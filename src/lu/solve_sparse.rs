// Copyright (C) 2016-2018 ERGO-Code
// Copyright (C) 2022-2023 Richard Lincoln

use crate::lu::lu::*;
use crate::lu::solve_symbolic::solve_symbolic;
use crate::lu::solve_triangular::solve_triangular;
use crate::LUInt;
use std::mem::size_of;
use std::time::Instant;

pub(crate) fn solve_sparse(
    lu: &mut LU,
    nrhs: usize,
    irhs: &[LUInt],
    xrhs: &[f64],
    p_nlhs: &mut usize,
    ilhs: &mut [LUInt],
    xlhs: &mut [f64],
    trans: char,
) {
    let m = lu.m;
    let nforrest = lu.nforrest;
    let pivotlen = lu.pivotlen;
    let nz_sparse = (lu.sparse_thres * m as f64) as usize;
    let droptol = lu.droptol;
    let p = &p!(lu);
    let pmap = &pmap!(lu);
    let qmap = &qmap!(lu);
    let eta_row = &eta_row!(lu);
    let pivotcol = &pivotcol!(lu);
    let pivotrow = &pivotrow!(lu);
    let l_begin = &l_begin!(lu);
    let lt_begin = &lt_begin!(lu);
    let lt_begin_p = &lt_begin_p!(lu);
    let u_begin = &lu.u_begin;
    let r_begin = &r_begin!(lu);
    let w_begin = &lu.w_begin;
    let w_end = &lu.w_end;
    let col_pivot = &lu.col_pivot;
    let row_pivot = &lu.row_pivot;
    let l_index = &lu.l_index;
    let l_value = &lu.l_value;
    let u_index = &lu.u_index;
    let u_value = &lu.u_value;
    let w_index = &lu.w_index;
    let w_value = &lu.w_value;
    let marked = &mut marked!(lu);

    let (mut l_flops, mut u_flops, mut r_flops) = (0, 0, 0);
    let tic = Instant::now();

    if trans == 't' || trans == 'T' {
        // Solve transposed system //

        // let pattern_symb = &mut lu.iwork1;
        // let pattern = &mut lu.iwork1[m as usize..];
        let (pattern_symb, pattern) = iwork1!(lu).split_at_mut(m as usize);
        let work = &mut lu.work0;
        // lu_int *pstack = (void *) lu.work1;
        let pstack = &mut lu.work1;
        assert!(size_of::<LUInt>() <= size_of::<f64>());

        // Sparse triangular solve with U'.
        // Solution scattered into work, indices in pattern[0..nz-1].
        // M = ++lu.marker;
        lu.marker += 1;
        let marker = lu.marker;
        let top = solve_symbolic(
            m,
            w_begin,
            Some(w_end),
            w_index,
            nrhs,
            irhs,
            pattern_symb,
            pstack,
            marked,
            marker,
        );
        let nz_symb = m - top;

        for n in 0..nrhs {
            work[irhs[n as usize] as usize] = xrhs[n as usize];
        }
        let mut nz = solve_triangular(
            nz_symb,
            &pattern_symb[top as usize..],
            w_begin,
            Some(w_end),
            w_index,
            w_value,
            Some(col_pivot),
            droptol,
            work,
            pattern,
            &mut u_flops,
        );

        // Permute solution into xlhs.
        // Map pattern from column indices to row indices.
        // M = ++lu.marker;
        lu.marker += 1;
        let marker = lu.marker;
        for n in 0..nz {
            let j = pattern[n as usize];
            let i = pmap[j as usize];
            pattern[n as usize] = i;
            xlhs[i as usize] = work[j as usize];
            work[j as usize] = 0.0;
            marked[i as usize] = marker;
        }

        // Solve with update etas.
        // Append fill-in to pattern.
        // for (t = nforrest-1; t >= 0; t--)
        for t in (0..nforrest as usize).rev() {
            let ipivot = eta_row[t];
            if xlhs[ipivot as usize] != 0.0 {
                let x = xlhs[ipivot as usize];
                for pos in r_begin[t]..r_begin[t + 1] {
                    let i = l_index[pos as usize];
                    if marked[i as usize] != marker {
                        marked[i as usize] = marker;
                        pattern[nz as usize] = i;
                        nz += 1;
                    }
                    xlhs[i as usize] -= x * l_value[pos as usize];
                    r_flops += 1;
                }
            }
        }

        if nz <= nz_sparse {
            // Sparse triangular solve with L'.
            // Solution scattered into xlhs, indices in ilhs[0..nz-1].
            // M = ++lu.marker;
            lu.marker += 1;
            let marker = lu.marker;
            let top = solve_symbolic(
                m,
                lt_begin,
                None,
                l_index,
                nz,
                pattern,
                pattern_symb,
                pstack,
                marked,
                marker,
            );
            let nz_symb = m - top;

            nz = solve_triangular(
                nz_symb,
                &pattern_symb[top as usize..],
                lt_begin,
                None,
                l_index,
                l_value,
                None,
                droptol,
                xlhs,
                ilhs,
                &mut l_flops,
            );
            *p_nlhs = nz;
        } else {
            // Sequential triangular solve with L'.
            // Solution scattered into xlhs, indices in ilhs[0..nz-1].
            nz = 0;
            // for (k = m-1; k >= 0; k--)
            for k in (0..m).rev() {
                let ipivot = p[k as usize];
                if xlhs[ipivot as usize] != 0.0 {
                    let x = xlhs[ipivot as usize];
                    // for (pos = lt_begin_p[k]; (i = Lindex[pos]) >= 0; pos++)
                    let mut pos = lt_begin_p[k as usize];
                    while l_index[pos as usize] >= 0 {
                        let i = l_index[pos as usize];
                        xlhs[i as usize] -= x * l_value[pos as usize];
                        l_flops += 1;
                        pos += 1;
                    }
                    if x.abs() > droptol {
                        ilhs[nz as usize] = ipivot;
                        nz += 1;
                    } else {
                        xlhs[ipivot as usize] = 0.0;
                    }
                }
            }
            *p_nlhs = nz;
        }
    } else {
        // Solve forward system //

        // let pattern_symb = &mut lu.iwork1;
        // let pattern = &mut lu.iwork1[m as usize..];
        let (pattern_symb, pattern) = iwork1!(lu).split_at_mut(m as usize);
        let work = &mut lu.work0;
        // lu_int *pstack       = (void *) lu.work1;
        let pstack = &mut lu.work1;
        assert!(size_of::<LUInt>() <= size_of::<f64>());

        // Sparse triangular solve with L.
        // Solution scattered into work, indices in pattern[0..nz-1].
        // M = ++lu.marker;
        lu.marker += 1;
        let marker = lu.marker;
        let top = solve_symbolic(
            m,
            l_begin,
            None,
            l_index,
            nrhs,
            irhs,
            pattern_symb,
            pstack,
            marked,
            marker,
        );
        let nz_symb = m - top;

        for n in 0..nrhs as usize {
            work[irhs[n] as usize] = xrhs[n];
        }
        let mut nz = solve_triangular(
            nz_symb,
            &pattern_symb[top as usize..],
            l_begin,
            None,
            l_index,
            l_value,
            None,
            droptol,
            work,
            pattern,
            &mut l_flops,
        );

        // unmark cancellation
        if nz < nz_symb {
            let mut t = top;
            let mut n = 0;
            while n < nz {
                let i = pattern_symb[t as usize];
                if i == pattern[n as usize] {
                    n += 1;
                } else {
                    marked[i as usize] -= 1;
                }
                t += 1;
            }
            while t < m {
                marked[pattern_symb[t as usize] as usize] -= 1;
                t += 1;
            }
        }

        // Solve with update etas.
        // Append fill-in to pattern.
        let mut pos = r_begin[0];
        for t in 0..nforrest as usize {
            let ipivot = eta_row[t];
            let mut x = 0.0;
            while pos < r_begin[t + 1] {
                x += work[l_index[pos as usize] as usize] * l_value[pos as usize];
                pos += 1;
            }
            work[ipivot as usize] -= x;
            if x != 0.0 && marked[ipivot as usize] != marker {
                marked[ipivot as usize] = marker;
                pattern[nz as usize] = ipivot;
                nz += 1;
            }
        }
        r_flops += (r_begin[nforrest as usize] - r_begin[0]) as usize;

        if nz <= nz_sparse {
            // Sparse triangular solve with U.
            // Solution scattered into work, indices in ilhs[0..nz-1].
            // M = ++lu.marker;
            lu.marker += 1;
            let marker = lu.marker;
            let top = solve_symbolic(
                m,
                u_begin,
                None,
                u_index,
                nz,
                pattern,
                pattern_symb,
                pstack,
                marked,
                marker,
            );
            let nz_symb = m - top;

            nz = solve_triangular(
                nz_symb,
                &pattern_symb[top as usize..],
                u_begin,
                None,
                u_index,
                u_value,
                Some(row_pivot),
                droptol,
                work,
                ilhs,
                &mut u_flops,
            );

            // Permute solution into xlhs.
            // Map pattern from row indices to column indices.
            for n in 0..nz {
                let i = ilhs[n as usize];
                let j = qmap[i as usize];
                ilhs[n as usize] = j;
                xlhs[j as usize] = work[i as usize];
                work[i as usize] = 0.0;
            }
        } else {
            // Sequential triangular solve with U.
            // Solution computed in work and permuted into xlhs.
            // Pattern (in column indices) stored in ilhs[0..nz-1].
            nz = 0;
            // for (k = pivotlen-1; k >= 0; k--)
            for k in (0..pivotlen).rev() {
                let ipivot = pivotrow[k as usize];
                let jpivot = pivotcol[k as usize];
                if work[ipivot as usize] != 0.0 {
                    let x = work[ipivot as usize] / row_pivot[ipivot as usize];
                    work[ipivot as usize] = 0.0;
                    // for (pos = Ubegin[ipivot]; (i = Uindex[pos]) >= 0; pos++)
                    let mut pos = u_begin[ipivot as usize];
                    while u_index[pos as usize] >= 0 {
                        let i = u_index[pos as usize];
                        work[i as usize] -= x * u_value[pos as usize];
                        u_flops += 1;
                        pos += 1;
                    }
                    if x.abs() > droptol {
                        ilhs[nz as usize] = jpivot;
                        nz += 1;
                        xlhs[jpivot as usize] = x;
                    }
                }
            }
        }
        *p_nlhs = nz;
    }

    let elapsed = tic.elapsed().as_secs_f64();
    lu.time_solve += elapsed;
    lu.time_solve_total += elapsed;
    lu.l_flops += l_flops;
    lu.u_flops += u_flops;
    lu.r_flops += r_flops;
    lu.update_cost_numer += r_flops as f64;
}
