// Copyright (C) 2016-2018  ERGO-Code

use crate::basiclu::*;
use crate::lu_internal::*;
use crate::lu_solve_symbolic::lu_solve_symbolic;
use crate::lu_solve_triangular::lu_solve_triangular;
use std::mem::size_of;
use std::time::Instant;

pub(crate) fn lu_solve_for_update(
    lu: &mut LU,
    nrhs: LUInt,
    irhs: &[LUInt],
    xrhs: Option<&[f64]>,
    p_nlhs: Option<&mut LUInt>,
    ilhs: Option<&mut [LUInt]>,
    xlhs: Option<&mut [f64]>,
    trans: char,
) -> LUInt {
    let m = lu.m;
    let nforrest = lu.nforrest;
    let pivotlen = lu.pivotlen;
    let nz_sparse = (lu.sparse_thres as LUInt) * m;
    let droptol = lu.droptol;
    let p = &p!(lu);
    let pmap = &pmap!(lu);
    let qmap = &qmap!(lu);
    // let eta_row = &mut eta_row!(lu);
    let pivotcol = &pivotcol!(lu);
    let pivotrow = &pivotrow!(lu);
    let l_begin = &l_begin!(lu);
    let lt_begin = &lt_begin!(lu);
    let lt_begin_p = &lt_begin_p!(lu);
    let u_begin = &lu.u_begin;
    // let r_begin = &mut r_begin!(lu);
    let w_begin = &lu.w_begin;
    let w_end = &lu.w_end;
    let col_pivot = &lu.col_pivot;
    let row_pivot = &lu.row_pivot;
    let l_index = &mut lu.l_index;
    let l_value = &mut lu.l_value;
    let u_index = &mut lu.u_index;
    let u_value = &mut lu.u_value;
    let w_index = &mut lu.w_index;
    let w_value = &mut lu.w_value;
    let marked = &mut marked!(lu);

    // lu_int i, j, k, n, t, top, pos, put, ipivot, jpivot, nz, nz_symb, M,
    //     room, need, jbegin, jend;
    // double x, xdrop, pivot;

    let want_solution = p_nlhs.is_some() && ilhs.is_some() && xlhs.is_some();
    let (mut l_flops, mut u_flops, mut r_flops) = (0, 0, 0);
    // double tic[2], elapsed;
    let tic = Instant::now();

    if trans == 't' || trans == 'T' {
        // Solve transposed system //

        // let pattern_symb = &mut lu.iwork1;
        // let pattern = &mut lu.iwork1[m as usize..];
        let (pattern_symb, pattern) = iwork1!(lu).split_at_mut(m as usize);
        let work = &mut lu.work0;
        // lu_int *pstack       = (void *) lu.work1;
        let pstack = &mut lu.work1;
        assert!(size_of::<LUInt>() <= size_of::<f64>());

        let jpivot = irhs[0];
        let ipivot = pmap[jpivot as usize];
        let jbegin = w_begin[jpivot as usize];
        let jend = w_end[jpivot as usize];

        // Compute row eta vector.
        // Symbolic pattern in pattern_symb[top..m-1], indices of (actual)
        // nonzeros in pattern[0..nz-1], values scattered into work.
        // We do not drop small elements to zero, but the symbolic and the
        // numeric pattern will still be different when we have exact
        // cancellation.
        // M = ++lu.marker;
        lu.marker += 1;
        let marker = lu.marker;
        let top = lu_solve_symbolic(
            m,
            w_begin,
            Some(w_end),
            w_index,
            jend - jbegin,
            &w_index[jbegin as usize..],
            pattern_symb,
            pstack,
            marked,
            marker,
        );
        let nz_symb = m - top;

        // reallocate if not enough memory in Li, Lx (where we store R)
        let room = lu.l_mem - r_begin!(lu)[nforrest as usize];
        if room < nz_symb {
            lu.addmem_l = nz_symb - room;
            return BASICLU_REALLOCATE;
        }

        for pos in jbegin..jend {
            work[w_index[pos as usize] as usize] = w_value[pos as usize];
        }
        lu_solve_triangular(
            nz_symb,
            &pattern_symb[top as usize..],
            w_begin,
            Some(w_end),
            w_index,
            w_value,
            Some(col_pivot),
            0.0,
            work,
            pattern,
            &mut u_flops,
        );

        // Compress row eta into L, pattern mapped from column to row indices.
        // The triangularity test in lu_update requires the symbolic pattern.
        let mut put = r_begin!(lu)[nforrest as usize];
        for t in top..m {
            let j = pattern_symb[t as usize];
            let i = pmap[j as usize];
            l_index[put as usize] = i;
            l_value[put as usize] = work[j as usize];
            put += 1;
            work[j as usize] = 0.0;
        }
        r_begin!(lu)[nforrest as usize + 1] = put;
        eta_row!(lu)[nforrest as usize] = ipivot;
        lu.btran_for_update = jpivot;

        if !want_solution {
            return done(tic, lu, l_flops, u_flops, r_flops);
        }
        let p_nlhs = p_nlhs.unwrap();
        let ilhs = ilhs.unwrap();
        let xlhs = xlhs.unwrap();

        // Scatter the row eta into xlhs and scale it to become the solution
        // to U^{-1}*[unit vector]. Now we can drop small entries to zero and
        // recompute the numerical pattern.
        // M = ++lu.marker;
        lu.marker += 1;
        let marker = lu.marker;
        pattern[0] = ipivot;
        marked[ipivot as usize] = marker;
        let pivot = col_pivot[jpivot as usize];
        xlhs[ipivot as usize] = 1.0 / pivot;

        let xdrop = droptol * pivot.abs();
        let mut nz: LUInt = 1;
        for pos in r_begin!(lu)[nforrest as usize]..r_begin!(lu)[(nforrest + 1) as usize] {
            if l_value[pos as usize].abs() > xdrop {
                let i = l_index[pos as usize];
                pattern[nz as usize] = i;
                nz += 1;
                marked[i as usize] = marker;
                xlhs[i as usize] = -l_value[pos as usize] / pivot;
            }
        }

        // Solve with update etas.
        // Append fill-in to pattern.
        // for (t = nforrest-1; t >= 0; t--)
        for t in (0..nforrest as usize).rev() {
            let ipivot = eta_row!(lu)[t];
            if xlhs[ipivot as usize] != 0.0 {
                let x = xlhs[ipivot as usize];
                for pos in r_begin!(lu)[t]..r_begin!(lu)[t + 1] {
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
            let top = lu_solve_symbolic(
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
            let nz = lu_solve_triangular(
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
            let mut nz: LUInt = 0;
            // for (k = m-1; k >= 0; k--)
            for k in (0..m).rev() {
                let ipivot = p[k as usize];
                if xlhs[ipivot as usize] != 0.0 {
                    let x = xlhs[ipivot as usize];
                    let mut pos = lt_begin_p[k as usize] as usize;
                    while l_index[pos] >= 0 {
                        let i = l_index[pos];
                        xlhs[i as usize] -= x * l_value[pos];
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
        let top = lu_solve_symbolic(
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
            work[irhs[n] as usize] = xrhs.unwrap()[n];
        }
        let mut nz = lu_solve_triangular(
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
                t += 1
            }
        }

        // Solve with update etas.
        // Append fill-in to pattern.
        let mut pos = r_begin!(lu)[0];
        for t in 0..nforrest as usize {
            let ipivot = eta_row!(lu)[t];
            let mut x = 0.0;
            while pos < r_begin!(lu)[t + 1] {
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
        r_flops += r_begin!(lu)[nforrest as usize] - r_begin!(lu)[0];

        // reallocate if not enough memory in U
        let room = lu.u_mem - u_begin[m as usize];
        let need = nz + 1;
        if room < need {
            for n in 0..nz {
                work[pattern[n as usize] as usize] = 0.0;
            }
            lu.addmem_u = need - room;
            return BASICLU_REALLOCATE;
        }

        // Compress spike into U.
        let mut put = u_begin[m as usize];
        for n in 0..nz {
            let i = pattern[n as usize];
            u_index[put as usize] = i;
            u_value[put as usize] = work[i as usize];
            put += 1;
            if !want_solution {
                work[i as usize] = 0.0;
            }
        }
        u_index[put as usize] = -1; // terminate column
        put += 1;
        lu.ftran_for_update = 0;

        if !want_solution {
            return done(tic, lu, l_flops, u_flops, r_flops);
        }
        let ilhs = ilhs.unwrap();
        let xlhs = xlhs.unwrap();

        if nz <= nz_sparse {
            // Sparse triangular solve with U.
            // Solution scattered into work, indices in ilhs[0..nz-1].
            // M = ++lu.marker;
            lu.marker += 1;
            let marker = lu.marker;
            let top = lu_solve_symbolic(
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

            nz = lu_solve_triangular(
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
                        let i = u_index[pos as usize] as usize;
                        work[i] -= x * u_value[pos as usize];
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
        *p_nlhs.unwrap() = nz;
    }

    done(tic, lu, l_flops, u_flops, r_flops)
}

fn done(tic: Instant, lu: &mut LU, l_flops: LUInt, u_flops: LUInt, r_flops: LUInt) -> LUInt {
    let elapsed = tic.elapsed().as_secs_f64();
    lu.time_solve += elapsed;
    lu.time_solve_total += elapsed;
    lu.l_flops += l_flops;
    lu.u_flops += u_flops;
    lu.r_flops += r_flops;
    lu.update_cost_numer += r_flops as f64;
    return BASICLU_OK;
}
