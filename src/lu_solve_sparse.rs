// Copyright (C) 2016-2018  ERGO-Code

use crate::basiclu::lu_int;
use crate::lu_internal::lu;
use crate::lu_solve_symbolic::lu_solve_symbolic;
use crate::lu_solve_triangular::lu_solve_triangular;
use std::mem::size_of;
use std::time::Instant;

pub(crate) fn lu_solve_sparse(
    this: &mut lu,
    nrhs: lu_int,
    irhs: &[lu_int],
    xrhs: &[f64],
    p_nlhs: &mut lu_int,
    ilhs: &mut [lu_int],
    xlhs: &mut [f64],
    trans: char,
) {
    let m = this.m;
    let nforrest = this.nforrest;
    let pivotlen = this.pivotlen;
    let nz_sparse = (this.sparse_thres as lu_int) * m;
    let droptol = this.droptol;
    let p = &this.p;
    let pmap = &this.pmap;
    let qmap = &this.qmap;
    let eta_row = &this.eta_row;
    let pivotcol = &this.pivotcol;
    let pivotrow = &this.pivotrow;
    let Lbegin = &this.Lbegin;
    let Ltbegin = &this.Ltbegin;
    let Ltbegin_p = &this.Ltbegin_p;
    let Ubegin = &this.Ubegin;
    let Rbegin = &this.Rbegin;
    let Wbegin = &this.Wbegin;
    let Wend = &this.Wend;
    let col_pivot = &this.col_pivot;
    let row_pivot = &this.row_pivot;
    let Lindex = this.Lindex.as_ref().unwrap();
    let Lvalue = this.Lvalue.as_ref().unwrap();
    let Uindex = this.Uindex.as_ref().unwrap();
    let Uvalue = this.Uvalue.as_ref().unwrap();
    let Windex = this.Windex.as_ref().unwrap();
    let Wvalue = this.Wvalue.as_ref().unwrap();
    let marked = &mut this.marked;

    let (mut Lflops, mut Uflops, mut Rflops) = (0, 0, 0);
    let tic = Instant::now();

    if trans == 't' || trans == 'T' {
        // Solve transposed system //

        let pattern_symb = &mut this.iwork1;
        let pattern = &mut this.iwork1[m as usize..];
        let work = &mut this.work0;
        // lu_int *pstack = (void *) this.work1;
        let pstack = &mut this.work1;
        assert!(size_of::<lu_int>() <= size_of::<f64>());

        // Sparse triangular solve with U'.
        // Solution scattered into work, indices in pattern[0..nz-1].
        // M = ++this.marker;
        this.marker += 1;
        let M = this.marker;
        let top = lu_solve_symbolic(
            m,
            Wbegin,
            Some(Wend),
            Windex,
            nrhs,
            irhs,
            pattern_symb,
            pstack,
            marked,
            M,
        );
        let nz_symb = m - top;

        for n in 0..nrhs {
            work[irhs[n as usize] as usize] = xrhs[n as usize];
        }
        let mut nz = lu_solve_triangular(
            nz_symb,
            &pattern_symb[top as usize..],
            Wbegin,
            Some(Wend),
            Windex,
            Wvalue,
            Some(col_pivot),
            droptol,
            work,
            pattern,
            &mut Uflops,
        );

        // Permute solution into xlhs.
        // Map pattern from column indices to row indices.
        // M = ++this.marker;
        this.marker += 1;
        let M = this.marker;
        for n in 0..nz {
            let j = pattern[n as usize];
            let i = pmap[j as usize];
            pattern[n as usize] = i;
            xlhs[i as usize] = work[j as usize];
            work[j as usize] = 0.0;
            marked[i as usize] = M;
        }

        // Solve with update etas.
        // Append fill-in to pattern.
        // for (t = nforrest-1; t >= 0; t--)
        for t in (0..nforrest as usize).rev() {
            let ipivot = eta_row[t];
            if xlhs[ipivot as usize] != 0.0 {
                let x = xlhs[ipivot as usize];
                for pos in Rbegin[t]..Rbegin[t + 1] {
                    let i = Lindex[pos as usize];
                    if marked[i as usize] != M {
                        marked[i as usize] = M;
                        pattern[nz as usize] = i;
                        nz += 1;
                    }
                    xlhs[i as usize] -= x * Lvalue[pos as usize];
                    Rflops += 1;
                }
            }
        }

        if nz <= nz_sparse {
            // Sparse triangular solve with L'.
            // Solution scattered into xlhs, indices in ilhs[0..nz-1].
            // M = ++this.marker;
            this.marker += 1;
            let M = this.marker;
            let top = lu_solve_symbolic(
                m,
                Ltbegin,
                None,
                Lindex,
                nz,
                pattern,
                pattern_symb,
                pstack,
                marked,
                M,
            );
            let nz_symb = m - top;

            nz = lu_solve_triangular(
                nz_symb,
                &pattern_symb[top as usize..],
                Ltbegin,
                None,
                Lindex,
                Lvalue,
                None,
                droptol,
                xlhs,
                ilhs,
                &mut Lflops,
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
                    // for (pos = Ltbegin_p[k]; (i = Lindex[pos]) >= 0; pos++)
                    let mut pos = Ltbegin_p[k as usize];
                    while Lindex[pos as usize] >= 0 {
                        let i = Lindex[pos as usize];
                        xlhs[i as usize] -= x * Lvalue[pos as usize];
                        Lflops += 1;
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

        let pattern_symb = &mut this.iwork1;
        let pattern = &mut this.iwork1[m as usize..];
        let work = &mut this.work0;
        // lu_int *pstack       = (void *) this.work1;
        let pstack = &mut this.work1;
        assert!(size_of::<lu_int>() <= size_of::<f64>());

        // Sparse triangular solve with L.
        // Solution scattered into work, indices in pattern[0..nz-1].
        // M = ++this.marker;
        this.marker += 1;
        let M = this.marker;
        let top = lu_solve_symbolic(
            m,
            Lbegin,
            None,
            Lindex,
            nrhs,
            irhs,
            pattern_symb,
            pstack,
            marked,
            M,
        );
        let nz_symb = m - top;

        for n in 0..nrhs as usize {
            work[irhs[n] as usize] = xrhs[n];
        }
        let mut nz = lu_solve_triangular(
            nz_symb,
            &pattern_symb[top as usize..],
            Lbegin,
            None,
            Lindex,
            Lvalue,
            None,
            droptol,
            work,
            pattern,
            &mut Lflops,
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
        let mut pos = Rbegin[0];
        for t in 0..nforrest as usize {
            let ipivot = eta_row[t];
            let mut x = 0.0;
            while pos < Rbegin[t + 1] {
                x += work[Lindex[pos as usize] as usize] * Lvalue[pos as usize];
                pos += 1;
            }
            work[ipivot as usize] -= x;
            if x != 0.0 && marked[ipivot as usize] != M {
                marked[ipivot as usize] = M;
                pattern[nz as usize] = ipivot;
                nz += 1;
            }
        }
        Rflops += Rbegin[nforrest as usize] - Rbegin[0];

        if nz <= nz_sparse {
            // Sparse triangular solve with U.
            // Solution scattered into work, indices in ilhs[0..nz-1].
            // M = ++this.marker;
            this.marker += 1;
            let M = this.marker;
            let top = lu_solve_symbolic(
                m,
                Ubegin,
                None,
                Uindex,
                nz,
                pattern,
                pattern_symb,
                pstack,
                marked,
                M,
            );
            let nz_symb = m - top;

            nz = lu_solve_triangular(
                nz_symb,
                &pattern_symb[top as usize..],
                Ubegin,
                None,
                Uindex,
                Uvalue,
                Some(row_pivot),
                droptol,
                work,
                ilhs,
                &mut Uflops,
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
                    let mut pos = Ubegin[ipivot as usize];
                    while Uindex[pos as usize] >= 0 {
                        let i = Uindex[pos as usize];
                        work[i as usize] -= x * Uvalue[pos as usize];
                        Uflops += 1;
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
    this.time_solve += elapsed;
    this.time_solve_total += elapsed;
    this.Lflops += Lflops;
    this.Uflops += Uflops;
    this.Rflops += Rflops;
    this.update_cost_numer += Rflops as f64;
}
