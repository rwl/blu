// Copyright (C) 2016-2018  ERGO-Code

use crate::basiclu::lu_int;
use crate::lu_internal::lu;
use std::mem::size_of;
use std::time::Instant;

pub(crate) fn lu_solve_sparse(
    this: &mut lu,
    nrhs: lu_int,
    irhs: &[lu_int],
    xrhs: &[f64],
    p_nlhs: &mut lu_int,
    ilhs: &mut [lu_int],
    xlhs: &[f64],
    trans: char,
) {
    let m = this.m;
    let nforrest = this.nforrest;
    let pivotlen = this.pivotlen;
    let nz_sparse = this.sparse_thres * m;
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

        let pattern_symb = &this.iwork1;
        let pattern = &this.iwork1[m..];
        let work = &this.work0;
        // lu_int *pstack = (void *) this.work1;
        let pstack = &this.work1;
        assert!(size_of::<lu_int>() <= size_of::<f64>());

        // Sparse triangular solve with U'.
        // Solution scattered into work, indices in pattern[0..nz-1].
        // M = ++this.marker;
        this.marker += 1;
        let M = this.marker;
        let top = lu_solve_symbolic(
            m,
            Wbegin,
            Wend,
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
            work[irhs[n]] = xrhs[n];
        }
        let mut nz = lu_solve_triangular(
            nz_symb,
            pattern_symb + top,
            Wbegin,
            Wend,
            Windex,
            Wvalue,
            col_pivot,
            droptol,
            work,
            pattern,
            &Uflops,
        );

        // Permute solution into xlhs.
        // Map pattern from column indices to row indices.
        // M = ++this.marker;
        this.marker += 1;
        let M = this.marker;
        for n in 0..nz {
            let j = pattern[n];
            let i = pmap[j];
            pattern[n] = i;
            xlhs[i] = work[j];
            work[j] = 0;
            marked[i] = M;
        }

        // Solve with update etas.
        // Append fill-in to pattern.
        // for (t = nforrest-1; t >= 0; t--)
        for t in (0..nforrest).rev() {
            let ipivot = eta_row[t];
            if xlhs[ipivot] {
                let x = xlhs[ipivot];
                for pos in Rbegin[t]..Rbegin[t + 1] {
                    let i = Lindex[pos];
                    if marked[i] != M {
                        marked[i] = M;
                        pattern[nz] = i;
                        nz += 1;
                    }
                    xlhs[i] -= x * Lvalue[pos];
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
                pattern_symb + top,
                Ltbegin,
                None,
                Lindex,
                Lvalue,
                None,
                droptol,
                xlhs,
                ilhs,
                &Lflops,
            );
            *p_nlhs = nz;
        } else {
            // Sequential triangular solve with L'.
            // Solution scattered into xlhs, indices in ilhs[0..nz-1].
            nz = 0;
            // for (k = m-1; k >= 0; k--)
            for k in (0..m).rev() {
                let ipivot = p[k];
                if xlhs[ipivot] {
                    let x = xlhs[ipivot];
                    // for (pos = Ltbegin_p[k]; (i = Lindex[pos]) >= 0; pos++)
                    let pos = Ltbegin_p[k];
                    while Lindex[pos] >= 0 {
                        let i = Lindex[pos];
                        xlhs[i] -= x * Lvalue[pos];
                        Lflops += 1;
                        pos += 1;
                    }
                    if x.abs() > droptol {
                        ilhs[nz] = ipivot;
                        nz += 1;
                    } else {
                        xlhs[ipivot] = 0.0;
                    }
                }
            }
            *p_nlhs = nz;
        }
    } else {
        // Solve forward system //

        let pattern_symb = &this.iwork1;
        let pattern = &this.iwork1[m..];
        let work = &this.work0;
        // lu_int *pstack       = (void *) this.work1;
        let pstack = &this.work1;
        assert!(size_of::<lu_int>() <= size_of::<f64>());

        // Sparse triangular solve with L.
        // Solution scattered into work, indices in pattern[0..nz-1].
        // M = ++this.marker;
        this.marker += 1;
        M = this.marker;
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

        for n in 0..nrhs {
            work[irhs[n]] = xrhs[n];
        }
        nz = lu_solve_triangular(
            nz_symb,
            pattern_symb + top,
            Lbegin,
            None,
            Lindex,
            Lvalue,
            None,
            droptol,
            work,
            pattern,
            &Lflops,
        );

        // unmark cancellation
        if nz < nz_symb {
            let mut t = top;
            let mut n = 0;
            while n < nz {
                i = pattern_symb[t];
                if i == pattern[n] {
                    n += 1;
                } else {
                    marked[i] -= 1;
                }
                t += 1;
            }
            while t < m {
                marked[pattern_symb[t]] -= 1;
                t += 1;
            }
        }

        // Solve with update etas.
        // Append fill-in to pattern.
        let mut pos = Rbegin[0];
        for t in 0..nforrest {
            let ipivot = eta_row[t];
            let mut x = 0.0;
            while pos < Rbegin[t + 1] {
                x += work[Lindex[pos]] * Lvalue[pos];
                pos += 1;
            }
            work[ipivot] -= x;
            if x != 0.0 && marked[ipivot] != M {
                marked[ipivot] = M;
                pattern[nz] = ipivot;
                nz += 1;
            }
        }
        Rflops += Rbegin[nforrest] - Rbegin[0];

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
                pattern_symb + top,
                Ubegin,
                None,
                Uindex,
                Uvalue,
                row_pivot,
                droptol,
                work,
                ilhs,
                &Uflops,
            );

            // Permute solution into xlhs.
            // Map pattern from row indices to column indices.
            for n in 0..nz {
                i = ilhs[n];
                j = qmap[i];
                ilhs[n] = j;
                xlhs[j] = work[i];
                work[i] = 0;
            }
        } else {
            // Sequential triangular solve with U.
            // Solution computed in work and permuted into xlhs.
            // Pattern (in column indices) stored in ilhs[0..nz-1].
            nz = 0;
            // for (k = pivotlen-1; k >= 0; k--)
            for k in (0..pivotlen).rev() {
                let ipivot = pivotrow[k];
                let jpivot = pivotcol[k];
                if work[ipivot] {
                    let x = work[ipivot] / row_pivot[ipivot];
                    work[ipivot] = 0.0;
                    // for (pos = Ubegin[ipivot]; (i = Uindex[pos]) >= 0; pos++)
                    let mut pos = Ubegin[ipivot];
                    while Uindex[pos] >= 0 {
                        let i = Uindex[pos];
                        work[i] -= x * Uvalue[pos];
                        Uflops += 1;
                        pos += 1;
                    }
                    if x.abs() > droptol {
                        ilhs[nz] = jpivot;
                        nz += 1;
                        xlhs[jpivot] = x;
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
    this.update_cost_numer += Rflops;
}
