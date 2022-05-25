// Copyright (C) 2016-2018  ERGO-Code

use crate::basiclu::*;
use crate::lu_internal::lu;
use std::mem::size_of;
use std::time::Instant;

pub(crate) fn lu_solve_for_update(
    this: &mut lu,
    nrhs: lu_int,
    irhs: &[lu_int],
    xrhs: Option<&[f64]>,
    p_nlhs: Option<&mut lu_int>,
    ilhs: Option<&mut [lu_int]>,
    xlhs: Option<&[f64]>,
    trans: char,
) -> lu_int {
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

    // lu_int i, j, k, n, t, top, pos, put, ipivot, jpivot, nz, nz_symb, M,
    //     room, need, jbegin, jend;
    // double x, xdrop, pivot;

    let want_solution = p_nlhs.is_some() && ilhs.is_some() && xlhs.is_some();
    let (mut Lflops, mut Uflops, mut Rflops) = (0, 0, 0);
    // double tic[2], elapsed;
    let tic = Instant::now();

    if trans == 't' || trans == 'T' {
        // Solve transposed system //

        let pattern_symb = &this.iwork1;
        let pattern = &this.iwork1[m..];
        let work = &this.work0;
        // lu_int *pstack       = (void *) this.work1;
        let pstack = &this.work1;
        assert!(size_of::<lu_int>() <= size_of::<f64>());

        let jpivot = irhs[0];
        let ipivot = pmap[jpivot];
        let jbegin = Wbegin[jpivot];
        let jend = Wend[jpivot];

        // Compute row eta vector.
        // Symbolic pattern in pattern_symb[top..m-1], indices of (actual)
        // nonzeros in pattern[0..nz-1], values scattered into work.
        // We do not drop small elements to zero, but the symbolic and the
        // numeric pattern will still be different when we have exact
        // cancellation.
        // M = ++this.marker;
        this.marker += 1;
        let M = this.marker;
        let top = lu_solve_symbolic(
            m,
            Wbegin,
            Wend,
            Windex,
            jend - jbegin,
            Windex + jbegin,
            pattern_symb,
            pstack,
            marked,
            M,
        );
        let nz_symb = m - top;

        // reallocate if not enough memory in Li, Lx (where we store R)
        let room = this.Lmem - Rbegin[nforrest];
        if room < nz_symb {
            this.addmemL = nz_symb - room;
            return BASICLU_REALLOCATE;
        }

        for pos in jbegin..jend {
            work[Windex[pos]] = Wvalue[pos];
        }
        lu_solve_triangular(
            nz_symb,
            pattern_symb + top,
            Wbegin,
            Wend,
            Windex,
            Wvalue,
            col_pivot,
            0.0,
            work,
            pattern,
            &Uflops,
        );

        // Compress row eta into L, pattern mapped from column to row indices.
        // The triangularity test in lu_update requires the symbolic pattern.
        let put = Rbegin[nforrest];
        for t in top..m {
            j = pattern_symb[t];
            i = pmap[j];
            Lindex[put] = i;
            Lvalue[put] = work[j];
            put += 1;
            work[j] = 0;
        }
        Rbegin[nforrest + 1] = put;
        eta_row[nforrest] = ipivot;
        this.btran_for_update = jpivot;

        if !want_solution {
            return done(tic, this, Lflops, Uflops, Rflops);
        }
        let p_nlhs = p_nlhs.unwrap();
        let ilhs = ilhs.unwrap();
        let xlhs = xlhs.unwrap();

        // Scatter the row eta into xlhs and scale it to become the solution
        // to U^{-1}*[unit vector]. Now we can drop small entries to zero and
        // recompute the numerical pattern.
        // M = ++this.marker;
        this.marker += 1;
        let M = this.marker;
        pattern[0] = ipivot;
        marked[ipivot] = M;
        pivot = col_pivot[jpivot];
        xlhs[ipivot] = 1.0 / pivot;

        xdrop = droptol * pivot.abs();
        let mut nz = 1;
        for pos in Rbegin[nforrest]..Rbegin[nforrest + 1] {
            if Lvalue[pos].abs() > xdrop {
                pattern[nz] = i = Lindex[pos];
                nz += 1;
                marked[i] = M;
                xlhs[i] = -Lvalue[pos] / pivot;
            }
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
            let nz = lu_solve_triangular(
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
            let mut nz: lu_int = 0;
            // for (k = m-1; k >= 0; k--)
            for k in (0..m).rev() {
                let ipivot = p[k];
                if xlhs[ipivot] {
                    let x = xlhs[ipivot];
                    let mut pos = Ltbegin_p[k];
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

        for n in 0..nrhs {
            work[irhs[n]] = xrhs[n];
        }
        let mut nz = lu_solve_triangular(
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
            let t = top;
            let mut n = 0;
            while n < nz {
                let i = pattern_symb[t];
                if i == pattern[n] {
                    n += 1;
                } else {
                    marked[i] -= 1;
                }
                t += 1;
            }
            while t < m {
                marked[pattern_symb[t]] -= 1;
                t += 1
            }
        }

        // Solve with update etas.
        // Append fill-in to pattern.
        pos = Rbegin[0];
        for t in 0..nforrest {
            ipivot = eta_row[t];
            x = 0.0;
            while pos < Rbegin[t + 1] {
                x += work[Lindex[pos]] * Lvalue[pos];
                pos += 1;
            }
            work[ipivot] -= x;
            if x && marked[ipivot] != M {
                marked[ipivot] = M;
                pattern[nz] = ipivot;
                nz += 1;
            }
        }
        Rflops += Rbegin[nforrest] - Rbegin[0];

        // reallocate if not enough memory in U
        room = this.Umem - Ubegin[m];
        need = nz + 1;
        if room < need {
            for n in 0..nz {
                work[pattern[n]] = 0.0;
            }
            this.addmemU = need - room;
            return BASICLU_REALLOCATE;
        }

        // Compress spike into U.
        put = Ubegin[m];
        for n in 0..nz {
            i = pattern[n];
            Uindex[put] = i;
            Uvalue[put] = work[i];
            put += 1;
            if !want_solution {
                work[i] = 0.0;
            }
        }
        Uindex[put] = -1; // terminate column
        put += 1;
        this.ftran_for_update = 0;

        if !want_solution {
            return done(tic, this, Lflops, Uflops, Rflops);
        }
        let ilhs = ilhs.unwrap();
        let xlhs = xlhs.unwrap();

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
                    x = work[ipivot] / row_pivot[ipivot];
                    work[ipivot] = 0.0;
                    // for (pos = Ubegin[ipivot]; (i = Uindex[pos]) >= 0; pos++)
                    let pos = Ubegin[ipivot];
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
        *p_nlhs.unwrap() = nz;
    }

    done(tic, this, Lflops, Uflops, Rflops)
}

fn done(tic: Instant, this: &mut lu, Lflops: lu_int, Uflops: lu_int, Rflops: lu_int) -> lu_int {
    elapsed = lu_toc(tic);
    this.time_solve += elapsed;
    this.time_solve_total += elapsed;
    this.Lflops += Lflops;
    this.Uflops += Uflops;
    this.Rflops += Rflops;
    this.update_cost_numer += Rflops;
    return BASICLU_OK;
}
