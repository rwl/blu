// Copyright (C) 2016-2018  ERGO-Code

use crate::basiclu::*;
use crate::lu_internal::lu;
use crate::lu_solve_symbolic::lu_solve_symbolic;
use crate::lu_solve_triangular::lu_solve_triangular;
use std::mem::size_of;
use std::time::Instant;

pub(crate) fn lu_solve_for_update(
    this: &mut lu,
    nrhs: lu_int,
    irhs: &[lu_int],
    xrhs: Option<&[f64]>,
    p_nlhs: Option<&mut lu_int>,
    ilhs: Option<&mut [lu_int]>,
    xlhs: Option<&mut [f64]>,
    trans: char,
) -> lu_int {
    let m = this.m;
    let nforrest = this.nforrest;
    let pivotlen = this.pivotlen;
    let nz_sparse = (this.sparse_thres as lu_int) * m;
    let droptol = this.droptol;
    let p = this.p.as_ref().unwrap();
    let pmap = this.pmap.as_ref().unwrap();
    let qmap = this.qmap.as_ref().unwrap();
    let eta_row = &mut this.eta_row;
    let pivotcol = this.pivotcol.as_ref().unwrap();
    let pivotrow = this.pivotrow.as_ref().unwrap();
    let Lbegin = this.Lbegin.as_ref().unwrap();
    let Ltbegin = this.Ltbegin.as_ref().unwrap();
    let Ltbegin_p = this.Ltbegin_p.as_ref().unwrap();
    let Ubegin = &this.Ubegin;
    let Rbegin = this.Rbegin.as_mut().unwrap();
    let Wbegin = this.Wbegin.as_ref().unwrap();
    let Wend = this.Wend.as_ref().unwrap();
    let col_pivot = &this.col_pivot;
    let row_pivot = &this.row_pivot;
    let Lindex = this.Lindex.as_mut().unwrap();
    let Lvalue = this.Lvalue.as_mut().unwrap();
    let Uindex = this.Uindex.as_mut().unwrap();
    let Uvalue = this.Uvalue.as_mut().unwrap();
    let Windex = this.Windex.as_mut().unwrap();
    let Wvalue = this.Wvalue.as_mut().unwrap();
    let marked = this.marked.as_mut().unwrap();

    // lu_int i, j, k, n, t, top, pos, put, ipivot, jpivot, nz, nz_symb, M,
    //     room, need, jbegin, jend;
    // double x, xdrop, pivot;

    let want_solution = p_nlhs.is_some() && ilhs.is_some() && xlhs.is_some();
    let (mut Lflops, mut Uflops, mut Rflops) = (0, 0, 0);
    // double tic[2], elapsed;
    let tic = Instant::now();

    if trans == 't' || trans == 'T' {
        // Solve transposed system //

        // let pattern_symb = &mut this.iwork1;
        // let pattern = &mut this.iwork1[m as usize..];
        let (pattern_symb, pattern) = this.iwork1.as_mut().unwrap().split_at_mut(m as usize);
        let work = &mut this.work0;
        // lu_int *pstack       = (void *) this.work1;
        let pstack = &mut this.work1;
        assert!(size_of::<lu_int>() <= size_of::<f64>());

        let jpivot = irhs[0];
        let ipivot = pmap[jpivot as usize];
        let jbegin = Wbegin[jpivot as usize];
        let jend = Wend[jpivot as usize];

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
            Some(Wend),
            Windex,
            jend - jbegin,
            &Windex[jbegin as usize..],
            pattern_symb,
            pstack,
            marked,
            M,
        );
        let nz_symb = m - top;

        // reallocate if not enough memory in Li, Lx (where we store R)
        let room = this.Lmem - Rbegin[nforrest as usize];
        if room < nz_symb {
            this.addmemL = nz_symb - room;
            return BASICLU_REALLOCATE;
        }

        for pos in jbegin..jend {
            work[Windex[pos as usize] as usize] = Wvalue[pos as usize];
        }
        lu_solve_triangular(
            nz_symb,
            &pattern_symb[top as usize..],
            Wbegin,
            Some(Wend),
            Windex,
            Wvalue,
            Some(col_pivot),
            0.0,
            work,
            pattern,
            &mut Uflops,
        );

        // Compress row eta into L, pattern mapped from column to row indices.
        // The triangularity test in lu_update requires the symbolic pattern.
        let mut put = Rbegin[nforrest as usize];
        for t in top..m {
            let j = pattern_symb[t as usize];
            let i = pmap[j as usize];
            Lindex[put as usize] = i;
            Lvalue[put as usize] = work[j as usize];
            put += 1;
            work[j as usize] = 0.0;
        }
        Rbegin[nforrest as usize + 1] = put;
        eta_row[nforrest as usize] = ipivot;
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
        marked[ipivot as usize] = M;
        let pivot = col_pivot[jpivot as usize];
        xlhs[ipivot as usize] = 1.0 / pivot;

        let xdrop = droptol * pivot.abs();
        let mut nz: lu_int = 1;
        for pos in Rbegin[nforrest as usize]..Rbegin[(nforrest + 1) as usize] {
            if Lvalue[pos as usize].abs() > xdrop {
                let i = Lindex[pos as usize];
                pattern[nz as usize] = i;
                nz += 1;
                marked[i as usize] = M;
                xlhs[i as usize] = -Lvalue[pos as usize] / pivot;
            }
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
            let nz = lu_solve_triangular(
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
            let mut nz: lu_int = 0;
            // for (k = m-1; k >= 0; k--)
            for k in (0..m).rev() {
                let ipivot = p[k as usize];
                if xlhs[ipivot as usize] != 0.0 {
                    let x = xlhs[ipivot as usize];
                    let mut pos = Ltbegin_p[k as usize] as usize;
                    while Lindex[pos] >= 0 {
                        let i = Lindex[pos];
                        xlhs[i as usize] -= x * Lvalue[pos];
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

        // let pattern_symb = &mut this.iwork1;
        // let pattern = &mut this.iwork1[m as usize..];
        let (pattern_symb, pattern) = this.iwork1.as_mut().unwrap().split_at_mut(m as usize);
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
            work[irhs[n] as usize] = xrhs.unwrap()[n];
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
                t += 1
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

        // reallocate if not enough memory in U
        let room = this.Umem - Ubegin[m as usize];
        let need = nz + 1;
        if room < need {
            for n in 0..nz {
                work[pattern[n as usize] as usize] = 0.0;
            }
            this.addmemU = need - room;
            return BASICLU_REALLOCATE;
        }

        // Compress spike into U.
        let mut put = Ubegin[m as usize];
        for n in 0..nz {
            let i = pattern[n as usize];
            Uindex[put as usize] = i;
            Uvalue[put as usize] = work[i as usize];
            put += 1;
            if !want_solution {
                work[i as usize] = 0.0;
            }
        }
        Uindex[put as usize] = -1; // terminate column
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
                        let i = Uindex[pos as usize] as usize;
                        work[i] -= x * Uvalue[pos as usize];
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
        *p_nlhs.unwrap() = nz;
    }

    done(tic, this, Lflops, Uflops, Rflops)
}

fn done(tic: Instant, this: &mut lu, Lflops: lu_int, Uflops: lu_int, Rflops: lu_int) -> lu_int {
    let elapsed = tic.elapsed().as_secs_f64();
    this.time_solve += elapsed;
    this.time_solve_total += elapsed;
    this.Lflops += Lflops;
    this.Uflops += Uflops;
    this.Rflops += Rflops;
    this.update_cost_numer += Rflops as f64;
    return BASICLU_OK;
}
