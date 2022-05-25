// Copyright (C) 2016-2018  ERGO-Code

use crate::lu_garbage_perm::lu_garbage_perm;
use crate::lu_internal::lu;

pub(crate) fn lu_solve_dense(this: &mut lu, rhs: &[f64], lhs: &mut [f64], trans: char) {
    let m = this.m;
    let nforrest = this.nforrest;
    let p = &this.p;
    let eta_row = &this.eta_row;
    let pivotcol = &this.pivotcol;
    let pivotrow = &this.pivotrow;
    let Lbegin_p = &this.Lbegin_p;
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
    let work1 = &mut this.work1;

    lu_garbage_perm(this);
    assert_eq!(this.pivotlen, m);

    if trans == 't' || trans == 'T' {
        // Solve transposed system

        // memcpy(work1, rhs, m*sizeof(double));
        work1[..m].copy_from_slice(rhs);

        // Solve with U'.
        for k in 0..m {
            let jpivot = pivotcol[k];
            let ipivot = pivotrow[k];
            let x = work1[jpivot] / col_pivot[jpivot];
            for pos in Wbegin[jpivot]..Wend[jpivot] {
                work1[Windex[pos]] -= x * Wvalue[pos];
            }
            lhs[ipivot] = x;
        }

        // Solve with update ETAs backwards.
        // for (t = nforrest-1; t >= 0; t--)
        for t in (0..nforrest).rev() {
            let ipivot = eta_row[t];
            let x = lhs[ipivot];
            for pos in Rbegin[t]..Rbegin[t + 1] {
                let i = Lindex[pos];
                lhs[i] -= x * Lvalue[pos];
            }
        }

        // Solve with L'.
        // for (k = m-1; k >= 0; k--)
        for k in (0..m).rev() {
            let mut x = 0.0;
            // for (pos = Lbegin_p[k]; (i = Lindex[pos]) >= 0; pos++)
            let mut pos = Lbegin_p[k];
            while Lindex[pos] >= 0 {
                let i = Lindex[pos];
                x += lhs[i] * Lvalue[pos];
                pos += 1;
            }
            lhs[p[k]] -= x;
        }
    } else {
        // Solve forward system //

        // memcpy(work1, rhs, m*sizeof(double));
        work1[..m].copy_from_slice(rhs);

        // Solve with L.
        for k in 0..m {
            let mut x = 0.0;
            let mut pos = Ltbegin_p[k];
            while Lindex[pos] >= 0 {
                let i = Lindex[pos];
                x += work1[i] * Lvalue[pos];
                pos += 1;
            }
            work1[p[k]] -= x;
        }

        // Solve with update ETAs.
        let mut pos = Rbegin[0];
        for t in 0..nforrest {
            let ipivot = eta_row[t];
            let mut x = 0.0;
            while pos < Rbegin[t + 1] {
                x += work1[Lindex[pos]] * Lvalue[pos];
                pos += 1;
            }
            work1[ipivot] -= x;
        }

        // Solve with U.
        // for (k = m-1; k >= 0; k--)
        for k in (0..m).rev() {
            let jpivot = pivotcol[k];
            let ipivot = pivotrow[k];
            let x = work1[ipivot] / row_pivot[ipivot];
            // for (pos = Ubegin[ipivot]; (i = Uindex[pos]) >= 0; pos++)
            let mut pos = Ubegin[ipivot];
            while Uindex[pos] >= 0 {
                let i = Uindex[pos];
                work1[i] -= x * Uvalue[pos];
                pos += 1;
            }
            lhs[jpivot] = x;
        }
    }
}
