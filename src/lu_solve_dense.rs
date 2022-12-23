// Copyright (C) 2016-2018  ERGO-Code

use crate::basiclu::lu_int;
use crate::lu_garbage_perm::lu_garbage_perm;
use crate::lu_internal::lu;

pub(crate) fn lu_solve_dense(
    this: &mut lu,
    rhs: &[f64],
    lhs: &mut [f64],
    trans: char,
    Li: &[lu_int],
    Lx: &[f64],
    Ui: &[lu_int],
    Ux: &[f64],
    Wi: &[lu_int],
    Wx: &[f64],
) {
    lu_garbage_perm(this);
    assert_eq!(this.pivotlen, this.m);

    let m = this.m;
    let nforrest = this.nforrest;
    let p = &this.solve.p;
    let eta_row = &this.solve.eta_row;
    let pivotcol = &this.solve.pivotcol;
    let pivotrow = &this.solve.pivotrow;
    let Lbegin_p = &this.solve.Lbegin_p;
    let Ltbegin_p = &this.solve.Ltbegin_p;
    let Ubegin = &this.solve.Ubegin;
    let Rbegin = &this.solve.Rbegin;
    let Wbegin = &this.factor.Wbegin;
    let Wend = &this.factor.Wend;
    let col_pivot = &this.xstore.col_pivot;
    let row_pivot = &this.xstore.row_pivot;
    let Lindex = Li;
    let Lvalue = Lx;
    let Uindex = Ui;
    let Uvalue = Ux;
    let Windex = Wi;
    let Wvalue = Wx;
    // let Lindex = this.Lindex.as_ref().unwrap();
    // let Lvalue = this.Lvalue.as_ref().unwrap();
    // let Uindex = this.Uindex.as_ref().unwrap();
    // let Uvalue = this.Uvalue.as_ref().unwrap();
    // let Windex = this.Windex.as_ref().unwrap();
    // let Wvalue = this.Wvalue.as_ref().unwrap();
    let work1 = &mut this.xstore.work1;

    if trans == 't' || trans == 'T' {
        // Solve transposed system

        // memcpy(work1, rhs, m*sizeof(double));
        work1[..m as usize].copy_from_slice(rhs);

        // Solve with U'.
        for k in 0..m {
            let jpivot = pivotcol[k as usize] as usize;
            let ipivot = pivotrow[k as usize] as usize;
            let x = work1[jpivot] / col_pivot[jpivot];
            for pos in Wbegin[jpivot]..Wend[jpivot] {
                work1[Windex[pos as usize] as usize] -= x * Wvalue[pos as usize];
            }
            lhs[ipivot] = x;
        }

        // Solve with update ETAs backwards.
        // for (t = nforrest-1; t >= 0; t--)
        for t in (0..nforrest).rev() {
            let ipivot = eta_row[t as usize];
            let x = lhs[ipivot as usize];
            for pos in Rbegin[t as usize]..Rbegin[(t + 1) as usize] {
                let i = Lindex[pos as usize] as usize;
                lhs[i] -= x * Lvalue[pos as usize];
            }
        }

        // Solve with L'.
        // for (k = m-1; k >= 0; k--)
        for k in (0..m).rev() {
            let mut x = 0.0;
            // for (pos = Lbegin_p[k]; (i = Lindex[pos]) >= 0; pos++)
            let mut pos = Lbegin_p[k as usize] as usize;
            while Lindex[pos] >= 0 {
                let i = Lindex[pos];
                x += lhs[i as usize] * Lvalue[pos];
                pos += 1;
            }
            lhs[p[k as usize] as usize] -= x;
        }
    } else {
        // Solve forward system //

        // memcpy(work1, rhs, m*sizeof(double));
        work1[..m as usize].copy_from_slice(rhs);

        // Solve with L.
        for k in 0..m {
            let mut x = 0.0;
            let mut pos = Ltbegin_p[k as usize] as usize;
            while Lindex[pos] >= 0 {
                let i = Lindex[pos];
                x += work1[i as usize] * Lvalue[pos];
                pos += 1;
            }
            work1[p[k as usize] as usize] -= x;
        }

        // Solve with update ETAs.
        let mut pos = Rbegin[0];
        for t in 0..nforrest as usize {
            let ipivot = eta_row[t];
            let mut x = 0.0;
            while pos < Rbegin[t + 1] {
                x += work1[Lindex[pos as usize] as usize] * Lvalue[pos as usize];
                pos += 1;
            }
            work1[ipivot as usize] -= x;
        }

        // Solve with U.
        // for (k = m-1; k >= 0; k--)
        for k in (0..m).rev() {
            let jpivot = pivotcol[k as usize] as usize;
            let ipivot = pivotrow[k as usize] as usize;
            let x = work1[ipivot] / row_pivot[ipivot];
            // for (pos = Ubegin[ipivot]; (i = Uindex[pos]) >= 0; pos++)
            let mut pos = Ubegin[ipivot] as usize;
            while Uindex[pos] >= 0 {
                let i = Uindex[pos];
                work1[i as usize] -= x * Uvalue[pos];
                pos += 1;
            }
            lhs[jpivot] = x;
        }
    }
}
