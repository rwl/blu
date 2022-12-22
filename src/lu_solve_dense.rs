// Copyright (C) 2016-2018  ERGO-Code

use crate::lu_garbage_perm::lu_garbage_perm;
use crate::lu_internal::lu;

pub(crate) fn lu_solve_dense(this: &mut lu, rhs: &[f64], lhs: &mut [f64], trans: char) {
    lu_garbage_perm(this);
    assert_eq!(this.pivotlen, this.m);

    let m = this.m;
    let nforrest = this.nforrest;
    let p = this.p.as_ref().unwrap();
    let eta_row = &this.eta_row;
    let pivotcol = this.pivotcol.as_ref().unwrap();
    let pivotrow = this.pivotrow.as_ref().unwrap();
    let Lbegin_p = &this.Lbegin_p;
    let Ltbegin_p = this.Ltbegin_p.as_ref().unwrap();
    let Ubegin = &this.Ubegin;
    let Rbegin = this.Rbegin.as_ref().unwrap();
    let Wbegin = this.Wbegin.as_ref().unwrap();
    let Wend = this.Wend.as_ref().unwrap();
    let col_pivot = &this.col_pivot;
    let row_pivot = &this.row_pivot;
    let Lindex = this.Lindex.as_ref().unwrap();
    let Lvalue = this.Lvalue.as_ref().unwrap();
    let Uindex = this.Uindex.as_ref().unwrap();
    let Uvalue = this.Uvalue.as_ref().unwrap();
    let Windex = this.Windex.as_ref().unwrap();
    let Wvalue = this.Wvalue.as_ref().unwrap();
    let work1 = &mut this.work1;

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
