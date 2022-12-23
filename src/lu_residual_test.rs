// Copyright (C) 2016-2018  ERGO-Code
//
// Stability test of fresh LU factorization based on relative residual.

use crate::basiclu::lu_int;
use crate::lu_internal::lu;
use crate::lu_matrix_norm::lu_matrix_norm;

fn lu_onenorm(m: lu_int, x: &[f64]) -> f64 {
    let mut d = 0.0;
    for i in 0..m {
        d += x[i as usize].abs();
    }
    d
}

pub(crate) fn lu_residual_test(
    this: &mut lu,
    Bbegin: &[lu_int],
    Bend: &[lu_int],
    Bi: &[lu_int],
    Bx: &[f64],
    Li: &[lu_int],
    Lx: &[f64],
    Ui: &[lu_int],
    Ux: &[f64],
) {
    let m = this.m;
    let rank = this.rank;
    let p = this.p.as_ref().unwrap();
    let pivotcol = this.pivotcol.as_ref().unwrap();
    let pivotrow = this.pivotrow.as_ref().unwrap();
    let Lbegin_p = &this.Lbegin_p;
    let Ltbegin_p = this.Ltbegin_p.as_ref().unwrap();
    let Ubegin = &this.Ubegin;
    let row_pivot = &this.row_pivot;
    let Lindex = Li;
    let Lvalue = Lx;
    let Uindex = Ui;
    let Uvalue = Ux;
    // let Lindex = this.Lindex.as_ref().unwrap();
    // let Lvalue = this.Lvalue.as_ref().unwrap();
    // let Uindex = this.Uindex.as_ref().unwrap();
    // let Uvalue = this.Uvalue.as_ref().unwrap();
    let rhs = &mut this.work0;
    let lhs = &mut this.work1;

    // lu_int i, k, ipivot, jpivot, pos;
    // double norm_ftran, norm_ftran_res, norm_btran, norm_btran_res, d;

    assert_eq!(this.nupdate, 0);

    // Residual Test with Forward System //

    // Compute lhs = L\rhs and build rhs on-the-fly.
    for k in 0..m as usize {
        let mut d = 0.0;
        // for (pos = Ltbegin_p[k]; (i = Lindex[pos]) >= 0; pos++)
        let mut pos = Ltbegin_p[k] as usize;
        while Lindex[pos] >= 0 {
            let i = Lindex[pos];
            d += lhs[i as usize] * Lvalue[pos];
            pos += 1;
        }
        let ipivot = p[k];
        rhs[ipivot as usize] = if d <= 0.0 { 1.0 } else { -1.0 };
        lhs[ipivot as usize] = rhs[ipivot as usize] - d;
    }

    // Overwrite lhs by U\lhs.
    // for (k = m-1; k >= 0; k--) TODO: check
    for k in (0..m as usize).rev() {
        let ipivot = pivotrow[k];
        lhs[ipivot as usize] /= row_pivot[ipivot as usize]; // TODO: check
        let d = lhs[ipivot as usize];
        // for (pos = Ubegin[ipivot]; (i = Uindex[pos]) >= 0; pos++)
        let mut pos = Ubegin[ipivot as usize] as usize;
        while Uindex[pos] >= 0 {
            let i = Uindex[pos];
            lhs[i as usize] -= d * Uvalue[pos];
            pos += 1;
        }
    }

    // Overwrite rhs by the residual rhs-B*lhs.
    for k in 0..rank {
        let ipivot = pivotrow[k as usize];
        let jpivot = pivotcol[k as usize];
        let d = lhs[ipivot as usize];
        for pos in Bbegin[jpivot as usize]..Bend[jpivot as usize] {
            rhs[Bi[pos as usize] as usize] -= d * Bx[pos as usize];
        }
    }
    for k in rank..m {
        let ipivot = pivotrow[k as usize] as usize;
        rhs[ipivot] -= lhs[ipivot];
    }
    let norm_ftran = lu_onenorm(m, lhs);
    let norm_ftran_res = lu_onenorm(m, rhs);

    // Residual Test with Backward System //

    // Compute lhs = U'\rhs and build rhs on-the-fly.
    for k in 0..m as usize {
        let ipivot = pivotrow[k] as usize;
        let mut d = 0.0;
        // for (pos = Ubegin[ipivot]; (i = Uindex[pos]) >= 0; pos++)
        let mut pos = Ubegin[ipivot] as usize;
        while Uindex[pos] >= 0 {
            let i = Uindex[pos];
            d += lhs[i as usize] * Uvalue[pos];
            pos += 1;
        }
        rhs[ipivot] = if d <= 0.0 { 1.0 } else { -1.0 };
        lhs[ipivot] = (rhs[ipivot] - d) / row_pivot[ipivot];
    }

    // Overwrite lhs by L'\lhs.
    // for (k = m-1; k >= 0; k--)
    for k in (0..m as usize).rev() {
        let mut d = 0.0;
        // for (pos = Lbegin_p[k]; (i = Lindex[pos]) >= 0; pos++)
        let mut pos = Lbegin_p[k] as usize;
        while Lindex[pos] >= 0 {
            let i = Lindex[pos];
            d += lhs[i as usize] * Lvalue[pos];
            pos += 1;
        }
        lhs[p[k] as usize] -= d;
    }

    // Overwrite rhs by the residual rhs-B'*lhs.
    for k in 0..rank as usize {
        let ipivot = pivotrow[k] as usize;
        let jpivot = pivotcol[k] as usize;
        let mut d = 0.0;
        for pos in Bbegin[jpivot]..Bend[jpivot] {
            d += lhs[Bi[pos as usize] as usize] * Bx[pos as usize];
        }
        rhs[ipivot] -= d;
    }
    for k in rank..m {
        let ipivot = pivotrow[k as usize];
        rhs[ipivot as usize] -= lhs[ipivot as usize];
    }
    let norm_btran = lu_onenorm(m, lhs);
    let norm_btran_res = lu_onenorm(m, rhs);

    // Finalize //

    lu_matrix_norm(this, Bbegin, Bend, Bi, Bx);
    assert!(this.onenorm > 0.0);
    assert!(this.infnorm > 0.0);
    this.residual_test = f64::max(
        norm_ftran_res / ((m as f64) + this.onenorm * norm_ftran),
        norm_btran_res / ((m as f64) + this.infnorm * norm_btran),
    );

    // reset workspace
    for i in 0..m {
        // rhs[i as usize] = 0.0;
        this.work0[i as usize] = 0.0;
    }
}
