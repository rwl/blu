// Copyright (C) 2016-2018 ERGO-Code
// Copyright (C) 2022-2023 Richard Lincoln

use crate::lu::lu::*;
use crate::lu::matrix_norm::matrix_norm;

fn onenorm(m: usize, x: &[f64]) -> f64 {
    let mut d = 0.0;
    for i in 0..m {
        d += x[i].abs();
    }
    d
}

// Stability test of fresh LU factorization based on relative residual.
pub(crate) fn residual_test(
    lu: &mut LU,
    b_begin: &[usize],
    b_end: &[usize],
    b_i: &[usize],
    b_x: &[f64],
) {
    let m = lu.m;
    let rank = lu.rank;
    let p = &p!(lu);
    let pivotcol = &pivotcol!(lu);
    let pivotrow = &pivotrow!(lu);
    let l_begin_p = &lu.l_begin_p;
    let lt_begin_p = &lt_begin_p!(lu);
    let u_begin = &lu.u_begin;
    let row_pivot = &lu.row_pivot;
    let l_index = &lu.l_index;
    let l_value = &lu.l_value;
    let u_index = &lu.u_index;
    let u_value = &lu.u_value;
    let rhs = &mut lu.work0;
    let lhs = &mut lu.work1;

    assert_eq!(lu.nupdate.unwrap(), 0);

    // Residual Test with Forward System //

    // Compute lhs = L\rhs and build rhs on-the-fly.
    for k in 0..m as usize {
        let mut d = 0.0;
        // for (pos = lt_begin_p[k]; (i = Lindex[pos]) >= 0; pos++)
        let mut pos = lt_begin_p[k] as usize;
        while l_index[pos] >= 0 {
            let i = l_index[pos];
            d += lhs[i as usize] * l_value[pos];
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
        let mut pos = u_begin[ipivot as usize] as usize;
        while u_index[pos] >= 0 {
            let i = u_index[pos];
            lhs[i as usize] -= d * u_value[pos];
            pos += 1;
        }
    }

    // Overwrite rhs by the residual rhs-B*lhs.
    for k in 0..rank {
        let ipivot = pivotrow[k as usize];
        let jpivot = pivotcol[k as usize];
        let d = lhs[ipivot as usize];
        for pos in b_begin[jpivot as usize]..b_end[jpivot as usize] {
            rhs[b_i[pos as usize] as usize] -= d * b_x[pos as usize];
        }
    }
    for k in rank..m {
        let ipivot = pivotrow[k as usize] as usize;
        rhs[ipivot] -= lhs[ipivot];
    }
    let norm_ftran = onenorm(m, lhs);
    let norm_ftran_res = onenorm(m, rhs);

    // Residual Test with Backward System //

    // Compute lhs = U'\rhs and build rhs on-the-fly.
    for k in 0..m as usize {
        let ipivot = pivotrow[k] as usize;
        let mut d = 0.0;
        // for (pos = Ubegin[ipivot]; (i = Uindex[pos]) >= 0; pos++)
        let mut pos = u_begin[ipivot] as usize;
        while u_index[pos] >= 0 {
            let i = u_index[pos];
            d += lhs[i as usize] * u_value[pos];
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
        let mut pos = l_begin_p[k] as usize;
        while l_index[pos] >= 0 {
            let i = l_index[pos];
            d += lhs[i as usize] * l_value[pos];
            pos += 1;
        }
        lhs[p[k] as usize] -= d;
    }

    // Overwrite rhs by the residual rhs-B'*lhs.
    for k in 0..rank as usize {
        let ipivot = pivotrow[k] as usize;
        let jpivot = pivotcol[k] as usize;
        let mut d = 0.0;
        for pos in b_begin[jpivot]..b_end[jpivot] {
            d += lhs[b_i[pos as usize] as usize] * b_x[pos as usize];
        }
        rhs[ipivot] -= d;
    }
    for k in rank..m {
        let ipivot = pivotrow[k as usize];
        rhs[ipivot as usize] -= lhs[ipivot as usize];
    }
    let norm_btran = onenorm(m, lhs);
    let norm_btran_res = onenorm(m, rhs);

    // Finalize //

    matrix_norm(lu, b_begin, b_end, b_i, b_x);
    assert!(lu.onenorm > 0.0);
    assert!(lu.infnorm > 0.0);
    lu.residual_test = f64::max(
        norm_ftran_res / ((m as f64) + lu.onenorm * norm_ftran),
        norm_btran_res / ((m as f64) + lu.infnorm * norm_btran),
    );

    // reset workspace
    for i in 0..m {
        // rhs[i as usize] = 0.0;
        lu.work0[i as usize] = 0.0;
    }
}
