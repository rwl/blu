// Copyright (C) 2016-2018 ERGO-Code
// Copyright (C) 2022-2023 Richard Lincoln

use crate::lu::lu::*;
use crate::LUInt;

// Computes the 1-norm and infinity-norm of the matrix that was freshly
// factorized. Unit cols inserted by the factorization are handled implicitly.
pub(crate) fn matrix_norm(
    lu: &mut LU,
    b_begin: &[LUInt],
    b_end: &[LUInt],
    b_i: &[LUInt],
    b_x: &[f64],
) {
    let m = lu.m;
    let rank = lu.rank;
    let pivotcol = &pivotcol!(lu);
    let pivotrow = &pivotrow!(lu);
    let rowsum = &mut lu.work1;

    assert_eq!(lu.nupdate, 0);

    for i in 0..m {
        rowsum[i as usize] = 0.0;
    }
    let mut onenorm = 0.0;
    let mut infnorm = 0.0;
    for k in 0..rank {
        let jpivot = pivotcol[k as usize] as usize;
        let mut colsum = 0.0;
        for pos in b_begin[jpivot]..b_end[jpivot] {
            colsum += b_x[pos as usize].abs();
            rowsum[b_i[pos as usize] as usize] += b_x[pos as usize].abs();
        }
        onenorm = f64::max(onenorm, colsum);
    }
    for k in rank..m {
        let ipivot = pivotrow[k as usize] as usize;
        rowsum[ipivot] += 1.0;
        onenorm = f64::max(onenorm, 1.0);
    }
    for i in 0..m {
        infnorm = f64::max(infnorm, rowsum[i as usize]);
    }

    lu.onenorm = onenorm;
    lu.infnorm = infnorm;
}
