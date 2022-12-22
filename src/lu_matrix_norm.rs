// Copyright (C) 2016-2018  ERGO-Code
//
// Computes the 1-norm and infinity-norm of the matrix that was freshly
// factorized. Unit cols inserted by the factorization are handled implicitly.

use crate::basiclu::lu_int;
use crate::lu_internal::lu;

pub(crate) fn lu_matrix_norm(
    this: &mut lu,
    Bbegin: &[lu_int],
    Bend: &[lu_int],
    Bi: &[lu_int],
    Bx: &[f64],
) {
    let m = this.m;
    let rank = this.rank;
    let pivotcol = this.pivotcol.as_ref().unwrap();
    let pivotrow = this.pivotrow.as_ref().unwrap();
    let rowsum = &mut this.work1;

    assert_eq!(this.nupdate, 0);

    for i in 0..m {
        rowsum[i as usize] = 0.0;
    }
    let mut onenorm = 0.0;
    let mut infnorm = 0.0;
    for k in 0..rank {
        let jpivot = pivotcol[k as usize] as usize;
        let mut colsum = 0.0;
        for pos in Bbegin[jpivot]..Bend[jpivot] {
            colsum += Bx[pos as usize].abs();
            rowsum[Bi[pos as usize] as usize] += Bx[pos as usize].abs();
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

    this.onenorm = onenorm;
    this.infnorm = infnorm;
}
