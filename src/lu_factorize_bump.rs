use crate::basiclu::*;
use crate::lu_internal::lu;
use crate::lu_list::lu_list_remove;
use crate::lu_markowitz::lu_markowitz;

/// Bump factorization driver routine.
pub(crate) fn lu_factorize_bump(this: &mut lu) -> lu_int {
    let m = this.m;
    let colcount_flink = this.colcount_flink;
    let colcount_blink = this.colcount_blink;
    let pinv = this.pinv;
    let qinv = this.qinv;
    let mut status = BASICLU_OK;

    while this.rank + this.rankdef < m {
        // Find pivot element. Markowitz search need not be called if the
        // previous call to lu_pivot() returned for reallocation. In this case
        // this.pivot_col is valid.
        if this.pivot_col < 0 {
            lu_markowitz(this);
        }
        assert!(this.pivot_col >= 0);

        if this.pivot_row < 0 {
            // Eliminate empty column without choosing a pivot.
            lu_list_remove(colcount_flink, colcount_blink, this.pivot_col);
            this.pivot_col = -1;
            this.rankdef += 1;
        } else {
            // Eliminate pivot. This may require reallocation.
            assert_eq!(pinv[this.pivot_row], -1);
            assert_eq!(qinv[this.pivot_col], -1);
            status = lu_pivot(this);
            if status != BASICLU_OK {
                break;
            }
            pinv[this.pivot_row] = this.rank;
            qinv[this.pivot_col] = this.rank;
            this.pivot_col = -1;
            this.pivot_row = -1;
            this.rank += 1;
        }
    }
    status
}
