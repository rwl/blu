use crate::basiclu::*;
use crate::lu_internal::lu;
use crate::lu_list::lu_list_remove;
use crate::lu_markowitz::lu_markowitz;
use crate::lu_pivot::lu_pivot;

/// Bump factorization driver routine.
pub(crate) fn lu_factorize_bump(
    this: &mut lu,
    Li: &mut [lu_int],
    Lx: &mut [f64],
    Ui: &mut [lu_int],
    Ux: &mut [f64],
    Wi: &mut [lu_int],
    Wx: &mut [f64],
) -> lu_int {
    let m = this.m;
    let mut status = BASICLU_OK;

    while this.rank + this.rankdef < m {
        // Find pivot element. Markowitz search need not be called if the
        // previous call to lu_pivot() returned for reallocation. In this case
        // this.pivot_col is valid.
        if this.pivot_col < 0 {
            lu_markowitz(this, Wi, Wx);
        }
        assert!(this.pivot_col >= 0);

        if this.pivot_row < 0 {
            // Eliminate empty column without choosing a pivot.
            lu_list_remove(
                &mut this.colcount_flink,
                &mut this.colcount_blink,
                this.pivot_col,
            );
            this.pivot_col = -1;
            this.rankdef += 1;
        } else {
            // Eliminate pivot. This may require reallocation.
            assert_eq!(this.pinv[this.pivot_row as usize], -1);
            assert_eq!(this.qinv[this.pivot_col as usize], -1);
            status = lu_pivot(this, Li, Lx, Ui, Ux, Wi, Wx);
            if status != BASICLU_OK {
                break;
            }
            this.pinv[this.pivot_row as usize] = this.rank;
            this.qinv[this.pivot_col as usize] = this.rank;
            this.pivot_col = -1;
            this.pivot_row = -1;
            this.rank += 1;
        }
    }
    status
}
