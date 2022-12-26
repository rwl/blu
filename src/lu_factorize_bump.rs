use crate::blu::*;
use crate::lu_internal::LU;
use crate::lu_list::lu_list_remove;
use crate::lu_markowitz::lu_markowitz;
use crate::lu_pivot::lu_pivot;

/// Bump factorization driver routine.
pub(crate) fn lu_factorize_bump(lu: &mut LU) -> LUInt {
    let m = lu.m;
    let mut status = BLU_OK;

    while lu.rank + lu.rankdef < m {
        // Find pivot element. Markowitz search need not be called if the
        // previous call to lu_pivot() returned for reallocation. In this case
        // this.pivot_col is valid.
        if lu.pivot_col < 0 {
            lu_markowitz(lu);
        }
        assert!(lu.pivot_col >= 0);

        if lu.pivot_row < 0 {
            // Eliminate empty column without choosing a pivot.
            lu_list_remove(&mut lu.colcount_flink, &mut lu.colcount_blink, lu.pivot_col);
            lu.pivot_col = -1;
            lu.rankdef += 1;
        } else {
            // Eliminate pivot. This may require reallocation.
            assert_eq!(lu.pinv[lu.pivot_row as usize], -1);
            assert_eq!(lu.qinv[lu.pivot_col as usize], -1);
            status = lu_pivot(lu);
            if status != BLU_OK {
                break;
            }
            lu.pinv[lu.pivot_row as usize] = lu.rank;
            lu.qinv[lu.pivot_col as usize] = lu.rank;
            lu.pivot_col = -1;
            lu.pivot_row = -1;
            lu.rank += 1;
        }
    }
    status
}
