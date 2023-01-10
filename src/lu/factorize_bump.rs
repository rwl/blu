// Copyright (C) 2016-2019 ERGO-Code
// Copyright (C) 2022-2023 Richard Lincoln

use crate::lu::list::list_remove;
use crate::lu::markowitz::markowitz;
use crate::lu::pivot::pivot;
use crate::lu::LU;
use crate::LUInt;
use crate::Status;

// Bump factorization driver routine.
pub(crate) fn factorize_bump(lu: &mut LU) -> Status {
    let m = lu.m;
    let mut status = Status::OK;

    while lu.rank + lu.rankdef < m {
        // Find pivot element. Markowitz search need not be called if the
        // previous call to pivot() returned for reallocation. In this case
        // this.pivot_col is valid.
        if lu.pivot_col.is_none() {
            markowitz(lu);
        }
        assert!(lu.pivot_col.is_some());

        if lu.pivot_row.is_none() {
            // Eliminate empty column without choosing a pivot.
            list_remove(
                &mut lu.colcount_flink,
                &mut lu.colcount_blink,
                lu.pivot_col.unwrap(),
            );
            lu.pivot_col = None;
            lu.rankdef += 1;
        } else {
            // Eliminate pivot. This may require reallocation.
            assert_eq!(lu.pinv[lu.pivot_row.unwrap()], -1);
            assert_eq!(lu.qinv[lu.pivot_col.unwrap()], -1);
            status = pivot(lu);
            if status != Status::OK {
                break;
            }
            lu.pinv[lu.pivot_row.unwrap()] = lu.rank as LUInt;
            lu.qinv[lu.pivot_col.unwrap()] = lu.rank as LUInt;
            lu.pivot_col = None;
            lu.pivot_row = None;
            lu.rank += 1;
        }
    }
    status
}
