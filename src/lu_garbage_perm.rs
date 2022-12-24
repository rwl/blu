// Copyright (C) 2016-2018  ERGO-Code

use crate::lu_internal::*;

// The sequence of pivot columns and pivot rows is stored in
//
//  pivotcol[0..pivotlen-1], pivotrow[0..pivotlen-1],
//
// where pivotlen >= m. When pivotlen > m, then the arrays contain duplicates.
// For each index its last occurence in the arrays is its position in the pivot
// sequence and occurences before mark unused slots.
//
// This routine removes duplicates and compresses the indices such that
// pivotlen == m.
pub(crate) fn lu_garbage_perm(this: &mut lu) {
    let m = this.m;
    let pivotlen = this.pivotlen;
    let pivotcol = &mut pivotcol!(this);
    let pivotrow = &mut pivotrow!(this);
    let marked = &mut marked!(this);

    if pivotlen > m {
        // marker = ++this.marker;
        this.marker += 1;
        let marker = this.marker;
        let mut put = pivotlen;
        // for (get = pivotlen-1; get >= 0; get--) {
        for get in (0..pivotlen).rev() {
            if marked[pivotcol[get as usize] as usize] != marker {
                let j = pivotcol[get as usize];
                marked[j as usize] = marker;
                // pivotcol[--put] = j;
                put -= 1;
                pivotcol[put as usize] = j;
                pivotrow[put as usize] = pivotrow[get as usize];
            }
        }
        assert_eq!(put + m, pivotlen);

        // memmove(pivotcol, pivotcol + put, m * sizeof(lu_int));  TODO: check
        pivotcol.copy_within((put as usize)..(put + m) as usize, 0);
        // memmove(pivotrow, pivotrow + put, m * sizeof(lu_int));
        pivotrow.copy_within((put as usize)..(put + m) as usize, 0);

        this.pivotlen = m;
    }
}
