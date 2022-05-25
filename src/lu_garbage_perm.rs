// Copyright (C) 2016-2018  ERGO-Code

use crate::lu_internal::lu;

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
    let pivotcol = &this.pivotcol;
    let pivotrow = &this.pivotrow;
    let marked = &this.marked;

    if pivotlen > m {
        // marker = ++this.marker;
        this.marker += 1;
        let marker = this.marker;
        let mut put = pivotlen;
        // for (get = pivotlen-1; get >= 0; get--) {
        for get in (0..pivotlen).rev() {
            if marked[pivotcol[get]] != marker {
                let j = pivotcol[get];
                marked[j] = marker;
                // pivotcol[--put] = j;
                put -= 1;
                pivotcol[put] = j;
                pivotrow[put] = pivotrow[get];
            }
        }
        assert_eq!(put + m, pivotlen);

        // memmove(pivotcol, pivotcol + put, m * sizeof(lu_int));  TODO: check
        pivotcol[..m].copy_from_slice(pivotcol[put..]);
        // memmove(pivotrow, pivotrow + put, m * sizeof(lu_int));
        pivotrow[..m].copy_from_slice(pivotrow[put..]);

        this.pivotlen = m;
    }
}
