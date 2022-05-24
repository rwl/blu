// Copyright (C) 2016-2018  ERGO-Code
//
// Data file implementation
//
// A data file stores lines of (index,value) pairs. Entries of each line are
// contiguous in memory. Lines can be in any order in memory and there can be
// gaps between consecutive lines.
//
// The data file implementation uses arrays
//
//     index[0:fmem-1], value[0:fmem-1],
//     begin[0:nlines], end[0:nlines],
//     next[0:nlines], prev[0:nlines]
//
//     index, value    storing (index,value) pairs
//     begin[k]        pointer to first element in line 0 <= k < nlines,
//     end[k]          pointer to one past the last element in line k.
//     begin[nlines]   pointer to the first element of unused space
//     end[nlines]     holds fmem
//
// next, prev hold the line numbers (0..nlines-1) in a double linked list in
// the order in which they appear in memory. That is, for line 0 <= k < nlines,
// next[k] and prev[k] are the line which comes before respectively after line
// k in memory. next[nlines] and prev[nlines] are the first respectively last
// line in memory order.

use crate::basiclu::lu_int;
use crate::lu_list::lu_list_move;

/// Initialize empty file with @fmem memory space.
pub(crate) fn lu_file_empty(
    nlines: lu_int,
    begin: &mut [lu_int],
    end: &mut [lu_int],
    next: &mut [lu_int],
    prev: &mut [lu_int],
    fmem: lu_int,
) {
    begin[nlines as usize] = 0;
    end[nlines as usize] = fmem;
    for i in 0..nlines {
        begin[i as usize] = 0;
        end[i as usize] = 0;
    }
    for i in 0..nlines {
        next[i as usize] = i + 1;
        prev[i as usize + 1] = i;
    }
    next[nlines as usize] = 0;
    prev[0] = nlines;
}

/// Reappend line to file end and add @extra_space elements room. The file must
/// have at least length(line) + @extra_space elements free space.
pub(crate) fn lu_file_reappend(
    line: lu_int,
    nlines: lu_int,
    begin: &mut [lu_int],
    end: &mut [lu_int],
    next: &mut [lu_int],
    prev: &mut [lu_int],
    index: &mut [lu_int],
    value: &mut [f64],
    extra_space: lu_int,
) {
    let fmem = end[nlines as usize];
    let mut used = begin[nlines as usize];
    let room = fmem - used;
    let ibeg = begin[line as usize]; // old beginning of line
    let iend = end[line as usize];
    begin[line as usize] = used; // new beginning of line
    assert!(iend - ibeg <= room);
    for pos in ibeg..iend {
        index[used as usize] = index[pos as usize];
        value[used as usize] = value[pos as usize];
        used += 1
    }
    end[line as usize] = used;
    let room = fmem - used;
    assert!(room >= extra_space);
    used += extra_space;
    begin[nlines as usize] = used; // beginning of unused space
    lu_list_move(line, 0, next, prev, nlines, None);
}

/// Compress file to reuse memory gaps. The ordering of lines in the file is
/// unchanged. To each line with nz entries add @stretch*nz+@pad elements extra
/// space. Chop extra space if it would overlap the following line in memory.
///
/// Return: number of entries in file
pub(crate) fn lu_file_compress(
    nlines: lu_int,
    begin: &mut [lu_int],
    end: &mut [lu_int],
    next: &[lu_int],
    index: &mut [lu_int],
    value: &mut [f64],
    stretch: f64,
    pad: lu_int,
) -> lu_int {
    let mut nz = 0;

    let mut used = 0;
    let mut extra_space = 0;
    let mut i = next[nlines as usize];
    while i < nlines {
        // move line i
        let ibeg = begin[i as usize];
        let iend = end[i as usize];
        assert!(ibeg >= used);
        used += extra_space;
        if used > ibeg {
            used = ibeg; // chop extra space added before
        }
        begin[i as usize] = used;
        for pos in ibeg..iend {
            index[used as usize] = index[pos as usize];
            value[used as usize] = value[pos as usize];
            used += 1;
        }
        end[i as usize] = used;
        extra_space = (stretch as lu_int) * (iend - ibeg) + pad;
        nz += iend - ibeg;

        i = next[i as usize];
    }
    assert!(used <= begin[nlines as usize]);
    used += extra_space;
    if used > begin[nlines as usize] {
        used = begin[nlines as usize]; // never use more space than before
    }
    begin[nlines as usize] = used;
    nz
}

// lu_file_diff (for debugging)
//
// @begin_row, @end_row, @begin_col, @end_col are pointer into @index, @value,
// defining lines of the "row file" and the "column file".
//
// Task:
//
//  val == NULL: count row file entries that are missing in column file.
//  val != NULL: count row file entries that are missing in column file
//               or which values are different.
//
// The method does a column file search for each row file entry. To check
// consistency of rowwise and columnwise storage, the method must be called
// twice with row pointers and column pointers swapped.
pub(crate) fn lu_file_diff(
    nrow: lu_int,
    begin_row: &[lu_int],
    end_row: &[lu_int],
    begin_col: &[lu_int],
    end_col: &[lu_int],
    index: &[lu_int],
    value: Option<&[f64]>,
) -> lu_int {
    let mut ndiff = 0;

    for i in 0..nrow {
        for pos in begin_row[i as usize]..end_row[i as usize] {
            let j = index[pos as usize] as usize;
            let mut where_ = begin_col[j];
            while where_ < end_col[j] && index[where_ as usize] != i {
                where_ += 1;
            }

            // if where_ == end_col[j] || (value && value[pos] != value[where_]) {
            if where_ == end_col[j] {
                ndiff += 1;
            } else if let Some(value) = value {
                if value[pos as usize] != value[where_ as usize] {
                    ndiff += 1;
                }
            }
        }
    }
    ndiff
}
