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
    begin[nlines] = 0;
    end[nlines] = fmem;
    for i in 0..nlines {
        begin[i] = 0;
        end[i] = 0;
    }
    for i in 0..nlines {
        next[i] = i + 1;
        prev[i + 1] = i;
    }
    next[nlines] = 0;
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
    let fmem = end[nlines];
    let used = begin[nlines];
    let room = fmem - used;
    let ibeg = begin[line]; // old beginning of line
    let iend = end[line];
    begin[line] = used; // new beginning of line
    assert!(iend - ibeg <= room);
    for pos in ibeg..iend {
        index[used] = index[pos];
        value[used] = value[pos];
        used += 1
    }
    end[line] = used;
    room = fmem - used;
    assert!(room >= extra_space);
    used += extra_space;
    begin[nlines] = used; // beginning of unused space
    lu_list_move(line, 0, next, prev, nlines, NULL);
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
    let mut i = next[nlines];
    while i < nlines {
        // move line i
        ibeg = begin[i];
        iend = end[i];
        assert!(ibeg >= used);
        used += extra_space;
        if (used > ibeg) {
            used = ibeg; // chop extra space added before
        }
        begin[i] = used;
        for pos in ibeg..iend {
            index[used] = index[pos];
            value[used] = value[pos];
            used += 1;
        }
        end[i] = used;
        extra_space = stretch * (iend - ibeg) + pad;
        nz += iend - ibeg;

        i = next[i];
    }
    assert!(used <= begin[nlines]);
    used += extra_space;
    if (used > begin[nlines]) {
        used = begin[nlines]; // never use more space than before
    }
    begin[nlines] = used;
    nz
}
