// Copyright (C) 2016-2018 ERGO-Code
// Copyright (C) 2022-2023 Richard Lincoln

use crate::lu::list::list_move;
use crate::LUInt;

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
// `next`, `prev` hold the line numbers (`0..nlines-1`) in a double linked list in
// the order in which they appear in memory. That is, for line `0 <= k < nlines`,
// `next[k]` and `prev[k]` are the line which comes before respectively after line
// `k` in memory. `next[nlines]` and `prev[nlines]` are the first respectively last
// line in memory order.

// Initialize empty file with `fmem` memory space.
pub(crate) fn file_empty(
    nlines: usize,
    begin: &mut [LUInt],
    end: &mut [LUInt],
    next: &mut [LUInt],
    prev: &mut [LUInt],
    fmem: LUInt,
) {
    begin[nlines as usize] = 0;
    end[nlines as usize] = fmem;
    for i in 0..nlines {
        begin[i as usize] = 0;
        end[i as usize] = 0;
    }
    for i in 0..nlines {
        next[i as usize] = (i + 1) as LUInt;
        prev[i as usize + 1] = i as LUInt;
    }
    next[nlines as usize] = 0;
    prev[0] = nlines as LUInt;
}

// Reappend line to file end and add `extra_space` elements room. The file must
// have at least length(line) + `extra_space` elements free space.
pub(crate) fn file_reappend(
    line: usize,
    nlines: usize,
    begin: &mut [LUInt],
    end: &mut [LUInt],
    next: &mut [LUInt],
    prev: &mut [LUInt],
    index: &mut [LUInt],
    value: &mut [f64],
    extra_space: usize,
) {
    let fmem = end[nlines];
    let mut used = begin[nlines];
    let room = fmem - used;
    let ibeg = begin[line]; // old beginning of line
    let iend = end[line];
    begin[line] = used; // new beginning of line
    assert!(iend - ibeg <= room);
    for pos in ibeg..iend {
        index[used as usize] = index[pos as usize];
        value[used as usize] = value[pos as usize];
        used += 1
    }
    end[line] = used;
    let room = fmem - used;
    assert!(room >= extra_space as LUInt);
    used += extra_space as LUInt;
    begin[nlines] = used; // beginning of unused space
    list_move(line, 0, next, prev, nlines, None);
}

// Compress file to reuse memory gaps. The ordering of lines in the file is
// unchanged. To each line with `nz` entries add `stretch*nz+pad` elements extra
// space. Chop extra space if it would overlap the following line in memory.
//
// Return: number of entries in file
pub(crate) fn file_compress(
    nlines: usize,
    begin: &mut [LUInt],
    end: &mut [LUInt],
    next: &[LUInt],
    index: &mut [LUInt],
    value: &mut [f64],
    stretch: f64,
    pad: usize,
) -> usize {
    let mut nz = 0;

    let mut used = 0;
    let mut extra_space = 0;
    let mut i = next[nlines as usize];
    while i < nlines as LUInt {
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
        extra_space = (stretch * (iend - ibeg) as f64) as LUInt + pad as LUInt;
        nz += (iend - ibeg) as usize;

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

// file_diff (for debugging)
//
// `begin_row`, `end_row`, `begin_col`, `end_col` are pointer into `index`, `value`,
// defining lines of the "row file" and the "column file".
//
// Task:
//
// - val == None: count row file entries that are missing in column file.
// - val != None: count row file entries that are missing in column file
//                or which values are different.
//
// The method does a column file search for each row file entry. To check
// consistency of rowwise and columnwise storage, the method must be called
// twice with row pointers and column pointers swapped.
pub(crate) fn file_diff(
    nrow: usize,
    begin_row: &[LUInt],
    end_row: &[LUInt],
    begin_col: &[LUInt],
    end_col: &[LUInt],
    index: &[LUInt],
    value: Option<&[f64]>,
) -> LUInt {
    let mut ndiff = 0;

    for i in 0..nrow {
        for pos in begin_row[i]..end_row[i] {
            let j = index[pos as usize] as usize;
            let mut where_ = begin_col[j];
            while where_ < end_col[j] && index[where_ as usize] != i as LUInt {
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
