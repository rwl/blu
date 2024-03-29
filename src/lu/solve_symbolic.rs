// Copyright (C) 2016-2018 ERGO-Code
// Copyright (C) 2022-2023 Richard Lincoln

use crate::lu::dfs::dfs;
use crate::LUInt;

// Symbolic solve with triangular matrix.
//
// The pattern of the right-hand side is given in `irhs[0..nrhs-1]`. The
// pattern of the solution is returned in `ilhs[top..m-1]` in topological
// order, top is the function return value.
//
// When `end` is not `None`, then the pattern of column `j` of the matrix must be
// given in `index[begin[j]..end[j]-1]`. When end is `None`, then each column must
// be terminated by a negative index.
//
// The method is due to J. Gilbert and T. Peierls, "Sparse partial pivoting
// in time proportional to arithmetic operations", (1988).
pub(crate) fn solve_symbolic(
    m: usize,
    begin: &[LUInt],
    end: Option<&[LUInt]>,
    index: &[LUInt],
    nrhs: usize,
    irhs: &[LUInt],
    ilhs: &mut [LUInt],
    // pstack: &[lu_int], // size m workspace
    pstack: &mut [f64],   // size m workspace
    marked: &mut [LUInt], // marked[i] != M on entry
    marker: LUInt,
) -> usize {
    let mut top = m;
    for n in 0..nrhs {
        if marked[irhs[n] as usize] != marker {
            let i = irhs[n] as usize;
            top = dfs(i, begin, end, index, top, ilhs, pstack, marked, marker);
        }
    }
    top
}
