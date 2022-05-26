// Copyright (C) 2016-2018  ERGO-Code
//
// Symbolic solve with triangular matrix

use crate::basiclu::lu_int;
use crate::lu_dfs::lu_dfs;

// The pattern of the right-hand side is given in irhs[0..nrhs-1]. The
// pattern of the solution is returned in ilhs[top..m-1] in topological
// order, top is the function return value.
//
// When end is not NULL, then the pattern of column j of the matrix must be
// given in index[begin[j]..end[j]-1]. When end is NULL, then each column must
// be terminated by a negative index.
//
// The method is due to J. Gilbert and T. Peierls, "Sparse partial pivoting
// in time proportional to arithmetic operations", (1988).
pub(crate) fn lu_solve_symbolic(
    m: lu_int,
    begin: &[lu_int],
    end: Option<&[lu_int]>,
    index: &[lu_int],
    nrhs: lu_int,
    irhs: &[lu_int],
    ilhs: &mut [lu_int],
    // pstack: &[lu_int], // size m workspace
    pstack: &mut [f64],    // size m workspace
    marked: &mut [lu_int], // marked[i] != M on entry
    M: lu_int,
) -> lu_int {
    let mut top = m;
    for n in 0..nrhs {
        if marked[irhs[n as usize] as usize] != M {
            let i = irhs[n as usize];
            top = lu_dfs(i, begin, end, index, top, ilhs, pstack, marked, M);
        }
    }
    top
}
