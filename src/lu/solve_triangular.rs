// Copyright (C) 2016-2018  ERGO-Code

use crate::blu::LUInt;

// Substitution with triangular matrix.
//
// The symbolic nonzero pattern of the solution must be given in topological
// order in `pattern_symb[0..nz_symb-1]`. On return `pattern[0..nz-1]` holds the
// nonzero pattern of the solution after dropping numerical zeros; `nz` is
// returned. `pattern` and `pattern_symb` can point to the same array.
//
// Entries in the solution that are less than or equal to `droptol` are set to
// zero. When `droptol` is zero or negative, then no entries will be set to zero.
//
// Note: The nonzero pattern of the solution never includes zeros. That means,
//       even if `droptol` is negative, the output pattern is not identical to
//       the symbolic pattern when exact cancellation happens.
//
// The pivot elements must be stored separately to the matrix. When `pivot` is
// `None`, then the pivot elements are assumed to be 1. The matrix is given in
// parallel arrays index, value. When `end` is not `None`, column `j` has elements
//
//     index[begin[j]..end[j]-1], value[begin[j]..end[j]-1].
//
// When `end` is `None`, then each column must be terminated by a negative index.
pub(crate) fn solve_triangular(
    nz_symb: LUInt,
    pattern_symb: &[LUInt],
    begin: &[LUInt],
    end: Option<&[LUInt]>,
    index: &[LUInt],
    value: &[f64],
    pivot: Option<&[f64]>,
    droptol: f64,
    lhs: &mut [f64], // solution overwrites RHS
    pattern: &mut [LUInt],
    flops: &mut LUInt, // add flop count
) -> LUInt {
    let mut nz: LUInt = 0;
    let mut flop_count = 0;

    if pivot.is_some() && end.is_some() {
        let pivot = pivot.unwrap();
        let end = end.unwrap();

        for n in 0..nz_symb {
            let ipivot = pattern_symb[n as usize];
            if lhs[ipivot as usize] != 0.0 {
                // x = lhs[ipivot] /= pivot[ipivot];
                lhs[ipivot as usize] /= pivot[ipivot as usize];
                let x = lhs[ipivot as usize];

                flop_count += 1;
                for pos in begin[ipivot as usize]..end[ipivot as usize] {
                    let i = index[pos as usize];
                    lhs[i as usize] -= x * value[pos as usize];
                    flop_count += 1;
                }
                if x.abs() > droptol {
                    pattern[nz as usize] = ipivot;
                    nz += 1;
                } else {
                    lhs[ipivot as usize] = 0.0;
                }
            }
        }
    } else if let Some(pivot) = pivot {
        for n in 0..nz_symb {
            let ipivot = pattern_symb[n as usize];
            if lhs[ipivot as usize] != 0.0 {
                // let x = lhs[ipivot] /= pivot[ipivot]; TODO check
                lhs[ipivot as usize] /= pivot[ipivot as usize];
                let x = lhs[ipivot as usize];
                flop_count += 1;
                // for (pos = begin[ipivot]; (i = index[pos]) >= 0; pos++)
                let mut pos = begin[ipivot as usize];
                while index[pos as usize] >= 0 {
                    let i = index[pos as usize];
                    lhs[i as usize] -= x * value[pos as usize];
                    flop_count += 1;
                    pos += 1;
                }
                if x.abs() > droptol {
                    pattern[nz as usize] = ipivot;
                    nz += 1;
                } else {
                    lhs[ipivot as usize] = 0.0;
                }
            }
        }
    } else if let Some(end) = end {
        for n in 0..nz_symb {
            let ipivot = pattern_symb[n as usize];
            if lhs[ipivot as usize] != 0.0 {
                let x = lhs[ipivot as usize];
                // for (pos = begin[ipivot]; pos < end[ipivot]; pos++)
                for pos in begin[ipivot as usize]..end[ipivot as usize] {
                    let i = index[pos as usize];
                    lhs[i as usize] -= x * value[pos as usize];
                    flop_count += 1;
                }
                if x.abs() > droptol {
                    pattern[nz as usize] = ipivot;
                    nz += 1;
                } else {
                    lhs[ipivot as usize] = 0.0;
                }
            }
        }
    } else {
        for n in 0..nz_symb {
            let ipivot = pattern_symb[n as usize];
            if lhs[ipivot as usize] != 0.0 {
                let x = lhs[ipivot as usize];
                // for (pos = begin[ipivot]; (i = index[pos]) >= 0; pos++)
                let mut pos = begin[ipivot as usize];
                while index[pos as usize] >= 0 {
                    let i = index[pos as usize];
                    lhs[i as usize] -= x * value[pos as usize];
                    flop_count += 1;
                    pos += 1;
                }
                if x.abs() > droptol {
                    pattern[nz as usize] = ipivot;
                    nz += 1;
                } else {
                    lhs[ipivot as usize] = 0.0;
                }
            }
        }
    }

    *flops += flop_count;
    nz
}
