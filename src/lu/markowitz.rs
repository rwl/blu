// Copyright (C) 2016-2019 ERGO-Code
// Copyright (C) 2022-2023 Richard Lincoln

use crate::lu::list::list_move;
use crate::lu::LU;
use crate::{IntLeast64, LUInt, Status};
use std::time::Instant;

// Search for pivot element with small Markowitz cost. An eligible pivot
// must be nonzero and satisfy
//
// 1. `abs(piv) >= abstol`,
// 2. `abs(piv) >= reltol * max[pivot column]`.
//
// From all eligible pivots search for one that minimizes
//
//     mc := (nnz[pivot row] - 1) * (nnz[pivot column] - 1).
//
// The search is terminated when `maxsearch` rows or columns with eligible pivots
// have been searched (if not before). The row and column of the cheapest one
// found is stored in `pivot_row` and `pivot_col`.
//
// When the active submatrix contains columns with column count = 0, then such a
// column is chosen immediately and `pivot_row` = -1 is returned. Otherwise, when
// the Markowitz search does not find a pivot that is nonzero and >= `abstol`,
// then `pivot_col` = `pivot_row` = -1 is returned. (The latter cannot happen in the
// current version of the code because [`lu_pivot()`] erases columns of the active
// submatrix whose maximum absolute value drops below `abstol`.)
//
// The Markowitz search is implemented as described in [1].
//
// [1] U. Suhl, L. Suhl, "Computing Sparse LU Factorizations for Large-Scale
//     Linear Programming Bases", ORSA Journal on Computing (1990)
pub(crate) fn markowitz(lu: &mut LU) -> Status {
    let m = lu.m;
    let w_begin = &lu.w_begin;
    let w_end = &lu.w_end;
    let w_index = &lu.w_index;
    let w_value = &lu.w_value;
    let colcount_flink = &lu.colcount_flink;
    let rowcount_flink = &mut lu.rowcount_flink;
    let rowcount_blink = &mut lu.rowcount_blink;
    let colmax = &lu.col_pivot;
    let abstol = lu.abstol;
    let reltol = lu.reltol;
    let maxsearch = lu.maxsearch;
    let search_rows = lu.search_rows;
    let nz_start = if search_rows != 0 {
        usize::min(lu.min_colnz, lu.min_rownz)
    } else {
        lu.min_colnz
    };

    // lu_int i, j, pos, where_, inext, nz, pivot_row, pivot_col;
    // lu_int nsearch, cheap, found, min_colnz, min_rownz;
    // double cmx, x, tol, tic[2];

    // integers for Markowitz cost must be 64 bit to prevent overflow
    let m64 = m as IntLeast64;

    // lu_tic(tic);
    let tic = Instant::now();
    let mut pivot_row: Option<usize> = None; // row of best pivot so far
    let mut pivot_col: Option<usize> = None; // col of best pivot so far
    let mut mc64: IntLeast64 = m64 * m64; // Markowitz cost of best pivot so far
    let mut nsearch = 0; // count rows/columns searched
    let mut min_colnz: Option<usize> = None; // minimum col count in active submatrix
    let mut min_rownz: Option<usize> = None; // minimum row count in active submatrix
    assert!(nz_start >= 1);

    // If the active submatrix contains empty columns, choose one and return
    // with pivot_row = -1.
    if colcount_flink[m] != m as LUInt {
        pivot_col = Some(colcount_flink[m] as usize);
        assert!(pivot_col.is_some() && pivot_col.unwrap() < m);
        assert_eq!(w_end[pivot_col.unwrap()], w_begin[pivot_col.unwrap()]);
        return done(lu, pivot_row, pivot_col, nsearch, min_colnz, min_rownz, tic);
    }

    for nz in nz_start..=m {
        // Search columns with nz nonzeros.
        let mut j = colcount_flink[(m + nz) as usize];
        while j < m as LUInt {
            if min_colnz.is_none() {
                min_colnz = Some(nz);
            }
            assert_eq!(w_end[j as usize] - w_begin[j as usize], nz as LUInt);
            let cmx = colmax[j as usize];
            assert!(cmx >= 0.0);
            if cmx == 0.0 || cmx < abstol {
                continue;
            }
            let tol = f64::max(abstol, reltol * cmx);
            for pos in w_begin[j as usize]..w_end[j as usize] {
                let x = w_value[pos as usize].abs();
                if x == 0.0 || x < tol {
                    continue;
                }
                let i = w_index[pos as usize] as usize;
                assert!(/*i >= 0 &&*/ i < m);
                let nz1: IntLeast64 = nz as IntLeast64;
                let nz2: IntLeast64 = w_end[m + i] - w_begin[m + i];
                assert!(nz2 >= 1);
                let mc: IntLeast64 = (nz1 - 1) * (nz2 - 1);
                if mc < mc64 {
                    mc64 = mc;
                    pivot_row = Some(i);
                    pivot_col = Some(j as usize);
                    if search_rows != 0 && mc64 <= (nz1 - 1) * (nz1 - 1) {
                        return done(lu, pivot_row, pivot_col, nsearch, min_colnz, min_rownz, tic);
                    }
                }
            }
            // We have seen at least one eligible pivot in column j.
            assert!(mc64 < m64 * m64);
            // if (++nsearch >= maxsearch) {
            nsearch += 1;
            if nsearch >= maxsearch {
                return done(lu, pivot_row, pivot_col, nsearch, min_colnz, min_rownz, tic);
            }
            j = colcount_flink[j as usize];
        }
        assert_eq!(j, (m + nz) as LUInt);

        if search_rows == 0 {
            continue;
        }

        // Search rows with nz nonzeros.
        let mut i = rowcount_flink[(m + nz) as usize];
        while i < m as LUInt {
            if min_rownz.is_none() {
                min_rownz = Some(nz);
            }
            // rowcount_flink[i] might be changed below, so keep a copy
            let inext = rowcount_flink[i as usize];
            assert_eq!(
                w_end[((m as LUInt) + i) as usize] - w_begin[((m as LUInt) + i) as usize],
                nz as LUInt
            );
            let mut cheap = 0; // row has entries with Markowitz cost < MC?
            let mut found = 0; // eligible pivot found?
            for pos in w_begin[((m as LUInt) + i) as usize]..w_end[((m as LUInt) + i) as usize] {
                let j = w_index[pos as usize] as usize;
                assert!(/*j >= 0 &&*/ j < m);
                let nz1: IntLeast64 = nz as IntLeast64;
                let nz2: IntLeast64 = w_end[j] - w_begin[j];
                assert!(nz2 >= 1);
                let mc: IntLeast64 = (nz1 - 1) * (nz2 - 1);
                if mc >= mc64 {
                    continue;
                }
                cheap = 1;
                let cmx = colmax[j];
                assert!(cmx >= 0.0);
                if cmx == 0.0 || cmx < abstol {
                    continue;
                }
                // find position of pivot in column file
                let mut where_ = w_begin[j];
                while w_index[where_ as usize] != i {
                    assert!(where_ < w_end[j] - 1);
                    where_ += 1;
                }
                let x = w_value[where_ as usize].abs();
                if x >= abstol && x >= reltol * cmx {
                    found = 1;
                    mc64 = mc;
                    pivot_row = Some(i as usize);
                    pivot_col = Some(j);
                    if mc64 <= nz1 * (nz1 - 1) {
                        return done(lu, pivot_row, pivot_col, nsearch, min_colnz, min_rownz, tic);
                    }
                }
            }
            // If row i has cheap entries but none of them is numerically
            // acceptable, then don't search the row again until updated.
            if cheap != 0 && found == 0 {
                list_move(i as usize, m + 1, rowcount_flink, rowcount_blink, m, None);
            } else {
                assert!(mc64 < m64 * m64);
                // if (++nsearch >= maxsearch)
                nsearch += 1;
                if nsearch >= maxsearch {
                    return done(lu, pivot_row, pivot_col, nsearch, min_colnz, min_rownz, tic);
                }
            }
            i = inext;
        }
        assert_eq!(i, (m + nz) as LUInt);
    }
    done(lu, pivot_row, pivot_col, nsearch, min_colnz, min_rownz, tic)
}

fn done(
    lu: &mut LU,
    pivot_row: Option<usize>,
    pivot_col: Option<usize>,
    nsearch: usize,
    min_colnz: Option<usize>,
    min_rownz: Option<usize>,
    tic: Instant,
) -> Status {
    lu.pivot_row = pivot_row;
    lu.pivot_col = pivot_col;

    lu.nsearch_pivot += nsearch;

    if let Some(min_colnz) = min_colnz {
        lu.min_colnz = min_colnz;
    }
    if let Some(min_rownz) = min_rownz {
        lu.min_rownz = min_rownz;
    }

    lu.time_search_pivot += tic.elapsed().as_secs_f64();

    Status::OK
}
