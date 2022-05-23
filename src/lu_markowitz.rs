// Copyright (C) 2016-2019  ERGO-Code

use crate::basiclu::*;
use crate::lu_internal::lu;
use crate::lu_list::lu_list_move;
use std::time::Instant;

/// Search for pivot element with small Markowitz cost. An eligible pivot
/// must be nonzero and satisfy
///
///  (1) abs(piv) >= abstol,
///  (2) abs(piv) >= reltol * max[pivot column].
///
///  From all eligible pivots search for one that minimizes
///
///   mc := (nnz[pivot row] - 1) * (nnz[pivot column] - 1).
///
/// The search is terminated when maxsearch rows or columns with eligible pivots
/// have been searched (if not before). The row and column of the cheapest one
/// found is stored in pivot_row and pivot_col.
///
/// When the active submatrix contains columns with column count = 0, then such a
/// column is chosen immediately and pivot_row = -1 is returned. Otherwise, when
/// the Markowitz search does not find a pivot that is nonzero and >= abstol,
/// then pivot_col = pivot_row = -1 is returned. (The latter cannot happen in the
/// current version of the code because lu_pivot() erases columns of the active
/// submatrix whose maximum absolute value drops below abstol.)
///
/// The Markowitz search is implemented as described in [1].
///
/// [1] U. Suhl, L. Suhl, "Computing Sparse LU Factorizations for Large-Scale
///     Linear Programming Bases", ORSA Journal on Computing (1990)
pub(crate) fn lu_markowitz(this: &mut lu) -> lu_int {
    let m = this.m;
    let Wbegin = this.Wbegin;
    let Wend = this.Wend;
    let Windex = this.Windex.unwrap();
    let Wvalue = this.Wvalue.unwrap();
    let colcount_flink = this.colcount_flink;
    let rowcount_flink = &mut this.rowcount_flink;
    let rowcount_blink = &mut this.rowcount_blink;
    let colmax = this.col_pivot;
    let abstol = this.abstol;
    let reltol = this.reltol;
    let maxsearch = this.maxsearch;
    let search_rows = this.search_rows;
    let nz_start = if search_rows != 0 {
        lu_int::min(this.min_colnz, this.min_rownz)
    } else {
        this.min_colnz
    };

    // lu_int i, j, pos, where_, inext, nz, pivot_row, pivot_col;
    // lu_int nsearch, cheap, found, min_colnz, min_rownz;
    // double cmx, x, tol, tic[2];

    /* integers for Markowitz cost must be 64 bit to prevent overflow */
    let M = m as int_least64_t;
    let nz1: int_least64_t;
    let nz2: int_least64_t;
    let mc: int_least64_t;
    let MC: int_least64_t;

    // lu_tic(tic);
    let tic = Instant::now();
    let mut pivot_row = -1; // row of best pivot so far
    let mut pivot_col = -1; // col of best pivot so far
    let mut MC = M * M; // Markowitz cost of best pivot so far
    let mut nsearch = 0; // count rows/columns searched
    let mut min_colnz = -1; // minimum col count in active submatrix
    let mut min_rownz = -1; // minimum row count in active submatrix
    assert!(nz_start >= 1);

    let done = || {
        this.pivot_row = pivot_row;
        this.pivot_col = pivot_col;
        this.nsearch_pivot += nsearch;
        if min_colnz >= 0 {
            this.min_colnz = min_colnz;
        }
        if min_rownz >= 0 {
            this.min_rownz = min_rownz;
        }
        this.time_search_pivot += lu_toc(tic);
        return BASICLU_OK;
    };

    // If the active submatrix contains empty columns, choose one and return
    // with pivot_row = -1.
    if colcount_flink[m] != m {
        pivot_col = colcount_flink[m];
        assert(pivot_col >= 0 && pivot_col < m);
        assert(Wend[pivot_col] == Wbegin[pivot_col]);
        return done();
    }

    for nz in nz_start..=m {
        // Search columns with nz nonzeros.
        let mut j = colcount_flink[m + nz];
        while j < m {
            if min_colnz == -1 {
                min_colnz = nz;
            }
            assert_eq!(Wend[j] - Wbegin[j], nz);
            cmx = colmax[j];
            assert(cmx >= 0);
            if !cmx || cmx < abstol {
                continue;
            }
            tol = fmax(abstol, reltol * cmx);
            for pos in Wbegin[j]..Wend[j] {
                x = fabs(Wvalue[pos]);
                if !x || x < tol {
                    continue;
                }
                i = Windex[pos];
                assert!(i >= 0 && i < m);
                nz1 = nz;
                nz2 = Wend[m + i] - Wbegin[m + i];
                assert!(nz2 >= 1);
                mc = (nz1 - 1) * (nz2 - 1);
                if mc < MC {
                    MC = mc;
                    pivot_row = i;
                    pivot_col = j;
                    if search_rows != 0 && MC <= (nz1 - 1) * (nz1 - 1) {
                        return done();
                    }
                }
            }
            // We have seen at least one eligible pivot in column j.
            assert!(MC < M * M);
            // if (++nsearch >= maxsearch) {
            nsearch += 1;
            if nsearch >= maxsearch {
                return done();
            }
            j = colcount_flink[j];
        }
        assert_eq!(j, m + nz);

        if search_rows == 0 {
            continue;
        }

        // Search rows with nz nonzeros.
        let mut i = rowcount_flink[m + nz];
        while i < m {
            if min_rownz == -1 {
                min_rownz = nz;
            }
            // rowcount_flink[i] might be changed below, so keep a copy
            inext = rowcount_flink[i];
            assert_eq!(Wend[m + i] - Wbegin[m + i], nz);
            cheap = 0; // row has entries with Markowitz cost < MC?
            found = 0; // eligible pivot found?
            for pos in Wbegin[m + i]..Wend[m + i] {
                j = Windex[pos];
                assert!(j >= 0 && j < m);
                nz1 = nz;
                nz2 = Wend[j] - Wbegin[j];
                assert!(nz2 >= 1);
                mc = (nz1 - 1) * (nz2 - 1);
                if mc >= MC {
                    continue;
                }
                cheap = 1;
                cmx = colmax[j];
                assert!(cmx >= 0);
                if !cmx || cmx < abstol {
                    continue;
                }
                // find position of pivot in column file
                let mut where_ = Wbegin[j];
                while Windex[where_] != i {
                    assert!(where_ < Wend[j] - 1);
                    where_ += 1;
                }
                x = fabs(Wvalue[where_]);
                if x >= abstol && x >= reltol * cmx {
                    found = 1;
                    MC = mc;
                    pivot_row = i;
                    pivot_col = j;
                    if MC <= nz1 * (nz1 - 1) {
                        return done();
                    }
                }
            }
            // If row i has cheap entries but none of them is numerically
            // acceptable, then don't search the row again until updated.
            if cheap && !found {
                lu_list_move(i, m + 1, rowcount_flink, rowcount_blink, m, None);
            } else {
                assert!(MC < M * M);
                // if (++nsearch >= maxsearch)
                nsearch += 1;
                if nsearch >= maxsearch {
                    return done();
                }
            }
            i = inext;
        }
        assert_eq!(i, m + nz);
    }
    done()
}
