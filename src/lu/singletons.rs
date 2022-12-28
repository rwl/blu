// Copyright (C) 2016-2018 ERGO-Code
// Copyright (C) 2022-2023 Richard Lincoln

use crate::lu::lu::*;
use crate::LUInt;
use crate::Status;
use std::time::Instant;

// Initialize the data structures which store the LU factors during
// factorization and eliminate pivots with Markowitz cost zero.
//
// During factorization the inverse pivot sequence is recorded in `pinv`, `qinv`:
//
// - `pinv[i]` >=  0   if row i was pivot row in stage `pinv[i]`
// - `pinv[i]` == -1   if row i has not been pivot row yet
// - `qinv[j]` >=  0   if col j was pivot col in stage `qinv[j]`
// - `qinv[j]` == -1   if col j has not been pivot col yet
//
// The lower triangular factor is composed columnwise in `l_index`, `l_value`.
// The upper triangular factor is composed rowwise in `u_index`, `u_value`.
// After rank steps of factorization:
//
// - `l_begin_p[rank]` is the next unused position in `l_index`, `l_value`.
//
// - `l_index[l_begin_p[k]..]`, `l_value[l_begin_p[k]..]` for `0 <= k < rank`
//   stores the column of `L` computed in stage `k` without the unit diagonal.
//   The column is terminated by a negative index.
//
// - `u_begin[rank]` is the next unused position in `u_index`, `u_value`.
//
// - `u_index[u_begin[k]..u_begin[k+1]-1]`, `u_value[u_begin[k]..u_begin[k+1]-1]`
//   stores the row of `U` computed in stage `k` without the pivot element.
//
// `singletons()` does `rank >= 0` steps of factorization until no singletons are
// left. We can either eliminate singleton columns before singleton rows or vice
// versa. When `nzbias >= 0`, then eliminate singleton columns first to keep `L`
// sparse. Otherwise eliminate singleton rows first. The resulting permutations
// `P`, `Q` (stored in inverse form) make `PBQ'` of the form
//
//             \uuuuuuuuuuuuuuuuuuuuuuu
//              \u                    u
//               \u                   u
//                \u                  u
//                 \u                 u
//     PBQ' =       \uuuuuuu__________u               singleton columns before
//                   \     |          |               singleton rows
//                   l\    |          |
//                   ll\   |          |
//                   l l\  |   BUMP   |
//                   l  l\ |          |
//                   lllll\|__________|
//
//             \
//             l\
//             ll\
//             l l\
//             l  l\
//             l   l\       __________
//     PBQ' =  l    l\uuuuu|          |               singleton rows before
//             l    l \u  u|          |               singleton columns
//             l    l  \u u|          |
//             l    l   \uu|   BUMP   |
//             l    l    \u|          |
//             llllll     \|__________|
//
// Off-diagonals from singleton columns (`u`) are stored in `U`, off-diagonals from
// singleton rows (`l`) are stored in `L` and divided by the diagonal. Diagonals (\)
// are stored in `col_pivot`.
//
// Do not pivot on elements which are zero or less than `abstol` in magnitude.
// When such pivots occur, the row/column remains in the active submatrix and
// the bump factorization will detect the singularity.
//
// Return:
//
// - `Reallocate`              less than `nnz(B)` memory in `L`, `U` or `W`
// - `ErrorInvalidArgument`  matrix `B` is invalid (negative number of
//                                     entries in column, index out of range,
//                                     duplicates)
// - `OK`
pub(crate) fn singletons(
    lu: &mut LU,
    b_begin: &[LUInt],
    b_end: &[LUInt],
    b_i: &[LUInt],
    b_x: &[f64],
) -> Status {
    let m = lu.m;
    let l_mem = lu.l_mem;
    let u_mem = lu.u_mem;
    let w_mem = lu.w_mem;
    let abstol = lu.abstol;
    let nzbias = lu.nzbias;
    let pinv = &mut lu.pinv;
    let qinv = &mut lu.qinv;
    let l_begin_p = &mut lu.l_begin_p;
    let u_begin = &mut lu.u_begin;
    let col_pivot = &mut lu.col_pivot;
    let l_index = &mut lu.l_index;
    let l_value = &mut lu.l_value;
    let u_index = &mut lu.u_index;
    let u_value = &mut lu.u_value;
    // let iwork1 = &mut lu.iwork1;
    // let iwork2 = iwork1 + m;
    let (iwork1, iwork2) = iwork1!(lu).split_at_mut(m as usize);

    let b_tp = &mut lu.w_begin; // build B rowwise in W
    let b_ti = &mut lu.w_index;
    let b_tx = &mut lu.w_value;

    // lu_int i, j, pos, put, rank, Bnz, ok;
    // double tic[2];
    // lu_tic(tic);
    let tic = Instant::now();

    // Check matrix and build transpose //

    // Check pointers and count nnz(B).
    let mut b_nz = 0;
    let mut ok = 1;
    let mut j = 0;
    while j < m && ok != 0 {
        if b_end[j as usize] < b_begin[j as usize] {
            ok = 0;
        } else {
            b_nz += b_end[j as usize] - b_begin[j as usize];
        }
        j += 1;
    }
    if ok == 0 {
        return Status::ErrorInvalidArgument;
    }

    // Check if sufficient memory in L, U, W.
    let mut ok = 1;
    if l_mem < b_nz {
        lu.addmem_l = b_nz - l_mem;
        ok = 0;
    }
    if u_mem < b_nz {
        lu.addmem_u = b_nz - u_mem;
        ok = 0;
    }
    if w_mem < b_nz {
        lu.addmem_w = b_nz - w_mem;
        ok = 0;
    }
    if ok == 0 {
        return Status::Reallocate;
    }

    // Count nz per row, check indices.
    // memset(iwork1, 0, m); // row counts
    iwork1.fill(0); // row counts
    let mut ok = 1;
    let mut j = 0;
    while j < m && ok != 0 {
        let mut pos = b_begin[j as usize];
        while pos < b_end[j as usize] && ok != 0 {
            let i = b_i[pos as usize];
            if i < 0 || i >= m {
                ok = 0;
            } else {
                iwork1[i as usize] += 1;
            }
            pos += 1;
        }
        j += 1;
    }
    if ok == 0 {
        return Status::ErrorInvalidArgument;
    }

    // Pack matrix rowwise, check for duplicates.
    let mut put = 0;
    for i in 0..m as usize {
        // set row pointers
        b_tp[i] = put;
        put += iwork1[i];
        iwork1[i] = b_tp[i];
    }
    b_tp[m as usize] = put;
    assert_eq!(put, b_nz);
    let mut ok = 1;
    for j in 0..m {
        // fill rows
        for pos in b_begin[j as usize]..b_end[j as usize] {
            let i = b_i[pos as usize] as usize;
            put = iwork1[i];
            iwork1[i] += 1;
            b_ti[put as usize] = j;
            b_tx[put as usize] = b_x[pos as usize];
            if put > b_tp[i] && b_ti[(put - 1) as usize] == j {
                ok = 0;
            }
        }
    }
    if ok == 0 {
        return Status::ErrorInvalidArgument;
    }

    // Pivot singletons //

    // No pivot rows or pivot columns so far.
    for i in 0..m {
        pinv[i as usize] = -1;
    }
    for j in 0..m {
        qinv[j as usize] = -1;
    }

    let rank = if nzbias >= 0 {
        // put more in U
        l_begin_p[0] = 0;
        u_begin[0] = 0;
        let rank = 0;

        let rank = singleton_cols(
            m, b_begin, b_end, b_i, b_x, b_tp, b_ti, b_tx, u_begin, u_index, u_value, l_begin_p,
            l_index, l_value, col_pivot, pinv, qinv, iwork1, iwork2, rank, abstol,
        );

        let rank = singleton_rows(
            m, b_begin, b_end, b_i, b_x, b_tp, b_ti, b_tx, u_begin, u_index, u_value, l_begin_p,
            l_index, l_value, col_pivot, pinv, qinv, iwork1, iwork2, rank, abstol,
        );
        rank
    } else {
        // put more in L
        l_begin_p[0] = 0;
        u_begin[0] = 0;
        let rank = 0;

        let rank = singleton_rows(
            m, b_begin, b_end, b_i, b_x, b_tp, b_ti, b_tx, u_begin, u_index, u_value, l_begin_p,
            l_index, l_value, col_pivot, pinv, qinv, iwork1, iwork2, rank, abstol,
        );

        let rank = singleton_cols(
            m, b_begin, b_end, b_i, b_x, b_tp, b_ti, b_tx, u_begin, u_index, u_value, l_begin_p,
            l_index, l_value, col_pivot, pinv, qinv, iwork1, iwork2, rank, abstol,
        );
        rank
    };

    // pinv, qinv were used as nonzero counters. Reset to -1 if not pivoted.
    for i in 0..m as usize {
        if pinv[i] < 0 {
            pinv[i] = -1;
        }
    }
    for j in 0..m as usize {
        if qinv[j] < 0 {
            qinv[j] = -1;
        }
    }

    lu.matrix_nz = b_nz;
    lu.rank = rank;
    lu.time_singletons = tic.elapsed().as_secs_f64();
    Status::OK
}

// The method successively removes singleton cols from an active submatrix.
// The active submatrix is composed of columns `j` for which `qinv[j] < 0` and
// rows `i` for which `pinv[i] < 0`. When removing a singleton column and its
// associated row generates new singleton columns, these are appended to a
// queue. The method stops when the active submatrix has no more singleton
// columns.
//
// For each active column `j` `iset[j]` is the XOR of row indices in the column
// in the active submatrix. For a singleton column, this is its single row
// index. The technique is due to J. Gilbert and described in [1], ex 3.7.
//
// For each eliminated column its associated row is stored in `U` without the
// pivot element. The pivot elements are stored in `col_pivot`. For each
// eliminated pivot an empty column is appended to `L`.
//
// Pivot elements which are zero or less than `abstol`, and empty columns in
// the active submatrix are not eliminated. In these cases the matrix is
// numerically or structurally singular and the bump factorization handles
// it. (We want singularities at the end of the pivot sequence.)
//
// [1] T. Davis, "Direct methods for sparse linear systems"
pub(crate) fn singleton_cols(
    m: LUInt,
    b_begin: &[LUInt], // B columnwise
    b_end: &[LUInt],
    b_i: &[LUInt],
    _b_x: &[f64],
    b_tp: &[LUInt], /* B rowwise */
    b_ti: &[LUInt],
    b_tx: &[f64],
    u_p: &mut [LUInt],
    u_i: &mut [LUInt],
    u_x: &mut [f64],
    l_p: &mut [LUInt],
    l_i: &mut [LUInt],
    _l_x: &mut [f64],
    col_pivot: &mut [f64],
    pinv: &mut [LUInt],
    qinv: &mut [LUInt],
    iset: &mut [LUInt],  // size m workspace
    queue: &mut [LUInt], // size m workspace
    mut rank: LUInt,
    abstol: f64,
) -> LUInt {
    // lu_int i, j, j2, nz, pos, put, end, front, tail;
    // double piv;
    let mut rk = rank;

    // Build index sets and initialize queue.
    let mut tail = 0;
    for j in 0..m {
        if qinv[j as usize] < 0 {
            let nz = b_end[j as usize] - b_begin[j as usize];
            let mut i = 0;
            for pos in b_begin[j as usize]..b_end[j as usize] {
                i ^= b_i[pos as usize]; // put row into set j
            }
            iset[j as usize] = i;
            qinv[j as usize] = -nz - 1; // use as nonzero counter
            if nz == 1 {
                queue[tail] = j;
                tail += 1;
            }
        }
    }

    // Eliminate singleton columns.
    let mut put = u_p[rank as usize];
    for front in 0..tail {
        let j = queue[front];
        assert!(qinv[j as usize] == -2 || qinv[j as usize] == -1);
        if qinv[j as usize] == -1 {
            continue; // empty column in active submatrix
        }
        let i = iset[j as usize];
        assert!(i >= 0 && i < m);
        assert!(pinv[i as usize] < 0);
        let end = b_tp[(i + 1) as usize];

        let mut pos = b_tp[i as usize];
        while b_ti[pos as usize] != j {
            // find pivot
            assert!(pos < end - 1);
            pos += 1;
        }

        let piv = b_tx[pos as usize];
        if piv == 0.0 || piv.abs() < abstol {
            continue; // skip singularity
        }

        // Eliminate pivot.
        qinv[j as usize] = rank;
        pinv[i as usize] = rank;
        for pos in b_tp[i as usize]..end {
            let j2 = b_ti[pos as usize];
            if qinv[j2 as usize] < 0 {
                // test is mandatory because the initial active submatrix may
                // not be the entire matrix (rows eliminated before)

                u_i[put as usize] = j2;
                u_x[put as usize] = b_tx[pos as usize];
                put += 1;
                iset[j2 as usize] ^= i; // remove i from set j2

                // if (++qinv[j2] == -2) {
                qinv[j2 as usize] += 1;
                if qinv[j2 as usize] == -2 {
                    queue[tail as usize] = j2; // new singleton
                    tail += 1;
                }
            }
        }
        u_p[(rank + 1) as usize] = put;
        col_pivot[j as usize] = piv;
        rank += 1;
    }

    // Put empty columns into L.
    let mut pos = l_p[rk as usize];
    while rk < rank {
        l_i[pos as usize] = -1;
        pos += 1;
        l_p[(rk + 1) as usize] = pos;
        rk += 1;
    }
    rank
}

// Analogous [`singleton_cols`] except that for each singleton row the
// associated column is stored in `L` and divided by the pivot element. The
// pivot element is stored in `col_pivot`.
fn singleton_rows(
    m: LUInt,
    b_begin: &[LUInt], // B columnwise
    b_end: &[LUInt],
    b_i: &[LUInt],
    b_x: &[f64],
    b_tp: &[LUInt], // B rowwise
    b_ti: &[LUInt],
    _b_tx: &[f64],
    u_p: &mut [LUInt],
    _u_i: &mut [LUInt],
    _u_x: &mut [f64],
    l_p: &mut [LUInt],
    l_i: &mut [LUInt],
    l_x: &mut [f64],
    col_pivot: &mut [f64],
    pinv: &mut [LUInt],
    qinv: &mut [LUInt],
    iset: &mut [LUInt],  // size m workspace
    queue: &mut [LUInt], // size m workspace
    mut rank: LUInt,
    abstol: f64,
) -> LUInt {
    // lu_int i, j, i2, nz, pos, put, end, front, tail, rk = rank;
    // double piv;
    let mut rk = rank;

    // Build index sets and initialize queue.
    let mut tail = 0;
    for i in 0..m {
        if pinv[i as usize] < 0 {
            let nz = b_tp[(i + 1) as usize] - b_tp[i as usize];
            let mut j = 0;
            for pos in b_tp[i as usize]..b_tp[(i + 1) as usize] {
                j ^= b_ti[pos as usize]; // put column into set i
            }
            iset[i as usize] = j;
            pinv[i as usize] = -nz - 1; /* use as nonzero counter */
            if nz == 1 {
                queue[tail as usize] = i;
                tail += 1;
            }
        }
    }

    // Eliminate singleton rows.
    let mut put = l_p[rank as usize];
    for front in 0..tail {
        let i = queue[front];
        assert!(pinv[i as usize] == -2 || pinv[i as usize] == -1);
        if pinv[i as usize] == -1 {
            continue; // empty column in active submatrix
        }
        let j = iset[i as usize];
        assert!(j >= 0 && j < m);
        assert!(qinv[j as usize] < 0);
        let end = b_end[j as usize];

        let mut pos = b_begin[j as usize];
        while b_i[pos as usize] != i {
            // find pivot
            assert!(pos < end - 1);
            pos += 1;
        }
        let piv = b_x[pos as usize];
        if piv == 0.0 || piv.abs() < abstol {
            continue; // skip singularity
        }

        // Eliminate pivot.
        qinv[j as usize] = rank;
        pinv[i as usize] = rank;
        for pos in b_begin[j as usize]..end {
            let i2 = b_i[pos as usize];
            if pinv[i2 as usize] < 0 {
                // test is mandatory because the initial active submatrix may
                // not be the entire matrix (columns eliminated before)
                l_i[put as usize] = i2;
                l_x[put as usize] = b_x[pos as usize] / piv;
                put += 1;
                iset[i2 as usize] ^= j; // remove j from set i2

                //if (++pinv[i2] == -2)
                pinv[i2 as usize] += 1;
                if pinv[i2 as usize] == -2 {
                    queue[tail] = i2; // new singleton
                    tail += 1;
                }
            }
        }
        l_i[put as usize] = -1; // terminate column
        put += 1;
        l_p[(rank + 1) as usize] = put;
        col_pivot[j as usize] = piv;
        rank += 1;
    }

    // Put empty rows into U.
    let pos = u_p[rk as usize];
    while rk < rank {
        u_p[(rk + 1) as usize] = pos;
        rk += 1;
    }

    rank
}
