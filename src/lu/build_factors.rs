// Copyright (C) 2016-2018 ERGO-Code
// Copyright (C) 2022-2023 Richard Lincoln

use crate::lu::file::file_empty;
use crate::lu::list::list_move;
use crate::lu::lu::*;
use crate::LUInt;
use crate::Status;

// Build rowwise and columnwise form of L and U //

// Build data structures for `L`, `R`, `U` and permutations.
//
// BLU maintains the factorization in the form
//
//     B = L * R^1 * R^2 * ... * R^{nforrest} * U,
//
// where `L[p,p]` is unit lower triangular and `U[pivotrow,pivotcol]` is upper
// triangular. After refactorization `nforrest` = 0 and `p` and `pivotrow` hold the
// same permutation. `pivotrow` and `pivotcol` are modified by updates, `p` is not.
//
// The permutations are stored as follows:
// ---------------------------------------
//
// `p[0..m-1]` is a vector.
//
//     pivotrow[0..pivotlen-1],
//     pivotcol[0..pivotlen-1]
//
// are vectors of length `m <= pivotlen < 2*m` which may contain duplicate
// indices. For each index its last occurrence is its position in the pivot
// sequence, see [`lu_garbage_perm()`].
//
//     pmap[0..m-1],
//     qmap[0..m-1]
//
// are vectors such that `i = pmap[j]` and `j = qmap[i]` when element (`i`,`j`) of
// `U` is pivot element.
//
// The matrix `L` is stored as follows:
// ------------------------------------
//
//     l_index[0..l_nz+m-1],
//     l_value[0..l_nz+m-1]
//
// hold `L` columnwise without the unit diagonal. Each column is terminated
// by index -1. Row indices are row indices of `B`.
//
//     l_begin[i]       points to the first element in column i.
//     l_begin_p[k]     points to the first element in column p[k].
//
//     l_index[l_nz+m..2*(l_nz+m)-1],
//     l_value[l_nz+m..2*(l_nz+m)-1]
//
// hold `L` rowwise without the unit diagonal. Each row is terminated
// by index -1. Column indices are such that column `i` holds the elimination
// factors from the pivot step in which row i was pivot row.
//
//     lt_begin[i]      points to the first element in row i.
//     lt_begin_p[k]    points to the first element in row p[k].
//
// The matrices `R^k` are stored as follows:
// -----------------------------------------
//
//     l_index[r_begin[k]..r_begin[k+1]-1],
//     l_value[r_begin[k]..r_begin[k+1]-1]
//
// hold the nontrivial column of `R^k` without the unit diagonal.
// Row indices are row indices of `B`. `r_begin[0]` is one past the last
// element of the `L` storage.
//
//     eta_row[k]
//
// holds the row index of the diagonal element of the nontrivial column
// of `R^k`. These are row indices of `B`.
//
// The matrix `U` is stored as follows:
// ------------------------------------
//
//     u_index[1..u_nz+m],
//     u_value[1..u_nz+m]
//
// hold `U` columnwise without the pivot elements. Each column is terminated
// by index -1. Row indices are row indices of `B`. Updates will introduce
// gaps into the data structure.
//
// `u_index[0]` stores index -1. All empty columns (ie. column in `U` with no
// off-diagonal elements) are stored in `u_index[0]`. For each empty column
// the length of the data structure decreases by 1.
//
// `u_begin[i]` points to the first element in column qmap[i].
//
//     w_index[..],
//     w_value[..]
//
// hold `U` rowwise without the pivot elements. Column indices are column
// indices of `B`. The rows are stored in a dynamic file structure with gaps
// between them and out of order.
//
// `w_begin[j]` points to the first element in row pmap[j].
// `w_end[j]` points to one past the last element in row pmap[j].
// `w_flink`, `w_blink` double linked list of rows in memory order.
//
//     col_pivot[0..m-1],
//     row_pivot[0..m-1]
//
// hold the pivot elements by column and by row index.
//
// Return:
//
// - `Reallocate`  require more memory in `L`, `U`, and/or `W`
// - `OK`
pub(crate) fn build_factors(lu: &mut LU) -> Result<(), Status> {
    let m = lu.m;
    let rank = lu.rank;
    let l_mem = lu.l_mem;
    let u_mem = lu.u_mem;
    let w_mem = lu.w_mem;
    let pad = lu.pad;
    let stretch = lu.stretch;
    // let pinv = &mut lu.pinv;
    // let qinv = &mut lu.qinv;
    // let pmap = &mut pmap!(lu); // shares memory with pinv
    // let qmap = &mut qmap!(lu); // shares memory with qinv
    let pivotcol = &mut pivotcol!(lu);
    let pivotrow = &mut pivotrow!(lu);
    let l_begin = &mut l_begin!(lu);
    // let Lbegin_p = &mut lu.Lbegin_p;
    let lt_begin = &mut lt_begin!(lu);
    let lt_begin_p = &mut lt_begin_p!(lu);
    // let Ubegin = &mut lu.Ubegin;
    let r_begin = &mut r_begin!(lu);
    // let Wbegin = &mut lu.Wbegin;
    // let Wend = &mut lu.Wend;
    // let Wflink = &mut lu.Wflink;
    // let Wblink = &mut lu.Wblink;
    // let col_pivot = &mut lu.xstore.col_pivot;
    // let row_pivot = &mut lu.xstore.row_pivot;
    let l_index = &mut lu.l_index;
    let l_value = &mut lu.l_value;
    let u_index = &mut lu.u_index;
    let u_value = &mut lu.u_value;
    let w_index = &mut lu.w_index;
    let w_value = &mut lu.w_value;
    let iwork1 = &mut iwork1!(lu);

    // lu_int i, j, ipivot, jpivot, k, lrank, nz, Lnz, Unz, need, get, put, pos;
    // double pivot, min_pivot, max_pivot;

    // So far L is stored columnwise in Lindex, Lvalue and U stored rowwise
    // in Uindex, Uvalue. The factorization has computed rank columns of L
    // and rank rows of U. If rank < m, then the columns which have not been
    // pivotal will be removed from U.
    let mut l_nz = lu.l_begin_p[rank as usize] as usize;
    l_nz -= rank; // because each column is terminated by -1
    let mut u_nz = lu.u_begin[rank as usize] as usize; // might be decreased when rank < m

    // Calculate memory and reallocate. The rowwise and columnwise storage of
    // L both need space for Lnz nonzeros + m terminators. The same for the
    // columnwise storage of U except that Uindex[0] = -1 is reserved to
    // accomodate pointers to empty columns. In the rowwise storage of U each
    // row with nz nonzeros is padded by stretch*nz + pad elements.
    let need = 2 * (l_nz + m);
    if l_mem < need {
        lu.addmem_l = need - l_mem;
        return Err(Status::Reallocate);
    }
    let need = u_nz + m + 1;
    if u_mem < need {
        lu.addmem_u = need - u_mem;
        return Err(Status::Reallocate);
    }
    let need = u_nz + (stretch * u_nz as f64) as usize + m * pad;
    if w_mem < need {
        lu.addmem_w = need - w_mem;
        return Err(Status::Reallocate);
    }

    // Build permutations //

    // Append columns/rows which have not been pivotal to the end of the
    // pivot sequence. Build pivotrow, pivotcol as inverse of pinv, qinv.
    if cfg!(feature = "debug") {
        for k in 0..m {
            pivotrow[k] = -1;
        }
        for k in 0..m {
            pivotcol[k] = -1;
        }
    }

    let mut lrank = rank;
    for i in 0..m {
        if lu.pinv[i] < 0 {
            lu.pinv[i] = lrank as LUInt;
            lrank += 1;
        }
        pivotrow[lu.pinv[i] as usize] = i as LUInt;
    }
    assert_eq!(lrank, m);
    let mut lrank = rank;
    for j in 0..m {
        if lu.qinv[j] < 0 {
            lu.qinv[j] = lrank as LUInt;
            lrank += 1;
        }
        pivotcol[lu.qinv[j] as usize] = j as LUInt;
    }
    assert_eq!(lrank, m);

    if cfg!(feature = "debug") {
        for k in 0..m {
            assert!(pivotrow[k] >= 0);
        }
        for k in 0..m {
            assert!(pivotcol[k] >= 0);
        }
    }

    // Dependent columns get unit pivot elements.
    for k in rank..m {
        lu.col_pivot[pivotcol[k] as usize] = 1.0;
    }

    // Lower triangular factor //

    // L columnwise. If rank < m, then complete with unit columns (no
    // off-diagonals, so nothing to store here).
    let mut put = lu.l_begin_p[rank as usize];
    for k in rank..m {
        l_index[put as usize] = -1;
        put += 1;
        lu.l_begin_p[k + 1] = put;
    }
    assert_eq!(lu.l_begin_p[m as usize], (l_nz + m) as LUInt);
    for i in 0..m {
        l_begin[i] = lu.l_begin_p[lu.pinv[i] as usize];
    }

    // L rowwise.
    // memset(iwork1, 0, m*sizeof(lu_int)); /* row counts */
    iwork1.fill(0); // row counts
    for get in 0..l_nz + m {
        let i = l_index[get];
        if i >= 0 {
            iwork1[i as usize] += 1;
        }
    }
    let mut put = (l_nz + m) as LUInt; // L rowwise starts here
    for k in 0..m {
        let i = pivotrow[k] as usize;
        lt_begin_p[k] = put;
        lt_begin[i] = put;
        put += iwork1[i];
        l_index[put as usize] = -1; // terminate row
        put += 1;
        iwork1[i] = lt_begin_p[k];
    }
    assert_eq!(put as usize, 2 * (l_nz + m));
    for k in 0..m {
        // fill rows
        let ipivot = pivotrow[k];
        // for (get = Lbegin_p[k]; (i = Lindex[get]) >= 0; get++)
        let mut get = lu.l_begin_p[k] as usize;
        while l_index[get] >= 0 {
            // put = iwork1[i]++;  /* put into row i */
            let put = iwork1[l_index[get] as usize] as usize; // put into row i
            iwork1[l_index[get] as usize] += 1;

            l_index[put] = ipivot;
            l_value[put] = l_value[get];
            get += 1;
        }
    }

    if cfg!(feature = "debug") {
        for i in 0..m {
            assert_eq!(l_index[iwork1[i] as usize], -1);
        }
    }
    r_begin[0] = 2 * (l_nz + m) as LUInt; // beginning of update etas

    // Upper triangular factor //

    // U rowwise.
    file_empty(
        m,
        &mut lu.w_begin,
        &mut lu.w_end,
        &mut lu.w_flink,
        &mut lu.w_blink,
        w_mem as LUInt,
    );
    // memset(iwork1, 0, m*sizeof(lu_int)); /* column counts */
    iwork1.fill(0); // column counts
    let mut put = 0;

    // Use separate loops for full rank and rank deficient factorizations. In
    // the first case no elements are removed from U, so skip the test.
    if rank == m {
        for k in 0..m {
            let jpivot = pivotcol[k] as usize;
            lu.w_begin[jpivot] = put;
            let mut nz = 0;
            for pos in lu.u_begin[k]..lu.u_begin[k + 1] {
                let j = u_index[pos as usize];
                w_index[put as usize] = j;
                // Wvalue[put++] = Uvalue[pos];
                let put0 = put;
                put += 1;
                w_value[put0 as usize] = u_value[pos as usize];
                iwork1[j as usize] += 1;
                nz += 1;
            }
            lu.w_end[jpivot] = put;
            put += (stretch * nz as f64) as LUInt + pad as LUInt;
            list_move(jpivot, 0, &mut lu.w_flink, &mut lu.w_blink, m, None);
        }
    } else {
        u_nz = 0; // actual number of nonzeros
        for k in 0..rank as usize {
            let jpivot = pivotcol[k] as usize;
            lu.w_begin[jpivot] = put;
            let mut nz = 0;
            for pos in lu.u_begin[k]..lu.u_begin[k + 1] {
                let j = u_index[pos as usize];
                if lu.qinv[j as usize] < rank as LUInt {
                    w_index[put as usize] = j;
                    // Wvalue[put++] = Uvalue[pos];
                    let put0 = put;
                    put += 1;
                    w_value[put0 as usize] = u_value[pos as usize];
                    iwork1[j as usize] += 1;
                    nz += 1;
                }
            }
            lu.w_end[jpivot] = put;
            put += (stretch * nz as f64) as LUInt + pad as LUInt;
            list_move(jpivot, 0, &mut lu.w_flink, &mut lu.w_blink, m, None);
            u_nz += nz;
        }
        for k in rank..m {
            let jpivot = pivotcol[k] as usize;
            lu.w_begin[jpivot] = put;
            lu.w_end[jpivot] = put;
            put += pad as LUInt;
            list_move(jpivot, 0, &mut lu.w_flink, &mut lu.w_blink, m, None);
        }
    }
    assert!(put <= lu.w_end[m]);
    lu.w_begin[m] = put; // beginning of free space

    // U columnwise.
    u_index[0] = -1;
    let mut put = 1;
    for k in 0..m {
        // set column pointers
        let j = pivotcol[k] as usize;
        let i = pivotrow[k] as usize;
        let nz = iwork1[j];
        if nz == 0 {
            lu.u_begin[i] = 0; // empty columns all in position 0
        } else {
            lu.u_begin[i] = put;
            put += nz;
            u_index[put as usize] = -1; // terminate column
            put += 1;
        }
        iwork1[j] = lu.u_begin[i];
    }
    lu.u_begin[m] = put;
    for k in 0..m {
        // fill columns
        let jpivot = pivotcol[k] as usize;
        let i = pivotrow[k];
        for pos in lu.w_begin[jpivot]..lu.w_end[jpivot] {
            let j = w_index[pos as usize] as usize;
            let put = iwork1[j] as usize;
            iwork1[j] += 1;
            assert!(put >= 1);
            u_index[put] = i;
            u_value[put] = w_value[pos as usize];
        }
    }

    if cfg!(feature = "debug") {
        for j in 0..m {
            assert_eq!(u_index[iwork1[j] as usize], -1);
        }
    }

    // Build pivot sequence //

    // Build row-column mappings, overwriting pinv, qinv.
    for k in 0..m {
        let i = pivotrow[k];
        let j = pivotcol[k];
        pmap!(lu)[j as usize] = i;
        qmap!(lu)[i as usize] = j;
    }

    // Build pivots by row index.
    let mut max_pivot = 0.0;
    let mut min_pivot = f64::INFINITY;
    for i in 0..m {
        lu.row_pivot[i] = lu.col_pivot[qmap![lu][i] as usize];
        let pivot = lu.row_pivot[i].abs();
        max_pivot = f64::max(pivot, max_pivot);
        min_pivot = f64::min(pivot, min_pivot);
    }

    // memcpy(lu.p, pivotrow, m*sizeof(lu_int));
    p![lu][..m].copy_from_slice(&pivotrow[..m]);

    lu.min_pivot = min_pivot;
    lu.max_pivot = max_pivot;
    lu.pivotlen = m;
    lu.l_nz = l_nz;
    lu.u_nz = u_nz;
    lu.r_nz = 0;

    Ok(())
}
