// Copyright (C) 2016-2018  ERGO-Code
//
// Build rowwise and columnwise form of L and U

// BASICLU maintains the factorization in the form
//
//   B = L * R^1 * R^2 * ... * R^{nforrest} * U,
//
// where L[p,p] is unit lower triangular and U[pivotrow,pivotcol] is upper
// triangular. After refactorization nforrest = 0 and p and pivotrow hold the
// same permutation. pivotrow and pivotcol are modified by updates, p is not.
//
// The permutations are stored as follows:
// ---------------------------------------
//
//   p[0..m-1] is a vector.
//
//   pivotrow[0..pivotlen-1],
//   pivotcol[0..pivotlen-1]
//
//     are vectors of length m <= pivotlen < 2*m which may contain duplicate
//     indices. For each index its last occurance is its position in the pivot
//     sequence, see lu_garbage_perm().
//
//   pmap[0..m-1],
//   qmap[0..m-1]
//
//     are vectors such that i = pmap[j] and j = qmap[i] when element (i,j) of
//     U is pivot element.
//
// The matrix L is stored as follows:
// ----------------------------------
//
//   Lindex[0..Lnz+m-1],
//   Lvalue[0..Lnz+m-1]
//
//     hold L columnwise without the unit diagonal. Each column is terminated
//     by index -1. Row indices are row indices of B.
//
//   Lbegin[i]       points to the first element in column i.
//   Lbegin_p[k]     points to the first element in column p[k].
//
//   Lindex[Lnz+m..2*(Lnz+m)-1],
//   Lvalue[Lnz+m..2*(Lnz+m)-1]
//
//     hold L rowwise without the unit diagonal. Each row is terminated
//     by index -1. Column indices are such that column i holds the elimination
//     factors from the pivot step in which row i was pivot row.
//
//   Ltbegin[i]      points to the first element in row i.
//   Ltbegin_p[k]    points to the first element in row p[k].
//
// The matrices R^k are stored as follows:
// ---------------------------------------
//
//   Lindex[Rbegin[k]..Rbegin[k+1]-1],
//   Lvalue[Rbegin[k]..Rbegin[k+1]-1]
//
//     hold the nontrivial column of R^k without the unit diagonal.
//     Row indices are row indices of B. Rbegin[0] is one past the last
//     element of the L storage.
//
//   eta_row[k]
//
//     holds the row index of the diagonal element of the nontrivial column
//     of R^k. These are row indices of B.
//
// The matrix U is stored as follows:
// ----------------------------------
//
//   Uindex[1..Unz+m],
//   Uvalue[1..Unz+m]
//
//     hold U columnwise without the pivot elements. Each column is terminated
//     by index -1. Row indices are row indices of B. Updates will introduce
//     gaps into the data structure.
//
//     Uindex[0] stores index -1. All empty columns (ie. column in U with no
//     off-diagonal elements) are stored in Uindex[0]. For each empty column
//     the length of the data structure decreases by 1.
//
//   Ubegin[i]       points to the first element in column qmap[i].
//
//   Windex[..],
//   Wvalue[..]
//
//     hold U rowwise without the pivot elements. Column indices are column
//     indices of B. The rows are stored in a dynamic file structure with gaps
//     between them and out of order.
//
//   Wbegin[j]       points to the first element in row pmap[j].
//   Wend[j]         points to one past the last element in row pmap[j].
//   Wflink, Wblink  double linked list of rows in memory order.
//
//   col_pivot[0..m-1],
//   row_pivot[0..m-1]
//
//     hold the pivot elements by column and by row index.

use crate::basiclu::{lu_int, BASICLU_OK, BASICLU_REALLOCATE};
use crate::lu_file::lu_file_empty;
use crate::lu_internal::lu;
use crate::lu_list::lu_list_move;

/// lu_build_factors() - build data structures for L, R, U and permutations
///
/// Return:
///
///  BASICLU_REALLOCATE  require more memory in L, U, and/or W
///  BASICLU_OK
pub(crate) fn lu_build_factors(
    this: &mut lu,
    Li: &mut [lu_int],
    Lx: &mut [f64],
    Ui: &mut [lu_int],
    Ux: &mut [f64],
    Wi: &mut [lu_int],
    Wx: &mut [f64],
) -> lu_int {
    let m = this.m;
    let rank = this.rank;
    let Lmem = this.Lmem;
    let Umem = this.Umem;
    let Wmem = this.Wmem;
    let pad = this.pad;
    let stretch = this.stretch as lu_int;
    let pinv = &mut this.factor.pinv;
    let qinv = &mut this.factor.qinv;
    let pmap = &mut this.solve.pmap; // shares memory with pinv
    let qmap = &mut this.solve.qmap; // shares memory with qinv
    let pivotcol = &mut this.solve.pivotcol;
    let pivotrow = &mut this.solve.pivotrow;
    let Lbegin = &mut this.solve.Lbegin;
    let Lbegin_p = &mut this.solve.Lbegin_p;
    let Ltbegin = &mut this.solve.Ltbegin;
    let Ltbegin_p = &mut this.solve.Ltbegin_p;
    let Ubegin = &mut this.solve.Ubegin;
    let Rbegin = &mut this.solve.Rbegin;
    let Wbegin = &mut this.factor.Wbegin;
    let Wend = &mut this.factor.Wend;
    let Wflink = &mut this.factor.Wflink;
    let Wblink = &mut this.factor.Wblink;
    let col_pivot = &mut this.xstore.col_pivot;
    let row_pivot = &mut this.xstore.row_pivot;
    let Lindex = Li;
    let Lvalue = Lx;
    let Uindex = Ui;
    let Uvalue = Ux;
    let Windex = Wi;
    let Wvalue = Wx;
    let iwork1 = &mut this.solve.iwork1;

    // lu_int i, j, ipivot, jpivot, k, lrank, nz, Lnz, Unz, need, get, put, pos;
    // double pivot, min_pivot, max_pivot;
    let mut status = BASICLU_OK;

    // So far L is stored columnwise in Lindex, Lvalue and U stored rowwise
    // in Uindex, Uvalue. The factorization has computed rank columns of L
    // and rank rows of U. If rank < m, then the columns which have not been
    // pivotal will be removed from U.
    let mut Lnz = Lbegin_p[rank as usize];
    Lnz -= rank; // because each column is terminated by -1
    let mut Unz = Ubegin[rank as usize]; // might be decreased when rank < m

    // Calculate memory and reallocate. The rowwise and columnwise storage of
    // L both need space for Lnz nonzeros + m terminators. The same for the
    // columnwise storage of U except that Uindex[0] = -1 is reserved to
    // accomodate pointers to empty columns. In the rowwise storage of U each
    // row with nz nonzeros is padded by stretch*nz + pad elements.
    let need = 2 * (Lnz + m);
    if Lmem < need {
        this.addmemL = need - Lmem;
        status = BASICLU_REALLOCATE;
    }
    let need = Unz + m + 1;
    if Umem < need {
        this.addmemU = need - Umem;
        status = BASICLU_REALLOCATE;
    }
    let need = Unz + stretch * Unz + m * pad;
    if Wmem < need {
        this.addmemW = need - Wmem;
        status = BASICLU_REALLOCATE;
    }
    if status != BASICLU_OK {
        return status;
    }

    // Build permutations //

    // Append columns/rows which have not been pivotal to the end of the
    // pivot sequence. Build pivotrow, pivotcol as inverse of pinv, qinv.
    if cfg!(feature = "debug") {
        for k in 0..m as usize {
            pivotrow[k] = -1;
        }
        for k in 0..m as usize {
            pivotcol[k] = -1;
        }
    }

    let mut lrank = rank;
    for i in 0..m as usize {
        if pinv[i] < 0 {
            pinv[i] = lrank;
            lrank += 1;
        }
        pivotrow[pinv[i] as usize] = i as lu_int;
    }
    assert_eq!(lrank, m);
    let mut lrank = rank;
    for j in 0..m as usize {
        if qinv[j] < 0 {
            qinv[j] = lrank;
            lrank += 1;
        }
        pivotcol[qinv[j] as usize] = j as lu_int;
    }
    assert_eq!(lrank, m);

    if cfg!(feature = "debug") {
        for k in 0..m as usize {
            assert!(pivotrow[k] >= 0);
        }
        for k in 0..m as usize {
            assert!(pivotcol[k] >= 0);
        }
    }

    // Dependent columns get unit pivot elements.
    for k in rank..m {
        col_pivot[pivotcol[k as usize] as usize] = 1.0;
    }

    // Lower triangular factor //

    // L columnwise. If rank < m, then complete with unit columns (no
    // off-diagonals, so nothing to store here).
    let mut put = Lbegin_p[rank as usize];
    for k in rank..m {
        Lindex[put as usize] = -1;
        put += 1;
        Lbegin_p[k as usize + 1] = put;
    }
    assert_eq!(Lbegin_p[m as usize], Lnz + m);
    for i in 0..m as usize {
        Lbegin[i] = Lbegin_p[pinv[i] as usize];
    }

    // L rowwise.
    // memset(iwork1, 0, m*sizeof(lu_int)); /* row counts */
    iwork1.fill(0); // row counts
    for get in 0..Lnz + m {
        let i = Lindex[get as usize];
        if i >= 0 {
            iwork1[i as usize] += 1;
        }
    }
    put = Lnz + m; // L rowwise starts here
    for k in 0..m as usize {
        let i = pivotrow[k] as usize;
        Ltbegin_p[k] = put;
        Ltbegin[i] = put;
        put += iwork1[i];
        Lindex[put as usize] = -1; // terminate row
        put += 1;
        iwork1[i] = Ltbegin_p[k];
    }
    assert_eq!(put, 2 * (Lnz + m));
    for k in 0..m as usize {
        // fill rows
        let ipivot = pivotrow[k];
        // for (get = Lbegin_p[k]; (i = Lindex[get]) >= 0; get++)
        let mut get = Lbegin_p[k] as usize;
        while Lindex[get] >= 0 {
            // put = iwork1[i]++;  /* put into row i */
            put = iwork1[Lindex[get] as usize]; // put into row i
            iwork1[Lindex[get] as usize] += 1;
            Lindex[put as usize] = ipivot;
            Lvalue[put as usize] = Lvalue[get];
            get += 1;
        }
    }

    if cfg!(feature = "debug") {
        for i in 0..m as usize {
            assert_eq!(Lindex[iwork1[i] as usize], -1);
        }
    }
    Rbegin[0] = 2 * (Lnz + m); // beginning of update etas

    // Upper triangular factor //

    // U rowwise.
    lu_file_empty(m, Wbegin, Wend, Wflink, Wblink, Wmem);
    // memset(iwork1, 0, m*sizeof(lu_int)); /* column counts */
    iwork1.fill(0); // column counts
    put = 0;

    // Use separate loops for full rank and rank deficient factorizations. In
    // the first case no elements are removed from U, so skip the test.
    if rank == m {
        for k in 0..m as usize {
            let jpivot = pivotcol[k];
            Wbegin[jpivot as usize] = put;
            let mut nz = 0;
            for pos in Ubegin[k]..Ubegin[k + 1] {
                let j = Uindex[pos as usize];
                Windex[put as usize] = j;
                // Wvalue[put++] = Uvalue[pos];
                let put0 = put;
                put += 1;
                Wvalue[put0 as usize] = Uvalue[pos as usize];
                iwork1[j as usize] += 1;
                nz += 1;
            }
            Wend[jpivot as usize] = put;
            put += stretch * nz + pad;
            lu_list_move(jpivot, 0, Wflink, Wblink, m, None);
        }
    } else {
        Unz = 0; // actual number of nonzeros
        for k in 0..rank as usize {
            let jpivot = pivotcol[k];
            Wbegin[jpivot as usize] = put;
            let mut nz = 0;
            for pos in Ubegin[k]..Ubegin[k + 1] {
                let j = Uindex[pos as usize];
                if qinv[j as usize] < rank {
                    Windex[put as usize] = j;
                    // Wvalue[put++] = Uvalue[pos];
                    let put0 = put;
                    put += 1;
                    Wvalue[put0 as usize] = Uvalue[pos as usize];
                    iwork1[j as usize] += 1;
                    nz += 1;
                }
            }
            Wend[jpivot as usize] = put;
            put += stretch * nz + pad;
            lu_list_move(jpivot, 0, Wflink, Wblink, m, None);
            Unz += nz;
        }
        for k in rank..m {
            let jpivot = pivotcol[k as usize];
            Wbegin[jpivot as usize] = put;
            Wend[jpivot as usize] = put;
            put += pad;
            lu_list_move(jpivot, 0, Wflink, Wblink, m, None);
        }
    }
    assert!(put <= Wend[m as usize]);
    Wbegin[m as usize] = put; // beginning of free space

    // U columnwise.
    Uindex[0] = -1;
    put = 1;
    for k in 0..m as usize {
        // set column pointers
        let j = pivotcol[k] as usize;
        let i = pivotrow[k] as usize;
        let nz = iwork1[j];
        if nz == 0 {
            Ubegin[i] = 0; // empty columns all in position 0
        } else {
            Ubegin[i] = put;
            put += nz;
            Uindex[put as usize] = -1; // terminate column
            put += 1;
        }
        iwork1[j] = Ubegin[i];
    }
    Ubegin[m as usize] = put;
    for k in 0..m as usize {
        // fill columns
        let jpivot = pivotcol[k] as usize;
        let i = pivotrow[k];
        for pos in Wbegin[jpivot]..Wend[jpivot] {
            let j = Windex[pos as usize] as usize;
            put = iwork1[j];
            iwork1[j] += 1;
            assert!(put >= 1);
            Uindex[put as usize] = i;
            Uvalue[put as usize] = Wvalue[pos as usize];
        }
    }

    if cfg!(feature = "debug") {
        for j in 0..m as usize {
            assert_eq!(Uindex[iwork1[j] as usize], -1);
        }
    }

    // Build pivot sequence //

    // Build row-column mappings, overwriting pinv, qinv.
    for k in 0..m as usize {
        let i = pivotrow[k];
        let j = pivotcol[k];
        pmap[j as usize] = i;
        qmap[i as usize] = j;
    }

    // Build pivots by row index.
    let mut max_pivot = 0.0;
    let mut min_pivot = f64::INFINITY;
    for i in 0..m as usize {
        row_pivot[i] = col_pivot[qmap[i] as usize];
        let pivot = row_pivot[i].abs();
        max_pivot = f64::max(pivot, max_pivot);
        min_pivot = f64::min(pivot, min_pivot);
    }

    // memcpy(this.p, pivotrow, m*sizeof(lu_int));
    this.solve.p.copy_from_slice(pivotrow); // TODO: check

    this.min_pivot = min_pivot;
    this.max_pivot = max_pivot;
    this.pivotlen = m;
    this.Lnz = Lnz;
    this.Unz = Unz;
    this.Rnz = 0;

    status
}
