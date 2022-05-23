// Copyright (C) 2016-2019  ERGO-Code
//
// Pivot elimination from active submatrix
//
// lu_pivot() is the only routine callable from extern. It branches out the
// implementation of the pivot operation. The pivot operation removes row
// this.pivot_row and column this.pivot_col from the active submatrix and
// applies a rank-1 update to the remaining active submatrix. It updates the
// row and column counts and the column maxima.
//
// Each pivot elimination adds one column to L and one row to U. On entry
// Lbegin_p[rank] and Ubegin[rank] must point to the next position in Lindex,
// Lvalue respectively Uindex, Uvalue. On return the column in L is terminated
// by index -1 and Lbegin_p[rank+1] and Ubegin[rank+1] point to the next free
// position.
//
// The routines check if memory is sufficient and request reallocation before
// manipulating data structures otherwise.
//
// Updating the columns of the active submatrix is implemented like in the
// Coin-OR code Clp (J. Forrest) by compressing unmodified entries and then
// appending the entries updated or filled-in by the pivot column. Compared
// to the technique from the Suhl/Suhl paper, this method often produces
// sparser factors. I assume the reason is tie breaking in the Markowitz
// search and that in Forrest's method updated elements are moved to the end
// of the column (and likewise for rows).

use crate::basiclu::*;
use crate::lu_def::{lu_fswap, lu_iswap};
use crate::lu_file::{lu_file_compress, lu_file_reappend};
use crate::lu_internal::lu;
use crate::lu_list::{lu_list_move, lu_list_remove};
use std::time::Instant;

// MAXROW_SMALL is the maximum number of off-diagonal elements in the pivot
// column handled by lu_pivot_small(). lu_pivot_small() uses int64_t integers
// for bit masking. Since each row to be updated requires one bit, the routine
// can handle pivot operations for up to 64 rows (excluding pivot row).
//
// Since int64_t is optional in the C99 standard, using it limits portability
// of the code. However, using a fixed threshold to switch between
// lu_pivot_small() and lu_pivot_any() guarantees identical pivot operations
// on all architectures. If int64_t does not exist, then the user can adapt it
// by hand and is aware of it.
const MAXROW_SMALL: lu_int = 64;

pub(crate) fn lu_pivot(this: &mut lu) -> lu_int {
    let m = this.m;
    let rank = this.rank;
    let Lmem = this.Lmem;
    let Umem = this.Umem;
    let pivot_col = this.pivot_col;
    let pivot_row = this.pivot_row;
    let colmax = this.col_pivot;
    let Lbegin_p = this.Lbegin_p;
    let Ubegin = this.Ubegin;
    let Wbegin = this.Wbegin;
    let Wend = this.Wend;
    let Uindex = this.Uindex.unwrap();
    let nz_col = Wend[pivot_col] - Wbegin[pivot_col];
    let nz_row = Wend[m + pivot_row] - Wbegin[m + pivot_row];

    let mut status = BASICLU_OK;
    let tic = Instant::now();

    assert!(nz_row >= 1);
    assert!(nz_col >= 1);

    // Check if room is available in L and U.
    let room = Lmem - Lbegin_p[rank];
    let need = nz_col; // # off-diagonals in pivot col + end marker (-1)
    if room < need {
        this.addmemL = need - room;
        status = BASICLU_REALLOCATE;
    }
    let room = Umem - Ubegin[rank];
    let need = nz_row - 1; // # off-diagonals in pivot row
    if room < need {
        this.addmemU = need - room;
        status = BASICLU_REALLOCATE;
    }
    if status != BASICLU_OK {
        return status;
    }

    // Branch out implementation of pivot operation.
    if nz_row == 1 {
        status = lu_pivot_singleton_row(this);
    } else if nz_col == 1 {
        status = lu_pivot_singleton_col(this);
    } else if nz_col == 2 {
        status = lu_pivot_doubleton_col(this);
    } else if nz_col - 1 <= MAXROW_SMALL {
        status = lu_pivot_small(this);
    } else {
        status = lu_pivot_any(this);
    }

    // Remove all entries in columns whose maximum entry has dropped below
    // absolute pivot tolerance.
    if status == BASICLU_OK {
        for pos in Ubegin[rank]..Ubegin[rank + 1] {
            let j = Uindex[pos];
            assert_ne!(j, pivot_col);
            if colmax[j] == 0.0 || colmax[j] < this.abstol {
                lu_remove_col(this, j);
            }
        }
    }

    this.factor_flops += (nz_col - 1) * (nz_row - 1);
    this.time_elim_pivot += tic.elapsed();
    status
}

fn lu_pivot_any(this: &mut lu) -> lu_int {
    let m = this.m;
    let rank = this.rank;
    let droptol = this.droptol;
    let pad = this.pad;
    let stretch = this.stretch;
    let pivot_col = this.pivot_col;
    let pivot_row = this.pivot_row;
    let colcount_flink = this.colcount_flink;
    let colcount_blink = this.colcount_blink;
    let rowcount_flink = this.rowcount_flink;
    let rowcount_blink = this.rowcount_blink;
    let colmax = this.col_pivot;
    let Lbegin_p = this.Lbegin_p;
    let Ubegin = this.Ubegin;
    let Wbegin = this.Wbegin;
    let Wend = this.Wend;
    let Wflink = this.Wflink;
    let Wblink = this.Wblink;
    let Lindex = this.Lindex;
    let Lvalue = this.Lvalue;
    let Uindex = this.Uindex;
    let Uvalue = this.Uvalue;
    let Windex = this.Windex;
    let Wvalue = this.Wvalue;
    let marked = this.iwork0;
    let work = this.work0;

    let cbeg = Wbegin[pivot_col]; // changed by file compression
    let cend = Wend[pivot_col];
    let rbeg = Wbegin[m + pivot_row];
    let rend = Wend[m + pivot_row];
    let cnz1 = cend - cbeg - 1; // nz in pivot column except pivot
    let rnz1 = rend - rbeg - 1; // nz in pivot row except pivot

    // lu_int i, j, pos, pos1, rpos, put, Uput, where, nz, *wi;
    // lu_int grow, room, found, position;
    // double pivot, a, x, cmx, xrj, *wx;

    // Check if room is available in W. At most each updated row and each
    // updated column will be reappended and filled-in with rnz1 respectively
    // cnz1 elements. Move pivot to the front of pivot row and pivot column.
    let mut grow = 0;
    let mut where_ = -1;
    for pos in cbeg..cend {
        // if ((i = Windex[pos]) == pivot_row) TODO: check
        i = Windex[pos];
        if i == pivot_row {
            where_ = pos;
        } else {
            nz = Wend[m + i] - Wbegin[m + i];
            grow += nz + rnz1 + stretch * (nz + rnz1) + pad;
        }
    }
    assert!(where_ >= 0);
    lu_iswap(Windex, cbeg, where_);
    lu_fswap(Wvalue, cbeg, where_);
    pivot = Wvalue[cbeg];
    assert!(pivot != 0);
    where_ = -1;
    for rpos in rbeg..rend {
        // if ((j = Windex[rpos]) == pivot_col) TODO: check
        j = Windex[rpos];
        if j == pivot_col {
            where_ = rpos;
        } else {
            nz = Wend[j] - Wbegin[j];
            grow += nz + cnz1 + stretch * (nz + cnz1) + pad;
        }
    }
    assert!(where_ >= 0);
    lu_iswap(Windex, rbeg, where_);
    room = Wend[2 * m] - Wbegin[2 * m];
    if grow > room {
        lu_file_compress(2 * m, Wbegin, Wend, Wflink, Windex, Wvalue, stretch, pad);
        cbeg = Wbegin[pivot_col];
        cend = Wend[pivot_col];
        rbeg = Wbegin[m + pivot_row];
        rend = Wend[m + pivot_row];
        room = Wend[2 * m] - Wbegin[2 * m];
        this.ngarbage += 1;
    }
    if grow > room {
        this.addmemW = grow - room;
        return BASICLU_REALLOCATE;
    }

    // get pointer to U
    Uput = Ubegin[rank];
    assert!(Uput >= 0);
    assert!(Uput < this.Umem);

    // Column file update //

    // For each row i to be updated set marked[i] > 0 to its position
    // in the (packed) pivot column.
    position = 1;
    for pos in (cbeg + 1)..cend {
        i = Windex[pos];
        marked[i] = position;
        position += 1;
    }

    wi = Windex + cbeg;
    wx = Wvalue + cbeg;
    for rpos in (rbeg + 1)..rend {
        j = Windex[rpos];
        assert_ne!(j, pivot_col);
        cmx = 0.0; // column maximum

        // Compress unmodified column entries. Store entries to be updated
        // in workspace. Move pivot row entry to the front of column.
        where_ = -1;
        put = pos1 = Wbegin[j];
        for pos in pos1..Wend[j] {
            i = Windex[pos];
            // if ((position = marked[i]) > 0) {
            position = marked[i];
            if position > 0 {
                assert_ne!(i, pivot_row);
                work[position] = Wvalue[pos];
            } else {
                assert_eq!(position, 0);
                x = fabs(Wvalue[pos]);
                if i == pivot_row {
                    where_ = put;
                // } else if ((x = fabs(Wvalue[pos])) > cmx) {
                } else if x > cmx {
                    cmx = x;
                }
                Windex[put] = Windex[pos];
                Wvalue[put] = Wvalue[pos];
                put += 1;
            }
        }
        assert!(where_ >= 0);
        Wend[j] = put;
        lu_iswap(Windex, pos1, where_);
        lu_fswap(Wvalue, pos1, where_);
        xrj = Wvalue[pos1]; // pivot row entry

        // Reappend column if no room for update.
        room = Wbegin[Wflink[j]] - put;
        if room < cnz1 {
            nz = Wend[j] - Wbegin[j];
            room = cnz1 + stretch * (nz + cnz1) + pad;
            lu_file_reappend(j, 2 * m, Wbegin, Wend, Wflink, Wblink, Windex, Wvalue, room);
            put = Wend[j];
            assert(Wbegin[Wflink[j]] - put == room);
            this.nexpand += 1;
        }

        // Compute update in workspace and append to column.
        a = xrj / pivot;
        for pos in 1..=cnz1 {
            work[pos] -= a * wx[pos];
        }
        for pos in 1..=cnz1 {
            Windex[put] = wi[pos];
            Wvalue[put] = work[pos];
            put += 1;
            // if ((x = fabs(work[pos])) > cmx) {
            x = fabs(work[pos]);
            if x > cmx {
                cmx = x;
            }
            work[pos] = 0.0;
        }
        Wend[j] = put;

        // Write pivot row entry to U and remove from file.
        if fabs(xrj) > droptol {
            assert!(Uput < this.Umem);
            Uindex[Uput] = j;
            Uvalue[Uput] = xrj;
            Uput += 1;
        }
        assert_eq!(Windex[Wbegin[j]], pivot_row);
        Wbegin[j] += 1;

        // Move column to new list and update min_colnz.
        nz = Wend[j] - Wbegin[j];
        lu_list_move(j, nz, colcount_flink, colcount_blink, m, &this.min_colnz);

        colmax[j] = cmx;
    }
    for pos in (cbeg + 1)..cend {
        marked[Windex[pos]] = 0;
    }

    // Row file update //

    for rpos in rbeg..rend {
        marked[Windex[rpos]] = 1;
    }
    assert_eq!(marked[pivot_col], 1);

    for pos in (cbeg + 1)..cend {
        i = Windex[pos];
        assert_ne!(i, pivot_row);

        // Compress unmodified row entries (not marked). Remove
        // overlap with pivot row, including pivot column entry.
        found = 0;
        put = Wbegin[m + i];
        for rpos in Wbegin[m + i]..Wend[m + i] {
            j = Windex[rpos];
            if j == pivot_col {
                found = 1;
            }
            if !marked[j] {
                Windex[put] = j;
                put += 1;
            }
        }
        assert!(found);
        Wend[m + i] = put;

        // Reappend row if no room for update. Append pattern of pivot row.
        room = Wbegin[Wflink[m + i]] - put;
        if room < rnz1 {
            nz = Wend[m + i] - Wbegin[m + i];
            room = rnz1 + stretch * (nz + rnz1) + pad;
            lu_file_reappend(
                m + i,
                2 * m,
                Wbegin,
                Wend,
                Wflink,
                Wblink,
                Windex,
                Wvalue,
                room,
            );
            put = Wend[m + i];
            assert_eq!(Wbegin[Wflink[m + i]] - put, room);
            this.nexpand += 1;
        }
        for rpos in (rbeg + 1)..rend {
            Windex[put] = Windex[rpos];
            put += 1;
        }
        Wend[m + i] = put;

        // Move to new list. The row must be reinserted even if nz are
        // unchanged since it might have been taken out in Markowitz search.
        nz = Wend[m + i] - Wbegin[m + i];
        lu_list_move(i, nz, rowcount_flink, rowcount_blink, m, &this.min_rownz);
    }
    for rpos in rbeg..rend {
        marked[Windex[rpos]] = 0;
    }

    // Store column in L.
    put = Lbegin_p[rank];
    for pos in (cbeg + 1)..cend {
        x = Wvalue[pos] / pivot;
        if fabs(x) > droptol {
            Lindex[put] = Windex[pos];
            Lvalue[put] = x;
            put += 1;
        }
    }
    Lindex[put] = -1; // terminate column
    put += 1;
    Lbegin_p[rank + 1] = put;
    Ubegin[rank + 1] = Uput;

    // Cleanup:
    // store pivot element;
    // remove pivot colum from column file, pivot row from row file;
    // remove pivot column/row from column/row counts
    colmax[pivot_col] = pivot;
    Wend[pivot_col] = cbeg;
    Wend[m + pivot_row] = rbeg;
    lu_list_remove(colcount_flink, colcount_blink, pivot_col);
    lu_list_remove(rowcount_flink, rowcount_blink, pivot_row);

    // Check that row file and column file are consistent. Only use when
    // DEBUG_EXTRA since this check is really expensive.
    if cfg!(feature == debug_extra) {
        assert_eq!(
            lu_file_diff(m, Wbegin + m, Wend + m, Wbegin, Wend, Windex, None),
            0
        );
        assert_eq!(
            lu_file_diff(m, Wbegin, Wend, Wbegin + m, Wend + m, Windex, None),
            0
        );
    }

    BASICLU_OK
}

fn lu_pivot_small(this: &mut lu) -> lu_int {
    let m = this.m;
    let rank = this.rank;
    let droptol = this.droptol;
    let pad = this.pad;
    let stretch = this.stretch;
    let pivot_col = this.pivot_col;
    let pivot_row = this.pivot_row;
    let colcount_flink = this.colcount_flink;
    let colcount_blink = this.colcount_blink;
    let rowcount_flink = this.rowcount_flink;
    let rowcount_blink = this.rowcount_blink;
    let colmax = this.col_pivot;
    let Lbegin_p = this.Lbegin_p;
    let Ubegin = this.Ubegin;
    let Wbegin = this.Wbegin;
    let Wend = this.Wend;
    let Wflink = this.Wflink;
    let Wblink = this.Wblink;
    let Lindex = this.Lindex;
    let Lvalue = this.Lvalue;
    let Uindex = this.Uindex;
    let Uvalue = this.Uvalue;
    let Windex = this.Windex;
    let Wvalue = this.Wvalue;
    let marked = this.iwork0;
    let work = this.work0;
    // int64_t *cancelled      = (void *) this.row_pivot;
    let cancelled = this.row_pivot;

    let cbeg = Wbegin[pivot_col]; // changed by file compression
    let cend = Wend[pivot_col];
    let rbeg = Wbegin[m + pivot_row];
    let rend = Wend[m + pivot_row];
    let cnz1 = cend - cbeg - 1; // nz in pivot column except pivot
    let rnz1 = rend - rbeg - 1; // nz in pivot row except pivot

    // lu_int i, j, pos, pos1, rpos, put, Uput, where_, nz, *wi;
    // lu_int grow, room, found, position, col_number;
    // double pivot, a, x, cmx, xrj, *wx;
    // int64_t mask;

    assert!(cnz1 <= MAXROW_SMALL);

    // Check if room is available in W. At most each updated row and each
    // updated column will be reappended and filled-in with rnz1 respectively
    // cnz1 elements. Move pivot to the front of pivot row and pivot column.
    grow = 0;
    where_ = -1;
    for pos in cbeg..cend {
        i = Windex[pos];
        if i == pivot_row {
            where_ = pos;
        } else {
            nz = Wend[m + i] - Wbegin[m + i];
            grow += nz + rnz1 + stretch * (nz + rnz1) + pad;
        }
    }
    assert!(where_ >= 0);
    lu_iswap(Windex, cbeg, where_);
    lu_fswap(Wvalue, cbeg, where_);
    pivot = Wvalue[cbeg];
    assert!(pivot);
    where_ = -1;
    for rpos in rbeg..rend {
        j = Windex[rpos];
        // if ((j = Windex[rpos]) == pivot_col)
        if j == pivot_col {
            where_ = rpos;
        } else {
            nz = Wend[j] - Wbegin[j];
            grow += nz + cnz1 + stretch * (nz + cnz1) + pad;
        }
    }
    assert!(where_ >= 0);
    lu_iswap(Windex, rbeg, where_);
    room = Wend[2 * m] - Wbegin[2 * m];
    if grow > room {
        lu_file_compress(2 * m, Wbegin, Wend, Wflink, Windex, Wvalue, stretch, pad);
        cbeg = Wbegin[pivot_col];
        cend = Wend[pivot_col];
        rbeg = Wbegin[m + pivot_row];
        rend = Wend[m + pivot_row];
        room = Wend[2 * m] - Wbegin[2 * m];
        this.ngarbage += 1;
    }
    if grow > room {
        this.addmemW = grow - room;
        return BASICLU_REALLOCATE;
    }

    // get pointer to U
    Uput = Ubegin[rank];
    assert!(Uput >= 0);
    assert!(Uput < this.Umem);

    // Column file update //

    // For each row i to be updated set marked[i] > 0 to its position
    // in the (packed) pivot column.
    position = 1;
    for pos in (cbeg + 1)..cend {
        i = Windex[pos];
        marked[i] = position;
        position += 1;
    }

    wi = Windex + cbeg;
    wx = Wvalue + cbeg;
    col_number = 0; // mask cancelled[col_number]
                    // for (rpos = rbeg+1; rpos < rend; rpos++, col_number++) {
    for rpos in (rbeg + 1)..rend {
        j = Windex[rpos];
        assert_ne(j, pivot_col);
        cmx = 0.0; // column maximum

        // Compress unmodified column entries. Store entries to be updated
        // in workspace. Move pivot row entry to the front of column.
        where_ = -1;
        put = pos1 = Wbegin[j];
        for pos in pos1..Wend[j] {
            i = Windex[pos];
            // if ((position = marked[i]) > 0)
            position = marked[i];
            if position > 0 {
                assert_ne!(i, pivot_row);
                work[position] = Wvalue[pos];
            } else {
                assert_eq!(position, 0);
                x = fabs(Wvalue[pos]);
                if i == pivot_row {
                    where_ = put;
                // } else if ((x = fabs(Wvalue[pos])) > cmx) {
                } else if x > cmx {
                    cmx = x;
                }
                Windex[put] = Windex[pos];
                Wvalue[put] = Wvalue[pos];
                put += 1;
            }
        }
        assert!(where_ >= 0);
        Wend[j] = put;
        lu_iswap(Windex, pos1, where_);
        lu_fswap(Wvalue, pos1, where_);
        xrj = Wvalue[pos1]; // pivot row entry

        // Reappend column if no room for update.
        room = Wbegin[Wflink[j]] - put;
        if room < cnz1 {
            nz = Wend[j] - Wbegin[j];
            room = cnz1 + stretch * (nz + cnz1) + pad;
            lu_file_reappend(j, 2 * m, Wbegin, Wend, Wflink, Wblink, Windex, Wvalue, room);
            put = Wend[j];
            assert(Wbegin[Wflink[j]] - put == room);
            this.nexpand += 1;
        }

        // Compute update in workspace and append to column.
        a = xrj / pivot;
        for pos in 1..=cnz1 {
            work[pos] -= a * wx[pos];
        }
        mask = 0;
        for pos in 1..=cnz1 {
            x = fabs(work[pos]);
            if x > droptol {
                Windex[put] = wi[pos];
                Wvalue[put] = work[pos];
                put += 1;
                if x > cmx {
                    cmx = x;
                }
            } else {
                // cancellation in row wi[pos]
                // mask |= (int64_t) 1 << (pos-1);
                mask |= (1 << (pos - 1) as i64);
            }
            work[pos] = 0.0;
        }
        Wend[j] = put;
        cancelled[col_number] = mask;

        // Write pivot row entry to U and remove from file.
        if fabs(xrj) > droptol {
            assert!(Uput < this.Umem);
            Uindex[Uput] = j;
            Uvalue[Uput] = xrj;
            Uput += 1;
        }
        assert_eq!(Windex[Wbegin[j]], pivot_row);
        Wbegin[j] += 1;

        // Move column to new list and update min_colnz.
        nz = Wend[j] - Wbegin[j];
        lu_list_move(j, nz, colcount_flink, colcount_blink, m, &this.min_colnz);

        colmax[j] = cmx;

        col_number += 1;
    }
    for pos in (cbeg + 1)..cend {
        marked[Windex[pos]] = 0;
    }

    // Row file update //

    for rpos in rbeg..rend {
        marked[Windex[rpos]] = 1;
    }
    assert_eq!(marked[pivot_col], 1);

    mask = 1;
    // for (pos = cbeg+1; pos < cend; pos++, mask <<= 1)
    for pos in (cbeg + 1)..cend {
        assert!(mask);
        i = Windex[pos];
        assert_ne!(i, pivot_row);

        // Compress unmodified row entries (not marked). Remove
        // overlap with pivot row, including pivot column entry.
        found = 0;
        put = Wbegin[m + i];
        for rpos in Wbegin[m + i]..Wend[m + i] {
            j = Windex[rpos];
            // if ((j = Windex[rpos]) == pivot_col)
            if j == pivot_col {
                found = 1;
            }
            if !marked[j] {
                Windex[put] = j;
                put += 1;
            }
        }
        assert!(found);
        Wend[m + i] = put;

        // Reappend row if no room for update. Append pattern of pivot row.
        room = Wbegin[Wflink[m + i]] - put;
        if room < rnz1 {
            nz = Wend[m + i] - Wbegin[m + i];
            room = rnz1 + stretch * (nz + rnz1) + pad;
            lu_file_reappend(
                m + i,
                2 * m,
                Wbegin,
                Wend,
                Wflink,
                Wblink,
                Windex,
                Wvalue,
                room,
            );
            put = Wend[m + i];
            assert(Wbegin[Wflink[m + i]] - put == room);
            this.nexpand += 1;
        }

        col_number = 0;
        for rpos in (rbeg + 1)..rend {
            if !(cancelled[col_number] & mask) {
                Windex[put] = Windex[rpos];
                put += 1;
            }
            col_number += 1;
        }
        Wend[m + i] = put;

        // Move to new list. The row must be reinserted even if nz are
        // unchanged since it might have been taken out in Markowitz search.
        nz = Wend[m + i] - Wbegin[m + i];
        lu_list_move(i, nz, rowcount_flink, rowcount_blink, m, &this.min_rownz);

        mask <<= 1;
    }
    for rpos in rbeg..rend {
        marked[Windex[rpos]] = 0;
    }

    // Store column in L.
    put = Lbegin_p[rank];
    for pos in (cbeg + 1)..cend {
        x = Wvalue[pos] / pivot;
        if fabs(x) > droptol {
            Lindex[put] = Windex[pos];
            Lvalue[put] = x;
            put += 1;
        }
    }
    Lindex[put] = -1; // terminate column
    put += 1;
    Lbegin_p[rank + 1] = put;
    Ubegin[rank + 1] = Uput;

    // Cleanup:
    // store pivot elemnt;
    // remove pivot colum from column file, pivot row from row file;
    // remove pivot column/row from column/row counts
    colmax[pivot_col] = pivot;
    Wend[pivot_col] = cbeg;
    Wend[m + pivot_row] = rbeg;
    lu_list_remove(colcount_flink, colcount_blink, pivot_col);
    lu_list_remove(rowcount_flink, rowcount_blink, pivot_row);

    // Check that row file and column file are consistent. Only use when
    // DEBUG_EXTRA since this check is really expensive.
    if cfg!(feature = debug_extra) {
        assert_eq!(
            lu_file_diff(m, Wbegin + m, Wend + m, Wbegin, Wend, Windex, None),
            0
        );
        assert_eq!(
            lu_file_diff(m, Wbegin, Wend, Wbegin + m, Wend + m, Windex, None),
            0
        );
    }

    BASICLU_OK
}

fn lu_pivot_singleton_row(this: &mut lu) -> lu_int {
    let m = this.m;
    let rank = this.rank;
    let droptol = this.droptol;
    let pivot_col = this.pivot_col;
    let pivot_row = this.pivot_row;
    let colcount_flink = this.colcount_flink;
    let colcount_blink = this.colcount_blink;
    let rowcount_flink = this.rowcount_flink;
    let rowcount_blink = this.rowcount_blink;
    let colmax = this.col_pivot;
    let Lbegin_p = this.Lbegin_p;
    let Ubegin = this.Ubegin;
    let Wbegin = this.Wbegin;
    let Wend = this.Wend;
    let Lindex = this.Lindex.unwrap();
    let Lvalue = this.Lvalue.unwrap();
    let Windex = this.Windex.unwrap();
    let Wvalue = this.Wvalue.unwrap();

    let cbeg = Wbegin[pivot_col];
    let cend = Wend[pivot_col];
    let rbeg = Wbegin[m + pivot_row];
    let rend = Wend[m + pivot_row];
    let rnz1 = rend - rbeg - 1; /* nz in pivot row except pivot */

    // lu_int i, pos, put, nz, where_;
    // double pivot, x;

    assert_eq!(rnz1, 0);

    // Find pivot.
    where_ = cbeg;
    while Windex[where_] != pivot_row {
        assert!(where_ < cend - 1);
        where_ += 1;
    }
    pivot = Wvalue[where_];
    assert!(pivot);

    // Store column in L.
    put = Lbegin_p[rank];
    for pos in cbeg..cend {
        x = Wvalue[pos] / pivot;
        if pos != where_ && fabs(x) > droptol {
            Lindex[put] = Windex[pos];
            Lvalue[put] = x;
            put += 1;
        }
    }
    Lindex[put] = -1; // terminate column
    put += 1;
    Lbegin_p[rank + 1] = put;
    Ubegin[rank + 1] = Ubegin[rank];

    // Remove pivot column from row file. Update row lists.
    for pos in cbeg..cend {
        i = Windex[pos];
        if i == pivot_row {
            continue;
        }
        where_ = Wbegin[m + i];
        while Windex[where_] != pivot_col {
            assert(where_ < Wend[m + i] - 1);
            where_ += 1;
        }
        // Windex[where_] = Windex[--Wend[m+i]];
        Wend[m + i] -= 1;
        Windex[where_] = Windex[Wend[m + i]];
        nz = Wend[m + i] - Wbegin[m + i];
        lu_list_move(i, nz, rowcount_flink, rowcount_blink, m, &this.min_rownz);
    }

    // Cleanup:
    // store pivot elemnt;
    // remove pivot colum from column file, pivot row from row file;
    // remove pivot column/row from column/row counts
    colmax[pivot_col] = pivot;
    Wend[pivot_col] = cbeg;
    Wend[m + pivot_row] = rbeg;
    lu_list_remove(colcount_flink, colcount_blink, pivot_col);
    lu_list_remove(rowcount_flink, rowcount_blink, pivot_row);

    BASICLU_OK
}

fn lu_pivot_singleton_col(this: &mut lu) -> lu_int {
    let m = this.m;
    let rank = this.rank;
    let droptol = this.droptol;
    let pivot_col = this.pivot_col;
    let pivot_row = this.pivot_row;
    let colcount_flink = this.colcount_flink;
    let colcount_blink = this.colcount_blink;
    let rowcount_flink = this.rowcount_flink;
    let rowcount_blink = this.rowcount_blink;
    let colmax = this.col_pivot;
    let Lbegin_p = this.Lbegin_p;
    let Ubegin = this.Ubegin;
    let Wbegin = this.Wbegin;
    let Wend = this.Wend;
    let Lindex = this.Lindex;
    let Uindex = this.Uindex;
    let Uvalue = this.Uvalue;
    let Windex = this.Windex;
    let Wvalue = this.Wvalue;

    let cbeg = Wbegin[pivot_col];
    let cend = Wend[pivot_col];
    let rbeg = Wbegin[m + pivot_row];
    let rend = Wend[m + pivot_row];
    let cnz1 = cend - cbeg - 1; /* nz in pivot column except pivot */

    // lu_int j, pos, rpos, put, nz, where_, found;
    // double pivot, cmx, x, xrj;

    assert_eq!(cnz1, 0);

    // Remove pivot row from column file and store in U. Update column lists.
    put = Ubegin[rank];
    pivot = Wvalue[cbeg];
    assert(pivot);
    found = 0;
    xrj = 0.0; // initialize to make gcc happy
    for rpos in rbeg..rend {
        j = Windex[rpos];
        if j == pivot_col {
            found = 1;
            continue;
        }
        where_ = -1;
        cmx = 0.0; // column maximum
        for pos in Wbegin[j]..Wend[j] {
            x = fabs(Wvalue[pos]);
            if Windex[pos] == pivot_row {
                where_ = pos;
                xrj = Wvalue[pos];
            // } else if ((x = fabs(Wvalue[pos])) > cmx) {
            } else if x > cmx {
                cmx = x;
            }
        }
        assert!(where_ >= 0);
        if fabs(xrj) > droptol {
            Uindex[put] = j;
            Uvalue[put] = xrj;
            put += 1;
        }
        // Windex[where_] = Windex[--Wend [j]];
        Wend[j] -= 1;
        Windex[where_] = Windex[Wend[j]];
        Wvalue[where_] = Wvalue[Wend[j]];
        nz = Wend[j] - Wbegin[j];
        lu_list_move(j, nz, colcount_flink, colcount_blink, m, &this.min_colnz);
        colmax[j] = cmx;
    }
    assert!(found);
    Ubegin[rank + 1] = put;

    // Store empty column in L.
    put = Lbegin_p[rank];
    Lindex[put] = -1; // terminate column
    put += 1;
    Lbegin_p[rank + 1] = put;

    // Cleanup:
    // store pivot element;
    // remove pivot colum from column file, pivot row from row file;
    // remove pivot column/row from column/row counts
    colmax[pivot_col] = pivot;
    Wend[pivot_col] = cbeg;
    Wend[m + pivot_row] = rbeg;
    lu_list_remove(colcount_flink, colcount_blink, pivot_col);
    lu_list_remove(rowcount_flink, rowcount_blink, pivot_row);

    BASICLU_OK
}

fn lu_pivot_doubleton_col(this: &mut lu) -> lu_int {
    let m = this.m;
    let rank = this.rank;
    let droptol = this.droptol;
    let pad = this.pad;
    let stretch = this.stretch;
    let pivot_col = this.pivot_col;
    let pivot_row = this.pivot_row;
    let colcount_flink = this.colcount_flink;
    let colcount_blink = this.colcount_blink;
    let rowcount_flink = this.rowcount_flink;
    let rowcount_blink = this.rowcount_blink;
    let colmax = this.col_pivot;
    let Lbegin_p = this.Lbegin_p;
    let Ubegin = this.Ubegin;
    let Wbegin = this.Wbegin;
    let Wend = this.Wend;
    let Wflink = this.Wflink;
    let Wblink = this.Wblink;
    let Lindex = this.Lindex;
    let Lvalue = this.Lvalue;
    let Uindex = this.Uindex;
    let Uvalue = this.Uvalue;
    let Windex = this.Windex;
    let Wvalue = this.Wvalue;
    let marked = this.iwork0;

    let cbeg = Wbegin[pivot_col]; // changed by file compression
    let cend = Wend[pivot_col];
    let rbeg = Wbegin[m + pivot_row];
    let rend = Wend[m + pivot_row];
    let cnz1 = cend - cbeg - 1; // nz in pivot column except pivot
    let rnz1 = rend - rbeg - 1; // nz in pivot row except pivot

    // lu_int j, pos, rpos, put, Uput, nz, nfill, where_, where_pivot, where_other;
    // lu_int other_row, grow, room, space, end, ncancelled;
    // double pivot, other_value, xrj, cmx, x, xabs;

    assert_eq!(cnz1, 1);

    /* Move pivot element to front of pivot column and pivot row. */
    if Windex[cbeg] != pivot_row {
        lu_iswap(Windex, cbeg, cbeg + 1);
        lu_fswap(Wvalue, cbeg, cbeg + 1);
    }
    assert_eq!(Windex[cbeg], pivot_row);
    let pivot = Wvalue[cbeg];
    assert!(pivot);
    other_row = Windex[cbeg + 1];
    other_value = Wvalue[cbeg + 1];
    where_ = rbeg;
    while Windex[where_] != pivot_col {
        assert!(where_ < rend - 1);
        where_ += 1;
    }
    lu_iswap(Windex, rbeg, where_);

    // Check if room is available in W.
    // Columns can be updated in place but the updated row may need to be
    // expanded.
    nz = Wend[m + other_row] - Wbegin[m + other_row];
    grow = nz + rnz1 + stretch * (nz + rnz1) + pad;
    room = Wend[2 * m] - Wbegin[2 * m];
    if grow > room {
        lu_file_compress(2 * m, Wbegin, Wend, Wflink, Windex, Wvalue, stretch, pad);
        cbeg = Wbegin[pivot_col];
        cend = Wend[pivot_col];
        rbeg = Wbegin[m + pivot_row];
        rend = Wend[m + pivot_row];
        room = Wend[2 * m] - Wbegin[2 * m];
        this.ngarbage += 1;
    }
    if grow > room {
        this.addmemW = grow - room;
        return BASICLU_REALLOCATE;
    }

    // Column file update //

    Uput = Ubegin[rank];
    put = rbeg + 1;
    ncancelled = 0;
    for rpos in (rbeg + 1)..rend {
        j = Windex[rpos];
        assert_ne!(j, pivot_col);
        cmx = 0.0; // column maximum

        // Find position of pivot row entry and possibly other row entry in
        // column j.
        where_pivot = -1;
        where_other = -1;
        end = Wend[j];
        for pos in Wbegin[j]..end {
            x = fabs(Wvalue[pos]);
            if Windex[pos] == pivot_row {
                where_pivot = pos;
            } else if Windex[pos] == other_row {
                where_other = pos;
            // } else if ((x = fabs(Wvalue[pos])) > cmx) {
            } else if x > cmx {
                cmx = x;
            }
        }
        assert!(where_pivot >= 0);
        xrj = Wvalue[where_pivot];

        // Store pivot row entry in U.
        if fabs(Wvalue[where_pivot]) > droptol {
            Uindex[Uput] = j;
            Uvalue[Uput] = Wvalue[where_pivot];
            Uput += 1;
        }

        if where_other == -1 {
            // Compute fill-in element.
            x = -xrj * (other_value / pivot);
            xabs = fabs(x);
            if xabs > droptol {
                // Store fill-in where pivot row entry was.
                Windex[where_pivot] = other_row;
                Wvalue[where_pivot] = x;
                Windex[put] = j;
                put += 1;
                if xabs > cmx {
                    cmx = xabs;
                }
            } else {
                // Remove pivot row entry.
                // end = --Wend[j]; TODO: check
                Wend[j] -= 1;
                end = Wend[j];
                Windex[where_pivot] = Windex[end];
                Wvalue[where_pivot] = Wvalue[end];

                // Decrease column count.
                nz = end - Wbegin[j];
                lu_list_move(j, nz, colcount_flink, colcount_blink, m, &this.min_colnz);
            }
        } else {
            // Remove pivot row entry and update other row entry.
            // end = --Wend[j]; TODO: check
            Wend[j] -= 1;
            end = Wend[j];
            Windex[where_pivot] = Windex[end];
            Wvalue[where_pivot] = Wvalue[end];
            if where_other == end {
                where_other = where_pivot;
            }
            Wvalue[where_other] -= xrj * (other_value / pivot);

            // If we have numerical cancellation, then remove the entry and mark
            // the column.
            x = fabs(Wvalue[where_other]);
            if x <= droptol {
                // end = --Wend[j]; TODO
                Wend[j] -= 1;
                end = Wend[j];
                Windex[where_other] = Windex[end];
                Wvalue[where_other] = Wvalue[end];
                marked[j] = 1;
                ncancelled += 1;
            } else if x > cmx {
                cmx = x;
            }

            // Decrease column count.
            nz = Wend[j] - Wbegin[j];
            lu_list_move(j, nz, colcount_flink, colcount_blink, m, &this.min_colnz);
        }
        colmax[j] = cmx;
    }
    rend = put;
    Ubegin[rank + 1] = Uput;

    // Row file update //

    // If we have numerical cancellation, then we have to remove these entries
    // (marked) from the row pattern. In any case remove pivot column entry.
    if ncancelled {
        assert_eq!(marked[pivot_col], 0);
        marked[pivot_col] = 1; // treat as cancelled
        put = Wbegin[m + other_row]; // compress remaining entries
        end = Wend[m + other_row];
        for pos in put..end {
            j = Windex[pos];
            if marked[j] {
                marked[j] = 0;
            } else {
                Windex[put] = j;
                put += 1;
            }
        }
        assert_eq!(end - put, ncancelled + 1);
        Wend[m + other_row] = put;
    } else {
        where_ = Wbegin[m + other_row];
        while Windex[where_] != pivot_col {
            assert!(where_ < Wend[m + other_row] - 1);
            where_ += 1;
        }
        // end = --Wend[m+other_row]; TODO: check
        Wend[m + other_row] -= 1;
        end = Wend[m + other_row];
        Windex[where_] = Windex[end];
    }

    // Reappend row if no room for update.
    nfill = rend - (rbeg + 1);
    room = Wbegin[Wflink[m + other_row]] - Wend[m + other_row];
    if nfill > room {
        nz = Wend[m + other_row] - Wbegin[m + other_row];
        space = nfill + stretch * (nz + nfill) + pad;
        lu_file_reappend(
            m + other_row,
            2 * m,
            Wbegin,
            Wend,
            Wflink,
            Wblink,
            Windex,
            Wvalue,
            space,
        );
        this.nexpand += 1;
    }

    // Append fill-in to row pattern.
    put = Wend[m + other_row];
    for pos in (rbeg + 1)..rend {
        Windex[put] = Windex[pos];
        put += 1;
    }
    Wend[m + other_row] = put;

    // Reinsert other row into row counts.
    nz = Wend[m + other_row] - Wbegin[m + other_row];
    lu_list_move(
        other_row,
        nz,
        rowcount_flink,
        rowcount_blink,
        m,
        &this.min_rownz,
    );

    // Store column in L.
    put = Lbegin_p[rank];
    x = other_value / pivot;
    if fabs(x) > droptol {
        Lindex[put] = other_row;
        Lvalue[put] = x;
        put += 1;
    }
    Lindex[put] = -1; // terminate column
    put += 1;
    Lbegin_p[rank + 1] = put;

    // Cleanup:
    // store pivot elemnt;
    // remove pivot colum from column file, pivot row from row file;
    // remove pivot column/row from column/row counts
    colmax[pivot_col] = pivot;
    Wend[pivot_col] = cbeg;
    Wend[m + pivot_row] = rbeg;
    lu_list_remove(colcount_flink, colcount_blink, pivot_col);
    lu_list_remove(rowcount_flink, rowcount_blink, pivot_row);

    // Check that row file and column file are consistent. Only use when
    // DEBUG_EXTRA since this check is really expensive.
    if cfg!(feature = debug_extra) {
        assert_eq!(
            lu_file_diff(m, Wbegin + m, Wend + m, Wbegin, Wend, Windex, None),
            0
        );
        assert_eq!(
            lu_file_diff(m, Wbegin, Wend, Wbegin + m, Wend + m, Windex, None),
            0
        );
    }

    BASICLU_OK
}

fn lu_remove_col(this: &mut lu, j: lu_int) {
    let m = this.m;
    let colcount_flink = this.colcount_flink;
    let colcount_blink = this.colcount_blink;
    let rowcount_flink = this.rowcount_flink;
    let rowcount_blink = this.rowcount_blink;
    let colmax = this.col_pivot;
    let Wbegin = this.Wbegin;
    let Wend = this.Wend;
    let Windex = this.Windex;
    let cbeg = Wbegin[j];
    let cend = Wend[j];

    // lu_int i, pos, nz, where_;

    // Remove column j from row file.
    for pos in cbeg..cend {
        i = Windex[pos];
        where_ = Wbegin[m + i];
        while Windex[where_] != j {
            assert!(where_ < Wend[m + i] - 1);
            where_ += 1;
        }
        // Windex[where_] = Windex[--Wend[m+i]];
        Wend[m + i] -= 1;
        Windex[where_] = Windex[Wend[m + i]];
        nz = Wend[m + i] - Wbegin[m + i];
        lu_list_move(i, nz, rowcount_flink, rowcount_blink, m, &this.min_rownz);
    }

    // Remove column j from column file.
    colmax[j] = 0.0;
    Wend[j] = cbeg;
    lu_list_move(j, 0, colcount_flink, colcount_blink, m, &this.min_colnz);
}
