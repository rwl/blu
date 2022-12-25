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
use crate::lu_def::*;
use crate::lu_file::*;
use crate::lu_internal::LU;
use crate::lu_list::*;
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
const MAXROW_SMALL: LUInt = 64;

pub(crate) fn lu_pivot(this: &mut LU) -> LUInt {
    let m = this.m;
    let rank = this.rank;
    let l_mem = this.l_mem;
    let u_mem = this.u_mem;
    let pivot_col = this.pivot_col;
    let pivot_row = this.pivot_row;
    // let colmax = &this.col_pivot;
    let l_begin_p = &this.l_begin_p;
    // let Ubegin = &this.Ubegin;
    let w_begin = &this.w_begin;
    let w_end = &this.w_end;
    // let Uindex = this.Uindex.as_ref().unwrap();
    let nz_col = w_end[pivot_col as usize] - w_begin[pivot_col as usize];
    let nz_row = w_end[(m + pivot_row) as usize] - w_begin[(m + pivot_row) as usize];

    let mut status = BASICLU_OK;
    let tic = Instant::now();

    assert!(nz_row >= 1);
    assert!(nz_col >= 1);

    // Check if room is available in L and U.
    let room = l_mem - l_begin_p[rank as usize];
    let need = nz_col; // # off-diagonals in pivot col + end marker (-1)
    if room < need {
        this.addmem_l = need - room;
        status = BASICLU_REALLOCATE;
    }
    let room = u_mem - this.u_begin[rank as usize];
    let need = nz_row - 1; // # off-diagonals in pivot row
    if room < need {
        this.addmem_u = need - room;
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
        for pos in this.u_begin[rank as usize]..this.u_begin[(rank + 1) as usize] {
            let j = this.u_index[pos as usize];
            assert_ne!(j, pivot_col);
            if this.col_pivot[j as usize] == 0.0 || this.col_pivot[j as usize] < this.abstol {
                lu_remove_col(this, j);
            }
        }
    }

    this.factor_flops += (nz_col - 1) * (nz_row - 1);
    this.time_elim_pivot += tic.elapsed().as_secs_f64();
    status
}

fn lu_pivot_any(this: &mut LU) -> LUInt {
    let m = this.m;
    let rank = this.rank;
    let droptol = this.droptol;
    let pad = this.pad;
    let stretch = this.stretch;
    let pivot_col = this.pivot_col;
    let pivot_row = this.pivot_row;
    let colcount_flink = &mut this.colcount_flink;
    let colcount_blink = &mut this.colcount_blink;
    let rowcount_flink = &mut this.rowcount_flink;
    let rowcount_blink = &mut this.rowcount_blink;
    let colmax = &mut this.col_pivot;
    let l_begin_p = &mut this.l_begin_p;
    let u_begin = &mut this.u_begin;
    let w_begin = &mut this.w_begin;
    let w_end = &mut this.w_end;
    let w_flink = &mut this.w_flink;
    let w_blink = &mut this.w_blink;
    let l_index = &mut this.l_index;
    let l_value = &mut this.l_value;
    let u_index = &mut this.u_index;
    let u_value = &mut this.u_value;
    let w_index = &mut this.w_index;
    let w_value = &mut this.w_value;
    let marked = &mut this.iwork0;
    let work = &mut this.work0;

    let mut cbeg = w_begin[pivot_col as usize]; // changed by file compression
    let mut cend = w_end[pivot_col as usize];
    let mut rbeg = w_begin[(m + pivot_row) as usize];
    let mut rend = w_end[(m + pivot_row) as usize];
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
        let i = w_index[pos as usize];
        if i == pivot_row {
            where_ = pos;
        } else {
            let nz = w_end[(m + i) as usize] - w_begin[(m + i) as usize];
            grow += nz + rnz1 + (stretch as LUInt) * (nz + rnz1) + pad;
        }
    }
    assert!(where_ >= 0);
    lu_iswap(w_index, cbeg, where_);
    lu_fswap(w_value, cbeg, where_);
    let pivot = w_value[cbeg as usize];
    assert_ne!(pivot, 0.0);
    where_ = -1;
    for rpos in rbeg..rend {
        // if ((j = Windex[rpos]) == pivot_col) TODO: check
        let j = w_index[rpos as usize];
        if j == pivot_col {
            where_ = rpos;
        } else {
            let nz = w_end[j as usize] - w_begin[j as usize];
            grow += nz + cnz1 + (stretch as LUInt) * (nz + cnz1) + pad;
        }
    }
    assert!(where_ >= 0);
    lu_iswap(w_index, rbeg, where_);
    let mut room = w_end[(2 * m) as usize] - w_begin[(2 * m) as usize];
    if grow > room {
        lu_file_compress(
            2 * m,
            w_begin,
            w_end,
            w_flink,
            w_index,
            w_value,
            stretch,
            pad,
        );
        cbeg = w_begin[pivot_col as usize];
        cend = w_end[pivot_col as usize];
        rbeg = w_begin[(m + pivot_row) as usize];
        rend = w_end[(m + pivot_row) as usize];
        room = w_end[(2 * m) as usize] - w_begin[(2 * m) as usize];
        this.ngarbage += 1;
    }
    if grow > room {
        this.addmem_w = grow - room;
        return BASICLU_REALLOCATE;
    }

    // get pointer to U
    let mut u_put = u_begin[rank as usize];
    assert!(u_put >= 0);
    assert!(u_put < this.u_mem);

    // Column file update //

    // For each row i to be updated set marked[i] > 0 to its position
    // in the (packed) pivot column.
    let mut position = 1;
    for pos in (cbeg + 1)..cend {
        let i = w_index[pos as usize];
        marked[i as usize] = position;
        position += 1;
    }

    // let wi = Windex + cbeg;
    // let wx = Wvalue + cbeg;
    for rpos in (rbeg + 1)..rend {
        let j = w_index[rpos as usize];
        assert_ne!(j, pivot_col);
        let mut cmx = 0.0; // column maximum

        // Compress unmodified column entries. Store entries to be updated
        // in workspace. Move pivot row entry to the front of column.
        where_ = -1;
        let mut put = w_begin[j as usize];
        let pos1 = w_begin[j as usize];
        for pos in pos1..w_end[j as usize] {
            let i = w_index[pos as usize];
            // if ((position = marked[i]) > 0) {
            position = marked[i as usize];
            if position > 0 {
                assert_ne!(i, pivot_row);
                work[position as usize] = w_value[pos as usize];
            } else {
                assert_eq!(position, 0);
                let x = w_value[pos as usize].abs();
                if i == pivot_row {
                    where_ = put;
                // } else if ((x = fabs(Wvalue[pos])) > cmx) {
                } else if x > cmx {
                    cmx = x;
                }
                w_index[put as usize] = w_index[pos as usize];
                w_value[put as usize] = w_value[pos as usize];
                put += 1;
            }
        }
        assert!(where_ >= 0);
        w_end[j as usize] = put;
        lu_iswap(w_index, pos1, where_);
        lu_fswap(w_value, pos1, where_);
        let xrj = w_value[pos1 as usize]; // pivot row entry

        // Reappend column if no room for update.
        room = w_begin[w_flink[j as usize] as usize] - put;
        if room < cnz1 {
            let nz = w_end[j as usize] - w_begin[j as usize];
            room = cnz1 + (stretch as LUInt) * (nz + cnz1) + pad;
            lu_file_reappend(
                j,
                2 * m,
                w_begin,
                w_end,
                w_flink,
                w_blink,
                w_index,
                w_value,
                room,
            );
            put = w_end[j as usize];
            assert_eq!(w_begin[w_flink[j as usize] as usize] - put, room);
            this.nexpand += 1;
        }

        // Compute update in workspace and append to column.
        let a = xrj / pivot;
        for pos in 1..=cnz1 {
            // work[pos as usize] -= a * wx[pos as usize];
            work[pos as usize] -= a * w_value[cbeg as usize..][pos as usize];
        }
        for pos in 1..=cnz1 {
            // Windex[put as usize] = wi[pos as usize];
            w_index[put as usize] = w_index[cbeg as usize..][pos as usize];
            w_value[put as usize] = work[pos as usize];
            put += 1;
            // if ((x = fabs(work[pos])) > cmx) {
            let x = work[pos as usize].abs();
            if x > cmx {
                cmx = x;
            }
            work[pos as usize] = 0.0;
        }
        w_end[j as usize] = put;

        // Write pivot row entry to U and remove from file.
        if xrj.abs() > droptol {
            assert!(u_put < this.u_mem);
            u_index[u_put as usize] = j;
            u_value[u_put as usize] = xrj;
            u_put += 1;
        }
        assert_eq!(w_index[w_begin[j as usize] as usize], pivot_row);
        w_begin[j as usize] += 1;

        // Move column to new list and update min_colnz.
        let nz = w_end[j as usize] - w_begin[j as usize];
        lu_list_move(
            j,
            nz,
            colcount_flink,
            colcount_blink,
            m,
            Some(&mut this.min_colnz),
        );

        colmax[j as usize] = cmx;
    }
    for pos in (cbeg + 1)..cend {
        marked[w_index[pos as usize] as usize] = 0;
    }

    // Row file update //

    for rpos in rbeg..rend {
        marked[w_index[rpos as usize] as usize] = 1;
    }
    assert_eq!(marked[pivot_col as usize], 1);

    for pos in (cbeg + 1)..cend {
        let i = w_index[pos as usize];
        assert_ne!(i, pivot_row);

        // Compress unmodified row entries (not marked). Remove
        // overlap with pivot row, including pivot column entry.
        let mut found = 0;
        let mut put = w_begin[(m + i) as usize];
        for rpos in w_begin[(m + i) as usize]..w_end[(m + i) as usize] {
            let j = w_index[rpos as usize];
            if j == pivot_col {
                found = 1;
            }
            if marked[j as usize] == 0 {
                w_index[put as usize] = j;
                put += 1;
            }
        }
        assert_ne!(found, 0);
        w_end[(m + i) as usize] = put;

        // Reappend row if no room for update. Append pattern of pivot row.
        room = w_begin[w_flink[(m + i) as usize] as usize] - put;
        if room < rnz1 {
            let nz = w_end[(m + i) as usize] - w_begin[(m + i) as usize];
            room = rnz1 + (stretch as LUInt) * (nz + rnz1) + pad;
            lu_file_reappend(
                m + i,
                2 * m,
                w_begin,
                w_end,
                w_flink,
                w_blink,
                w_index,
                w_value,
                room,
            );
            put = w_end[(m + i) as usize];
            assert_eq!(w_begin[w_flink[(m + i) as usize] as usize] - put, room);
            this.nexpand += 1;
        }
        for rpos in (rbeg + 1)..rend {
            w_index[put as usize] = w_index[rpos as usize];
            put += 1;
        }
        w_end[(m + i) as usize] = put;

        // Move to new list. The row must be reinserted even if nz are
        // unchanged since it might have been taken out in Markowitz search.
        let nz = w_end[(m + i) as usize] - w_begin[(m + i) as usize];
        lu_list_move(
            i,
            nz,
            rowcount_flink,
            rowcount_blink,
            m,
            Some(&mut this.min_rownz),
        );
    }
    for rpos in rbeg..rend {
        marked[w_index[rpos as usize] as usize] = 0;
    }

    // Store column in L.
    let mut put = l_begin_p[rank as usize];
    for pos in (cbeg + 1)..cend {
        let x = w_value[pos as usize] / pivot;
        if x.abs() > droptol {
            l_index[put as usize] = w_index[pos as usize];
            l_value[put as usize] = x;
            put += 1;
        }
    }
    l_index[put as usize] = -1; // terminate column
    put += 1;
    l_begin_p[(rank + 1) as usize] = put;
    u_begin[(rank + 1) as usize] = u_put;

    // Cleanup:
    // store pivot element;
    // remove pivot colum from column file, pivot row from row file;
    // remove pivot column/row from column/row counts
    colmax[pivot_col as usize] = pivot;
    w_end[pivot_col as usize] = cbeg;
    w_end[(m + pivot_row) as usize] = rbeg;
    lu_list_remove(colcount_flink, colcount_blink, pivot_col);
    lu_list_remove(rowcount_flink, rowcount_blink, pivot_row);

    // Check that row file and column file are consistent. Only use when
    // DEBUG_EXTRA since this check is really expensive.
    if cfg!(feature = "debug_extra") {
        assert_eq!(
            lu_file_diff(
                m,
                &w_begin[m as usize..],
                &w_end[m as usize..],
                w_begin,
                w_end,
                w_index,
                None
            ),
            0
        );
        assert_eq!(
            lu_file_diff(
                m,
                w_begin,
                w_end,
                &w_begin[m as usize..],
                &w_end[m as usize..],
                w_index,
                None
            ),
            0
        );
    }

    BASICLU_OK
}

fn lu_pivot_small(this: &mut LU) -> LUInt {
    let m = this.m;
    let rank = this.rank;
    let droptol = this.droptol;
    let pad = this.pad;
    let stretch = this.stretch;
    let pivot_col = this.pivot_col;
    let pivot_row = this.pivot_row;
    let colcount_flink = &mut this.colcount_flink;
    let colcount_blink = &mut this.colcount_blink;
    let rowcount_flink = &mut this.rowcount_flink;
    let rowcount_blink = &mut this.rowcount_blink;
    let colmax = &mut this.col_pivot;
    let l_begin_p = &mut this.l_begin_p;
    let u_begin = &mut this.u_begin;
    let w_begin = &mut this.w_begin;
    let w_end = &mut this.w_end;
    let w_flink = &mut this.w_flink;
    let w_blink = &mut this.w_blink;
    let l_index = &mut this.l_index;
    let l_value = &mut this.l_value;
    let u_index = &mut this.u_index;
    let u_value = &mut this.u_value;
    let w_index = &mut this.w_index;
    let w_value = &mut this.w_value;
    let marked = &mut this.iwork0;
    let work = &mut this.work0;
    // int64_t *cancelled      = (void *) this.row_pivot;
    let cancelled = &mut this.row_pivot;

    let mut cbeg = w_begin[pivot_col as usize]; // changed by file compression
    let mut cend = w_end[pivot_col as usize];
    let mut rbeg = w_begin[(m + pivot_row) as usize];
    let mut rend = w_end[(m + pivot_row) as usize];
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
    let mut grow = 0;
    let mut where_ = -1;
    for pos in cbeg..cend {
        let i = w_index[pos as usize];
        if i == pivot_row {
            where_ = pos;
        } else {
            let nz = w_end[(m + i) as usize] - w_begin[(m + i) as usize];
            grow += nz + rnz1 + (stretch as LUInt) * (nz + rnz1) + pad;
        }
    }
    assert!(where_ >= 0);
    lu_iswap(w_index, cbeg, where_);
    lu_fswap(w_value, cbeg, where_);
    let pivot = w_value[cbeg as usize];
    assert_ne!(pivot, 0.0);
    where_ = -1;
    for rpos in rbeg..rend {
        let j = w_index[rpos as usize];
        // if ((j = Windex[rpos]) == pivot_col)
        if j == pivot_col {
            where_ = rpos;
        } else {
            let nz = w_end[j as usize] - w_begin[j as usize];
            grow += nz + cnz1 + (stretch as LUInt) * (nz + cnz1) + pad;
        }
    }
    assert!(where_ >= 0);
    lu_iswap(w_index, rbeg, where_);
    let mut room = w_end[(2 * m) as usize] - w_begin[(2 * m) as usize];
    if grow > room {
        lu_file_compress(
            2 * m,
            w_begin,
            w_end,
            w_flink,
            w_index,
            w_value,
            stretch,
            pad,
        );
        cbeg = w_begin[pivot_col as usize];
        cend = w_end[pivot_col as usize];
        rbeg = w_begin[(m + pivot_row) as usize];
        rend = w_end[(m + pivot_row) as usize];
        room = w_end[(2 * m) as usize] - w_begin[(2 * m) as usize];
        this.ngarbage += 1;
    }
    if grow > room {
        this.addmem_w = grow - room;
        return BASICLU_REALLOCATE;
    }

    // get pointer to U
    let mut u_put = u_begin[rank as usize];
    assert!(u_put >= 0);
    assert!(u_put < this.u_mem);

    // Column file update //

    // For each row i to be updated set marked[i] > 0 to its position
    // in the (packed) pivot column.
    let mut position = 1;
    for pos in (cbeg + 1)..cend {
        let i = w_index[pos as usize];
        marked[i as usize] = position;
        position += 1;
    }

    // let wi = Windex + cbeg;
    // let wx = Wvalue + cbeg;
    let mut col_number = 0; // mask cancelled[col_number]

    // for (rpos = rbeg+1; rpos < rend; rpos++, col_number++) {
    for rpos in (rbeg + 1)..rend {
        let j = w_index[rpos as usize];
        assert_ne!(j, pivot_col);
        let mut cmx = 0.0; // column maximum

        // Compress unmodified column entries. Store entries to be updated
        // in workspace. Move pivot row entry to the front of column.
        where_ = -1;
        let mut put = w_begin[j as usize];
        let pos1 = w_begin[j as usize];
        for pos in pos1..w_end[j as usize] {
            let i = w_index[pos as usize];
            // if ((position = marked[i]) > 0)
            position = marked[i as usize];
            if position > 0 {
                assert_ne!(i, pivot_row);
                work[position as usize] = w_value[pos as usize];
            } else {
                assert_eq!(position, 0);
                let x = w_value[pos as usize].abs();
                if i == pivot_row {
                    where_ = put;
                // } else if ((x = fabs(Wvalue[pos])) > cmx) {
                } else if x > cmx {
                    cmx = x;
                }
                w_index[put as usize] = w_index[pos as usize];
                w_value[put as usize] = w_value[pos as usize];
                put += 1;
            }
        }
        assert!(where_ >= 0);
        w_end[j as usize] = put;
        lu_iswap(w_index, pos1, where_);
        lu_fswap(w_value, pos1, where_);
        let xrj = w_value[pos1 as usize]; // pivot row entry

        // Reappend column if no room for update.
        room = w_begin[w_flink[j as usize] as usize] - put;
        if room < cnz1 {
            let nz = w_end[j as usize] - w_begin[j as usize];
            room = cnz1 + (stretch as LUInt) * (nz + cnz1) + pad;
            lu_file_reappend(
                j,
                2 * m,
                w_begin,
                w_end,
                w_flink,
                w_blink,
                w_index,
                w_value,
                room,
            );
            put = w_end[j as usize];
            assert_eq!(w_begin[w_flink[j as usize] as usize] - put, room);
            this.nexpand += 1;
        }

        // Compute update in workspace and append to column.
        let a = xrj / pivot;
        for pos in 1..=cnz1 {
            // work[pos as usize] -= a * wx[pos as usize];
            work[pos as usize] -= a * w_value[cbeg as usize..][pos as usize];
        }
        let mut mask = 0;
        for pos in 1..=cnz1 {
            let x = work[pos as usize].abs();
            if x > droptol {
                // Windex[put as usize] = wi[pos as usize];
                w_index[put as usize] = w_index[cbeg as usize..][pos as usize];
                w_value[put as usize] = work[pos as usize];
                put += 1;
                if x > cmx {
                    cmx = x;
                }
            } else {
                // cancellation in row wi[pos]
                // mask |= (int64_t) 1 << (pos-1);
                mask |= 1 << (pos - 1) as i64;
            }
            work[pos as usize] = 0.0;
        }
        w_end[j as usize] = put;
        cancelled[col_number] = mask as f64;

        // Write pivot row entry to U and remove from file.
        if xrj.abs() > droptol {
            assert!(u_put < this.u_mem);
            u_index[u_put as usize] = j;
            u_value[u_put as usize] = xrj;
            u_put += 1;
        }
        assert_eq!(w_index[w_begin[j as usize] as usize], pivot_row);
        w_begin[j as usize] += 1;

        // Move column to new list and update min_colnz.
        let nz = w_end[j as usize] - w_begin[j as usize];
        lu_list_move(
            j,
            nz,
            colcount_flink,
            colcount_blink,
            m,
            Some(&mut this.min_colnz),
        );

        colmax[j as usize] = cmx;

        col_number += 1;
    }
    for pos in (cbeg + 1)..cend {
        marked[w_index[pos as usize] as usize] = 0;
    }

    // Row file update //

    for rpos in rbeg..rend {
        marked[w_index[rpos as usize] as usize] = 1;
    }
    assert_eq!(marked[pivot_col as usize], 1);

    let mut mask = 1;
    // for (pos = cbeg+1; pos < cend; pos++, mask <<= 1)
    for pos in (cbeg + 1)..cend {
        assert_ne!(mask, 0);
        let i = w_index[pos as usize];
        assert_ne!(i, pivot_row);

        // Compress unmodified row entries (not marked). Remove
        // overlap with pivot row, including pivot column entry.
        let mut found = 0;
        let mut put = w_begin[(m + i) as usize];
        for rpos in w_begin[(m + i) as usize]..w_end[(m + i) as usize] {
            let j = w_index[rpos as usize];
            // if ((j = Windex[rpos]) == pivot_col)
            if j == pivot_col {
                found = 1;
            }
            if marked[j as usize] == 0 {
                w_index[put as usize] = j;
                put += 1;
            }
        }
        assert_ne!(found, 0);
        w_end[(m + i) as usize] = put;

        // Reappend row if no room for update. Append pattern of pivot row.
        room = w_begin[w_flink[(m + i) as usize] as usize] - put;
        if room < rnz1 {
            let nz = w_end[(m + i) as usize] - w_begin[(m + i) as usize];
            room = rnz1 + (stretch as LUInt) * (nz + rnz1) + pad;
            lu_file_reappend(
                m + i,
                2 * m,
                w_begin,
                w_end,
                w_flink,
                w_blink,
                w_index,
                w_value,
                room,
            );
            put = w_end[(m + i) as usize];
            assert_eq!(w_begin[w_flink[(m + i) as usize] as usize] - put, room);
            this.nexpand += 1;
        }

        col_number = 0;
        for rpos in (rbeg + 1)..rend {
            if (cancelled[col_number] as i64 & mask) == 0 {
                w_index[put as usize] = w_index[rpos as usize];
                put += 1;
            }
            col_number += 1;
        }
        w_end[(m + i) as usize] = put;

        // Move to new list. The row must be reinserted even if nz are
        // unchanged since it might have been taken out in Markowitz search.
        let nz = w_end[(m + i) as usize] - w_begin[(m + i) as usize];
        lu_list_move(
            i,
            nz,
            rowcount_flink,
            rowcount_blink,
            m,
            Some(&mut this.min_rownz),
        );

        mask <<= 1;
    }
    for rpos in rbeg..rend {
        marked[w_index[rpos as usize] as usize] = 0;
    }

    // Store column in L.
    let mut put = l_begin_p[rank as usize];
    for pos in (cbeg + 1)..cend {
        let x = w_value[pos as usize] / pivot;
        if x.abs() > droptol {
            l_index[put as usize] = w_index[pos as usize];
            l_value[put as usize] = x;
            put += 1;
        }
    }
    l_index[put as usize] = -1; // terminate column
    put += 1;
    l_begin_p[(rank + 1) as usize] = put;
    u_begin[(rank + 1) as usize] = u_put;

    // Cleanup:
    // store pivot elemnt;
    // remove pivot colum from column file, pivot row from row file;
    // remove pivot column/row from column/row counts
    colmax[pivot_col as usize] = pivot;
    w_end[pivot_col as usize] = cbeg;
    w_end[(m + pivot_row) as usize] = rbeg;
    lu_list_remove(colcount_flink, colcount_blink, pivot_col);
    lu_list_remove(rowcount_flink, rowcount_blink, pivot_row);

    // Check that row file and column file are consistent. Only use when
    // DEBUG_EXTRA since this check is really expensive.
    if cfg!(feature = "debug_extra") {
        // let (_, Wbegin_m) = Wbegin.split_at(m as usize);
        // let (_, Wend_m) = Wend.split_at(m as usize);
        assert_eq!(
            lu_file_diff(
                m,
                &w_begin[m as usize..],
                &w_end[m as usize..],
                w_begin,
                w_end,
                w_index,
                None
            ),
            0
        );
        assert_eq!(
            lu_file_diff(
                m,
                w_begin,
                w_end,
                &w_begin[m as usize..],
                &w_end[m as usize..],
                w_index,
                None
            ),
            0
        );
    }

    BASICLU_OK
}

fn lu_pivot_singleton_row(this: &mut LU) -> LUInt {
    let m = this.m;
    let rank = this.rank;
    let droptol = this.droptol;
    let pivot_col = this.pivot_col;
    let pivot_row = this.pivot_row;
    let colcount_flink = &mut this.colcount_flink;
    let colcount_blink = &mut this.colcount_blink;
    let rowcount_flink = &mut this.rowcount_flink;
    let rowcount_blink = &mut this.rowcount_blink;
    let colmax = &mut this.col_pivot;
    let l_begin_p = &mut this.l_begin_p;
    let u_begin = &mut this.u_begin;
    let w_begin = &mut this.w_begin;
    let w_end = &mut this.w_end;
    let l_index = &mut this.l_index;
    let l_value = &mut this.l_value;
    let w_index = &mut this.w_index;
    let w_value = &mut this.w_value;

    let cbeg = w_begin[pivot_col as usize];
    let cend = w_end[pivot_col as usize];
    let rbeg = w_begin[(m + pivot_row) as usize];
    let rend = w_end[(m + pivot_row) as usize];
    let rnz1 = rend - rbeg - 1; /* nz in pivot row except pivot */

    // lu_int i, pos, put, nz, where_;
    // double pivot, x;

    assert_eq!(rnz1, 0);

    // Find pivot.
    let mut where_ = cbeg;
    while w_index[where_ as usize] != pivot_row {
        assert!(where_ < cend - 1);
        where_ += 1;
    }
    let pivot = w_value[where_ as usize];
    assert_ne!(pivot, 0.0);

    // Store column in L.
    let mut put = l_begin_p[rank as usize];
    for pos in cbeg..cend {
        let x = w_value[pos as usize] / pivot;
        if pos != where_ && x.abs() > droptol {
            l_index[put as usize] = w_index[pos as usize];
            l_value[put as usize] = x;
            put += 1;
        }
    }
    l_index[put as usize] = -1; // terminate column
    put += 1;
    l_begin_p[(rank + 1) as usize] = put;
    u_begin[(rank + 1) as usize] = u_begin[rank as usize];

    // Remove pivot column from row file. Update row lists.
    for pos in cbeg..cend {
        let i = w_index[pos as usize];
        if i == pivot_row {
            continue;
        }
        where_ = w_begin[(m + i) as usize];
        while w_index[where_ as usize] != pivot_col {
            assert!(where_ < w_end[(m + i) as usize] - 1);
            where_ += 1;
        }
        // Windex[where_] = Windex[--Wend[m+i]];
        w_end[(m + i) as usize] -= 1;
        w_index[where_ as usize] = w_index[w_end[(m + i) as usize] as usize];
        let nz = w_end[(m + i) as usize] - w_begin[(m + i) as usize];
        lu_list_move(
            i,
            nz,
            rowcount_flink,
            rowcount_blink,
            m,
            Some(&mut this.min_rownz),
        );
    }

    // Cleanup:
    // store pivot elemnt;
    // remove pivot colum from column file, pivot row from row file;
    // remove pivot column/row from column/row counts
    colmax[pivot_col as usize] = pivot;
    w_end[pivot_col as usize] = cbeg;
    w_end[(m + pivot_row) as usize] = rbeg;
    lu_list_remove(colcount_flink, colcount_blink, pivot_col);
    lu_list_remove(rowcount_flink, rowcount_blink, pivot_row);

    BASICLU_OK
}

fn lu_pivot_singleton_col(this: &mut LU) -> LUInt {
    let m = this.m;
    let rank = this.rank;
    let droptol = this.droptol;
    let pivot_col = this.pivot_col;
    let pivot_row = this.pivot_row;
    let colcount_flink = &mut this.colcount_flink;
    let colcount_blink = &mut this.colcount_blink;
    let rowcount_flink = &mut this.rowcount_flink;
    let rowcount_blink = &mut this.rowcount_blink;
    let colmax = &mut this.col_pivot;
    let l_begin_p = &mut this.l_begin_p;
    let u_begin = &mut this.u_begin;
    let w_begin = &mut this.w_begin;
    let w_end = &mut this.w_end;
    let l_index = &mut this.l_index;
    let u_index = &mut this.u_index;
    let u_value = &mut this.u_value;
    let w_index = &mut this.w_index;
    let w_value = &mut this.w_value;

    let cbeg = w_begin[pivot_col as usize];
    let cend = w_end[pivot_col as usize];
    let rbeg = w_begin[(m + pivot_row) as usize];
    let rend = w_end[(m + pivot_row) as usize];
    let cnz1 = cend - cbeg - 1; /* nz in pivot column except pivot */

    // lu_int j, pos, rpos, put, nz, where_, found;
    // double pivot, cmx, x, xrj;

    assert_eq!(cnz1, 0);

    // Remove pivot row from column file and store in U. Update column lists.
    let mut put = u_begin[rank as usize];
    let pivot = w_value[cbeg as usize];
    assert_ne!(pivot, 0.0);
    let mut found = 0;
    let mut xrj = 0.0; // initialize to make gcc happy
    for rpos in rbeg..rend {
        let j = w_index[rpos as usize];
        if j == pivot_col {
            found = 1;
            continue;
        }
        let mut where_ = -1;
        let mut cmx = 0.0; // column maximum
        for pos in w_begin[j as usize]..w_end[j as usize] {
            let x = w_value[pos as usize].abs();
            if w_index[pos as usize] == pivot_row {
                where_ = pos;
                xrj = w_value[pos as usize];
            // } else if ((x = fabs(Wvalue[pos])) > cmx) {
            } else if x > cmx {
                cmx = x;
            }
        }
        assert!(where_ >= 0);
        if xrj.abs() > droptol {
            u_index[put as usize] = j;
            u_value[put as usize] = xrj;
            put += 1;
        }
        // Windex[where_] = Windex[--Wend [j]];
        w_end[j as usize] -= 1;
        w_index[where_ as usize] = w_index[w_end[j as usize] as usize];
        w_value[where_ as usize] = w_value[w_end[j as usize] as usize];
        let nz = w_end[j as usize] - w_begin[j as usize];
        lu_list_move(
            j,
            nz,
            colcount_flink,
            colcount_blink,
            m,
            Some(&mut this.min_colnz),
        );
        colmax[j as usize] = cmx;
    }
    assert_ne!(found, 0);
    u_begin[(rank + 1) as usize] = put;

    // Store empty column in L.
    put = l_begin_p[rank as usize];
    l_index[put as usize] = -1; // terminate column
    put += 1;
    l_begin_p[(rank + 1) as usize] = put;

    // Cleanup:
    // store pivot element;
    // remove pivot colum from column file, pivot row from row file;
    // remove pivot column/row from column/row counts
    colmax[pivot_col as usize] = pivot;
    w_end[pivot_col as usize] = cbeg;
    w_end[(m + pivot_row) as usize] = rbeg;
    lu_list_remove(colcount_flink, colcount_blink, pivot_col);
    lu_list_remove(rowcount_flink, rowcount_blink, pivot_row);

    BASICLU_OK
}

fn lu_pivot_doubleton_col(this: &mut LU) -> LUInt {
    let m = this.m;
    let rank = this.rank;
    let droptol = this.droptol;
    let pad = this.pad;
    let stretch = this.stretch;
    let pivot_col = this.pivot_col;
    let pivot_row = this.pivot_row;
    let colcount_flink = &mut this.colcount_flink;
    let colcount_blink = &mut this.colcount_blink;
    let rowcount_flink = &mut this.rowcount_flink;
    let rowcount_blink = &mut this.rowcount_blink;
    let colmax = &mut this.col_pivot;
    let l_begin_p = &mut this.l_begin_p;
    let u_begin = &mut this.u_begin;
    let w_begin = &mut this.w_begin;
    let w_end = &mut this.w_end;
    let w_flink = &mut this.w_flink;
    let w_blink = &mut this.w_blink;
    let l_index = &mut this.l_index;
    let l_value = &mut this.l_value;
    let u_index = &mut this.u_index;
    let u_value = &mut this.u_value;
    let w_index = &mut this.w_index;
    let w_value = &mut this.w_value;
    let marked = &mut this.iwork0;

    let mut cbeg = w_begin[pivot_col as usize]; // changed by file compression
    let mut cend = w_end[pivot_col as usize];
    let mut rbeg = w_begin[(m + pivot_row) as usize];
    let mut rend = w_end[(m + pivot_row) as usize];
    let cnz1 = cend - cbeg - 1; // nz in pivot column except pivot
    let rnz1 = rend - rbeg - 1; // nz in pivot row except pivot

    // lu_int j, pos, rpos, put, Uput, nz, nfill, where_, where_pivot, where_other;
    // lu_int other_row, grow, room, space, end, ncancelled;
    // double pivot, other_value, xrj, cmx, x, xabs;

    assert_eq!(cnz1, 1);

    /* Move pivot element to front of pivot column and pivot row. */
    if w_index[cbeg as usize] != pivot_row {
        lu_iswap(w_index, cbeg, cbeg + 1);
        lu_fswap(w_value, cbeg, cbeg + 1);
    }
    assert_eq!(w_index[cbeg as usize], pivot_row);
    let pivot = w_value[cbeg as usize];
    assert_ne!(pivot, 0.0);
    let other_row = w_index[(cbeg + 1) as usize];
    let other_value = w_value[(cbeg + 1) as usize];
    let mut where_ = rbeg;
    while w_index[where_ as usize] != pivot_col {
        assert!(where_ < rend - 1);
        where_ += 1;
    }
    lu_iswap(w_index, rbeg, where_);

    // Check if room is available in W.
    // Columns can be updated in place but the updated row may need to be
    // expanded.
    let mut nz = w_end[(m + other_row) as usize] - w_begin[(m + other_row) as usize];
    let grow = nz + rnz1 + (stretch as LUInt) * (nz + rnz1) + pad;
    let mut room = w_end[(2 * m) as usize] - w_begin[(2 * m) as usize];
    if grow > room {
        lu_file_compress(
            2 * m,
            w_begin,
            w_end,
            w_flink,
            w_index,
            w_value,
            stretch,
            pad,
        );
        cbeg = w_begin[pivot_col as usize];
        cend = w_end[pivot_col as usize];
        rbeg = w_begin[(m + pivot_row) as usize];
        rend = w_end[(m + pivot_row) as usize];
        room = w_end[(2 * m) as usize] - w_begin[(2 * m) as usize];
        this.ngarbage += 1;
    }
    if grow > room {
        this.addmem_w = grow - room;
        return BASICLU_REALLOCATE;
    }

    // Column file update //

    let mut u_put = u_begin[rank as usize];
    let mut put = rbeg + 1;
    let mut ncancelled = 0;
    for rpos in (rbeg + 1)..rend {
        let j = w_index[rpos as usize];
        assert_ne!(j, pivot_col);
        let mut cmx = 0.0; // column maximum

        // Find position of pivot row entry and possibly other row entry in
        // column j.
        let mut where_pivot = -1;
        let mut where_other = -1;
        let mut end = w_end[j as usize];
        for pos in w_begin[j as usize]..end {
            let x = w_value[pos as usize].abs();
            if w_index[pos as usize] == pivot_row {
                where_pivot = pos;
            } else if w_index[pos as usize] == other_row {
                where_other = pos;
            // } else if ((x = fabs(Wvalue[pos])) > cmx) {
            } else if x > cmx {
                cmx = x;
            }
        }
        assert!(where_pivot >= 0);
        let xrj = w_value[where_pivot as usize];

        // Store pivot row entry in U.
        if w_value[where_pivot as usize].abs() > droptol {
            u_index[u_put as usize] = j;
            u_value[u_put as usize] = w_value[where_pivot as usize];
            u_put += 1;
        }

        if where_other == -1 {
            // Compute fill-in element.
            let x = -xrj * (other_value / pivot);
            let xabs = x.abs();
            if xabs > droptol {
                // Store fill-in where pivot row entry was.
                w_index[where_pivot as usize] = other_row;
                w_value[where_pivot as usize] = x;
                w_index[put as usize] = j;
                put += 1;
                if xabs > cmx {
                    cmx = xabs;
                }
            } else {
                // Remove pivot row entry.
                // end = --Wend[j]; TODO: check
                w_end[j as usize] -= 1;
                end = w_end[j as usize];
                w_index[where_pivot as usize] = w_index[end as usize];
                w_value[where_pivot as usize] = w_value[end as usize];

                // Decrease column count.
                nz = end - w_begin[j as usize];
                lu_list_move(
                    j,
                    nz,
                    colcount_flink,
                    colcount_blink,
                    m,
                    Some(&mut this.min_colnz),
                );
            }
        } else {
            // Remove pivot row entry and update other row entry.
            // end = --Wend[j]; TODO: check
            w_end[j as usize] -= 1;
            end = w_end[j as usize];
            w_index[where_pivot as usize] = w_index[end as usize];
            w_value[where_pivot as usize] = w_value[end as usize];
            if where_other == end {
                where_other = where_pivot;
            }
            w_value[where_other as usize] -= xrj * (other_value / pivot);

            // If we have numerical cancellation, then remove the entry and mark
            // the column.
            let x = w_value[where_other as usize].abs();
            if x <= droptol {
                // end = --Wend[j]; TODO
                w_end[j as usize] -= 1;
                end = w_end[j as usize];
                w_index[where_other as usize] = w_index[end as usize];
                w_value[where_other as usize] = w_value[end as usize];
                marked[j as usize] = 1;
                ncancelled += 1;
            } else if x > cmx {
                cmx = x;
            }

            // Decrease column count.
            nz = w_end[j as usize] - w_begin[j as usize];
            lu_list_move(
                j,
                nz,
                colcount_flink,
                colcount_blink,
                m,
                Some(&mut this.min_colnz),
            );
        }
        colmax[j as usize] = cmx;
    }
    rend = put;
    u_begin[(rank + 1) as usize] = u_put;

    // Row file update //

    // If we have numerical cancellation, then we have to remove these entries
    // (marked) from the row pattern. In any case remove pivot column entry.
    if ncancelled != 0 {
        assert_eq!(marked[pivot_col as usize], 0);
        marked[pivot_col as usize] = 1; // treat as cancelled
        let mut put = w_begin[(m + other_row) as usize]; // compress remaining entries
        let end = w_end[(m + other_row) as usize];
        for pos in put..end {
            let j = w_index[pos as usize];
            if marked[j as usize] != 0 {
                marked[j as usize] = 0;
            } else {
                w_index[put as usize] = j;
                put += 1;
            }
        }
        assert_eq!(end - put, ncancelled + 1);
        w_end[(m + other_row) as usize] = put;
    } else {
        where_ = w_begin[(m + other_row) as usize];
        while w_index[where_ as usize] != pivot_col {
            assert!(where_ < w_end[(m + other_row) as usize] - 1);
            where_ += 1;
        }
        // end = --Wend[m+other_row]; TODO: check
        w_end[(m + other_row) as usize] -= 1;
        let end = w_end[(m + other_row) as usize];
        w_index[where_ as usize] = w_index[end as usize];
    }

    // Reappend row if no room for update.
    let nfill = rend - (rbeg + 1);
    let room =
        w_begin[w_flink[(m + other_row) as usize] as usize] - w_end[(m + other_row) as usize];
    if nfill > room {
        let nz = w_end[(m + other_row) as usize] - w_begin[(m + other_row) as usize];
        let space = nfill + (stretch as LUInt) * (nz + nfill) + pad;
        lu_file_reappend(
            m + other_row,
            2 * m,
            w_begin,
            w_end,
            w_flink,
            w_blink,
            w_index,
            w_value,
            space,
        );
        this.nexpand += 1;
    }

    // Append fill-in to row pattern.
    let mut put = w_end[(m + other_row) as usize];
    for pos in (rbeg + 1)..rend {
        w_index[put as usize] = w_index[pos as usize];
        put += 1;
    }
    w_end[(m + other_row) as usize] = put;

    // Reinsert other row into row counts.
    let nz = w_end[(m + other_row) as usize] - w_begin[(m + other_row) as usize];
    lu_list_move(
        other_row,
        nz,
        rowcount_flink,
        rowcount_blink,
        m,
        Some(&mut this.min_rownz),
    );

    // Store column in L.
    let mut put = l_begin_p[rank as usize];
    let x = other_value / pivot;
    if x.abs() > droptol {
        l_index[put as usize] = other_row;
        l_value[put as usize] = x;
        put += 1;
    }
    l_index[put as usize] = -1; // terminate column
    put += 1;
    l_begin_p[(rank + 1) as usize] = put;

    // Cleanup:
    // store pivot elemnt;
    // remove pivot colum from column file, pivot row from row file;
    // remove pivot column/row from column/row counts
    colmax[pivot_col as usize] = pivot;
    w_end[pivot_col as usize] = cbeg;
    w_end[(m + pivot_row) as usize] = rbeg;
    lu_list_remove(colcount_flink, colcount_blink, pivot_col);
    lu_list_remove(rowcount_flink, rowcount_blink, pivot_row);

    // Check that row file and column file are consistent. Only use when
    // DEBUG_EXTRA since this check is really expensive.
    if cfg!(feature = "debug_extra") {
        assert_eq!(
            lu_file_diff(
                m,
                &w_begin[m as usize..],
                &w_end[m as usize..],
                w_begin,
                w_end,
                w_index,
                None
            ),
            0
        );
        assert_eq!(
            lu_file_diff(
                m,
                w_begin,
                w_end,
                &w_begin[m as usize..],
                &w_end[m as usize..],
                w_index,
                None
            ),
            0
        );
    }

    BASICLU_OK
}

fn lu_remove_col(this: &mut LU, j: LUInt) {
    let m = this.m;
    let colcount_flink = &mut this.colcount_flink;
    let colcount_blink = &mut this.colcount_blink;
    let rowcount_flink = &mut this.rowcount_flink;
    let rowcount_blink = &mut this.rowcount_blink;
    let colmax = &mut this.col_pivot;
    let w_begin = &mut this.w_begin;
    let w_end = &mut this.w_end;
    let w_index = &mut this.w_index;
    let cbeg = w_begin[j as usize];
    let cend = w_end[j as usize];

    // lu_int i, pos, nz, where_;

    // Remove column j from row file.
    for pos in cbeg..cend {
        let i = w_index[pos as usize];
        let mut where_ = w_begin[(m + i) as usize];
        while w_index[where_ as usize] != j {
            assert!(where_ < w_end[(m + i) as usize] - 1);
            where_ += 1;
        }
        // Windex[where_] = Windex[--Wend[m+i]];
        w_end[(m + i) as usize] -= 1;
        w_index[where_ as usize] = w_index[w_end[(m + i) as usize] as usize];
        let nz = w_end[(m + i) as usize] - w_begin[(m + i) as usize];
        lu_list_move(
            i,
            nz,
            rowcount_flink,
            rowcount_blink,
            m,
            Some(&mut this.min_rownz),
        );
    }

    // Remove column j from column file.
    colmax[j as usize] = 0.0;
    w_end[j as usize] = cbeg;
    lu_list_move(
        j,
        0,
        colcount_flink,
        colcount_blink,
        m,
        Some(&mut this.min_colnz),
    );
}
