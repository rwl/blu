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
use crate::lu_internal::lu;
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
const MAXROW_SMALL: lu_int = 64;

pub(crate) fn lu_pivot(this: &mut lu) -> lu_int {
    let m = this.m;
    let rank = this.rank;
    let Lmem = this.Lmem;
    let Umem = this.Umem;
    let pivot_col = this.pivot_col;
    let pivot_row = this.pivot_row;
    // let colmax = &this.col_pivot;
    let Lbegin_p = &this.Lbegin_p;
    // let Ubegin = &this.Ubegin;
    let Wbegin = this.Wbegin.as_ref().unwrap();
    let Wend = this.Wend.as_ref().unwrap();
    // let Uindex = this.Uindex.as_ref().unwrap();
    let nz_col = Wend[pivot_col as usize] - Wbegin[pivot_col as usize];
    let nz_row = Wend[(m + pivot_row) as usize] - Wbegin[(m + pivot_row) as usize];

    let mut status = BASICLU_OK;
    let tic = Instant::now();

    assert!(nz_row >= 1);
    assert!(nz_col >= 1);

    // Check if room is available in L and U.
    let room = Lmem - Lbegin_p[rank as usize];
    let need = nz_col; // # off-diagonals in pivot col + end marker (-1)
    if room < need {
        this.addmemL = need - room;
        status = BASICLU_REALLOCATE;
    }
    let room = Umem - this.Ubegin[rank as usize];
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
        for pos in this.Ubegin[rank as usize]..this.Ubegin[(rank + 1) as usize] {
            let j = this.Uindex.as_ref().unwrap()[pos as usize];
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

fn lu_pivot_any(this: &mut lu) -> lu_int {
    let m = this.m;
    let rank = this.rank;
    let droptol = this.droptol;
    let pad = this.pad;
    let stretch = this.stretch;
    let pivot_col = this.pivot_col;
    let pivot_row = this.pivot_row;
    let colcount_flink = this.colcount_flink.as_mut().unwrap();
    let colcount_blink = this.colcount_blink.as_mut().unwrap();
    let rowcount_flink = this.rowcount_flink.as_mut().unwrap();
    let rowcount_blink = this.rowcount_blink.as_mut().unwrap();
    let colmax = &mut this.col_pivot;
    let Lbegin_p = &mut this.Lbegin_p;
    let Ubegin = &mut this.Ubegin;
    let Wbegin = this.Wbegin.as_mut().unwrap();
    let Wend = this.Wend.as_mut().unwrap();
    let Wflink = this.Wflink.as_mut().unwrap();
    let Wblink = this.Wblink.as_mut().unwrap();
    let Lindex = this.Lindex.as_mut().unwrap();
    let Lvalue = this.Lvalue.as_mut().unwrap();
    let Uindex = this.Uindex.as_mut().unwrap();
    let Uvalue = this.Uvalue.as_mut().unwrap();
    let Windex = this.Windex.as_mut().unwrap();
    let Wvalue = this.Wvalue.as_mut().unwrap();
    let marked = this.iwork0.as_mut().unwrap();
    let work = &mut this.work0;

    let mut cbeg = Wbegin[pivot_col as usize]; // changed by file compression
    let mut cend = Wend[pivot_col as usize];
    let mut rbeg = Wbegin[(m + pivot_row) as usize];
    let mut rend = Wend[(m + pivot_row) as usize];
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
        let i = Windex[pos as usize];
        if i == pivot_row {
            where_ = pos;
        } else {
            let nz = Wend[(m + i) as usize] - Wbegin[(m + i) as usize];
            grow += nz + rnz1 + (stretch as lu_int) * (nz + rnz1) + pad;
        }
    }
    assert!(where_ >= 0);
    lu_iswap(Windex, cbeg, where_);
    lu_fswap(Wvalue, cbeg, where_);
    let pivot = Wvalue[cbeg as usize];
    assert_ne!(pivot, 0.0);
    where_ = -1;
    for rpos in rbeg..rend {
        // if ((j = Windex[rpos]) == pivot_col) TODO: check
        let j = Windex[rpos as usize];
        if j == pivot_col {
            where_ = rpos;
        } else {
            let nz = Wend[j as usize] - Wbegin[j as usize];
            grow += nz + cnz1 + (stretch as lu_int) * (nz + cnz1) + pad;
        }
    }
    assert!(where_ >= 0);
    lu_iswap(Windex, rbeg, where_);
    let mut room = Wend[(2 * m) as usize] - Wbegin[(2 * m) as usize];
    if grow > room {
        lu_file_compress(2 * m, Wbegin, Wend, Wflink, Windex, Wvalue, stretch, pad);
        cbeg = Wbegin[pivot_col as usize];
        cend = Wend[pivot_col as usize];
        rbeg = Wbegin[(m + pivot_row) as usize];
        rend = Wend[(m + pivot_row) as usize];
        room = Wend[(2 * m) as usize] - Wbegin[(2 * m) as usize];
        this.ngarbage += 1;
    }
    if grow > room {
        this.addmemW = grow - room;
        return BASICLU_REALLOCATE;
    }

    // get pointer to U
    let mut Uput = Ubegin[rank as usize];
    assert!(Uput >= 0);
    assert!(Uput < this.Umem);

    // Column file update //

    // For each row i to be updated set marked[i] > 0 to its position
    // in the (packed) pivot column.
    let mut position = 1;
    for pos in (cbeg + 1)..cend {
        let i = Windex[pos as usize];
        marked[i as usize] = position;
        position += 1;
    }

    // let wi = Windex + cbeg;
    // let wx = Wvalue + cbeg;
    for rpos in (rbeg + 1)..rend {
        let j = Windex[rpos as usize];
        assert_ne!(j, pivot_col);
        let mut cmx = 0.0; // column maximum

        // Compress unmodified column entries. Store entries to be updated
        // in workspace. Move pivot row entry to the front of column.
        where_ = -1;
        let mut put = Wbegin[j as usize];
        let pos1 = Wbegin[j as usize];
        for pos in pos1..Wend[j as usize] {
            let i = Windex[pos as usize];
            // if ((position = marked[i]) > 0) {
            position = marked[i as usize];
            if position > 0 {
                assert_ne!(i, pivot_row);
                work[position as usize] = Wvalue[pos as usize];
            } else {
                assert_eq!(position, 0);
                let x = Wvalue[pos as usize].abs();
                if i == pivot_row {
                    where_ = put;
                // } else if ((x = fabs(Wvalue[pos])) > cmx) {
                } else if x > cmx {
                    cmx = x;
                }
                Windex[put as usize] = Windex[pos as usize];
                Wvalue[put as usize] = Wvalue[pos as usize];
                put += 1;
            }
        }
        assert!(where_ >= 0);
        Wend[j as usize] = put;
        lu_iswap(Windex, pos1, where_);
        lu_fswap(Wvalue, pos1, where_);
        let xrj = Wvalue[pos1 as usize]; // pivot row entry

        // Reappend column if no room for update.
        room = Wbegin[Wflink[j as usize] as usize] - put;
        if room < cnz1 {
            let nz = Wend[j as usize] - Wbegin[j as usize];
            room = cnz1 + (stretch as lu_int) * (nz + cnz1) + pad;
            lu_file_reappend(j, 2 * m, Wbegin, Wend, Wflink, Wblink, Windex, Wvalue, room);
            put = Wend[j as usize];
            assert_eq!(Wbegin[Wflink[j as usize] as usize] - put, room);
            this.nexpand += 1;
        }

        // Compute update in workspace and append to column.
        let a = xrj / pivot;
        for pos in 1..=cnz1 {
            // work[pos as usize] -= a * wx[pos as usize];
            work[pos as usize] -= a * Wvalue[cbeg as usize..][pos as usize];
        }
        for pos in 1..=cnz1 {
            // Windex[put as usize] = wi[pos as usize];
            Windex[put as usize] = Windex[cbeg as usize..][pos as usize];
            Wvalue[put as usize] = work[pos as usize];
            put += 1;
            // if ((x = fabs(work[pos])) > cmx) {
            let x = work[pos as usize].abs();
            if x > cmx {
                cmx = x;
            }
            work[pos as usize] = 0.0;
        }
        Wend[j as usize] = put;

        // Write pivot row entry to U and remove from file.
        if xrj.abs() > droptol {
            assert!(Uput < this.Umem);
            Uindex[Uput as usize] = j;
            Uvalue[Uput as usize] = xrj;
            Uput += 1;
        }
        assert_eq!(Windex[Wbegin[j as usize] as usize], pivot_row);
        Wbegin[j as usize] += 1;

        // Move column to new list and update min_colnz.
        let nz = Wend[j as usize] - Wbegin[j as usize];
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
        marked[Windex[pos as usize] as usize] = 0;
    }

    // Row file update //

    for rpos in rbeg..rend {
        marked[Windex[rpos as usize] as usize] = 1;
    }
    assert_eq!(marked[pivot_col as usize], 1);

    for pos in (cbeg + 1)..cend {
        let i = Windex[pos as usize];
        assert_ne!(i, pivot_row);

        // Compress unmodified row entries (not marked). Remove
        // overlap with pivot row, including pivot column entry.
        let mut found = 0;
        let mut put = Wbegin[(m + i) as usize];
        for rpos in Wbegin[(m + i) as usize]..Wend[(m + i) as usize] {
            let j = Windex[rpos as usize];
            if j == pivot_col {
                found = 1;
            }
            if marked[j as usize] == 0 {
                Windex[put as usize] = j;
                put += 1;
            }
        }
        assert_ne!(found, 0);
        Wend[(m + i) as usize] = put;

        // Reappend row if no room for update. Append pattern of pivot row.
        room = Wbegin[Wflink[(m + i) as usize] as usize] - put;
        if room < rnz1 {
            let nz = Wend[(m + i) as usize] - Wbegin[(m + i) as usize];
            room = rnz1 + (stretch as lu_int) * (nz + rnz1) + pad;
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
            put = Wend[(m + i) as usize];
            assert_eq!(Wbegin[Wflink[(m + i) as usize] as usize] - put, room);
            this.nexpand += 1;
        }
        for rpos in (rbeg + 1)..rend {
            Windex[put as usize] = Windex[rpos as usize];
            put += 1;
        }
        Wend[(m + i) as usize] = put;

        // Move to new list. The row must be reinserted even if nz are
        // unchanged since it might have been taken out in Markowitz search.
        let nz = Wend[(m + i) as usize] - Wbegin[(m + i) as usize];
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
        marked[Windex[rpos as usize] as usize] = 0;
    }

    // Store column in L.
    let mut put = Lbegin_p[rank as usize];
    for pos in (cbeg + 1)..cend {
        let x = Wvalue[pos as usize] / pivot;
        if x.abs() > droptol {
            Lindex[put as usize] = Windex[pos as usize];
            Lvalue[put as usize] = x;
            put += 1;
        }
    }
    Lindex[put as usize] = -1; // terminate column
    put += 1;
    Lbegin_p[(rank + 1) as usize] = put;
    Ubegin[(rank + 1) as usize] = Uput;

    // Cleanup:
    // store pivot element;
    // remove pivot colum from column file, pivot row from row file;
    // remove pivot column/row from column/row counts
    colmax[pivot_col as usize] = pivot;
    Wend[pivot_col as usize] = cbeg;
    Wend[(m + pivot_row) as usize] = rbeg;
    lu_list_remove(colcount_flink, colcount_blink, pivot_col);
    lu_list_remove(rowcount_flink, rowcount_blink, pivot_row);

    // Check that row file and column file are consistent. Only use when
    // DEBUG_EXTRA since this check is really expensive.
    if cfg!(feature = "debug_extra") {
        assert_eq!(
            lu_file_diff(
                m,
                &Wbegin[m as usize..],
                &Wend[m as usize..],
                Wbegin,
                Wend,
                Windex,
                None
            ),
            0
        );
        assert_eq!(
            lu_file_diff(
                m,
                Wbegin,
                Wend,
                &Wbegin[m as usize..],
                &Wend[m as usize..],
                Windex,
                None
            ),
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
    let colcount_flink = this.colcount_flink.as_mut().unwrap();
    let colcount_blink = this.colcount_blink.as_mut().unwrap();
    let rowcount_flink = this.rowcount_flink.as_mut().unwrap();
    let rowcount_blink = this.rowcount_blink.as_mut().unwrap();
    let colmax = &mut this.col_pivot;
    let Lbegin_p = &mut this.Lbegin_p;
    let Ubegin = &mut this.Ubegin;
    let Wbegin = this.Wbegin.as_mut().unwrap();
    let Wend = this.Wend.as_mut().unwrap();
    let Wflink = this.Wflink.as_mut().unwrap();
    let Wblink = this.Wblink.as_mut().unwrap();
    let Lindex = this.Lindex.as_mut().unwrap();
    let Lvalue = this.Lvalue.as_mut().unwrap();
    let Uindex = this.Uindex.as_mut().unwrap();
    let Uvalue = this.Uvalue.as_mut().unwrap();
    let Windex = this.Windex.as_mut().unwrap();
    let Wvalue = this.Wvalue.as_mut().unwrap();
    let marked = this.iwork0.as_mut().unwrap();
    let work = &mut this.work0;
    // int64_t *cancelled      = (void *) this.row_pivot;
    let cancelled = &mut this.row_pivot;

    let mut cbeg = Wbegin[pivot_col as usize]; // changed by file compression
    let mut cend = Wend[pivot_col as usize];
    let mut rbeg = Wbegin[(m + pivot_row) as usize];
    let mut rend = Wend[(m + pivot_row) as usize];
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
        let i = Windex[pos as usize];
        if i == pivot_row {
            where_ = pos;
        } else {
            let nz = Wend[(m + i) as usize] - Wbegin[(m + i) as usize];
            grow += nz + rnz1 + (stretch as lu_int) * (nz + rnz1) + pad;
        }
    }
    assert!(where_ >= 0);
    lu_iswap(Windex, cbeg, where_);
    lu_fswap(Wvalue, cbeg, where_);
    let pivot = Wvalue[cbeg as usize];
    assert_ne!(pivot, 0.0);
    where_ = -1;
    for rpos in rbeg..rend {
        let j = Windex[rpos as usize];
        // if ((j = Windex[rpos]) == pivot_col)
        if j == pivot_col {
            where_ = rpos;
        } else {
            let nz = Wend[j as usize] - Wbegin[j as usize];
            grow += nz + cnz1 + (stretch as lu_int) * (nz + cnz1) + pad;
        }
    }
    assert!(where_ >= 0);
    lu_iswap(Windex, rbeg, where_);
    let mut room = Wend[(2 * m) as usize] - Wbegin[(2 * m) as usize];
    if grow > room {
        lu_file_compress(2 * m, Wbegin, Wend, Wflink, Windex, Wvalue, stretch, pad);
        cbeg = Wbegin[pivot_col as usize];
        cend = Wend[pivot_col as usize];
        rbeg = Wbegin[(m + pivot_row) as usize];
        rend = Wend[(m + pivot_row) as usize];
        room = Wend[(2 * m) as usize] - Wbegin[(2 * m) as usize];
        this.ngarbage += 1;
    }
    if grow > room {
        this.addmemW = grow - room;
        return BASICLU_REALLOCATE;
    }

    // get pointer to U
    let mut Uput = Ubegin[rank as usize];
    assert!(Uput >= 0);
    assert!(Uput < this.Umem);

    // Column file update //

    // For each row i to be updated set marked[i] > 0 to its position
    // in the (packed) pivot column.
    let mut position = 1;
    for pos in (cbeg + 1)..cend {
        let i = Windex[pos as usize];
        marked[i as usize] = position;
        position += 1;
    }

    // let wi = Windex + cbeg;
    // let wx = Wvalue + cbeg;
    let mut col_number = 0; // mask cancelled[col_number]

    // for (rpos = rbeg+1; rpos < rend; rpos++, col_number++) {
    for rpos in (rbeg + 1)..rend {
        let j = Windex[rpos as usize];
        assert_ne!(j, pivot_col);
        let mut cmx = 0.0; // column maximum

        // Compress unmodified column entries. Store entries to be updated
        // in workspace. Move pivot row entry to the front of column.
        where_ = -1;
        let mut put = Wbegin[j as usize];
        let pos1 = Wbegin[j as usize];
        for pos in pos1..Wend[j as usize] {
            let i = Windex[pos as usize];
            // if ((position = marked[i]) > 0)
            position = marked[i as usize];
            if position > 0 {
                assert_ne!(i, pivot_row);
                work[position as usize] = Wvalue[pos as usize];
            } else {
                assert_eq!(position, 0);
                let x = Wvalue[pos as usize].abs();
                if i == pivot_row {
                    where_ = put;
                // } else if ((x = fabs(Wvalue[pos])) > cmx) {
                } else if x > cmx {
                    cmx = x;
                }
                Windex[put as usize] = Windex[pos as usize];
                Wvalue[put as usize] = Wvalue[pos as usize];
                put += 1;
            }
        }
        assert!(where_ >= 0);
        Wend[j as usize] = put;
        lu_iswap(Windex, pos1, where_);
        lu_fswap(Wvalue, pos1, where_);
        let xrj = Wvalue[pos1 as usize]; // pivot row entry

        // Reappend column if no room for update.
        room = Wbegin[Wflink[j as usize] as usize] - put;
        if room < cnz1 {
            let nz = Wend[j as usize] - Wbegin[j as usize];
            room = cnz1 + (stretch as lu_int) * (nz + cnz1) + pad;
            lu_file_reappend(j, 2 * m, Wbegin, Wend, Wflink, Wblink, Windex, Wvalue, room);
            put = Wend[j as usize];
            assert_eq!(Wbegin[Wflink[j as usize] as usize] - put, room);
            this.nexpand += 1;
        }

        // Compute update in workspace and append to column.
        let a = xrj / pivot;
        for pos in 1..=cnz1 {
            // work[pos as usize] -= a * wx[pos as usize];
            work[pos as usize] -= a * Wvalue[cbeg as usize..][pos as usize];
        }
        let mut mask = 0;
        for pos in 1..=cnz1 {
            let x = work[pos as usize].abs();
            if x > droptol {
                // Windex[put as usize] = wi[pos as usize];
                Windex[put as usize] = Windex[cbeg as usize..][pos as usize];
                Wvalue[put as usize] = work[pos as usize];
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
        Wend[j as usize] = put;
        cancelled[col_number] = mask as f64;

        // Write pivot row entry to U and remove from file.
        if xrj.abs() > droptol {
            assert!(Uput < this.Umem);
            Uindex[Uput as usize] = j;
            Uvalue[Uput as usize] = xrj;
            Uput += 1;
        }
        assert_eq!(Windex[Wbegin[j as usize] as usize], pivot_row);
        Wbegin[j as usize] += 1;

        // Move column to new list and update min_colnz.
        let nz = Wend[j as usize] - Wbegin[j as usize];
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
        marked[Windex[pos as usize] as usize] = 0;
    }

    // Row file update //

    for rpos in rbeg..rend {
        marked[Windex[rpos as usize] as usize] = 1;
    }
    assert_eq!(marked[pivot_col as usize], 1);

    let mut mask = 1;
    // for (pos = cbeg+1; pos < cend; pos++, mask <<= 1)
    for pos in (cbeg + 1)..cend {
        assert_ne!(mask, 0);
        let i = Windex[pos as usize];
        assert_ne!(i, pivot_row);

        // Compress unmodified row entries (not marked). Remove
        // overlap with pivot row, including pivot column entry.
        let mut found = 0;
        let mut put = Wbegin[(m + i) as usize];
        for rpos in Wbegin[(m + i) as usize]..Wend[(m + i) as usize] {
            let j = Windex[rpos as usize];
            // if ((j = Windex[rpos]) == pivot_col)
            if j == pivot_col {
                found = 1;
            }
            if marked[j as usize] == 0 {
                Windex[put as usize] = j;
                put += 1;
            }
        }
        assert_ne!(found, 0);
        Wend[(m + i) as usize] = put;

        // Reappend row if no room for update. Append pattern of pivot row.
        room = Wbegin[Wflink[(m + i) as usize] as usize] - put;
        if room < rnz1 {
            let nz = Wend[(m + i) as usize] - Wbegin[(m + i) as usize];
            room = rnz1 + (stretch as lu_int) * (nz + rnz1) + pad;
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
            put = Wend[(m + i) as usize];
            assert_eq!(Wbegin[Wflink[(m + i) as usize] as usize] - put, room);
            this.nexpand += 1;
        }

        col_number = 0;
        for rpos in (rbeg + 1)..rend {
            if (cancelled[col_number] as i64 & mask) == 0 {
                Windex[put as usize] = Windex[rpos as usize];
                put += 1;
            }
            col_number += 1;
        }
        Wend[(m + i) as usize] = put;

        // Move to new list. The row must be reinserted even if nz are
        // unchanged since it might have been taken out in Markowitz search.
        let nz = Wend[(m + i) as usize] - Wbegin[(m + i) as usize];
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
        marked[Windex[rpos as usize] as usize] = 0;
    }

    // Store column in L.
    let mut put = Lbegin_p[rank as usize];
    for pos in (cbeg + 1)..cend {
        let x = Wvalue[pos as usize] / pivot;
        if x.abs() > droptol {
            Lindex[put as usize] = Windex[pos as usize];
            Lvalue[put as usize] = x;
            put += 1;
        }
    }
    Lindex[put as usize] = -1; // terminate column
    put += 1;
    Lbegin_p[(rank + 1) as usize] = put;
    Ubegin[(rank + 1) as usize] = Uput;

    // Cleanup:
    // store pivot elemnt;
    // remove pivot colum from column file, pivot row from row file;
    // remove pivot column/row from column/row counts
    colmax[pivot_col as usize] = pivot;
    Wend[pivot_col as usize] = cbeg;
    Wend[(m + pivot_row) as usize] = rbeg;
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
                &Wbegin[m as usize..],
                &Wend[m as usize..],
                Wbegin,
                Wend,
                Windex,
                None
            ),
            0
        );
        assert_eq!(
            lu_file_diff(
                m,
                Wbegin,
                Wend,
                &Wbegin[m as usize..],
                &Wend[m as usize..],
                Windex,
                None
            ),
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
    let colcount_flink = this.colcount_flink.as_mut().unwrap();
    let colcount_blink = this.colcount_blink.as_mut().unwrap();
    let rowcount_flink = this.rowcount_flink.as_mut().unwrap();
    let rowcount_blink = this.rowcount_blink.as_mut().unwrap();
    let colmax = &mut this.col_pivot;
    let Lbegin_p = &mut this.Lbegin_p;
    let Ubegin = &mut this.Ubegin;
    let Wbegin = this.Wbegin.as_mut().unwrap();
    let Wend = this.Wend.as_mut().unwrap();
    let Lindex = this.Lindex.as_mut().unwrap();
    let Lvalue = this.Lvalue.as_mut().unwrap();
    let Windex = this.Windex.as_mut().unwrap();
    let Wvalue = this.Wvalue.as_mut().unwrap();

    let cbeg = Wbegin[pivot_col as usize];
    let cend = Wend[pivot_col as usize];
    let rbeg = Wbegin[(m + pivot_row) as usize];
    let rend = Wend[(m + pivot_row) as usize];
    let rnz1 = rend - rbeg - 1; /* nz in pivot row except pivot */

    // lu_int i, pos, put, nz, where_;
    // double pivot, x;

    assert_eq!(rnz1, 0);

    // Find pivot.
    let mut where_ = cbeg;
    while Windex[where_ as usize] != pivot_row {
        assert!(where_ < cend - 1);
        where_ += 1;
    }
    let pivot = Wvalue[where_ as usize];
    assert_ne!(pivot, 0.0);

    // Store column in L.
    let mut put = Lbegin_p[rank as usize];
    for pos in cbeg..cend {
        let x = Wvalue[pos as usize] / pivot;
        if pos != where_ && x.abs() > droptol {
            Lindex[put as usize] = Windex[pos as usize];
            Lvalue[put as usize] = x;
            put += 1;
        }
    }
    Lindex[put as usize] = -1; // terminate column
    put += 1;
    Lbegin_p[(rank + 1) as usize] = put;
    Ubegin[(rank + 1) as usize] = Ubegin[rank as usize];

    // Remove pivot column from row file. Update row lists.
    for pos in cbeg..cend {
        let i = Windex[pos as usize];
        if i == pivot_row {
            continue;
        }
        where_ = Wbegin[(m + i) as usize];
        while Windex[where_ as usize] != pivot_col {
            assert!(where_ < Wend[(m + i) as usize] - 1);
            where_ += 1;
        }
        // Windex[where_] = Windex[--Wend[m+i]];
        Wend[(m + i) as usize] -= 1;
        Windex[where_ as usize] = Windex[Wend[(m + i) as usize] as usize];
        let nz = Wend[(m + i) as usize] - Wbegin[(m + i) as usize];
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
    Wend[pivot_col as usize] = cbeg;
    Wend[(m + pivot_row) as usize] = rbeg;
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
    let colcount_flink = this.colcount_flink.as_mut().unwrap();
    let colcount_blink = this.colcount_blink.as_mut().unwrap();
    let rowcount_flink = this.rowcount_flink.as_mut().unwrap();
    let rowcount_blink = this.rowcount_blink.as_mut().unwrap();
    let colmax = &mut this.col_pivot;
    let Lbegin_p = &mut this.Lbegin_p;
    let Ubegin = &mut this.Ubegin;
    let Wbegin = this.Wbegin.as_mut().unwrap();
    let Wend = this.Wend.as_mut().unwrap();
    let Lindex = this.Lindex.as_mut().unwrap();
    let Uindex = this.Uindex.as_mut().unwrap();
    let Uvalue = this.Uvalue.as_mut().unwrap();
    let Windex = this.Windex.as_mut().unwrap();
    let Wvalue = this.Wvalue.as_mut().unwrap();

    let cbeg = Wbegin[pivot_col as usize];
    let cend = Wend[pivot_col as usize];
    let rbeg = Wbegin[(m + pivot_row) as usize];
    let rend = Wend[(m + pivot_row) as usize];
    let cnz1 = cend - cbeg - 1; /* nz in pivot column except pivot */

    // lu_int j, pos, rpos, put, nz, where_, found;
    // double pivot, cmx, x, xrj;

    assert_eq!(cnz1, 0);

    // Remove pivot row from column file and store in U. Update column lists.
    let mut put = Ubegin[rank as usize];
    let pivot = Wvalue[cbeg as usize];
    assert_ne!(pivot, 0.0);
    let mut found = 0;
    let mut xrj = 0.0; // initialize to make gcc happy
    for rpos in rbeg..rend {
        let j = Windex[rpos as usize];
        if j == pivot_col {
            found = 1;
            continue;
        }
        let mut where_ = -1;
        let mut cmx = 0.0; // column maximum
        for pos in Wbegin[j as usize]..Wend[j as usize] {
            let x = Wvalue[pos as usize].abs();
            if Windex[pos as usize] == pivot_row {
                where_ = pos;
                xrj = Wvalue[pos as usize];
            // } else if ((x = fabs(Wvalue[pos])) > cmx) {
            } else if x > cmx {
                cmx = x;
            }
        }
        assert!(where_ >= 0);
        if xrj.abs() > droptol {
            Uindex[put as usize] = j;
            Uvalue[put as usize] = xrj;
            put += 1;
        }
        // Windex[where_] = Windex[--Wend [j]];
        Wend[j as usize] -= 1;
        Windex[where_ as usize] = Windex[Wend[j as usize] as usize];
        Wvalue[where_ as usize] = Wvalue[Wend[j as usize] as usize];
        let nz = Wend[j as usize] - Wbegin[j as usize];
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
    Ubegin[(rank + 1) as usize] = put;

    // Store empty column in L.
    put = Lbegin_p[rank as usize];
    Lindex[put as usize] = -1; // terminate column
    put += 1;
    Lbegin_p[(rank + 1) as usize] = put;

    // Cleanup:
    // store pivot element;
    // remove pivot colum from column file, pivot row from row file;
    // remove pivot column/row from column/row counts
    colmax[pivot_col as usize] = pivot;
    Wend[pivot_col as usize] = cbeg;
    Wend[(m + pivot_row) as usize] = rbeg;
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
    let colcount_flink = this.colcount_flink.as_mut().unwrap();
    let colcount_blink = this.colcount_blink.as_mut().unwrap();
    let rowcount_flink = this.rowcount_flink.as_mut().unwrap();
    let rowcount_blink = this.rowcount_blink.as_mut().unwrap();
    let colmax = &mut this.col_pivot;
    let Lbegin_p = &mut this.Lbegin_p;
    let Ubegin = &mut this.Ubegin;
    let Wbegin = this.Wbegin.as_mut().unwrap();
    let Wend = this.Wend.as_mut().unwrap();
    let Wflink = this.Wflink.as_mut().unwrap();
    let Wblink = this.Wblink.as_mut().unwrap();
    let Lindex = this.Lindex.as_mut().unwrap();
    let Lvalue = this.Lvalue.as_mut().unwrap();
    let Uindex = this.Uindex.as_mut().unwrap();
    let Uvalue = this.Uvalue.as_mut().unwrap();
    let Windex = this.Windex.as_mut().unwrap();
    let Wvalue = this.Wvalue.as_mut().unwrap();
    let marked = this.iwork0.as_mut().unwrap();

    let mut cbeg = Wbegin[pivot_col as usize]; // changed by file compression
    let mut cend = Wend[pivot_col as usize];
    let mut rbeg = Wbegin[(m + pivot_row) as usize];
    let mut rend = Wend[(m + pivot_row) as usize];
    let cnz1 = cend - cbeg - 1; // nz in pivot column except pivot
    let rnz1 = rend - rbeg - 1; // nz in pivot row except pivot

    // lu_int j, pos, rpos, put, Uput, nz, nfill, where_, where_pivot, where_other;
    // lu_int other_row, grow, room, space, end, ncancelled;
    // double pivot, other_value, xrj, cmx, x, xabs;

    assert_eq!(cnz1, 1);

    /* Move pivot element to front of pivot column and pivot row. */
    if Windex[cbeg as usize] != pivot_row {
        lu_iswap(Windex, cbeg, cbeg + 1);
        lu_fswap(Wvalue, cbeg, cbeg + 1);
    }
    assert_eq!(Windex[cbeg as usize], pivot_row);
    let pivot = Wvalue[cbeg as usize];
    assert_ne!(pivot, 0.0);
    let other_row = Windex[(cbeg + 1) as usize];
    let other_value = Wvalue[(cbeg + 1) as usize];
    let mut where_ = rbeg;
    while Windex[where_ as usize] != pivot_col {
        assert!(where_ < rend - 1);
        where_ += 1;
    }
    lu_iswap(Windex, rbeg, where_);

    // Check if room is available in W.
    // Columns can be updated in place but the updated row may need to be
    // expanded.
    let mut nz = Wend[(m + other_row) as usize] - Wbegin[(m + other_row) as usize];
    let grow = nz + rnz1 + (stretch as lu_int) * (nz + rnz1) + pad;
    let mut room = Wend[(2 * m) as usize] - Wbegin[(2 * m) as usize];
    if grow > room {
        lu_file_compress(2 * m, Wbegin, Wend, Wflink, Windex, Wvalue, stretch, pad);
        cbeg = Wbegin[pivot_col as usize];
        cend = Wend[pivot_col as usize];
        rbeg = Wbegin[(m + pivot_row) as usize];
        rend = Wend[(m + pivot_row) as usize];
        room = Wend[(2 * m) as usize] - Wbegin[(2 * m) as usize];
        this.ngarbage += 1;
    }
    if grow > room {
        this.addmemW = grow - room;
        return BASICLU_REALLOCATE;
    }

    // Column file update //

    let mut Uput = Ubegin[rank as usize];
    let mut put = rbeg + 1;
    let mut ncancelled = 0;
    for rpos in (rbeg + 1)..rend {
        let j = Windex[rpos as usize];
        assert_ne!(j, pivot_col);
        let mut cmx = 0.0; // column maximum

        // Find position of pivot row entry and possibly other row entry in
        // column j.
        let mut where_pivot = -1;
        let mut where_other = -1;
        let mut end = Wend[j as usize];
        for pos in Wbegin[j as usize]..end {
            let x = Wvalue[pos as usize].abs();
            if Windex[pos as usize] == pivot_row {
                where_pivot = pos;
            } else if Windex[pos as usize] == other_row {
                where_other = pos;
            // } else if ((x = fabs(Wvalue[pos])) > cmx) {
            } else if x > cmx {
                cmx = x;
            }
        }
        assert!(where_pivot >= 0);
        let xrj = Wvalue[where_pivot as usize];

        // Store pivot row entry in U.
        if Wvalue[where_pivot as usize].abs() > droptol {
            Uindex[Uput as usize] = j;
            Uvalue[Uput as usize] = Wvalue[where_pivot as usize];
            Uput += 1;
        }

        if where_other == -1 {
            // Compute fill-in element.
            let x = -xrj * (other_value / pivot);
            let xabs = x.abs();
            if xabs > droptol {
                // Store fill-in where pivot row entry was.
                Windex[where_pivot as usize] = other_row;
                Wvalue[where_pivot as usize] = x;
                Windex[put as usize] = j;
                put += 1;
                if xabs > cmx {
                    cmx = xabs;
                }
            } else {
                // Remove pivot row entry.
                // end = --Wend[j]; TODO: check
                Wend[j as usize] -= 1;
                end = Wend[j as usize];
                Windex[where_pivot as usize] = Windex[end as usize];
                Wvalue[where_pivot as usize] = Wvalue[end as usize];

                // Decrease column count.
                nz = end - Wbegin[j as usize];
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
            Wend[j as usize] -= 1;
            end = Wend[j as usize];
            Windex[where_pivot as usize] = Windex[end as usize];
            Wvalue[where_pivot as usize] = Wvalue[end as usize];
            if where_other == end {
                where_other = where_pivot;
            }
            Wvalue[where_other as usize] -= xrj * (other_value / pivot);

            // If we have numerical cancellation, then remove the entry and mark
            // the column.
            let x = Wvalue[where_other as usize].abs();
            if x <= droptol {
                // end = --Wend[j]; TODO
                Wend[j as usize] -= 1;
                end = Wend[j as usize];
                Windex[where_other as usize] = Windex[end as usize];
                Wvalue[where_other as usize] = Wvalue[end as usize];
                marked[j as usize] = 1;
                ncancelled += 1;
            } else if x > cmx {
                cmx = x;
            }

            // Decrease column count.
            nz = Wend[j as usize] - Wbegin[j as usize];
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
    Ubegin[(rank + 1) as usize] = Uput;

    // Row file update //

    // If we have numerical cancellation, then we have to remove these entries
    // (marked) from the row pattern. In any case remove pivot column entry.
    if ncancelled != 0 {
        assert_eq!(marked[pivot_col as usize], 0);
        marked[pivot_col as usize] = 1; // treat as cancelled
        let mut put = Wbegin[(m + other_row) as usize]; // compress remaining entries
        let end = Wend[(m + other_row) as usize];
        for pos in put..end {
            let j = Windex[pos as usize];
            if marked[j as usize] != 0 {
                marked[j as usize] = 0;
            } else {
                Windex[put as usize] = j;
                put += 1;
            }
        }
        assert_eq!(end - put, ncancelled + 1);
        Wend[(m + other_row) as usize] = put;
    } else {
        where_ = Wbegin[(m + other_row) as usize];
        while Windex[where_ as usize] != pivot_col {
            assert!(where_ < Wend[(m + other_row) as usize] - 1);
            where_ += 1;
        }
        // end = --Wend[m+other_row]; TODO: check
        Wend[(m + other_row) as usize] -= 1;
        let end = Wend[(m + other_row) as usize];
        Windex[where_ as usize] = Windex[end as usize];
    }

    // Reappend row if no room for update.
    let nfill = rend - (rbeg + 1);
    let room = Wbegin[Wflink[(m + other_row) as usize] as usize] - Wend[(m + other_row) as usize];
    if nfill > room {
        let nz = Wend[(m + other_row) as usize] - Wbegin[(m + other_row) as usize];
        let space = nfill + (stretch as lu_int) * (nz + nfill) + pad;
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
    let mut put = Wend[(m + other_row) as usize];
    for pos in (rbeg + 1)..rend {
        Windex[put as usize] = Windex[pos as usize];
        put += 1;
    }
    Wend[(m + other_row) as usize] = put;

    // Reinsert other row into row counts.
    let nz = Wend[(m + other_row) as usize] - Wbegin[(m + other_row) as usize];
    lu_list_move(
        other_row,
        nz,
        rowcount_flink,
        rowcount_blink,
        m,
        Some(&mut this.min_rownz),
    );

    // Store column in L.
    let mut put = Lbegin_p[rank as usize];
    let x = other_value / pivot;
    if x.abs() > droptol {
        Lindex[put as usize] = other_row;
        Lvalue[put as usize] = x;
        put += 1;
    }
    Lindex[put as usize] = -1; // terminate column
    put += 1;
    Lbegin_p[(rank + 1) as usize] = put;

    // Cleanup:
    // store pivot elemnt;
    // remove pivot colum from column file, pivot row from row file;
    // remove pivot column/row from column/row counts
    colmax[pivot_col as usize] = pivot;
    Wend[pivot_col as usize] = cbeg;
    Wend[(m + pivot_row) as usize] = rbeg;
    lu_list_remove(colcount_flink, colcount_blink, pivot_col);
    lu_list_remove(rowcount_flink, rowcount_blink, pivot_row);

    // Check that row file and column file are consistent. Only use when
    // DEBUG_EXTRA since this check is really expensive.
    if cfg!(feature = "debug_extra") {
        assert_eq!(
            lu_file_diff(
                m,
                &Wbegin[m as usize..],
                &Wend[m as usize..],
                Wbegin,
                Wend,
                Windex,
                None
            ),
            0
        );
        assert_eq!(
            lu_file_diff(
                m,
                Wbegin,
                Wend,
                &Wbegin[m as usize..],
                &Wend[m as usize..],
                Windex,
                None
            ),
            0
        );
    }

    BASICLU_OK
}

fn lu_remove_col(this: &mut lu, j: lu_int) {
    let m = this.m;
    let colcount_flink = this.colcount_flink.as_mut().unwrap();
    let colcount_blink = this.colcount_blink.as_mut().unwrap();
    let rowcount_flink = this.rowcount_flink.as_mut().unwrap();
    let rowcount_blink = this.rowcount_blink.as_mut().unwrap();
    let colmax = &mut this.col_pivot;
    let Wbegin = this.Wbegin.as_mut().unwrap();
    let Wend = this.Wend.as_mut().unwrap();
    let Windex = this.Windex.as_mut().unwrap();
    let cbeg = Wbegin[j as usize];
    let cend = Wend[j as usize];

    // lu_int i, pos, nz, where_;

    // Remove column j from row file.
    for pos in cbeg..cend {
        let i = Windex[pos as usize];
        let mut where_ = Wbegin[(m + i) as usize];
        while Windex[where_ as usize] != j {
            assert!(where_ < Wend[(m + i) as usize] - 1);
            where_ += 1;
        }
        // Windex[where_] = Windex[--Wend[m+i]];
        Wend[(m + i) as usize] -= 1;
        Windex[where_ as usize] = Windex[Wend[(m + i) as usize] as usize];
        let nz = Wend[(m + i) as usize] - Wbegin[(m + i) as usize];
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
    Wend[j as usize] = cbeg;
    lu_list_move(
        j,
        0,
        colcount_flink,
        colcount_blink,
        m,
        Some(&mut this.min_colnz),
    );
}
