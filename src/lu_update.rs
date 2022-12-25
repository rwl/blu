// Copyright (C) 2016-2018  ERGO-Code
//
// Forrest-Tomlin update with reordering.

use crate::basiclu::*;
use crate::lu_dfs::lu_dfs;
use crate::lu_file::{lu_file_compress, lu_file_reappend};
use crate::lu_garbage_perm::lu_garbage_perm;
use crate::lu_internal::*;
use crate::lu_list::lu_list_swap;
use std::time::Instant;

const GAP: LUInt = -1;

macro_rules! flip {
    ($i:expr) => {
        -($i) - 1
    };
}

// Find position of index j in index[start..end-1].
// If end < 0, then the search stops at the first nonnegative index.
// Return end if not found.
fn find(j: LUInt, index: &[LUInt], mut start: LUInt, end: LUInt) -> LUInt {
    if end >= 0 {
        while start < end && index[start as usize] != j {
            start += 1;
        }
        start
    } else {
        while index[start as usize] != j && index[start as usize] >= 0 {
            start += 1;
        }
        if index[start as usize] == j {
            start
        } else {
            end
        }
    }
}

// Find a path from j0 to j0 by a breadth first search. When top < m is
// returned, then the indices in the path (excluding the final j0) are
// jlist[top..m-1]. When top == m is returned, then no such path exists.
//
// The neighbours of node j are index[begin[j]..end[j]-1]. On entry
// marked[j] >= 0 for all nodes j. On return some elements of marked
// are set to zero.
fn bfs_path(
    m: LUInt, // graph has m nodes
    j0: LUInt,
    begin: &[LUInt],
    end: &[LUInt],
    index: &[LUInt],
    jlist: &mut [LUInt],
    marked: &mut [LUInt],
    queue: &mut [LUInt], // size m workspace
) -> LUInt {
    // lu_int j, k, pos, front, tail = 1, top = m, found = 0;
    let mut j: LUInt = -1;
    let mut tail: LUInt = 1;
    let mut top: LUInt = m;
    let mut found: LUInt = 0;

    queue[0] = j0;
    // for (front = 0; front < tail && !found; front++)
    for front in 0..tail {
        if found != 0 {
            break;
        }
        j = queue[front as usize];
        for pos in begin[j as usize]..end[j as usize] {
            let k = index[pos as usize];
            if k == j0 {
                found = 1;
                break;
            }
            if marked[k as usize] >= 0 {
                // not in queue yet
                marked[k as usize] = flip!(j); // parent[k] = j
                queue[tail as usize] = k; // append to queue
                tail += 1;
            }
        }
    }
    if found != 0 {
        // build path (j0,..,j)
        while j != j0 {
            // jlist[--top] = j;
            top -= 1;
            jlist[top as usize] = j;
            j = flip!(marked[j as usize]); // go to parent
            assert!(j >= 0);
        }
        // jlist[--top] = j0;
        top -= 1;
        jlist[top as usize] = j0;
    }
    for pos in 0..tail {
        marked[queue[pos as usize] as usize] = 0; // reset
    }
    top
}

// Compress matrix file to reuse memory gaps. Data line 0 <= i < m begins at
// position begin[i] and ends before the first slot with index[slot] == GAP.
// begin[m] points to the beginning of unused space at file end.
//
// An unused slot in the file must have index[slot] == GAP. All other slots
// must have index[slot] > GAP. index[0] must be unused. On return
// index[1..begin[m]-1] contains the data of nonempty lines. All empty lines
// begin at slot 0. Each two subsequent lines are separated by one gap.
fn compress_packed(m: LUInt, begin: &mut [LUInt], index: &mut [LUInt], value: &mut [f64]) -> LUInt {
    let mut nz = 0;
    let end = begin[m as usize];

    // Mark the beginning of each nonempty line.
    for i in 0..m {
        let p = begin[i as usize];
        if index[p as usize] == GAP {
            begin[i as usize] = 0;
        } else {
            assert!(index[p as usize] > GAP);
            begin[i as usize] = index[p as usize]; // temporarily store index here
            index[p as usize] = GAP - i - 1; // mark beginning of line i
        }
    }

    // Compress nonempty lines.
    assert_eq!(index[0], GAP);
    let mut i = -1;
    let mut put = 1;
    for get in 1..end {
        if index[get as usize] > GAP {
            // shift entry of line i
            assert!(i >= 0);
            index[put as usize] = index[get as usize];
            value[put as usize] = value[get as usize];
            put += 1;
            nz += 1;
        } else if index[get as usize] < GAP {
            // beginning of line i
            assert_eq!(i, -1);
            i = GAP - index[get as usize] - 1;
            index[put as usize] = begin[i as usize]; // store back
            begin[i as usize] = put;
            value[put as usize] = value[get as usize];
            put += 1;
            nz += 1;
        } else if i >= 0 {
            // line i ended at a gap
            i = -1;
            index[put as usize] = GAP;
            put += 1;
        }
    }
    assert_eq!(i, -1);
    begin[m as usize] = put;
    nz
}

// Change row-column mappings for columns jlist[0..nswap]. When row i was
// mapped to column jlist[n], then it will be mapped to column jlist[n+1].
// When row i was mapped to column jlist[nswap], then it will be mapped to
// column jlist[0].
//
// This requires to update pmap, qmap and the rowwise and columwise storage
// of U. It also changes the pivot elements.
//
// Note: This is the most ugly part of the update code and looks horribly
//       inefficient (in particular the list swaps). However, usually nswap
//       is a small number (2, 3, 4, ..), so we don't need to give too much
//       attention to it.
fn permute(this: &mut LU, jlist: &[LUInt], nswap: LUInt) {
    let pmap = &mut pmap!(this);
    let qmap = &mut qmap!(this);
    let u_begin = &mut this.u_begin;
    let w_begin = &mut this.w_begin;
    let w_end = &mut this.w_end;
    let w_flink = &mut this.w_flink;
    let w_blink = &mut this.w_blink;
    let col_pivot = &mut this.col_pivot;
    let row_pivot = &mut this.row_pivot;
    let u_index = &mut this.u_index;
    let u_value = &mut this.u_value;
    let w_index = &mut this.w_index;
    let w_value = &mut this.w_value;

    let j0 = jlist[0];
    let jn = jlist[nswap as usize];
    let i0 = pmap[j0 as usize];
    let in_ = pmap[jn as usize];

    // lu_int begin, end, i, inext, j, jprev, n, where_;
    // double piv;

    assert!(nswap >= 1);
    assert_eq!(qmap[i0 as usize], j0);
    assert_eq!(qmap[in_ as usize], jn);
    assert_eq!(row_pivot[i0 as usize], 0.0);
    assert_eq!(col_pivot[j0 as usize], 0.0);

    // Update row file //

    let begin = w_begin[jn as usize]; // keep for later
    let mut end = w_end[jn as usize];
    let piv = col_pivot[jn as usize];

    // for (n = nswap; n > 0; n--)  TODO: check
    for n in (1..=nswap).rev() {
        let j = jlist[n as usize];
        let jprev = jlist[(n - 1) as usize];

        // When row i was indexed by jprev in the row file before,
        // then it is indexed by j now.
        w_begin[j as usize] = w_begin[jprev as usize];
        w_end[j as usize] = w_end[jprev as usize];
        lu_list_swap(w_flink, w_blink, j, jprev);

        // That row must have an entry in column j because (jprev,j) is an
        // edge in the augmenting path. This entry becomes a pivot element.
        // If jprev is not the first node in the path, then it has an entry
        // in the row (the old pivot) which becomes an off-diagonal entry now.
        let where_ = find(j, w_index, w_begin[j as usize], w_end[j as usize]);
        assert!(where_ < w_end[j as usize]);
        if n > 1 {
            assert_ne!(jprev, j0);
            w_index[where_ as usize] = jprev;
            col_pivot[j as usize] = w_value[where_ as usize];
            assert_ne!(col_pivot[j as usize], 0.0);
            w_value[where_ as usize] = col_pivot[jprev as usize];
        } else {
            assert_eq!(jprev, j0);
            col_pivot[j as usize] = w_value[where_ as usize];
            assert_ne!(col_pivot[j as usize], 0.0);
            w_end[j as usize] -= 1;
            w_index[where_ as usize] = w_index[w_end[j as usize] as usize];
            w_value[where_ as usize] = w_value[w_end[j as usize] as usize];
        }
        this.min_pivot = f64::min(this.min_pivot, col_pivot[j as usize].abs());
        this.max_pivot = f64::max(this.max_pivot, col_pivot[j as usize].abs());
    }

    w_begin[j0 as usize] = begin;
    w_end[j0 as usize] = end;
    let where_ = find(j0, w_index, w_begin[j0 as usize], w_end[j0 as usize]);
    assert!(where_ < w_end[j0 as usize]);
    w_index[where_ as usize] = jn;
    col_pivot[j0 as usize] = w_value[where_ as usize];
    assert_ne!(col_pivot[j0 as usize], 0.0);
    w_value[where_ as usize] = piv;
    this.min_pivot = f64::min(this.min_pivot, col_pivot[j0 as usize].abs());
    this.max_pivot = f64::max(this.max_pivot, col_pivot[j0 as usize].abs());

    // Update column file //

    let begin = u_begin[i0 as usize]; // keep for later

    for n in 0..nswap {
        let i = pmap[jlist[n as usize] as usize];
        let inext = pmap[jlist[(n + 1) as usize] as usize];

        // When column j indexed by inext in the column file before,
        // then it is indexed by i now.
        u_begin[i as usize] = u_begin[inext as usize];

        // That column must have an entry in row i because there is an
        // edge in the augmenting path. This entry becomes a pivot element.
        // There is also an entry in row inext (the old pivot), which now
        // becomes an off-diagonal entry.
        let where_ = find(i, u_index, u_begin[i as usize], -1);
        assert!(where_ >= 0);
        u_index[where_ as usize] = inext;
        row_pivot[i as usize] = u_value[where_ as usize];
        assert_ne!(row_pivot[i as usize], 0.0);
        u_value[where_ as usize] = row_pivot[inext as usize];
    }

    u_begin[in_ as usize] = begin;
    let where_ = find(in_, u_index, u_begin[in_ as usize], -1);
    assert!(where_ >= 0);
    row_pivot[in_ as usize] = u_value[where_ as usize];
    assert_ne!(row_pivot[in_ as usize], 0.0);
    // for (end = where_; Uindex[end] >= 0; end++) ;
    end = where_;
    while u_index[end as usize] >= 0 {
        end += 1;
    }
    u_index[where_ as usize] = u_index[(end - 1) as usize];
    u_value[where_ as usize] = u_value[(end - 1) as usize];
    u_index[(end - 1) as usize] = -1;

    // Update row-column mappings //

    // for (n = nswap; n > 0; n--)
    for n in (1..=nswap).rev() {
        let j = jlist[n as usize];
        let i = pmap[jlist[(n - 1) as usize] as usize];
        pmap[j as usize] = i;
        qmap[i as usize] = j;
    }
    pmap[j0 as usize] = in_;
    qmap[in_ as usize] = j0;

    if cfg!(feature = "debug") {
        for n in 0..=nswap {
            let j = jlist[n as usize];
            let i = pmap[j as usize];
            assert_eq!(row_pivot[i as usize], col_pivot[j as usize]);
        }
    }
}

// Search for column file entry that is missing or different in row file,
// and for row file entry that is missing or different in column file.
// If such an entry is found, then its column and row are returned in *p_col
// and *p_row. If no such entry exists, then *p_col and *p_row are negative.
#[cfg(feature = "debug_extra")]
fn check_consistency(this: &LU, p_col: &mut LUInt, p_row: &mut LUInt) {
    let m = this.m;
    let pmap = &this.pmap;
    let qmap = &this.qmap;
    let u_begin = &this.u_begin;
    let w_begin = &this.w_begin;
    let w_end = &this.w_end;
    let u_index = this.u_index.as_ref().unwrap();
    let u_value = this.u_value.as_ref().unwrap();
    let w_index = this.w_index.as_ref().unwrap();
    let w_value = this.w_value.as_ref().unwrap();
    // lu_int i, ientry, j, jentry, pos, where_, found;

    for i in 0..m {
        // process column file entries
        let j = qmap[i];
        // for (pos = Ubegin[i]; (ientry = Uindex[pos]) >= 0; pos++)
        let mut pos = u_begin[i];
        while u_index[pos] >= 0 {
            let ientry = u_index[pos];
            jentry = qmap[ientry];
            where_ = w_begin[jentry];
            while where_ < w_end[jentry] && w_index[where_] != j {
                where_ += 1;
            }
            let found = where_ < w_end[jentry] && w_value[where_] == u_value[pos];
            if !found {
                *p_col = j;
                *p_row = ientry;
                return;
            }
            pos += 1;
        }
    }
    for j in 0..m {
        // process row file entries
        let i = pmap[j];
        for pos in w_begin[j]..w_end[j] {
            let jentry = w_index[pos];
            let ientry = pmap[jentry];
            // for (where_ = Ubegin[ientry]; Uindex[where_] >= 0 && Uindex[where_] != i; where_++);
            let where_ = u_begin[ientry];
            while u_index[where_] >= 0 && u_index[where_] != i {
                where_ += 1;
            }
            let found = u_index[where_] == i && u_value[where_] == w_value[pos];
            if !found {
                *p_col = jentry;
                *p_row = i;
                return;
            }
        }
    }
    *p_col = -1;
    *p_row = -1;
}

// Insert spike into U and restore triangularity. If the spiked matrix
// is permuted triangular, then only permutations are updated. If the
// spiked matrix is not permuted triangular, then the Forrest-Tomlin
// update is used and the number of row eta matrices increases by 1.
//
// Return:
//
//  BASICLU_OK                      update successfully completed
//  BASICLU_REALLOCATE              require more memory in W
//  BASICLU_ERROR_SINGULAR_UPDATE   new pivot element is zero or < abstol
pub(crate) fn lu_update(this: &mut LU, xtbl: f64) -> LUInt {
    let m = this.m;
    let nforrest = this.nforrest;
    let mut u_nz = this.u_nz;
    let pad = this.pad;
    let stretch = this.stretch;
    // let pmap = &mut this.pmap;
    // let qmap = &mut this.solve.qmap;
    // let pivotcol = &mut this.pivotcol;
    // let pivotrow = &mut this.pivotrow;
    // let Ubegin = &mut this.Ubegin;
    // let r_begin = &mut this.r_begin;
    // let Wbegin = &mut this.Wbegin;
    // let Wend = &mut this.Wend;
    // let Wflink = &mut this.Wflink;
    // let Wblink = &mut this.Wblink;
    // let col_pivot = &mut this.col_pivot;
    // let row_pivot = &mut this.row_pivot;
    // let Lindex = &mut this.Lindex;
    // let Lvalue = &mut this.Lvalue;
    // let Uindex = &mut this.Uindex;
    // let Uvalue = &mut this.Uvalue;
    // let Windex = &mut this.Windex;
    // let Wvalue = &mut this.Wvalue;
    // let marked = &mut this.marked;
    // let iwork1 = &mut this.iwork1;
    // let iwork2 = &mut iwork1[m as usize..];
    // let (iwork1, iwork2) = this.iwork1.split_at_mut(m as usize);
    // let work1 = &mut this.work1;

    let jpivot = this.btran_for_update;
    let ipivot = pmap!(this)[jpivot as usize];
    let oldpiv = this.col_pivot[jpivot as usize];
    let mut status = BASICLU_OK;

    let mut ipivot_vec = vec![ipivot]; // FIXME
    let mut jpivot_vec = vec![jpivot];

    // lu_int i, j, jnext, n, nz, t, put, pos, end, where_, room, grow, used,need,M;
    // lu_int have_diag, intersect, istriangular, nz_roweta, nz_spike;
    // lu_int nreach, *col_reach, *row_reach;
    // double spike_diag, newpiv, piverr;
    // double tic[2], elapsed;
    let tic = Instant::now();

    assert!(nforrest < m);

    // Note: If the singularity test fails or memory is insufficient, then the
    //       update is aborted and the user may call this routine a second time.
    //       Changes made to data structures in the first call must not violate
    //       the logic in the second call.

    // Prepare //

    // if present, move diagonal element to end of spike
    let mut spike_diag = 0.0;
    let mut have_diag = 0;
    let mut put = this.u_begin[m as usize];
    // for (pos = put; (i = Uindex[pos]) >= 0; pos++)
    let mut pos = put;
    while this.u_index[pos as usize] >= 0 {
        let i = this.u_index[pos as usize];
        if i != ipivot {
            this.u_index[put as usize] = i;
            this.u_value[put as usize] = this.u_value[pos as usize];
            put += 1;
        } else {
            spike_diag = this.u_value[pos as usize];
            have_diag = 1;
        }
        pos += 1;
    }
    if have_diag != 0 {
        this.u_index[put as usize] = ipivot;
        this.u_value[put as usize] = spike_diag;
    }
    let nz_spike = put - this.u_begin[m as usize]; // nz excluding diagonal

    let nz_roweta = r_begin!(this)[(nforrest + 1) as usize] - r_begin!(this)[nforrest as usize];

    // Compute pivot //

    // newpiv is the diagonal element in the spike column after the
    // Forrest-Tomlin update has been applied. It can be computed as
    //
    //    newpiv = spike_diag - dot(spike,row eta)                (1)
    // or
    //    newpiv = xtbl * oldpiv,                                 (2)
    //
    // where spike_diag is the diaognal element in the spike column
    // before the Forrest-Tomlin update and oldpiv was the pivot element
    // in column jpivot before inserting the spike. This routine uses
    // newpiv from (1) and reports the difference to (2) to the user
    // to monitor numerical stability.
    //
    // While computing (1), count intersection of patterns of spike and
    // row eta.

    // scatter row eta into work1 and mark positions
    // M = ++this.marker;
    this.marker += 1;
    let marker = this.marker;
    for pos in r_begin!(this)[nforrest as usize]..r_begin!(this)[(nforrest + 1) as usize] {
        let i = this.l_index[pos as usize];
        marked!(this)[i as usize] = marker;
        this.work1[i as usize] = this.l_value[pos as usize];
    }

    // compute newpiv and count intersection
    let mut newpiv = spike_diag;
    let mut intersect = 0;
    for pos in this.u_begin[m as usize]..this.u_begin[m as usize] + nz_spike {
        let i = this.u_index[pos as usize];
        assert_ne!(i, ipivot);
        if marked!(this)[i as usize] == marker {
            newpiv -= this.u_value[pos as usize] * this.work1[i as usize];
            intersect += 1;
        }
    }

    // singularity test
    if newpiv == 0.0 || newpiv.abs() < this.abstol {
        status = BASICLU_ERROR_SINGULAR_UPDATE;
        return status;
    }

    // stability measure
    let piverr = (newpiv - xtbl * oldpiv).abs();

    // Insert spike //

    // calculate bound on file growth
    let mut grow = 0;
    for pos in this.u_begin[m as usize]..(this.u_begin[m as usize] + nz_spike) {
        let i = this.u_index[pos as usize];
        assert_ne!(i, ipivot);
        let j = qmap!(this)[i as usize];
        let jnext = this.w_flink[j as usize];
        if this.w_end[j as usize] == this.w_begin[jnext as usize] {
            let nz = this.w_end[j as usize] - this.w_begin[j as usize];
            grow += nz + 1; // row including spike entry
            grow += (stretch as LUInt) * (nz + 1) + pad; // extra room
        }
    }

    // reallocate if necessary
    let room = this.w_end[m as usize] - this.w_begin[m as usize];
    if grow > room {
        this.addmem_w = grow - room;
        status = BASICLU_REALLOCATE;
        return status;
    }

    // remove column jpivot from row file
    let mut nz = 0;
    // for (pos = Ubegin[ipivot]; (i = Uindex[pos]) >= 0; pos++)
    pos = this.u_begin[ipivot as usize];
    while this.u_index[pos as usize] >= 0 {
        let i = this.u_index[pos as usize];
        let j = qmap!(this)[i as usize];
        // end = Wend[j]--;
        let end = this.w_end[j as usize];
        this.w_end[j as usize] -= 1;
        let where_ = find(jpivot, &this.w_index, this.w_begin[j as usize], end);
        assert!(where_ < end);
        this.w_index[where_ as usize] = this.w_index[(end - 1) as usize];
        this.w_value[where_ as usize] = this.w_value[(end - 1) as usize];
        nz += 1;
        pos += 1;
    }
    u_nz -= nz;

    // erase column jpivot in column file
    // for (pos = Ubegin[ipivot]; Uindex[pos] >= 0; pos++)
    pos = this.u_begin[ipivot as usize];
    while this.u_index[pos as usize] >= 0 {
        this.u_index[pos as usize] = GAP;
        pos += 1;
    }

    // set column pointers to spike, chop off diagonal
    this.u_begin[ipivot as usize] = this.u_begin[m as usize];
    this.u_begin[m as usize] += nz_spike;
    // Uindex[Ubegin[m]++] = GAP;
    this.u_index[this.u_begin[m as usize] as usize] = GAP;
    this.u_begin[m as usize] += 1;

    // insert spike into row file
    // for (pos = Ubegin[ipivot]; (i = Uindex[pos]) >= 0; pos++)
    pos = this.u_begin[ipivot as usize];
    while this.u_index[pos as usize] >= 0 {
        let i = this.u_index[pos as usize];
        let j = qmap!(this)[i as usize];
        let jnext = this.w_flink[j as usize];
        if this.w_end[j as usize] == this.w_begin[jnext as usize] {
            nz = this.w_end[j as usize] - this.w_begin[j as usize];
            let room = 1 + (stretch as LUInt) * (nz + 1) + pad;
            lu_file_reappend(
                j,
                m,
                &mut this.w_begin,
                &mut this.w_end,
                &mut this.w_flink,
                &mut this.w_blink,
                &mut this.w_index,
                &mut this.w_value,
                room,
            );
        }
        // end = Wend[j]++;
        let end = this.w_end[j as usize];
        this.w_end[j as usize] += 1;
        this.w_index[end as usize] = jpivot;
        this.w_value[end as usize] = this.u_value[pos as usize];
        pos += 1;
    }
    u_nz += nz_spike;

    // insert diagonal
    this.col_pivot[jpivot as usize] = spike_diag;
    this.row_pivot[ipivot as usize] = spike_diag;

    // Test triangularity //

    let (istriangular, nreach_opt, row_reach_opt, col_reach_opt) = if have_diag != 0 {
        // When the spike has a nonzero diagonal element, then the spiked matrix
        // is (symmetrically) permuted triangular if and only if reach(ipivot)
        // does not intersect with the spike pattern except for ipivot. Since
        // reach(ipivot) \ {ipivot} is the structural pattern of the row eta,
        // the matrix is permuted triangular iff the patterns of the row eta
        // and the spike do not intersect.
        //
        // To update the permutations below, we have to provide reach(ipivot)
        // and the associated column indices in topological order as arrays
        // row_reach[0..nreach-1] and col_reach[0..nreach-1]. Because the
        // pattern of the row eta was computed by a dfs, we obtain row_reach
        // simply by adding ipivot to the front. col_reach can then be obtained
        // through qmap.
        let istriangular = intersect == 0;
        if istriangular {
            this.min_pivot = f64::min(this.min_pivot, newpiv.abs());
            this.max_pivot = f64::max(this.max_pivot, newpiv.abs());

            // build row_reach and col_reach in topological order
            let nreach = nz_roweta + 1;
            // let row_reach = &mut iwork1[..];
            // let (iwork1, iwork2) = this.iwork1.split_at_mut(m as usize);
            // let row_reach = iwork1;
            // let col_reach = iwork2;
            let mut row_reach = vec![0; nreach as usize - 1]; // FIXME: iwork1
            let mut col_reach = vec![0; nreach as usize - 1];
            row_reach[0] = ipivot;
            col_reach[0] = jpivot;
            pos = r_begin!(this)[nforrest as usize];
            for n in 1..nreach {
                let i = this.l_index[pos as usize];
                pos += 1;
                row_reach[n as usize] = i;
                col_reach[n as usize] = qmap!(this)[i as usize];
            }
            this.nsymperm_total += 1;

            (istriangular, Some(nreach), Some(row_reach), Some(col_reach))
        } else {
            (istriangular, None, None, None)
        }
    } else {
        // The spike has a zero diagonal element, so the spiked matrix may only
        // be the *un*symmetric permutation of an upper triangular matrix.
        //
        // Part 1:
        //
        // Find an augmenting path in U[pmap,:] starting from jpivot.
        // An augmenting path is a sequence of column indices such that there
        // is an edge from each node to the next, and an edge from the final
        // node back to jpivot.
        //
        // bfs_path computes such a path in path[top..m-1].
        //
        // Because jpivot has no self-edge, the path must have at least two
        // nodes. The path must exist because otherwise the spiked matrix was
        // structurally singular and the singularity test above had failed.
        let top = {
            let (iwork1, iwork2) = iwork1!(this).split_at_mut(m as usize);
            // lu_int *path = iwork1, top;
            // let path = iwork1;
            let path = iwork1;
            // lu_int *reach = iwork2, rtop;
            // let reach = iwork2;
            // lu_int *pstack = (void *) work1;
            // let pstack = work1;

            let top = bfs_path(
                m,
                jpivot,
                &this.w_begin,
                &this.w_end,
                &this.w_index,
                path,
                &mut marked!(this),
                iwork2,
            );
            assert!(top < m - 1);
            assert_eq!(path[top as usize], jpivot);

            // let reach = iwork2;

            top
        };

        // Part 2a:
        //
        // For each path index j (except the final one) mark the nodes in
        // reach(j), where the reach is computed in U[pmap,:] without the path
        // edges. If a path index is contained in the reach of an index that
        // comes before it in the path, then U is not permuted triangular.
        //
        // At the same time assemble the combined reach of all path nodes
        // (except the final one) in U[pmap_new,:], where pmap_new is the
        // column-row mapping after applying the permutation associated with
        // the augmenting path. We only have to replace each index where the
        // dfs starts by the next index in the path. The combined reach is
        // then assembled in topological order in
        //
        //    reach[rtop..m-1].
        let (mut istriangular, mut rtop) = {
            let (iwork1, iwork2) = iwork1!(this).split_at_mut(m as usize);
            let path = iwork1;
            let reach = iwork2;
            let pstack = &mut this.work1;

            let mut istriangular = true;
            let mut rtop = m;
            // M = ++this.marker;
            this.marker += 1;
            let marker = this.marker;
            // for (t = top; t < m-1 && istriangular; t++)
            for t in top..m - 1 {
                if !istriangular {
                    break;
                }
                let j = path[t as usize];
                let jnext = path[(t + 1) as usize];
                let where_ = find(
                    jnext,
                    &this.w_index,
                    this.w_begin[j as usize],
                    this.w_end[j as usize],
                );
                assert!(where_ < this.w_end[j as usize]);
                this.w_index[where_ as usize] = j; // take out for a moment
                rtop = lu_dfs(
                    j,
                    &this.w_begin,
                    Some(&this.w_end),
                    &this.w_index,
                    rtop,
                    reach,
                    pstack,
                    &mut marked!(this),
                    marker,
                );
                assert_eq!(reach[rtop as usize], j);
                reach[rtop as usize] = jnext;
                this.w_index[where_ as usize] = jnext; /* restore */
                istriangular = marked!(this)[jnext as usize] != marker;
            }
            (istriangular, rtop)
        };

        // Part 2b:
        //
        // If the matrix looks triangular so far, then also mark the reach of
        // the final path node, which is reach(jpivot) in U[pmap_new,:].
        // U is then permuted triangular iff the combined reach does not
        // intersect the spike pattern except in the final path index.
        if istriangular {
            let (iwork1, iwork2) = iwork1!(this).split_at_mut(m as usize);
            let path = iwork1;
            let reach = iwork2;
            let pstack = &mut this.work1;

            let j = path[(m - 1) as usize];
            rtop = lu_dfs(
                j,
                &this.w_begin,
                Some(&this.w_end),
                &this.w_index,
                rtop,
                reach,
                pstack,
                &mut marked!(this),
                marker,
            );
            assert_eq!(reach[rtop as usize], j);
            reach[rtop as usize] = jpivot;
            marked!(this)[j as usize] -= 1; // unmark for a moment
                                            // for (pos = Ubegin[ipivot]; (i = Uindex[pos]) >= 0; pos++)
            pos = this.u_begin[ipivot as usize];
            while this.u_index[pos as usize] >= 0 {
                let i = this.u_index[pos as usize];
                if marked!(this)[qmap!(this)[i as usize] as usize] == marker {
                    istriangular = false;
                }
                pos += 1;
            }
            marked!(this)[j as usize] += 1; /* restore */
        }

        // If U is permuted triangular, then permute to zero-free diagonal.
        // Set up row_reach[0..nreach-1] and col_reach[0..nreach-1] for
        // updating the permutations below. The column reach is the combined
        // reach of the path nodes. The row reach is is given through pmap.
        if istriangular {
            let nswap = m - top - 1;
            // permute(this, &path[top as usize..], nswap);
            permute(
                this,
                &iwork1!(this)[top as usize..(top + nswap) as usize].to_vec(),
                nswap,
            ); // usually nswap is a small number
            u_nz -= 1;

            let (iwork1, iwork2) = iwork1!(this).split_at_mut(m as usize);
            // let path = iwork1;
            let reach = iwork2;

            assert_eq!(reach[rtop as usize], jpivot);
            // let col_reach = &mut reach[rtop as usize..]; /* stored in iwork2 */
            // let row_reach = &mut iwork1[rtop as usize..];
            let nreach = m - rtop;
            let col_reach = reach[rtop as usize..(rtop + nreach) as usize].to_vec(); /* stored in iwork2 */
            // FIXME: iwork1
            let mut row_reach = iwork1[rtop as usize..(rtop + nreach) as usize].to_vec();
            for n in 0..nreach {
                row_reach[n as usize] = pmap!(this)[col_reach[n as usize] as usize];
            }
            (istriangular, Some(nreach), Some(row_reach), Some(col_reach))
        } else {
            (istriangular, None, None, None)
        }
    };

    // Forrest-Tomlin update //

    let (nreach, row_reach, col_reach) = if !istriangular {
        // remove row ipivot from column file
        // for (pos = Wbegin[jpivot]; pos < Wend[jpivot]; pos++)
        pos = this.w_begin[jpivot as usize];
        while pos < this.w_end[jpivot as usize] {
            let j = this.w_index[pos as usize];
            assert_ne!(j, jpivot);
            let mut where_ = -1;
            // for (end = Ubegin[pmap[j]]; (i = Uindex[end]) >= 0; end++)
            let mut end = this.u_begin[pmap!(this)[j as usize] as usize];
            while this.u_index[end as usize] >= 0 {
                let i = this.u_index[end as usize];
                if i == ipivot {
                    where_ = end;
                }
                end += 1;
            }
            assert!(where_ >= 0);
            this.u_index[where_ as usize] = this.u_index[(end - 1) as usize];
            this.u_value[where_ as usize] = this.u_value[(end - 1) as usize];
            this.u_index[(end - 1) as usize] = -1;
            u_nz -= 1;
            pos += 1;
        }

        // remove row ipivot from row file
        this.w_end[jpivot as usize] = this.w_begin[jpivot as usize];

        // replace pivot
        this.col_pivot[jpivot as usize] = newpiv;
        this.row_pivot[ipivot as usize] = newpiv;
        this.min_pivot = f64::min(this.min_pivot, newpiv.abs());
        this.max_pivot = f64::max(this.max_pivot, newpiv.abs());

        // drop zeros from row eta; update max entry of row etas
        nz = 0;
        put = r_begin!(this)[nforrest as usize];
        let mut max_eta = 0.0;
        for pos in put..r_begin!(this)[(nforrest + 1) as usize] {
            if this.l_value[pos as usize] != 0.0 {
                max_eta = f64::max(max_eta, this.l_value[pos as usize].abs());
                this.l_index[put as usize] = this.l_index[pos as usize];
                this.l_value[put as usize] = this.l_value[pos as usize];
                put += 1;
                nz += 1;
            }
        }
        r_begin!(this)[(nforrest + 1) as usize] = put;
        this.r_nz += nz;
        this.max_eta = f64::max(this.max_eta, max_eta);

        // prepare permutation update
        let nreach: LUInt = 1;
        // let row_reach = &mut ipivot_vec[..];
        // let col_reach = &mut jpivot_vec[..];
        let row_reach = ipivot_vec.to_vec();
        let col_reach = jpivot_vec.to_vec();
        this.nforrest += 1;
        this.nforrest_total += 1;

        (nreach, row_reach, col_reach)
    } else {
        (
            nreach_opt.unwrap(),
            row_reach_opt.unwrap(),
            col_reach_opt.unwrap(),
        )
    };

    // Update permutations //

    if this.pivotlen + nreach > 2 * m {
        lu_garbage_perm(this);
    }

    // append row indices row_reach[0..nreach-1] to end of pivot sequence
    put = this.pivotlen;
    for n in 0..nreach {
        pivotrow!(this)[put as usize] = row_reach[n as usize];
        put += 1;
    }

    // append col indices col_reach[0..nreach-1] to end of pivot sequence
    put = this.pivotlen;
    for n in 0..nreach {
        pivotcol!(this)[put as usize] = col_reach[n as usize];
        put += 1;
    }

    this.pivotlen += nreach;

    // Clean up //

    // compress U if used memory is shrinked sufficiently
    let used = this.u_begin[m as usize];
    if used - u_nz - m > (this.compress_thres as LUInt) * used {
        nz = compress_packed(m, &mut this.u_begin, &mut this.u_index, &mut this.u_value);
        assert_eq!(nz, u_nz);
    }

    // compress W if used memory is shrinked suficiently
    let used = this.w_begin[m as usize];
    let need = u_nz + (stretch as LUInt) * u_nz + m * pad;
    if (used - need) > (this.compress_thres as LUInt) * used {
        nz = lu_file_compress(
            m,
            &mut this.w_begin,
            &mut this.w_end,
            &mut this.w_flink,
            &mut this.w_index,
            &mut this.w_value,
            stretch,
            pad,
        );
        assert_eq!(nz, u_nz);
    }

    let elapsed = tic.elapsed().as_secs_f64();
    this.time_update += elapsed;
    this.time_update_total += elapsed;
    this.pivot_error = piverr / (1.0 + newpiv.abs());
    this.u_nz = u_nz;
    this.btran_for_update = -1;
    this.ftran_for_update = -1;
    this.update_cost_numer += nz_roweta as f64;
    this.nupdate += 1;
    this.nupdate_total += 1;

    #[cfg(feature = "debug_extra")]
    {
        let mut col = -1;
        let mut row = -1;
        check_consistency(&this, &mut col, &mut row);
        assert!(col < 0 && row < 0);
    }

    status
}
