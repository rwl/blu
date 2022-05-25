// Copyright (C) 2016-2018  ERGO-Code
//
// Forrest-Tomlin update with reordering.

use crate::basiclu::*;
use crate::lu_dfs::lu_dfs;
use crate::lu_file::{lu_file_compress, lu_file_reappend};
use crate::lu_garbage_perm::lu_garbage_perm;
use crate::lu_internal::lu;
use crate::lu_list::lu_list_swap;
use std::time::Instant;

const GAP: lu_int = -1;

macro_rules! FLIP {
    ($i:expr) => {
        -($i) - 1
    };
}

// Find position of index j in index[start..end-1].
// If end < 0, then the search stops at the first nonnegative index.
// Return end if not found.
fn find(j: lu_int, index: &[lu_int], mut start: lu_int, end: lu_int) -> lu_int {
    if end >= 0 {
        while start < end && index[start] != j {
            start += 1;
        }
        start
    } else {
        while index[start] != j && index[start] >= 0 {
            start += 1;
        }
        if index[start] == j {
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
    m: lu_int, // graph has m nodes
    j0: lu_int,
    begin: &[lu_int],
    end: &[lu_int],
    index: &[lu_int],
    jlist: &[lu_int],
    marked: &[lu_int],
    queue: &mut [lu_int], // size m workspace
) -> lu_int {
    // lu_int j, k, pos, front, tail = 1, top = m, found = 0;
    let mut j = -1;
    let mut tail = 1;
    let mut top = m;
    let mut found = 0;

    queue[0] = j0;
    // for (front = 0; front < tail && !found; front++)
    for front in 0..tail {
        if found != 0 {
            break;
        }
        j = queue[front];
        for pos in begin[j]..end[j] {
            let k = index[pos];
            if k == j0 {
                found = 1;
                break;
            }
            if marked[k] >= 0 {
                // not in queue yet
                marked[k] = FLIP!(j); // parent[k] = j
                queue[tail] = k; // append to queue
                tail += 1;
            }
        }
    }
    if found != 0 {
        // build path (j0,..,j)
        while j != j0 {
            // jlist[--top] = j;
            top -= 1;
            jlist[top] = j;
            j = FLIP!(marked[j]); // go to parent
            assert!(j >= 0);
        }
        // jlist[--top] = j0;
        top -= 1;
        jlist[top] = j0;
    }
    for pos in 0..tail {
        marked[queue[pos]] = 0; // reset
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
fn compress_packed(m: lu_int, begin: &[lu_int], index: &mut [lu_int], value: &mut [f64]) -> lu_int {
    let mut nz = 0;
    let end = begin[m];

    // Mark the beginning of each nonempty line.
    for i in 0..m {
        p = begin[i];
        if index[p] == GAP {
            begin[i] = 0;
        } else {
            assert!(index[p] > GAP);
            begin[i] = index[p]; // temporarily store index here
            index[p] = GAP - i - 1; // mark beginning of line i
        }
    }

    // Compress nonempty lines.
    assert_eq!(index[0], GAP);
    let mut i = -1;
    let mut put = 1;
    for get in 1..end {
        if index[get] > GAP {
            // shift entry of line i
            assert!(i >= 0);
            index[put] = index[get];
            value[put] = value[get];
            put += 1;
            nz += 1;
        } else if index[get] < GAP {
            // beginning of line i
            assert_eq!(i, -1);
            i = GAP - index[get] - 1;
            index[put] = begin[i]; // store back
            begin[i] = put;
            value[put] = value[get];
            put += 1;
            nz += 1;
        } else if i >= 0 {
            // line i ended at a gap
            i = -1;
            index[put] = GAP;
            put += 1;
        }
    }
    assert_eq!(i, -1);
    begin[m] = put;
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
fn permute(this: &mut lu, jlist: &[lu_int], nswap: lu_int) {
    let pmap = &this.pmap;
    let qmap = &this.qmap;
    let Ubegin = &this.Ubegin;
    let Wbegin = &this.Wbegin;
    let Wend = &this.Wend;
    let Wflink = &mut this.Wflink;
    let Wblink = &mut this.Wblink;
    let col_pivot = &this.col_pivot;
    let row_pivot = &this.row_pivot;
    let Uindex = this.Uindex.as_ref().unwrap();
    let Uvalue = this.Uvalue.as_ref().unwrap();
    let Windex = this.Windex.as_ref().unwrap();
    let Wvalue = this.Wvalue.as_ref().unwrap();

    let j0 = jlist[0];
    let jn = jlist[nswap];
    let i0 = pmap[j0];
    let in_ = pmap[jn];

    // lu_int begin, end, i, inext, j, jprev, n, where_;
    // double piv;

    assert!(nswap >= 1);
    assert_eq!(qmap[i0], j0);
    assert_eq!(qmap[in_], jn);
    assert_eq!(row_pivot[i0], 0);
    assert_eq!(col_pivot[j0], 0);

    // Update row file //

    let begin = Wbegin[jn]; // keep for later
    let mut end = Wend[jn];
    let piv = col_pivot[jn];

    // for (n = nswap; n > 0; n--)  TODO: check
    for n in (1..=nswap).rev() {
        let j = jlist[n];
        let jprev = jlist[n - 1];

        // When row i was indexed by jprev in the row file before,
        // then it is indexed by j now.
        Wbegin[j] = Wbegin[jprev];
        Wend[j] = Wend[jprev];
        lu_list_swap(Wflink, Wblink, j, jprev);

        // That row must have an entry in column j because (jprev,j) is an
        // edge in the augmenting path. This entry becomes a pivot element.
        // If jprev is not the first node in the path, then it has an entry
        // in the row (the old pivot) which becomes an off-diagonal entry now.
        where_ = find(j, Windex, Wbegin[j], Wend[j]);
        assert!(where_ < Wend[j]);
        if n > 1 {
            assert_ne!(jprev, j0);
            Windex[where_] = jprev;
            col_pivot[j] = Wvalue[where_];
            assert!(col_pivot[j]);
            Wvalue[where_] = col_pivot[jprev];
        } else {
            assert_eq!(jprev, j0);
            col_pivot[j] = Wvalue[where_];
            assert!(col_pivot[j]);
            Wend[j] -= 1;
            Windex[where_] = Windex[Wend[j]];
            Wvalue[where_] = Wvalue[Wend[j]];
        }
        this.min_pivot = f64::min(this.min_pivot, col_pivot[j].abs());
        this.max_pivot = f64::max(this.max_pivot, col_pivot[j].abs());
    }

    Wbegin[j0] = begin;
    Wend[j0] = end;
    let where_ = find(j0, Windex, Wbegin[j0], Wend[j0]);
    assert!(where_ < Wend[j0]);
    Windex[where_] = jn;
    col_pivot[j0] = Wvalue[where_];
    assert!(col_pivot[j0]);
    Wvalue[where_] = piv;
    this.min_pivot = f64::min(this.min_pivot, col_pivot[j0].abs());
    this.max_pivot = f64::max(this.max_pivot, col_pivot[j0].abs());

    // Update column file //

    let begin = Ubegin[i0]; // keep for later

    for n in 0..nswap {
        let i = pmap[jlist[n]];
        let inext = pmap[jlist[n + 1]];

        // When column j indexed by inext in the column file before,
        // then it is indexed by i now.
        Ubegin[i] = Ubegin[inext];

        // That column must have an entry in row i because there is an
        // edge in the augmenting path. This entry becomes a pivot element.
        // There is also an entry in row inext (the old pivot), which now
        // becomes an off-diagonal entry.
        let where_ = find(i, Uindex, Ubegin[i], -1);
        assert!(where_ >= 0);
        Uindex[where_] = inext;
        row_pivot[i] = Uvalue[where_];
        assert!(row_pivot[i]);
        Uvalue[where_] = row_pivot[inext];
    }

    Ubegin[in_] = begin;
    let where_ = find(in_, Uindex, Ubegin[in_], -1);
    assert!(where_ >= 0);
    row_pivot[in_] = Uvalue[where_];
    assert!(row_pivot[in_]);
    // for (end = where_; Uindex[end] >= 0; end++) ;
    end = where_;
    while Uindex[end] >= 0 {
        end += 1;
    }
    Uindex[where_] = Uindex[end - 1];
    Uvalue[where_] = Uvalue[end - 1];
    Uindex[end - 1] = -1;

    // Update row-column mappings //

    // for (n = nswap; n > 0; n--)
    for n in (1..=nswap).rev() {
        j = jlist[n];
        i = pmap[jlist[n - 1]];
        pmap[j] = i;
        qmap[i] = j;
    }
    pmap[j0] = in_;
    qmap[in_] = j0;

    if cfg!(feature = "debug") {
        for n in 0..=nswap {
            j = jlist[n];
            i = pmap[j];
            assert_eq!(row_pivot[i], col_pivot[j]);
        }
    }
}

// Search for column file entry that is missing or different in row file,
// and for row file entry that is missing or different in column file.
// If such an entry is found, then its column and row are returned in *p_col
// and *p_row. If no such entry exists, then *p_col and *p_row are negative.
#[cfg(feature = "debug_extra")]
fn check_consistency(this: &mut lu, p_col: &mut lu_int, p_row: &mut lu_int) {
    let m = this.m;
    let pmap = &this.pmap;
    let qmap = &this.qmap;
    let Ubegin = &this.Ubegin;
    let Wbegin = &this.Wbegin;
    let Wend = &this.Wend;
    let Uindex = this.Uindex.as_ref().unwrap();
    let Uvalue = this.Uvalue.as_ref().unwrap();
    let Windex = this.Windex.as_ref().unwrap();
    let Wvalue = this.Wvalue.as_ref().unwrap();
    // lu_int i, ientry, j, jentry, pos, where_, found;

    for i in 0..m {
        // process column file entries
        let j = qmap[i];
        // for (pos = Ubegin[i]; (ientry = Uindex[pos]) >= 0; pos++)
        let mut pos = Ubegin[i];
        while Uindex[pos] >= 0 {
            let ientry = Uindex[pos];
            jentry = qmap[ientry];
            where_ = Wbegin[jentry];
            while where_ < Wend[jentry] && Windex[where_] != j {
                where_ += 1;
            }
            let found = where_ < Wend[jentry] && Wvalue[where_] == Uvalue[pos];
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
        for pos in Wbegin[j]..Wend[j] {
            let jentry = Windex[pos];
            let ientry = pmap[jentry];
            // for (where_ = Ubegin[ientry]; Uindex[where_] >= 0 && Uindex[where_] != i; where_++);
            let where_ = Ubegin[ientry];
            while Uindex[where_] >= 0 && Uindex[where_] != i {
                where_ += 1;
            }
            let found = Uindex[where_] == i && Uvalue[where_] == Wvalue[pos];
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
//  BASICLU_ERROR_singular_update   new pivot element is zero or < abstol
pub(crate) fn lu_update(this: &mut lu, xtbl: f64) -> lu_int {
    let m = this.m;
    let nforrest = this.nforrest;
    let mut Unz = this.Unz;
    let pad = this.pad;
    let stretch = this.stretch;
    let pmap = &mut this.pmap;
    let qmap = &mut this.qmap;
    let pivotcol = &mut this.pivotcol;
    let pivotrow = &mut this.pivotrow;
    let Ubegin = &mut this.Ubegin;
    let Rbegin = &mut this.Rbegin;
    let Wbegin = &mut this.Wbegin;
    let Wend = &mut this.Wend;
    let Wflink = &mut this.Wflink;
    let Wblink = &mut this.Wblink;
    let col_pivot = &mut this.col_pivot;
    let row_pivot = &mut this.row_pivot;
    let Lindex = this.Lindex.as_mut().unwrap();
    let Lvalue = this.Lvalue.as_mut().unwrap();
    let Uindex = this.Uindex.as_mut().unwrap();
    let Uvalue = this.Uvalue.as_mut().unwrap();
    let Windex = this.Windex.as_mut().unwrap();
    let Wvalue = this.Wvalue.as_mut().unwrap();
    let marked = &mut this.marked;
    let iwork1 = &mut this.iwork1;
    let iwork2 = &mut iwork1[m..];
    let work1 = &mut this.work1;

    let jpivot = this.btran_for_update;
    let ipivot = pmap[jpivot];
    let oldpiv = col_pivot[jpivot];
    let mut status = BASICLU_OK;

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
    let mut put = Ubegin[m];
    // for (pos = put; (i = Uindex[pos]) >= 0; pos++)
    let mut pos = put;
    while Uindex[pos] >= 0 {
        let i = Uindex[pos];
        if i != ipivot {
            Uindex[put] = i;
            Uvalue[put] = Uvalue[pos];
            put += 1;
        } else {
            spike_diag = Uvalue[pos];
            have_diag = 1;
        }
        pos += 1;
    }
    if have_diag != 0 {
        Uindex[put] = ipivot;
        Uvalue[put] = spike_diag;
    }
    nz_spike = put - Ubegin[m]; // nz excluding diagonal

    nz_roweta = Rbegin[nforrest + 1] - Rbegin[nforrest];

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
    M = this.marker;
    for pos in Rbegin[nforrest]..Rbegin[nforrest + 1] {
        i = Lindex[pos];
        marked[i] = M;
        work1[i] = Lvalue[pos];
    }

    // compute newpiv and count intersection
    newpiv = spike_diag;
    intersect = 0;
    for pos in Ubegin[m]..Ubegin[m] + nz_spike {
        i = Uindex[pos];
        assert_ne!(i, ipivot);
        if marked[i] == M {
            newpiv -= Uvalue[pos] * work1[i];
            intersect += 1;
        }
    }

    // singularity test
    if newpiv == 0 || newpiv.abs() < this.abstol {
        status = BASICLU_ERROR_singular_update;
        return status;
    }

    // stability measure
    piverr = (newpiv - xtbl * oldpiv).abs();

    // Insert spike //

    // calculate bound on file growth
    let mut grow = 0;
    for pos in Ubegin[m]..(Ubegin[m] + nz_spike) {
        i = Uindex[pos];
        assert_ne!(i, ipivot);
        j = qmap[i];
        jnext = Wflink[j];
        if Wend[j] == Wbegin[jnext] {
            nz = Wend[j] - Wbegin[j];
            grow += nz + 1; // row including spike entry
            grow += stretch * (nz + 1) + pad; // extra room
        }
    }

    // reallocate if necessary
    room = Wend[m] - Wbegin[m];
    if grow > room {
        this.addmemW = grow - room;
        status = BASICLU_REALLOCATE;
        return status;
    }

    // remove column jpivot from row file
    nz = 0;
    // for (pos = Ubegin[ipivot]; (i = Uindex[pos]) >= 0; pos++)
    pos = Ubegin[ipivot];
    while Uindex[pos] >= 0 {
        i = Uindex[pos];
        j = qmap[i];
        // end = Wend[j]--;
        end = Wend[j];
        Wend[j] -= 1;
        where_ = find(jpivot, Windex, Wbegin[j], end);
        assert!(where_ < end);
        Windex[where_] = Windex[end - 1];
        Wvalue[where_] = Wvalue[end - 1];
        nz += 1;
        pos += 1;
    }
    Unz -= nz;

    // erase column jpivot in column file
    // for (pos = Ubegin[ipivot]; Uindex[pos] >= 0; pos++)
    pos = Ubegin[ipivot];
    while Uindex[pos] >= 0 {
        Uindex[pos] = GAP;
        pos += 1;
    }

    // set column pointers to spike, chop off diagonal
    Ubegin[ipivot] = Ubegin[m];
    Ubegin[m] += nz_spike;
    // Uindex[Ubegin[m]++] = GAP;
    Uindex[Ubegin[m]] = GAP;
    Ubegin[m] += 1;

    // insert spike into row file
    // for (pos = Ubegin[ipivot]; (i = Uindex[pos]) >= 0; pos++)
    pos = Ubegin[ipivot];
    while Uindex[pos] >= 0 {
        i = Uindex[pos];
        j = qmap[i];
        jnext = Wflink[j];
        if Wend[j] == Wbegin[jnext] {
            nz = Wend[j] - Wbegin[j];
            room = 1 + stretch * (nz + 1) + pad;
            lu_file_reappend(j, m, Wbegin, Wend, Wflink, Wblink, Windex, Wvalue, room);
        }
        // end = Wend[j]++;
        end = Wend[j];
        Wend[j] += 1;
        Windex[end] = jpivot;
        Wvalue[end] = Uvalue[pos];
        pos += 1;
    }
    Unz += nz_spike;

    // insert diagonal
    col_pivot[jpivot] = spike_diag;
    row_pivot[ipivot] = spike_diag;

    // Test triangularity //

    if have_diag != 0 {
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
            nreach = nz_roweta + 1;
            row_reach = iwork1;
            col_reach = iwork2;
            row_reach[0] = ipivot;
            col_reach[0] = jpivot;
            pos = Rbegin[nforrest];
            for n in 1..nreach {
                i = Lindex[pos];
                pos += 1;
                row_reach[n] = i;
                col_reach[n] = qmap[i];
            }
            this.nsymperm_total += 1;
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
        // lu_int *path = iwork1, top;
        let path = iwork1;
        // lu_int *reach = iwork2, rtop;
        let reach = iwork2;
        // lu_int *pstack = (void *) work1;
        let pstack = work1;

        let top = bfs_path(m, jpivot, Wbegin, Wend, Windex, path, marked, iwork2);
        assert!(top < m - 1);
        assert_eq!(path[top], jpivot);

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
        let mut istriangular = true;
        let mut rtop = m;
        // M = ++this.marker;
        this.marker += 1;
        M = this.marker;
        // for (t = top; t < m-1 && istriangular; t++)
        for t in top..m - 1 {
            if !istriangular {
                break;
            }
            j = path[t];
            jnext = path[t + 1];
            where_ = find(jnext, Windex, Wbegin[j], Wend[j]);
            assert!(where_ < Wend[j]);
            Windex[where_] = j; // take out for a moment
            rtop = lu_dfs(
                j,
                Wbegin,
                Some(Wend),
                Windex,
                rtop,
                reach,
                pstack,
                marked,
                M,
            );
            assert(reach[rtop] == j);
            reach[rtop] = jnext;
            Windex[where_] = jnext; /* restore */
            istriangular = marked[jnext] != M;
        }

        // Part 2b:
        //
        // If the matrix looks triangular so far, then also mark the reach of
        // the final path node, which is reach(jpivot) in U[pmap_new,:].
        // U is then permuted triangular iff the combined reach does not
        // intersect the spike pattern except in the final path index.
        if istriangular {
            j = path[m - 1];
            rtop = lu_dfs(
                j,
                Wbegin,
                Some(Wend),
                Windex,
                rtop,
                reach,
                pstack,
                marked,
                M,
            );
            assert_eq!(reach[rtop], j);
            reach[rtop] = jpivot;
            marked[j] -= 1; // unmark for a moment
                            // for (pos = Ubegin[ipivot]; (i = Uindex[pos]) >= 0; pos++)
            pos = Ubegin[ipivot];
            while Uindex[pos] >= 0 {
                i = Uindex[pos];
                if marked[qmap[i]] == M {
                    istriangular = false;
                }
                pos += 1;
            }
            marked[j] += 1; /* restore */
        }

        // If U is permuted triangular, then permute to zero-free diagonal.
        // Set up row_reach[0..nreach-1] and col_reach[0..nreach-1] for
        // updating the permutations below. The column reach is the combined
        // reach of the path nodes. The row reach is is given through pmap.
        if istriangular {
            let nswap = m - top - 1;
            permute(this, path + top, nswap);
            Unz -= 1;
            assert_eq!(reach[rtop], jpivot);
            col_reach = reach + rtop; /* stored in iwork2 */
            row_reach = iwork1 + rtop;
            nreach = m - rtop;
            for n in 0..nreach {
                row_reach[n] = pmap[col_reach[n]];
            }
        }
    }

    // Forrest-Tomlin update //

    if !istriangular {
        // remove row ipivot from column file
        // for (pos = Wbegin[jpivot]; pos < Wend[jpivot]; pos++)
        pos = Wbegin[jpivot];
        while pos < Wend[jpivot] {
            j = Windex[pos];
            assert_ne!(j, jpivot);
            where_ = -1;
            // for (end = Ubegin[pmap[j]]; (i = Uindex[end]) >= 0; end++)
            end = Ubegin[pmap[j]];
            while Uindex[end] >= 0 {
                i = Uindex[end];
                if i == ipivot {
                    where_ = end;
                }
                end += 1;
            }
            assert!(where_ >= 0);
            Uindex[where_] = Uindex[end - 1];
            Uvalue[where_] = Uvalue[end - 1];
            Uindex[end - 1] = -1;
            Unz -= 1;
            pos += 1;
        }

        // remove row ipivot from row file
        Wend[jpivot] = Wbegin[jpivot];

        // replace pivot
        col_pivot[jpivot] = newpiv;
        row_pivot[ipivot] = newpiv;
        this.min_pivot = f64::min(this.min_pivot, newpiv.abs());
        this.max_pivot = f64::max(this.max_pivot, newpiv.abs());

        // drop zeros from row eta; update max entry of row etas
        nz = 0;
        put = Rbegin[nforrest];
        let mut max_eta = 0.0;
        for pos in put..Rbegin[nforrest + 1] {
            if Lvalue[pos] {
                max_eta = f64::max(max_eta, Lvalue[pos].abs());
                Lindex[put] = Lindex[pos];
                Lvalue[put] = Lvalue[pos];
                put += 1;
                nz += 1;
            }
        }
        Rbegin[nforrest + 1] = put;
        this.Rnz += nz;
        this.max_eta = f64::max(this.max_eta, max_eta);

        // prepare permutation update
        nreach = 1;
        row_reach = &ipivot;
        col_reach = &jpivot;
        this.nforrest += 1;
        this.nforrest_total += 1;
    }

    // Update permutations //

    if this.pivotlen + nreach > 2 * m {
        lu_garbage_perm(this);
    }

    // append row indices row_reach[0..nreach-1] to end of pivot sequence
    put = this.pivotlen;
    for n in 0..nreach {
        pivotrow[put] = row_reach[n];
        put += 1;
    }

    // append col indices col_reach[0..nreach-1] to end of pivot sequence
    put = this.pivotlen;
    for n in 0..nreach {
        pivotcol[put] = col_reach[n];
        put += 1;
    }

    this.pivotlen += nreach;

    // Clean up //

    // compress U if used memory is shrinked sufficiently
    used = Ubegin[m];
    if used - Unz - m > this.compress_thres * used {
        nz = compress_packed(m, Ubegin, Uindex, Uvalue);
        assert_eq!(nz, Unz);
    }

    // compress W if used memory is shrinked suficiently
    used = Wbegin[m];
    need = Unz + stretch * Unz + m * pad;
    if (used - need) > this.compress_thres * used {
        nz = lu_file_compress(m, Wbegin, Wend, Wflink, Windex, Wvalue, stretch, pad);
        assert_eq!(nz, Unz);
    }

    let elapsed = tic.elapsed().as_secs_f64();
    this.time_update += elapsed;
    this.time_update_total += elapsed;
    this.pivot_error = piverr / (1.0 + newpiv.abs());
    this.Unz = Unz;
    this.btran_for_update = -1;
    this.ftran_for_update = -1;
    this.update_cost_numer += nz_roweta;
    this.nupdate += 1;
    this.nupdate_total += 1;

    if cfg!(feature = "debug_extra") {
        let (mut col, mut row) = (-1, -1);
        check_consistency(this, &mut col, &mut row);
        assert!(col < 0 && row < 0);
    }

    status
}
