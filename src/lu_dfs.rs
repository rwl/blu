// Copyright (C) 2016-2018  ERGO-Code
//
// Depth first search in a graph.

use crate::basiclu::lu_int;

/// compute reach(i) in a graph by depth first search
///
/// @begin, @end, @index define the graph. When end is not NULL, then node j
/// has neighbours index[begin[j]..end[j]-1]. When end is NULL, then the
/// neighbour list is terminated by a negative index.
///
/// On return xi[newtop..top-1] hold reach(i) in topological order; newtop is
/// the function return value. Nodes that were already marked are excluded from
/// the reach.
///
/// @pstack is size m workspace (m the number of nodes in the graph); the
/// contents of pstack is undefined on entry/return.
///
/// @marked is size m array. Node j is marked iff marked[j] == M.
/// On return nodes xi[newtop..top-1] are marked.
///
/// If node i is marked on entry, the function does nothing.
pub(crate) fn lu_dfs(
    i: lu_int,
    begin: &[lu_int],
    end: Option<&[lu_int]>,
    index: &[lu_int],
    top: lu_int,
    xi: &mut [lu_int],
    pstack: &mut [lu_int],
    marked: &[lu_int],
    M: lu_int,
) -> lu_int {
    if marked[i] == M {
        return top;
    }

    if let Some(end) = end {
        dfs_end(i, begin, end, index, top, xi, pstack, marked, M)
    } else {
        dfs(i, begin, index, top, xi, pstack, marked, M)
    }
}

// adapted from T. Davis, CSPARSE
fn dfs_end(
    mut i: lu_int,
    begin: &[lu_int],
    end: &[lu_int],
    index: &[lu_int],
    mut top: lu_int,
    xi: &mut [lu_int],
    pstack: &mut [lu_int],
    marked: &[lu_int],
    M: lu_int,
) -> lu_int {
    let mut head = 0;
    assert_ne!(marked[i], M);

    xi[0] = i;
    while head >= 0 {
        i = xi[head];
        if marked[i] != M {
            // node i has not been visited
            marked[i] = M;
            pstack[head] = begin[i];
        }
        let mut done = 1;
        // continue dfs at node i
        for p in pstack[head]..end[i] {
            let inext = index[p];
            if marked[inext] == M {
                continue; // skip visited node
            }
            pstack[head] = p + 1;
            // xi[++head] = inext; /* start dfs at node inext */
            head += 1;
            xi[head] = inext; // start dfs at node inext
            done = 0;
            break;
        }
        if done != 0 {
            // node i has no unvisited neighbours
            head -= 1;
            // xi[--top] = i;
            top -= 1;
            xi[top] = i;
        }
    }

    return top;
}

// adapted from T. Davis, CSPARSE
fn dfs(
    mut i: lu_int,
    begin: &[lu_int],
    index: &[lu_int],
    mut top: lu_int,
    xi: &mut [lu_int],
    pstack: &mut [lu_int],
    marked: &[lu_int],
    M: lu_int,
) -> lu_int {
    let mut head = 0;
    assert_ne!(marked[i], M);

    xi[0] = i;
    while head >= 0 {
        i = xi[head];
        if marked[i] != M {
            // node i has not been visited
            marked[i] = M;
            pstack[head] = begin[i];
        }
        let mut done = 1;
        // continue dfs at node i
        // for (p = pstack[head]; (inext = index[p]) >= 0; p++)
        let mut p = pstack[head];
        while index[p] >= 0 {
            let inext = index[p];
            if marked[inext] == M {
                continue; // skip visited node
            }
            pstack[head] = p + 1;
            // xi[++head] = inext; /* start dfs at node inext */
            head += 1;
            xi[head] = inext; // start dfs at node inext
            done = 0;
            break;
            p += 1;
        }
        if done != 0 {
            // node i has no unvisited neighbours
            head -= 1;
            top -= 1;
            xi[top] = i;
        }
    }

    return top;
}
