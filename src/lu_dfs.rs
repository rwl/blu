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
    // pstack: &mut [lu_int],
    pstack: &mut [f64],
    marked: &mut [lu_int],
    M: lu_int,
) -> lu_int {
    if marked[i as usize] == M {
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
    // pstack: &mut [lu_int],
    pstack: &mut [f64],
    marked: &mut [lu_int],
    M: lu_int,
) -> lu_int {
    let mut head: lu_int = 0;
    assert_ne!(marked[i as usize], M);

    xi[0] = i;
    while head >= 0 {
        i = xi[head as usize];
        if marked[i as usize] != M {
            // node i has not been visited
            marked[i as usize] = M;
            pstack[head as usize] = begin[i as usize] as f64;
        }
        let mut done = 1;
        // continue dfs at node i
        for p in (pstack[head as usize] as lu_int)..end[i as usize] {
            let inext = index[p as usize];
            if marked[inext as usize] == M {
                continue; // skip visited node
            }
            pstack[head as usize] = (p + 1) as f64;
            // xi[++head] = inext; /* start dfs at node inext */
            head += 1;
            xi[head as usize] = inext; // start dfs at node inext
            done = 0;
            break;
        }
        if done != 0 {
            // node i has no unvisited neighbours
            head -= 1;
            // xi[--top] = i;
            top -= 1;
            xi[top as usize] = i;
        }
    }
    top
}

// adapted from T. Davis, CSPARSE
fn dfs(
    mut i: lu_int,
    begin: &[lu_int],
    index: &[lu_int],
    mut top: lu_int,
    xi: &mut [lu_int],
    // pstack: &mut [lu_int],
    pstack: &mut [f64],
    marked: &mut [lu_int],
    M: lu_int,
) -> lu_int {
    let mut head: lu_int = 0;
    assert_ne!(marked[i as usize], M);

    xi[0] = i;
    while head >= 0 {
        i = xi[head as usize];
        if marked[i as usize] != M {
            // node i has not been visited
            marked[i as usize] = M;
            pstack[head as usize] = begin[i as usize] as f64;
        }
        let mut done = 1;
        // continue dfs at node i
        // for (p = pstack[head]; (inext = index[p]) >= 0; p++)
        let mut p = pstack[head as usize] as lu_int;
        while index[p as usize] >= 0 {
            let inext = index[p as usize];
            if marked[inext as usize] == M {
                p += 1; // TODO: check
                continue; // skip visited node
            }
            pstack[head as usize] = (p + 1) as f64;
            // xi[++head] = inext; /* start dfs at node inext */
            head += 1;
            xi[head as usize] = inext; // start dfs at node inext
            done = 0;
            break;
        }
        if done != 0 {
            // node i has no unvisited neighbours
            head -= 1;
            top -= 1;
            xi[top as usize] = i;
        }
    }

    return top;
}
