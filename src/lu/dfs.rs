// Copyright (C) 2016-2018  ERGO-Code
//
// Depth first search in a graph.

use crate::blu::LUInt;

/// Compute `reach(i)` in a graph by depth first search.
///
/// `begin`, `end`, `index` define the graph. When `end` is not None, then node `j`
/// has neighbours `index[begin[j]..end[j]-1]`. When `end` is None, then the
/// neighbour list is terminated by a negative index.
///
/// On return `xi[newtop..top-1]` hold `reach(i)` in topological order; `newtop` is
/// the function return value. Nodes that were already marked are excluded from
/// the reach.
///
/// `pstack` is size `m` workspace (`m` the number of nodes in the graph); the
/// contents of `pstack` is undefined on entry/return.
///
/// `marked` is size m array. Node `j` is marked iff `marked[j] == m`.
/// On return nodes `xi[newtop..top-1]` are marked.
///
/// If node `i` is marked on entry, the function does nothing.
pub(crate) fn dfs(
    i: LUInt,
    begin: &[LUInt],
    end: Option<&[LUInt]>,
    index: &[LUInt],
    top: LUInt,
    xi: &mut [LUInt],
    // pstack: &mut [lu_int],
    pstack: &mut [f64],
    marked: &mut [LUInt],
    m: LUInt,
) -> LUInt {
    if marked[i as usize] == m {
        return top;
    }

    if let Some(end) = end {
        dfs_end(i, begin, end, index, top, xi, pstack, marked, m)
    } else {
        dfs_begin(i, begin, index, top, xi, pstack, marked, m)
    }
}

// adapted from T. Davis, CSPARSE
fn dfs_end(
    mut i: LUInt,
    begin: &[LUInt],
    end: &[LUInt],
    index: &[LUInt],
    mut top: LUInt,
    xi: &mut [LUInt],
    // pstack: &mut [lu_int],
    pstack: &mut [f64],
    marked: &mut [LUInt],
    m: LUInt,
) -> LUInt {
    let mut head: LUInt = 0;
    assert_ne!(marked[i as usize], m);

    xi[0] = i;
    while head >= 0 {
        i = xi[head as usize];
        if marked[i as usize] != m {
            // node i has not been visited
            marked[i as usize] = m;
            pstack[head as usize] = begin[i as usize] as f64;
        }
        let mut done = 1;
        // continue dfs at node i
        for p in (pstack[head as usize] as LUInt)..end[i as usize] {
            let inext = index[p as usize];
            if marked[inext as usize] == m {
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
fn dfs_begin(
    mut i: LUInt,
    begin: &[LUInt],
    index: &[LUInt],
    mut top: LUInt,
    xi: &mut [LUInt],
    // pstack: &mut [lu_int],
    pstack: &mut [f64],
    marked: &mut [LUInt],
    m: LUInt,
) -> LUInt {
    let mut head: LUInt = 0;
    assert_ne!(marked[i as usize], m);

    xi[0] = i;
    while head >= 0 {
        i = xi[head as usize];
        if marked[i as usize] != m {
            // node i has not been visited
            marked[i as usize] = m;
            pstack[head as usize] = begin[i as usize] as f64;
        }
        let mut done = 1;
        // continue dfs at node i
        // for (p = pstack[head]; (inext = index[p]) >= 0; p++)
        let mut p = pstack[head as usize] as LUInt;
        while index[p as usize] >= 0 {
            let inext = index[p as usize];
            if marked[inext as usize] == m {
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
