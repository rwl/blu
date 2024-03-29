// Copyright (C) 2016-2018 ERGO-Code
// Copyright (C) 2022-2023 Richard Lincoln

use crate::LUInt;
use std::usize;

// Implementation of doubly linked lists (see [1] section 5.5)
//
// Maintain nelem elements in `nlist` doubly linked lists. Each element can belong
// to zero or one list at a time.
//
// The implementation uses arrays
//
//     flink[0..nelem+nlist-1],
//     blink[0..nelem+nlist-1].
//
// In each array, the leading `nelem` entries store links, the trailing `nlist`
// entries store heads. That is, for 0 <= i < nelem and 0 <= j < nlist:
//
//     flink[i]        next element in the list containing element i
//     blink[i]        previous element in the list containing element i
//     flink[nelem+j]  first element in list j
//     blink[nelem+j]  last element in list j
//
// The forward link of the last element in a list points to its flink-head. The
// backward link of the first element in a list points to its blink-head. For
// empty lists the heads point to themselves. When an element is not in any list
// its links point to itself.
//
// Optionally the quantity `min_list` >= 1 can be updated such that lists
// `1..min_list-1` are empty. Notice that list 0 is not covered by `min_list`.
//
//    [1] Istvan Maros, Computational Techniques of the Simplex Method

// Initialize all lists to empty.
pub(crate) fn list_init(
    flink: &mut [LUInt],
    blink: &mut [LUInt],
    nelem: usize,
    nlist: usize,
    min_list: Option<&mut usize>,
) {
    for i in 0..nelem + nlist {
        flink[i] = i as LUInt;
        blink[i] = i as LUInt;
    }
    if let Some(min_list) = min_list {
        *min_list = usize::max(1, nlist);
    }
}

// Add element `elem` to list `list`. `elem` must not be in any list already.
// If `list` > 0 and `min_list` != None, update `*min_list = min(*min_list, list)`.
pub(crate) fn list_add(
    elem: usize,
    list: usize,
    flink: &mut [LUInt],
    blink: &mut [LUInt],
    nelem: usize,
    min_list: Option<&mut usize>,
) {
    // lu_int temp;
    assert_eq!(flink[elem] as usize, elem);
    assert_eq!(blink[elem] as usize, elem);
    // append elem to the end of list
    let temp = blink[nelem + list];
    blink[nelem + list] = elem as LUInt;
    blink[elem] = temp;
    flink[temp as usize] = elem as LUInt;
    flink[elem] = (nelem + list) as LUInt;
    if let Some(min_list) = min_list {
        // if (list > 0 && min_list && list < *min_list) TODO: check translation
        if list > 0 && (list as usize) < *min_list {
            *min_list = list as usize;
        }
    }
}

// Remove element `elem` from its list. If `elem` was not in a list before,
// then do nothing.
pub(crate) fn list_remove(flink: &mut [LUInt], blink: &mut [LUInt], elem: usize) {
    flink[blink[elem] as usize] = flink[elem];
    blink[flink[elem] as usize] = blink[elem];
    flink[elem] = elem as LUInt;
    blink[elem] = elem as LUInt;
}

// Remove element `elem` from its list (if in a list) and add it to list `list`.
pub(crate) fn list_move(
    elem: usize,
    list: usize,
    flink: &mut [LUInt],
    blink: &mut [LUInt],
    nelem: usize,
    min_list: Option<&mut usize>,
) {
    list_remove(flink, blink, elem);
    list_add(elem, list, flink, blink, nelem, min_list);
}

// Swap elements `e1` and `e2`, which both must be in a list. If `e1` and `e2`
// are in the same list, then their positions are swapped. If they are in
// different lists, then each is moved to the other's list.
pub(crate) fn list_swap(flink: &mut [LUInt], blink: &mut [LUInt], e1: usize, e2: usize) {
    let e1next = flink[e1] as usize;
    let e2next = flink[e2] as usize;
    let e1prev = blink[e1] as usize;
    let e2prev = blink[e2] as usize;

    assert_ne!(e1next, e1); // must be in a list
    assert_ne!(e2next, e2);

    if e1next == e2 {
        flink[e2] = e1 as LUInt;
        blink[e1] = e2 as LUInt;
        flink[e1prev] = e2 as LUInt;
        blink[e2] = e1prev as LUInt;
        flink[e1] = e2next as LUInt;
        blink[e2next] = e1 as LUInt;
    } else if e2next == e1 {
        flink[e1] = e2 as LUInt;
        blink[e2] = e1 as LUInt;
        flink[e2] = e1next as LUInt;
        blink[e1next] = e2 as LUInt;
        flink[e2prev] = e1 as LUInt;
        blink[e1] = e2prev as LUInt;
    } else {
        flink[e2] = e1next as LUInt;
        blink[e1next] = e2 as LUInt;
        flink[e2prev] = e1 as LUInt;
        blink[e1] = e2prev as LUInt;
        flink[e1prev] = e2 as LUInt;
        blink[e2] = e1prev as LUInt;
        flink[e1] = e2next as LUInt;
        blink[e2next] = e1 as LUInt;
    }
}
