use crate::basiclu::*;
use crate::lu_file::{lu_file_diff, lu_file_empty};
use crate::lu_internal::LU;
use crate::lu_list::{lu_list_add, lu_list_init, lu_list_move};

/// The bump is composed of rows i and columns j for which pinv[i] < 0 and
/// qinv[j] < 0. For the factorization, the bump is stored in Windex, Wvalue
/// columnwise and additionally the nonzero pattern rowwise:
///
///  Wbegin[j]   points to the first element in column j.
///  Wend[j]     points to one past the last element in colum j.
///  Wbegin[m+i] points to the first element in row i.
///  Wend[m+i]   points to one past the last element in row i.
///
///  Wflink, Wblink hold the 2*m lines in a double linked list in memory order.
///
/// When a row or column is empty, then Wbegin == Wend. In the rowwise storage
/// the entries in Wvalue are undefined.
///
/// The Markowitz search requires double linked lists of columns with equal
/// column counts and rows with equal row counts:
///
///  colcount_flink, colcount_blink
///  rowcount_flink, rowcount_blink
///
/// They organize m elements (cols/rows) in m+2 lists. Column j is in list
/// 0 <= nz <= m when it has nz nonzeros in the active submatrix. Row i can
/// alternatively be in list m+1 to exclude it temporarily from the search.
/// A column/row not in the active submatrix is not in any list.
///
/// The Markowitz search also requires the maximum in each column of the
/// active submatrix. For column j the maximum is stored in col_pivot[j].
/// When j becomes pivot column, the maximum is replaced by the pivot element.
///
/// Return:
///
///  BASICLU_REALLOCATE  require more memory in W
///  BASICLU_OK
pub(crate) fn lu_setup_bump(
    lu: &mut LU,
    b_begin: &[LUInt],
    b_end: &[LUInt],
    b_i: &[LUInt],
    b_x: &[f64],
) -> LUInt {
    let m = lu.m;
    let rank = lu.rank;
    let w_mem = lu.w_mem;
    let b_nz = lu.matrix_nz;
    let l_nz = lu.l_begin_p[rank as usize] - rank;
    let u_nz = lu.u_begin[rank as usize];
    let abstol = lu.abstol;
    let pad = lu.pad;
    let stretch = lu.stretch;
    let colcount_flink = &mut lu.colcount_flink;
    let colcount_blink = &mut lu.colcount_blink;
    let rowcount_flink = &mut lu.rowcount_flink;
    let rowcount_blink = &mut lu.rowcount_blink;
    let pinv = &lu.pinv;
    let qinv = &lu.qinv;
    // let Wbegin = &mut lu.Wbegin;
    // let Wend = &mut lu.Wend;
    // let Wbegin2 = Wbegin + m; /* alias for row file */
    // let Wend2 = Wend + m;
    let (w_begin, w_begin2) = lu.w_begin.split_at_mut(m as usize);
    let (w_end, w_end2) = lu.w_end.split_at_mut(m as usize);
    let w_flink = &mut lu.w_flink;
    let w_blink = &mut lu.w_blink;
    let w_index = &mut lu.w_index;
    let w_value = &mut lu.w_value;
    let colmax = &mut lu.col_pivot;
    let iwork0 = &mut lu.iwork0;

    let mut bump_nz = b_nz - l_nz - u_nz - rank; // will change if columns are dropped

    // lu_int i, j, pos, put, cnz, rnz, need, min_rownz, min_colnz;
    // double cmx;
    let mut min_rownz = 0;
    let mut min_colnz = 0;

    assert!(l_nz >= 0);
    assert!(u_nz >= 0);
    assert!(bump_nz >= 0);
    #[cfg(feature = "debug")]
    for i in 0..m {
        assert_eq!(iwork0[i], 0);
    }

    // Calculate memory and reallocate. For each row/column with nz nonzeros
    // add stretch*nz+pad elements extra space for fill-in.
    let need = bump_nz + (stretch as LUInt) * bump_nz + (m - rank) * pad;
    let need = 2 * need; // rowwise + columnwise
    if need > w_mem {
        lu.addmem_w = need - w_mem;
        return BASICLU_REALLOCATE;
    }

    lu_file_empty(2 * m, w_begin, w_end, w_flink, w_blink, w_mem);

    // Build columnwise storage. Build row counts in iwork0.
    lu_list_init(
        colcount_flink,
        colcount_blink,
        m,
        m + 2,
        Some(&mut min_colnz),
    );
    let mut put = 0;
    for j in 0..m {
        if qinv[j as usize] >= 0 {
            continue;
        }
        let mut cnz = 0; // count nz per column
        let mut cmx = 0.0; // find column maximum
        for pos in b_begin[j as usize]..b_end[j as usize] {
            let i = b_i[pos as usize] as usize;
            if pinv[i] >= 0 {
                continue;
            }
            cmx = f64::max(cmx, b_x[pos as usize].abs());
            cnz += 1;
        }
        if cmx == 0.0 || cmx < abstol {
            // Leave column of active submatrix empty.
            colmax[j as usize] = 0.0;
            lu_list_add(
                j,
                0,
                colcount_flink,
                colcount_blink,
                m,
                Some(&mut min_colnz),
            );
            bump_nz -= cnz;
        } else {
            // Copy column into active submatrix.
            colmax[j as usize] = cmx;
            lu_list_add(
                j,
                cnz,
                colcount_flink,
                colcount_blink,
                m,
                Some(&mut min_colnz),
            );
            w_begin[j as usize] = put;
            for pos in b_begin[j as usize]..b_end[j as usize] {
                let i = b_i[pos as usize];
                if pinv[i as usize] >= 0 {
                    continue;
                }
                w_index[put as usize] = i;
                w_value[put as usize] = b_x[pos as usize];
                put += 1;
                iwork0[i as usize] += 1;
            }
            w_end[j as usize] = put;
            put += (stretch as LUInt) * cnz + pad;
            // reappend line to list end
            lu_list_move(j, 0, w_flink, w_blink, 2 * m, None);
        }
    }

    //  Build rowwise storage (pattern only).
    lu_list_init(
        rowcount_flink,
        rowcount_blink,
        m,
        m + 2,
        Some(&mut min_rownz),
    );
    for i in 0..m {
        // set row pointers
        if pinv[i as usize] >= 0 {
            continue;
        }
        let rnz = iwork0[i as usize];
        iwork0[i as usize] = 0;
        lu_list_add(
            i,
            rnz,
            rowcount_flink,
            rowcount_blink,
            m,
            Some(&mut min_rownz),
        );
        w_begin2[i as usize] = put;
        w_end2[i as usize] = put;
        put += rnz;
        // reappend line to list end
        lu_list_move(m + i, 0, w_flink, w_blink, 2 * m, None);
        put += (stretch as LUInt) * rnz + pad;
    }
    for j in 0..m {
        // fill rows
        for pos in w_begin[j as usize]..w_end[j as usize] {
            let i = w_index[pos as usize] as usize;
            w_index[w_end2[i] as usize] = j;
            w_end2[i] += 1;
        }
    }
    w_begin[(2 * m) as usize] = put; // set beginning of free space
    assert!(w_begin[(2 * m) as usize] <= w_end[(2 * m) as usize]);

    assert_eq!(
        lu_file_diff(m, w_begin, w_end, w_begin2, w_end2, w_index, None),
        0
    );
    assert_eq!(
        lu_file_diff(m, w_begin2, w_end2, w_begin, w_end, w_index, None),
        0
    );

    #[cfg(feature = "debug")]
    for i in 0..m {
        assert_eq!(iwork0[i], 0);
    }

    lu.bump_nz = bump_nz;
    lu.bump_size = m - rank;
    lu.min_colnz = min_colnz;
    lu.min_rownz = min_rownz;

    BASICLU_OK
}
