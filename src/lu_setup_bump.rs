use crate::basiclu::*;
use crate::lu_file::lu_file_empty;
use crate::lu_internal::lu;
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
    this: &mut lu,
    Bbegin: &[lu_int],
    Bend: &[lu_int],
    Bi: &[lu_int],
    Bx: &[f64],
) -> lu_int {
    let m = this.m;
    let rank = this.rank;
    let Wmem = this.Wmem;
    let Bnz = this.matrix_nz;
    let Lnz = this.Lbegin_p[rank] - rank;
    let Unz = this.Ubegin[rank];
    let abstol = this.abstol;
    let pad = this.pad;
    let stretch = this.stretch;
    let colcount_flink = this.colcount_flink;
    let colcount_blink = this.colcount_blink;
    let rowcount_flink = this.rowcount_flink;
    let rowcount_blink = this.rowcount_blink;
    /*const*/
    let pinv = this.pinv;
    /*const*/
    let qinv = this.qinv;
    let Wbegin = this.Wbegin;
    let Wend = this.Wend;
    let Wbegin2 = Wbegin + m; /* alias for row file */
    let Wend2 = Wend + m;
    let Wflink = this.Wflink;
    let Wblink = this.Wblink;
    let Windex = this.Windex.unwrap();
    let Wvalue = this.Wvalue.unwrap();
    let colmax = this.col_pivot;
    let iwork0 = this.iwork0;

    let bump_nz = Bnz - Lnz - Unz - rank; // will change if columns are dropped
                                          // lu_int i, j, pos, put, cnz, rnz, need, min_rownz, min_colnz;
                                          // double cmx;

    assert!(Lnz >= 0);
    assert!(Unz >= 0);
    assert!(bump_nz >= 0);
    #[cfg(feature = debug)]
    for i in 0..m {
        assert_eq!(iwork0[i], 0);
    }

    // Calculate memory and reallocate. For each row/column with nz nonzeros
    // add stretch*nz+pad elements extra space for fill-in.
    need = bump_nz + stretch * bump_nz + (m - rank) * pad;
    need = 2 * need; /* rowwise + columnwise */
    if need > Wmem {
        this.addmemW = need - Wmem;
        return BASICLU_REALLOCATE;
    }

    lu_file_empty(2 * m, Wbegin, Wend, Wflink, Wblink, Wmem);

    // Build columnwise storage. Build row counts in iwork0.
    lu_list_init(colcount_flink, colcount_blink, m, m + 2, &min_colnz);
    put = 0;
    for j in 0..m {
        if qinv[j] >= 0 {
            continue;
        }
        cnz = 0; // count nz per column
        cmx = 0.0; // find column maximum
        for pos in Bbegin[j]..Bend[j] {
            i = Bi[pos];
            if pinv[i] >= 0 {
                continue;
            }
            cmx = fmax(cmx, fabs(Bx[pos]));
            cnz += 1;
        }
        if cmx == 0.0 || cmx < abstol {
            // Leave column of active submatrix empty.
            colmax[j] = 0.0;
            lu_list_add(j, 0, colcount_flink, colcount_blink, m, &min_colnz);
            bump_nz -= cnz;
        } else {
            // Copy column into active submatrix.
            colmax[j] = cmx;
            lu_list_add(j, cnz, colcount_flink, colcount_blink, m, &min_colnz);
            Wbegin[j] = put;
            for pos in Bbegin[j]..Bend[j] {
                i = Bi[pos];
                if pinv[i] >= 0 {
                    continue;
                }
                Windex[put] = i;
                Wvalue[put] = Bx[pos];
                put += 1;
                iwork0[i] += 1;
            }
            Wend[j] = put;
            put += stretch * cnz + pad;
            // reappend line to list end
            lu_list_move(j, 0, Wflink, Wblink, 2 * m, NULL);
        }
    }

    //  Build rowwise storage (pattern only).
    lu_list_init(rowcount_flink, rowcount_blink, m, m + 2, &min_rownz);
    for i in 0..m {
        // set row pointers
        if pinv[i] >= 0 {
            continue;
        }
        rnz = iwork0[i];
        iwork0[i] = 0;
        lu_list_add(i, rnz, rowcount_flink, rowcount_blink, m, &min_rownz);
        Wbegin2[i] = put;
        Wend2[i] = put;
        put += rnz;
        // reappend line to list end
        lu_list_move(m + i, 0, Wflink, Wblink, 2 * m, NULL);
        put += stretch * rnz + pad;
    }
    for j in 0..m {
        // fill rows
        for pos in Wbegin[j]..Wend[j] {
            i = Windex[pos];
            Windex[Wend2[i]] = j;
            Wend2[i] += 1;
        }
    }
    Wbegin[2 * m] = put; // set beginning of free space
    assert!(Wbegin[2 * m] <= Wend[2 * m]);

    assert!(lu_file_diff(m, Wbegin, Wend, Wbegin2, Wend2, Windex, NULL) == 0);
    assert!(lu_file_diff(m, Wbegin2, Wend2, Wbegin, Wend, Windex, NULL) == 0);

    #[cfg(feature = debug)]
    for i in 0..m {
        assert_eq!(iwork0[i], 0);
    }

    this.bump_nz = bump_nz;
    this.bump_size = m - rank;
    this.min_colnz = min_colnz;
    this.min_rownz = min_rownz;

    BASICLU_OK
}
