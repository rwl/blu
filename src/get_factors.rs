// Copyright (C) 2016-2018 ERGO-Code
// Copyright (C) 2022-2023 Richard Lincoln

use crate::blu::*;
use crate::lu::lu::*;

/// Extract the row and column permutation and the LU factors. This routine can
/// be used only after [`factorize()`] has completed and before a call to
/// [`update()`]. At that point the factorized form of matrix `B` is
///
///     B[rowperm,colperm] = L*U,
///
/// where `L` is unit lower triangular and `U` is upper triangular. If the
/// factorization was singular (`rank` < `m`), then columns `colperm[rank..m-1]`
/// of `B` have been replaced by unit columns with entry 1 in position
/// `rowperm[rank..m-1]`.
///
/// `get_factors()` is intended when the user needs direct access to the
/// matrix factors. It is not required to solve linear systems with the factors
/// (see [`blu_solve_dense()`] and [`blu_solve_sparse()`] instead).
///
/// Return:
///
///     BLU_ERROR_INVALID_STORE if istore, xstore do not hold a BLU
///     instance. In this case xstore[BLU_STATUS] is not set.
///
///     Otherwise return the status code. See xstore[BLU_STATUS] below.
///
/// Arguments:
///
///     LU lu
///
///         The BLU instance after factorize() has completed.
///
///     lu_int rowperm[m]
///
///         Returns the row permutation. If the row permutation is not required,
///         then NULL can be passed (this is not an error).
///
///     lu_int colperm[m]
///
///         Returns the column permutation. If the column permutation is not
///         required, then NULL can be passed (this is not an error).
///
///     lu_int l_colptr[m+1]
///     lu_int l_rowidx[m+Lnz]
///     double Lvalue[m+Lnz], where Lnz = xstore[BLU_LNZ]
///
///         If all three arguments are not NULL, then they are filled with L in
///         compressed column form. The indices in each column are sorted with the
///         unit diagonal element at the front.
///
///         If any of the three arguments is NULL, then L is not returned
///         (this is not an error).
///
///     lu_int u_colptr[m+1]
///     lu_int u_rowidx[m+Unz]
///     double Uvalue[m+Unz], where Unz = xstore[BLU_UNZ]
///
///         If all three arguments are not NULL, then they are filled with U in
///         compressed column form. The indices in each column are sorted with the
///         diagonal element at the end.
///
///         If any of the three arguments is NULL, then U is not returned
///         (this is not an error).
///
/// Info:
///
///     xstore[BLU_STATUS]: status code.
///
///         BLU_OK
///
///             The requested quantities have been returned successfully.
///
///         BLU_ERROR_ARGUMENT_MISSING
///
///             One or more of the mandatory pointer/array arguments are NULL.
///
///         BLU_ERROR_INVALID_CALL
///
///             The BLU instance does not hold a fresh factorization (either
///             factorize() has not completed or update() has been
///             called in the meanwhile).
pub fn get_factors(
    lu: &mut LU,
    rowperm: Option<&[LUInt]>,
    colperm: Option<&[LUInt]>,
    l_colptr: Option<&mut [LUInt]>,
    l_rowidx: Option<&mut [LUInt]>,
    l_value_: Option<&mut [f64]>,
    u_colptr: Option<&mut [LUInt]>,
    u_rowidx: Option<&mut [LUInt]>,
    u_value_: Option<&mut [f64]>,
) -> LUInt {
    // let status = lu.load(xstore);
    // if status != BLU_OK {
    //     return status;
    // }
    assert_eq!(lu.nupdate, 0);
    // if lu.nupdate != 0 {
    //     let status = BLU_ERROR_INVALID_CALL;
    //     return lu.save(xstore, status);
    // }
    let m = lu.m;

    if let Some(rowperm) = rowperm {
        // memcpy(rowperm, lu.pivotrow, m * sizeof(lu_int));
        pivotrow!(lu).copy_from_slice(rowperm);
    }
    if let Some(colperm) = colperm {
        // memcpy(colperm, lu.pivotcol, m * sizeof(lu_int));
        pivotcol!(lu).copy_from_slice(colperm);
    }

    if l_colptr.is_some() && l_rowidx.is_some() && l_value_.is_some() {
        let l_colptr = l_colptr.unwrap();
        let l_rowidx = l_rowidx.unwrap();
        let l_value_ = l_value_.unwrap();

        // let Lbegin_p = &lu.Lbegin_p;
        let lt_begin_p = &lt_begin_p!(lu);
        let l_index = &lu.l_index;
        let l_value = &lu.l_value;
        let p = &p!(lu);
        let colptr = &mut iwork1!(lu); // size m workspace

        // L[:,k] will hold the elimination factors from the k-th pivot step.
        // First set the column pointers and store the unit diagonal elements
        // at the front of each column. Then scatter each row of L' into the
        // columnwise L so that the row indices become sorted.
        let mut put = 0;
        for k in 0..m {
            l_colptr[k as usize] = put;
            l_rowidx[put as usize] = k;
            l_value_[put as usize] = 1.0;
            put += 1;
            colptr[p[k as usize] as usize] = put; // next free position in column
            put += lu.l_begin_p[(k + 1) as usize] - lu.l_begin_p[k as usize] - 1;
            // subtract 1 because internal storage uses (-1) terminators
        }
        l_colptr[m as usize] = put;
        assert_eq!(put, lu.l_nz + m);

        for k in 0..m {
            let mut pos = lt_begin_p[k as usize];
            while l_index[pos as usize] >= 0 {
                let i = l_index[pos as usize];
                // put = colptr[i]++; TODO: check
                put = colptr[i as usize];
                colptr[i as usize] += 1;
                l_rowidx[put as usize] = k;
                l_value_[put as usize] = l_value[pos as usize];
                pos += 1;
            }
        }

        if cfg!(feature = "debug") {
            for k in 0..m {
                assert_eq!(colptr[p[k as usize] as usize], l_colptr[(k + 1) as usize]);
            }
        }
    }

    if u_colptr.is_some() && u_rowidx.is_some() && u_value_.is_some() {
        let u_colptr = u_colptr.unwrap();
        let u_rowidx = u_rowidx.unwrap();
        let u_value_ = u_value_.unwrap();

        // let Wbegin = &lu.Wbegin;
        // let Wend = &lu.Wend;
        let w_index = &lu.w_index;
        let w_value = &lu.w_value;
        // let col_pivot = &lu.xstore.col_pivot;
        let pivotcol = &pivotcol!(lu);
        let colptr = &mut iwork1!(lu); // size m workspace

        // U[:,k] will hold the column of B from the k-th pivot step.
        // First set the column pointers and store the pivot element at the end
        // of each column. Then scatter each row of U' into the columnwise U so
        // that the row indices become sorted.
        // memset(colptr, 0, m*sizeof(lu_int)); /* column counts */
        colptr.fill(0); // column counts
        for j in 0..m {
            for pos in lu.w_begin[j as usize]..lu.w_end[j as usize] {
                colptr[w_index[pos as usize] as usize] += 1;
            }
        }
        let mut put = 0;
        for k in 0..m {
            // set column pointers
            let j = pivotcol[k as usize];
            u_colptr[k as usize] = put;
            put += colptr[j as usize];
            colptr[j as usize] = u_colptr[k as usize]; // next free position in column
            u_rowidx[put as usize] = k;
            u_value_[put as usize] = lu.col_pivot[j as usize];
            put += 1;
        }
        u_colptr[m as usize] = put;
        assert_eq!(put, lu.u_nz + m);
        for k in 0..m {
            // scatter row k
            let j = pivotcol[k as usize];
            for pos in lu.w_begin[j as usize]..lu.w_end[j as usize] {
                // put = colptr[Windex[pos]]++;  TODO: check
                put = colptr[w_index[pos as usize] as usize];
                colptr[w_index[pos as usize] as usize] += 1;
                u_rowidx[put as usize] = k;
                u_value_[put as usize] = w_value[pos as usize];
            }
        }

        if cfg!(feature = "debug") {
            for k in 0..m {
                assert_eq!(
                    colptr[pivotcol[k as usize] as usize],
                    u_colptr[(k + 1) as usize] - 1
                );
            }
            for k in 0..m {
                assert_eq!(u_rowidx[u_colptr[(k + 1) as usize] as usize - 1], k);
            }
        }
    }

    BLU_OK
}
