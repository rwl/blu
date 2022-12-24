// Copyright (C) 2016-2018  ERGO-Code

use crate::basiclu::*;
use crate::lu_internal::*;

/// Purpose:
///
///     Extract the row and column permutation and the LU factors. This routine can
///     be used only after basiclu_factorize() has completed and before a call to
///     basiclu_update(). At that point the factorized form of matrix B is
///
///         B[rowperm,colperm] = L*U,
///
///     where L is unit lower triangular and U is upper triangular. If the
///     factorization was singular (rank < m), then columns colperm[rank..m-1]
///     of B have been replaced by unit columns with entry 1 in position
///     rowperm[rank..m-1].
///
///     basiclu_get_factors() is intended when the user needs direct access to the
///     matrix factors. It is not required to solve linear systems with the factors
///     (see basiclu_solve_dense() and basiclu_solve_sparse() instead).
///
/// Return:
///
///     BASICLU_ERROR_invalid_store if istore, xstore do not hold a BASICLU
///     instance. In this case xstore[BASICLU_STATUS] is not set.
///
///     Otherwise return the status code. See xstore[BASICLU_STATUS] below.
///
/// Arguments:
///
///     lu_int istore[]
///     double xstore[]
///     lu_int Li[]
///     double Lx[]
///     lu_int Ui[]
///     double Ux[]
///     lu_int Wi[]
///     double Wx[]
///
///         The BASICLU instance after basiclu_factorize() has completed.
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
///     lu_int Lcolptr[m+1]
///     lu_int Lrowidx[m+Lnz]
///     double Lvalue[m+Lnz], where Lnz = xstore[BASICLU_LNZ]
///
///         If all three arguments are not NULL, then they are filled with L in
///         compressed column form. The indices in each column are sorted with the
///         unit diagonal element at the front.
///
///         If any of the three arguments is NULL, then L is not returned
///         (this is not an error).
///
///     lu_int Ucolptr[m+1]
///     lu_int Urowidx[m+Unz]
///     double Uvalue[m+Unz], where Unz = xstore[BASICLU_UNZ]
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
///     xstore[BASICLU_STATUS]: status code.
///
///         BASICLU_OK
///
///             The requested quantities have been returned successfully.
///
///         BASICLU_ERROR_argument_missing
///
///             One or more of the mandatory pointer/array arguments are NULL.
///
///         BASICLU_ERROR_invalid_call
///
///             The BASICLU instance does not hold a fresh factorization (either
///             basiclu_factorize() has not completed or basiclu_update() has been
///             called in the meanwhile).
pub fn basiclu_get_factors(
    istore: &mut [lu_int],
    xstore: &mut [f64],
    Li: &[lu_int],
    Lx: &[f64],
    Wi: &[lu_int],
    Wx: &[f64],
    rowperm: Option<&[lu_int]>,
    colperm: Option<&[lu_int]>,
    Lcolptr: Option<&mut [lu_int]>,
    Lrowidx: Option<&mut [lu_int]>,
    Lvalue_: Option<&mut [f64]>,
    Ucolptr: Option<&mut [lu_int]>,
    Urowidx: Option<&mut [lu_int]>,
    Uvalue_: Option<&mut [f64]>,
) -> lu_int {
    let mut this = lu {
        ..Default::default()
    };

    let status = lu_load(
        &mut this, /*istore,*/ xstore, /*, Li, Lx, Ui, Ux, Wi, Wx*/
    );
    if status != BASICLU_OK {
        return status;
    }
    if this.nupdate != 0 {
        let status = BASICLU_ERROR_invalid_call;
        return lu_save(&this, /*istore,*/ xstore, status);
    }
    let m = this.m;

    if let Some(rowperm) = rowperm {
        // memcpy(rowperm, this.pivotrow, m * sizeof(lu_int));
        pivotrow!(this).copy_from_slice(rowperm);
    }
    if let Some(colperm) = colperm {
        // memcpy(colperm, this.pivotcol, m * sizeof(lu_int));
        pivotcol!(this).copy_from_slice(colperm);
    }

    if Lcolptr.is_some() && Lrowidx.is_some() && Lvalue_.is_some() {
        let Lcolptr = Lcolptr.unwrap();
        let Lrowidx = Lrowidx.unwrap();
        let Lvalue_ = Lvalue_.unwrap();

        // let Lbegin_p = &this.Lbegin_p;
        let Ltbegin_p = &Ltbegin_p!(this);
        // let Lindex = this.Lindex.as_ref().unwrap();
        // let Lvalue = this.Lvalue.as_ref().unwrap();
        let Lindex = Li;
        let Lvalue = Lx;
        let p = &p!(this);
        let colptr = &mut iwork1!(this); // size m workspace

        // L[:,k] will hold the elimination factors from the k-th pivot step.
        // First set the column pointers and store the unit diagonal elements
        // at the front of each column. Then scatter each row of L' into the
        // columnwise L so that the row indices become sorted.
        let mut put = 0;
        for k in 0..m {
            Lcolptr[k as usize] = put;
            Lrowidx[put as usize] = k;
            Lvalue_[put as usize] = 1.0;
            put += 1;
            colptr[p[k as usize] as usize] = put; // next free position in column
            put += this.Lbegin_p[(k + 1) as usize] - this.Lbegin_p[k as usize] - 1;
            // subtract 1 because internal storage uses (-1) terminators
        }
        Lcolptr[m as usize] = put;
        assert_eq!(put, this.Lnz + m);

        for k in 0..m {
            let mut pos = Ltbegin_p[k as usize];
            while Lindex[pos as usize] >= 0 {
                let i = Lindex[pos as usize];
                // put = colptr[i]++; TODO: check
                put = colptr[i as usize];
                colptr[i as usize] += 1;
                Lrowidx[put as usize] = k;
                Lvalue_[put as usize] = Lvalue[pos as usize];
                pos += 1;
            }
        }

        if cfg!(feature = "debug") {
            for k in 0..m {
                assert_eq!(colptr[p[k as usize] as usize], Lcolptr[(k + 1) as usize]);
            }
        }
    }

    if Ucolptr.is_some() && Urowidx.is_some() && Uvalue_.is_some() {
        let Ucolptr = Ucolptr.unwrap();
        let Urowidx = Urowidx.unwrap();
        let Uvalue_ = Uvalue_.unwrap();

        // let Wbegin = &this.Wbegin;
        // let Wend = &this.Wend;
        // let Windex = this.Windex.as_ref().unwrap();
        // let Wvalue = this.Wvalue.as_ref().unwrap();
        let Windex = Wi;
        let Wvalue = Wx;
        // let col_pivot = &this.xstore.col_pivot;
        let pivotcol = &pivotcol!(this);
        let colptr = &mut iwork1!(this); // size m workspace

        // U[:,k] will hold the column of B from the k-th pivot step.
        // First set the column pointers and store the pivot element at the end
        // of each column. Then scatter each row of U' into the columnwise U so
        // that the row indices become sorted.
        // memset(colptr, 0, m*sizeof(lu_int)); /* column counts */
        colptr.fill(0); // column counts
        for j in 0..m {
            for pos in this.Wbegin[j as usize]..this.Wend[j as usize] {
                colptr[Windex[pos as usize] as usize] += 1;
            }
        }
        let mut put = 0;
        for k in 0..m {
            // set column pointers
            let j = pivotcol[k as usize];
            Ucolptr[k as usize] = put;
            put += colptr[j as usize];
            colptr[j as usize] = Ucolptr[k as usize]; // next free position in column
            Urowidx[put as usize] = k;
            Uvalue_[put as usize] = this.col_pivot[j as usize];
            put += 1;
        }
        Ucolptr[m as usize] = put;
        assert_eq!(put, this.Unz + m);
        for k in 0..m {
            // scatter row k
            let j = pivotcol[k as usize];
            for pos in this.Wbegin[j as usize]..this.Wend[j as usize] {
                // put = colptr[Windex[pos]]++;  TODO: check
                put = colptr[Windex[pos as usize] as usize];
                colptr[Windex[pos as usize] as usize] += 1;
                Urowidx[put as usize] = k;
                Uvalue_[put as usize] = Wvalue[pos as usize];
            }
        }

        if cfg!(feature = "debug") {
            for k in 0..m {
                assert_eq!(
                    colptr[pivotcol[k as usize] as usize],
                    Ucolptr[(k + 1) as usize] - 1
                );
            }
            for k in 0..m {
                assert_eq!(Urowidx[Ucolptr[(k + 1) as usize] as usize - 1], k);
            }
        }
    }

    BASICLU_OK
}
