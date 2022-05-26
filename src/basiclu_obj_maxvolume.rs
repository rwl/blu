// Copyright (C) 2016-2017  ERGO-Code

use crate::basiclu::*;
use crate::basiclu_object::basiclu_object;
use crate::{basiclu_obj_factorize, basiclu_obj_solve_for_update, basiclu_obj_update};

/// Purpose:
///
///     Make one pass over the columns of a rectangular (ncol >= nrow) matrix and
///     pivot each nonbasic column into the basis when it increases the volume (i.e.
///     the absolute value of the determinant) of the basis matrix. This is one main
///     loop of the "maximum volume" algorithm described in [1,2].
///
///     [1] C. T. Pan, "On the existence and computation of rank-revealing LU
///         factorizations". Linear Algebra Appl., 316(1-3), pp. 199-222, 2000
///
///     [2] S. A. Goreinov, I. V. Oseledets, D. V. Savostyanov, E. E. Tyrtyshnikov,
///         N. L. Zamarashkin, "How to find a good submatrix". In "Matrix methods:
///         theory, algorithms and applications", pp. 247-256. World Sci. Publ.,
///         Hackensack, NJ, 2010.
///
/// Return:
///
///     BASICLU_ERROR_invalid_argument when volumetol is less than 1.0.
///     BASICLU_ERROR_out_of_memory when memory allocation in this function failed.
///
///     The return code from a basiclu_obj_* function called when not BASICLU_OK.
///     (Note that BASICLU_WARNING_singular_matrix means that the algorithm failed.)
///
///     BASICLU_OK otherwise.
///
/// Arguments:
///
///     struct basiclu_object *obj
///
///         Pointer to an initialized BASICLU object. The dimension of the object
///         specifies the number of rows of the matrix.
///
///     lu_int ncol
///     const lu_int Ap[ncol+1]
///     const lu_int Ai[]
///     const double Ax[]
///
///         Matrix A in compressed sparse column format. Column j contains elements
///
///             Ai[Ap[j] .. Ap[j+1]-1], Ax[Ap[j] .. Ap[j+1]-1].
///
///         The columns must not contain duplicate row indices. The row indices per
///         column need not be sorted.
///
///    lu_int basis[nrow]
///
///         On entry holds the column indices of A that form the initial basis. On
///         return holds the updated basis. A basis defines a square nonsingular
///         submatrix of A. If the initial basis is (numerically) singular, then the
///         initial LU factorization will fail and BASICLU_WARNING_singular_matrix
///         is returned.
///
///    lu_int isbasic[ncol]
///
///         This array must be consistent with basis[] on entry, and is consistent
///         on return. isbasic[j] must be nonzero iff column j appears in the basis.
///
///    double volumetol
///
///         A column is pivoted into the basis when it increases the absolute value
///         of the determinant of the basis matrix by more than a factor volumetol.
///         This parameter must be >= 1.0. In pratcice typical values are 2.0, 10.0
///         or even 100.0. The closer the tolerances to 1.0, the more basis changes
///         will usually be necessary to find a maximum volume basis for this
///         tolerance (using repeated calls to basiclu_obj_maxvolume(), see below).
///
///     lu_int *p_nupdate
///
///         On return *p_nupdate holds the number of basis updates performed. When
///         this is zero and BASICLU_OK is returned, then the volume of the initial
///         basis is locally (within one basis change) maximum up to a factor
///         volumetol. To find such a basis, basiclu_obj_maxvolume() must be called
///         repeatedly starting from an arbitrary basis until *p_nupdate is zero.
///         This will happen eventually because each basis update strictly increases
///         the volume of the basis matrix. Hence a basis cannot repeat.
///
///         p_nupdate can be NULL, in which case it is not accessed. This is not an
///         error condition. The number of updates performed can be obtained as the
///         increment to obj.xstore[BASICLU_NUPDATE_TOTAL] caused by the call to
///         basiclu_obj_maxvolume().
pub fn basiclu_obj_maxvolume(
    obj: &mut basiclu_object,
    ncol: lu_int,
    Ap: &[lu_int],
    Ai: &[lu_int],
    Ax: &[f64],
    basis: &mut [lu_int],
    isbasic: &mut [lu_int],
    volumetol: f64,
    p_nupdate: Option<&mut lu_int>,
) -> lu_int {
    // one pass over columns of A doing basis updates
    //
    // For each column a_j not in B, compute lhs = B^{-1}*a_j and find the maximum
    // entry lhs[imax]. If it is bigger than @volumetol in absolute value, then
    // replace position imax of the basis by index j. On return *p_nupdate is the
    // number of basis updates done.

    let mut nupdate = 0;
    let mut status = BASICLU_OK;

    if volumetol < 1.0 {
        status = BASICLU_ERROR_invalid_argument;
        // goto_cleanup();
        if let Some(p_nupdate) = p_nupdate {
            *p_nupdate = nupdate;
        }
        return status;
    }

    // Compute initial factorization.
    status = factorize(obj, Ap, Ai, Ax, basis);
    if status != BASICLU_OK {
        // goto_cleanup;
        if let Some(p_nupdate) = p_nupdate {
            *p_nupdate = nupdate;
        }
        return status;
    }

    for j in 0..ncol {
        if isbasic[j as usize] != 0 {
            continue;
        }

        // compute B^{-1}*a_j
        let nzrhs = Ap[(j + 1) as usize] - Ap[j as usize];
        let begin = Ap[j as usize] as usize;
        status = basiclu_obj_solve_for_update(obj, nzrhs, &Ai[begin..], Some(&Ax[begin..]), 'N', 1);
        if status != BASICLU_OK {
            // goto_cleanup;
            if let Some(p_nupdate) = p_nupdate {
                *p_nupdate = nupdate;
            }
            return status;
        }

        // Find the maximum entry.
        let mut xmax = 0.0;
        let mut xtbl = 0.0;
        let mut imax = 0;
        for k in 0..obj.nzlhs {
            let i = obj.ilhs[k as usize];
            if obj.lhs[i as usize].abs() > xmax {
                xtbl = obj.lhs[i as usize];
                xmax = xtbl.abs();
                imax = i;
            }
        }
        let imax_vec = vec![imax];

        if xmax <= volumetol {
            continue;
        }

        // Update basis.
        isbasic[basis[imax as usize] as usize] = 0;
        isbasic[j as usize] = 1;
        basis[imax as usize] = j;
        nupdate += 1;

        // Prepare to update factorization.
        status = basiclu_obj_solve_for_update(obj, 0, &imax_vec, None, 'T', 0);
        if status != BASICLU_OK {
            // goto_cleanup;
            if let Some(p_nupdate) = p_nupdate {
                *p_nupdate = nupdate;
            }
            return status;
        }

        status = basiclu_obj_update(obj, xtbl);
        if status != BASICLU_OK {
            // goto_cleanup;
            if let Some(p_nupdate) = p_nupdate {
                *p_nupdate = nupdate;
            }
            return status;
        }

        status = refactorize_if_needed(obj, Ap, Ai, Ax, basis);
        if status != BASICLU_OK {
            // goto_cleanup;
            if let Some(p_nupdate) = p_nupdate {
                *p_nupdate = nupdate;
            }
            return status;
        }
    }

    if let Some(p_nupdate) = p_nupdate {
        *p_nupdate = nupdate;
    }
    status
}

// factorize A[:,basis]
fn factorize(
    obj: &mut basiclu_object,
    Ap: &[lu_int],
    Ai: &[lu_int],
    Ax: &[f64],
    basis: &[lu_int],
) -> lu_int {
    let xstore = &obj.xstore;
    let m = xstore[BASICLU_DIM] as lu_int;
    // lu_int *begin = NULL;
    // lu_int *end = NULL;
    // lu_int i, status = BASICLU_OK;
    let mut status = BASICLU_OK;

    let mut begin = vec![0; m as usize];
    let mut end = vec![0; m as usize];
    // if (!begin || !end) {
    //     status = BASICLU_ERROR_out_of_memory;
    //     goto cleanup;
    // }
    for i in 0..m {
        begin[i as usize] = Ap[basis[i as usize] as usize];
        end[i as usize] = Ap[basis[i as usize] as usize + 1];
    }

    basiclu_obj_factorize(obj, &begin, &end, Ai, Ax)
}

// refactorize the basis if required or favourable
//
// The basis matrix is refactorized if
// - the maximum number of updates is reached, or
// - the previous update had a large pivot error, or
// - it is favourable for performance
//
// factorize() is called for the actual factorization.
//
// Note: refactorize_if_needed() will not do an initial factorization.
fn refactorize_if_needed(
    obj: &mut basiclu_object,
    Ap: &[lu_int],
    Ai: &[lu_int],
    Ax: &[f64],
    basis: &[lu_int],
) -> lu_int {
    let mut status = BASICLU_OK;
    let piverr_tol = 1e-8;
    let xstore = &obj.xstore;

    if xstore[BASICLU_NFORREST] == xstore[BASICLU_DIM]
        || xstore[BASICLU_PIVOT_ERROR] > piverr_tol
        || xstore[BASICLU_UPDATE_COST] > 1.0
    {
        status = factorize(obj, Ap, Ai, Ax, basis);
    }
    status
}
