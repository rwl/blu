// Copyright (C) 2016-2018  ERGO-Code

use crate::basiclu::*;
use crate::lu_internal::*;
use crate::lu_solve_for_update::lu_solve_for_update;

/// Purpose:
///
///     Given the factorization computed by basiclu_factorize() or basiclu_update(),
///     solve a linear system in preparation to update the factorization.
///
///     When the forward system is solved, then the right-hand side is the column
///     to be inserted into the factorized matrix. When the transposed system is
///     solved, then the right-hand side is a unit vector with entry 1 in position
///     of the column to be replaced in the factorized matrix.
///
///     For BASICLU to prepare the update, it is sufficient to compute only a
///     partial solution. If the left-hand side is not requested by the user (see
///     below), then only one triangular solve is done. If the left-hand side is
///     requested, then a second triangular solve is required.
///
/// Return:
///
///     BASICLU_ERROR_INVALID_STORE if istore, xstore do not hold a BASICLU
///     instance. In this case xstore[BASICLU_STATUS] is not set.
///
///     Otherwise return the status code. See xstore[BASICLU_STATUS] below.
///
/// Arguments:
///
///     lu_int istore[]
///     double xstore[]
///     lu_int l_i[]
///     double l_x[]
///     lu_int u_i[]
///     double u_x[]
///     lu_int w_i[]
///     double w_x[]
///
///         Factorization computed by basiclu_factorize() or basiclu_update().
///
///     lu_int nzrhs
///     const lu_int irhs[nzrhs]
///     const double xrhs[nzrhs]
///
///         The right-hand side vector in compressed format.
///
///         When the forward system is solved, irhs[0..nzrhs-1] are the indices of
///         nonzeros and xrhs[0..nzrhs-1] the corresponding values. irhs must not
///         contain duplicates.
///
///         When the transposed system is solved, the right-hand side is a unit
///         vector with entry 1 in position irhs[0]. nzrhs, xrhs and elements of
///         irhs other than irhs[0] are not accessed. xrhs can be NULL.
///
///     lu_int *p_nzlhs
///     lu_int ilhs[m]
///     lu_int lhs[m]
///
///         If any of p_nzlhs, ilhs or lhs is NULL, then the solution to the linear
///         system is not requested. In this case only the update is prepared.
///
///         Otherwise:
///
///         *p_nzlhs is uninitialized on entry. On return *p_nzlhs holds
///         the number of nonzeros in the solution.
///         The contents of ilhs is uninitialized on entry. On return
///         ilhs[0..*p_nzlhs-1] holds the indices of nonzeros in the solution.
///         The contents of lhs must be initialized to zero on entry. On return
///         the solution is  scattered into lhs.
///
///     char trans
///
///         Defines which system to solve. 't' or 'T' for the transposed system,
///         any other character for the forward system.
///
/// Parameters:
///
///     xstore[BASICLU_MEMORYL]: length of l_i and l_x
///     xstore[BASICLU_MEMORYU]: length of u_i and u_x
///     xstore[BASICLU_MEMORYW]: length of w_i and w_x
///
///     xstore[BASICLU_SPARSE_THRESHOLD]
///
///         Defines which method is used for solving a triangular system. A
///         triangular solve can be done either by the two phase method of Gilbert
///         and Peierls ("sparse solve") or by a sequential pass through the vector
///         ("sequential solve").
///
///         When the solution to the linear system is requested, then two triangular
///         systems are solved. The first triangular solve is done sparse. The
///         second triangular solve is done sparse if its right-hand side has not
///         more than m * xstore[BASICLU_SPARSE_THRESHOLD] nonzeros. Otherwise the
///         sequential solve is used.
///
///         When the solution to the linear system is not requested, then this
///         parameter has no effect.
///
///         Default: 0.05
///
///     xstore[BASICLU_DROP_TOLERANCE]
///
///         Nonzeros which magnitude is less than or equal to the drop tolerance
///         are removed after each triangular solve. Default: 1e-20
///
/// Info:
///
///     xstore[BASICLU_STATUS]: status code.
///
///         BASICLU_OK
///
///             The updated has been successfully prepared and, if requested, the
///             solution to the linear system has been computed.
///
///         BASICLU_ERROR_ARGUMENT_MISSING
///
///             One or more of the mandatory pointer/array arguments are NULL.
///
///         BASICLU_ERROR_INVALID_CALL
///
///             The factorization is invalid.
///
///         BASICLU_ERROR_MAXIMUM_UPDATES
///
///             There have already been m Forrest-Tomlin updates, see
///             xstore[BASICLU_NFORREST]. The factorization cannot be updated any
///             more and must be recomputed by basiclu_factorize().
///             The solution to the linear system has not been computed.
///
///         BASICLU_ERROR_INVALID_ARGUMENT
///
///             The right-hand side is invalid (forward system: nzrhs < 0 or
///             nzrhs > m or one or more indices out of range; backward system:
///             irhs[0] out of range).
///
///         BASICLU_REALLOCATE
///
///             The solve was aborted because of insufficient memory in l_i,l_x or
///             u_i,u_x to store data for basiclu_update(). The number of additional
///             elements required is given by
///
///                 xstore[BASICLU_ADD_MEMORYL] >= 0
///                 xstore[BASICLU_ADD_MEMORYU] >= 0
///
///             The user must reallocate the arrays for which additional memory is
///             required. It is recommended to reallocate for the requested number
///             of additional elements plus some extra space for further updates
///             (e.g. 0.5 times the current array length). The new array lengths
///             must be provided in
///
///                 xstore[BASICLU_MEMORYL]: length of l_i and l_x
///                 xstore[BASICLU_MEMORYU]: length of u_i and u_x
///
///             basiclu_solve_for_update() will start from scratch in the next call.
pub fn basiclu_solve_for_update(
    _istore: &mut [LUInt],
    xstore: &mut [f64],
    _l_i: &mut [LUInt],
    _l_x: &mut [f64],
    _u_i: &mut [LUInt],
    _u_x: &mut [f64],
    _w_i: &mut [LUInt],
    _w_x: &mut [f64],
    nzrhs: LUInt,
    irhs: &[LUInt],
    xrhs: Option<&[f64]>,
    p_nzlhs: Option<&mut LUInt>,
    ilhs: Option<&mut [LUInt]>,
    lhs: Option<&mut [f64]>,
    trans: char,
) -> LUInt {
    let mut lu = LU {
        ..Default::default()
    };
    // lu_int status, n, ok;

    let status = lu_load(
        &mut lu,
        // istore,
        xstore,
        // Some(l_i.to_vec()), // FIXME
        // Some(l_x.to_vec()),
        // Some(u_i.to_vec()),
        // Some(u_x.to_vec()),
        // Some(w_i.to_vec()),
        // Some(w_x.to_vec()),
    );
    if status != BASICLU_OK {
        return status;
    }

    // if (! (l_i && l_x && u_i && u_x && w_i && w_x && irhs)) {
    //     status = BASICLU_ERROR_ARGUMENT_MISSING;
    // }
    let mut status = if trans != 't' && trans != 'T'
    /*&& !xrhs*/
    {
        BASICLU_ERROR_ARGUMENT_MISSING
    } else if lu.nupdate < 0 {
        BASICLU_ERROR_INVALID_CALL
    } else if lu.nforrest == lu.m {
        BASICLU_ERROR_MAXIMUM_UPDATES
    } else {
        // check RHS indices
        let ok = if trans == 't' || trans == 'T' {
            irhs[0] >= 0 && irhs[0] < lu.m
        } else {
            let mut ok = nzrhs >= 0 && nzrhs <= lu.m;
            for n in 0..nzrhs as usize {
                if !ok {
                    break;
                }
                ok = ok && irhs[n] >= 0 && irhs[n] < lu.m;
            }
            ok
        };
        if !ok {
            BASICLU_ERROR_INVALID_ARGUMENT
        } else {
            BASICLU_OK
        }
    };

    if status == BASICLU_OK {
        // may request reallocation
        status = lu_solve_for_update(&mut lu, nzrhs, irhs, xrhs, p_nzlhs, ilhs, lhs, trans);
    }

    lu_save(&lu, /*istore,*/ xstore, status)
}
