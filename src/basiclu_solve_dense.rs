// Copyright (C) 2016-2018  ERGO-Code

use crate::basiclu::*;
use crate::lu_internal::{lu_load, lu_save, LU};
use crate::lu_solve_dense::lu_solve_dense;

/// Purpose:
///
///     Given the factorization computed by basiclu_factorize() or basiclu_update()
///     and the dense right-hand side, rhs, solve a linear system for the solution
///     lhs.
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
///     const double rhs[m]
///
///         The right-hand side vector.
///
///     double lhs[m]
///
///         Uninitialized on entry. On return lhs holds the solution to the linear
///         system.
///
///         lhs and rhs are allowed to overlap. To overwrite rhs with the solution
///         pass pointers to the same array.
///
///     char trans
///
///         Defines which system to solve. 't' or 'T' for the transposed system, any
///         other character for the forward system.
///
/// Info:
///
///     xstore[BASICLU_STATUS]: status code.
///
///         BASICLU_OK
///
///             The linear system has been successfully solved.
///
///         BASICLU_ERROR_ARGUMENT_MISSING
///
///             One or more of the pointer/array arguments are NULL.
///
///         BASICLU_ERROR_INVALID_CALL
///
///             The factorization is invalid.
pub fn basiclu_solve_dense(
    // istore: &mut [lu_int],
    xstore: &mut [f64],
    _l_i: &[LUInt],
    _l_x: &[f64],
    _u_i: &[LUInt],
    _u_x: &[f64],
    _w_i: &[LUInt],
    _w_x: &[f64],
    rhs: &[f64],
    lhs: &mut [f64],
    trans: char,
) -> LUInt {
    let mut lu = LU {
        ..Default::default()
    };

    let status = lu_load(
        &mut lu,
        // istore,
        xstore,
        // Some(l_i),
        // Some(l_x),
        // Some(u_i),
        // Some(u_x),
        // Some(w_i),
        // Some(w_x),
    );
    if status != BASICLU_OK {
        return status;
    }

    // if (! (l_i && l_x && u_i && u_x && w_i && w_x && rhs && lhs)) {
    //     status = BASICLU_ERROR_ARGUMENT_MISSING;
    // }
    if lu.nupdate < 0 {
        let status = BASICLU_ERROR_INVALID_CALL;
        return lu_save(&lu, /*istore,*/ xstore, status);
    }

    lu_solve_dense(&mut lu, rhs, lhs, trans);

    lu_save(&lu, /*istore,*/ xstore, status)
}
