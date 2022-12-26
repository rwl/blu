// Copyright (C) 2016-2018  ERGO-Code

use crate::blu::*;
use crate::lu;
use crate::lu::LU;
// use crate::lu::solve_dense::solve_dense;

/// Purpose:
///
///     Given the factorization computed by factorize() or update()
///     and the dense right-hand side, rhs, solve a linear system for the solution
///     lhs.
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
///     lu_int istore[]
///     double xstore[]
///     lu_int l_i[]
///     double l_x[]
///     lu_int u_i[]
///     double u_x[]
///     lu_int w_i[]
///     double w_x[]
///
///         Factorization computed by factorize() or update().
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
///     xstore[BLU_STATUS]: status code.
///
///         BLU_OK
///
///             The linear system has been successfully solved.
///
///         BLU_ERROR_ARGUMENT_MISSING
///
///             One or more of the pointer/array arguments are NULL.
///
///         BLU_ERROR_INVALID_CALL
///
///             The factorization is invalid.
pub fn solve_dense(lu: &mut LU, rhs: &[f64], lhs: &mut [f64], trans: char) -> LUInt {
    // let status = lu.load(xstore);
    // if status != BLU_OK {
    //     return status;
    // }

    // if (! (l_i && l_x && u_i && u_x && w_i && w_x && rhs && lhs)) {
    //     status = BLU_ERROR_ARGUMENT_MISSING;
    // }
    assert!(lu.nupdate >= 0);
    // if lu.nupdate < 0 {
    //     let status = BLU_ERROR_INVALID_CALL;
    //     return lu_save(&lu, /*istore,*/ xstore, status);
    // }

    lu::solve_dense(lu, rhs, lhs, trans);

    // lu_save(&lu, /*istore,*/ xstore, status)
    BLU_OK
}
