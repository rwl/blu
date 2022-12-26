// Copyright (C) 2016-2018  ERGO-Code

use crate::basiclu::{LUInt, BASICLU_ERROR_INVALID_CALL};
use crate::lu_internal::LU;
use crate::lu_update::lu_update;

/// Update the factorization to replace one column of the factorized matrix.
/// A call to `basiclu_update()` must be preceded by calls to
/// [`basiclu_solve_for_update()`] to provide the column to be inserted and the
/// index of the column to be replaced.
///
/// The column to be inserted is defined as the right-hand side in the last call
/// to [`basiclu_solve_for_update()`] in which the forward system was solved.
///
/// The index of the column to be replaced is defined by the unit vector in the
/// last call to [`basiclu_solve_for_update()`] in which the transposed system was
/// solved.
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
///     double xtbl
///
///         This is an optional argument to monitor numerical stability. xtbl can be
///         either of
///
///         (a) element j0 of the solution to the forward system computed by
///             basiclu_solve_for_update(), where j0 is the column to be replaced;
///
///         (b) the dot product of the incoming column and the solution to the
///             transposed system computed by basiclu_solve_for_update().
///
///         In either case xstore[BASICLU_PIVOT_ERROR] (see below) has a defined
///         value. If monitoring stability is not desired, xtbl can be any value.
///
/// Parameters:
///
///     xstore[BASICLU_MEMORYL]: length of l_i and l_x
///     xstore[BASICLU_MEMORYU]: length of u_i and u_x
///     xstore[BASICLU_MEMORYW]: length of w_i and w_x
///
///     xstore[BASICLU_DROP_TOLERANCE]
///
///         Nonzeros which magnitude is less than or equal to the drop tolerance
///         are removed from the row eta matrix. Default: 1e-20
///
/// Info:
///
///     xstore[BASICLU_STATUS]: status code.
///
///         BASICLU_OK
///
///             The update has successfully completed.
///
///         BASICLU_ERROR_ARGUMENT_MISSING
///
///             One or more of the pointer/array arguments are NULL.
///
///         BASICLU_ERROR_INVALID_CALL
///
///             The factorization is invalid or the update was not prepared by two
///             calls to basiclu_solve_for_update().
///
///         BASICLU_REALLOCATE
///
///             Insufficient memory in w_i,w_x. The number of additional elements
///             required is given by
///
///                 xstore[BASICLU_ADD_MEMORYW] > 0
///
///             The user must reallocate w_i,w_x. It is recommended to reallocate for
///             the requested number of additional elements plus some extra space
///             for further updates (e.g. 0.5 times the current array length). The
///             new array length must be provided in
///
///                 xstore[BASICLU_MEMORYW]: length of w_i and w_x
///
///             basiclu_update will start from scratch in the next call.
///
///         BASICLU_ERROR_SINGULAR_UPDATE
///
///             The updated factorization would be (numerically) singular. No update
///             has been computed and the old factorization is still valid.
///
///     xstore[BASICLU_PIVOT_ERROR]
///
///         When xtbl was given (see above), then xstore[BASICLU_PIVOT_ERROR] is a
///         measure for numerical stability. It is the difference between two
///         computations of the new pivot element relative to the new pivot element.
///         A value larger than 1e-10 indicates numerical instability and suggests
///         refactorization (and possibly tightening the pivot tolerance).
///
///     xstore[BASICLU_MAX_ETA]
///
///         The maximum entry (in absolute value) in the eta vectors from the
///         Forrest-Tomlin update. A large value, say > 1e6, indicates that pivoting
///         on diagonal element was unstable and refactorization might be necessary.
pub fn basiclu_update(lu: &mut LU, xtbl: f64) -> LUInt {
    // let status = lu.load(xstore);
    // if status != BASICLU_OK {
    //     return status;
    // }

    // if (! (l_i && l_x && u_i && u_x && w_i && w_x)) {
    //     status = BASICLU_ERROR_ARGUMENT_MISSING;
    // }
    let status = if lu.nupdate < 0 || lu.ftran_for_update < 0 || lu.btran_for_update < 0 {
        BASICLU_ERROR_INVALID_CALL
    } else {
        lu_update(lu, xtbl)
    };
    // lu_save(&mut lu, /*istore,*/ xstore, status)
    status
}
