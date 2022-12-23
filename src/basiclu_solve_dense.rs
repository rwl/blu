// Copyright (C) 2016-2018  ERGO-Code

use crate::basiclu::*;
use crate::lu_internal::{lu, lu_load, lu_save};
use crate::lu_solve_dense::lu_solve_dense;

/// Purpose:
///
///     Given the factorization computed by basiclu_factorize() or basiclu_update()
///     and the dense right-hand side, rhs, solve a linear system for the solution
///     lhs.
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
///         BASICLU_ERROR_argument_missing
///
///             One or more of the pointer/array arguments are NULL.
///
///         BASICLU_ERROR_invalid_call
///
///             The factorization is invalid.
pub fn basiclu_solve_dense(
    // istore: &mut [lu_int],
    xstore: &mut [f64],
    Li: &[lu_int],
    Lx: &[f64],
    Ui: &[lu_int],
    Ux: &[f64],
    Wi: &[lu_int],
    Wx: &[f64],
    rhs: &[f64],
    lhs: &mut [f64],
    trans: char,
) -> lu_int {
    let mut this = lu {
        ..Default::default()
    };

    let status = lu_load(
        &mut this,
        // istore,
        xstore,
        // Some(Li),
        // Some(Lx),
        // Some(Ui),
        // Some(Ux),
        // Some(Wi),
        // Some(Wx),
    );
    if status != BASICLU_OK {
        return status;
    }

    // if (! (Li && Lx && Ui && Ux && Wi && Wx && rhs && lhs)) {
    //     status = BASICLU_ERROR_argument_missing;
    // }
    if this.nupdate < 0 {
        let status = BASICLU_ERROR_invalid_call;
        return lu_save(&this, /*istore,*/ xstore, status);
    }

    lu_solve_dense(&mut this, rhs, lhs, trans, Li, Lx, Ui, Ux, Wi, Wx);

    lu_save(&this, /*istore,*/ xstore, status)
}
