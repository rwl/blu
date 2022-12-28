// Copyright (C) 2016-2018 ERGO-Code
// Copyright (C) 2022-2023 Richard Lincoln

use crate::lu;
use crate::lu::LU;
use crate::Status;

/// Update the factorization to replace one column of the factorized matrix.
/// A call to `update()` must be preceded by calls to
/// [`solve_for_update()`] to provide the column to be inserted and the
/// index of the column to be replaced.
///
/// The column to be inserted is defined as the right-hand side in the last call
/// to [`solve_for_update()`] in which the forward system was solved.
///
/// The index of the column to be replaced is defined by the unit vector in the
/// last call to [`solve_for_update()`] in which the transposed system was
/// solved.
///
/// `xtbl` is an optional argument to monitor numerical stability. `xtbl` can be
/// either of:
///
/// - element `j0` of the solution to the forward system computed by
///   [`solve_for_update()`], where `j0` is the column to be replaced;
/// - the dot product of the incoming column and the solution to the
///   transposed system computed by [`solve_for_update()`].
///
/// In either case [`LU.pivot_error`] has a defined value. If monitoring
/// stability is not desired, `xtbl` can be any value.
///
/// Returns:
///
/// [`ErrorInvalidCall`] if the factorization is invalid or the update was not
/// prepared by two calls to [`solve_for_update()`].
///
/// [`Reallocate`] for insufficient memory in `w_i`,`w_x`. The number of additional
/// elements required is given by [`LU.addmem_w`]. The user must reallocate
/// `w_i`,`w_x`. It is recommended to reallocate for the requested number of additional
/// elements plus some extra space for further updates (e.g. 0.5 times the current array length).
/// The new array length must be provided in [`LU.w_mem`]. [`update()`] will start
/// from scratch in the next call.
///
/// [`ErrorSingularUpdate`] if the updated factorization would be (numerically) singular.
/// No update has been computed and the old factorization is still valid.
pub fn update(lu: &mut LU, xtbl: f64) -> Status {
    if lu.nupdate < 0 || lu.ftran_for_update < 0 || lu.btran_for_update < 0 {
        Status::ErrorInvalidCall
    } else {
        lu::update(lu, xtbl)
    }
}
