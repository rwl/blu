// Copyright (C) 2016-2018 ERGO-Code
// Copyright (C) 2022-2023 Richard Lincoln

use crate::lu;
use crate::lu::LU;
use crate::Status;

/// Given the factorization computed by [`factorize()`](crate::factorize()) or
/// [`update()`](crate::update()) and the dense right-hand side, `rhs`, solve a linear
/// system for the solution `lhs`.
///
/// ## Arguments
///
/// On return `lhs` holds the solution to the linear system. Uninitialized on entry.
/// `lhs` and `rhs` are allowed to overlap. To overwrite `rhs` with the solution
/// pass references to the same array.
///
/// `trans` defines which system to solve. `'t'` or `'T'` for the transposed system, any
/// other character for the forward system.
///
/// ## Returns
///
/// [`Status::ErrorInvalidCall`] if the factorization is invalid.
pub fn solve_dense(lu: &mut LU, rhs: &[f64], lhs: &mut [f64], trans: char) -> Result<(), Status> {
    if lu.nupdate.is_none() {
        return Err(Status::ErrorInvalidCall);
    }

    lu::solve_dense(lu, rhs, lhs, trans);

    Ok(())
}
