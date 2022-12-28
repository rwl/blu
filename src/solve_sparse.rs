// Copyright (C) 2016-2018 ERGO-Code
// Copyright (C) 2022-2023 Richard Lincoln

use crate::lu;
use crate::lu::LU;
use crate::LUInt;
use crate::Status;

/// Given the factorization computed by `factorize()` or `update()`
/// and the sparse right-hand side, [`rhs`], solve a linear system for the
/// solution [`lhs`].
///
/// The right-hand side vector is provided in compressed format. `irhs[0..nzrhs-1]`
/// are the indices of nonzeros and `xrhs[0..nzrhs-1]` the corresponding values.
/// `irhs` must not contain duplicates.
///
/// `p_nzlhs` is uninitialized on entry. On return `p_nzlhs` holds the number of
/// nonzeros in the solution. The contents of `ilhs` is uninitialized on entry. On
/// return `ilhs[0..p_nzlhs-1]` holds the indices of nonzeros in the solution.
/// The contents `lhs` must be initialized to zero on entry. On return the solution
/// is scattered into `lhs`.
///
/// `trans` defines which system to solve. 't' or 'T' for the transposed system,
/// any other character for the forward system.
///
/// See parameters [`LU.sparse_thres`] and [`LU.droptol`].
///
/// Returns [`ErrorInvalidCall`] if the factorization is invalid, [`ErrorInvalidArgument`]
/// if the right-hand side is invalid (`nzrhs` < 0 or `nzrhs` > `m` or one or
/// more indices out of range).
pub fn solve_sparse(
    lu: &mut LU,
    nzrhs: LUInt,
    irhs: &[LUInt],
    xrhs: &[f64],
    p_nzlhs: &mut LUInt,
    ilhs: &mut [LUInt],
    lhs: &mut [f64],
    trans: char,
) -> Status {
    let status = if lu.nupdate < 0 {
        Status::ErrorInvalidCall
    } else {
        // check RHS indices
        let mut ok = nzrhs >= 0 && nzrhs <= lu.m;
        for n in 0..nzrhs as usize {
            if !ok {
                break;
            }
            ok = ok && irhs[n] >= 0 && irhs[n] < lu.m;
        }
        if !ok {
            Status::ErrorInvalidArgument
        } else {
            Status::OK
        }
    };

    if status == Status::OK {
        lu::solve_sparse(lu, nzrhs, irhs, xrhs, p_nzlhs, ilhs, lhs, trans);
    }

    status
}
