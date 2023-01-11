// Copyright (C) 2016-2018 ERGO-Code
// Copyright (C) 2022-2023 Richard Lincoln

use crate::lu;
use crate::lu::lu::*;
use crate::LUInt;
use crate::Status;

/// Given the factorization computed by [`factorize()`](crate::factorize()) or
/// [`update()`](crate::update()), solve a linear system in preparation to update
/// the factorization.
///
/// When the forward system is solved, then the right-hand side is the column
/// to be inserted into the factorized matrix. When the transposed system is
/// solved, then the right-hand side is a unit vector with entry 1 in position
/// of the column to be replaced in the factorized matrix.
///
/// For BLU to prepare the update, it is sufficient to compute only a
/// partial solution. If the left-hand side is not requested by the user (see
/// below), then only one triangular solve is done. If the left-hand side is
/// requested, then a second triangular solve is required.
///
/// ## Arguments
///
/// The right-hand side vector is provided in compressed format. When the forward
/// system is solved, `irhs[0..nzrhs-1]` are the indices of nonzeros and
/// `xrhs[0..nzrhs-1]` the corresponding values. `irhs` must not contain duplicates.
///
/// When the transposed system is solved, the right-hand side is a unit
/// vector with entry 1 in position `irhs[0]`. `nzrhs`, `xrhs` and elements of
/// `irhs` other than `irhs[0]` are not accessed.
///
/// If any of `p_nzlhs`, `ilhs` or `lhs` is `None`, then the solution to the linear
/// system is not requested. In this case only the update is prepared.
///
/// Otherwise, `p_nzlhs` is uninitialized on entry. On return `p_nzlhs` holds
/// the number of nonzeros in the solution. The contents of `ilhs` is uninitialized
/// on entry. On return `ilhs[0..p_nzlhs-1]` holds the indices of nonzeros in the
/// solution. The contents of lhs must be initialized to zero on entry. On return
/// the solution is scattered into `lhs`.
///
/// `trans` defines which system to solve. 't' or 'T' for the transposed system,
/// any other character for the forward system.
///
/// See parameters [`LU::sparse_thres`] and [`LU::droptol`].
///
/// ## Returns
///
/// [`Status::ErrorInvalidCall`] if the factorization is invalid.
///
/// [`Status::ErrorMaximumUpdates`] if rhere have already been `m` Forrest-Tomlin updates,
/// see [`LU::nforrest`]. The factorization cannot be updated any more and must be
/// recomputed by [`factorize()`](crate::factorize()). The solution to the linear
/// system has not been computed.
///
/// [`Status::ErrorInvalidArgument`] if the right-hand side is invalid
/// (forward system: `nzrhs` < 0 or `nzrhs` > [`LU::m`] or one or more indices out of range;
/// backward system: `irhs[0]` out of range).
///
/// [`Status::Reallocate`] if the solve was aborted because of insufficient memory in
/// `l_i`,`l_x` or `u_i`,`u_x` to store data for [`update()`](crate::update()).
/// The number of additional elements required is given by [`LU::addmem_l`] and
/// [`LU::addmem_u`].
///
/// The user must reallocate the arrays for which additional memory is
/// required. It is recommended to reallocate for the requested number
/// of additional elements plus some extra space for further updates
/// (e.g. 0.5 times the current array length). The new array lengths
/// must be provided in [`LU::l_mem`] and [`LU::u_mem`].
/// [`solve_for_update()`](crate::solve_for_update()) will start from
/// scratch in the next call.
pub fn solve_for_update(
    lu: &mut LU,
    nzrhs: LUInt,
    irhs: &[usize],
    xrhs: Option<&[f64]>,
    p_nzlhs: Option<&mut usize>,
    ilhs: Option<&mut [LUInt]>,
    lhs: Option<&mut [f64]>,
    trans: char,
) -> Status {
    let mut status = if trans != 't' && trans != 'T' && xrhs.is_none() {
        Status::ErrorArgumentMissing
    } else if lu.nupdate.is_none() {
        Status::ErrorInvalidCall
    } else if lu.nforrest == lu.m {
        Status::ErrorMaximumUpdates
    } else {
        // check RHS indices
        let ok = if trans == 't' || trans == 'T' {
            /*irhs[0] >= 0 &&*/
            irhs[0] < lu.m
        } else {
            let mut ok = nzrhs >= 0 && nzrhs <= lu.m as LUInt;
            for n in 0..nzrhs as usize {
                if !ok {
                    break;
                }
                ok = ok && /*irhs[n] >= 0 &&*/ irhs[n] < lu.m;
            }
            ok
        };
        if !ok {
            Status::ErrorInvalidArgument
        } else {
            Status::OK
        }
    };

    if status == Status::OK {
        // may request reallocation
        status = lu::solve_for_update(
            lu,
            nzrhs as usize,
            &irhs.iter().map(|&i| i as LUInt).collect::<Vec<LUInt>>(),
            xrhs,
            p_nzlhs,
            ilhs,
            lhs,
            trans,
        );
    }

    status
}
