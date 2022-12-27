// Copyright (C) 2016-2018 ERGO-Code
// Copyright (C) 2022-2023 Richard Lincoln

use crate::blu::*;
use crate::lu::build_factors;
use crate::lu::condest;
use crate::lu::def::*;
use crate::lu::factorize_bump;
use crate::lu::lu::*;
use crate::lu::residual_test;
use crate::lu::setup_bump;
use crate::lu::singletons;
use std::time::Instant;

/// Factorize the matrix B into its LU factors. Choose pivot elements by a
/// Markowitz criterion subject to columnwise threshold pivoting (the pivot
/// may not be smaller than a factor of the largest entry in its column).
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
///
///         BLU instance. The instance determines the dimension of matrix B
///         (stored in xstore[BLU_DIM]).
///
///     lu_int l_i[]
///     double l_x[]
///     lu_int u_i[]
///     double u_x[]
///     lu_int w_i[]
///     double w_x[]
///
///         Arrays used for workspace during the factorization and to store the
///         final factors. They must be allocated by the user and their length
///         must be provided as parameters:
///
///             xstore[BLU_MEMORYL]: length of l_i and l_x
///             xstore[BLU_MEMORYU]: length of u_i and u_x
///             xstore[BLU_MEMORYW]: length of w_i and w_x
///
///         When the allocated length is insufficient to complete the factorization,
///         factorize() returns to the caller for reallocation (see
///         xstore[BLU_STATUS] below). A successful factorization requires at
///         least nnz(B) length for each of the arrays.
///
///     const lu_int b_begin[]
///     const lu_int b_end[]
///     const lu_int b_i[]
///     const double b_x[]
///
///         Matrix B in packed column form. b_i and b_x are arrays of row indices
///         and nonzero values. Column j of matrix B contains elements
///
///             b_i[b_begin[j] .. b_end[j]-1], b_x[b_begin[j] .. b_end[j]-1].
///
///         The columns must not contain duplicate row indices. The arrays b_begin
///         and b_end may overlap, so that it is valid to pass Bp, Bp+1 for a matrix
///         stored in compressed column form (Bp, b_i, b_x).
///
///     lu_int c0ntinue
///
///         zero to start a new factorization; nonzero to continue a factorization
///         after reallocation.
pub fn factorize(
    lu: &mut LU,
    b_begin: &[LUInt],
    b_end: &[LUInt],
    b_i: &[LUInt],
    b_x: &[f64],
    c0ntinue: LUInt,
) -> LUInt {
    let tic = Instant::now();

    // let status = lu.load(xstore);
    // if status != BLU_OK {
    //     return status;
    // }

    // if !(l_i && l_x && u_i && u_x && w_i && w_x && b_begin && b_end && b_i && b_x) {
    //     let status = BLU_ERROR_ARGUMENT_MISSING;
    //     return lu_save(&lu, istore, xstore, status);
    // }
    if c0ntinue == 0 {
        lu.reset();
        lu.task = SINGLETONS;
    }

    fn return_to_caller(
        tic: Instant,
        lu: &mut LU,
        /*xstore: &mut [f64],*/ status: LUInt,
    ) -> LUInt {
        let elapsed = tic.elapsed().as_secs_f64();
        lu.time_factorize += elapsed;
        lu.time_factorize_total += elapsed;
        // return lu_save(&lu, /*istore,*/ xstore, status);
        status
    }

    // continue factorization
    match lu.task {
        SINGLETONS => {
            // lu.task = SINGLETONS;
            let status = singletons(lu, b_begin, b_end, b_i, b_x);
            if status != BLU_OK {
                return return_to_caller(tic, lu, /*xstore,*/ status);
            }

            lu.task = SETUP_BUMP;
            let status = setup_bump(lu, b_begin, b_end, b_i, b_x);
            if status != BLU_OK {
                return return_to_caller(tic, lu, /*xstore,*/ status);
            }

            lu.task = FACTORIZE_BUMP;
            let status = factorize_bump(lu);
            if status != BLU_OK {
                return return_to_caller(tic, lu, /*xstore,*/ status);
            }
        }
        SETUP_BUMP => {
            // lu.task = SETUP_BUMP;
            let status = setup_bump(lu, b_begin, b_end, b_i, b_x);
            if status != BLU_OK {
                return return_to_caller(tic, lu, /*xstore,*/ status);
            }

            lu.task = FACTORIZE_BUMP;
            let status = factorize_bump(lu);
            if status != BLU_OK {
                return return_to_caller(tic, lu, /*xstore,*/ status);
            }
        }
        FACTORIZE_BUMP => {
            // lu.task = FACTORIZE_BUMP;
            let status = factorize_bump(lu);
            if status != BLU_OK {
                return return_to_caller(tic, lu, /*xstore,*/ status);
            }
        }
        BUILD_FACTORS => {}
        _ => {
            let status = BLU_ERROR_INVALID_CALL;
            // return lu_save(&lu, xstore, status);
            return status;
        }
    };

    lu.task = BUILD_FACTORS;
    let status = build_factors(lu);
    if status != BLU_OK {
        return return_to_caller(tic, lu, /*xstore,*/ status);
    }

    // factorization successfully finished
    lu.task = NO_TASK;
    lu.nupdate = 0; // make factorization valid
    lu.ftran_for_update = -1;
    lu.btran_for_update = -1;
    lu.nfactorize += 1;

    lu.condest_l = condest(
        lu.m,
        &l_begin!(lu),
        &lu.l_index,
        &lu.l_value,
        None,
        Some(&p!(lu)),
        0,
        &mut lu.work1,
        Some(&mut lu.norm_l),
        Some(&mut lu.normest_l_inv),
    );
    lu.condest_u = condest(
        lu.m,
        &lu.u_begin,
        &lu.u_index,
        &lu.u_value,
        Some(&lu.row_pivot),
        Some(&p!(lu)),
        1,
        &mut lu.work1,
        Some(&mut lu.norm_u),
        Some(&mut lu.normest_u_inv),
    );

    // measure numerical stability of the factorization
    residual_test(lu, b_begin, b_end, b_i, b_x);

    // factor_cost is a deterministic measure of the factorization cost.
    // The parameters have been adjusted such that (on my computer)
    // 1e-6 * factor_cost =~ time_factorize.
    //
    // update_cost measures the accumulated cost of updates/solves compared
    // to the last factorization. It is computed from
    //
    //   update_cost = update_cost_numer / update_cost_denom.
    //
    // update_cost_denom is fixed here.
    // update_cost_numer is zero here and increased by solves/updates.
    let factor_cost = 0.04 * (lu.m as f64)
        + 0.07 * (lu.matrix_nz as f64)
        + 0.20 * (lu.bump_nz as f64)
        + 0.20 * (lu.nsearch_pivot as f64)
        + 0.008 * (lu.factor_flops as f64);

    lu.update_cost_denom = factor_cost * 250.0;

    if cfg!(feature = "debug") {
        let elapsed = lu.time_factorize + tic.elapsed().as_secs_f64();
        println!(
            " 1e-6 * factor_cost / time_factorize: {}",
            1e-6 * factor_cost / elapsed,
        );
    }

    if lu.rank < lu.m {
        let status = BLU_WARNING_SINGULAR_MATRIX;
        return_to_caller(tic, lu, /*xstore,*/ status);
    }

    return_to_caller(tic, lu, /*xstore,*/ status)
}
