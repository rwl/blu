// Copyright (C) 2016-2018 ERGO-Code
// Copyright (C) 2022-2023 Richard Lincoln

use crate::lu::build_factors;
use crate::lu::condest;
use crate::lu::def::*;
use crate::lu::factorize_bump;
use crate::lu::lu::*;
use crate::lu::residual_test;
use crate::lu::setup_bump;
use crate::lu::singletons;
use crate::Status;
use std::time::Instant;

/// Factorize the matrix `B` into its LU factors. Choose pivot elements by a
/// Markowitz criterion subject to columnwise threshold pivoting (the pivot
/// may not be smaller than a factor of the largest entry in its column).
///
/// ## Arguments
///
/// Matrix `B` must be in packed column form. `b_i` and `b_x` are arrays of row
/// indices and nonzero values. Column `j` of matrix `B` contains elements
///
/// ```txt
///     b_i[b_begin[j] .. b_end[j]-1], b_x[b_begin[j] .. b_end[j]-1].
/// ```
///
/// The columns must not contain duplicate row indices. The arrays `b_begin`
/// and `b_end` may overlap, so that it is valid to pass `b_p`, `b_p[1..]` for
/// a matrix stored in compressed column form (`b_p`, `b_i`, `b_x`).
///
/// `c0ntinue`: `false` to start a new factorization; `true` to continue a
/// factorization after reallocation.
pub fn factorize(
    lu: &mut LU,
    b_begin: &[usize],
    b_end: &[usize],
    b_i: &[usize],
    b_x: &[f64],
    c0ntinue: bool,
) -> Result<(), Status> {
    let tic = Instant::now();

    if !c0ntinue {
        lu.reset();
        lu.task = Task::Singletons;
    }

    fn return_to_caller(
        tic: Instant,
        lu: &mut LU,
        status: Result<(), Status>,
    ) -> Result<(), Status> {
        let elapsed = tic.elapsed().as_secs_f64();
        lu.time_factorize += elapsed;
        lu.time_factorize_total += elapsed;
        status
    }

    // continue factorization
    match lu.task {
        Task::Singletons => {
            // lu.task = SINGLETONS;
            let status = singletons(lu, b_begin, b_end, b_i, b_x);
            if status.is_err() {
                return return_to_caller(tic, lu, status);
            }

            lu.task = Task::SetupBump;
            let status = setup_bump(lu, b_begin, b_end, b_i, b_x);
            if status.is_err() {
                return return_to_caller(tic, lu, status);
            }

            lu.task = Task::FactorizeBump;
            let status = factorize_bump(lu);
            if status.is_err() {
                return return_to_caller(tic, lu, status);
            }
        }
        Task::SetupBump => {
            // lu.task = SETUP_BUMP;
            let status = setup_bump(lu, b_begin, b_end, b_i, b_x);
            if status.is_err() {
                return return_to_caller(tic, lu, status);
            }

            lu.task = Task::FactorizeBump;
            let status = factorize_bump(lu);
            if status.is_err() {
                return return_to_caller(tic, lu, status);
            }
        }
        Task::FactorizeBump => {
            // lu.task = FACTORIZE_BUMP;
            let status = factorize_bump(lu);
            if status.is_err() {
                return return_to_caller(tic, lu, status);
            }
        }
        Task::BuildFactors => {}
        _ => {
            let status = Status::ErrorInvalidCall;
            return Err(status);
        }
    };

    lu.task = Task::BuildFactors;
    let status = build_factors(lu);
    if status.is_err() {
        return return_to_caller(tic, lu, status);
    }

    // factorization successfully finished
    lu.task = Task::NoTask;
    lu.nupdate = Some(0); // make factorization valid
    lu.ftran_for_update = None;
    lu.btran_for_update = None;
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
        let status = Err(Status::WarningSingularMatrix);
        return_to_caller(tic, lu, status)
    } else {
        return_to_caller(tic, lu, Ok(()))
    }
}
