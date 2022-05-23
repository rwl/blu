use crate::basiclu::*;
use crate::lu_def::*;
use crate::lu_internal::{lu, lu_load, lu_reset, lu_save};
use crate::lu_setup_bump::lu_setup_bump;
use crate::lu_singletons::lu_singletons;
use std::time::Instant;

/// Purpose:
///
///     Factorize the matrix B into its LU factors. Choose pivot elements by a
///     Markowitz criterion subject to columnwise threshold pivoting (the pivot may
///     not be smaller than a factor of the largest entry in its column).
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
///
///         BASICLU instance. The instance determines the dimension of matrix B
///         (stored in xstore[BASICLU_DIM]).
///
///     lu_int Li[]
///     double Lx[]
///     lu_int Ui[]
///     double Ux[]
///     lu_int Wi[]
///     double Wx[]
///
///         Arrays used for workspace during the factorization and to store the
///         final factors. They must be allocated by the user and their length
///         must be provided as parameters:
///
///             xstore[BASICLU_MEMORYL]: length of Li and Lx
///             xstore[BASICLU_MEMORYU]: length of Ui and Ux
///             xstore[BASICLU_MEMORYW]: length of Wi and Wx
///
///         When the allocated length is insufficient to complete the factorization,
///         basiclu_factorize() returns to the caller for reallocation (see
///         xstore[BASICLU_STATUS] below). A successful factorization requires at
///         least nnz(B) length for each of the arrays.
///
///     const lu_int Bbegin[]
///     const lu_int Bend[]
///     const lu_int Bi[]
///     const double Bx[]
///
///         Matrix B in packed column form. Bi and Bx are arrays of row indices
///         and nonzero values. Column j of matrix B contains elements
///
///             Bi[Bbegin[j] .. Bend[j]-1], Bx[Bbegin[j] .. Bend[j]-1].
///
///         The columns must not contain duplicate row indices. The arrays Bbegin
///         and Bend may overlap, so that it is valid to pass Bp, Bp+1 for a matrix
///         stored in compressed column form (Bp, Bi, Bx).
///
///     lu_int c0ntinue
///
///         zero to start a new factorization; nonzero to continue a factorization
///         after reallocation.
///
/// Parameters:
///
///     xstore[BASICLU_DROP_TOLERANCE]
///
///         Nonzeros which magnitude is less than or equal to the drop tolerance can
///         be removed after each pivot step. They are guaranteed removed at the end
///         of the factorization. Default: 1e-20
///
///     xstore[BASICLU_ABS_PIVOT_TOLERANCE]
///
///         A pivot element must be nonzero and in absolute value must be greater
///         than or equal to xstore[BASICLU_ABS_PIVOT_TOLERANCE]. Default: 1e-14
///
///     xstore[BASICLU_REL_PIVOT_TOLERANCE]
///
///         A pivot element must be (in absolute value) greater than or equal to
///         xstore[BASICLU_REL_PIVOT_TOLERANCE] times the largest entry in its
///         column. A value greater than or equal to 1.0 is treated as 1.0 and
///         enforces partial pivoting. Default: 0.1
///
///     xstore[BASICLU_BIAS_NONZEROS]
///
///         When this value is greater than or equal to zero, the pivot choice
///         attempts to keep L sparse, putting entries into U when possible.
///         When this value is less than zero, the pivot choice attempts to keep U
///         sparse, putting entries into L when possible. Default: 1
///
///     xstore[BASICLU_MAXN_SEARCH_PIVOT]
///
///         The Markowitz search is terminated after searching
///         xstore[BASICLU_MAXN_SERACH_PIVOT] rows or columns if a numerically
///         stable pivot element has been found. Default: 3
///
///     xstore[BASICLU_SEARCH_ROWS]
///
///         If xstore[BASICLU_SEARCH_ROWS] is zero, then the Markowitz search only
///         scans columns. If nonzero, then both columns and rows are searched in
///         increasing order of number of entries. Default: 1
///
///     xstore[BASICLU_PAD]
///     xstore[BASICLU_STRETCH]
///
///         When a row or column cannot be updated by the pivot operation in place,
///         it is appended to the end of the workspace. For a row or column with nz
///         elements, xstore[BASICLU_PAD] + nz * xstore[BASICLU_STRETCH] elements
///         extra space are added for later fill-in.
///         Default: xstore[BASICLU_PAD] = 4, xstore[BASICLU_STRETCH] = 0.3
///
///     xstore[BASICLU_REMOVE_COLUMNS]
///
///         This parameter is present for compatibility to previous versions but has
///         no effect. If during factorization the maximum entry of a column of the
///         active submatrix becomes zero or less than
///         xstore[BASICLU_ABS_PIVOT_TOLERANCE], then that column is immediately
///         removed without choosing a pivot.
///
/// Info:
///
///     xstore[BASICLU_STATUS]: status code.
///
///         BASICLU_OK
///
///             The factorization has successfully completed.
///
///         BASICLU_WARNING_singular_matrix
///
///             The factorization did xstore[BASICLU_RANK] < xstore[BASICLU_DIM]
///             pivot steps. The remaining elements in the active submatrix are zero
///             or less than xstore[BASICLU_ABS_PIVOT_TOLERANCE]. The factors have
///             been augmented by unit columns to form a square matrix. See
///             basiclu_get_factors() on how to get the indices of linearly
///             dependent columns.
///
///         BASICLU_ERROR_argument_missing
///
///             One or more of the pointer/array arguments are NULL.
///
///         BASICLU_ERROR_invalid_call
///
///             c0ntinue is nonzero, but the factorization was not started before.
///
///         BASICLU_ERROR_invalid_argument
///
///             The matrix is invalid (a column has a negative number of entries,
///             a row index is out of range, or a column has duplicate entries).
///
///         BASICLU_REALLOCATE
///
///             Factorization requires more memory in Li,Lx and/or Ui,Ux and/or
///             Wi,Wx. The number of additional elements in each of the array pairs
///             required for the next pivot operation is given by:
///
///                 xstore[BASICLU_ADD_MEMORYL] >= 0
///                 xstore[BASICLU_ADD_MEMORYU] >= 0
///                 xstore[BASICLU_ADD_MEMORYW] >= 0
///
///             The user must reallocate the arrays for which additional memory is
///             required. It is recommended to reallocate for the requested number
///             of additional elements plus some extra space (e.g. 0.5 times the
///             current array length). The new array lengths must be provided in
///
///                 xstore[BASICLU_MEMORYL]: length of Li and Lx
///                 xstore[BASICLU_MEMORYU]: length of Ui and Ux
///                 xstore[BASICLU_MEMORYW]: length of Wi and Wx
///
///             basiclu_factorize() can be called again with c0ntinue not equal to
///             zero to continue the factorization.
///
///     xstore[BASICLU_MATRIX_NZ] number of nonzeros in B
///
///     xstore[BASICLU_MATRIX_ONENORM]
///     xstore[BASICLU_MATRIX_INFNORM] 1-norm and inf-norm of the input matrix
///                                    after replacing dependent columns by unit
///                                    columns.
///
///     xstore[BASICLU_RANK] number of pivot steps performed
///
///     xstore[BASICLU_BUMP_SIZE] dimension of matrix after removing singletons
///
///     xstore[BASICLU_BUMP_NZ] # nonzeros in matrix after removing singletons
///
///     xstore[BASICLU_NSEARCH_PIVOT] total # columns/rows searched for pivots
///
///     xstore[BASICLU_NEXPAND] # columns/rows which had to be appended to the end
///                             of the workspace for the rank-1 update
///
///     xstore[BASICLU_NGARBAGE] # garbage collections
///
///     xstore[BASICLU_FACTOR_FLOPS] # floating point operations performed,
///                                  counting multiply-add as one flop
///
///     xstore[BASICLU_TIME_SINGLETONS] wall clock time for removing the initial
///                                     triangular factors
///
///     xstore[BASICLU_TIME_SEARCH_PIVOT] wall clock time for Markowitz search
///
///     xstore[BASIClU_TIME_ELIM_PIVOT] wall clock time for pivot elimination
///
///     xstore[BASICLU_RESIDUAL_TEST]
///
///             An estimate for numerical stability of the factorization.
///             xstore[BASICLU_RESIDUAL_TEST] is the maximum of the scaled residuals
///
///               ||b-Bx|| / (||b|| + ||B||*||x||)
///
///             and
///
///               ||c-B'y|| / (||c|| + ||B'||*||y||),
///
///             where x=B\b and y=B'\c are computed from the LU factors, b and c
///             have components +/-1 that are chosen to make x respectively y large,
///             and ||.|| is the 1-norm. Here B is the input matrix after replacing
///             dependent columns by unit columns.
///
///             If xstore[BASICLU_RESIDUAL_TEST] > 1e-12, say, the factorization is
///             numerically unstable. (This is independent of the condition number
///             of B.) In this case tightening the relative pivot tolerance and
///             refactorizing is appropriate.
///
///     xstore[BASICLU_NORM_L]
///     xstore[BASICLU_NORM_U] 1-norm of L and U.
///
///     xstore[BASICLU_NORMEST_LINV]
///     xstore[BASICLU_NORMEST_UINV] Estimated 1-norm of L^{-1} and U^{-1},
///                                  computed by the LINPACK algorithm.
///
///     xstore[BASICLU_CONDEST_L]
///     xstore[BASICLU_CONDEST_U] Estimated 1-norm condition number of L and U.
pub fn basiclu_factorize(
    istore: &mut [lu_int],
    xstore: &mut [f64],
    Li: &mut [lu_int],
    Lx: &mut [f64],
    Ui: &mut [lu_int],
    Ux: &mut [f64],
    Wi: &mut [lu_int],
    Wx: &mut [f64],
    Bbegin: &[lu_int],
    Bend: &[lu_int],
    Bi: &[lu_int],
    Bx: &[f64],
    c0ntinue: lu_int,
) -> lu_int {
    let mut this = lu {
        ..Default::default()
    };
    let tic = Instant::now();

    let mut status = lu_load(
        &mut this,
        istore,
        xstore,
        Some(Li),
        Some(Lx),
        Some(Ui),
        Some(Ux),
        Some(Wi),
        Some(Wx),
    );
    if status != BASICLU_OK {
        return status;
    }

    // if !(Li && Lx && Ui && Ux && Wi && Wx && Bbegin && Bend && Bi && Bx) {
    //     let status = BASICLU_ERROR_argument_missing;
    //     return lu_save(&this, istore, xstore, status);
    // }
    if !c0ntinue {
        lu_reset(&mut this);
        this.task = SINGLETONS;
    }

    let return_to_caller = || {
        elapsed = lu_toc(tic);
        this.time_factorize += elapsed;
        this.time_factorize_total += elapsed;
        return lu_save(&this, istore, xstore, status);
    };

    // continue factorization
    match this.task {
        SINGLETONS => {
            // this.task = SINGLETONS;
            status = lu_singletons(&mut this, Bbegin, Bend, Bi, Bx);
            if status != BASICLU_OK {
                return return_to_caller();
            }

            this.task = SETUP_BUMP;
            status = lu_setup_bump(&mut this, Bbegin, Bend, Bi, Bx);
            if status != BASICLU_OK {
                return return_to_caller();
            }

            this.task = FACTORIZE_BUMP;
            status = lu_factorize_bump(&this);
            if status != BASICLU_OK {
                return return_to_caller();
            }
        }
        SETUP_BUMP => {
            // this.task = SETUP_BUMP;
            status = lu_setup_bump(&mut this, Bbegin, Bend, Bi, Bx);
            if status != BASICLU_OK {
                return return_to_caller();
            }

            this.task = FACTORIZE_BUMP;
            status = lu_factorize_bump(&this);
            if status != BASICLU_OK {
                return return_to_caller();
            }
        }
        FACTORIZE_BUMP => {
            // this.task = FACTORIZE_BUMP;
            status = lu_factorize_bump(&this);
            if status != BASICLU_OK {
                return return_to_caller();
            }
        }
        BUILD_FACTORS => {}
        _ => {
            status = BASICLU_ERROR_invalid_call;
            return lu_save(&this, istore, xstore, status);
        }
    };

    this.task = BUILD_FACTORS;
    status = lu_build_factors(&this);
    if status != BASICLU_OK {
        return return_to_caller();
    }

    // factorization successfully finished
    this.task = NO_TASK;
    this.nupdate = 0; // make factorization valid
    this.ftran_for_update = -1;
    this.btran_for_update = -1;
    this.nfactorize += 1;

    this.condestL = lu_condest(
        this.m,
        this.Lbegin,
        this.Lindex,
        this.Lvalue,
        NULL,
        this.p,
        0,
        this.work1,
        &this.normL,
        &this.normestLinv,
    );
    this.condestU = lu_condest(
        this.m,
        this.Ubegin,
        this.Uindex,
        this.Uvalue,
        this.row_pivot,
        this.p,
        1,
        this.work1,
        &this.normU,
        &this.normestUinv,
    );

    // measure numerical stability of the factorization
    lu_residual_test(&this, Bbegin, Bend, Bi, Bx);

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
    let factor_cost = 0.04 * this.m
        + 0.07 * this.matrix_nz
        + 0.20 * this.bump_nz
        + 0.20 * this.nsearch_pivot
        + 0.008 * this.factor_flops;

    this.update_cost_denom = factor_cost * 250;

    if cfg!(feature = "debug") {
        let elapsed = this.time_factorize + tic.elapsed();
        println!(
            " 1e-6 * factor_cost / time_factorize: {}",
            1e-6 * factor_cost / elapsed,
        );
    }

    if this.rank < this.m {
        status = BASICLU_WARNING_singular_matrix;
    }

    return_to_caller()
}
