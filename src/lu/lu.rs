// Copyright (C) 2016-2018 ERGO-Code
// Copyright (C) 2022-2023 Richard Lincoln

use crate::lu::def::Task;
use crate::LUInt;
use crate::LU_INT_MAX;

#[derive(Default)]
pub struct LU {
    /// length of `l_i` and `l_x`
    pub l_mem: LUInt,
    /// length of `u_i` and `u_x`
    pub u_mem: LUInt,
    /// length of `w_i` and `w_x`
    pub w_mem: LUInt,

    /// Nonzeros which magnitude is less than or equal to the drop tolerance
    /// are removed. They are guaranteed removed at the end of the factorization.
    /// Default: 1e-20
    pub droptol: f64,

    /// A pivot element must be nonzero and in absolute value must be greater
    /// than or equal to `abstol`. Default: 1e-14
    pub abstol: f64,
    /// A pivot element must be (in absolute value) greater than or equal to
    /// `reltol` times the largest entry in its column. A value greater than or
    /// equal to 1.0 is treated as 1.0 and enforces partial pivoting. Default: 0.1
    pub reltol: f64,

    /// When this value is greater than or equal to zero, the pivot choice
    /// attempts to keep L sparse, putting entries into U when possible.
    /// When this value is less than zero, the pivot choice attempts to keep U
    /// sparse, putting entries into L when possible. Default: 1
    pub nzbias: LUInt,

    /// The Markowitz search is terminated after searching `maxsearch` rows or
    /// columns if a numerically stable pivot element has been found. Default: 3
    pub maxsearch: LUInt,

    /// When a row or column cannot be updated by the pivot operation in place,
    /// it is appended to the end of the workspace. For a row or column with nz
    /// elements, `pad` + nz * `stretch` elements extra space are added for later
    /// fill-in. Default: 4
    pub pad: LUInt,
    /// Default: 0.3
    pub stretch: f64,

    pub compress_thres: f64,

    /// Defines which method is used for solving a triangular system. A
    /// triangular solve can be done either by the two phase method of Gilbert
    /// and Peierls ("sparse solve") or by a sequential pass through the vector
    /// ("sequential solve").
    ///
    /// Solving `B*x=b` requires two triangular solves. The first triangular solve
    /// is done sparse. The second triangular solve is done sparse if its
    /// right-hand side has not more than `m * sparse_thres` nonzeros. Otherwise
    /// the sequential solve is used.
    ///
    /// Default: 0.05
    pub sparse_thres: f64,

    /// If `search_rows` is zero, then the Markowitz search only scans columns.
    /// If nonzero, then both columns and rows are searched in increasing order of
    /// number of entries. Default: 1
    pub search_rows: LUInt,

    // user readable //
    pub(crate) m: LUInt,

    pub(crate) addmem_l: LUInt,
    pub(crate) addmem_u: LUInt,
    pub(crate) addmem_w: LUInt,

    pub(crate) nupdate: LUInt,
    pub(crate) nforrest: LUInt,
    pub(crate) nfactorize: LUInt,
    pub(crate) nupdate_total: LUInt,
    pub(crate) nforrest_total: LUInt,
    pub(crate) nsymperm_total: LUInt,
    pub(crate) l_nz: LUInt, // nz in L excluding diagonal
    pub(crate) u_nz: LUInt, // nz in U excluding diagonal
    pub(crate) r_nz: LUInt, // nz in update etas excluding diagonal
    pub(crate) min_pivot: f64,
    pub(crate) max_pivot: f64,
    pub(crate) max_eta: f64,
    pub(crate) update_cost_numer: f64,
    pub(crate) update_cost_denom: f64,
    pub(crate) time_factorize: f64,
    pub(crate) time_solve: f64,
    pub(crate) time_update: f64,
    pub(crate) time_factorize_total: f64,
    pub(crate) time_solve_total: f64,
    pub(crate) time_update_total: f64,
    pub(crate) l_flops: LUInt,
    pub(crate) u_flops: LUInt,
    pub(crate) r_flops: LUInt,
    pub(crate) condest_l: f64,
    pub(crate) condest_u: f64,
    pub(crate) norm_l: f64,
    pub(crate) norm_u: f64,
    pub(crate) normest_l_inv: f64,
    pub(crate) normest_u_inv: f64,
    pub(crate) onenorm: f64, // 1-norm and inf-norm of matrix after fresh
    pub(crate) infnorm: f64, // factorization with dependent cols replaced
    pub(crate) residual_test: f64, // computed by residual_test()

    pub(crate) matrix_nz: LUInt, // nz in basis matrix when factorized
    pub(crate) rank: LUInt,      // rank of basis matrix when factorized
    pub(crate) bump_size: LUInt,
    pub(crate) bump_nz: LUInt,
    pub(crate) nsearch_pivot: LUInt, // # rows/cols searched for pivot
    pub(crate) nexpand: LUInt,       // # rows/cols expanded in factorize
    pub(crate) ngarbage: LUInt,      // # garbage collections in factorize
    pub(crate) factor_flops: LUInt,  // # flops in factorize
    pub(crate) time_singletons: f64,
    pub(crate) time_search_pivot: f64,
    pub(crate) time_elim_pivot: f64,

    pub(crate) pivot_error: f64, // error estimate for pivot in last update

    // private //
    pub(crate) task: Task,              // the part of factorization in progress
    pub(crate) pivot_row: LUInt,        // chosen pivot row
    pub(crate) pivot_col: LUInt,        // chosen pivot column
    pub(crate) ftran_for_update: LUInt, // >= 0 if FTRAN prepared for update
    pub(crate) btran_for_update: LUInt, // >= 0 if BTRAN prepared for update
    pub(crate) marker: LUInt,           // see @marked, below
    pub(crate) pivotlen: LUInt,         // length of @pivotcol, @pivotrow; <= 2*m
    pub(crate) rankdef: LUInt,          // # columns removed from active submatrix
    // because maximum was 0 or < abstol
    pub(crate) min_colnz: LUInt, // colcount lists 1..min_colnz-1 are empty
    pub(crate) min_rownz: LUInt, // rowcount lists 1..min_rownz-1 are empty

    // Arrays used for workspace during the factorization and to store the
    // final factors.
    //
    // When the allocated length is insufficient to complete the factorization,
    // [`blu_factorize()`] returns to the caller for reallocation. A successful
    // factorization requires at least `nnz(B)` length for each of the arrays.
    pub(crate) l_index: Vec<LUInt>,
    pub(crate) u_index: Vec<LUInt>,
    pub(crate) w_index: Vec<LUInt>,

    pub(crate) l_value: Vec<f64>,
    pub(crate) u_value: Vec<f64>,
    pub(crate) w_value: Vec<f64>,

    pub(crate) colcount_flink: Vec<LUInt>, // pivotcol!
    pub(crate) colcount_blink: Vec<LUInt>, // pivotrow!
    pub(crate) rowcount_flink: Vec<LUInt>, // r_begin! eta_row!
    pub(crate) rowcount_blink: Vec<LUInt>, // iwork1!
    pub(crate) w_begin: Vec<LUInt>,        // l_begin!
    pub(crate) w_end: Vec<LUInt>,          // lt_begin!
    pub(crate) w_flink: Vec<LUInt>,        // lt_begin_p!
    pub(crate) w_blink: Vec<LUInt>,        // p!
    pub(crate) pinv: Vec<LUInt>,           // pmap!
    pub(crate) qinv: Vec<LUInt>,           // qmap!
    pub(crate) l_begin_p: Vec<LUInt>,
    pub(crate) u_begin: Vec<LUInt>,

    // !marked
    // iwork0: size m workspace, zeroed
    // marked: size m workspace, 0 <= marked[i] <= marker
    pub(crate) iwork0: Vec<LUInt>,

    pub(crate) work0: Vec<f64>,     // size m workspace, zeroed
    pub(crate) work1: Vec<f64>,     // size m workspace, uninitialized
    pub(crate) col_pivot: Vec<f64>, // pivot elements by column index
    pub(crate) row_pivot: Vec<f64>, // pivot elements by row index
}

macro_rules! pivotcol {
    ($lu:ident) => {
        $lu.colcount_flink
    };
}

macro_rules! pivotrow {
    ($lu:ident) => {
        $lu.colcount_blink
    };
}
macro_rules! r_begin {
    ($lu:ident) => {
        $lu.rowcount_flink
    };
}
macro_rules! eta_row {
    ($lu:ident) => {
        $lu.rowcount_flink
    };
}
macro_rules! iwork1 {
    ($lu:ident) => {
        $lu.rowcount_blink
    };
}
macro_rules! l_begin {
    ($lu:ident) => {
        $lu.w_begin[$lu.m as usize + 1..]
    };
}
macro_rules! lt_begin {
    ($lu:ident) => {
        $lu.w_end[$lu.m as usize + 1..]
    };
}
macro_rules! lt_begin_p {
    ($lu:ident) => {
        $lu.w_flink[$lu.m as usize + 1..]
    };
}
macro_rules! p {
    ($lu:ident) => {
        $lu.w_blink[$lu.m as usize + 1..]
    };
}
macro_rules! pmap {
    ($lu:ident) => {
        $lu.pinv
    };
}
macro_rules! qmap {
    ($lu:ident) => {
        $lu.qinv
    };
}
macro_rules! marked {
    ($lu:ident) => {
        $lu.iwork0
    };
}

pub(crate) use {
    eta_row, iwork1, l_begin, lt_begin, lt_begin_p, marked, p, pivotcol, pivotrow, pmap, qmap,
    r_begin,
};

impl LU {
    /// Make a BLU instance. Set parameters to defaults and initialize global counters.
    /// Reset instance for a fresh factorization.
    pub fn new(m: LUInt, b_nz: LUInt) -> Self {
        let mut lu = LU {
            l_mem: b_nz,
            u_mem: b_nz,
            w_mem: b_nz,

            // set default parameters
            droptol: 1e-20,
            abstol: 1e-14,
            reltol: 0.1,
            nzbias: 1,
            maxsearch: 3,
            pad: 4,
            stretch: 0.3,
            compress_thres: 0.5,
            sparse_thres: 0.05,
            search_rows: 0,

            m,

            // initialize global counters
            nfactorize: 0,
            nupdate_total: 0,
            nforrest_total: 0,
            nsymperm_total: 0,
            time_factorize_total: 0.0,
            time_solve_total: 0.0,
            time_update_total: 0.0,

            l_index: vec![0; b_nz as usize],
            u_index: vec![0; b_nz as usize],
            w_index: vec![0; b_nz as usize],

            l_value: vec![0.0; b_nz as usize],
            u_value: vec![0.0; b_nz as usize],
            w_value: vec![0.0; b_nz as usize],

            colcount_flink: vec![0; 2 * m as usize + 2],
            colcount_blink: vec![0; 2 * m as usize + 2],
            rowcount_flink: vec![0; 2 * m as usize + 2],
            rowcount_blink: vec![0; 2 * m as usize + 2],
            w_begin: vec![0; 2 * m as usize + 2],
            w_end: vec![0; 2 * m as usize + 2],
            w_flink: vec![0; 2 * m as usize + 2],
            w_blink: vec![0; 2 * m as usize + 2],
            pinv: vec![0; m as usize],
            qinv: vec![0; m as usize],
            l_begin_p: vec![0; m as usize + 1],
            u_begin: vec![0; m as usize + 1],
            iwork0: vec![0; m as usize],

            work0: vec![0.0; m as usize],
            work1: vec![0.0; m as usize],
            col_pivot: vec![0.0; m as usize],
            row_pivot: vec![0.0; m as usize],

            ..Default::default()
        };

        // Reset marked if increasing marker by four causes overflow.
        if lu.marker > LU_INT_MAX - 4 {
            marked!(lu).fill(0);
            lu.marker = 0;
        }

        // One past the final position in @Wend must hold the file size.
        // The file has 2*m lines while factorizing and m lines otherwise.
        if lu.nupdate >= 0 {
            lu.w_end[m as usize] = lu.w_mem;
        } else {
            lu.w_end[2 * m as usize] = lu.w_mem;
        }

        lu.reset();

        lu
    }

    /// Deterministic measure of solve/update cost compared to cost of last factorization. This
    /// value is zero after factorization and monotonically increases with solves/updates.
    /// When > 1.0, then a refactorization is good for performance.
    pub fn update_cost(&mut self) -> f64 {
        self.update_cost_numer / self.update_cost_denom
    }

    /// Reset for a new factorization. Invalidate current factorization.
    pub fn reset(&mut self) {
        // user readable
        self.nupdate = -1; // invalidate factorization
        self.nforrest = 0;
        self.l_nz = 0;
        self.u_nz = 0;
        self.r_nz = 0;
        self.min_pivot = 0.0;
        self.max_pivot = 0.0;
        self.max_eta = 0.0;
        self.update_cost_numer = 0.0;
        self.update_cost_denom = 1.0;
        self.time_factorize = 0.0;
        self.time_solve = 0.0;
        self.time_update = 0.0;
        self.l_flops = 0;
        self.u_flops = 0;
        self.r_flops = 0;
        self.condest_l = 0.0;
        self.condest_u = 0.0;
        self.norm_l = 0.0;
        self.norm_u = 0.0;
        self.normest_l_inv = 0.0;
        self.normest_u_inv = 0.0;
        self.onenorm = 0.0;
        self.infnorm = 0.0;
        self.residual_test = 0.0;

        self.matrix_nz = 0;
        self.rank = 0;
        self.bump_size = 0;
        self.bump_nz = 0;
        self.nsearch_pivot = 0;
        self.nexpand = 0;
        self.ngarbage = 0;
        self.factor_flops = 0;
        self.time_singletons = 0.0;
        self.time_search_pivot = 0.0;
        self.time_elim_pivot = 0.0;

        self.pivot_error = 0.0;

        // private
        self.task = Task::NoTask;
        self.pivot_row = -1;
        self.pivot_col = -1;
        self.ftran_for_update = -1;
        self.btran_for_update = -1;
        self.marker = 0;
        self.pivotlen = 0;
        self.rankdef = 0;
        self.min_colnz = 1;
        self.min_rownz = 1;

        // One past the final position in @Wend must hold the file size.
        // The file has 2*m lines during factorization.
        self.w_end[2 * self.m as usize] = self.w_mem;

        // The integer workspace iwork0 must be zeroed for a new factorization.
        // The double workspace work0 actually needs only be zeroed once in the
        // initialization of xstore. However, it is easier and more consistent
        // to do that here as well.
        // memset(self.iwork0, 0, self.m);
        self.iwork0.fill(0);

        // memset(self.work0, 0, self.m);
        self.work0.fill(0.0);
    }

    /// Matrix dimension (constant).
    pub fn m(&self) -> LUInt {
        self.m
    }

    /// Factorization requires more memory in `l_i`,`l_x`. The number of additional
    /// elements in each of the array pairs required for the next pivot operation.
    ///
    /// The user must reallocate the arrays for which additional memory is
    /// required. It is recommended to reallocate for the requested number
    /// of additional elements plus some extra space (e.g. 0.5 times the
    /// current array length). The new array lengths must be provided in `l_mem`.
    ///
    /// [`blu_factorize()`] can be called again with `c0ntinue` not equal to
    /// zero to continue the factorization.
    pub fn addmem_l(&self) -> LUInt {
        self.addmem_l
    }

    /// Factorization requires more memory in `u_i`,`u_x`. The number of additional
    /// elements in each of the array pairs required for the next pivot operation.
    pub fn addmem_u(&self) -> LUInt {
        self.addmem_u
    }

    /// Factorization requires more memory in `w_i`,`w_x`. The number of additional
    /// elements in each of the array pairs required for the next pivot operation.
    pub fn addmem_w(&self) -> LUInt {
        self.addmem_w
    }

    /// Number of updates since last factorization. This is
    /// the sum of Forrest-Tomlin updates and permutation
    /// updates.
    pub fn nupdate(&self) -> LUInt {
        self.nupdate
    }

    /// Number of Forrest-Tomlin updates since last factorization.
    /// The upper limit on Forrest-Tomlin updates before refactorization is `m`,
    /// but that is far too much for performance reasons and numerical stability.
    pub fn nforrest(&self) -> LUInt {
        self.nforrest
    }

    /// Number of factorizations since initialization.
    pub fn nfactorize(&self) -> LUInt {
        self.nfactorize
    }

    /// Number of updates since initialization.
    pub fn nupdate_total(&self) -> LUInt {
        self.nupdate_total
    }

    /// Number of Forrest-Tomlin updates since initialization.
    pub fn nforrest_total(&self) -> LUInt {
        self.nforrest_total
    }

    /// Number of symmetric permutation updates since initialization.
    /// A permutation update is "symmetric" if the row and column
    /// permutation can be updated symmetrically.
    pub fn nsymperm_total(&self) -> LUInt {
        self.nsymperm_total
    }

    /// Number of nonzeros in `L` excluding diagonal elements (not changed by updates).
    pub fn l_nz(&self) -> LUInt {
        self.l_nz
    }

    /// Number of nonzeros in `U` excluding diagonal elements (changed by updates).
    pub fn u_nz(&self) -> LUInt {
        self.u_nz
    }

    /// Number of nonzeros in update ETA vectors excluding diagonal elements (zero after
    /// factorization, increased by Forrest-Tomlin updates).
    pub fn r_nz(&self) -> LUInt {
        self.r_nz
    }

    /// The smallest pivot element after factorization.
    /// Replaced when a smaller pivot occurs in an update.
    pub fn min_pivot(&self) -> f64 {
        self.min_pivot
    }

    /// The largest pivot element after factorization.
    /// Replaced when a larger pivot occurs in an update.
    pub fn max_pivot(&self) -> f64 {
        self.max_pivot
    }

    /// The maximum entry (in absolute value) in the eta vectors from the
    /// Forrest-Tomlin update. A large value, say > 1e6, indicates that pivoting
    /// on diagonal element was unstable and refactorization might be necessary.
    pub fn max_eta(&self) -> f64 {
        self.max_eta
    }

    /// Wall clock time for last factorization.
    pub fn time_factorize(&self) -> f64 {
        self.time_factorize
    }

    /// Wall clock time for all calls to [`blu_solve_sparse()`] and
    /// [`blu_solve_for_update`] since last factorization.
    pub fn time_solve(&self) -> f64 {
        self.time_solve
    }

    /// Wall clock time for all calls to [`blu_update`] since last factorization.
    pub fn time_update(&self) -> f64 {
        self.time_update
    }

    /// Analogous to above, but summing up all calls since initialization.
    pub fn time_factorize_total(&self) -> f64 {
        self.time_factorize_total
    }

    /// Analogous to above, but summing up all calls since initialization.
    pub fn time_solve_total(&self) -> f64 {
        self.time_solve_total
    }

    /// Analogous to above, but summing up all calls since initialization.
    pub fn time_update_total(&self) -> f64 {
        self.time_update_total
    }

    /// Number of flops for operations with `L` vectors in calls to
    /// [`blu_solve_sparse`] and [`blu_solve_for_update`] since last factorization.
    pub fn l_flops(&self) -> LUInt {
        self.l_flops
    }

    /// Number of flops for operations with `U` vectors in calls to
    /// [`blu_solve_sparse`] and [`blu_solve_for_update`] since last factorization.
    pub fn u_flops(&self) -> LUInt {
        self.u_flops
    }

    /// Number of flops for operations with update ETA vectors in calls to
    /// [`blu_solve_sparse`] and [`blu_solve_for_update`] since last factorization.
    pub fn r_flops(&self) -> LUInt {
        self.r_flops
    }

    /// Estimated 1-norm condition number of `L`.
    pub fn condest_l(&self) -> f64 {
        self.condest_l
    }

    /// Estimated 1-norm condition number of `U`.
    pub fn condest_u(&self) -> f64 {
        self.condest_u
    }

    /// 1-norm of `L`.
    pub fn norm_l(&self) -> f64 {
        self.norm_l
    }

    /// 1-norm of `U`.
    pub fn norm_u(&self) -> f64 {
        self.norm_u
    }

    /// Estimated 1-norm of `L^{-1}`, computed by the LINPACK algorithm.
    pub fn normest_l_inv(&self) -> f64 {
        self.normest_l_inv
    }

    /// Estimated 1-norm of `U^{-1}`, computed by the LINPACK algorithm.
    pub fn normest_u_inv(&self) -> f64 {
        self.normest_u_inv
    }

    /// 1-norm of the input matrix after replacing dependent columns by unit columns.
    pub fn onenorm(&self) -> f64 {
        self.onenorm
    }

    /// Inf-norm of the input matrix after replacing dependent columns by unit columns.
    pub fn infnorm(&self) -> f64 {
        self.infnorm
    }

    /// An estimate for numerical stability of the factorization.
    /// `residual_test` is the maximum of the scaled residuals
    ///
    ///     ||b-b_x|| / (||b|| + ||B||*||x||)
    ///
    /// and
    ///
    ///     ||c-B'y|| / (||c|| + ||B'||*||y||),
    ///
    /// where `x=B\b` and `y=B'\c` are computed from the LU factors, `b` and `c`
    /// have components +/-1 that are chosen to make `x` respectively `y` large,
    /// and `||.||` is the 1-norm. Here `B` is the input matrix after replacing
    /// dependent columns by unit columns.
    ///
    /// If `residual_test` > 1e-12, say, the factorization is numerically unstable.
    /// (This is independent of the condition number of B.) In this case tightening
    /// the relative pivot tolerance and refactorizing is appropriate.
    pub fn residual_test(&self) -> f64 {
        // computed by residual_test()
        self.residual_test
    }

    /// Number of nonzeros in basis matrix (`B`) when factorized.
    pub fn matrix_nz(&self) -> LUInt {
        self.matrix_nz
    }

    /// number of pivot steps performed
    pub fn rank(&self) -> LUInt {
        // rank of basis matrix when factorized
        self.rank
    }

    /// Dimension of matrix after removing singletons.
    pub fn bump_size(&self) -> LUInt {
        self.bump_size
    }

    /// Number of nonzeros in matrix after removing singletons.
    pub fn bump_nz(&self) -> LUInt {
        self.bump_nz
    }

    /// Total number of columns/rows searched for pivots.
    pub fn nsearch_pivot(&self) -> LUInt {
        self.nsearch_pivot
    }

    /// Number of columns/rows which had to be appended to the end
    /// of the workspace for the rank-1 update.
    pub fn nexpand(&self) -> LUInt {
        self.nexpand
    }

    /// Number of garbage collections in factorize.
    pub fn ngarbage(&self) -> LUInt {
        self.ngarbage
    }

    /// Number of floating point operations performed in factorize,
    /// counting multiply-add as one flop.
    pub fn factor_flops(&self) -> LUInt {
        self.factor_flops
    }

    /// Wall clock time for removing the initial triangular factors.
    pub fn time_singletons(&self) -> f64 {
        self.time_singletons
    }

    /// Wall clock time for Markowitz search.
    pub fn time_search_pivot(&self) -> f64 {
        self.time_search_pivot
    }

    /// Wall clock time for pivot elimination.
    pub fn time_elim_pivot(&self) -> f64 {
        self.time_elim_pivot
    }

    /// A measure for numerical stability. It is the difference between two
    /// computations of the new pivot element relative to the new pivot element.
    /// A value larger than 1e-10 indicates numerical instability and suggests
    /// refactorization (and possibly tightening the pivot tolerance).
    pub fn pivot_error(&self) -> f64 {
        // error estimate for pivot in last update
        self.pivot_error
    }
}
