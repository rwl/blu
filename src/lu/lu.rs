// Copyright (C) 2016-2018 ERGO-Code
// Copyright (C) 2022-2023 Richard Lincoln

use crate::blu::*;
use crate::lu::def::{BLU_HASH, NO_TASK};

// private entries in xstore
pub(crate) const BLU_TASK: usize = 256;
pub(crate) const BLU_FTCOLUMN_IN: usize = 257;
pub(crate) const BLU_FTCOLUMN_OUT: usize = 258;
pub(crate) const BLU_PIVOT_ROW: usize = 259;
pub(crate) const BLU_PIVOT_COL: usize = 260;
pub(crate) const BLU_RANKDEF: usize = 261;
pub(crate) const BLU_MIN_COLNZ: usize = 262;
pub(crate) const BLU_MIN_ROWNZ: usize = 263;
pub(crate) const BLU_MARKER: usize = 266;
pub(crate) const BLU_UPDATE_COST_NUMER: usize = 267;
pub(crate) const BLU_UPDATE_COST_DENOM: usize = 268;
pub(crate) const BLU_PIVOTLEN: usize = 269;

/// This data structure provides access to istore, xstore.
///
/// lu_* routines do not access istore, xstore directly. Instead, they operate
/// on a struct lu object. Scalar quantities stored in istore, xstore are copied
/// to a struct lu object by lu_load() and copied back by lu_save(). Subarrays
/// of istore, xstore and the user arrays Li, Lx, Ui, Ux, Wi, Wx are aliased by
/// pointers in struct lu.
#[derive(Default)]
pub struct LU {
    // user parameters, not modified //
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
    /// Matrix dimension (constant).
    pub m: LUInt,

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
    pub addmem_l: LUInt,
    /// Factorization requires more memory in `u_i`,`u_x`. The number of additional
    /// elements in each of the array pairs required for the next pivot operation.
    pub addmem_u: LUInt,
    /// Factorization requires more memory in `w_i`,`w_x`. The number of additional
    /// elements in each of the array pairs required for the next pivot operation.
    pub addmem_w: LUInt,

    /// Number of updates since last factorization. This is
    /// the sum of Forrest-Tomlin updates and permutation
    /// updates.
    pub nupdate: LUInt,
    /// Number of Forrest-Tomlin updates since last factorization.
    /// The upper limit on Forrest-Tomlin updates before refactorization is `m`,
    /// but that is far too much for performance reasons and numerical stability.
    pub nforrest: LUInt,
    /// Number of factorizations since initialization.
    pub nfactorize: LUInt,
    /// Number of updates since initialization.
    pub nupdate_total: LUInt,
    /// Number of Forrest-Tomlin updates since initialization.
    pub nforrest_total: LUInt,
    /// Number of symmetric permutation updates since initialization.
    /// A permutation update is "symmetric" if the row and column
    /// permutation can be updated symmetrically.
    pub nsymperm_total: LUInt,
    /// Number of nonzeros in `L` excluding diagonal elements (not changed by updates).
    pub l_nz: LUInt,
    /// Number of nonzeros in `U` excluding diagonal elements (changed by updates).
    pub u_nz: LUInt,
    /// Number of nonzeros in update ETA vectors excluding diagonal elements (zero after
    /// factorization, increased by Forrest-Tomlin updates).
    pub r_nz: LUInt,
    /// The smallest pivot element after factorization.
    /// Replaced when a smaller pivot occurs in an update.
    pub min_pivot: f64,
    /// The largest pivot element after factorization.
    /// Replaced when a larger pivot occurs in an update.
    pub max_pivot: f64,
    /// The maximum entry (in absolute value) in the eta vectors from the
    /// Forrest-Tomlin update. A large value, say > 1e6, indicates that pivoting
    /// on diagonal element was unstable and refactorization might be necessary.
    pub max_eta: f64,
    pub update_cost_numer: f64,
    pub update_cost_denom: f64,
    /// Wall clock time for last factorization.
    pub time_factorize: f64,
    /// Wall clock time for all calls to [`blu_solve_sparse()`] and
    /// [`blu_solve_for_update`] since last factorization.
    pub time_solve: f64,
    /// Wall clock time for all calls to [`blu_update`] since last factorization.
    pub time_update: f64,
    /// Analogous to above, but summing up all calls since initialization.
    pub time_factorize_total: f64,
    /// Analogous to above, but summing up all calls since initialization.
    pub time_solve_total: f64,
    /// Analogous to above, but summing up all calls since initialization.
    pub time_update_total: f64,
    /// Number of flops for operations with `L` vectors in calls to
    /// [`blu_solve_sparse`] and [`blu_solve_for_update`] since last factorization.
    pub l_flops: LUInt,
    /// Number of flops for operations with `U` vectors in calls to
    /// [`blu_solve_sparse`] and [`blu_solve_for_update`] since last factorization.
    pub u_flops: LUInt,
    /// Number of flops for operations with update ETA vectors in calls to
    /// [`blu_solve_sparse`] and [`blu_solve_for_update`] since last factorization.
    pub r_flops: LUInt,
    /// Estimated 1-norm condition number of `L`.
    pub condest_l: f64,
    /// Estimated 1-norm condition number of `U`.
    pub condest_u: f64,
    /// 1-norm of `L`.
    pub norm_l: f64,
    /// 1-norm of `U`.
    pub norm_u: f64,
    /// Estimated 1-norm of `L^{-1}`, computed by the LINPACK algorithm.
    pub normest_l_inv: f64,
    /// Estimated 1-norm of `U^{-1}`, computed by the LINPACK algorithm.
    pub normest_u_inv: f64,
    /// 1-norm of the input matrix after replacing dependent columns by unit columns.
    pub onenorm: f64,
    /// Inf-norm of the input matrix after replacing dependent columns by unit columns.
    pub infnorm: f64,
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
    pub residual_test: f64, // computed by residual_test()

    /// Number of nonzeros in basis matrix (`B`) when factorized.
    pub matrix_nz: LUInt,
    /// number of pivot steps performed
    pub rank: LUInt, // rank of basis matrix when factorized
    /// Dimension of matrix after removing singletons.
    pub bump_size: LUInt,
    /// Number of nonzeros in matrix after removing singletons.
    pub bump_nz: LUInt,
    /// Total number of columns/rows searched for pivots.
    pub nsearch_pivot: LUInt,
    /// Number of columns/rows which had to be appended to the end
    /// of the workspace for the rank-1 update.
    pub nexpand: LUInt,
    /// Number of garbage collections in factorize.
    pub ngarbage: LUInt,
    /// Number of floating point operations performed in factorize,
    /// counting multiply-add as one flop.
    pub factor_flops: LUInt,
    /// Wall clock time for removing the initial triangular factors.
    pub time_singletons: f64,
    /// Wall clock time for Markowitz search.
    pub time_search_pivot: f64,
    /// Wall clock time for pivot elimination.
    pub time_elim_pivot: f64,

    /// A measure for numerical stability. It is the difference between two
    /// computations of the new pivot element relative to the new pivot element.
    /// A value larger than 1e-10 indicates numerical instability and suggests
    /// refactorization (and possibly tightening the pivot tolerance).
    pub pivot_error: f64, // error estimate for pivot in last update

    // private //
    pub(crate) task: LUInt,      // the part of factorization in progress
    pub(crate) pivot_row: LUInt, // chosen pivot row
    pub(crate) pivot_col: LUInt, // chosen pivot column
    pub(crate) ftran_for_update: LUInt, // >= 0 if FTRAN prepared for update
    pub(crate) btran_for_update: LUInt, // >= 0 if BTRAN prepared for update
    pub(crate) marker: LUInt,    // see @marked, below
    pub(crate) pivotlen: LUInt,  // length of @pivotcol, @pivotrow; <= 2*m
    pub(crate) rankdef: LUInt,   // # columns removed from active submatrix
    // because maximum was 0 or < abstol
    pub(crate) min_colnz: LUInt, // colcount lists 1..min_colnz-1 are empty
    pub(crate) min_rownz: LUInt, // rowcount lists 1..min_rownz-1 are empty

    /// Arrays used for workspace during the factorization and to store the
    /// final factors.
    ///
    /// When the allocated length is insufficient to complete the factorization,
    /// [`blu_factorize()`] returns to the caller for reallocation. A successful
    /// factorization requires at least `nnz(B)` length for each of the arrays.
    pub(crate) l_index: Vec<LUInt>,
    pub(crate) u_index: Vec<LUInt>,
    pub(crate) w_index: Vec<LUInt>,

    pub(crate) l_value: Vec<f64>,
    pub(crate) u_value: Vec<f64>,
    pub(crate) w_value: Vec<f64>,

    pub(crate) colcount_flink: Vec<LUInt>,
    // pub(crate) pivotcol: Vec<lu_int>,
    pub(crate) colcount_blink: Vec<LUInt>,
    // pub(crate) pivotrow: Vec<lu_int>,
    pub(crate) rowcount_flink: Vec<LUInt>,
    // pub(crate) r_begin: Vec<lu_int>,
    // pub(crate) eta_row: Vec<lu_int>,
    pub(crate) rowcount_blink: Vec<LUInt>,
    // pub(crate) iwork1: Vec<lu_int>,
    pub(crate) w_begin: Vec<LUInt>,
    // pub(crate) l_begin: Vec<lu_int>, // + Wbegin reused
    pub(crate) w_end: Vec<LUInt>,
    // pub(crate) lt_begin: Vec<lu_int>, // + Wend   reused
    pub(crate) w_flink: Vec<LUInt>,
    // pub(crate) lt_begin_p: Vec<lu_int>, // + Wflink reused
    pub(crate) w_blink: Vec<LUInt>,
    // pub(crate) p: Vec<lu_int>, // + Wblink reused
    pub(crate) pinv: Vec<LUInt>,
    // pub(crate) pmap: Vec<lu_int>,
    pub(crate) qinv: Vec<LUInt>,
    // pub(crate) qmap: Vec<lu_int>,
    pub(crate) l_begin_p: Vec<LUInt>, // Lbegin_p reused
    pub(crate) u_begin: Vec<LUInt>,   // Ubegin   reused

    pub(crate) iwork0: Vec<LUInt>,
    // pub(crate) marked: Vec<lu_int>,
    // iwork0: size m workspace, zeroed
    // marked: size m workspace, 0 <= marked[i] <= @marker
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
    // Initialize @lu from @istore, @xstore if these are a valid BLU
    // instance. The remaining arguments are copied only and can be NULL.
    //
    // Return `BLU_OK` or `BLU_ERROR_INVALID_STORE`
    pub(crate) fn load(&mut self, xstore: &[f64]) -> LUInt {
        if xstore[0] != BLU_HASH as f64 {
            return BLU_ERROR_INVALID_STORE;
        }

        // user parameters
        self.l_mem = xstore[BLU_MEMORYL] as LUInt;
        self.u_mem = xstore[BLU_MEMORYU] as LUInt;
        self.w_mem = xstore[BLU_MEMORYW] as LUInt;
        self.droptol = xstore[BLU_DROP_TOLERANCE];
        self.abstol = xstore[BLU_ABS_PIVOT_TOLERANCE];
        self.reltol = xstore[BLU_REL_PIVOT_TOLERANCE];
        self.reltol = f64::min(self.reltol, 1.0);
        self.nzbias = xstore[BLU_BIAS_NONZEROS] as LUInt;
        self.maxsearch = xstore[BLU_MAXN_SEARCH_PIVOT] as LUInt;
        self.pad = xstore[BLU_PAD] as LUInt;
        self.stretch = xstore[BLU_STRETCH];
        self.compress_thres = xstore[BLU_COMPRESSION_THRESHOLD];
        self.sparse_thres = xstore[BLU_SPARSE_THRESHOLD];
        self.search_rows = if xstore[BLU_SEARCH_ROWS] != 0.0 { 1 } else { 0 };

        // user readable
        let m = xstore[BLU_DIM];
        self.m = m as LUInt;
        self.addmem_l = 0;
        self.addmem_u = 0;
        self.addmem_w = 0;

        self.nupdate = xstore[BLU_NUPDATE] as LUInt;
        self.nforrest = xstore[BLU_NFORREST] as LUInt;
        self.nfactorize = xstore[BLU_NFACTORIZE] as LUInt;
        self.nupdate_total = xstore[BLU_NUPDATE_TOTAL] as LUInt;
        self.nforrest_total = xstore[BLU_NFORREST_TOTAL] as LUInt;
        self.nsymperm_total = xstore[BLU_NSYMPERM_TOTAL] as LUInt;
        self.l_nz = xstore[BLU_LNZ] as LUInt;
        self.u_nz = xstore[BLU_UNZ] as LUInt;
        self.r_nz = xstore[BLU_RNZ] as LUInt;
        self.min_pivot = xstore[BLU_MIN_PIVOT];
        self.max_pivot = xstore[BLU_MAX_PIVOT];
        self.max_eta = xstore[BLU_MAX_ETA];
        self.update_cost_numer = xstore[BLU_UPDATE_COST_NUMER];
        self.update_cost_denom = xstore[BLU_UPDATE_COST_DENOM];
        self.time_factorize = xstore[BLU_TIME_FACTORIZE];
        self.time_solve = xstore[BLU_TIME_SOLVE];
        self.time_update = xstore[BLU_TIME_UPDATE];
        self.time_factorize_total = xstore[BLU_TIME_FACTORIZE_TOTAL];
        self.time_solve_total = xstore[BLU_TIME_SOLVE_TOTAL];
        self.time_update_total = xstore[BLU_TIME_UPDATE_TOTAL];
        self.l_flops = xstore[BLU_LFLOPS] as LUInt;
        self.u_flops = xstore[BLU_UFLOPS] as LUInt;
        self.r_flops = xstore[BLU_RFLOPS] as LUInt;
        self.condest_l = xstore[BLU_CONDEST_L];
        self.condest_u = xstore[BLU_CONDEST_U];
        self.norm_l = xstore[BLU_NORM_L];
        self.norm_u = xstore[BLU_NORM_U];
        self.normest_l_inv = xstore[BLU_NORMEST_LINV];
        self.normest_u_inv = xstore[BLU_NORMEST_UINV];
        self.onenorm = xstore[BLU_MATRIX_ONENORM];
        self.infnorm = xstore[BLU_MATRIX_INFNORM];
        self.residual_test = xstore[BLU_RESIDUAL_TEST];

        self.matrix_nz = xstore[BLU_MATRIX_NZ] as LUInt;
        self.rank = xstore[BLU_RANK] as LUInt;
        self.bump_size = xstore[BLU_BUMP_SIZE] as LUInt;
        self.bump_nz = xstore[BLU_BUMP_NZ] as LUInt;
        self.nsearch_pivot = xstore[BLU_NSEARCH_PIVOT] as LUInt;
        self.nexpand = xstore[BLU_NEXPAND] as LUInt;
        self.ngarbage = xstore[BLU_NGARBAGE] as LUInt;
        self.factor_flops = xstore[BLU_FACTOR_FLOPS] as LUInt;
        self.time_singletons = xstore[BLU_TIME_SINGLETONS];
        self.time_search_pivot = xstore[BLU_TIME_SEARCH_PIVOT];
        self.time_elim_pivot = xstore[BLU_TIME_ELIM_PIVOT];

        self.pivot_error = xstore[BLU_PIVOT_ERROR];

        // private
        self.task = xstore[BLU_TASK] as LUInt;
        self.pivot_row = xstore[BLU_PIVOT_ROW] as LUInt;
        self.pivot_col = xstore[BLU_PIVOT_COL] as LUInt;
        self.ftran_for_update = xstore[BLU_FTCOLUMN_IN] as LUInt;
        self.btran_for_update = xstore[BLU_FTCOLUMN_OUT] as LUInt;
        self.marker = xstore[BLU_MARKER] as LUInt;
        self.pivotlen = xstore[BLU_PIVOTLEN] as LUInt;
        self.rankdef = xstore[BLU_RANKDEF] as LUInt;
        self.min_colnz = xstore[BLU_MIN_COLNZ] as LUInt;
        self.min_rownz = xstore[BLU_MIN_ROWNZ] as LUInt;

        // aliases to user arrays
        // self.Lindex = Li;
        // self.Lvalue = Lx;
        // self.Uindex = Ui;
        // self.Uvalue = Ux;
        // self.Windex = Wi;
        // self.Wvalue = Wx;
        // self.Lindex = match Li {
        //     Some(Li) => Some(Li.to_vec()),
        //     None => None,
        // };
        // self.Lvalue = match Lx {
        //     Some(Lx) => Some(Lx.to_vec()),
        //     None => None,
        // };
        // self.Uindex = match Ui {
        //     Some(Ui) => Some(Ui.to_vec()),
        //     None => None,
        // };
        // self.Uvalue = match Ux {
        //     Some(Ux) => Some(Ux.to_vec()),
        //     None => None,
        // };
        // self.Windex = match Wi {
        //     Some(Wi) => Some(Wi.to_vec()),
        //     None => None,
        // };
        // self.Wvalue = match Wx {
        //     Some(Wx) => Some(Wx.to_vec()),
        //     None => None,
        // };

        // // partition istore for factorize
        // self.colcount_flink = vec![0; 2 * m as usize + 2];
        // self.pivotcol = vec![];
        // // iptr += 2 * m + 2;
        // self.colcount_blink = Some(vec![0; 2 * m as usize + 2]);
        // // iptr += 2 * m + 2;
        // self.rowcount_flink = Some(vec![0; 2 * m as usize + 2]);
        // // iptr += 2 * m + 2;
        // self.rowcount_blink = Some(vec![0; 2 * m as usize + 2]);
        // // iptr += 2 * m + 2;
        // self.Wbegin = Some(vec![0; 2 * m as usize + 2]);
        // // iptr += 2 * m + 1;
        // self.Wend = Some(vec![0; 2 * m as usize + 2]);
        // // iptr += 2 * m + 1;
        // self.Wflink = Some(vec![0; 2 * m as usize + 2]);
        // // iptr += 2 * m + 1;
        // self.Wblink = Some(vec![0; 2 * m as usize + 2]);
        // // iptr += 2 * m + 1;
        // self.pinv = Some(vec![0; m as usize]);
        // // iptr += m;
        // self.qinv = Some(vec![0; m as usize]);
        // // iptr += m;
        // self.Lbegin_p = vec![0; m as usize + 1];
        // // iptr += m + 1;
        // self.Ubegin = vec![0; m as usize + 1];
        // // iptr += m + 1;
        // self.iwork0 = Some(vec![0; m as usize]);
        // // iptr += m;
        //
        // // share istore memory for solve/update
        // swap(&mut self.pivotcol, &mut self.colcount_flink);
        // self.pivotrow = self.colcount_blink.take();
        // self.r_begin = self.rowcount_flink.take(); // FIXME: [..m+1]
        //                                           // self.eta_row = self.rowcount_flink + m + 1;
        //                                           // self.eta_row = self.rowcount_flink[m as usize + 1..].to_vec();
        // self.eta_row = vec![0; m as usize + 1]; // FIXME: rowcount_flink[m+1..]
        // self.iwork1 = self.rowcount_blink.take();
        // // self.l_begin = self.Wbegin + m + 1;
        // // self.l_begin = self.Wbegin[m as usize + 1..].to_vec();
        // self.l_begin = self.Wbegin.take(); // [m+1..]
        //                                   // self.lt_begin = self.Wend + m + 1;
        //                                   // self.lt_begin = self.Wend[m as usize + 1..].to_vec();
        // self.lt_begin = self.Wend.take(); // [m+1..]
        //                                  // self.lt_begin_p = self.Wflink + m + 1;
        //                                  // self.lt_begin_p = self.Wflink[m as usize + 1..].to_vec();
        // self.lt_begin_p = self.Wflink.take(); // [m+1..]
        //                                      // self.p = self.Wblink + m + 1;
        //                                      // self.p = self.Wblink[m as usize + 1..].to_vec();
        // self.p = self.Wblink.take(); // [m+1..]
        // self.pmap = self.pinv.take();
        // self.qmap = self.qinv.take();
        // self.marked = self.iwork0.take();
        //
        // // partition xstore for factorize and update
        // // let xptr = xstore + 512;
        // // let (_, xstore) = xstore.split_at(512);
        // // let (work0, xstore) = xstore.split_at(m as usize);
        // self.work0 = vec![0.0; m as usize];
        // // xptr += m;
        // let (work1, xstore) = xstore.split_at(m as usize);
        // // self.work1 = Vec::from(work1);
        // self.work1 = vec![0.0; m as usize];
        // // xptr += m;
        // // let (col_pivot, xstore) = xstore.split_at(m as usize);
        // // self.col_pivot = Vec::from(col_pivot);
        // self.col_pivot = vec![0.0; m as usize];
        // // xptr += m;
        // // let (row_pivot, _) = xstore.split_at(m as usize);
        // // self.row_pivot = Vec::from(row_pivot);
        // self.row_pivot = vec![0.0; m as usize];
        // // xptr += m;

        // Reset @marked if increasing @marker by four causes overflow.
        if self.marker > LU_INT_MAX - 4 {
            // memset(self.marked, 0, m * sizeof(lu_int));
            marked!(self).fill(0);
            self.marker = 0;
        }

        // One past the final position in @Wend must hold the file size.
        // The file has 2*m lines while factorizing and m lines otherwise.
        if self.nupdate >= 0 {
            self.w_end[m as usize] = self.w_mem;
        } else {
            self.w_end[2 * m as usize] = self.w_mem;
        }

        BLU_OK
    }

    /// Copy scalar entries (except for user parameters) from @lu to @istore,
    /// @xstore. Store status code.
    ///
    /// Return @status
    pub(crate) fn save(&mut self, status: LUInt) -> (Vec<f64>, LUInt) {
        let mut xstore = vec![0.0; BLU_SIZE_XSTORE_1 as usize];

        xstore[0] = BLU_HASH as f64;

        // user readable
        xstore[BLU_STATUS] = status as f64;
        xstore[BLU_ADD_MEMORYL] = self.addmem_l as f64;
        xstore[BLU_ADD_MEMORYU] = self.addmem_u as f64;
        xstore[BLU_ADD_MEMORYW] = self.addmem_w as f64;

        xstore[BLU_NUPDATE] = self.nupdate as f64;
        xstore[BLU_NFORREST] = self.nforrest as f64;
        xstore[BLU_NFACTORIZE] = self.nfactorize as f64;
        xstore[BLU_NUPDATE_TOTAL] = self.nupdate_total as f64;
        xstore[BLU_NFORREST_TOTAL] = self.nforrest_total as f64;
        xstore[BLU_NSYMPERM_TOTAL] = self.nsymperm_total as f64;
        xstore[BLU_LNZ] = self.l_nz as f64;
        xstore[BLU_UNZ] = self.u_nz as f64;
        xstore[BLU_RNZ] = self.r_nz as f64;
        xstore[BLU_MIN_PIVOT] = self.min_pivot;
        xstore[BLU_MAX_PIVOT] = self.max_pivot;
        xstore[BLU_MAX_ETA] = self.max_eta;
        xstore[BLU_UPDATE_COST_NUMER] = self.update_cost_numer;
        xstore[BLU_UPDATE_COST_DENOM] = self.update_cost_denom;
        xstore[BLU_UPDATE_COST] = self.update_cost_numer / self.update_cost_denom;
        xstore[BLU_TIME_FACTORIZE] = self.time_factorize;
        xstore[BLU_TIME_SOLVE] = self.time_solve;
        xstore[BLU_TIME_UPDATE] = self.time_update;
        xstore[BLU_TIME_FACTORIZE_TOTAL] = self.time_factorize_total;
        xstore[BLU_TIME_SOLVE_TOTAL] = self.time_solve_total;
        xstore[BLU_TIME_UPDATE_TOTAL] = self.time_update_total;
        xstore[BLU_LFLOPS] = self.l_flops as f64;
        xstore[BLU_UFLOPS] = self.u_flops as f64;
        xstore[BLU_RFLOPS] = self.r_flops as f64;
        xstore[BLU_CONDEST_L] = self.condest_l;
        xstore[BLU_CONDEST_U] = self.condest_u;
        xstore[BLU_NORM_L] = self.norm_l;
        xstore[BLU_NORM_U] = self.norm_u;
        xstore[BLU_NORMEST_LINV] = self.normest_l_inv;
        xstore[BLU_NORMEST_UINV] = self.normest_u_inv;
        xstore[BLU_MATRIX_ONENORM] = self.onenorm;
        xstore[BLU_MATRIX_INFNORM] = self.infnorm;
        xstore[BLU_RESIDUAL_TEST] = self.residual_test;

        xstore[BLU_MATRIX_NZ] = self.matrix_nz as f64;
        xstore[BLU_RANK] = self.rank as f64;
        xstore[BLU_BUMP_SIZE] = self.bump_size as f64;
        xstore[BLU_BUMP_NZ] = self.bump_nz as f64;
        xstore[BLU_NSEARCH_PIVOT] = self.nsearch_pivot as f64;
        xstore[BLU_NEXPAND] = self.nexpand as f64;
        xstore[BLU_NGARBAGE] = self.ngarbage as f64;
        xstore[BLU_FACTOR_FLOPS] = self.factor_flops as f64;
        xstore[BLU_TIME_SINGLETONS] = self.time_singletons;
        xstore[BLU_TIME_SEARCH_PIVOT] = self.time_search_pivot;
        xstore[BLU_TIME_ELIM_PIVOT] = self.time_elim_pivot;

        xstore[BLU_PIVOT_ERROR] = self.pivot_error;

        // private
        xstore[BLU_TASK] = self.task as f64;
        xstore[BLU_PIVOT_ROW] = self.pivot_row as f64;
        xstore[BLU_PIVOT_COL] = self.pivot_col as f64;
        xstore[BLU_FTCOLUMN_IN] = self.ftran_for_update as f64;
        xstore[BLU_FTCOLUMN_OUT] = self.btran_for_update as f64;
        xstore[BLU_MARKER] = self.marker as f64;
        xstore[BLU_PIVOTLEN] = self.pivotlen as f64;
        xstore[BLU_RANKDEF] = self.rankdef as f64;
        xstore[BLU_MIN_COLNZ] = self.min_colnz as f64;
        xstore[BLU_MIN_ROWNZ] = self.min_rownz as f64;

        (xstore, status)
    }

    /// Deterministic measure of solve/update cost compared to cost of last factorization. This
    /// value is zero after factorization and monotonically increases with solves/updates.
    /// When > 1.0, then a refactorization is good for performance.
    pub(crate) fn update_cost(&mut self) -> f64 {
        self.update_cost_numer / self.update_cost_denom
    }

    /// Reset @lu for a new factorization. Invalidate current factorization.
    pub(crate) fn reset(&mut self) {
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
        self.task = NO_TASK;
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
}
