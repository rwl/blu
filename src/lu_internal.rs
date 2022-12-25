// Functions to load/save/reset struct lu objects //

use crate::basiclu::*;
use crate::lu_def::{BASICLU_HASH, NO_TASK};

// private entries in xstore
pub(crate) const BASICLU_TASK: usize = 256;
pub(crate) const BASICLU_FTCOLUMN_IN: usize = 257;
pub(crate) const BASICLU_FTCOLUMN_OUT: usize = 258;
pub(crate) const BASICLU_PIVOT_ROW: usize = 259;
pub(crate) const BASICLU_PIVOT_COL: usize = 260;
pub(crate) const BASICLU_RANKDEF: usize = 261;
pub(crate) const BASICLU_MIN_COLNZ: usize = 262;
pub(crate) const BASICLU_MIN_ROWNZ: usize = 263;
pub(crate) const BASICLU_MARKER: usize = 266;
pub(crate) const BASICLU_UPDATE_COST_NUMER: usize = 267;
pub(crate) const BASICLU_UPDATE_COST_DENOM: usize = 268;
pub(crate) const BASICLU_PIVOTLEN: usize = 269;

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
    pub l_mem: LUInt,
    pub u_mem: LUInt,
    pub w_mem: LUInt,
    pub droptol: f64,
    pub abstol: f64,
    pub reltol: f64,
    pub nzbias: LUInt,
    pub maxsearch: LUInt,
    pub pad: LUInt,
    pub stretch: f64,
    pub compress_thres: f64,
    pub sparse_thres: f64,
    pub search_rows: LUInt,

    // user readable //
    pub m: LUInt,
    pub addmem_l: LUInt,
    pub addmem_u: LUInt,
    pub addmem_w: LUInt,

    pub nupdate: LUInt,
    pub nforrest: LUInt,
    pub nfactorize: LUInt,
    pub nupdate_total: LUInt,
    pub nforrest_total: LUInt,
    pub nsymperm_total: LUInt,
    pub l_nz: LUInt, // nz in L excluding diagonal
    pub u_nz: LUInt, // nz in U excluding diagonal
    pub r_nz: LUInt, // nz in update etas excluding diagonal
    pub min_pivot: f64,
    pub max_pivot: f64,
    pub max_eta: f64,
    pub update_cost_numer: f64,
    pub update_cost_denom: f64,
    pub time_factorize: f64,
    pub time_solve: f64,
    pub time_update: f64,
    pub time_factorize_total: f64,
    pub time_solve_total: f64,
    pub time_update_total: f64,
    pub l_flops: LUInt,
    pub u_flops: LUInt,
    pub r_flops: LUInt,
    pub condest_l: f64,
    pub condest_u: f64,
    pub norm_l: f64,
    pub norm_u: f64,
    pub normest_l_inv: f64,
    pub normest_u_inv: f64,
    pub onenorm: f64,       // 1-norm and inf-norm of matrix after fresh
    pub infnorm: f64,       // factorization with dependent cols replaced
    pub residual_test: f64, // computed by lu_residual_test()

    pub matrix_nz: LUInt, // nz in basis matrix when factorized
    pub rank: LUInt,      // rank of basis matrix when factorized
    pub bump_size: LUInt,
    pub bump_nz: LUInt,
    pub nsearch_pivot: LUInt, // # rows/cols searched for pivot
    pub nexpand: LUInt,       // # rows/cols expanded in factorize
    pub ngarbage: LUInt,      // # garbage collections in factorize
    pub factor_flops: LUInt,  // # flops in factorize
    pub time_singletons: f64,
    pub time_search_pivot: f64,
    pub time_elim_pivot: f64,

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

    // aliases to user arrays //
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
    // Initialize @lu from @istore, @xstore if these are a valid BASICLU
    // instance. The remaining arguments are copied only and can be NULL.
    //
    // Return `BASICLU_OK` or `BASICLU_ERROR_INVALID_STORE`
    pub(crate) fn load(&mut self, xstore: &[f64]) -> LUInt {
        if xstore[0] != BASICLU_HASH as f64 {
            return BASICLU_ERROR_INVALID_STORE;
        }

        // user parameters
        self.l_mem = xstore[BASICLU_MEMORYL] as LUInt;
        self.u_mem = xstore[BASICLU_MEMORYU] as LUInt;
        self.w_mem = xstore[BASICLU_MEMORYW] as LUInt;
        self.droptol = xstore[BASICLU_DROP_TOLERANCE];
        self.abstol = xstore[BASICLU_ABS_PIVOT_TOLERANCE];
        self.reltol = xstore[BASICLU_REL_PIVOT_TOLERANCE];
        self.reltol = f64::min(self.reltol, 1.0);
        self.nzbias = xstore[BASICLU_BIAS_NONZEROS] as LUInt;
        self.maxsearch = xstore[BASICLU_MAXN_SEARCH_PIVOT] as LUInt;
        self.pad = xstore[BASICLU_PAD] as LUInt;
        self.stretch = xstore[BASICLU_STRETCH];
        self.compress_thres = xstore[BASICLU_COMPRESSION_THRESHOLD];
        self.sparse_thres = xstore[BASICLU_SPARSE_THRESHOLD];
        self.search_rows = if xstore[BASICLU_SEARCH_ROWS] != 0.0 {
            1
        } else {
            0
        };

        // user readable
        let m = xstore[BASICLU_DIM];
        self.m = m as LUInt;
        self.addmem_l = 0;
        self.addmem_u = 0;
        self.addmem_w = 0;

        self.nupdate = xstore[BASICLU_NUPDATE] as LUInt;
        self.nforrest = xstore[BASICLU_NFORREST] as LUInt;
        self.nfactorize = xstore[BASICLU_NFACTORIZE] as LUInt;
        self.nupdate_total = xstore[BASICLU_NUPDATE_TOTAL] as LUInt;
        self.nforrest_total = xstore[BASICLU_NFORREST_TOTAL] as LUInt;
        self.nsymperm_total = xstore[BASICLU_NSYMPERM_TOTAL] as LUInt;
        self.l_nz = xstore[BASICLU_LNZ] as LUInt;
        self.u_nz = xstore[BASICLU_UNZ] as LUInt;
        self.r_nz = xstore[BASICLU_RNZ] as LUInt;
        self.min_pivot = xstore[BASICLU_MIN_PIVOT];
        self.max_pivot = xstore[BASICLU_MAX_PIVOT];
        self.max_eta = xstore[BASICLU_MAX_ETA];
        self.update_cost_numer = xstore[BASICLU_UPDATE_COST_NUMER];
        self.update_cost_denom = xstore[BASICLU_UPDATE_COST_DENOM];
        self.time_factorize = xstore[BASICLU_TIME_FACTORIZE];
        self.time_solve = xstore[BASICLU_TIME_SOLVE];
        self.time_update = xstore[BASICLU_TIME_UPDATE];
        self.time_factorize_total = xstore[BASICLU_TIME_FACTORIZE_TOTAL];
        self.time_solve_total = xstore[BASICLU_TIME_SOLVE_TOTAL];
        self.time_update_total = xstore[BASICLU_TIME_UPDATE_TOTAL];
        self.l_flops = xstore[BASICLU_LFLOPS] as LUInt;
        self.u_flops = xstore[BASICLU_UFLOPS] as LUInt;
        self.r_flops = xstore[BASICLU_RFLOPS] as LUInt;
        self.condest_l = xstore[BASICLU_CONDEST_L];
        self.condest_u = xstore[BASICLU_CONDEST_U];
        self.norm_l = xstore[BASICLU_NORM_L];
        self.norm_u = xstore[BASICLU_NORM_U];
        self.normest_l_inv = xstore[BASICLU_NORMEST_LINV];
        self.normest_u_inv = xstore[BASICLU_NORMEST_UINV];
        self.onenorm = xstore[BASICLU_MATRIX_ONENORM];
        self.infnorm = xstore[BASICLU_MATRIX_INFNORM];
        self.residual_test = xstore[BASICLU_RESIDUAL_TEST];

        self.matrix_nz = xstore[BASICLU_MATRIX_NZ] as LUInt;
        self.rank = xstore[BASICLU_RANK] as LUInt;
        self.bump_size = xstore[BASICLU_BUMP_SIZE] as LUInt;
        self.bump_nz = xstore[BASICLU_BUMP_NZ] as LUInt;
        self.nsearch_pivot = xstore[BASICLU_NSEARCH_PIVOT] as LUInt;
        self.nexpand = xstore[BASICLU_NEXPAND] as LUInt;
        self.ngarbage = xstore[BASICLU_NGARBAGE] as LUInt;
        self.factor_flops = xstore[BASICLU_FACTOR_FLOPS] as LUInt;
        self.time_singletons = xstore[BASICLU_TIME_SINGLETONS];
        self.time_search_pivot = xstore[BASICLU_TIME_SEARCH_PIVOT];
        self.time_elim_pivot = xstore[BASICLU_TIME_ELIM_PIVOT];

        self.pivot_error = xstore[BASICLU_PIVOT_ERROR];

        // private
        self.task = xstore[BASICLU_TASK] as LUInt;
        self.pivot_row = xstore[BASICLU_PIVOT_ROW] as LUInt;
        self.pivot_col = xstore[BASICLU_PIVOT_COL] as LUInt;
        self.ftran_for_update = xstore[BASICLU_FTCOLUMN_IN] as LUInt;
        self.btran_for_update = xstore[BASICLU_FTCOLUMN_OUT] as LUInt;
        self.marker = xstore[BASICLU_MARKER] as LUInt;
        self.pivotlen = xstore[BASICLU_PIVOTLEN] as LUInt;
        self.rankdef = xstore[BASICLU_RANKDEF] as LUInt;
        self.min_colnz = xstore[BASICLU_MIN_COLNZ] as LUInt;
        self.min_rownz = xstore[BASICLU_MIN_ROWNZ] as LUInt;

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

        BASICLU_OK
    }

    /// Copy scalar entries (except for user parameters) from @lu to @istore,
    /// @xstore. Store status code.
    ///
    /// Return @status
    pub(crate) fn save(&mut self, status: LUInt) -> (Vec<f64>, LUInt) {
        let mut xstore = vec![0.0; BASICLU_SIZE_XSTORE_1 as usize];

        xstore[0] = BASICLU_HASH as f64;

        // user readable
        xstore[BASICLU_STATUS] = status as f64;
        xstore[BASICLU_ADD_MEMORYL] = self.addmem_l as f64;
        xstore[BASICLU_ADD_MEMORYU] = self.addmem_u as f64;
        xstore[BASICLU_ADD_MEMORYW] = self.addmem_w as f64;

        xstore[BASICLU_NUPDATE] = self.nupdate as f64;
        xstore[BASICLU_NFORREST] = self.nforrest as f64;
        xstore[BASICLU_NFACTORIZE] = self.nfactorize as f64;
        xstore[BASICLU_NUPDATE_TOTAL] = self.nupdate_total as f64;
        xstore[BASICLU_NFORREST_TOTAL] = self.nforrest_total as f64;
        xstore[BASICLU_NSYMPERM_TOTAL] = self.nsymperm_total as f64;
        xstore[BASICLU_LNZ] = self.l_nz as f64;
        xstore[BASICLU_UNZ] = self.u_nz as f64;
        xstore[BASICLU_RNZ] = self.r_nz as f64;
        xstore[BASICLU_MIN_PIVOT] = self.min_pivot;
        xstore[BASICLU_MAX_PIVOT] = self.max_pivot;
        xstore[BASICLU_MAX_ETA] = self.max_eta;
        xstore[BASICLU_UPDATE_COST_NUMER] = self.update_cost_numer;
        xstore[BASICLU_UPDATE_COST_DENOM] = self.update_cost_denom;
        xstore[BASICLU_UPDATE_COST] = self.update_cost_numer / self.update_cost_denom;
        xstore[BASICLU_TIME_FACTORIZE] = self.time_factorize;
        xstore[BASICLU_TIME_SOLVE] = self.time_solve;
        xstore[BASICLU_TIME_UPDATE] = self.time_update;
        xstore[BASICLU_TIME_FACTORIZE_TOTAL] = self.time_factorize_total;
        xstore[BASICLU_TIME_SOLVE_TOTAL] = self.time_solve_total;
        xstore[BASICLU_TIME_UPDATE_TOTAL] = self.time_update_total;
        xstore[BASICLU_LFLOPS] = self.l_flops as f64;
        xstore[BASICLU_UFLOPS] = self.u_flops as f64;
        xstore[BASICLU_RFLOPS] = self.r_flops as f64;
        xstore[BASICLU_CONDEST_L] = self.condest_l;
        xstore[BASICLU_CONDEST_U] = self.condest_u;
        xstore[BASICLU_NORM_L] = self.norm_l;
        xstore[BASICLU_NORM_U] = self.norm_u;
        xstore[BASICLU_NORMEST_LINV] = self.normest_l_inv;
        xstore[BASICLU_NORMEST_UINV] = self.normest_u_inv;
        xstore[BASICLU_MATRIX_ONENORM] = self.onenorm;
        xstore[BASICLU_MATRIX_INFNORM] = self.infnorm;
        xstore[BASICLU_RESIDUAL_TEST] = self.residual_test;

        xstore[BASICLU_MATRIX_NZ] = self.matrix_nz as f64;
        xstore[BASICLU_RANK] = self.rank as f64;
        xstore[BASICLU_BUMP_SIZE] = self.bump_size as f64;
        xstore[BASICLU_BUMP_NZ] = self.bump_nz as f64;
        xstore[BASICLU_NSEARCH_PIVOT] = self.nsearch_pivot as f64;
        xstore[BASICLU_NEXPAND] = self.nexpand as f64;
        xstore[BASICLU_NGARBAGE] = self.ngarbage as f64;
        xstore[BASICLU_FACTOR_FLOPS] = self.factor_flops as f64;
        xstore[BASICLU_TIME_SINGLETONS] = self.time_singletons;
        xstore[BASICLU_TIME_SEARCH_PIVOT] = self.time_search_pivot;
        xstore[BASICLU_TIME_ELIM_PIVOT] = self.time_elim_pivot;

        xstore[BASICLU_PIVOT_ERROR] = self.pivot_error;

        // private
        xstore[BASICLU_TASK] = self.task as f64;
        xstore[BASICLU_PIVOT_ROW] = self.pivot_row as f64;
        xstore[BASICLU_PIVOT_COL] = self.pivot_col as f64;
        xstore[BASICLU_FTCOLUMN_IN] = self.ftran_for_update as f64;
        xstore[BASICLU_FTCOLUMN_OUT] = self.btran_for_update as f64;
        xstore[BASICLU_MARKER] = self.marker as f64;
        xstore[BASICLU_PIVOTLEN] = self.pivotlen as f64;
        xstore[BASICLU_RANKDEF] = self.rankdef as f64;
        xstore[BASICLU_MIN_COLNZ] = self.min_colnz as f64;
        xstore[BASICLU_MIN_ROWNZ] = self.min_rownz as f64;

        (xstore, status)
    }

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
