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
    ($this:ident) => {
        $this.colcount_flink
    };
}

macro_rules! pivotrow {
    ($this:ident) => {
        $this.colcount_blink
    };
}
macro_rules! r_begin {
    ($this:ident) => {
        $this.rowcount_flink
    };
}
macro_rules! eta_row {
    ($this:ident) => {
        $this.rowcount_flink
    };
}
macro_rules! iwork1 {
    ($this:ident) => {
        $this.rowcount_blink
    };
}
macro_rules! l_begin {
    ($this:ident) => {
        $this.Wbegin[$this.m as usize + 1..]
    };
}
macro_rules! lt_begin {
    ($this:ident) => {
        $this.Wend[$this.m as usize + 1..]
    };
}
macro_rules! lt_begin_p {
    ($this:ident) => {
        $this.Wflink[$this.m as usize + 1..]
    };
}
macro_rules! p {
    ($this:ident) => {
        $this.Wblink[$this.m as usize + 1..]
    };
}
macro_rules! pmap {
    ($this:ident) => {
        $this.pinv
    };
}
macro_rules! qmap {
    ($this:ident) => {
        $this.qinv
    };
}
macro_rules! marked {
    ($this:ident) => {
        $this.iwork0
    };
}

pub(crate) use {
    eta_row, iwork1, l_begin, lt_begin, lt_begin_p, marked, p, pivotcol, pivotrow, pmap, qmap,
    r_begin,
};

/// Initialize @this from @istore, @xstore if these are a valid BASICLU
/// instance. The remaining arguments are copied only and can be NULL.
///
/// Return `BASICLU_OK` or `BASICLU_ERROR_INVALID_STORE`
pub(crate) fn lu_load(
    this: &mut LU,
    // istore: &[lu_int],
    xstore: &[f64],
    // Li: Option<&[lu_int]>,
    // Lx: Option<&[f64]>,
    // Ui: Option<&[lu_int]>,
    // Ux: Option<&[f64]>,
    // Wi: Option<&[lu_int]>,
    // Wx: Option<&[f64]>,
) -> LUInt {
    if
    /*istore[0] != BASICLU_HASH ||*/
    xstore[0] != BASICLU_HASH as f64 {
        return BASICLU_ERROR_INVALID_STORE;
    }

    // user parameters
    this.l_mem = xstore[BASICLU_MEMORYL] as LUInt;
    this.u_mem = xstore[BASICLU_MEMORYU] as LUInt;
    this.w_mem = xstore[BASICLU_MEMORYW] as LUInt;
    this.droptol = xstore[BASICLU_DROP_TOLERANCE];
    this.abstol = xstore[BASICLU_ABS_PIVOT_TOLERANCE];
    this.reltol = xstore[BASICLU_REL_PIVOT_TOLERANCE];
    this.reltol = f64::min(this.reltol, 1.0);
    this.nzbias = xstore[BASICLU_BIAS_NONZEROS] as LUInt;
    this.maxsearch = xstore[BASICLU_MAXN_SEARCH_PIVOT] as LUInt;
    this.pad = xstore[BASICLU_PAD] as LUInt;
    this.stretch = xstore[BASICLU_STRETCH];
    this.compress_thres = xstore[BASICLU_COMPRESSION_THRESHOLD];
    this.sparse_thres = xstore[BASICLU_SPARSE_THRESHOLD];
    this.search_rows = if xstore[BASICLU_SEARCH_ROWS] != 0.0 {
        1
    } else {
        0
    };

    // user readable
    let m = xstore[BASICLU_DIM];
    this.m = m as LUInt;
    this.addmem_l = 0;
    this.addmem_u = 0;
    this.addmem_w = 0;

    this.nupdate = xstore[BASICLU_NUPDATE] as LUInt;
    this.nforrest = xstore[BASICLU_NFORREST] as LUInt;
    this.nfactorize = xstore[BASICLU_NFACTORIZE] as LUInt;
    this.nupdate_total = xstore[BASICLU_NUPDATE_TOTAL] as LUInt;
    this.nforrest_total = xstore[BASICLU_NFORREST_TOTAL] as LUInt;
    this.nsymperm_total = xstore[BASICLU_NSYMPERM_TOTAL] as LUInt;
    this.l_nz = xstore[BASICLU_LNZ] as LUInt;
    this.u_nz = xstore[BASICLU_UNZ] as LUInt;
    this.r_nz = xstore[BASICLU_RNZ] as LUInt;
    this.min_pivot = xstore[BASICLU_MIN_PIVOT];
    this.max_pivot = xstore[BASICLU_MAX_PIVOT];
    this.max_eta = xstore[BASICLU_MAX_ETA];
    this.update_cost_numer = xstore[BASICLU_UPDATE_COST_NUMER];
    this.update_cost_denom = xstore[BASICLU_UPDATE_COST_DENOM];
    this.time_factorize = xstore[BASICLU_TIME_FACTORIZE];
    this.time_solve = xstore[BASICLU_TIME_SOLVE];
    this.time_update = xstore[BASICLU_TIME_UPDATE];
    this.time_factorize_total = xstore[BASICLU_TIME_FACTORIZE_TOTAL];
    this.time_solve_total = xstore[BASICLU_TIME_SOLVE_TOTAL];
    this.time_update_total = xstore[BASICLU_TIME_UPDATE_TOTAL];
    this.l_flops = xstore[BASICLU_LFLOPS] as LUInt;
    this.u_flops = xstore[BASICLU_UFLOPS] as LUInt;
    this.r_flops = xstore[BASICLU_RFLOPS] as LUInt;
    this.condest_l = xstore[BASICLU_CONDEST_L];
    this.condest_u = xstore[BASICLU_CONDEST_U];
    this.norm_l = xstore[BASICLU_NORM_L];
    this.norm_u = xstore[BASICLU_NORM_U];
    this.normest_l_inv = xstore[BASICLU_NORMEST_LINV];
    this.normest_u_inv = xstore[BASICLU_NORMEST_UINV];
    this.onenorm = xstore[BASICLU_MATRIX_ONENORM];
    this.infnorm = xstore[BASICLU_MATRIX_INFNORM];
    this.residual_test = xstore[BASICLU_RESIDUAL_TEST];

    this.matrix_nz = xstore[BASICLU_MATRIX_NZ] as LUInt;
    this.rank = xstore[BASICLU_RANK] as LUInt;
    this.bump_size = xstore[BASICLU_BUMP_SIZE] as LUInt;
    this.bump_nz = xstore[BASICLU_BUMP_NZ] as LUInt;
    this.nsearch_pivot = xstore[BASICLU_NSEARCH_PIVOT] as LUInt;
    this.nexpand = xstore[BASICLU_NEXPAND] as LUInt;
    this.ngarbage = xstore[BASICLU_NGARBAGE] as LUInt;
    this.factor_flops = xstore[BASICLU_FACTOR_FLOPS] as LUInt;
    this.time_singletons = xstore[BASICLU_TIME_SINGLETONS];
    this.time_search_pivot = xstore[BASICLU_TIME_SEARCH_PIVOT];
    this.time_elim_pivot = xstore[BASICLU_TIME_ELIM_PIVOT];

    this.pivot_error = xstore[BASICLU_PIVOT_ERROR];

    // private
    this.task = xstore[BASICLU_TASK] as LUInt;
    this.pivot_row = xstore[BASICLU_PIVOT_ROW] as LUInt;
    this.pivot_col = xstore[BASICLU_PIVOT_COL] as LUInt;
    this.ftran_for_update = xstore[BASICLU_FTCOLUMN_IN] as LUInt;
    this.btran_for_update = xstore[BASICLU_FTCOLUMN_OUT] as LUInt;
    this.marker = xstore[BASICLU_MARKER] as LUInt;
    this.pivotlen = xstore[BASICLU_PIVOTLEN] as LUInt;
    this.rankdef = xstore[BASICLU_RANKDEF] as LUInt;
    this.min_colnz = xstore[BASICLU_MIN_COLNZ] as LUInt;
    this.min_rownz = xstore[BASICLU_MIN_ROWNZ] as LUInt;

    // aliases to user arrays
    // this.Lindex = Li;
    // this.Lvalue = Lx;
    // this.Uindex = Ui;
    // this.Uvalue = Ux;
    // this.Windex = Wi;
    // this.Wvalue = Wx;
    // this.Lindex = match Li {
    //     Some(Li) => Some(Li.to_vec()),
    //     None => None,
    // };
    // this.Lvalue = match Lx {
    //     Some(Lx) => Some(Lx.to_vec()),
    //     None => None,
    // };
    // this.Uindex = match Ui {
    //     Some(Ui) => Some(Ui.to_vec()),
    //     None => None,
    // };
    // this.Uvalue = match Ux {
    //     Some(Ux) => Some(Ux.to_vec()),
    //     None => None,
    // };
    // this.Windex = match Wi {
    //     Some(Wi) => Some(Wi.to_vec()),
    //     None => None,
    // };
    // this.Wvalue = match Wx {
    //     Some(Wx) => Some(Wx.to_vec()),
    //     None => None,
    // };

    // // partition istore for factorize
    // this.colcount_flink = vec![0; 2 * m as usize + 2];
    // this.pivotcol = vec![];
    // // iptr += 2 * m + 2;
    // this.colcount_blink = Some(vec![0; 2 * m as usize + 2]);
    // // iptr += 2 * m + 2;
    // this.rowcount_flink = Some(vec![0; 2 * m as usize + 2]);
    // // iptr += 2 * m + 2;
    // this.rowcount_blink = Some(vec![0; 2 * m as usize + 2]);
    // // iptr += 2 * m + 2;
    // this.Wbegin = Some(vec![0; 2 * m as usize + 2]);
    // // iptr += 2 * m + 1;
    // this.Wend = Some(vec![0; 2 * m as usize + 2]);
    // // iptr += 2 * m + 1;
    // this.Wflink = Some(vec![0; 2 * m as usize + 2]);
    // // iptr += 2 * m + 1;
    // this.Wblink = Some(vec![0; 2 * m as usize + 2]);
    // // iptr += 2 * m + 1;
    // this.pinv = Some(vec![0; m as usize]);
    // // iptr += m;
    // this.qinv = Some(vec![0; m as usize]);
    // // iptr += m;
    // this.Lbegin_p = vec![0; m as usize + 1];
    // // iptr += m + 1;
    // this.Ubegin = vec![0; m as usize + 1];
    // // iptr += m + 1;
    // this.iwork0 = Some(vec![0; m as usize]);
    // // iptr += m;
    //
    // // share istore memory for solve/update
    // swap(&mut this.pivotcol, &mut this.colcount_flink);
    // this.pivotrow = this.colcount_blink.take();
    // this.r_begin = this.rowcount_flink.take(); // FIXME: [..m+1]
    //                                           // this.eta_row = this.rowcount_flink + m + 1;
    //                                           // this.eta_row = this.rowcount_flink[m as usize + 1..].to_vec();
    // this.eta_row = vec![0; m as usize + 1]; // FIXME: rowcount_flink[m+1..]
    // this.iwork1 = this.rowcount_blink.take();
    // // this.l_begin = this.Wbegin + m + 1;
    // // this.l_begin = this.Wbegin[m as usize + 1..].to_vec();
    // this.l_begin = this.Wbegin.take(); // [m+1..]
    //                                   // this.lt_begin = this.Wend + m + 1;
    //                                   // this.lt_begin = this.Wend[m as usize + 1..].to_vec();
    // this.lt_begin = this.Wend.take(); // [m+1..]
    //                                  // this.lt_begin_p = this.Wflink + m + 1;
    //                                  // this.lt_begin_p = this.Wflink[m as usize + 1..].to_vec();
    // this.lt_begin_p = this.Wflink.take(); // [m+1..]
    //                                      // this.p = this.Wblink + m + 1;
    //                                      // this.p = this.Wblink[m as usize + 1..].to_vec();
    // this.p = this.Wblink.take(); // [m+1..]
    // this.pmap = this.pinv.take();
    // this.qmap = this.qinv.take();
    // this.marked = this.iwork0.take();
    //
    // // partition xstore for factorize and update
    // // let xptr = xstore + 512;
    // // let (_, xstore) = xstore.split_at(512);
    // // let (work0, xstore) = xstore.split_at(m as usize);
    // this.work0 = vec![0.0; m as usize];
    // // xptr += m;
    // let (work1, xstore) = xstore.split_at(m as usize);
    // // this.work1 = Vec::from(work1);
    // this.work1 = vec![0.0; m as usize];
    // // xptr += m;
    // // let (col_pivot, xstore) = xstore.split_at(m as usize);
    // // this.col_pivot = Vec::from(col_pivot);
    // this.col_pivot = vec![0.0; m as usize];
    // // xptr += m;
    // // let (row_pivot, _) = xstore.split_at(m as usize);
    // // this.row_pivot = Vec::from(row_pivot);
    // this.row_pivot = vec![0.0; m as usize];
    // // xptr += m;
    //
    // // Reset @marked if increasing @marker by four causes overflow.
    // if this.marker > LU_INT_MAX - 4 {
    //     // memset(this.marked, 0, m * sizeof(lu_int));
    //     this.marked.as_mut().unwrap().fill(0);
    //     this.marker = 0;
    // }
    //
    // // One past the final position in @Wend must hold the file size.
    // // The file has 2*m lines while factorizing and m lines otherwise.
    // let Wend = this.Wend.as_mut().unwrap();
    // if this.nupdate >= 0 {
    //     Wend[m as usize] = this.wmem;
    // } else {
    //     Wend[2 * m as usize] = this.wmem;
    // }

    BASICLU_OK
}

/// Copy scalar entries (except for user parameters) from @this to @istore,
/// @xstore. Store status code.
///
/// Return @status
pub(crate) fn lu_save(
    this: &LU,
    // _istore: &mut [lu_int],
    xstore: &mut [f64],
    status: LUInt,
) -> LUInt {
    // user readable
    xstore[BASICLU_STATUS] = status as f64;
    xstore[BASICLU_ADD_MEMORYL] = this.addmem_l as f64;
    xstore[BASICLU_ADD_MEMORYU] = this.addmem_u as f64;
    xstore[BASICLU_ADD_MEMORYW] = this.addmem_w as f64;

    xstore[BASICLU_NUPDATE] = this.nupdate as f64;
    xstore[BASICLU_NFORREST] = this.nforrest as f64;
    xstore[BASICLU_NFACTORIZE] = this.nfactorize as f64;
    xstore[BASICLU_NUPDATE_TOTAL] = this.nupdate_total as f64;
    xstore[BASICLU_NFORREST_TOTAL] = this.nforrest_total as f64;
    xstore[BASICLU_NSYMPERM_TOTAL] = this.nsymperm_total as f64;
    xstore[BASICLU_LNZ] = this.l_nz as f64;
    xstore[BASICLU_UNZ] = this.u_nz as f64;
    xstore[BASICLU_RNZ] = this.r_nz as f64;
    xstore[BASICLU_MIN_PIVOT] = this.min_pivot;
    xstore[BASICLU_MAX_PIVOT] = this.max_pivot;
    xstore[BASICLU_MAX_ETA] = this.max_eta;
    xstore[BASICLU_UPDATE_COST_NUMER] = this.update_cost_numer;
    xstore[BASICLU_UPDATE_COST_DENOM] = this.update_cost_denom;
    xstore[BASICLU_UPDATE_COST] = this.update_cost_numer / this.update_cost_denom;
    xstore[BASICLU_TIME_FACTORIZE] = this.time_factorize;
    xstore[BASICLU_TIME_SOLVE] = this.time_solve;
    xstore[BASICLU_TIME_UPDATE] = this.time_update;
    xstore[BASICLU_TIME_FACTORIZE_TOTAL] = this.time_factorize_total;
    xstore[BASICLU_TIME_SOLVE_TOTAL] = this.time_solve_total;
    xstore[BASICLU_TIME_UPDATE_TOTAL] = this.time_update_total;
    xstore[BASICLU_LFLOPS] = this.l_flops as f64;
    xstore[BASICLU_UFLOPS] = this.u_flops as f64;
    xstore[BASICLU_RFLOPS] = this.r_flops as f64;
    xstore[BASICLU_CONDEST_L] = this.condest_l;
    xstore[BASICLU_CONDEST_U] = this.condest_u;
    xstore[BASICLU_NORM_L] = this.norm_l;
    xstore[BASICLU_NORM_U] = this.norm_u;
    xstore[BASICLU_NORMEST_LINV] = this.normest_l_inv;
    xstore[BASICLU_NORMEST_UINV] = this.normest_u_inv;
    xstore[BASICLU_MATRIX_ONENORM] = this.onenorm;
    xstore[BASICLU_MATRIX_INFNORM] = this.infnorm;
    xstore[BASICLU_RESIDUAL_TEST] = this.residual_test;

    xstore[BASICLU_MATRIX_NZ] = this.matrix_nz as f64;
    xstore[BASICLU_RANK] = this.rank as f64;
    xstore[BASICLU_BUMP_SIZE] = this.bump_size as f64;
    xstore[BASICLU_BUMP_NZ] = this.bump_nz as f64;
    xstore[BASICLU_NSEARCH_PIVOT] = this.nsearch_pivot as f64;
    xstore[BASICLU_NEXPAND] = this.nexpand as f64;
    xstore[BASICLU_NGARBAGE] = this.ngarbage as f64;
    xstore[BASICLU_FACTOR_FLOPS] = this.factor_flops as f64;
    xstore[BASICLU_TIME_SINGLETONS] = this.time_singletons;
    xstore[BASICLU_TIME_SEARCH_PIVOT] = this.time_search_pivot;
    xstore[BASICLU_TIME_ELIM_PIVOT] = this.time_elim_pivot;

    xstore[BASICLU_PIVOT_ERROR] = this.pivot_error;

    // private
    xstore[BASICLU_TASK] = this.task as f64;
    xstore[BASICLU_PIVOT_ROW] = this.pivot_row as f64;
    xstore[BASICLU_PIVOT_COL] = this.pivot_col as f64;
    xstore[BASICLU_FTCOLUMN_IN] = this.ftran_for_update as f64;
    xstore[BASICLU_FTCOLUMN_OUT] = this.btran_for_update as f64;
    xstore[BASICLU_MARKER] = this.marker as f64;
    xstore[BASICLU_PIVOTLEN] = this.pivotlen as f64;
    xstore[BASICLU_RANKDEF] = this.rankdef as f64;
    xstore[BASICLU_MIN_COLNZ] = this.min_colnz as f64;
    xstore[BASICLU_MIN_ROWNZ] = this.min_rownz as f64;

    status
}

/// Reset @this for a new factorization. Invalidate current factorization.
pub(crate) fn lu_reset(this: &mut LU) {
    // user readable
    this.nupdate = -1; // invalidate factorization
    this.nforrest = 0;
    this.l_nz = 0;
    this.u_nz = 0;
    this.r_nz = 0;
    this.min_pivot = 0.0;
    this.max_pivot = 0.0;
    this.max_eta = 0.0;
    this.update_cost_numer = 0.0;
    this.update_cost_denom = 1.0;
    this.time_factorize = 0.0;
    this.time_solve = 0.0;
    this.time_update = 0.0;
    this.l_flops = 0;
    this.u_flops = 0;
    this.r_flops = 0;
    this.condest_l = 0.0;
    this.condest_u = 0.0;
    this.norm_l = 0.0;
    this.norm_u = 0.0;
    this.normest_l_inv = 0.0;
    this.normest_u_inv = 0.0;
    this.onenorm = 0.0;
    this.infnorm = 0.0;
    this.residual_test = 0.0;

    this.matrix_nz = 0;
    this.rank = 0;
    this.bump_size = 0;
    this.bump_nz = 0;
    this.nsearch_pivot = 0;
    this.nexpand = 0;
    this.ngarbage = 0;
    this.factor_flops = 0;
    this.time_singletons = 0.0;
    this.time_search_pivot = 0.0;
    this.time_elim_pivot = 0.0;

    this.pivot_error = 0.0;

    // private
    this.task = NO_TASK;
    this.pivot_row = -1;
    this.pivot_col = -1;
    this.ftran_for_update = -1;
    this.btran_for_update = -1;
    this.marker = 0;
    this.pivotlen = 0;
    this.rankdef = 0;
    this.min_colnz = 1;
    this.min_rownz = 1;

    // One past the final position in @Wend must hold the file size.
    // The file has 2*m lines during factorization.
    this.w_end[2 * this.m as usize] = this.w_mem;

    // The integer workspace iwork0 must be zeroed for a new factorization.
    // The double workspace work0 actually needs only be zeroed once in the
    // initialization of xstore. However, it is easier and more consistent
    // to do that here as well.
    // memset(this.iwork0, 0, this.m);
    this.iwork0.fill(0);

    // memset(this.work0, 0, this.m);
    this.work0.fill(0.0);
}
