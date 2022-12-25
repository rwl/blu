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
        $lu.Wbegin[$lu.m as usize + 1..]
    };
}
macro_rules! lt_begin {
    ($lu:ident) => {
        $lu.Wend[$lu.m as usize + 1..]
    };
}
macro_rules! lt_begin_p {
    ($lu:ident) => {
        $lu.Wflink[$lu.m as usize + 1..]
    };
}
macro_rules! p {
    ($lu:ident) => {
        $lu.Wblink[$lu.m as usize + 1..]
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

/// Initialize @lu from @istore, @xstore if these are a valid BASICLU
/// instance. The remaining arguments are copied only and can be NULL.
///
/// Return `BASICLU_OK` or `BASICLU_ERROR_INVALID_STORE`
pub(crate) fn lu_load(
    lu: &mut LU,
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
    lu.l_mem = xstore[BASICLU_MEMORYL] as LUInt;
    lu.u_mem = xstore[BASICLU_MEMORYU] as LUInt;
    lu.w_mem = xstore[BASICLU_MEMORYW] as LUInt;
    lu.droptol = xstore[BASICLU_DROP_TOLERANCE];
    lu.abstol = xstore[BASICLU_ABS_PIVOT_TOLERANCE];
    lu.reltol = xstore[BASICLU_REL_PIVOT_TOLERANCE];
    lu.reltol = f64::min(lu.reltol, 1.0);
    lu.nzbias = xstore[BASICLU_BIAS_NONZEROS] as LUInt;
    lu.maxsearch = xstore[BASICLU_MAXN_SEARCH_PIVOT] as LUInt;
    lu.pad = xstore[BASICLU_PAD] as LUInt;
    lu.stretch = xstore[BASICLU_STRETCH];
    lu.compress_thres = xstore[BASICLU_COMPRESSION_THRESHOLD];
    lu.sparse_thres = xstore[BASICLU_SPARSE_THRESHOLD];
    lu.search_rows = if xstore[BASICLU_SEARCH_ROWS] != 0.0 {
        1
    } else {
        0
    };

    // user readable
    let m = xstore[BASICLU_DIM];
    lu.m = m as LUInt;
    lu.addmem_l = 0;
    lu.addmem_u = 0;
    lu.addmem_w = 0;

    lu.nupdate = xstore[BASICLU_NUPDATE] as LUInt;
    lu.nforrest = xstore[BASICLU_NFORREST] as LUInt;
    lu.nfactorize = xstore[BASICLU_NFACTORIZE] as LUInt;
    lu.nupdate_total = xstore[BASICLU_NUPDATE_TOTAL] as LUInt;
    lu.nforrest_total = xstore[BASICLU_NFORREST_TOTAL] as LUInt;
    lu.nsymperm_total = xstore[BASICLU_NSYMPERM_TOTAL] as LUInt;
    lu.l_nz = xstore[BASICLU_LNZ] as LUInt;
    lu.u_nz = xstore[BASICLU_UNZ] as LUInt;
    lu.r_nz = xstore[BASICLU_RNZ] as LUInt;
    lu.min_pivot = xstore[BASICLU_MIN_PIVOT];
    lu.max_pivot = xstore[BASICLU_MAX_PIVOT];
    lu.max_eta = xstore[BASICLU_MAX_ETA];
    lu.update_cost_numer = xstore[BASICLU_UPDATE_COST_NUMER];
    lu.update_cost_denom = xstore[BASICLU_UPDATE_COST_DENOM];
    lu.time_factorize = xstore[BASICLU_TIME_FACTORIZE];
    lu.time_solve = xstore[BASICLU_TIME_SOLVE];
    lu.time_update = xstore[BASICLU_TIME_UPDATE];
    lu.time_factorize_total = xstore[BASICLU_TIME_FACTORIZE_TOTAL];
    lu.time_solve_total = xstore[BASICLU_TIME_SOLVE_TOTAL];
    lu.time_update_total = xstore[BASICLU_TIME_UPDATE_TOTAL];
    lu.l_flops = xstore[BASICLU_LFLOPS] as LUInt;
    lu.u_flops = xstore[BASICLU_UFLOPS] as LUInt;
    lu.r_flops = xstore[BASICLU_RFLOPS] as LUInt;
    lu.condest_l = xstore[BASICLU_CONDEST_L];
    lu.condest_u = xstore[BASICLU_CONDEST_U];
    lu.norm_l = xstore[BASICLU_NORM_L];
    lu.norm_u = xstore[BASICLU_NORM_U];
    lu.normest_l_inv = xstore[BASICLU_NORMEST_LINV];
    lu.normest_u_inv = xstore[BASICLU_NORMEST_UINV];
    lu.onenorm = xstore[BASICLU_MATRIX_ONENORM];
    lu.infnorm = xstore[BASICLU_MATRIX_INFNORM];
    lu.residual_test = xstore[BASICLU_RESIDUAL_TEST];

    lu.matrix_nz = xstore[BASICLU_MATRIX_NZ] as LUInt;
    lu.rank = xstore[BASICLU_RANK] as LUInt;
    lu.bump_size = xstore[BASICLU_BUMP_SIZE] as LUInt;
    lu.bump_nz = xstore[BASICLU_BUMP_NZ] as LUInt;
    lu.nsearch_pivot = xstore[BASICLU_NSEARCH_PIVOT] as LUInt;
    lu.nexpand = xstore[BASICLU_NEXPAND] as LUInt;
    lu.ngarbage = xstore[BASICLU_NGARBAGE] as LUInt;
    lu.factor_flops = xstore[BASICLU_FACTOR_FLOPS] as LUInt;
    lu.time_singletons = xstore[BASICLU_TIME_SINGLETONS];
    lu.time_search_pivot = xstore[BASICLU_TIME_SEARCH_PIVOT];
    lu.time_elim_pivot = xstore[BASICLU_TIME_ELIM_PIVOT];

    lu.pivot_error = xstore[BASICLU_PIVOT_ERROR];

    // private
    lu.task = xstore[BASICLU_TASK] as LUInt;
    lu.pivot_row = xstore[BASICLU_PIVOT_ROW] as LUInt;
    lu.pivot_col = xstore[BASICLU_PIVOT_COL] as LUInt;
    lu.ftran_for_update = xstore[BASICLU_FTCOLUMN_IN] as LUInt;
    lu.btran_for_update = xstore[BASICLU_FTCOLUMN_OUT] as LUInt;
    lu.marker = xstore[BASICLU_MARKER] as LUInt;
    lu.pivotlen = xstore[BASICLU_PIVOTLEN] as LUInt;
    lu.rankdef = xstore[BASICLU_RANKDEF] as LUInt;
    lu.min_colnz = xstore[BASICLU_MIN_COLNZ] as LUInt;
    lu.min_rownz = xstore[BASICLU_MIN_ROWNZ] as LUInt;

    // aliases to user arrays
    // lu.Lindex = Li;
    // lu.Lvalue = Lx;
    // lu.Uindex = Ui;
    // lu.Uvalue = Ux;
    // lu.Windex = Wi;
    // lu.Wvalue = Wx;
    // lu.Lindex = match Li {
    //     Some(Li) => Some(Li.to_vec()),
    //     None => None,
    // };
    // lu.Lvalue = match Lx {
    //     Some(Lx) => Some(Lx.to_vec()),
    //     None => None,
    // };
    // lu.Uindex = match Ui {
    //     Some(Ui) => Some(Ui.to_vec()),
    //     None => None,
    // };
    // lu.Uvalue = match Ux {
    //     Some(Ux) => Some(Ux.to_vec()),
    //     None => None,
    // };
    // lu.Windex = match Wi {
    //     Some(Wi) => Some(Wi.to_vec()),
    //     None => None,
    // };
    // lu.Wvalue = match Wx {
    //     Some(Wx) => Some(Wx.to_vec()),
    //     None => None,
    // };

    // // partition istore for factorize
    // lu.colcount_flink = vec![0; 2 * m as usize + 2];
    // lu.pivotcol = vec![];
    // // iptr += 2 * m + 2;
    // lu.colcount_blink = Some(vec![0; 2 * m as usize + 2]);
    // // iptr += 2 * m + 2;
    // lu.rowcount_flink = Some(vec![0; 2 * m as usize + 2]);
    // // iptr += 2 * m + 2;
    // lu.rowcount_blink = Some(vec![0; 2 * m as usize + 2]);
    // // iptr += 2 * m + 2;
    // lu.Wbegin = Some(vec![0; 2 * m as usize + 2]);
    // // iptr += 2 * m + 1;
    // lu.Wend = Some(vec![0; 2 * m as usize + 2]);
    // // iptr += 2 * m + 1;
    // lu.Wflink = Some(vec![0; 2 * m as usize + 2]);
    // // iptr += 2 * m + 1;
    // lu.Wblink = Some(vec![0; 2 * m as usize + 2]);
    // // iptr += 2 * m + 1;
    // lu.pinv = Some(vec![0; m as usize]);
    // // iptr += m;
    // lu.qinv = Some(vec![0; m as usize]);
    // // iptr += m;
    // lu.Lbegin_p = vec![0; m as usize + 1];
    // // iptr += m + 1;
    // lu.Ubegin = vec![0; m as usize + 1];
    // // iptr += m + 1;
    // lu.iwork0 = Some(vec![0; m as usize]);
    // // iptr += m;
    //
    // // share istore memory for solve/update
    // swap(&mut lu.pivotcol, &mut lu.colcount_flink);
    // lu.pivotrow = lu.colcount_blink.take();
    // lu.r_begin = lu.rowcount_flink.take(); // FIXME: [..m+1]
    //                                           // lu.eta_row = lu.rowcount_flink + m + 1;
    //                                           // lu.eta_row = lu.rowcount_flink[m as usize + 1..].to_vec();
    // lu.eta_row = vec![0; m as usize + 1]; // FIXME: rowcount_flink[m+1..]
    // lu.iwork1 = lu.rowcount_blink.take();
    // // lu.l_begin = lu.Wbegin + m + 1;
    // // lu.l_begin = lu.Wbegin[m as usize + 1..].to_vec();
    // lu.l_begin = lu.Wbegin.take(); // [m+1..]
    //                                   // lu.lt_begin = lu.Wend + m + 1;
    //                                   // lu.lt_begin = lu.Wend[m as usize + 1..].to_vec();
    // lu.lt_begin = lu.Wend.take(); // [m+1..]
    //                                  // lu.lt_begin_p = lu.Wflink + m + 1;
    //                                  // lu.lt_begin_p = lu.Wflink[m as usize + 1..].to_vec();
    // lu.lt_begin_p = lu.Wflink.take(); // [m+1..]
    //                                      // lu.p = lu.Wblink + m + 1;
    //                                      // lu.p = lu.Wblink[m as usize + 1..].to_vec();
    // lu.p = lu.Wblink.take(); // [m+1..]
    // lu.pmap = lu.pinv.take();
    // lu.qmap = lu.qinv.take();
    // lu.marked = lu.iwork0.take();
    //
    // // partition xstore for factorize and update
    // // let xptr = xstore + 512;
    // // let (_, xstore) = xstore.split_at(512);
    // // let (work0, xstore) = xstore.split_at(m as usize);
    // lu.work0 = vec![0.0; m as usize];
    // // xptr += m;
    // let (work1, xstore) = xstore.split_at(m as usize);
    // // lu.work1 = Vec::from(work1);
    // lu.work1 = vec![0.0; m as usize];
    // // xptr += m;
    // // let (col_pivot, xstore) = xstore.split_at(m as usize);
    // // lu.col_pivot = Vec::from(col_pivot);
    // lu.col_pivot = vec![0.0; m as usize];
    // // xptr += m;
    // // let (row_pivot, _) = xstore.split_at(m as usize);
    // // lu.row_pivot = Vec::from(row_pivot);
    // lu.row_pivot = vec![0.0; m as usize];
    // // xptr += m;
    //
    // // Reset @marked if increasing @marker by four causes overflow.
    // if lu.marker > LU_INT_MAX - 4 {
    //     // memset(lu.marked, 0, m * sizeof(lu_int));
    //     lu.marked.as_mut().unwrap().fill(0);
    //     lu.marker = 0;
    // }
    //
    // // One past the final position in @Wend must hold the file size.
    // // The file has 2*m lines while factorizing and m lines otherwise.
    // let Wend = lu.Wend.as_mut().unwrap();
    // if lu.nupdate >= 0 {
    //     Wend[m as usize] = lu.wmem;
    // } else {
    //     Wend[2 * m as usize] = lu.wmem;
    // }

    BASICLU_OK
}

/// Copy scalar entries (except for user parameters) from @lu to @istore,
/// @xstore. Store status code.
///
/// Return @status
pub(crate) fn lu_save(
    lu: &LU,
    // _istore: &mut [lu_int],
    xstore: &mut [f64],
    status: LUInt,
) -> LUInt {
    // user readable
    xstore[BASICLU_STATUS] = status as f64;
    xstore[BASICLU_ADD_MEMORYL] = lu.addmem_l as f64;
    xstore[BASICLU_ADD_MEMORYU] = lu.addmem_u as f64;
    xstore[BASICLU_ADD_MEMORYW] = lu.addmem_w as f64;

    xstore[BASICLU_NUPDATE] = lu.nupdate as f64;
    xstore[BASICLU_NFORREST] = lu.nforrest as f64;
    xstore[BASICLU_NFACTORIZE] = lu.nfactorize as f64;
    xstore[BASICLU_NUPDATE_TOTAL] = lu.nupdate_total as f64;
    xstore[BASICLU_NFORREST_TOTAL] = lu.nforrest_total as f64;
    xstore[BASICLU_NSYMPERM_TOTAL] = lu.nsymperm_total as f64;
    xstore[BASICLU_LNZ] = lu.l_nz as f64;
    xstore[BASICLU_UNZ] = lu.u_nz as f64;
    xstore[BASICLU_RNZ] = lu.r_nz as f64;
    xstore[BASICLU_MIN_PIVOT] = lu.min_pivot;
    xstore[BASICLU_MAX_PIVOT] = lu.max_pivot;
    xstore[BASICLU_MAX_ETA] = lu.max_eta;
    xstore[BASICLU_UPDATE_COST_NUMER] = lu.update_cost_numer;
    xstore[BASICLU_UPDATE_COST_DENOM] = lu.update_cost_denom;
    xstore[BASICLU_UPDATE_COST] = lu.update_cost_numer / lu.update_cost_denom;
    xstore[BASICLU_TIME_FACTORIZE] = lu.time_factorize;
    xstore[BASICLU_TIME_SOLVE] = lu.time_solve;
    xstore[BASICLU_TIME_UPDATE] = lu.time_update;
    xstore[BASICLU_TIME_FACTORIZE_TOTAL] = lu.time_factorize_total;
    xstore[BASICLU_TIME_SOLVE_TOTAL] = lu.time_solve_total;
    xstore[BASICLU_TIME_UPDATE_TOTAL] = lu.time_update_total;
    xstore[BASICLU_LFLOPS] = lu.l_flops as f64;
    xstore[BASICLU_UFLOPS] = lu.u_flops as f64;
    xstore[BASICLU_RFLOPS] = lu.r_flops as f64;
    xstore[BASICLU_CONDEST_L] = lu.condest_l;
    xstore[BASICLU_CONDEST_U] = lu.condest_u;
    xstore[BASICLU_NORM_L] = lu.norm_l;
    xstore[BASICLU_NORM_U] = lu.norm_u;
    xstore[BASICLU_NORMEST_LINV] = lu.normest_l_inv;
    xstore[BASICLU_NORMEST_UINV] = lu.normest_u_inv;
    xstore[BASICLU_MATRIX_ONENORM] = lu.onenorm;
    xstore[BASICLU_MATRIX_INFNORM] = lu.infnorm;
    xstore[BASICLU_RESIDUAL_TEST] = lu.residual_test;

    xstore[BASICLU_MATRIX_NZ] = lu.matrix_nz as f64;
    xstore[BASICLU_RANK] = lu.rank as f64;
    xstore[BASICLU_BUMP_SIZE] = lu.bump_size as f64;
    xstore[BASICLU_BUMP_NZ] = lu.bump_nz as f64;
    xstore[BASICLU_NSEARCH_PIVOT] = lu.nsearch_pivot as f64;
    xstore[BASICLU_NEXPAND] = lu.nexpand as f64;
    xstore[BASICLU_NGARBAGE] = lu.ngarbage as f64;
    xstore[BASICLU_FACTOR_FLOPS] = lu.factor_flops as f64;
    xstore[BASICLU_TIME_SINGLETONS] = lu.time_singletons;
    xstore[BASICLU_TIME_SEARCH_PIVOT] = lu.time_search_pivot;
    xstore[BASICLU_TIME_ELIM_PIVOT] = lu.time_elim_pivot;

    xstore[BASICLU_PIVOT_ERROR] = lu.pivot_error;

    // private
    xstore[BASICLU_TASK] = lu.task as f64;
    xstore[BASICLU_PIVOT_ROW] = lu.pivot_row as f64;
    xstore[BASICLU_PIVOT_COL] = lu.pivot_col as f64;
    xstore[BASICLU_FTCOLUMN_IN] = lu.ftran_for_update as f64;
    xstore[BASICLU_FTCOLUMN_OUT] = lu.btran_for_update as f64;
    xstore[BASICLU_MARKER] = lu.marker as f64;
    xstore[BASICLU_PIVOTLEN] = lu.pivotlen as f64;
    xstore[BASICLU_RANKDEF] = lu.rankdef as f64;
    xstore[BASICLU_MIN_COLNZ] = lu.min_colnz as f64;
    xstore[BASICLU_MIN_ROWNZ] = lu.min_rownz as f64;

    status
}

/// Reset @lu for a new factorization. Invalidate current factorization.
pub(crate) fn lu_reset(lu: &mut LU) {
    // user readable
    lu.nupdate = -1; // invalidate factorization
    lu.nforrest = 0;
    lu.l_nz = 0;
    lu.u_nz = 0;
    lu.r_nz = 0;
    lu.min_pivot = 0.0;
    lu.max_pivot = 0.0;
    lu.max_eta = 0.0;
    lu.update_cost_numer = 0.0;
    lu.update_cost_denom = 1.0;
    lu.time_factorize = 0.0;
    lu.time_solve = 0.0;
    lu.time_update = 0.0;
    lu.l_flops = 0;
    lu.u_flops = 0;
    lu.r_flops = 0;
    lu.condest_l = 0.0;
    lu.condest_u = 0.0;
    lu.norm_l = 0.0;
    lu.norm_u = 0.0;
    lu.normest_l_inv = 0.0;
    lu.normest_u_inv = 0.0;
    lu.onenorm = 0.0;
    lu.infnorm = 0.0;
    lu.residual_test = 0.0;

    lu.matrix_nz = 0;
    lu.rank = 0;
    lu.bump_size = 0;
    lu.bump_nz = 0;
    lu.nsearch_pivot = 0;
    lu.nexpand = 0;
    lu.ngarbage = 0;
    lu.factor_flops = 0;
    lu.time_singletons = 0.0;
    lu.time_search_pivot = 0.0;
    lu.time_elim_pivot = 0.0;

    lu.pivot_error = 0.0;

    // private
    lu.task = NO_TASK;
    lu.pivot_row = -1;
    lu.pivot_col = -1;
    lu.ftran_for_update = -1;
    lu.btran_for_update = -1;
    lu.marker = 0;
    lu.pivotlen = 0;
    lu.rankdef = 0;
    lu.min_colnz = 1;
    lu.min_rownz = 1;

    // One past the final position in @Wend must hold the file size.
    // The file has 2*m lines during factorization.
    lu.w_end[2 * lu.m as usize] = lu.w_mem;

    // The integer workspace iwork0 must be zeroed for a new factorization.
    // The double workspace work0 actually needs only be zeroed once in the
    // initialization of xstore. However, it is easier and more consistent
    // to do that here as well.
    // memset(lu.iwork0, 0, lu.m);
    lu.iwork0.fill(0);

    // memset(lu.work0, 0, lu.m);
    lu.work0.fill(0.0);
}
