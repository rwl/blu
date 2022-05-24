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
pub struct lu {
    // user parameters, not modified //
    pub Lmem: lu_int,
    pub Umem: lu_int,
    pub Wmem: lu_int,
    pub droptol: f64,
    pub abstol: f64,
    pub reltol: f64,
    pub nzbias: lu_int,
    pub maxsearch: lu_int,
    pub pad: lu_int,
    pub stretch: f64,
    pub compress_thres: f64,
    pub sparse_thres: f64,
    pub search_rows: lu_int,

    // user readable //
    pub m: lu_int,
    pub addmemL: lu_int,
    pub addmemU: lu_int,
    pub addmemW: lu_int,

    pub nupdate: lu_int,
    pub nforrest: lu_int,
    pub nfactorize: lu_int,
    pub nupdate_total: lu_int,
    pub nforrest_total: lu_int,
    pub nsymperm_total: lu_int,
    pub Lnz: lu_int, // nz in L excluding diagonal
    pub Unz: lu_int, // nz in U excluding diagonal
    pub Rnz: lu_int, // nz in update etas excluding diagonal
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
    pub Lflops: lu_int,
    pub Uflops: lu_int,
    pub Rflops: lu_int,
    pub condestL: f64,
    pub condestU: f64,
    pub normL: f64,
    pub normU: f64,
    pub normestLinv: f64,
    pub normestUinv: f64,
    pub onenorm: f64,       // 1-norm and inf-norm of matrix after fresh
    pub infnorm: f64,       // factorization with dependent cols replaced
    pub residual_test: f64, // computed by lu_residual_test()

    pub matrix_nz: lu_int, // nz in basis matrix when factorized
    pub rank: lu_int,      // rank of basis matrix when factorized
    pub bump_size: lu_int,
    pub bump_nz: lu_int,
    pub nsearch_pivot: lu_int, // # rows/cols searched for pivot
    pub nexpand: lu_int,       // # rows/cols expanded in factorize
    pub ngarbage: lu_int,      // # garbage collections in factorize
    pub factor_flops: lu_int,  // # flops in factorize
    pub time_singletons: f64,
    pub time_search_pivot: f64,
    pub time_elim_pivot: f64,

    pub pivot_error: f64, // error estimate for pivot in last update

    // private //
    pub(crate) task: lu_int,      // the part of factorization in progress
    pub(crate) pivot_row: lu_int, // chosen pivot row
    pub(crate) pivot_col: lu_int, // chosen pivot column
    pub(crate) ftran_for_update: lu_int, // >= 0 if FTRAN prepared for update
    pub(crate) btran_for_update: lu_int, // >= 0 if BTRAN prepared for update
    pub(crate) marker: lu_int,    // see @marked, below
    pub(crate) pivotlen: lu_int,  // length of @pivotcol, @pivotrow; <= 2*m
    pub(crate) rankdef: lu_int,   // # columns removed from active submatrix
    // because maximum was 0 or < abstol
    pub(crate) min_colnz: lu_int, // colcount lists 1..min_colnz-1 are empty
    pub(crate) min_rownz: lu_int, // rowcount lists 1..min_rownz-1 are empty

    // aliases to user arrays //
    pub(crate) Lindex: Option<Vec<lu_int>>,
    pub(crate) Uindex: Option<Vec<lu_int>>,
    pub(crate) Windex: Option<Vec<lu_int>>,

    pub(crate) Lvalue: Option<Vec<f64>>,
    pub(crate) Uvalue: Option<Vec<f64>>,
    pub(crate) Wvalue: Option<Vec<f64>>,

    // pointers into istore
    //
    // When two declaration lists are on one line, then the arrays from the
    // second list share memory with the array from the first list. The arrays
    // from the first lists are used during factorization, the arrays from the
    // second lists are used during solves/updates.

    // documented in lu_singletons.c, lu_setup_bump.c, lu_build_factors.c
    pub(crate) colcount_flink: Vec<lu_int>,
    pub(crate) pivotcol: Vec<lu_int>,
    pub(crate) colcount_blink: Vec<lu_int>,
    pub(crate) pivotrow: Vec<lu_int>,
    pub(crate) rowcount_flink: Vec<lu_int>,
    pub(crate) Rbegin: Vec<lu_int>,
    pub(crate) eta_row: Vec<lu_int>,
    pub(crate) rowcount_blink: Vec<lu_int>,
    pub(crate) iwork1: Vec<lu_int>,
    pub(crate) Wbegin: Vec<lu_int>,
    pub(crate) Lbegin: Vec<lu_int>, // + Wbegin reused
    pub(crate) Wend: Vec<lu_int>,
    pub(crate) Ltbegin: Vec<lu_int>, // + Wend   reused
    pub(crate) Wflink: Vec<lu_int>,
    pub(crate) Ltbegin_p: Vec<lu_int>, // + Wflink reused
    pub(crate) Wblink: Vec<lu_int>,
    pub(crate) p: Vec<lu_int>, // + Wblink reused
    pub(crate) pinv: Vec<lu_int>,
    pub(crate) pmap: Vec<lu_int>,
    pub(crate) qinv: Vec<lu_int>,
    pub(crate) qmap: Vec<lu_int>,
    pub(crate) Lbegin_p: Vec<lu_int>, // Lbegin_p reused
    pub(crate) Ubegin: Vec<lu_int>,   // Ubegin   reused

    pub(crate) iwork0: Vec<lu_int>,
    pub(crate) marked: Vec<lu_int>,
    // iwork0: size m workspace, zeroed
    // marked: size m workspace, 0 <= marked[i] <= @marker

    // pointers into xstore
    pub(crate) work0: Vec<f64>,     // size m workspace, zeroed
    pub(crate) work1: Vec<f64>,     // size m workspace, uninitialized
    pub(crate) col_pivot: Vec<f64>, // pivot elements by column index
    pub(crate) row_pivot: Vec<f64>, // pivot elements by row index
}

/// Initialize @this from @istore, @xstore if these are a valid BASICLU
/// instance. The remaining arguments are copied only and can be NULL.
///
/// Return `BASICLU_OK` or `BASICLU_ERROR_invalid_store`
pub(crate) fn lu_load(
    this: &mut lu,
    istore: &[lu_int],
    xstore: &[f64],
    Li: Option<Vec<lu_int>>,
    Lx: Option<Vec<f64>>,
    Ui: Option<Vec<lu_int>>,
    Ux: Option<Vec<f64>>,
    Wi: Option<Vec<lu_int>>,
    Wx: Option<Vec<f64>>,
) -> lu_int {
    if istore[0] != BASICLU_HASH || xstore[0] != BASICLU_HASH as f64 {
        return BASICLU_ERROR_invalid_store;
    }

    // user parameters
    this.Lmem = xstore[BASICLU_MEMORYL] as lu_int;
    this.Umem = xstore[BASICLU_MEMORYU] as lu_int;
    this.Wmem = xstore[BASICLU_MEMORYW] as lu_int;
    this.droptol = xstore[BASICLU_DROP_TOLERANCE];
    this.abstol = xstore[BASICLU_ABS_PIVOT_TOLERANCE];
    this.reltol = xstore[BASICLU_REL_PIVOT_TOLERANCE];
    this.reltol = f64::min(this.reltol, 1.0);
    this.nzbias = xstore[BASICLU_BIAS_NONZEROS] as lu_int;
    this.maxsearch = xstore[BASICLU_MAXN_SEARCH_PIVOT] as lu_int;
    this.pad = xstore[BASICLU_PAD] as lu_int;
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
    this.m = m as lu_int;
    this.addmemL = 0;
    this.addmemU = 0;
    this.addmemW = 0;

    this.nupdate = xstore[BASICLU_NUPDATE] as lu_int;
    this.nforrest = xstore[BASICLU_NFORREST] as lu_int;
    this.nfactorize = xstore[BASICLU_NFACTORIZE] as lu_int;
    this.nupdate_total = xstore[BASICLU_NUPDATE_TOTAL] as lu_int;
    this.nforrest_total = xstore[BASICLU_NFORREST_TOTAL] as lu_int;
    this.nsymperm_total = xstore[BASICLU_NSYMPERM_TOTAL] as lu_int;
    this.Lnz = xstore[BASICLU_LNZ] as lu_int;
    this.Unz = xstore[BASICLU_UNZ] as lu_int;
    this.Rnz = xstore[BASICLU_RNZ] as lu_int;
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
    this.Lflops = xstore[BASICLU_LFLOPS] as lu_int;
    this.Uflops = xstore[BASICLU_UFLOPS] as lu_int;
    this.Rflops = xstore[BASICLU_RFLOPS] as lu_int;
    this.condestL = xstore[BASICLU_CONDEST_L];
    this.condestU = xstore[BASICLU_CONDEST_U];
    this.normL = xstore[BASICLU_NORM_L];
    this.normU = xstore[BASICLU_NORM_U];
    this.normestLinv = xstore[BASICLU_NORMEST_LINV];
    this.normestUinv = xstore[BASICLU_NORMEST_UINV];
    this.onenorm = xstore[BASICLU_MATRIX_ONENORM];
    this.infnorm = xstore[BASICLU_MATRIX_INFNORM];
    this.residual_test = xstore[BASICLU_RESIDUAL_TEST];

    this.matrix_nz = xstore[BASICLU_MATRIX_NZ] as lu_int;
    this.rank = xstore[BASICLU_RANK] as lu_int;
    this.bump_size = xstore[BASICLU_BUMP_SIZE] as lu_int;
    this.bump_nz = xstore[BASICLU_BUMP_NZ] as lu_int;
    this.nsearch_pivot = xstore[BASICLU_NSEARCH_PIVOT] as lu_int;
    this.nexpand = xstore[BASICLU_NEXPAND] as lu_int;
    this.ngarbage = xstore[BASICLU_NGARBAGE] as lu_int;
    this.factor_flops = xstore[BASICLU_FACTOR_FLOPS] as lu_int;
    this.time_singletons = xstore[BASICLU_TIME_SINGLETONS];
    this.time_search_pivot = xstore[BASICLU_TIME_SEARCH_PIVOT];
    this.time_elim_pivot = xstore[BASICLU_TIME_ELIM_PIVOT];

    this.pivot_error = xstore[BASICLU_PIVOT_ERROR];

    // private
    this.task = xstore[BASICLU_TASK] as lu_int;
    this.pivot_row = xstore[BASICLU_PIVOT_ROW] as lu_int;
    this.pivot_col = xstore[BASICLU_PIVOT_COL] as lu_int;
    this.ftran_for_update = xstore[BASICLU_FTCOLUMN_IN] as lu_int;
    this.btran_for_update = xstore[BASICLU_FTCOLUMN_OUT] as lu_int;
    this.marker = xstore[BASICLU_MARKER] as lu_int;
    this.pivotlen = xstore[BASICLU_PIVOTLEN] as lu_int;
    this.rankdef = xstore[BASICLU_RANKDEF] as lu_int;
    this.min_colnz = xstore[BASICLU_MIN_COLNZ] as lu_int;
    this.min_rownz = xstore[BASICLU_MIN_ROWNZ] as lu_int;

    // aliases to user arrays
    this.Lindex = Li;
    this.Lvalue = Lx;
    this.Uindex = Ui;
    this.Uvalue = Ux;
    this.Windex = Wi;
    this.Wvalue = Wx;

    // partition istore for factorize
    // iptr = istore + 1;
    let (_, istore) = istore.split_at(1);
    let (colcount_flink, istore) = istore.split_at(2 * m as usize + 2);
    this.colcount_flink = Vec::from(colcount_flink);
    // iptr += 2 * m + 2;
    let (colcount_blink, istore) = istore.split_at(2 * m as usize + 2);
    this.colcount_blink = Vec::from(colcount_blink);
    // iptr += 2 * m + 2;
    let (rowcount_flink, istore) = istore.split_at(2 * m as usize + 2);
    this.rowcount_flink = Vec::from(rowcount_flink);
    // iptr += 2 * m + 2;
    let (rowcount_blink, istore) = istore.split_at(2 * m as usize + 2);
    this.rowcount_blink = Vec::from(rowcount_blink);
    // iptr += 2 * m + 2;
    let (Wbegin, istore) = istore.split_at(2 * m as usize + 1);
    this.Wbegin = Vec::from(Wbegin);
    // iptr += 2 * m + 1;
    let (Wend, istore) = istore.split_at(2 * m as usize + 1);
    this.Wend = Vec::from(Wend);
    // iptr += 2 * m + 1;
    let (Wflink, istore) = istore.split_at(2 * m as usize + 1);
    this.Wflink = Vec::from(Wflink);
    // iptr += 2 * m + 1;
    let (Wblink, istore) = istore.split_at(2 * m as usize + 1);
    this.Wblink = Vec::from(Wblink);
    // iptr += 2 * m + 1;
    let (pinv, istore) = istore.split_at(m as usize);
    this.pinv = Vec::from(pinv);
    // iptr += m;
    let (qinv, istore) = istore.split_at(m as usize);
    this.qinv = Vec::from(qinv);
    // iptr += m;
    let (Lbegin_p, istore) = istore.split_at(m as usize + 1);
    this.Lbegin_p = Vec::from(Lbegin_p);
    // iptr += m + 1;
    let (Ubegin, istore) = istore.split_at(m as usize + 1);
    this.Ubegin = Vec::from(Ubegin);
    // iptr += m + 1;
    let (iwork0, istore) = istore.split_at(m as usize);
    this.iwork0 = Vec::from(iwork0);
    // iptr += m;

    // share istore memory for solve/update
    this.pivotcol = this.colcount_flink.clone();
    this.pivotrow = this.colcount_blink.clone();
    this.Rbegin = this.rowcount_flink.clone();
    // this.eta_row = this.rowcount_flink + m + 1;
    this.eta_row = this.rowcount_flink[m as usize + 1..].to_vec();
    this.iwork1 = this.rowcount_blink.clone();
    // this.Lbegin = this.Wbegin + m + 1;
    this.Lbegin = this.Wbegin[m as usize + 1..].to_vec();
    // this.Ltbegin = this.Wend + m + 1;
    this.Ltbegin = this.Wend[m as usize + 1..].to_vec();
    // this.Ltbegin_p = this.Wflink + m + 1;
    this.Ltbegin_p = this.Wflink[m as usize + 1..].to_vec();
    // this.p = this.Wblink + m + 1;
    this.p = this.Wblink[m as usize + 1..].to_vec();
    this.pmap = this.pinv.clone();
    this.qmap = this.qinv.clone();
    this.marked = this.iwork0.clone();

    // partition xstore for factorize and update
    // let xptr = xstore + 512;
    let (_, xstore) = xstore.split_at(512);
    let (work0, xstore) = xstore.split_at(m as usize);
    this.work0 = Vec::from(work0);
    // xptr += m;
    let (work1, xstore) = xstore.split_at(m as usize);
    this.work1 = Vec::from(work1);
    // xptr += m;
    let (col_pivot, xstore) = xstore.split_at(m as usize);
    this.col_pivot = Vec::from(col_pivot);
    // xptr += m;
    let (row_pivot, xstore) = xstore.split_at(m as usize);
    this.row_pivot = Vec::from(row_pivot);
    // xptr += m;

    // Reset @marked if increasing @marker by four causes overflow.
    if this.marker > LU_INT_MAX - 4 {
        // memset(this.marked, 0, m * sizeof(lu_int));
        this.marked.fill(0);
        this.marker = 0;
    }

    // One past the final position in @Wend must hold the file size.
    // The file has 2*m lines while factorizing and m lines otherwise.
    if this.nupdate >= 0 {
        this.Wend[m as usize] = this.Wmem;
    } else {
        this.Wend[2 * m as usize] = this.Wmem;
    }

    BASICLU_OK
}

/// Copy scalar entries (except for user parameters) from @this to @istore,
/// @xstore. Store status code.
///
/// Return @status
pub(crate) fn lu_save(
    this: &lu,
    istore: &mut [lu_int],
    xstore: &mut [f64],
    status: lu_int,
) -> lu_int {
    // user readable
    xstore[BASICLU_STATUS] = status as f64;
    xstore[BASICLU_ADD_MEMORYL] = this.addmemL as f64;
    xstore[BASICLU_ADD_MEMORYU] = this.addmemU as f64;
    xstore[BASICLU_ADD_MEMORYW] = this.addmemW as f64;

    xstore[BASICLU_NUPDATE] = this.nupdate as f64;
    xstore[BASICLU_NFORREST] = this.nforrest as f64;
    xstore[BASICLU_NFACTORIZE] = this.nfactorize as f64;
    xstore[BASICLU_NUPDATE_TOTAL] = this.nupdate_total as f64;
    xstore[BASICLU_NFORREST_TOTAL] = this.nforrest_total as f64;
    xstore[BASICLU_NSYMPERM_TOTAL] = this.nsymperm_total as f64;
    xstore[BASICLU_LNZ] = this.Lnz as f64;
    xstore[BASICLU_UNZ] = this.Unz as f64;
    xstore[BASICLU_RNZ] = this.Rnz as f64;
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
    xstore[BASICLU_LFLOPS] = this.Lflops as f64;
    xstore[BASICLU_UFLOPS] = this.Uflops as f64;
    xstore[BASICLU_RFLOPS] = this.Rflops as f64;
    xstore[BASICLU_CONDEST_L] = this.condestL;
    xstore[BASICLU_CONDEST_U] = this.condestU;
    xstore[BASICLU_NORM_L] = this.normL;
    xstore[BASICLU_NORM_U] = this.normU;
    xstore[BASICLU_NORMEST_LINV] = this.normestLinv;
    xstore[BASICLU_NORMEST_UINV] = this.normestUinv;
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
pub(crate) fn lu_reset(this: &mut lu) {
    // user readable
    this.nupdate = -1; // invalidate factorization
    this.nforrest = 0;
    this.Lnz = 0;
    this.Unz = 0;
    this.Rnz = 0;
    this.min_pivot = 0.0;
    this.max_pivot = 0.0;
    this.max_eta = 0.0;
    this.update_cost_numer = 0.0;
    this.update_cost_denom = 1.0;
    this.time_factorize = 0.0;
    this.time_solve = 0.0;
    this.time_update = 0.0;
    this.Lflops = 0;
    this.Uflops = 0;
    this.Rflops = 0;
    this.condestL = 0.0;
    this.condestU = 0.0;
    this.normL = 0.0;
    this.normU = 0.0;
    this.normestLinv = 0.0;
    this.normestUinv = 0.0;
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
    this.Wend[2 * this.m as usize] = this.Wmem;

    // The integer workspace iwork0 must be zeroed for a new factorization.
    // The double workspace work0 actually needs only be zeroed once in the
    // initialization of xstore. However, it is easier and more consistent
    // to do that here as well.
    // memset(this.iwork0, 0, this.m);
    this.iwork0.fill(0);
    // memset(this.work0, 0, this.m);
    this.work0.fill(0.0);
}
