use crate::basiclu::*;
use crate::lu_def::BASICLU_HASH;
use crate::lu_internal::{lu_load, lu_reset, lu_save, LU};

/// lu_initialize()
///
/// Make @istore, @xstore a BASICLU instance. Set parameters to defaults and
/// initialize global counters. Reset instance for a fresh factorization.
pub(crate) fn lu_initialize(m: LUInt, /*istore: &mut [lu_int],*/ xstore: &mut [f64]) {
    let mut lu = LU {
        ..Default::default()
    };

    // set constant entries
    // istore[0] = BASICLU_HASH;
    xstore[0] = BASICLU_HASH as f64;
    xstore[BASICLU_DIM] = m as f64;

    // set default parameters
    xstore[BASICLU_MEMORYL] = 0.0;
    xstore[BASICLU_MEMORYU] = 0.0;
    xstore[BASICLU_MEMORYW] = 0.0;
    xstore[BASICLU_DROP_TOLERANCE] = 1e-20;
    xstore[BASICLU_ABS_PIVOT_TOLERANCE] = 1e-14;
    xstore[BASICLU_REL_PIVOT_TOLERANCE] = 0.1;
    xstore[BASICLU_BIAS_NONZEROS] = 1.0;
    xstore[BASICLU_MAXN_SEARCH_PIVOT] = 3.0;
    xstore[BASICLU_PAD] = 4.0;
    xstore[BASICLU_STRETCH] = 0.3;
    xstore[BASICLU_COMPRESSION_THRESHOLD] = 0.5;
    xstore[BASICLU_SPARSE_THRESHOLD] = 0.05;
    xstore[BASICLU_REMOVE_COLUMNS] = 0.0;
    xstore[BASICLU_SEARCH_ROWS] = 1.0;

    // initialize global counters
    xstore[BASICLU_NFACTORIZE] = 0.0;
    xstore[BASICLU_NUPDATE_TOTAL] = 0.0;
    xstore[BASICLU_NFORREST_TOTAL] = 0.0;
    xstore[BASICLU_NSYMPERM_TOTAL] = 0.0;
    xstore[BASICLU_TIME_FACTORIZE_TOTAL] = 0.0;
    xstore[BASICLU_TIME_SOLVE_TOTAL] = 0.0;
    xstore[BASICLU_TIME_UPDATE_TOTAL] = 0.0;

    // lu_reset() and lu_save() initializes the remaining slots
    lu_load(
        &mut lu, /*istore,*/ xstore, //None, None, None, None, None, None,
    );
    lu_reset(&mut lu);
    lu_save(&lu, /*istore,*/ xstore, BASICLU_OK);
}
