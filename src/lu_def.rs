use crate::basiclu::lu_int;

pub(crate) const BASICLU_HASH: lu_int = 7743090; // hash in istore[0], xstore[0]

pub(crate) const NO_TASK: lu_int = 0;
pub(crate) const SINGLETONS: lu_int = 1;
pub(crate) const SETUP_BUMP: lu_int = 2;
pub(crate) const FACTORIZE_BUMP: lu_int = 3;
pub(crate) const BUILD_FACTORS: lu_int = 4;

pub(crate) fn lu_iswap(x: &mut [lu_int], i: lu_int, j: lu_int) {
    let t = x[i];
    x[i] = x[j];
    x[j] = t;
}

pub(crate) fn lu_fswap(x: &mut [f64], i: lu_int, j: lu_int) {
    let t = x[i];
    x[i] = x[j];
    x[j] = t;
}
