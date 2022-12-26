use crate::blu::LUInt;

pub(crate) const BLU_HASH: LUInt = 7743090; // hash in istore[0], xstore[0]

pub(crate) const NO_TASK: LUInt = 0;
pub(crate) const SINGLETONS: LUInt = 1;
pub(crate) const SETUP_BUMP: LUInt = 2;
pub(crate) const FACTORIZE_BUMP: LUInt = 3;
pub(crate) const BUILD_FACTORS: LUInt = 4;

pub(crate) fn iswap(x: &mut [LUInt], i: LUInt, j: LUInt) {
    let t = x[i as usize];
    x[i as usize] = x[j as usize];
    x[j as usize] = t;
}

pub(crate) fn fswap(x: &mut [f64], i: LUInt, j: LUInt) {
    let t = x[i as usize];
    x[i as usize] = x[j as usize];
    x[j as usize] = t;
}
