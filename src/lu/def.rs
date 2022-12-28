// Copyright (C) 2016-2018 ERGO-Code
// Copyright (C) 2022-2023 Richard Lincoln

use crate::LUInt;

pub(crate) enum Task {
    NoTask,
    Singletons,
    SetupBump,
    FactorizeBump,
    BuildFactors,
}

impl Default for Task {
    fn default() -> Self {
        Self::Singletons
    }
}

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
