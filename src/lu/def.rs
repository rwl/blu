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

pub(crate) fn iswap(x: &mut [LUInt], i: usize, j: usize) {
    let t = x[i];
    x[i] = x[j];
    x[j] = t;
}

pub(crate) fn fswap(x: &mut [f64], i: usize, j: usize) {
    let t = x[i];
    x[i] = x[j];
    x[j] = t;
}
