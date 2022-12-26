// Copyright (C) 2016-2018  ERGO-Code
//
// LINPACK condition number estimate

use crate::basiclu::LUInt;

/// Given `m`-by-`m` matrix `U` such that `U[perm,perm]` is upper triangular,
/// return estimate for 1-norm condition number of `U`.
/// If `norm` is not None, it holds the 1-norm of the matrix on return.
/// If `norminv` is not None, it holds the estimated 1-norm of the inverse on
/// return.
///  
/// The other function arguments are the same as in [`lu_normest()`].
pub(crate) fn lu_condest(
    m: LUInt,
    u_begin: &[LUInt],
    u_i: &[LUInt],
    u_x: &[f64],
    pivot: Option<&[f64]>,
    perm: Option<&[LUInt]>,
    upper: i32,
    work: &mut [f64],
    norm: Option<&mut f64>,
    norminv: Option<&mut f64>,
) -> f64 {
    // compute 1-norm of U
    let mut u_norm = 0.0;
    for j in 0..m as usize {
        let mut colsum: f64 = if let Some(pivot) = pivot {
            pivot[j].abs()
        } else {
            1.0
        };
        let mut p = u_begin[j] as usize;
        while u_i[p] >= 0 {
            colsum += u_x[p].abs();
            p += 1;
        }
        u_norm = f64::max(u_norm, colsum);
    }

    // estimate 1-norm of U^{-1}
    let u_invnorm = lu_normest(m, u_begin, u_i, u_x, pivot, perm, upper, work);

    if let Some(norm) = norm {
        *norm = u_norm;
    }
    if let Some(norminv) = norminv {
        *norminv = u_invnorm;
    }

    u_norm * u_invnorm
}

// Given `m`-by-`m` matrix `U` such that `U[perm,perm]` is triangular,
// estimate 1-norm of `U^{-1}` by computing
//
//     U'x = b, Uy = x, normest = max{norm(y)_1/norm(x)_1, norm(x)_inf},
//
// where the entries of `b` are +/-1 chosen dynamically to make `x` large.
// The method is described in [1].
//
// - `u_begin`, `u_`, `u_x` matrix `U` in compressed column format without pivots,
//                          columns are terminated by a negative index
// - `pivot` pivot elements by column index of `U`; None if unit pivots
// - `perm` permutation to triangular form; None if identity
// - `upper` nonzero if permuted matrix is upper triangular; zero if lower
// - `work` size `m` workspace, uninitialized on entry/return
//
// Return: estimate for 1-norm of `U^{-1}`
//
// [1] I. Duff, A. Erisman, J. Reid, "Direct Methods for Sparse Matrices"
pub(crate) fn lu_normest(
    m: LUInt,
    u_begin: &[LUInt],
    u_i: &[LUInt],
    u_x: &[f64],
    pivot: Option<&[f64]>,
    perm: Option<&[LUInt]>,
    upper: i32,
    work: &mut [f64],
) -> f64 {
    let mut x1norm = 0.0;
    let mut xinfnorm = 0.0;
    let (kbeg, kend, kinc): (LUInt, LUInt, LUInt) = if upper != 0 {
        let kbeg = 0;
        let kend = m;
        let kinc = 1;
        (kbeg, kend, kinc)
    } else {
        let kbeg = m - 1;
        let kend = -1;
        let kinc = -1;
        (kbeg, kend, kinc)
    };
    let mut k = kbeg;
    while k != kend {
        let j = if let Some(perm) = perm {
            perm[k as usize] as usize
        } else {
            k as usize
        };
        let mut temp = 0.0;
        // for (p = u_begin[j]; (i = u_i[p]) >= 0; p++) {
        let mut p = u_begin[j] as usize;
        while u_i[p] >= 0 {
            temp -= work[u_i[p] as usize] * u_x[p];
            p += 1;
        }
        temp += if temp >= 0.0 { 1.0 } else { -1.0 }; // choose b[i] = 1 or b[i] = -1
        if let Some(pivot) = pivot {
            temp /= pivot[j];
        }
        work[j] = temp;
        x1norm += temp.abs();
        xinfnorm = f64::max(xinfnorm, temp.abs());
        k += kinc;
    }

    let mut y1norm = 0.0;
    let (kbeg, kend, kinc): (LUInt, LUInt, LUInt) = if upper != 0 {
        let kbeg = m - 1;
        let kend = -1;
        let kinc = -1;
        (kbeg, kend, kinc)
    } else {
        let kbeg = 0;
        let kend = m;
        let kinc = 1;
        (kbeg, kend, kinc)
    };
    // for (k = kbeg; k != kend; k += kinc)
    let mut k = kbeg;
    while k != kend {
        let j = if let Some(perm) = perm {
            perm[k as usize] as usize
        } else {
            k as usize
        };
        if let Some(pivot) = pivot {
            work[j] /= pivot[j];
        }
        let temp = work[j];
        // for (p = u_begin[j]; (i = u_i[p]) >= 0; p++) {
        let mut p = u_begin[j] as usize;
        while u_i[p] >= 0 {
            work[u_i[p] as usize] -= temp * u_x[p];
            p += 1;
        }
        y1norm += temp.abs();

        k += kinc;
    }

    f64::max(y1norm / x1norm, xinfnorm)
}
