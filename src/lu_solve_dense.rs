// Copyright (C) 2016-2018  ERGO-Code

use crate::basiclu::LUInt;
use crate::lu_garbage_perm::lu_garbage_perm;
use crate::lu_internal::*;

pub(crate) fn lu_solve_dense(lu: &mut LU, rhs: &[f64], lhs: &mut [f64], trans: char) {
    lu_garbage_perm(lu);
    assert_eq!(lu.pivotlen, lu.m);

    let m = lu.m;
    let nforrest = lu.nforrest;
    let p = &p!(lu);
    let eta_row = &eta_row!(lu);
    let pivotcol = &pivotcol!(lu);
    let pivotrow = &pivotrow!(lu);
    let l_begin_p = &lu.l_begin_p;
    let lt_begin_p = &lt_begin_p!(lu);
    let u_begin = &lu.u_begin;
    let r_begin = &r_begin!(lu);
    let w_begin = &lu.w_begin;
    let w_end = &lu.w_end;
    let col_pivot = &lu.col_pivot;
    let row_pivot = &lu.row_pivot;
    let l_index = &lu.l_index;
    let l_value = &lu.l_value;
    let u_index = &lu.u_index;
    let u_value = &lu.u_value;
    let w_index = &lu.w_index;
    let w_value = &lu.w_value;
    let work1 = &mut lu.work1;

    if trans == 't' || trans == 'T' {
        // Solve transposed system

        // memcpy(work1, rhs, m*sizeof(double));
        work1[..m as usize].copy_from_slice(rhs);

        // Solve with U'.
        for k in 0..m {
            let jpivot = pivotcol[k as usize] as usize;
            let ipivot = pivotrow[k as usize] as usize;
            let x = work1[jpivot] / col_pivot[jpivot];
            for pos in w_begin[jpivot]..w_end[jpivot] {
                work1[w_index[pos as usize] as usize] -= x * w_value[pos as usize];
            }
            lhs[ipivot] = x;
        }

        // Solve with update ETAs backwards.
        // for (t = nforrest-1; t >= 0; t--)
        for t in (0..nforrest).rev() {
            let ipivot = eta_row[t as usize];
            let x = lhs[ipivot as usize];
            for pos in r_begin[t as usize]..r_begin[(t + 1) as usize] {
                let i = l_index[pos as usize] as usize;
                lhs[i] -= x * l_value[pos as usize];
            }
        }

        // Solve with L'.
        // for (k = m-1; k >= 0; k--)
        for k in (0..m).rev() {
            let mut x = 0.0;
            // for (pos = Lbegin_p[k]; (i = Lindex[pos]) >= 0; pos++)
            let mut pos = l_begin_p[k as usize] as usize;
            while l_index[pos] >= 0 {
                let i = l_index[pos];
                x += lhs[i as usize] * l_value[pos];
                pos += 1;
            }
            lhs[p[k as usize] as usize] -= x;
        }
    } else {
        // Solve forward system //

        // memcpy(work1, rhs, m*sizeof(double));
        work1[..m as usize].copy_from_slice(rhs);

        // Solve with L.
        for k in 0..m {
            let mut x = 0.0;
            let mut pos = lt_begin_p[k as usize] as usize;
            while l_index[pos] >= 0 {
                let i = l_index[pos];
                x += work1[i as usize] * l_value[pos];
                pos += 1;
            }
            work1[p[k as usize] as usize] -= x;
        }

        // Solve with update ETAs.
        let mut pos = r_begin[0];
        for t in 0..nforrest as usize {
            let ipivot = eta_row[t];
            let mut x = 0.0;
            while pos < r_begin[t + 1] {
                x += work1[l_index[pos as usize] as usize] * l_value[pos as usize];
                pos += 1;
            }
            work1[ipivot as usize] -= x;
        }

        // Solve with U.
        // for (k = m-1; k >= 0; k--)
        for k in (0..m).rev() {
            let jpivot = pivotcol[k as usize] as usize;
            let ipivot = pivotrow[k as usize] as usize;
            let x = work1[ipivot] / row_pivot[ipivot];
            // for (pos = Ubegin[ipivot]; (i = Uindex[pos]) >= 0; pos++)
            let mut pos = u_begin[ipivot] as usize;
            while u_index[pos] >= 0 {
                let i = u_index[pos];
                work1[i as usize] -= x * u_value[pos];
                pos += 1;
            }
            lhs[jpivot] = x;
        }
    }
}
