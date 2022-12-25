// Copyright (C) 2016-2018  ERGO-Code

use crate::basiclu::*;
use crate::lu_initialize::lu_initialize;
use crate::{
    basiclu_factorize, basiclu_get_factors, basiclu_solve_dense, basiclu_solve_for_update,
    basiclu_solve_sparse, basiclu_update,
};

/// A variable of type struct BasicLUObject must be defined in user code. Its
/// members are set and maintained by basiclu_obj_* routines. User code should only
/// access the following members:
///
///     xstore (read/write)
///
///         set parameters and get info values
///
///     lhs, ilhs, nzlhs (read only)
///
///         holds solution after solve_sparse() and solve_for_update()
///
///     realloc_factor (read/write)
///
///         Arrays are reallocated for max(realloc_factor, 1.0) times the
///         required size. Default: 1.5
pub struct BasicLUObject {
    pub istore: Vec<LUInt>,
    pub xstore: Vec<f64>,

    l_i: Vec<LUInt>,
    u_i: Vec<LUInt>,
    w_i: Vec<LUInt>,
    l_x: Vec<f64>,
    u_x: Vec<f64>,
    w_x: Vec<f64>,

    pub lhs: Vec<f64>,
    pub ilhs: Vec<LUInt>,
    pub nzlhs: LUInt,

    pub realloc_factor: f64,
}

/// Purpose:
///
///     Initialize a BASICLU object. When m is positive, then *obj is initialized to
///     process matrices of dimension m. When m is zero, then *obj is initialized to
///     a "null" object, which cannot be used for factorization, but can be passed
///     to basiclu_obj_free().
///
///     This routine must be called once before passing obj to any other
///     basiclu_obj_ routine. When obj is initialized to a null object, then the
///     routine can be called again to reinitialize obj.
///
/// Return:
///
///     BASICLU_OK
///
///         *obj successfully initialized.
///
///     BASICLU_ERROR_ARGUMENT_MISSING
///
///         obj is NULL.
///
///     BASICLU_ERROR_INVALID_ARGUMENT
///
///         m is negative.
///
///     BASICLU_ERROR_OUT_OF_MEMORY
///
///         insufficient memory to initialize object.
///
/// Arguments:
///
///     struct BasicLUObject *obj
///
///         Pointer to the object to be initialized.
///
///     lu_int m
///
///         The dimension of matrices which can be processed, or 0.
pub fn basiclu_obj_initialize(obj: &mut BasicLUObject, m: LUInt) -> LUInt {
    // lu_int imemsize, xmemsize, fmemsize;

    // if (!obj)
    //     return BASICLU_ERROR_ARGUMENT_MISSING;
    if m < 0 {
        return BASICLU_ERROR_INVALID_ARGUMENT;
    }

    let imemsize = BASICLU_SIZE_ISTORE_1 + BASICLU_SIZE_ISTORE_M * m;
    let xmemsize = BASICLU_SIZE_XSTORE_1 + BASICLU_SIZE_XSTORE_M * m;
    let fmemsize = m; // initial length of l_i, l_x, u_i, u_x, w_i, w_x

    obj.istore = vec![0; imemsize as usize];
    obj.xstore = vec![0.0; xmemsize as usize];
    obj.l_i = vec![0; fmemsize as usize];
    obj.l_x = vec![0.0; fmemsize as usize];
    obj.u_i = vec![0; fmemsize as usize];
    obj.u_x = vec![0.0; fmemsize as usize];
    obj.w_i = vec![0; fmemsize as usize];
    obj.w_x = vec![0.0; fmemsize as usize];
    obj.lhs = vec![0.0; m as usize];
    obj.ilhs = vec![0; m as usize];
    obj.nzlhs = 0;
    obj.realloc_factor = 1.5;

    lu_initialize(m, /*&mut obj.istore,*/ &mut obj.xstore);
    obj.xstore[BASICLU_MEMORYL] = fmemsize as f64;
    obj.xstore[BASICLU_MEMORYU] = fmemsize as f64;
    obj.xstore[BASICLU_MEMORYW] = fmemsize as f64;
    BASICLU_OK
}

/// Purpose:
///
///     Call basiclu_factorize() on a BASICLU object.
///
/// Return:
///
///     BASICLU_ERROR_INVALID_OBJECT
///
///         obj is NULL or initialized to a null object.
///
///     BASICLU_ERROR_OUT_OF_MEMORY
///
///         reallocation failed because of insufficient memory.
///
///     Other return codes are passed through from basiclu_factorize().
///
/// Arguments:
///
///     struct BasicLUObject *obj
///
///         Pointer to an initialized BASICLU object.
///
///     The other arguments are passed through to basiclu_factorize().
pub fn basiclu_obj_factorize(
    obj: &mut BasicLUObject,
    b_begin: &[LUInt],
    b_end: &[LUInt],
    b_i: &[LUInt],
    b_x: &[f64],
) -> LUInt {
    if !isvalid(obj) {
        return BASICLU_ERROR_INVALID_OBJECT;
    }

    let mut status = basiclu_factorize(
        &mut obj.istore,
        &mut obj.xstore,
        &mut obj.l_i,
        &mut obj.l_x,
        &mut obj.u_i,
        &mut obj.u_x,
        &mut obj.w_i,
        &mut obj.w_x,
        b_begin,
        b_end,
        b_i,
        b_x,
        0,
    );

    while status == BASICLU_REALLOCATE {
        status = lu_realloc_obj(obj);
        if status != BASICLU_OK {
            break;
        }
        status = basiclu_factorize(
            &mut obj.istore,
            &mut obj.xstore,
            &mut obj.l_i,
            &mut obj.l_x,
            &mut obj.u_i,
            &mut obj.u_x,
            &mut obj.w_i,
            &mut obj.w_x,
            b_begin,
            b_end,
            b_i,
            b_x,
            1,
        );
    }

    status
}

/// Purpose:
///
///     Call basiclu_get_factors() on a BASICLU object.
///
/// Return:
///
///     BASICLU_ERROR_INVALID_OBJECT
///
///         obj is NULL or initialized to a null object.
///
///     Other return codes are passed through from basiclu_get_factors().
///
/// Arguments:
///
///     struct BasicLUObject *obj
///
///         Pointer to an initialized BASICLU object.
///
///     The other arguments are passed through to basiclu_get_factors().
pub fn basiclu_obj_get_factors(
    obj: &mut BasicLUObject,
    rowperm: Option<&[LUInt]>,
    colperm: Option<&[LUInt]>,
    l_colptr: Option<&mut [LUInt]>,
    l_rowidx: Option<&mut [LUInt]>,
    l_value: Option<&mut [f64]>,
    u_colptr: Option<&mut [LUInt]>,
    u_rowidx: Option<&mut [LUInt]>,
    u_value: Option<&mut [f64]>,
) -> LUInt {
    if !isvalid(obj) {
        return BASICLU_ERROR_INVALID_OBJECT;
    }

    basiclu_get_factors(
        &mut obj.istore,
        &mut obj.xstore,
        &obj.l_i,
        &obj.l_x,
        &obj.w_i,
        &obj.w_x,
        rowperm,
        colperm,
        l_colptr,
        l_rowidx,
        l_value,
        u_colptr,
        u_rowidx,
        u_value,
    )
}

/// Purpose:
///
///     Call basiclu_solve_dense() on a BASICLU object.
///
/// Return:
///
///     BASICLU_ERROR_INVALID_OBJECT
///
///         obj is NULL or initialized to a null object.
///
///     Other return codes are passed through from basiclu_solve_dense().
///
/// Arguments:
///
///     struct BasicLUObject *obj
///
///         Pointer to an initialized BASICLU object.
///
///     The other arguments are passed through to basiclu_solve_dense().
pub fn basiclu_obj_solve_dense(
    obj: &mut BasicLUObject,
    rhs: &[f64],
    lhs: &mut [f64],
    trans: char,
) -> LUInt {
    if !isvalid(obj) {
        return BASICLU_ERROR_INVALID_OBJECT;
    }

    basiclu_solve_dense(
        // &mut obj.istore,
        &mut obj.xstore,
        &obj.l_i,
        &obj.l_x,
        &obj.u_i,
        &obj.u_x,
        &obj.w_i,
        &obj.w_x,
        rhs,
        lhs,
        trans,
    )
}

/// Purpose:
///
///     Call basiclu_solve_sparse() on a BASICLU object. On success, the solution
///     is provided in obj.lhs and the nonzero pattern is stored in
///     obj.ilhs[0..obj.nzlhs-1].
///
/// Return:
///
///     BASICLU_ERROR_INVALID_OBJECT
///
///         obj is NULL or initialized to a null object.
///
///     Other return codes are passed through from basiclu_solve_sparse().
///
/// Arguments:
///
///     struct BasicLUObject *obj
///
///         Pointer to an initialized BASICLU object.
///
///     The other arguments are passed through to basiclu_solve_sparse().
pub fn basiclu_obj_solve_sparse(
    obj: &mut BasicLUObject,
    nzrhs: LUInt,
    irhs: &[LUInt],
    xrhs: &[f64],
    trans: char,
) -> LUInt {
    if !isvalid(obj) {
        return BASICLU_ERROR_INVALID_OBJECT;
    }

    lu_clear_lhs(obj);
    basiclu_solve_sparse(
        &mut obj.istore,
        &mut obj.xstore,
        &obj.l_i,
        &obj.l_x,
        &obj.u_i,
        &obj.u_x,
        &obj.w_i,
        &obj.w_x,
        nzrhs,
        irhs,
        xrhs,
        &mut obj.nzlhs,
        &mut obj.ilhs,
        &mut obj.lhs,
        trans,
    )
}

/// Purpose:
///
///     Call basiclu_solve_for_update() on a BASICLU object. On success, if the
///     solution was requested, it is provided in obj.lhs and the nonzero pattern
///     is stored in obj.ilhs[0..obj.nzlhs-1].
///
/// Return:
///
///     BASICLU_ERROR_INVALID_OBJECT
///
///         obj is NULL or initialized to a null object.
///
///     BASICLU_ERROR_OUT_OF_MEMORY
///
///         reallocation failed because of insufficient memory.
///
///     Other return codes are passed through from basiclu_solve_for_update().
///
/// Arguments:
///
///     struct BasicLUObject *obj
///
///         Pointer to an initialized BASICLU object.
///
///     lu_int want_solution
///
///         Nonzero to compute the solution to the linear system,
///         zero to only prepare the update.
///
///     The other arguments are passed through to basiclu_solve_for_update().
pub fn basiclu_obj_solve_for_update(
    obj: &mut BasicLUObject,
    nzrhs: LUInt,
    irhs: &[LUInt],
    xrhs: Option<&[f64]>,
    trans: char,
    want_solution: LUInt,
) -> LUInt {
    let mut status = BASICLU_OK;

    if !isvalid(obj) {
        return BASICLU_ERROR_INVALID_OBJECT;
    }

    lu_clear_lhs(obj);
    while status == BASICLU_OK {
        let mut nzlhs: LUInt = -1;
        status = basiclu_solve_for_update(
            &mut obj.istore,
            &mut obj.xstore,
            &mut obj.l_i,
            &mut obj.l_x,
            &mut obj.u_i,
            &mut obj.u_x,
            &mut obj.w_i,
            &mut obj.w_x,
            nzrhs,
            irhs,
            xrhs,
            Some(&mut nzlhs),
            Some(&mut obj.ilhs),
            Some(&mut obj.lhs),
            trans,
        );
        if want_solution != 0 {
            obj.nzlhs = nzlhs;
        }
        if status != BASICLU_REALLOCATE {
            break;
        }
        status = lu_realloc_obj(obj);
    }

    status
}

/// Purpose:
///
///     Call basiclu_update() on a BASICLU object.
///
/// Return:
///
///     BASICLU_ERROR_INVALID_OBJECT
///
///         obj is NULL or initialized to a null object.
///
///     BASICLU_ERROR_OUT_OF_MEMORY
///
///         reallocation failed because of insufficient memory.
///
///     Other return codes are passed through from basiclu_update().
///
/// Arguments:
///
///     struct BasicLUObject *obj
///
///         Pointer to an initialized BASICLU object.
///
///     The other arguments are passed through to basiclu_update().
pub fn basiclu_obj_update(obj: &mut BasicLUObject, xtbl: f64) -> LUInt {
    let mut status = BASICLU_OK;

    if !isvalid(obj) {
        return BASICLU_ERROR_INVALID_OBJECT;
    }

    while status == BASICLU_OK {
        status = basiclu_update(
            &mut obj.istore,
            &mut obj.xstore,
            &mut obj.l_i,
            &mut obj.l_x,
            &mut obj.u_i,
            &mut obj.u_x,
            &mut obj.w_i,
            &mut obj.w_x,
            xtbl,
        );
        if status != BASICLU_REALLOCATE {
            break;
        }
        status = lu_realloc_obj(obj);
    }

    status
}

// reallocate two arrays
//
// @nz new number of elements per array
// @a_i location of pointer to integer memory
// @a_x location of pointer to floating point memory
//
// If a reallocation fails, then do not overwrite the old pointer.
//
// Return: BASICLU_OK or BASICLU_ERROR_OUT_OF_MEMORY
fn lu_reallocix(nz: LUInt, a_i: &mut Vec<LUInt>, a_x: &mut Vec<f64>) -> LUInt {
    // lu_int *Ainew;
    // double *Axnew;

    (*a_i).resize(nz as usize, 0);
    // Ainew = realloc(*a_i, nz * sizeof(lu_int));
    // if (Ainew)
    //     *a_i = Ainew;

    (*a_x).resize(nz as usize, 0.0);
    // Axnew = realloc(*a_x, nz * sizeof(double));
    // if (Axnew)
    //     *a_x = Axnew;

    // return Ainew && Axnew ? BASICLU_OK : BASICLU_ERROR_OUT_OF_MEMORY;
    BASICLU_OK
}

// Reallocate l_i,l_x and/or u_i,u_x and/or w_i,w_x as requested in xstore
//
// Return: BASICLU_OK or BASICLU_ERROR_OUT_OF_MEMORY
fn lu_realloc_obj(obj: &mut BasicLUObject) -> LUInt {
    let xstore = &mut obj.xstore;
    let addmem_l = xstore[BASICLU_ADD_MEMORYL];
    let addmem_u = xstore[BASICLU_ADD_MEMORYU];
    let addmem_w = xstore[BASICLU_ADD_MEMORYW];
    let realloc_factor = f64::max(1.0, obj.realloc_factor);
    // lu_int nelem;
    let mut status = BASICLU_OK;

    if status == BASICLU_OK && addmem_l > 0.0 {
        let mut nelem = xstore[BASICLU_MEMORYL] + addmem_l;
        nelem *= realloc_factor;
        status = lu_reallocix(nelem as LUInt, &mut obj.l_i, &mut obj.l_x);
        if status == BASICLU_OK {
            xstore[BASICLU_MEMORYL] = nelem;
        }
    }
    if status == BASICLU_OK && addmem_u > 0.0 {
        let mut nelem = xstore[BASICLU_MEMORYU] + addmem_u;
        nelem *= realloc_factor;
        status = lu_reallocix(nelem as LUInt, &mut obj.u_i, &mut obj.u_x);
        if status == BASICLU_OK {
            xstore[BASICLU_MEMORYU] = nelem;
        }
    }
    if status == BASICLU_OK && addmem_w > 0.0 {
        let mut nelem = xstore[BASICLU_MEMORYW] + addmem_w;
        nelem *= realloc_factor;
        status = lu_reallocix(nelem as LUInt, &mut obj.w_i, &mut obj.w_x);
        if status == BASICLU_OK {
            xstore[BASICLU_MEMORYW] = nelem;
        }
    }
    status
}

// Test if @obj is an allocated BASICLU object
fn isvalid(obj: &BasicLUObject) -> bool {
    !obj.istore.is_empty() && !obj.xstore.is_empty()
}

// reset contents of lhs to zero
fn lu_clear_lhs(obj: &mut BasicLUObject) {
    let m = obj.xstore[BASICLU_DIM];
    let nzsparse = (obj.xstore[BASICLU_SPARSE_THRESHOLD] * m) as LUInt;
    let nz = obj.nzlhs;

    if nz != 0 {
        if nz <= nzsparse {
            for p in 0..nz {
                obj.lhs[obj.ilhs[p as usize] as usize] = 0.0;
            }
        } else {
            // memset(obj.lhs, 0, m * sizeof(double));
            obj.lhs.fill(0.0);
        }
        obj.nzlhs = 0;
    }
}
