// Copyright (C) 2016-2018  ERGO-Code

use crate::basiclu::*;
use crate::lu_initialize::lu_initialize;
use crate::{
    basiclu_factorize, basiclu_get_factors, basiclu_solve_dense, basiclu_solve_for_update,
    basiclu_solve_sparse, basiclu_update,
};

/// A variable of type struct basiclu_object must be defined in user code. Its
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
pub struct basiclu_object {
    pub istore: Vec<lu_int>,
    pub xstore: Vec<f64>,

    Li: Vec<lu_int>,
    Ui: Vec<lu_int>,
    Wi: Vec<lu_int>,
    Lx: Vec<f64>,
    Ux: Vec<f64>,
    Wx: Vec<f64>,

    pub lhs: Vec<f64>,
    pub ilhs: Vec<lu_int>,
    pub nzlhs: lu_int,

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
///     BASICLU_ERROR_argument_missing
///
///         obj is NULL.
///
///     BASICLU_ERROR_invalid_argument
///
///         m is negative.
///
///     BASICLU_ERROR_out_of_memory
///
///         insufficient memory to initialize object.
///
/// Arguments:
///
///     struct basiclu_object *obj
///
///         Pointer to the object to be initialized.
///
///     lu_int m
///
///         The dimension of matrices which can be processed, or 0.
pub fn basiclu_obj_initialize(obj: &mut basiclu_object, m: lu_int) -> lu_int {
    // lu_int imemsize, xmemsize, fmemsize;

    // if (!obj)
    //     return BASICLU_ERROR_argument_missing;
    if m < 0 {
        return BASICLU_ERROR_invalid_argument;
    }

    let imemsize = BASICLU_SIZE_ISTORE_1 + BASICLU_SIZE_ISTORE_M * m;
    let xmemsize = BASICLU_SIZE_XSTORE_1 + BASICLU_SIZE_XSTORE_M * m;
    let fmemsize = m; // initial length of Li, Lx, Ui, Ux, Wi, Wx

    obj.istore = vec![0; imemsize as usize];
    obj.xstore = vec![0.0; xmemsize as usize];
    obj.Li = vec![0; fmemsize as usize];
    obj.Lx = vec![0.0; fmemsize as usize];
    obj.Ui = vec![0; fmemsize as usize];
    obj.Ux = vec![0.0; fmemsize as usize];
    obj.Wi = vec![0; fmemsize as usize];
    obj.Wx = vec![0.0; fmemsize as usize];
    obj.lhs = vec![0.0; m as usize];
    obj.ilhs = vec![0; m as usize];
    obj.nzlhs = 0;
    obj.realloc_factor = 1.5;

    lu_initialize(m, &mut obj.istore, &mut obj.xstore);
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
///     BASICLU_ERROR_invalid_object
///
///         obj is NULL or initialized to a null object.
///
///     BASICLU_ERROR_out_of_memory
///
///         reallocation failed because of insufficient memory.
///
///     Other return codes are passed through from basiclu_factorize().
///
/// Arguments:
///
///     struct basiclu_object *obj
///
///         Pointer to an initialized BASICLU object.
///
///     The other arguments are passed through to basiclu_factorize().
pub fn basiclu_obj_factorize(
    obj: &mut basiclu_object,
    Bbegin: &[lu_int],
    Bend: &[lu_int],
    Bi: &[lu_int],
    Bx: &[f64],
) -> lu_int {
    if !isvalid(obj) {
        return BASICLU_ERROR_invalid_object;
    }

    let mut status = basiclu_factorize(
        &mut obj.istore,
        &mut obj.xstore,
        &obj.Li,
        &obj.Lx,
        &obj.Ui,
        &obj.Ux,
        &obj.Wi,
        &obj.Wx,
        Bbegin,
        Bend,
        Bi,
        Bx,
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
            &obj.Li,
            &obj.Lx,
            &obj.Ui,
            &obj.Ux,
            &obj.Wi,
            &obj.Wx,
            Bbegin,
            Bend,
            Bi,
            Bx,
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
///     BASICLU_ERROR_invalid_object
///
///         obj is NULL or initialized to a null object.
///
///     Other return codes are passed through from basiclu_get_factors().
///
/// Arguments:
///
///     struct basiclu_object *obj
///
///         Pointer to an initialized BASICLU object.
///
///     The other arguments are passed through to basiclu_get_factors().
pub fn basiclu_obj_get_factors(
    obj: &mut basiclu_object,
    rowperm: Option<&[lu_int]>,
    colperm: Option<&[lu_int]>,
    Lcolptr: Option<&mut [lu_int]>,
    Lrowidx: Option<&mut [lu_int]>,
    Lvalue: Option<&mut [f64]>,
    Ucolptr: Option<&mut [lu_int]>,
    Urowidx: Option<&mut [lu_int]>,
    Uvalue: Option<&mut [f64]>,
) -> lu_int {
    if !isvalid(obj) {
        return BASICLU_ERROR_invalid_object;
    }

    basiclu_get_factors(
        &mut obj.istore,
        &mut obj.xstore,
        Some(&obj.Li),
        Some(&obj.Lx),
        Some(&obj.Ui),
        Some(&obj.Ux),
        Some(&obj.Wi),
        Some(&obj.Wx),
        rowperm,
        colperm,
        Lcolptr,
        Lrowidx,
        Lvalue,
        Ucolptr,
        Urowidx,
        Uvalue,
    )
}

/// Purpose:
///
///     Call basiclu_solve_dense() on a BASICLU object.
///
/// Return:
///
///     BASICLU_ERROR_invalid_object
///
///         obj is NULL or initialized to a null object.
///
///     Other return codes are passed through from basiclu_solve_dense().
///
/// Arguments:
///
///     struct basiclu_object *obj
///
///         Pointer to an initialized BASICLU object.
///
///     The other arguments are passed through to basiclu_solve_dense().
pub fn basiclu_obj_solve_dense(
    obj: &mut basiclu_object,
    rhs: &[f64],
    lhs: &mut [f64],
    trans: char,
) -> lu_int {
    if !isvalid(obj) {
        return BASICLU_ERROR_invalid_object;
    }

    basiclu_solve_dense(
        &mut obj.istore,
        &mut obj.xstore,
        &obj.Li,
        &obj.Lx,
        &obj.Ui,
        &obj.Ux,
        &obj.Wi,
        &obj.Wx,
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
///     BASICLU_ERROR_invalid_object
///
///         obj is NULL or initialized to a null object.
///
///     Other return codes are passed through from basiclu_solve_sparse().
///
/// Arguments:
///
///     struct basiclu_object *obj
///
///         Pointer to an initialized BASICLU object.
///
///     The other arguments are passed through to basiclu_solve_sparse().
pub fn basiclu_obj_solve_sparse(
    obj: &mut basiclu_object,
    nzrhs: lu_int,
    irhs: &[lu_int],
    xrhs: &[f64],
    trans: char,
) -> lu_int {
    if !isvalid(obj) {
        return BASICLU_ERROR_invalid_object;
    }

    lu_clear_lhs(obj);
    basiclu_solve_sparse(
        &mut obj.istore,
        &mut obj.xstore,
        &obj.Li,
        &obj.Lx,
        &obj.Ui,
        &obj.Ux,
        &obj.Wi,
        &obj.Wx,
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
///     BASICLU_ERROR_invalid_object
///
///         obj is NULL or initialized to a null object.
///
///     BASICLU_ERROR_out_of_memory
///
///         reallocation failed because of insufficient memory.
///
///     Other return codes are passed through from basiclu_solve_for_update().
///
/// Arguments:
///
///     struct basiclu_object *obj
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
    obj: &mut basiclu_object,
    nzrhs: lu_int,
    irhs: &[lu_int],
    xrhs: Option<&[f64]>,
    trans: char,
    want_solution: lu_int,
) -> lu_int {
    let mut status = BASICLU_OK;

    if !isvalid(obj) {
        return BASICLU_ERROR_invalid_object;
    }

    lu_clear_lhs(obj);
    while status == BASICLU_OK {
        let mut nzlhs: lu_int = -1;
        status = basiclu_solve_for_update(
            &mut obj.istore,
            &mut obj.xstore,
            &obj.Li,
            &obj.Lx,
            &obj.Ui,
            &obj.Ux,
            &obj.Wi,
            &obj.Wx,
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
///     BASICLU_ERROR_invalid_object
///
///         obj is NULL or initialized to a null object.
///
///     BASICLU_ERROR_out_of_memory
///
///         reallocation failed because of insufficient memory.
///
///     Other return codes are passed through from basiclu_update().
///
/// Arguments:
///
///     struct basiclu_object *obj
///
///         Pointer to an initialized BASICLU object.
///
///     The other arguments are passed through to basiclu_update().
pub fn basiclu_obj_update(obj: &mut basiclu_object, xtbl: f64) -> lu_int {
    let mut status = BASICLU_OK;

    if !isvalid(obj) {
        return BASICLU_ERROR_invalid_object;
    }

    while status == BASICLU_OK {
        status = basiclu_update(
            &mut obj.istore,
            &mut obj.xstore,
            &obj.Li,
            &obj.Lx,
            &obj.Ui,
            &obj.Ux,
            &obj.Wi,
            &obj.Wx,
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
// @p_Ai location of pointer to integer memory
// @p_Ax location of pointer to floating point memory
//
// If a reallocation fails, then do not overwrite the old pointer.
//
// Return: BASICLU_OK or BASICLU_ERROR_out_of_memory
fn lu_reallocix(nz: lu_int, p_Ai: &mut Vec<lu_int>, p_Ax: &mut Vec<f64>) -> lu_int {
    // lu_int *Ainew;
    // double *Axnew;

    (*p_Ai).resize(nz as usize, 0);
    // Ainew = realloc(*p_Ai, nz * sizeof(lu_int));
    // if (Ainew)
    //     *p_Ai = Ainew;

    (*p_Ax).resize(nz as usize, 0.0);
    // Axnew = realloc(*p_Ax, nz * sizeof(double));
    // if (Axnew)
    //     *p_Ax = Axnew;

    // return Ainew && Axnew ? BASICLU_OK : BASICLU_ERROR_out_of_memory;
    BASICLU_OK
}

// Reallocate Li,Lx and/or Ui,Ux and/or Wi,Wx as requested in xstore
//
// Return: BASICLU_OK or BASICLU_ERROR_out_of_memory
fn lu_realloc_obj(obj: &mut basiclu_object) -> lu_int {
    let xstore = &mut obj.xstore;
    let addmemL = xstore[BASICLU_ADD_MEMORYL];
    let addmemU = xstore[BASICLU_ADD_MEMORYU];
    let addmemW = xstore[BASICLU_ADD_MEMORYW];
    let realloc_factor = f64::max(1.0, obj.realloc_factor);
    // lu_int nelem;
    let mut status = BASICLU_OK;

    if status == BASICLU_OK && addmemL > 0.0 {
        let mut nelem = xstore[BASICLU_MEMORYL] + addmemL;
        nelem *= realloc_factor;
        status = lu_reallocix(nelem as lu_int, &mut obj.Li, &mut obj.Lx);
        if status == BASICLU_OK {
            xstore[BASICLU_MEMORYL] = nelem;
        }
    }
    if status == BASICLU_OK && addmemU > 0.0 {
        let mut nelem = xstore[BASICLU_MEMORYU] + addmemU;
        nelem *= realloc_factor;
        status = lu_reallocix(nelem as lu_int, &mut obj.Ui, &mut obj.Ux);
        if status == BASICLU_OK {
            xstore[BASICLU_MEMORYU] = nelem;
        }
    }
    if status == BASICLU_OK && addmemW > 0.0 {
        let mut nelem = xstore[BASICLU_MEMORYW] + addmemW;
        nelem *= realloc_factor;
        status = lu_reallocix(nelem as lu_int, &mut obj.Wi, &mut obj.Wx);
        if status == BASICLU_OK {
            xstore[BASICLU_MEMORYW] = nelem;
        }
    }
    status
}

// Test if @obj is an allocated BASICLU object
fn isvalid(obj: &basiclu_object) -> bool {
    !obj.istore.is_empty() && !obj.xstore.is_empty()
}

// reset contents of lhs to zero
fn lu_clear_lhs(obj: &mut basiclu_object) {
    let m = obj.xstore[BASICLU_DIM];
    let nzsparse = (obj.xstore[BASICLU_SPARSE_THRESHOLD] * m) as lu_int;
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
