// Copyright (C) 2016-2018  ERGO-Code

use crate::blu::*;
use crate::lu::LU;
use crate::{factorize, get_factors, solve_dense, solve_for_update, solve_sparse, update};

pub struct BLU {
    pub lu: LU,

    /// Holds solution after `solve_sparse()` and `solve_for_update()`.
    pub lhs: Vec<f64>,
    pub ilhs: Vec<LUInt>,
    pub nzlhs: LUInt,

    /// Arrays are reallocated for max(realloc_factor, 1.0) times the
    /// required size. Default: 1.5
    pub realloc_factor: f64,
}

impl BLU {
    /// Purpose:
    ///
    ///     Initialize a BLU object. When m is positive, then *obj is initialized to
    ///     process matrices of dimension m. When m is zero, then *obj is initialized to
    ///     a "null" object, which cannot be used for factorization, but can be passed
    ///     to blu_obj_free().
    ///
    ///     This routine must be called once before passing obj to any other
    ///     blu_obj_ routine. When obj is initialized to a null object, then the
    ///     routine can be called again to reinitialize obj.
    ///
    /// Return:
    ///
    ///     BLU_OK
    ///
    ///         *obj successfully initialized.
    ///
    ///     BLU_ERROR_ARGUMENT_MISSING
    ///
    ///         obj is NULL.
    ///
    ///     BLU_ERROR_INVALID_ARGUMENT
    ///
    ///         m is negative.
    ///
    ///     BLU_ERROR_OUT_OF_MEMORY
    ///
    ///         insufficient memory to initialize object.
    ///
    /// Arguments:
    ///
    ///     struct BLU *obj
    ///
    ///         Pointer to the object to be initialized.
    ///
    ///     lu_int m
    ///
    ///         The dimension of matrices which can be processed, or 0.
    pub fn new(m: LUInt) -> Self {
        // lu_int imemsize, xmemsize, fmemsize;

        // if (!obj)
        //     return BLU_ERROR_ARGUMENT_MISSING;
        // if m < 0 {
        //     return BLU_ERROR_INVALID_ARGUMENT;
        // }
        assert!(m >= 0);

        // let imemsize = BLU_SIZE_ISTORE_1 + BLU_SIZE_ISTORE_M * m;
        // let xmemsize = BLU_SIZE_XSTORE_1 + BLU_SIZE_XSTORE_M * m;
        // let fmemsize = m; // initial length of l_i, l_x, u_i, u_x, w_i, w_x

        // obj.istore = vec![0; imemsize as usize];
        // obj.xstore = vec![0.0; xmemsize as usize];
        // obj.l_i = vec![0; fmemsize as usize];
        // obj.l_x = vec![0.0; fmemsize as usize];
        // obj.u_i = vec![0; fmemsize as usize];
        // obj.u_x = vec![0.0; fmemsize as usize];
        // obj.w_i = vec![0; fmemsize as usize];
        // obj.w_x = vec![0.0; fmemsize as usize];
        // obj.lhs = vec![0.0; m as usize];
        // obj.ilhs = vec![0; m as usize];
        // obj.nzlhs = 0;
        // obj.realloc_factor = 1.5;

        // lu_initialize(m, /*&mut obj.istore,*/ &mut obj.xstore);
        // obj.xstore[BLU_MEMORYL] = fmemsize as f64;
        // obj.xstore[BLU_MEMORYU] = fmemsize as f64;
        // obj.xstore[BLU_MEMORYW] = fmemsize as f64;
        // BLU_OK

        Self {
            lu: LU::new(m),
            lhs: vec![0.0; m as usize],
            ilhs: vec![0; m as usize],
            nzlhs: 0,
            realloc_factor: 1.5,
        }
    }

    /// Purpose:
    ///
    ///     Call factorize() on a BLU object.
    ///
    /// Return:
    ///
    ///     BLU_ERROR_INVALID_OBJECT
    ///
    ///         obj is NULL or initialized to a null object.
    ///
    ///     BLU_ERROR_OUT_OF_MEMORY
    ///
    ///         reallocation failed because of insufficient memory.
    ///
    ///     Other return codes are passed through from factorize().
    ///
    /// Arguments:
    ///
    ///     struct BLU *obj
    ///
    ///         Pointer to an initialized BLU object.
    ///
    ///     The other arguments are passed through to factorize().
    pub fn factorize(
        &mut self,
        b_begin: &[LUInt],
        b_end: &[LUInt],
        b_i: &[LUInt],
        b_x: &[f64],
    ) -> LUInt {
        if !isvalid(self) {
            return BLU_ERROR_INVALID_OBJECT;
        }

        let mut status = factorize(&mut self.lu, b_begin, b_end, b_i, b_x, 0);

        while status == BLU_REALLOCATE {
            status = lu_realloc_obj(self);
            if status != BLU_OK {
                break;
            }
            status = factorize(&mut self.lu, b_begin, b_end, b_i, b_x, 1);
        }

        status
    }

    /// Purpose:
    ///
    ///     Call get_factors() on a BLU object.
    ///
    /// Return:
    ///
    ///     BLU_ERROR_INVALID_OBJECT
    ///
    ///         obj is NULL or initialized to a null object.
    ///
    ///     Other return codes are passed through from get_factors().
    ///
    /// Arguments:
    ///
    ///     struct BLU *obj
    ///
    ///         Pointer to an initialized BLU object.
    ///
    ///     The other arguments are passed through to get_factors().
    pub fn get_factors(
        &mut self,
        rowperm: Option<&[LUInt]>,
        colperm: Option<&[LUInt]>,
        l_colptr: Option<&mut [LUInt]>,
        l_rowidx: Option<&mut [LUInt]>,
        l_value: Option<&mut [f64]>,
        u_colptr: Option<&mut [LUInt]>,
        u_rowidx: Option<&mut [LUInt]>,
        u_value: Option<&mut [f64]>,
    ) -> LUInt {
        if !isvalid(self) {
            return BLU_ERROR_INVALID_OBJECT;
        }

        get_factors(
            &mut self.lu,
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
    ///     Call solve_dense() on a BLU object.
    ///
    /// Return:
    ///
    ///     BLU_ERROR_INVALID_OBJECT
    ///
    ///         obj is NULL or initialized to a null object.
    ///
    ///     Other return codes are passed through from solve_dense().
    ///
    /// Arguments:
    ///
    ///     struct BLU *obj
    ///
    ///         Pointer to an initialized BLU object.
    ///
    ///     The other arguments are passed through to solve_dense().
    pub fn solve_dense(&mut self, rhs: &[f64], lhs: &mut [f64], trans: char) -> LUInt {
        if !isvalid(self) {
            return BLU_ERROR_INVALID_OBJECT;
        }

        solve_dense(&mut self.lu, rhs, lhs, trans)
    }

    /// Purpose:
    ///
    ///     Call solve_sparse() on a BLU object. On success, the solution
    ///     is provided in obj.lhs and the nonzero pattern is stored in
    ///     obj.ilhs[0..obj.nzlhs-1].
    ///
    /// Return:
    ///
    ///     BLU_ERROR_INVALID_OBJECT
    ///
    ///         obj is NULL or initialized to a null object.
    ///
    ///     Other return codes are passed through from solve_sparse().
    ///
    /// Arguments:
    ///
    ///     struct BLU *obj
    ///
    ///         Pointer to an initialized BLU object.
    ///
    ///     The other arguments are passed through to solve_sparse().
    pub fn solve_sparse(
        &mut self,
        nzrhs: LUInt,
        irhs: &[LUInt],
        xrhs: &[f64],
        trans: char,
    ) -> LUInt {
        if !isvalid(self) {
            return BLU_ERROR_INVALID_OBJECT;
        }

        lu_clear_lhs(self);
        solve_sparse(
            &mut self.lu,
            nzrhs,
            irhs,
            xrhs,
            &mut self.nzlhs,
            &mut self.ilhs,
            &mut self.lhs,
            trans,
        )
    }

    /// Purpose:
    ///
    ///     Call solve_for_update() on a BLU object. On success, if the
    ///     solution was requested, it is provided in obj.lhs and the nonzero pattern
    ///     is stored in obj.ilhs[0..obj.nzlhs-1].
    ///
    /// Return:
    ///
    ///     BLU_ERROR_INVALID_OBJECT
    ///
    ///         obj is NULL or initialized to a null object.
    ///
    ///     BLU_ERROR_OUT_OF_MEMORY
    ///
    ///         reallocation failed because of insufficient memory.
    ///
    ///     Other return codes are passed through from solve_for_update().
    ///
    /// Arguments:
    ///
    ///     struct BLU *obj
    ///
    ///         Pointer to an initialized BLU object.
    ///
    ///     lu_int want_solution
    ///
    ///         Nonzero to compute the solution to the linear system,
    ///         zero to only prepare the update.
    ///
    ///     The other arguments are passed through to solve_for_update().
    pub fn solve_for_update(
        &mut self,
        nzrhs: LUInt,
        irhs: &[LUInt],
        xrhs: Option<&[f64]>,
        trans: char,
        want_solution: LUInt,
    ) -> LUInt {
        let mut status = BLU_OK;

        if !isvalid(self) {
            return BLU_ERROR_INVALID_OBJECT;
        }

        lu_clear_lhs(self);
        while status == BLU_OK {
            let mut nzlhs: LUInt = -1;
            status = solve_for_update(
                &mut self.lu,
                nzrhs,
                irhs,
                xrhs,
                Some(&mut nzlhs),
                Some(&mut self.ilhs),
                Some(&mut self.lhs),
                trans,
            );
            if want_solution != 0 {
                self.nzlhs = nzlhs;
            }
            if status != BLU_REALLOCATE {
                break;
            }
            status = lu_realloc_obj(self);
        }

        status
    }

    /// Purpose:
    ///
    ///     Call update() on a BLU object.
    ///
    /// Return:
    ///
    ///     BLU_ERROR_INVALID_OBJECT
    ///
    ///         obj is NULL or initialized to a null object.
    ///
    ///     BLU_ERROR_OUT_OF_MEMORY
    ///
    ///         reallocation failed because of insufficient memory.
    ///
    ///     Other return codes are passed through from update().
    ///
    /// Arguments:
    ///
    ///     struct BLU *obj
    ///
    ///         Pointer to an initialized BLU object.
    ///
    ///     The other arguments are passed through to update().
    pub fn update(&mut self, xtbl: f64) -> LUInt {
        let mut status = BLU_OK;

        if !isvalid(self) {
            return BLU_ERROR_INVALID_OBJECT;
        }

        while status == BLU_OK {
            status = update(&mut self.lu, xtbl);
            if status != BLU_REALLOCATE {
                break;
            }
            status = lu_realloc_obj(self);
        }

        status
    }
}

// reallocate two arrays
//
// @nz new number of elements per array
// @a_i location of pointer to integer memory
// @a_x location of pointer to floating point memory
//
// If a reallocation fails, then do not overwrite the old pointer.
//
// Return: BLU_OK or BLU_ERROR_OUT_OF_MEMORY
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

    // return Ainew && Axnew ? BLU_OK : BLU_ERROR_OUT_OF_MEMORY;
    BLU_OK
}

// Reallocate l_i,l_x and/or u_i,u_x and/or w_i,w_x as requested in xstore
//
// Return: BLU_OK or BLU_ERROR_OUT_OF_MEMORY
fn lu_realloc_obj(obj: &mut BLU) -> LUInt {
    // let xstore = &mut obj.xstore;
    // let addmem_l = xstore[BLU_ADD_MEMORYL];
    // let addmem_u = xstore[BLU_ADD_MEMORYU];
    // let addmem_w = xstore[BLU_ADD_MEMORYW];
    let addmem_l = obj.lu.addmem_l;
    let addmem_u = obj.lu.addmem_u;
    let addmem_w = obj.lu.addmem_w;
    let realloc_factor = f64::max(1.0, obj.realloc_factor);
    // lu_int nelem;
    let mut status = BLU_OK;

    if status == BLU_OK && addmem_l > 0 {
        let mut nelem = obj.lu.l_mem + addmem_l;
        nelem = ((nelem as f64) * realloc_factor) as LUInt;
        status = lu_reallocix(nelem as LUInt, &mut obj.lu.l_index, &mut obj.lu.l_value);
        if status == BLU_OK {
            // xstore[BLU_MEMORYL] = nelem;
            obj.lu.l_mem = nelem;
        }
    }
    if status == BLU_OK && addmem_u > 0 {
        let mut nelem = obj.lu.u_mem + addmem_u;
        nelem = ((nelem as f64) * realloc_factor) as LUInt;
        status = lu_reallocix(nelem as LUInt, &mut obj.lu.u_index, &mut obj.lu.u_value);
        if status == BLU_OK {
            // xstore[BLU_MEMORYU] = nelem;
            obj.lu.u_mem = nelem;
        }
    }
    if status == BLU_OK && addmem_w > 0 {
        let mut nelem = obj.lu.w_mem + addmem_w;
        nelem = ((nelem as f64) * realloc_factor) as LUInt;
        status = lu_reallocix(nelem as LUInt, &mut obj.lu.w_index, &mut obj.lu.w_value);
        if status == BLU_OK {
            // xstore[BLU_MEMORYW] = nelem;
            obj.lu.w_mem = nelem;
        }
    }
    status
}

// Test if @obj is an allocated BLU object
fn isvalid(obj: &BLU) -> bool {
    // !obj.istore.is_empty() && !obj.xstore.is_empty()
    true
}

// reset contents of lhs to zero
fn lu_clear_lhs(obj: &mut BLU) {
    let m = obj.lu.m as f64;
    let nzsparse = (obj.lu.sparse_thres * m) as LUInt;
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
