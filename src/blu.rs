// Copyright (C) 2016-2018 ERGO-Code
// Copyright (C) 2022-2023 Richard Lincoln

use crate::lu::LU;
use crate::{
    factorize, get_factors, solve_dense, solve_for_update, solve_sparse, update, LUInt, Status,
};

pub struct BLU {
    pub lu: LU,

    /// Holds solution after `solve_sparse()` and `solve_for_update()`.
    pub lhs: Vec<f64>,
    pub ilhs: Vec<LUInt>,
    pub nzlhs: usize,

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
    ///     OK
    ///
    ///         *obj successfully initialized.
    ///
    ///     ErrorArgumentMissing
    ///
    ///         obj is NULL.
    ///
    ///     ErrorInvalidArgument
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
    pub fn new(m: usize, b_nz: usize) -> Self {
        // assert!(m >= 0);
        Self {
            lu: LU::new(m, b_nz),
            lhs: vec![0.0; m],
            ilhs: vec![0; m],
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
        b_begin: &[usize],
        b_end: &[usize],
        b_i: &[usize],
        b_x: &[f64],
    ) -> Status {
        let mut status = factorize(&mut self.lu, b_begin, b_end, b_i, b_x, false);

        while status == Status::Reallocate {
            status = lu_realloc_obj(self);
            if status != Status::OK {
                break;
            }
            status = factorize(&mut self.lu, b_begin, b_end, b_i, b_x, true);
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
        rowperm: Option<&mut [LUInt]>,
        colperm: Option<&mut [LUInt]>,
        l_colptr: Option<&mut [LUInt]>,
        l_rowidx: Option<&mut [LUInt]>,
        l_value: Option<&mut [f64]>,
        u_colptr: Option<&mut [LUInt]>,
        u_rowidx: Option<&mut [LUInt]>,
        u_value: Option<&mut [f64]>,
    ) -> Status {
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
    pub fn solve_dense(&mut self, rhs: &[f64], lhs: &mut [f64], trans: char) -> Status {
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
        irhs: &[usize],
        xrhs: &[f64],
        trans: char,
    ) -> Status {
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
        nzrhs: usize,
        irhs: &[usize],
        xrhs: Option<&[f64]>,
        trans: char,
        want_solution: LUInt,
    ) -> Status {
        let mut status = Status::OK;

        lu_clear_lhs(self);
        while status == Status::OK {
            // let mut nzlhs: LUInt = -1;
            let mut nzlhs: usize = 0;
            status = solve_for_update(
                &mut self.lu,
                nzrhs as LUInt,
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
            if status != Status::Reallocate {
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
    pub fn update(&mut self, xtbl: f64) -> Status {
        let mut status = Status::OK;

        while status == Status::OK {
            status = update(&mut self.lu, xtbl);
            if status != Status::Reallocate {
                break;
            }
            status = lu_realloc_obj(self);
        }

        status
    }
}

// reallocate two arrays
fn lu_reallocix(nz: usize, a_i: &mut Vec<LUInt>, a_x: &mut Vec<f64>) -> Status {
    a_i.resize(nz, 0);
    a_x.resize(nz, 0.0);
    Status::OK
}

// Reallocate l_i,l_x and/or u_i,u_x and/or w_i,w_x as requested in LU.
fn lu_realloc_obj(obj: &mut BLU) -> Status {
    let addmem_l = obj.lu.addmem_l;
    let addmem_u = obj.lu.addmem_u;
    let addmem_w = obj.lu.addmem_w;
    let realloc_factor = f64::max(1.0, obj.realloc_factor);
    let mut status = Status::OK;

    if status == Status::OK && addmem_l > 0 {
        let nelem = obj.lu.l_mem + addmem_l;
        let nelem = ((nelem as f64) * realloc_factor) as usize;
        status = lu_reallocix(nelem, &mut obj.lu.l_index, &mut obj.lu.l_value);
        if status == Status::OK {
            obj.lu.l_mem = nelem;
        }
    }
    if status == Status::OK && addmem_u > 0 {
        let nelem = obj.lu.u_mem + addmem_u;
        let nelem = ((nelem as f64) * realloc_factor) as usize;
        status = lu_reallocix(nelem, &mut obj.lu.u_index, &mut obj.lu.u_value);
        if status == Status::OK {
            obj.lu.u_mem = nelem;
        }
    }
    if status == Status::OK && addmem_w > 0 {
        let nelem = obj.lu.w_mem + addmem_w;
        let nelem = ((nelem as f64) * realloc_factor) as usize;
        status = lu_reallocix(nelem, &mut obj.lu.w_index, &mut obj.lu.w_value);
        if status == Status::OK {
            obj.lu.w_mem = nelem;
        }
    }
    status
}

// reset contents of lhs to zero
fn lu_clear_lhs(obj: &mut BLU) {
    let m = obj.lu.m as f64;
    let nzsparse = (obj.lu.sparse_thres * m as f64) as usize;
    let nz = obj.nzlhs;

    if nz != 0 {
        if nz <= nzsparse {
            for p in 0..nz {
                obj.lhs[obj.ilhs[p as usize] as usize] = 0.0;
            }
        } else {
            obj.lhs.fill(0.0);
        }
        obj.nzlhs = 0;
    }
}
