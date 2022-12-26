use crate::blu::{LUInt, BLU_ERROR_INVALID_ARGUMENT, BLU_OK};

/// Purpose:
///
///     Initialize istore, xstore to a BLU instance. Set parameters to defaults
///     and reset counters. The initialization fixes the dimension of matrices
///     which can be processed by this instance.
///
///     This routine must be called once before passing istore, xstore to any other
///     blu_ routine.
///
/// Return:
///
///     BLU_OK
///
///         m, istore, xstore were valid arguments. Only in this case are istore,
///         xstore initialized.
///
///     BLU_ERROR_ARGUMENT_MISSING
///
///         istore or xstore is NULL.
///
///     BLU_ERROR_INVALID_ARGUMENT
///
///         m is less than or equal to zero.
///
/// Arguments:
///
///     lu_int m
///
///         The dimension of matrices which can be processed. m > 0.
///
///     lu_int istore[]
///     double xstore[]
///
///         Fixed size arrays. These must be allocated by the user as follows:
///
///           length of istore: BLU_SIZE_ISTORE_1 + BLU_SIZE_ISTORE_M * m
///           length of xstore: BLU_SIZE_XSTORE_1 + BLU_SIZE_XSTORE_M * m
pub fn initialize(m: LUInt, /*istore: &mut [lu_int],*/ xstore: &mut [f64]) -> LUInt {
    if m <= 0 {
        return BLU_ERROR_INVALID_ARGUMENT;
    }
    // lu_initialize(m, /*istore,*/ xstore);
    BLU_OK
}
