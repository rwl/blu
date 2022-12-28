mod blu;
mod factorize;
mod get_factors;
mod lu;
mod maxvolume;
mod solve_dense;
mod solve_for_update;
mod solve_sparse;
mod update;

pub use blu::BLU;
pub use factorize::factorize;
pub use get_factors::get_factors;
pub use maxvolume::maxvolume;
pub use solve_dense::solve_dense;
pub use solve_for_update::solve_for_update;
pub use solve_sparse::solve_sparse;
pub use update::update;

/// BLU integer type
///
/// All integers in BLU code are of type lu_int, which must be a signed
/// integer type. It is required that all integer values arising in the
/// computation can be stored in double variables and converted back to lu_int
/// without altering their value.
///
/// LU_INT_MAX must be the maximum value of a variable of type lu_int.
///
/// The default is 64 bit integers to make the code easily callable from Julia.
/// int64_t is optional in the C99 standard, but available on most systems.
pub type LUInt = i64;

pub const LU_INT_MAX: i64 = i64::MAX;

pub type IntLeast64 = i64;

#[derive(PartialEq, Debug)]
pub enum Status {
    OK,

    /// Insufficient memory in `w_i`, `w_x`. The number of additional elements
    /// required is given by `addmem_w`.
    ///
    /// The user must reallocate `w_i`, `w_x`. It is recommended to reallocate for
    /// the requested number of additional elements plus some extra space
    /// for further updates (e.g. 0.5 times the current array length). The
    /// new array length must be provided in `w_mem`.
    Reallocate,

    /// The factorization did [`rank`] < [`m`] pivot steps. The remaining elements
    /// in the active submatrix are zero or less than [`abstol`]. The factors have
    /// been augmented by unit columns to form a square matrix. See
    /// [`blu_get_factors()`] on how to get the indices of linearly dependent
    /// columns.
    WarningSingularMatrix,

    ErrorInvalidCall,
    ErrorArgumentMissing,
    ErrorInvalidArgument,
    ErrorMaximumUpdates,

    /// The updated factorization would be (numerically) singular. No update
    /// has been computed and the old factorization is still valid.
    ErrorSingularUpdate,
}
