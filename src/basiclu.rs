/// BASICLU integer type
///
/// All integers in BASICLU code are of type lu_int, which must be a signed
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

// size of istore
pub const BASICLU_SIZE_ISTORE_1: LUInt = 1024;
pub const BASICLU_SIZE_ISTORE_M: LUInt = 21;

// size of xstore
pub const BASICLU_SIZE_XSTORE_1: LUInt = 1024;
pub const BASICLU_SIZE_XSTORE_M: LUInt = 4;

// status codes //

pub const BASICLU_OK: LUInt = 0;
/// Insufficient memory in `w_i`, `w_x`. The number of additional elements
/// required is given by `addmem_w`.
///
/// The user must reallocate `w_i`, `w_x`. It is recommended to reallocate for
/// the requested number of additional elements plus some extra space
/// for further updates (e.g. 0.5 times the current array length). The
/// new array length must be provided in `w_mem`.
pub const BASICLU_REALLOCATE: LUInt = 1;
/// The factorization did [`rank`] < [`m`] pivot steps. The remaining elements
/// in the active submatrix are zero or less than [`abstol`]. The factors have
/// been augmented by unit columns to form a square matrix. See
/// [`basiclu_get_factors()`] on how to get the indices of linearly dependent
/// columns.
pub const BASICLU_WARNING_SINGULAR_MATRIX: LUInt = 2;
pub const BASICLU_ERROR_INVALID_STORE: LUInt = -1;
pub const BASICLU_ERROR_INVALID_CALL: LUInt = -2;
pub const BASICLU_ERROR_ARGUMENT_MISSING: LUInt = -3;
pub const BASICLU_ERROR_INVALID_ARGUMENT: LUInt = -4;
pub const BASICLU_ERROR_MAXIMUM_UPDATES: LUInt = -5;
/// The updated factorization would be (numerically) singular. No update
/// has been computed and the old factorization is still valid.
pub const BASICLU_ERROR_SINGULAR_UPDATE: LUInt = -6;

pub const BASICLU_ERROR_INVALID_OBJECT: LUInt = -8;
pub const BASICLU_ERROR_OUT_OF_MEMORY: LUInt = -9;

// public entries in xstore //

// user parameters
pub const BASICLU_MEMORYL: usize = 1;
pub const BASICLU_MEMORYU: usize = 2;
pub const BASICLU_MEMORYW: usize = 3;
pub const BASICLU_DROP_TOLERANCE: usize = 4;
pub const BASICLU_ABS_PIVOT_TOLERANCE: usize = 5;
pub const BASICLU_REL_PIVOT_TOLERANCE: usize = 6;
pub const BASICLU_BIAS_NONZEROS: usize = 7;
pub const BASICLU_MAXN_SEARCH_PIVOT: usize = 8;
pub const BASICLU_PAD: usize = 9;
pub const BASICLU_STRETCH: usize = 10;
pub const BASICLU_COMPRESSION_THRESHOLD: usize = 11;
pub const BASICLU_SPARSE_THRESHOLD: usize = 12;
pub const BASICLU_REMOVE_COLUMNS: usize = 13;
pub const BASICLU_SEARCH_ROWS: usize = 14;

// user readable
pub const BASICLU_DIM: usize = 64;
pub const BASICLU_STATUS: usize = 65;
pub const BASICLU_ADD_MEMORYL: usize = 66;
pub const BASICLU_ADD_MEMORYU: usize = 67;
pub const BASICLU_ADD_MEMORYW: usize = 68;

pub const BASICLU_NUPDATE: usize = 70;
pub const BASICLU_NFORREST: usize = 71;
pub const BASICLU_NFACTORIZE: usize = 72;
pub const BASICLU_NUPDATE_TOTAL: usize = 73;
pub const BASICLU_NFORREST_TOTAL: usize = 74;
pub const BASICLU_NSYMPERM_TOTAL: usize = 75;
pub const BASICLU_LNZ: usize = 76;
pub const BASICLU_UNZ: usize = 77;
pub const BASICLU_RNZ: usize = 78;
pub const BASICLU_MIN_PIVOT: usize = 79;
pub const BASICLU_MAX_PIVOT: usize = 80;
pub const BASICLU_UPDATE_COST: usize = 81;
pub const BASICLU_TIME_FACTORIZE: usize = 82;
pub const BASICLU_TIME_SOLVE: usize = 83;
pub const BASICLU_TIME_UPDATE: usize = 84;
pub const BASICLU_TIME_FACTORIZE_TOTAL: usize = 85;
pub const BASICLU_TIME_SOLVE_TOTAL: usize = 86;
pub const BASICLU_TIME_UPDATE_TOTAL: usize = 87;
pub const BASICLU_LFLOPS: usize = 88;
pub const BASICLU_UFLOPS: usize = 89;
pub const BASICLU_RFLOPS: usize = 90;
pub const BASICLU_CONDEST_L: usize = 91;
pub const BASICLU_CONDEST_U: usize = 92;
pub const BASICLU_MAX_ETA: usize = 93;
pub const BASICLU_NORM_L: usize = 94;
pub const BASICLU_NORM_U: usize = 95;
pub const BASICLU_NORMEST_LINV: usize = 96;
pub const BASICLU_NORMEST_UINV: usize = 97;
pub const BASICLU_MATRIX_ONENORM: usize = 98;
pub const BASICLU_MATRIX_INFNORM: usize = 99;
pub const BASICLU_RESIDUAL_TEST: usize = 111;

pub const BASICLU_MATRIX_NZ: usize = 100;
pub const BASICLU_RANK: usize = 101;
pub const BASICLU_BUMP_SIZE: usize = 102;
pub const BASICLU_BUMP_NZ: usize = 103;
pub const BASICLU_NSEARCH_PIVOT: usize = 104;
pub const BASICLU_NEXPAND: usize = 105;
pub const BASICLU_NGARBAGE: usize = 106;
pub const BASICLU_FACTOR_FLOPS: usize = 107;
pub const BASICLU_TIME_SINGLETONS: usize = 108;
pub const BASICLU_TIME_SEARCH_PIVOT: usize = 109;
pub const BASICLU_TIME_ELIM_PIVOT: usize = 110;

pub const BASICLU_PIVOT_ERROR: usize = 120;
