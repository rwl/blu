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
pub type lu_int = i64;
pub const LU_INT_MAX: i64 = i64::MAX;

pub type int_least64_t = i64;

// size of istore
pub const BASICLU_SIZE_ISTORE_1: lu_int = 1024;
pub const BASICLU_SIZE_ISTORE_M: lu_int = 21;

// size of xstore
pub const BASICLU_SIZE_XSTORE_1: lu_int = 1024;
pub const BASICLU_SIZE_XSTORE_M: lu_int = 4;

// status codes //

pub const BASICLU_OK: lu_int = 0;
pub const BASICLU_REALLOCATE: lu_int = 1;
pub const BASICLU_WARNING_singular_matrix: lu_int = 2;
pub const BASICLU_ERROR_invalid_store: lu_int = -1;
pub const BASICLU_ERROR_invalid_call: lu_int = -2;
pub const BASICLU_ERROR_argument_missing: lu_int = -3;
pub const BASICLU_ERROR_invalid_argument: lu_int = -4;
pub const BASICLU_ERROR_maximum_updates: lu_int = -5;
pub const BASICLU_ERROR_singular_update: lu_int = -6;

pub const BASICLU_ERROR_invalid_object: lu_int = -8;
pub const BASICLU_ERROR_out_of_memory: lu_int = -9;

// public entries in xstore //

// user parameters
pub const BASICLU_MEMORYL: lu_int = 1;
pub const BASICLU_MEMORYU: lu_int = 2;
pub const BASICLU_MEMORYW: lu_int = 3;
pub const BASICLU_DROP_TOLERANCE: lu_int = 4;
pub const BASICLU_ABS_PIVOT_TOLERANCE: lu_int = 5;
pub const BASICLU_REL_PIVOT_TOLERANCE: lu_int = 6;
pub const BASICLU_BIAS_NONZEROS: lu_int = 7;
pub const BASICLU_MAXN_SEARCH_PIVOT: lu_int = 8;
pub const BASICLU_PAD: lu_int = 9;
pub const BASICLU_STRETCH: lu_int = 10;
pub const BASICLU_COMPRESSION_THRESHOLD: lu_int = 11;
pub const BASICLU_SPARSE_THRESHOLD: lu_int = 12;
pub const BASICLU_REMOVE_COLUMNS: lu_int = 13;
pub const BASICLU_SEARCH_ROWS: lu_int = 14;

// user readable
pub const BASICLU_DIM: lu_int = 64;
pub const BASICLU_STATUS: lu_int = 65;
pub const BASICLU_ADD_MEMORYL: lu_int = 66;
pub const BASICLU_ADD_MEMORYU: lu_int = 67;
pub const BASICLU_ADD_MEMORYW: lu_int = 68;

pub const BASICLU_NUPDATE: lu_int = 70;
pub const BASICLU_NFORREST: lu_int = 71;
pub const BASICLU_NFACTORIZE: lu_int = 72;
pub const BASICLU_NUPDATE_TOTAL: lu_int = 73;
pub const BASICLU_NFORREST_TOTAL: lu_int = 74;
pub const BASICLU_NSYMPERM_TOTAL: lu_int = 75;
pub const BASICLU_LNZ: lu_int = 76;
pub const BASICLU_UNZ: lu_int = 77;
pub const BASICLU_RNZ: lu_int = 78;
pub const BASICLU_MIN_PIVOT: lu_int = 79;
pub const BASICLU_MAX_PIVOT: lu_int = 80;
pub const BASICLU_UPDATE_COST: lu_int = 81;
pub const BASICLU_TIME_FACTORIZE: lu_int = 82;
pub const BASICLU_TIME_SOLVE: lu_int = 83;
pub const BASICLU_TIME_UPDATE: lu_int = 84;
pub const BASICLU_TIME_FACTORIZE_TOTAL: lu_int = 85;
pub const BASICLU_TIME_SOLVE_TOTAL: lu_int = 86;
pub const BASICLU_TIME_UPDATE_TOTAL: lu_int = 87;
pub const BASICLU_LFLOPS: lu_int = 88;
pub const BASICLU_UFLOPS: lu_int = 89;
pub const BASICLU_RFLOPS: lu_int = 90;
pub const BASICLU_CONDEST_L: lu_int = 91;
pub const BASICLU_CONDEST_U: lu_int = 92;
pub const BASICLU_MAX_ETA: lu_int = 93;
pub const BASICLU_NORM_L: lu_int = 94;
pub const BASICLU_NORM_U: lu_int = 95;
pub const BASICLU_NORMEST_LINV: lu_int = 96;
pub const BASICLU_NORMEST_UINV: lu_int = 97;
pub const BASICLU_MATRIX_ONENORM: lu_int = 98;
pub const BASICLU_MATRIX_INFNORM: lu_int = 99;
pub const BASICLU_RESIDUAL_TEST: lu_int = 111;

pub const BASICLU_MATRIX_NZ: lu_int = 100;
pub const BASICLU_RANK: lu_int = 101;
pub const BASICLU_BUMP_SIZE: lu_int = 102;
pub const BASICLU_BUMP_NZ: lu_int = 103;
pub const BASICLU_NSEARCH_PIVOT: lu_int = 104;
pub const BASICLU_NEXPAND: lu_int = 105;
pub const BASICLU_NGARBAGE: lu_int = 106;
pub const BASICLU_FACTOR_FLOPS: lu_int = 107;
pub const BASICLU_TIME_SINGLETONS: lu_int = 108;
pub const BASICLU_TIME_SEARCH_PIVOT: lu_int = 109;
pub const BASICLU_TIME_ELIM_PIVOT: lu_int = 110;

pub const BASICLU_PIVOT_ERROR: lu_int = 120;
