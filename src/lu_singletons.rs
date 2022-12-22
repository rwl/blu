use crate::basiclu::*;
use crate::lu_internal::lu;
use std::time::Instant;

/// lu_singletons()
///
/// Initialize the data structures which store the LU factors during
/// factorization and eliminate pivots with Markowitz cost zero.
///
/// During factorization the inverse pivot sequence is recorded in pinv, qinv:
///
/// - `pinv[i]` >=  0   if row i was pivot row in stage `pinv[i]`
/// - `pinv[i]` == -1   if row i has not been pivot row yet
/// - `qinv[j]` >=  0   if col j was pivot col in stage `qinv[j]`
/// - `qinv[j]` == -1   if col j has not been pivot col yet
///
/// The lower triangular factor is composed columnwise in Lindex, Lvalue.
/// The upper triangular factor is composed rowwise in Uindex, Uvalue.
/// After rank steps of factorization:
///
/// - `Lbegin_p[rank]` is the next unused position in Lindex, Lvalue.
///
/// - `Lindex[Lbegin_p[k]..]`, `Lvalue[Lbegin_p[k]..]` for 0 <= k < rank
///   stores the column of L computed in stage k without the unit diagonal.
///   The column is terminated by a negative index.
///
/// - `Ubegin[rank]` is the next unused position in Uindex, Uvalue.
///
/// - `Uindex[Ubegin[k]..Ubegin[k+1]-1]`, `Uvalue[Ubegin[k]..Ubegin[k+1]-1]`
///   stores the row of U computed in stage k without the pivot element.
///
/// `lu_singletons()` does rank >= 0 steps of factorization until no singletons are
/// left. We can either eliminate singleton columns before singleton rows or vice
/// versa. When nzbias >= 0, then eliminate singleton columns first to keep L
/// sparse. Otherwise eliminate singleton rows first. The resulting permutations
/// P, Q (stored in inverse form) make PBQ' of the form
///
///             \uuuuuuuuuuuuuuuuuuuuuuu
///              \u                    u
///               \u                   u
///                \u                  u
///                 \u                 u
///     PBQ' =       \uuuuuuu__________u               singleton columns before
///                   \     |          |               singleton rows
///                   l\    |          |
///                   ll\   |          |
///                   l l\  |   BUMP   |
///                   l  l\ |          |
///                   lllll\|__________|
///
///             \
///             l\
///             ll\
///             l l\
///             l  l\
///             l   l\       __________
///     PBQ' =  l    l\uuuuu|          |               singleton rows before
///             l    l \u  u|          |               singleton columns
///             l    l  \u u|          |
///             l    l   \uu|   BUMP   |
///             l    l    \u|          |
///             llllll     \|__________|
///
/// Off-diagonals from singleton columns (u) are stored in U, off-diagonals from
/// singleton rows (l) are stored in L and divided by the diagonal. Diagonals (\)
/// are stored in col_pivot.
///
/// Do not pivot on elements which are zero or less than abstol in magnitude.
/// When such pivots occur, the row/column remains in the active submatrix and
/// the bump factorization will detect the singularity.
///
/// Return:
///
/// - BASICLU_REALLOCATE              less than nnz(B) memory in L, U or W
/// - BASICLU_ERROR_invalid_argument  matrix B is invalid (negative number of
///                                   entries in column, index out of range,
///                                   duplicates)
/// - BASICLU_OK
pub(crate) fn lu_singletons(
    this: &mut lu,
    Bbegin: &[lu_int],
    Bend: &[lu_int],
    Bi: &[lu_int],
    Bx: &[f64],
) -> lu_int {
    let m = this.m;
    let Lmem = this.Lmem;
    let Umem = this.Umem;
    let Wmem = this.Wmem;
    let abstol = this.abstol;
    let nzbias = this.nzbias;
    let pinv = this.pinv.as_mut().unwrap();
    let qinv = this.qinv.as_mut().unwrap();
    let Lbegin_p = &mut this.Lbegin_p;
    let Ubegin = &mut this.Ubegin;
    let col_pivot = &mut this.col_pivot;
    let Lindex = this.Lindex.as_mut().unwrap();
    let Lvalue = this.Lvalue.as_mut().unwrap();
    let Uindex = this.Uindex.as_mut().unwrap();
    let Uvalue = this.Uvalue.as_mut().unwrap();
    // let iwork1 = &mut this.iwork1;
    // let iwork2 = iwork1 + m;
    let (iwork1, iwork2) = this.iwork1.as_mut().unwrap().split_at_mut(m as usize);

    let Btp = this.Wbegin.as_mut().unwrap(); // build B rowwise in W
    let Bti = this.Windex.as_mut().unwrap();
    let Btx = this.Wvalue.as_mut().unwrap();

    // lu_int i, j, pos, put, rank, Bnz, ok;
    // double tic[2];
    // lu_tic(tic);
    let tic = Instant::now();

    // Check matrix and build transpose //

    // Check pointers and count nnz(B).
    let mut Bnz = 0;
    let mut ok = 1;
    let mut j = 0;
    while j < m && ok != 0 {
        if Bend[j as usize] < Bbegin[j as usize] {
            ok = 0;
        } else {
            Bnz += Bend[j as usize] - Bbegin[j as usize];
        }
        j += 1;
    }
    if ok == 0 {
        return BASICLU_ERROR_invalid_argument;
    }

    // Check if sufficient memory in L, U, W.
    let mut ok = 1;
    if Lmem < Bnz {
        this.addmemL = Bnz - Lmem;
        ok = 0;
    }
    if Umem < Bnz {
        this.addmemU = Bnz - Umem;
        ok = 0;
    }
    if Wmem < Bnz {
        this.addmemW = Bnz - Wmem;
        ok = 0;
    }
    if ok == 0 {
        return BASICLU_REALLOCATE;
    }

    // Count nz per row, check indices.
    // memset(iwork1, 0, m); // row counts
    iwork1.fill(0); // row counts
    let mut ok = 1;
    let mut j = 0;
    while j < m && ok != 0 {
        let mut pos = Bbegin[j as usize];
        while pos < Bend[j as usize] && ok != 0 {
            let i = Bi[pos as usize];
            if i < 0 || i >= m {
                ok = 0;
            } else {
                iwork1[i as usize] += 1;
            }
            pos += 1;
        }
        j += 1;
    }
    if ok == 0 {
        return BASICLU_ERROR_invalid_argument;
    }

    // Pack matrix rowwise, check for duplicates.
    let mut put = 0;
    for i in 0..m as usize {
        // set row pointers
        Btp[i] = put;
        put += iwork1[i];
        iwork1[i] = Btp[i];
    }
    Btp[m as usize] = put;
    assert_eq!(put, Bnz);
    let mut ok = 1;
    for j in 0..m {
        // fill rows
        for pos in Bbegin[j as usize]..Bend[j as usize] {
            let i = Bi[pos as usize] as usize;
            put = iwork1[i];
            iwork1[i] += 1;
            Bti[put as usize] = j;
            Btx[put as usize] = Bx[pos as usize];
            if put > Btp[i] && Bti[(put - 1) as usize] == j {
                ok = 0;
            }
        }
    }
    if ok == 0 {
        return BASICLU_ERROR_invalid_argument;
    }

    // Pivot singletons //

    // No pivot rows or pivot columns so far.
    for i in 0..m {
        pinv[i as usize] = -1;
    }
    for j in 0..m {
        qinv[j as usize] = -1;
    }

    let rank = if nzbias >= 0 {
        // put more in U
        Lbegin_p[0] = 0;
        Ubegin[0] = 0;
        let rank = 0;

        let rank = singleton_cols(
            m, Bbegin, Bend, Bi, Bx, Btp, Bti, Btx, Ubegin, Uindex, Uvalue, Lbegin_p, Lindex,
            Lvalue, col_pivot, pinv, qinv, iwork1, iwork2, rank, abstol,
        );

        let rank = singleton_rows(
            m, Bbegin, Bend, Bi, Bx, Btp, Bti, Btx, Ubegin, Uindex, Uvalue, Lbegin_p, Lindex,
            Lvalue, col_pivot, pinv, qinv, iwork1, iwork2, rank, abstol,
        );
        rank
    } else {
        // put more in L
        Lbegin_p[0] = 0;
        Ubegin[0] = 0;
        let rank = 0;

        let rank = singleton_rows(
            m, Bbegin, Bend, Bi, Bx, Btp, Bti, Btx, Ubegin, Uindex, Uvalue, Lbegin_p, Lindex,
            Lvalue, col_pivot, pinv, qinv, iwork1, iwork2, rank, abstol,
        );

        let rank = singleton_cols(
            m, Bbegin, Bend, Bi, Bx, Btp, Bti, Btx, Ubegin, Uindex, Uvalue, Lbegin_p, Lindex,
            Lvalue, col_pivot, pinv, qinv, iwork1, iwork2, rank, abstol,
        );
        rank
    };

    // pinv, qinv were used as nonzero counters. Reset to -1 if not pivoted.
    for i in 0..m as usize {
        if pinv[i] < 0 {
            pinv[i] = -1;
        }
    }
    for j in 0..m as usize {
        if qinv[j] < 0 {
            qinv[j] = -1;
        }
    }

    this.matrix_nz = Bnz;
    this.rank = rank;
    this.time_singletons = tic.elapsed().as_secs_f64();
    BASICLU_OK
}

/// singleton_cols()
///
/// The method successively removes singleton cols from an active submatrix.
/// The active submatrix is composed of columns j for which qinv[j] < 0 and
/// rows i for which pinv[i] < 0. When removing a singleton column and its
/// associated row generates new singleton columns, these are appended to a
/// queue. The method stops when the active submatrix has no more singleton
/// columns.
///
/// For each active column j iset[j] is the XOR of row indices in the column
/// in the active submatrix. For a singleton column, this is its single row
/// index. The technique is due to J. Gilbert and described in [1], ex 3.7.
///
/// For each eliminated column its associated row is stored in U without the
/// pivot element. The pivot elements are stored in col_pivot. For each
/// eliminated pivot an empty column is appended to L.
///
/// Pivot elements which are zero or less than abstol, and empty columns in
/// the active submatrix are not eliminated. In these cases the matrix is
/// numerically or structurally singular and the bump factorization handles
/// it. (We want singularities at the end of the pivot sequence.)
///
/// [1] T. Davis, "Direct methods for sparse linear systems"
pub(crate) fn singleton_cols(
    m: lu_int,
    Bbegin: &[lu_int], // B columnwise
    Bend: &[lu_int],
    Bi: &[lu_int],
    _Bx: &[f64],
    Btp: &[lu_int], /* B rowwise */
    Bti: &[lu_int],
    Btx: &[f64],
    Up: &mut [lu_int],
    Ui: &mut [lu_int],
    Ux: &mut [f64],
    Lp: &mut [lu_int],
    Li: &mut [lu_int],
    _Lx: &mut [f64],
    col_pivot: &mut [f64],
    pinv: &mut [lu_int],
    qinv: &mut [lu_int],
    iset: &mut [lu_int],  // size m workspace
    queue: &mut [lu_int], // size m workspace
    mut rank: lu_int,
    abstol: f64,
) -> lu_int {
    // lu_int i, j, j2, nz, pos, put, end, front, tail;
    // double piv;
    let mut rk = rank;

    // Build index sets and initialize queue.
    let mut tail = 0;
    for j in 0..m {
        if qinv[j as usize] < 0 {
            let nz = Bend[j as usize] - Bbegin[j as usize];
            let mut i = 0;
            for pos in Bbegin[j as usize]..Bend[j as usize] {
                i ^= Bi[pos as usize]; // put row into set j
            }
            iset[j as usize] = i;
            qinv[j as usize] = -nz - 1; // use as nonzero counter
            if nz == 1 {
                queue[tail] = j;
                tail += 1;
            }
        }
    }

    // Eliminate singleton columns.
    let mut put = Up[rank as usize];
    for front in 0..tail {
        let j = queue[front];
        assert!(qinv[j as usize] == -2 || qinv[j as usize] == -1);
        if qinv[j as usize] == -1 {
            continue; // empty column in active submatrix
        }
        let i = iset[j as usize];
        assert!(i >= 0 && i < m);
        assert!(pinv[i as usize] < 0);
        let end = Btp[(i + 1) as usize];

        let mut pos = Btp[i as usize];
        while Bti[pos as usize] != j {
            // find pivot
            assert!(pos < end - 1);
            pos += 1;
        }

        let piv = Btx[pos as usize];
        if piv == 0.0 || piv.abs() < abstol {
            continue; // skip singularity
        }

        // Eliminate pivot.
        qinv[j as usize] = rank;
        pinv[i as usize] = rank;
        for pos in Btp[i as usize]..end {
            let j2 = Bti[pos as usize];
            if qinv[j2 as usize] < 0 {
                // test is mandatory because the initial active submatrix may
                // not be the entire matrix (rows eliminated before)

                Ui[put as usize] = j2;
                Ux[put as usize] = Btx[pos as usize];
                put += 1;
                iset[j2 as usize] ^= i; // remove i from set j2

                // if (++qinv[j2] == -2) {
                qinv[j2 as usize] += 1;
                if qinv[j2 as usize] == -2 {
                    queue[tail as usize] = j2; // new singleton
                    tail += 1;
                }
            }
        }
        Up[(rank + 1) as usize] = put;
        col_pivot[j as usize] = piv;
        rank += 1;
    }

    // Put empty columns into L.
    let mut pos = Lp[rk as usize];
    while rk < rank {
        Li[pos as usize] = -1;
        pos += 1;
        Lp[(rk + 1) as usize] = pos;
        rk += 1;
    }
    rank
}

/// singleton_rows()
///
/// Analogeous singleton_cols except that for each singleton row the
/// associated column is stored in L and divided by the pivot element. The
/// pivot element is stored in col_pivot.
fn singleton_rows(
    m: lu_int,
    Bbegin: &[lu_int], // B columnwise
    Bend: &[lu_int],
    Bi: &[lu_int],
    Bx: &[f64],
    Btp: &[lu_int], // B rowwise
    Bti: &[lu_int],
    _Btx: &[f64],
    Up: &mut [lu_int],
    _Ui: &mut [lu_int],
    _Ux: &mut [f64],
    Lp: &mut [lu_int],
    Li: &mut [lu_int],
    Lx: &mut [f64],
    col_pivot: &mut [f64],
    pinv: &mut [lu_int],
    qinv: &mut [lu_int],
    iset: &mut [lu_int],  // size m workspace
    queue: &mut [lu_int], // size m workspace
    mut rank: lu_int,
    abstol: f64,
) -> lu_int {
    // lu_int i, j, i2, nz, pos, put, end, front, tail, rk = rank;
    // double piv;
    let mut rk = rank;

    // Build index sets and initialize queue.
    let mut tail = 0;
    for i in 0..m {
        if pinv[i as usize] < 0 {
            let nz = Btp[(i + 1) as usize] - Btp[i as usize];
            let mut j = 0;
            for pos in Btp[i as usize]..Btp[(i + 1) as usize] {
                j ^= Bti[pos as usize]; // put column into set i
            }
            iset[i as usize] = j;
            pinv[i as usize] = -nz - 1; /* use as nonzero counter */
            if nz == 1 {
                queue[tail as usize] = i;
                tail += 1;
            }
        }
    }

    // Eliminate singleton rows.
    let mut put = Lp[rank as usize];
    for front in 0..tail {
        let i = queue[front];
        assert!(pinv[i as usize] == -2 || pinv[i as usize] == -1);
        if pinv[i as usize] == -1 {
            continue; // empty column in active submatrix
        }
        let j = iset[i as usize];
        assert!(j >= 0 && j < m);
        assert!(qinv[j as usize] < 0);
        let end = Bend[j as usize];

        let mut pos = Bbegin[j as usize];
        while Bi[pos as usize] != i {
            // find pivot
            assert!(pos < end - 1);
            pos += 1;
        }
        let piv = Bx[pos as usize];
        if piv == 0.0 || piv.abs() < abstol {
            continue; // skip singularity
        }

        // Eliminate pivot.
        qinv[j as usize] = rank;
        pinv[i as usize] = rank;
        for pos in Bbegin[j as usize]..end {
            let i2 = Bi[pos as usize];
            if pinv[i2 as usize] < 0 {
                // test is mandatory because the initial active submatrix may
                // not be the entire matrix (columns eliminated before)
                Li[put as usize] = i2;
                Lx[put as usize] = Bx[pos as usize] / piv;
                put += 1;
                iset[i2 as usize] ^= j; // remove j from set i2

                //if (++pinv[i2] == -2)
                pinv[i2 as usize] += 1;
                if pinv[i2 as usize] == -2 {
                    queue[tail] = i2; // new singleton
                    tail += 1;
                }
            }
        }
        Li[put as usize] = -1; // terminate column
        put += 1;
        Lp[(rank + 1) as usize] = put;
        col_pivot[j as usize] = piv;
        rank += 1;
    }

    // Put empty rows into U.
    let pos = Up[rk as usize];
    while rk < rank {
        Up[(rk + 1) as usize] = pos;
        rk += 1;
    }

    rank
}
