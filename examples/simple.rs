// Copyright (C) 2022-2023 Richard Lincoln

use blu::{LUInt, Status, BLU};

/// Factorizes a 10x10 matrix, given a column permutation
/// vector, and solves for a single right-hand-side.
fn main() {
    // A = [
    //   [2.10                               0.14 0.09     ]
    //   [     1.10           0.06                     0.03]
    //   [          1.70                               0.04]
    //   [               1.00           0.32 0.19 0.32 0.44]
    //   [     0.06           1.60                         ]
    //   [                         2.20                    ]
    //   [               0.32           1.90           0.43]
    //   [0.14           0.19                1.10 0.22     ]
    //   [0.09           0.32                0.22 2.40     ]
    //   [     0.03 0.04 0.44           0.43           3.20]
    // ]
    let n = 10;
    let arow = vec![
        0, 7, 8, 1, 4, 9, 2, 9, 3, 6, 7, 8, 9, 1, 4, 5, 3, 6, 9, 0, 3, 7, 8, 0, 3, 7, 8, 1, 2, 3,
        6, 9,
    ];
    let acolst = vec![0, 3, 6, 8, 13, 15, 16, 19, 23, 27, 32];
    let a = vec![
        2.1, 0.14, 0.09, 1.1, 0.06, 0.03, 1.7, 0.04, 1.0, 0.32, 0.19, 0.32, 0.44, 0.06, 1.6, 2.2,
        0.32, 1.9, 0.43, 0.14, 0.19, 1.1, 0.22, 0.09, 0.32, 0.22, 2.4, 0.03, 0.04, 0.44, 0.43, 3.2,
    ];

    let b = vec![
        0.403, 0.28, 0.55, 1.504, 0.812, 1.32, 1.888, 1.168, 2.473, 3.695,
    ];
    let mut x = vec![0.0; n];

    let _col_perm = vec![6, 5, 2, 4, 1, 9, 7, 8, 0, 3];

    let mut blu = BLU::new(n as LUInt, a.len() as LUInt);

    let rv = blu.factorize(&acolst, &acolst[1..], &arow, &a);
    assert_eq!(rv, Status::OK);

    let rv = blu.solve_dense(&b, &mut x, 'N');
    assert_eq!(rv, Status::OK);

    println!("{:?}", x);
}
