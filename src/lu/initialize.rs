use crate::blu::*;
use crate::lu::LU;

impl LU {
    /// Make a BLU instance. Set parameters to defaults and initialize global counters.
    /// Reset instance for a fresh factorization.
    pub(crate) fn new(m: LUInt) -> Self {
        let mut lu = LU {
            l_mem: m,
            u_mem: m,
            w_mem: m,

            // set default parameters
            droptol: 1e-20,
            abstol: 1e-14,
            reltol: 0.1,
            nzbias: 1,
            maxsearch: 3,
            pad: 4,
            stretch: 0.3,
            compress_thres: 0.5,
            sparse_thres: 0.05,
            search_rows: 0,

            m,

            // initialize global counters
            nfactorize: 0,
            nupdate_total: 0,
            nforrest_total: 0,
            nsymperm_total: 0,
            time_factorize_total: 0.0,
            time_solve_total: 0.0,
            time_update_total: 0.0,

            l_index: vec![0; m as usize],
            u_index: vec![0; m as usize],
            w_index: vec![0; m as usize],

            l_value: vec![0.0; m as usize],
            u_value: vec![0.0; m as usize],
            w_value: vec![0.0; m as usize],

            colcount_flink: vec![0; 2 * m as usize + 2],
            colcount_blink: vec![0; 2 * m as usize + 2],
            rowcount_flink: vec![0; 2 * m as usize + 2],
            rowcount_blink: vec![0; 2 * m as usize + 2],
            w_begin: vec![0; 2 * m as usize + 2],
            w_end: vec![0; 2 * m as usize + 2],
            w_flink: vec![0; 2 * m as usize + 2],
            w_blink: vec![0; 2 * m as usize + 2],
            pinv: vec![0; m as usize],
            qinv: vec![0; m as usize],
            l_begin_p: vec![0; m as usize + 1],
            u_begin: vec![0; m as usize + 1],
            iwork0: vec![0; m as usize],

            work0: vec![0.0; m as usize],
            work1: vec![0.0; m as usize],
            col_pivot: vec![0.0; m as usize],
            row_pivot: vec![0.0; m as usize],

            ..Default::default()
        };

        // lu.reset() and lu.save() initializes the remaining slots
        // lu.load(xstore);
        lu.reset();
        // lu.save(xstore, BLU_OK);

        lu
    }
}
