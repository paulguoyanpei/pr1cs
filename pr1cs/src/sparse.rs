use ark_ff::PrimeField;
use util::poly::MlPoly;

use crate::circuit::Circuit;

#[derive(Debug, Clone)]
pub struct SparseEvals<F: PrimeField> {
    pub a_suf: F,
    pub b_suf: F,
    pub c_suf: F,
    pub d_suf: F,
    pub e_suf: F,
    pub a_pre: F,
    pub b_pre: F,
    pub c_pre: F,
    pub d_pre: F,
    pub e_pre: F,
}

pub fn sparse_commit<F: PrimeField>(_circuit: &Circuit<F>) {}

pub fn sparse_open<F: PrimeField>(
    circuit: &Circuit<F>,
    point1: &[F],
    point_logup_left: &[F],
    point_suf: &[F],
    point_pre: &[F],
    gamma: F,
) -> SparseEvals<F> {
    let row_eq_1 = MlPoly::new_eq(&point1.to_vec()).0;
    let row_eq_lu = MlPoly::new_eq(&point_logup_left.to_vec()).0;
    let col_eq_suf = MlPoly::new_eq(&point_suf.to_vec()).0;
    let col_eq_pre = MlPoly::new_eq(&point_pre.to_vec()).0;
    let wei_len = circuit.weight_len;

    SparseEvals {
        a_suf: circuit.a.mle(&row_eq_1, &col_eq_suf, wei_len, usize::MAX, gamma),
        b_suf: circuit.b.mle(&row_eq_1, &col_eq_suf, wei_len, usize::MAX, gamma),
        c_suf: circuit.c.mle(&row_eq_1, &col_eq_suf, wei_len, usize::MAX, gamma),
        d_suf: circuit
            .d
            .mle(&row_eq_lu, &col_eq_suf, wei_len, usize::MAX, gamma),
        e_suf: circuit
            .e
            .mle(&row_eq_lu, &col_eq_suf, wei_len, usize::MAX, gamma),
        a_pre: circuit.a.mle(&row_eq_1, &col_eq_pre, 0, wei_len, gamma),
        b_pre: circuit.b.mle(&row_eq_1, &col_eq_pre, 0, wei_len, gamma),
        c_pre: circuit.c.mle(&row_eq_1, &col_eq_pre, 0, wei_len, gamma),
        d_pre: circuit.d.mle(&row_eq_lu, &col_eq_pre, 0, wei_len, gamma),
        e_pre: circuit.e.mle(&row_eq_lu, &col_eq_pre, 0, wei_len, gamma),
    }
}

pub fn sparse_verify<F: PrimeField>(_evals: &SparseEvals<F>) -> bool {
    true
}
