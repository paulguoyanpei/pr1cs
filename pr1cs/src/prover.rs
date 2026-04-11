use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use util::{kzg::MkzgProveParams, oracle::RandomOracle, poly::MlPoly};

use crate::circuit::Circuit;

struct Prover<E: Pairing, F: PrimeField> {
    kzg_pp: MkzgProveParams<E>,
    circuit: Circuit<F>,
}

impl<E, F> Prover<E, F>
where
    E: Pairing<ScalarField = F>,
    F: PrimeField,
{
    // prove a*b*eq = c
    fn sumcheck_1(a: Vec<F>, b: Vec<F>, c: Vec<F>, ro: &mut RandomOracle<F>) -> (F, F, F, Vec<F>) {
        let len = a.len();
        for i in 0..len {
            assert_eq!(a[i] * b[i], c[i])
        }
        let point = ro.next_n_fields((a.len() - 1).ilog2() as usize + 1);
        let va = MlPoly::new(a).eval(&point);
        let vb = MlPoly::new(b).eval(&point);
        let vc = MlPoly::new(c).eval(&point);
        (va, vb, vc, point)
    }

    // prove sum m[i] * z[i]
    fn sumcheck_2(m: Vec<F>, z: Vec<F>, ro: &mut RandomOracle<F>) -> Vec<F> {
        let mut res = F::ZERO;
        assert_eq!(m.len(), z.len());
        for i in 0..m.len() {
            res += m[i] * z[i];
        }
        let point = ro.next_n_fields((m.len() - 1).ilog2() as usize + 1);
        point
    }

    pub fn prove(&self, z: Vec<F>, gamma: F, ro: &mut RandomOracle<F>) {
        let cons_line = self.circuit.a.len();
        let mut a = vec![];
        let mut b = vec![];
        let mut c = vec![];
        for cr in 0..cons_line {
            a.push(self.circuit.a.mult_vec_at(cr, &z, gamma));
            b.push(self.circuit.b.mult_vec_at(cr, &z, gamma));
            c.push(self.circuit.c.mult_vec_at(cr, &z, gamma));
        }
        let (va, vb, vc, point1) = Self::sumcheck_1(a, b, c, ro);
        let eq_v = MlPoly::new_eq(&point1).0;
        let a = self.circuit.a.vec_mult(&eq_v, z.len(), gamma);
        let b = self.circuit.b.vec_mult(&eq_v, z.len(), gamma);
        let c = self.circuit.c.vec_mult(&eq_v, z.len(), gamma);
        let wei_len = self.circuit.weight_len;
        let a_suf = a[wei_len..].to_vec();
        let b_suf = b[wei_len..].to_vec();
        let c_suf = c[wei_len..].to_vec();
        let z_suf = z[wei_len..].to_vec();
        let mut m = c_suf.clone();
        let mut a_suf_sum = F::ZERO;
        let mut b_suf_sum = F::ZERO;
        let mut c_suf_sum = F::ZERO;
        for i in 0..a_suf.len() {
            a_suf_sum += a_suf[i] * z_suf[i];
            b_suf_sum += b_suf[i] * z_suf[i];
            c_suf_sum += c_suf[i] * z_suf[i];
        }
        let r = ro.next_field();
        for i in 0..m.len() {
            m[i] *= r;
            m[i] += b_suf[i];
            m[i] *= r;
            m[i] += a_suf[i];
        }
        let point2 = Self::sumcheck_2(m, z_suf, ro);

        let a_pre = a[..wei_len].to_vec();
        let b_pre = b[..wei_len].to_vec();
        let c_pre = c[..wei_len].to_vec();
        let z_pre = z[..wei_len].to_vec();
        let mut m = c_pre.clone();
        let mut a_pre_sum = F::ZERO;
        let mut b_pre_sum = F::ZERO;
        let mut c_pre_sum = F::ZERO;
        for i in 0..a_suf.len() {
            a_pre_sum += a_pre[i] * z_pre[i];
            b_pre_sum += b_pre[i] * z_pre[i];
            c_pre_sum += c_pre[i] * z_pre[i];
        }
        let r = ro.next_field();
        for i in 0..m.len() {
            m[i] *= r;
            m[i] += b_pre[i];
            m[i] *= r;
            m[i] += a_pre[i];
        }
        let point2 = Self::sumcheck_2(m, z_pre, ro);

    }
}
