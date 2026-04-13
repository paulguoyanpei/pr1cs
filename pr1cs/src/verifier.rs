use ark_ec::pairing::Pairing;
use ark_ff::{Field, PrimeField, Zero};
use util::{
    kzg::MkzgVerParams,
    poly::MlPoly,
    util::{batch_inverse, Proof, RandomOracle},
};

use crate::circuit::Circuit;

pub struct Verifier<E: Pairing> {
    kzg_vp: MkzgVerParams<E>,
    circuit: Circuit<E::ScalarField>,
    weight_len: usize,
}

impl<E: Pairing> Verifier<E> {
    pub fn new(vp: MkzgVerParams<E>, circuit: Circuit<E::ScalarField>, weight_len: usize) -> Self {
        Verifier {
            kzg_vp: vp,
            circuit,
            weight_len,
        }
    }

    fn init_base(n: usize) -> Vec<E::ScalarField> {
        let mut res = vec![];
        for i in 0..n + 1 {
            let mut prod = E::ScalarField::ONE;
            for j in 0..n + 1 {
                if i != j {
                    prod *= E::ScalarField::from(i as u32) - E::ScalarField::from(j as u32);
                }
            }
            res.push(prod);
        }
        batch_inverse(&mut res);
        res
    }

    fn uni_extrapolate(
        base: &Vec<E::ScalarField>,
        v: &Vec<E::ScalarField>,
        x: E::ScalarField,
    ) -> E::ScalarField {
        let n = base.len() - 1;
        let mut prod = x;
        for i in 1..n + 1 {
            prod *= x - E::ScalarField::from(i as u32);
        }
        let mut numerator = (0..n + 1)
            .map(|y| x - E::ScalarField::from(y as u32))
            .collect::<Vec<_>>();
        batch_inverse(&mut numerator);
        let mut res = E::ScalarField::zero();
        for i in 0..n + 1 {
            res += numerator[i] * base[i] * v[i];
        }
        res * prod
    }

    fn sumcheck(
        mut y: E::ScalarField,
        nv: usize,
        degree: usize,
        proof: &mut Proof<E>,
        ro: &mut RandomOracle<E::ScalarField>,
    ) -> (Vec<E::ScalarField>, E::ScalarField) {
        let base = Self::init_base(degree);
        let mut new_point = vec![];
        for i in 0..nv {
            let sums = proof.next_n_fs(degree + 1);
            assert_eq!(sums[0] + sums[1], y);
            let challenge = ro.next_field();
            new_point.push(challenge);
            y = Self::uni_extrapolate(&base, &sums, challenge);
        }
        (new_point, y)
    }

    pub fn verify(
        &self,
        mut proof: Proof<E>,
        gamma: E::ScalarField,
        ro: &mut RandomOracle<E::ScalarField>,
    ) {
        ro.restart();
        let len = proof.next_u();
        let nv = (len - 1).ilog2() as usize + 1;
        let r = ro.next_n_fields(nv);
        let (point1, y) = Self::sumcheck(0.into(), nv, 3, &mut proof, ro);
        let sc1_values = proof.next_n_fs(3);
        assert_eq!(
            y,
            (sc1_values[0] * sc1_values[1] - sc1_values[2])
                * MlPoly::eval_eq_pref(&r, &point1, len)
        );

        let r_lut = ro.next_field();
        let alpha = ro.next_field();
        let sum = proof.next_f();
        let len = proof.next_u();
        let nv = (len - 1).ilog2() as usize + 1;
        let r = ro.next_n_fields(nv);
        let r_sum = ro.next_field();
        let (point_logup_left, y) = Self::sumcheck(sum * r_sum, nv, 3, &mut proof, ro);
        let ele = proof.next_f();
        let ele_inv = proof.next_f();
        assert_eq!(
            y,
            (ele * ele_inv - E::ScalarField::ONE)
                * MlPoly::eval_eq_pref(&r, &point_logup_left, len)
                + r_sum * ele_inv
        );
        let logup_values = proof.next_n_fs(3);
        assert_eq!(
            ele,
            logup_values
                .iter()
                .rev()
                .copied()
                .reduce(|acc, v| acc * r_lut + v)
                .unwrap() + alpha
        );
        let len = proof.next_u();
        let nv = (len - 1).ilog2() as usize + 1;
        let r = ro.next_n_fields(nv);
        let r_sum = ro.next_field();
        let (point_logup_right, y) = Self::sumcheck(sum * r_sum, nv, 3, &mut proof, ro);
        let tab = proof.next_f();
        let tab_inv = proof.next_f();
        let count = proof.next_f();
        assert_eq!(
            y,
            (tab * tab_inv - E::ScalarField::ONE)
                * MlPoly::eval_eq_pref(&r, &point_logup_right, len)
                + r_sum * tab_inv * count
        );

        let suf_values = proof.next_n_fs(5);
        let r_suf = ro.next_field();
        let (point_suf, y) = Self::sumcheck(
            suf_values
                .iter()
                .rev()
                .copied()
                .reduce(|acc, v| acc * r_suf + v)
                .unwrap(),
            proof.next_u(),
            2,
            &mut proof,
            ro,
        );
        let suf_m = proof.next_f();
        let suf_z = proof.next_f();
        assert_eq!(y, suf_m * suf_z);

        let pre_values = proof.next_n_fs(5);
        let r_pre = ro.next_field();
        let (point_pre, y) = Self::sumcheck(
            pre_values
                .iter()
                .rev()
                .copied()
                .reduce(|acc, v| acc * r_pre + v)
                .unwrap(),
            proof.next_u(),
            2,
            &mut proof,
            ro,
        );
        for i in 0..3 {
            assert_eq!(suf_values[i] + pre_values[i], sc1_values[i]);
        }
        assert_eq!(suf_values[3] + pre_values[3], logup_values[0]);
        assert_eq!(suf_values[4] + pre_values[4], logup_values[1]);
        let pre_m = proof.next_f();
        let pre_z = proof.next_f();
        assert_eq!(y, pre_m * pre_z);
    }
}
