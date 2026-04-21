use ark_ec::pairing::Pairing;
use ark_ff::{Field, PrimeField, Zero};
use util::{
    kzg::{Mkzg, MkzgCommit, MkzgProof, MkzgVerParams, SumcheckProof},
    poly::MlPoly,
    util::{batch_inverse, Proof, RandomOracle},
};

use crate::circuit::Circuit;

pub struct Verifier<E: Pairing> {
    kzg_vp: MkzgVerParams<E>,
    circuit: Circuit<E::ScalarField>,
    weights: Vec<E::ScalarField>,
}

impl<E: Pairing> Verifier<E> {
    pub fn new(
        vp: MkzgVerParams<E>,
        circuit: Circuit<E::ScalarField>,
        weights: Vec<E::ScalarField>,
    ) -> Self {
        Verifier {
            kzg_vp: vp,
            circuit,
            weights,
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

        // Read z_suf commitment.
        let z_suf_commit_len = proof.next_u();
        let z_suf_commit = MkzgCommit(proof.next_n_gs(z_suf_commit_len));

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

        // Read count, ele_inv, tab_inv commitments.
        let count_commit_len = proof.next_u();
        let count_commit = MkzgCommit(proof.next_n_gs(count_commit_len));
        let ele_inv_commit_len = proof.next_u();
        let ele_inv_commit = MkzgCommit(proof.next_n_gs(ele_inv_commit_len));
        let tab_inv_commit_len = proof.next_u();
        let tab_inv_commit = MkzgCommit(proof.next_n_gs(tab_inv_commit_len));

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
        // logup_values[2] is the claimed tp(point_logup_left); tp is public.
        assert_eq!(
            logup_values[2],
            MlPoly::new(self.circuit.tp.clone()).eval(&point_logup_left)
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
        // tab is public: tab[i] = table[i].0 + table[i].1 * r_lut + table[i].2 * r_lut^2,
        // with `alpha` added inside logup_sumcheck_right before the sumcheck.
        let tab_vec: Vec<E::ScalarField> = self
            .circuit
            .table
            .iter()
            .map(|&(i, j, k)| ((k * r_lut) + j) * r_lut + i + alpha)
            .collect();
        assert_eq!(tab, MlPoly::new(tab_vec).eval(&point_logup_right));

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

        // Verify suf_m against the sparse-matrix MLEs over the suffix column window.
        // Rows of A, B, C are reduced against eq(point1); D, E against eq(point_logup_left).
        let wei_len = self.circuit.weight_len;
        let row_eq_1 = MlPoly::new_eq(&point1).0;
        let row_eq_lu = MlPoly::new_eq(&point_logup_left).0;
        let col_eq_suf = MlPoly::new_eq(&point_suf).0;
        let a_suf_eval =
            self.circuit
                .a
                .mle(&row_eq_1, &col_eq_suf, wei_len, usize::MAX, gamma);
        let b_suf_eval =
            self.circuit
                .b
                .mle(&row_eq_1, &col_eq_suf, wei_len, usize::MAX, gamma);
        let c_suf_eval =
            self.circuit
                .c
                .mle(&row_eq_1, &col_eq_suf, wei_len, usize::MAX, gamma);
        let d_suf_eval =
            self.circuit
                .d
                .mle(&row_eq_lu, &col_eq_suf, wei_len, usize::MAX, gamma);
        let e_suf_eval =
            self.circuit
                .e
                .mle(&row_eq_lu, &col_eq_suf, wei_len, usize::MAX, gamma);
        let suf_m_expected = [
            a_suf_eval,
            b_suf_eval,
            c_suf_eval,
            d_suf_eval,
            e_suf_eval,
        ]
        .iter()
        .rev()
        .copied()
        .reduce(|acc, v| acc * r_suf + v)
        .unwrap();
        assert_eq!(suf_m, suf_m_expected);

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

        // Verify pre_m against the sparse-matrix MLEs over the prefix column window.
        let col_eq_pre = MlPoly::new_eq(&point_pre).0;
        let a_pre_eval = self.circuit.a.mle(&row_eq_1, &col_eq_pre, 0, wei_len, gamma);
        let b_pre_eval = self.circuit.b.mle(&row_eq_1, &col_eq_pre, 0, wei_len, gamma);
        let c_pre_eval = self.circuit.c.mle(&row_eq_1, &col_eq_pre, 0, wei_len, gamma);
        let d_pre_eval = self.circuit.d.mle(&row_eq_lu, &col_eq_pre, 0, wei_len, gamma);
        let e_pre_eval = self.circuit.e.mle(&row_eq_lu, &col_eq_pre, 0, wei_len, gamma);
        let pre_m_expected = [
            a_pre_eval,
            b_pre_eval,
            c_pre_eval,
            d_pre_eval,
            e_pre_eval,
        ]
        .iter()
        .rev()
        .copied()
        .reduce(|acc, v| acc * r_pre + v)
        .unwrap();
        assert_eq!(pre_m, pre_m_expected);

        // pre_z = weights(point_pre); weights are public.
        assert_eq!(pre_z, MlPoly::new(self.weights.clone()).eval(&point_pre));

        // Batch-verify the four PCS openings.
        let kzg_proof_len = proof.next_u();
        let kzg_proof = MkzgProof(proof.next_n_gs(kzg_proof_len));
        let sumcheck_proof_len = proof.next_u();
        let sumcheck_proof = SumcheckProof(proof.next_n_fs(sumcheck_proof_len));

        let points = vec![
            point_suf,
            point_logup_right.clone(),
            point_logup_left,
            point_logup_right,
        ];
        let comms = vec![z_suf_commit, count_commit, ele_inv_commit, tab_inv_commit];
        let values = vec![suf_z, count, ele_inv, tab_inv];
        assert!(Mkzg::batch_verify(
            &self.kzg_vp,
            &points,
            &comms,
            &values,
            (kzg_proof, sumcheck_proof),
            ro,
        ));
    }
}
