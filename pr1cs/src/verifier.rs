use ark_ec::pairing::Pairing;
use ark_ff::{Field, Zero};
use util::{
    kzg::{Mkzg, MkzgCommit, MkzgProof, SumcheckProof},
    poly::MlPoly,
    util::{batch_inverse, Proof, RandomOracle},
};

use crate::{preprocess::VerifierKey, sparse};

pub struct Verifier<E: Pairing> {
    vk: VerifierKey<E>,
}

impl<E: Pairing> Verifier<E> {
    pub fn new(vk: VerifierKey<E>) -> Self {
        Verifier { vk }
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
        for _ in 0..nv {
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
        let len_left = proof.next_u();
        let nv = (len_left - 1).ilog2() as usize + 1;
        let r = ro.next_n_fields(nv);
        let r_sum = ro.next_field();
        let (point_logup_left, y) = Self::sumcheck(sum * r_sum, nv, 3, &mut proof, ro);
        let ele = proof.next_f();
        let ele_inv = proof.next_f();
        assert_eq!(
            y,
            (ele * ele_inv - E::ScalarField::ONE)
                * MlPoly::eval_eq_pref(&r, &point_logup_left, len_left)
                + r_sum * ele_inv
        );
        let logup_values = proof.next_n_fs(3);
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

        // Verify witness-dependent PCS openings before consuming preprocessed challenges.
        let dependent_kzg_proof_len = proof.next_u();
        let dependent_kzg_proof = MkzgProof(proof.next_n_gs(dependent_kzg_proof_len));
        let dependent_sumcheck_proof_len = proof.next_u();
        let dependent_sumcheck_proof = SumcheckProof(proof.next_n_fs(dependent_sumcheck_proof_len));
        let dependent_points = vec![
            point_suf.clone(),
            point_logup_right.clone(),
            point_logup_left.clone(),
            point_logup_right.clone(),
        ];
        let dependent_comms = vec![z_suf_commit, count_commit, ele_inv_commit, tab_inv_commit];
        let dependent_values = vec![suf_z, count, ele_inv, tab_inv];
        assert!(Mkzg::batch_verify(
            &self.vk.kzg_vp,
            &dependent_points,
            &dependent_comms,
            &dependent_values,
            (dependent_kzg_proof, dependent_sumcheck_proof),
            ro,
        ));

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

        let sparse_evals = sparse::sparse_verify(
            &self.vk.kzg_vp,
            &self.vk.sparse_commits,
            &point1,
            &point_logup_left,
            &point_suf,
            &point_pre,
            gamma,
            &mut proof,
            ro,
        );

        let suf_m_expected = [
            sparse_evals.a_suf,
            sparse_evals.b_suf,
            sparse_evals.c_suf,
            sparse_evals.d_suf,
            sparse_evals.e_suf,
        ]
        .iter()
        .rev()
        .copied()
        .reduce(|acc, v| acc * r_suf + v)
        .unwrap();
        assert_eq!(suf_m, suf_m_expected);

        let pre_m_expected = [
            sparse_evals.a_pre,
            sparse_evals.b_pre,
            sparse_evals.c_pre,
            sparse_evals.d_pre,
            sparse_evals.e_pre,
        ]
        .iter()
        .rev()
        .copied()
        .reduce(|acc, v| acc * r_pre + v)
        .unwrap();
        assert_eq!(pre_m, pre_m_expected);

        let weights = proof.next_f();
        let tp = proof.next_f();
        let table_i = proof.next_f();
        let table_j = proof.next_f();
        let table_k = proof.next_f();
        let table_one = proof.next_f();
        assert_eq!(pre_z, weights);
        assert_eq!(logup_values[2], tp);
        assert_eq!(
            ele,
            logup_values
                .iter()
                .rev()
                .copied()
                .reduce(|acc, v| acc * r_lut + v)
                .unwrap()
                + alpha * MlPoly::<E::ScalarField>::eval_ones_pref(&point_logup_left, len_left)
        );
        assert_eq!(
            tab,
            ((table_k * r_lut) + table_j) * r_lut + table_i + alpha * table_one
        );

        // Then batch-verify instance-independent dense PCS openings.
        let independent_kzg_proof_len = proof.next_u();
        let independent_kzg_proof = MkzgProof(proof.next_n_gs(independent_kzg_proof_len));
        let independent_sumcheck_proof_len = proof.next_u();
        let independent_sumcheck_proof =
            SumcheckProof(proof.next_n_fs(independent_sumcheck_proof_len));
        let independent_points = vec![
            point_pre,
            point_logup_left.clone(),
            point_logup_right.clone(),
            point_logup_right.clone(),
            point_logup_right.clone(),
            point_logup_right,
        ];
        let independent_comms = vec![
            self.vk.weights_commit.clone(),
            self.vk.tp_commit.clone(),
            self.vk.table_i_commit.clone(),
            self.vk.table_j_commit.clone(),
            self.vk.table_k_commit.clone(),
            self.vk.table_one_commit.clone(),
        ];
        let independent_values = vec![weights, tp, table_i, table_j, table_k, table_one];
        assert!(Mkzg::batch_verify(
            &self.vk.kzg_vp,
            &independent_points,
            &independent_comms,
            &independent_values,
            (independent_kzg_proof, independent_sumcheck_proof),
            ro,
        ));
    }
}
