use ark_ec::pairing::Pairing;
use ark_ff::{Field, PrimeField};
use util::{
    kzg::{Mkzg, MkzgCommit, MkzgProveParams, MkzgVerParams},
    poly::MlPoly,
};

use crate::{
    circuit::Circuit,
    sparse::{self, SparseCommits, SparsePolys},
};

#[derive(Clone)]
pub struct DensePublicPolys<F: PrimeField> {
    pub weights: MlPoly<F>,
    pub tp: MlPoly<F>,
    pub table_i: MlPoly<F>,
    pub table_j: MlPoly<F>,
    pub table_k: MlPoly<F>,
    pub table_one: MlPoly<F>,
}

#[derive(Clone)]
pub struct ProvingKey<E: Pairing> {
    pub kzg_pp: MkzgProveParams<E>,
    pub circuit: Circuit<E::ScalarField>,
    pub dense_public_polys: DensePublicPolys<E::ScalarField>,
    pub sparse_polys: SparsePolys<E::ScalarField>,
}

#[derive(Clone)]
pub struct VerifierKey<E: Pairing> {
    pub kzg_vp: MkzgVerParams<E>,
    pub weight_len: usize,
    pub weights_commit: MkzgCommit<E>,
    pub tp_commit: MkzgCommit<E>,
    pub table_i_commit: MkzgCommit<E>,
    pub table_j_commit: MkzgCommit<E>,
    pub table_k_commit: MkzgCommit<E>,
    pub table_one_commit: MkzgCommit<E>,
    pub sparse_commits: SparseCommits<E>,
}

pub struct Preprocessor;

impl Preprocessor {
    pub fn build<E: Pairing>(
        kzg_pp: MkzgProveParams<E>,
        kzg_vp: MkzgVerParams<E>,
        circuit: Circuit<E::ScalarField>,
    ) -> (ProvingKey<E>, VerifierKey<E>) {
        let (sparse_polys, sparse_commits) = sparse::sparse_commit(&kzg_pp, &circuit);

        let dense_public_polys = DensePublicPolys {
            weights: MlPoly::new(circuit.weights.clone()),
            tp: MlPoly::new(circuit.tp.clone()),
            table_i: MlPoly::new(circuit.table.iter().map(|&(i, _, _)| i).collect()),
            table_j: MlPoly::new(circuit.table.iter().map(|&(_, j, _)| j).collect()),
            table_k: MlPoly::new(circuit.table.iter().map(|&(_, _, k)| k).collect()),
            table_one: MlPoly::new(vec![E::ScalarField::ONE; circuit.table.len()]),
        };

        let vk = VerifierKey {
            kzg_vp,
            weight_len: circuit.weight_len,
            weights_commit: Mkzg::<E>::commit(&kzg_pp, &dense_public_polys.weights),
            tp_commit: Mkzg::<E>::commit(&kzg_pp, &dense_public_polys.tp),
            table_i_commit: Mkzg::<E>::commit(&kzg_pp, &dense_public_polys.table_i),
            table_j_commit: Mkzg::<E>::commit(&kzg_pp, &dense_public_polys.table_j),
            table_k_commit: Mkzg::<E>::commit(&kzg_pp, &dense_public_polys.table_k),
            table_one_commit: Mkzg::<E>::commit(&kzg_pp, &dense_public_polys.table_one),
            sparse_commits,
        };
        let pk = ProvingKey {
            kzg_pp,
            circuit,
            dense_public_polys,
            sparse_polys,
        };
        (pk, vk)
    }
}

#[cfg(test)]
mod tests {
    use std::{cmp, panic::AssertUnwindSafe};

    use ark_bn254::{Bn254, Fr};
    use ark_ff::UniformRand;
    use rand::{rngs::StdRng, SeedableRng};
    use util::{kzg::Mkzg, poly::MlPoly, util::RandomOracle};

    use super::Preprocessor;
    use crate::{
        circuit::{Circuit, LookupType, SparseMatrix},
        instruction::Instruction,
        program::Program,
        prover::Prover,
        sparse,
        verifier::Verifier,
    };

    fn sample_fixture() -> (Program<Fr>, Vec<i64>, Circuit<Fr>, usize) {
        let n = 2;
        let m = 2;
        let in_channels = 2;
        let out_channels = 2;

        let raw_weights = vec![1, 2, 3, 4, 5, 6, 7, 8, 1, -1, 2, -2, 3, -3, 4, -4];
        let input = vec![1, 2, 3, 4, -1, 0, 2, 1];

        let mut weights = vec![1];
        weights.extend(raw_weights.iter().copied());
        let conv_output_start = weights.len() + input.len();
        let instructions = vec![
            Instruction::Conv {
                n,
                m,
                in_channels,
                out_channels,
                start1: weights.len(),
                start2: 1,
            },
            Instruction::Lookup {
                input: vec![(conv_output_start, 1)],
                tp: LookupType::Relu,
            },
            Instruction::Lookup {
                input: vec![(conv_output_start + 1, 1)],
                tp: LookupType::Relu,
            },
        ];
        let program = Program::<Fr>::new(instructions, weights);
        let trace = program.execute(input.clone());
        let aux_start = trace.len();
        let table = (-64i64..=64)
            .map(|i| (Fr::from(i), Fr::from(cmp::max(0, i)), Fr::from(2u64)))
            .collect::<Vec<_>>();
        let circuit = program.to_circuit(input.len(), aux_start, table);
        (program, input, circuit, conv_output_start)
    }

    fn dnn_like_fixture() -> Circuit<Fr> {
        let weights = vec![1, 1, 0, 0, 1];
        let input = vec![1, 2];
        let instructions = vec![
            Instruction::MatMult {
                m: 1,
                n: 2,
                k: 2,
                start1: weights.len(),
                start2: 1,
            },
            Instruction::Quant {
                input1: vec![(weights.len() + input.len(), 1)],
                input2: vec![(0, 1)],
            },
            Instruction::Lookup {
                input: vec![(weights.len() + input.len() + 2, 1)],
                tp: LookupType::Relu,
            },
        ];
        let program = Program::<Fr>::new(instructions, weights);
        let trace = program.execute(input.clone());
        let table = (-64i64..=64)
            .map(|i| (Fr::from(i), Fr::from(cmp::max(0, i)), Fr::from(2u64)))
            .collect::<Vec<_>>();
        program.to_circuit(input.len(), trace.len(), table)
    }

    fn log_len(len: usize) -> usize {
        (len.saturating_sub(1).ilog2() as usize) + 1
    }

    fn split_nonzeros(mat: &SparseMatrix<Fr>, weight_len: usize) -> (usize, usize) {
        let mut pre = 0;
        let mut suf = 0;
        for row in &mat.rows {
            for &(col, _, _) in &row.elems {
                if col < weight_len {
                    pre += 1;
                } else {
                    suf += 1;
                }
            }
        }
        (pre, suf)
    }

    fn assert_only_b_has_prefix_nonzeros(circuit: &Circuit<Fr>) {
        let (a_pre, _) = split_nonzeros(&circuit.a, circuit.weight_len);
        let (b_pre, _) = split_nonzeros(&circuit.b, circuit.weight_len);
        let (c_pre, _) = split_nonzeros(&circuit.c, circuit.weight_len);
        let (d_pre, _) = split_nonzeros(&circuit.d, circuit.weight_len);
        let (e_pre, _) = split_nonzeros(&circuit.e, circuit.weight_len);
        assert_eq!(a_pre, 0);
        assert!(b_pre > 0);
        assert_eq!(c_pre, 0);
        assert_eq!(d_pre, 0);
        assert_eq!(e_pre, 0);
    }

    #[test]
    fn dense_public_polys_match_direct_evals() {
        let (_program, _input, circuit, _output_start) = sample_fixture();
        let mut rng = StdRng::seed_from_u64(11);
        let (kzg_pp, kzg_vp) = Mkzg::<Bn254>::gen_srs(5, &mut rng);
        let (pk, _vk) = Preprocessor::build(kzg_pp, kzg_vp, circuit.clone());

        let weight_point = (0..log_len(circuit.weights.len()))
            .map(|_| Fr::rand(&mut rng))
            .collect::<Vec<_>>();
        let tp_point = (0..log_len(circuit.tp.len()))
            .map(|_| Fr::rand(&mut rng))
            .collect::<Vec<_>>();
        let table_point = (0..log_len(circuit.table.len()))
            .map(|_| Fr::rand(&mut rng))
            .collect::<Vec<_>>();

        assert_eq!(
            pk.dense_public_polys.weights.clone().eval(&weight_point),
            MlPoly::new(circuit.weights.clone()).eval(&weight_point)
        );
        assert_eq!(
            pk.dense_public_polys.tp.clone().eval(&tp_point),
            MlPoly::new(circuit.tp.clone()).eval(&tp_point)
        );
        assert_eq!(
            pk.dense_public_polys.table_i.clone().eval(&table_point),
            MlPoly::new(circuit.table.iter().map(|&(i, _, _)| i).collect()).eval(&table_point)
        );
        assert_eq!(
            pk.dense_public_polys.table_j.clone().eval(&table_point),
            MlPoly::new(circuit.table.iter().map(|&(_, j, _)| j).collect()).eval(&table_point)
        );
        assert_eq!(
            pk.dense_public_polys.table_k.clone().eval(&table_point),
            MlPoly::new(circuit.table.iter().map(|&(_, _, k)| k).collect()).eval(&table_point)
        );
    }

    #[test]
    fn sparse_open_matches_existing_matrix_mles() {
        let (program, input, circuit, output_start) = sample_fixture();
        let mut rng = StdRng::seed_from_u64(17);
        let gamma = Fr::rand(&mut rng);
        let trace = program.execute(input);
        let z = program.gen_z(output_start, trace, gamma);
        let point1 = (0..log_len(circuit.a.len()))
            .map(|_| Fr::rand(&mut rng))
            .collect::<Vec<_>>();
        let point_logup_left = (0..log_len(circuit.d.len()))
            .map(|_| Fr::rand(&mut rng))
            .collect::<Vec<_>>();
        let point_suf = (0..log_len(z.len() - circuit.weight_len))
            .map(|_| Fr::rand(&mut rng))
            .collect::<Vec<_>>();
        let point_pre = (0..log_len(circuit.weight_len))
            .map(|_| Fr::rand(&mut rng))
            .collect::<Vec<_>>();

        let (kzg_pp, kzg_vp) = Mkzg::<Bn254>::gen_srs(5, &mut rng);
        let (sparse_polys, sparse_commits) = sparse::sparse_commit(&kzg_pp, &circuit);
        let mut proof = util::util::Proof::<Bn254>::new();
        let mut ro = RandomOracle::new(&mut rng);
        let evals = sparse::sparse_open(
            &kzg_pp,
            &circuit,
            &sparse_polys,
            &point1,
            &point_logup_left,
            &point_suf,
            &point_pre,
            gamma,
            &mut proof,
            &mut ro,
        );
        let row_eq_1 = MlPoly::new_eq(&point1).0;
        let row_eq_lu = MlPoly::new_eq(&point_logup_left).0;
        let col_eq_suf = MlPoly::new_eq(&point_suf).0;
        let col_eq_pre = MlPoly::new_eq(&point_pre).0;
        let wei_len = circuit.weight_len;

        assert_eq!(
            evals.a_suf,
            circuit
                .a
                .mle(&row_eq_1, &col_eq_suf, wei_len, usize::MAX, gamma)
        );
        assert_eq!(
            evals.b_suf,
            circuit
                .b
                .mle(&row_eq_1, &col_eq_suf, wei_len, usize::MAX, gamma)
        );
        assert_eq!(
            evals.c_suf,
            circuit
                .c
                .mle(&row_eq_1, &col_eq_suf, wei_len, usize::MAX, gamma)
        );
        assert_eq!(
            evals.d_suf,
            circuit
                .d
                .mle(&row_eq_lu, &col_eq_suf, wei_len, usize::MAX, gamma)
        );
        assert_eq!(
            evals.e_suf,
            circuit
                .e
                .mle(&row_eq_lu, &col_eq_suf, wei_len, usize::MAX, gamma)
        );
        assert_eq!(
            evals.a_pre,
            circuit.a.mle(&row_eq_1, &col_eq_pre, 0, wei_len, gamma)
        );
        assert_eq!(
            evals.b_pre,
            circuit.b.mle(&row_eq_1, &col_eq_pre, 0, wei_len, gamma)
        );
        assert_eq!(
            evals.c_pre,
            circuit.c.mle(&row_eq_1, &col_eq_pre, 0, wei_len, gamma)
        );
        assert_eq!(
            evals.d_pre,
            circuit.d.mle(&row_eq_lu, &col_eq_pre, 0, wei_len, gamma)
        );
        assert_eq!(
            evals.e_pre,
            circuit.e.mle(&row_eq_lu, &col_eq_pre, 0, wei_len, gamma)
        );
        assert_eq!(evals.a_pre, Fr::from(0u64));
        assert_eq!(evals.c_pre, Fr::from(0u64));
        assert_eq!(evals.d_pre, Fr::from(0u64));
        assert_eq!(evals.e_pre, Fr::from(0u64));

        ro.restart();
        let verified = sparse::sparse_verify(
            &kzg_vp,
            &sparse_commits,
            &point1,
            &point_logup_left,
            &point_suf,
            &point_pre,
            gamma,
            &mut proof,
            &mut ro,
        );
        assert_eq!(verified.b_pre, evals.b_pre);
    }

    #[test]
    fn sparse_commit_uses_single_suffix_batch_layout() {
        let (_program, _input, conv_circuit, _output_start) = sample_fixture();
        let dnn_circuit = dnn_like_fixture();
        let mut rng = StdRng::seed_from_u64(19);
        let (kzg_pp, _kzg_vp) = Mkzg::<Bn254>::gen_srs(5, &mut rng);

        for circuit in [&conv_circuit, &dnn_circuit] {
            assert_only_b_has_prefix_nonzeros(circuit);
            let (polys, commits) = sparse::sparse_commit(&kzg_pp, circuit);
            assert_eq!(polys.supergroups.len(), 1);
            assert_eq!(commits.supergroups.len(), 1);
            assert_eq!(polys.supergroups[0].matrix_lens.len(), 5);
            assert_eq!(
                commits.supergroups[0].matrix_lens,
                polys.supergroups[0].matrix_lens
            );
            assert_eq!(polys.b_pre.len, commits.b_pre.len);
            assert_eq!(polys.b_pre.log_col, commits.b_pre.log_col);
            assert_eq!(polys.b_pre.count_col.0.len(), 1 << polys.b_pre.log_col);
            assert!(polys.b_pre.count_col.0.len() >= circuit.weight_len);
        }
    }

    #[test]
    fn sparse_verify_rejects_nonzero_unproved_prefix_claims() {
        let (_program, _input, mut circuit, _output_start) = sample_fixture();
        circuit.a.rows[0].elems.push((1, Fr::from(1u64), None));

        let mut rng = StdRng::seed_from_u64(29);
        let gamma = Fr::from(7u64);
        let point1 = vec![Fr::from(0u64); log_len(circuit.a.len())];
        let point_logup_left = vec![Fr::from(0u64); log_len(circuit.d.len())];
        let point_suf = vec![Fr::from(0u64); log_len(circuit.z_len - circuit.weight_len)];
        let mut point_pre = vec![Fr::from(0u64); log_len(circuit.weight_len)];
        point_pre[0] = Fr::from(1u64);

        let (kzg_pp, kzg_vp) = Mkzg::<Bn254>::gen_srs(5, &mut rng);
        let (sparse_polys, sparse_commits) = sparse::sparse_commit(&kzg_pp, &circuit);
        let mut proof = util::util::Proof::<Bn254>::new();
        let mut ro = RandomOracle::new(&mut rng);
        let evals = sparse::sparse_open(
            &kzg_pp,
            &circuit,
            &sparse_polys,
            &point1,
            &point_logup_left,
            &point_suf,
            &point_pre,
            gamma,
            &mut proof,
            &mut ro,
        );
        assert_eq!(evals.a_pre, Fr::from(1u64));

        ro.restart();
        let rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
            sparse::sparse_verify(
                &kzg_vp,
                &sparse_commits,
                &point1,
                &point_logup_left,
                &point_suf,
                &point_pre,
                gamma,
                &mut proof,
                &mut ro,
            );
        }));
        assert!(rejected.is_err());
    }

    #[test]
    fn preprocessing_reuses_verifier_key_across_gammas() {
        let (program, input, circuit, output_start) = sample_fixture();
        let mut rng = StdRng::seed_from_u64(23);
        let (kzg_pp, kzg_vp) = Mkzg::<Bn254>::gen_srs(5, &mut rng);
        let (pk, vk) = Preprocessor::build(kzg_pp, kzg_vp, circuit.clone());
        let prover = Prover::new(pk.clone());
        let verifier = Verifier::new(vk.clone());

        for gamma in [Fr::rand(&mut rng), Fr::rand(&mut rng)] {
            let trace = program.execute(input.clone());
            let z = program.gen_z(output_start, trace, gamma);
            let mut ro = RandomOracle::new(&mut rng);
            let proof = prover.prove(z, gamma, &mut ro);
            verifier.verify(proof, gamma, &mut ro);
        }
    }

    #[test]
    fn verify_with_non_power_of_two_lookup_count() {
        // Three lookups → lu_line = 3, which is NOT a power of two. Exercises the
        // logup-left indicator term that was missing from the verifier before.
        let mut weights = vec![1i64];
        weights.extend([1, -1, 2, -2, 3, -3, 4, -4]);
        let input = vec![1i64, 2, 3, 4, -1, 0, 2, 1];
        let conv_output_start = weights.len() + input.len();
        let instructions = vec![
            Instruction::Conv {
                n: 2,
                m: 2,
                in_channels: 2,
                out_channels: 1,
                start1: weights.len(),
                start2: 1,
            },
            Instruction::Lookup {
                input: vec![(conv_output_start, 1)],
                tp: LookupType::Relu,
            },
            Instruction::Lookup {
                input: vec![(conv_output_start + 1, 1)],
                tp: LookupType::Relu,
            },
            Instruction::Lookup {
                input: vec![(conv_output_start + 2, 1)],
                tp: LookupType::Relu,
            },
        ];
        let program = Program::<Fr>::new(instructions, weights);
        let trace = program.execute(input.clone());
        let aux_start = trace.len();
        let table = (-64i64..=64)
            .map(|i| (Fr::from(i), Fr::from(cmp::max(0, i)), Fr::from(2u64)))
            .collect::<Vec<_>>();
        let circuit = program.to_circuit(input.len(), aux_start, table);
        assert_eq!(circuit.d.len(), 3);

        let mut rng = StdRng::seed_from_u64(57);
        let gamma = Fr::rand(&mut rng);
        let z = program.gen_z(conv_output_start, trace, gamma);
        let (kzg_pp, kzg_vp) = Mkzg::<Bn254>::gen_srs(5, &mut rng);
        let (pk, vk) = Preprocessor::build(kzg_pp, kzg_vp, circuit);
        let prover = Prover::new(pk);
        let mut ro = RandomOracle::new(&mut rng);
        let proof = prover.prove(z, gamma, &mut ro);
        Verifier::new(vk).verify(proof, gamma, &mut ro);
    }

    #[test]
    fn wrong_weights_commit_rejects() {
        let (program, input, circuit, output_start) = sample_fixture();
        let mut rng = StdRng::seed_from_u64(31);
        let gamma = Fr::rand(&mut rng);
        let trace = program.execute(input);
        let z = program.gen_z(output_start, trace, gamma);
        let (kzg_pp, kzg_vp) = Mkzg::<Bn254>::gen_srs(5, &mut rng);
        let (pk, vk) = Preprocessor::build(kzg_pp, kzg_vp, circuit);
        let prover = Prover::new(pk);
        let mut ro = RandomOracle::new(&mut rng);
        let proof = prover.prove(z, gamma, &mut ro);

        let mut bad_vk = vk.clone();
        bad_vk.weights_commit = bad_vk.tp_commit.clone();
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            Verifier::new(bad_vk).verify(proof, gamma, &mut ro)
        }));
        assert!(result.is_err());
    }

    #[test]
    fn wrong_tp_commit_rejects() {
        let (program, input, circuit, output_start) = sample_fixture();
        let mut rng = StdRng::seed_from_u64(37);
        let gamma = Fr::rand(&mut rng);
        let trace = program.execute(input);
        let z = program.gen_z(output_start, trace, gamma);
        let (kzg_pp, kzg_vp) = Mkzg::<Bn254>::gen_srs(5, &mut rng);
        let (pk, vk) = Preprocessor::build(kzg_pp, kzg_vp, circuit);
        let prover = Prover::new(pk);
        let mut ro = RandomOracle::new(&mut rng);
        let proof = prover.prove(z, gamma, &mut ro);

        let mut bad_vk = vk.clone();
        bad_vk.tp_commit = bad_vk.weights_commit.clone();
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            Verifier::new(bad_vk).verify(proof, gamma, &mut ro)
        }));
        assert!(result.is_err());
    }

    #[test]
    fn wrong_table_commit_rejects() {
        let (program, input, circuit, output_start) = sample_fixture();
        let mut rng = StdRng::seed_from_u64(41);
        let gamma = Fr::rand(&mut rng);
        let trace = program.execute(input);
        let z = program.gen_z(output_start, trace, gamma);
        let (kzg_pp, kzg_vp) = Mkzg::<Bn254>::gen_srs(5, &mut rng);
        let (pk, vk) = Preprocessor::build(kzg_pp, kzg_vp, circuit);
        let prover = Prover::new(pk);
        let mut ro = RandomOracle::new(&mut rng);
        let proof = prover.prove(z, gamma, &mut ro);

        let mut bad_vk = vk.clone();
        bad_vk.table_i_commit = bad_vk.table_j_commit.clone();
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            Verifier::new(bad_vk).verify(proof, gamma, &mut ro)
        }));
        assert!(result.is_err());
    }

    #[test]
    fn wrong_b_pre_commit_rejects() {
        let (program, input, circuit, output_start) = sample_fixture();
        let mut rng = StdRng::seed_from_u64(43);
        let gamma = Fr::rand(&mut rng);
        let trace = program.execute(input);
        let z = program.gen_z(output_start, trace, gamma);
        let (kzg_pp, kzg_vp) = Mkzg::<Bn254>::gen_srs(5, &mut rng);
        let (pk, vk) = Preprocessor::build(kzg_pp, kzg_vp, circuit);
        let prover = Prover::new(pk);
        let mut ro = RandomOracle::new(&mut rng);
        let proof = prover.prove(z, gamma, &mut ro);

        let mut bad_vk = vk.clone();
        bad_vk.sparse_commits.b_pre.val = bad_vk.sparse_commits.b_pre.row.clone();
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            Verifier::new(bad_vk).verify(proof, gamma, &mut ro)
        }));
        assert!(result.is_err());
    }
}
