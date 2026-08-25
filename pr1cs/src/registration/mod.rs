//! Model registration: the one-time certificate that the committed pR1CS
//! circuit is the compilation of a valid private VM program.
//!
//! A committed circuit on its own only defines an NP relation, so the same
//! input may admit several accepting outputs. Registration closes that gap: it
//! binds the circuit commitment to a program of the deterministic VM, whose
//! execution fixes one output per input.
//!
//! The certificate proves the relation
//!
//! ```text
//! R_reg = {(cm_C, cm_p); (P, C, p) : C = Compile(P) != bottom, Open(cm_C, C), Open(cm_p, p)}
//! ```
//!
//! against the very commitments the online proof is verified against: the
//! sparse descriptor commitments in [`crate::preprocess::VerifierKey`] and the
//! parameter commitment `weights_commit`. The rules it proves are listed in
//! [`relation::build_rules`] and mirrored in the clear by
//! [`encode::check_compilation`].
//!
//! Scope note: like the rest of this crate, the argument targets binding and
//! soundness. The hiding layer (blinded commitments and re-masked sumchecks)
//! is not implemented for the online proof either, so a registration
//! transcript is not yet zero-knowledge.

pub(crate) mod args;
pub mod encode;
pub(crate) mod family;
pub(crate) mod relation;
pub mod witness;

use ark_ec::pairing::Pairing;
use ark_ff::{AdditiveGroup, PrimeField};

use util::{poly::MlPoly, util::Proof, util::RandomOracle};

use crate::{
    preprocess::{ProvingKey, VerifierKey},
    registration::{
        args::{eq_sumcheck_prove, eq_sumcheck_verify, frac_sum_prove, frac_sum_verify,
               ProverClaims, VerifierClaims},
        relation::{build_rules, Block, Tab, NUM_CHALLENGES},
        witness::{CTR_AUX, CTR_CR, CTR_LR, CTR_OUT},
    },
    sparse::{identity_mle, prefix_mle},
};

pub use encode::{check_compilation, Deltas, Opcode, ProgramEncoding, RuleViolation};
pub use witness::{ProgramCommits, ProgramWitness, RegistrationProfile};

/// Why a program cannot be registered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistrationError {
    /// The circuit is not the compilation of the program.
    NotCompiled(RuleViolation),
}

impl std::fmt::Display for RegistrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotCompiled(v) => write!(f, "{}", v),
        }
    }
}

/// What a registered model publishes next to its circuit commitment.
#[derive(Clone)]
pub struct RegistrationKey<E: Pairing> {
    pub profile: RegistrationProfile,
    pub commits: ProgramCommits<E>,
}

pub struct Registrar;

impl Registrar {
    /// Commits the private program and proves the committed circuit is its
    /// compilation. The proof is verified with [`Registrar::verify`] against
    /// the same verifier key the online proofs use.
    pub fn register<E: Pairing>(
        pk: &ProvingKey<E>,
        vk: &VerifierKey<E>,
        enc: &ProgramEncoding,
        ro: &mut RandomOracle<E::ScalarField>,
    ) -> Result<(RegistrationKey<E>, Proof<E>), RegistrationError> {
        check_compilation(enc, &pk.circuit).map_err(RegistrationError::NotCompiled)?;

        let profile = RegistrationProfile::new(vk, enc);
        let witness = ProgramWitness::build(enc, &pk.circuit, &profile);
        let commits = witness.commit(&trim_pp(&pk.kzg_pp, &profile));
        let key = RegistrationKey { profile, commits };
        let proof = Self::prove(pk, &key, &witness, ro);
        Ok((key, proof))
    }

    fn prove<E: Pairing>(
        pk: &ProvingKey<E>,
        key: &RegistrationKey<E>,
        witness: &ProgramWitness<E::ScalarField>,
        ro: &mut RandomOracle<E::ScalarField>,
    ) -> Proof<E> {
        let profile = &key.profile;
        let rules = build_rules::<E::ScalarField>(profile);
        let tables = ProverTables {
            witness,
            sg_row: &pk.sparse_polys.supergroups[0].row,
            sg_col: &pk.sparse_polys.supergroups[0].col,
            sg_val: &pk.sparse_polys.supergroups[0].val,
            sg_pow: &pk.sparse_polys.supergroups[0].pow,
            b_row: &pk.sparse_polys.b_pre.row,
            b_col: &pk.sparse_polys.b_pre.col,
            b_val: &pk.sparse_polys.b_pre.val,
            b_pow: &pk.sparse_polys.b_pre.pow,
            tp: &pk.dense_public_polys.tp,
        };

        let mut proof = Proof::new();
        let mut claims = ProverClaims::new();
        let chals = ro.next_n_fields(NUM_CHALLENGES);

        for check in &rules.zero_checks {
            let len = 1usize << check.log_len;
            let eq_point = ro.next_n_fields(check.log_len);
            let columns: Vec<Vec<E::ScalarField>> = check
                .tables
                .iter()
                .map(|&tab| tables.column(tab, len))
                .collect();
            let g = |v: &[E::ScalarField]| (check.g)(v, &chals);
            let (point, values) =
                eq_sumcheck_prove::<E, _>(&eq_point, columns, check.degree, g, &mut proof, ro);
            for (slot, &tab) in check.tables.iter().enumerate() {
                if tab.is_committed() {
                    claims.push(tables.poly(tab), point.clone(), values[slot]);
                }
            }
        }

        for equality in &rules.equalities {
            let alpha = ro.next_field();
            let beta = ro.next_field();
            let mut left = E::ScalarField::ZERO;
            let mut right = E::ScalarField::ZERO;
            for block in &equality.left {
                left += Self::prove_block(&tables, block, alpha, beta, &mut proof, &mut claims, ro);
            }
            for block in &equality.right {
                right +=
                    Self::prove_block(&tables, block, alpha, beta, &mut proof, &mut claims, ro);
            }
            assert_eq!(left, right, "equality `{}` does not hold", equality.name);
        }

        Self::boundary_claims(profile, |tab, point, value| {
            claims.push(tables.poly(tab), point, value)
        });

        claims.open(&trim_pp(&pk.kzg_pp, profile), &mut proof, ro);
        proof
    }

    /// Checks a registration certificate. Panics on a failed check, matching
    /// [`crate::verifier::Verifier`].
    pub fn verify<E: Pairing>(
        vk: &VerifierKey<E>,
        key: &RegistrationKey<E>,
        mut proof: Proof<E>,
        ro: &mut RandomOracle<E::ScalarField>,
    ) {
        ro.restart();
        let profile = &key.profile;
        assert_eq!(
            profile.weight_len, vk.weight_len,
            "profile disagrees with the verifier key"
        );
        let sg = &vk.sparse_commits.supergroups[0];
        assert_eq!(profile.matrix_lens, sg.matrix_lens, "descriptor layout differs");
        assert_eq!(profile.entry_len, sg.len, "descriptor length differs");
        assert_eq!(
            profile.b_pre_len, vk.sparse_commits.b_pre.len,
            "prefix descriptor length differs"
        );

        let rules = build_rules::<E::ScalarField>(profile);
        let tables = VerifierTables { vk, key };
        let mut claims = VerifierClaims::new();
        let chals = ro.next_n_fields(NUM_CHALLENGES);

        for check in &rules.zero_checks {
            let eq_point = ro.next_n_fields(check.log_len);
            let g = |v: &[E::ScalarField]| (check.g)(v, &chals);
            let (point, values) = eq_sumcheck_verify::<E, _>(
                &eq_point,
                E::ScalarField::ZERO,
                check.degree,
                check.tables.len(),
                g,
                &mut proof,
                ro,
            );
            for (slot, &tab) in check.tables.iter().enumerate() {
                if tab.is_committed() {
                    claims.push(tables.commit(tab), point.clone(), values[slot]);
                } else {
                    assert_eq!(
                        values[slot],
                        public_eval::<E::ScalarField>(tab, &point),
                        "public table {:?} misreported in `{}`",
                        tab,
                        check.name
                    );
                }
            }
        }

        for equality in &rules.equalities {
            let alpha = ro.next_field();
            let beta = ro.next_field();
            let mut left = E::ScalarField::ZERO;
            let mut right = E::ScalarField::ZERO;
            for block in &equality.left {
                left += Self::verify_block(&tables, block, alpha, beta, &mut proof, &mut claims, ro);
            }
            for block in &equality.right {
                right +=
                    Self::verify_block(&tables, block, alpha, beta, &mut proof, &mut claims, ro);
            }
            assert_eq!(left, right, "equality `{}` rejected", equality.name);
        }

        Self::boundary_claims(profile, |tab, point, value| {
            claims.push(tables.commit(tab), point, value)
        });

        assert!(
            claims.verify(
                &vk.kzg_vp
                    .trim(profile.log_srs().min(vk.kzg_vp.log_len())),
                &mut proof,
                ro
            ),
            "registration openings rejected"
        );
    }

    /// The counters start at the parameter block and end at the totals the
    /// public profile announces; both ends are read straight off the committed
    /// vectors at boolean points.
    fn boundary_claims<F: PrimeField>(
        profile: &RegistrationProfile,
        mut push: impl FnMut(Tab, Vec<F>, F),
    ) {
        let nv = profile.log_instr();
        let first = vec![F::ZERO; nv];
        let last = vec![F::ONE; nv];
        let starts = [
            profile.weight_len + profile.input_len,
            profile.aux_start,
            0,
            0,
        ];
        let ends = [
            profile.aux_start,
            profile.z_len,
            profile.cons_line,
            profile.lu_line,
        ];
        for counter in [CTR_OUT, CTR_AUX, CTR_CR, CTR_LR] {
            push(
                Tab::Ctr(counter),
                first.clone(),
                F::from(starts[counter] as u64),
            );
            push(Tab::Ctr(counter), last.clone(), F::from(ends[counter] as u64));
        }
        // Each family's runs start at zero and end at the family's size.
        for (f, len) in profile.family_lens.iter().enumerate() {
            push(Tab::Base(f), first.clone(), F::ZERO);
            push(Tab::Base(f), last.clone(), F::from(*len as u64));
        }
    }

    fn prove_block<E: Pairing>(
        tables: &ProverTables<'_, E::ScalarField>,
        block: &Block,
        alpha: E::ScalarField,
        beta: E::ScalarField,
        proof: &mut Proof<E>,
        claims: &mut ProverClaims<E::ScalarField>,
        ro: &mut RandomOracle<E::ScalarField>,
    ) -> E::ScalarField {
        let len = 1usize << block.log_len;
        let tabs = block_tabs(block);
        let columns: Vec<Vec<E::ScalarField>> =
            tabs.iter().map(|&tab| tables.column(tab, len)).collect();

        let mut num = Vec::with_capacity(len);
        let mut den = Vec::with_capacity(len);
        let mut row = Vec::with_capacity(tabs.len());
        for x in 0..len {
            row.clear();
            for (slot, &tab) in tabs.iter().enumerate() {
                row.push((tab, columns[slot][x]));
            }
            num.push(block.num.eval(&row));
            den.push(alpha + fingerprint(&block.comps, &row, beta));
        }

        let (sum, point, num_eval, den_eval) = frac_sum_prove::<E>(num, den, proof, ro);
        let mut values = Vec::new();
        for &tab in &tabs {
            let value = if tab.is_committed() {
                let v = tables.poly(tab).clone().eval(&point);
                claims.push(tables.poly(tab), point.clone(), v);
                v
            } else {
                public_eval::<E::ScalarField>(tab, &point)
            };
            values.push((tab, value));
        }
        let committed: Vec<E::ScalarField> = tabs
            .iter()
            .zip(values.iter())
            .filter(|(tab, _)| tab.is_committed())
            .map(|(_, (_, v))| *v)
            .collect();
        proof.push_f(&committed);

        debug_assert_eq!(num_eval, block.num.eval(&values));
        debug_assert_eq!(den_eval, alpha + fingerprint(&block.comps, &values, beta));
        sum
    }

    fn verify_block<E: Pairing>(
        tables: &VerifierTables<'_, E>,
        block: &Block,
        alpha: E::ScalarField,
        beta: E::ScalarField,
        proof: &mut Proof<E>,
        claims: &mut VerifierClaims<E>,
        ro: &mut RandomOracle<E::ScalarField>,
    ) -> E::ScalarField {
        let tabs = block_tabs(block);
        let (sum, point, num_eval, den_eval) = frac_sum_verify::<E>(block.log_len, proof, ro);
        let committed: Vec<Tab> = tabs.iter().copied().filter(|t| t.is_committed()).collect();
        let opened = proof.next_n_fs(committed.len());

        let mut values = Vec::with_capacity(tabs.len());
        let mut next = 0;
        for &tab in &tabs {
            if tab.is_committed() {
                let value = opened[next];
                next += 1;
                claims.push(tables.commit(tab), point.clone(), value);
                values.push((tab, value));
            } else {
                values.push((tab, public_eval::<E::ScalarField>(tab, &point)));
            }
        }

        assert_eq!(num_eval, block.num.eval(&values), "leaf numerator rejected");
        assert_eq!(
            den_eval,
            alpha + fingerprint(&block.comps, &values, beta),
            "leaf fingerprint rejected"
        );
        sum
    }
}

/// The proving SRS cut down to the widest domain a registration touches.
/// Committing a short polynomial against the trimmed SRS gives the same
/// commitment as the full one, since the padding scalars are zero.
fn trim_pp<E: Pairing>(
    kzg_pp: &util::kzg::MkzgProveParams<E>,
    profile: &RegistrationProfile,
) -> util::kzg::MkzgProveParams<E> {
    let log_chunk = kzg_pp.0.len().ilog2() as usize;
    kzg_pp.trim(profile.log_srs().min(log_chunk))
}

fn fingerprint<F: PrimeField>(comps: &[relation::Lin], values: &[(Tab, F)], beta: F) -> F {
    let mut acc = F::ZERO;
    let mut weight = F::ONE;
    for comp in comps {
        acc += weight * comp.eval(values);
        weight *= beta;
    }
    acc
}

/// Tables a block reads, in the order both sides walk them.
fn block_tabs(block: &Block) -> Vec<Tab> {
    let mut tabs = vec![];
    for lin in std::iter::once(&block.num).chain(block.comps.iter()) {
        for &(_, tab) in &lin.terms {
            if !tabs.contains(&tab) {
                tabs.push(tab);
            }
        }
    }
    tabs
}

/// Value of a public table's multilinear extension at `point`.
fn public_eval<F: PrimeField>(tab: Tab, point: &[F]) -> F {
    match tab {
        Tab::One => F::ONE,
        Tab::Identity => identity_mle(point),
        Tab::Ind(lo, hi) => prefix_mle::<F>(point, hi) - prefix_mle::<F>(point, lo),
        _ => panic!("{:?} is committed, not public", tab),
    }
}

struct ProverTables<'a, F: PrimeField> {
    witness: &'a ProgramWitness<F>,
    sg_row: &'a MlPoly<F>,
    sg_col: &'a MlPoly<F>,
    sg_val: &'a MlPoly<F>,
    sg_pow: &'a MlPoly<F>,
    b_row: &'a MlPoly<F>,
    b_col: &'a MlPoly<F>,
    b_val: &'a MlPoly<F>,
    b_pow: &'a MlPoly<F>,
    tp: &'a MlPoly<F>,
}

impl<'a, F: PrimeField> ProverTables<'a, F> {
    fn poly(&self, tab: Tab) -> &MlPoly<F> {
        let w = self.witness;
        match tab {
            Tab::Sel(j) => &w.sel[j],
            Tab::Opn(j) => &w.opn[j],
            Tab::Ctr(j) => &w.ctr[j],
            Tab::DCtr(j) => &w.dctr[j],
            Tab::Base(f) => &w.base[f],
            Tab::Mult(f) => &w.mult[f],
            Tab::RowOut => &w.row_out,
            Tab::RowPtr => &w.row_ptr,
            Tab::RowCr => &w.row_cr,
            Tab::RowDcr => &w.row_dcr,
            Tab::IsLin => &w.is_lin,
            Tab::RowSlackLo => &w.row_slack_lo,
            Tab::RowSlackHi => &w.row_slack_hi,
            Tab::BCnt => &w.b_cnt,
            Tab::EntLin => &w.ent_lin,
            Tab::MaskA => &w.mask_a,
            Tab::MaskB => &w.mask_b,
            Tab::MaskC => &w.mask_c,
            Tab::BEntLin => &w.b_ent_lin,
            Tab::MaskPre => &w.mask_pre,
            Tab::XPtr => &w.x_ptr,
            Tab::XBase => &w.x_base,
            Tab::XCr => &w.x_cr,
            Tab::XOut => &w.x_out,
            Tab::XAux => &w.x_aux,
            Tab::XOp(j) => &w.x_op[j],
            Tab::XDigit(j) => &w.x_digit[j],
            Tab::XDigitSlack(j) => &w.x_digit_slack[j],
            Tab::XPow => &w.x_pow,
            Tab::CntC => &w.cnt_c,
            Tab::LuOut => &w.lu_out,
            Tab::IsLookup => &w.is_lookup,
            Tab::CntL => &w.cnt_l,
            Tab::MaxRead => &w.max_read,
            Tab::Diff => &w.diff,
            Tab::DiffPre => &w.diff_pre,
            Tab::RangeCnt => &w.range_cnt,
            Tab::Row => self.sg_row,
            Tab::Col => self.sg_col,
            Tab::Val => self.sg_val,
            Tab::Pow => self.sg_pow,
            Tab::BRow => self.b_row,
            Tab::BCol => self.b_col,
            Tab::BVal => self.b_val,
            Tab::BPow => self.b_pow,
            Tab::Tp => self.tp,
            _ => panic!("{:?} is public, not committed", tab),
        }
    }

    /// The table's values over `[0, len)`, zero-padded exactly the way its
    /// multilinear extension is.
    fn column(&self, tab: Tab, len: usize) -> Vec<F> {
        match tab {
            Tab::One => vec![F::ONE; len],
            Tab::Identity => (0..len).map(|i| F::from(i as u64)).collect(),
            Tab::Ind(lo, hi) => (0..len)
                .map(|i| {
                    if i >= lo && i < hi {
                        F::ONE
                    } else {
                        F::ZERO
                    }
                })
                .collect(),
            _ => {
                let mut values = self.poly(tab).0.clone();
                assert!(
                    values.len() <= len,
                    "table {:?} is longer than the block it feeds",
                    tab
                );
                values.resize(len, F::ZERO);
                values
            }
        }
    }
}

struct VerifierTables<'a, E: Pairing> {
    vk: &'a VerifierKey<E>,
    key: &'a RegistrationKey<E>,
}

impl<'a, E: Pairing> VerifierTables<'a, E> {
    fn commit(&self, tab: Tab) -> &util::kzg::MkzgCommit<E> {
        let c = &self.key.commits;
        let sg = &self.vk.sparse_commits.supergroups[0];
        match tab {
            Tab::Sel(j) => &c.sel[j],
            Tab::Opn(j) => &c.opn[j],
            Tab::Ctr(j) => &c.ctr[j],
            Tab::DCtr(j) => &c.dctr[j],
            Tab::Base(f) => &c.base[f],
            Tab::Mult(f) => &c.mult[f],
            Tab::RowOut => &c.row_out,
            Tab::RowPtr => &c.row_ptr,
            Tab::RowCr => &c.row_cr,
            Tab::RowDcr => &c.row_dcr,
            Tab::IsLin => &c.is_lin,
            Tab::RowSlackLo => &c.row_slack_lo,
            Tab::RowSlackHi => &c.row_slack_hi,
            Tab::BCnt => &c.b_cnt,
            Tab::EntLin => &c.ent_lin,
            Tab::MaskA => &c.mask_a,
            Tab::MaskB => &c.mask_b,
            Tab::MaskC => &c.mask_c,
            Tab::BEntLin => &c.b_ent_lin,
            Tab::MaskPre => &c.mask_pre,
            Tab::XPtr => &c.x_ptr,
            Tab::XBase => &c.x_base,
            Tab::XCr => &c.x_cr,
            Tab::XOut => &c.x_out,
            Tab::XAux => &c.x_aux,
            Tab::XOp(j) => &c.x_op[j],
            Tab::XDigit(j) => &c.x_digit[j],
            Tab::XDigitSlack(j) => &c.x_digit_slack[j],
            Tab::XPow => &c.x_pow,
            Tab::CntC => &c.cnt_c,
            Tab::LuOut => &c.lu_out,
            Tab::IsLookup => &c.is_lookup,
            Tab::CntL => &c.cnt_l,
            Tab::MaxRead => &c.max_read,
            Tab::Diff => &c.diff,
            Tab::DiffPre => &c.diff_pre,
            Tab::RangeCnt => &c.range_cnt,
            Tab::Row => &sg.row,
            Tab::Col => &sg.col,
            Tab::Val => &sg.val,
            Tab::Pow => &sg.pow,
            Tab::BRow => &self.vk.sparse_commits.b_pre.row,
            Tab::BCol => &self.vk.sparse_commits.b_pre.col,
            Tab::BVal => &self.vk.sparse_commits.b_pre.val,
            Tab::BPow => &self.vk.sparse_commits.b_pre.pow,
            Tab::Tp => &self.vk.tp_commit,
            _ => panic!("{:?} is public, not committed", tab),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{cmp, panic::AssertUnwindSafe};

    use ark_bn254::{Bn254, Fr};
    use rand::{rngs::StdRng, SeedableRng};
    use util::kzg::Mkzg;

    use super::*;
    use crate::{
        circuit::{Circuit, LookupType, QUOTIENT_BOUND},
        instruction::Instruction,
        preprocess::Preprocessor,
        program::Program,
    };

    const DIVISOR: i64 = 64;

    fn table() -> Vec<(Fr, Fr, Fr)> {
        let mut rows = (0i64..DIVISOR)
            .map(|i| {
                (
                    Fr::from(0),
                    Fr::from(i),
                    Fr::from(LookupType::Range(DIVISOR).tag()),
                )
            })
            .collect::<Vec<_>>();
        rows.extend(
            (-64i64..=64).map(|i| (Fr::from(i), Fr::from(cmp::max(0, i)), Fr::from(2u64))),
        );
        rows.extend((-64i64..=64).map(|i| {
            (
                Fr::from(0),
                Fr::from(i),
                Fr::from(LookupType::Bound(QUOTIENT_BOUND).tag()),
            )
        }));
        rows
    }

    /// AddMult, then Div, then a ReLU lookup.
    fn fixture() -> (Program<Fr>, ProgramEncoding, Circuit<Fr>) {
        let weights = vec![1, 3];
        let input = vec![5];
        let instructions = vec![
            Instruction::AddMult {
                input1: vec![(2, 1)],
                input2: vec![(1, 1)],
            },
            Instruction::Div {
                input1: vec![(3, 1)],
                input2: vec![(0, 1)],
                divisor: DIVISOR,
            },
            Instruction::Lookup {
                input: vec![(4, 1)],
                tp: LookupType::Relu,
            },
        ];
        let program = Program::<Fr>::new(instructions, weights);
        let aux_start = program.execute(input.clone()).len();
        let circuit = program.compile(input.len(), aux_start, table()).unwrap();
        let enc = ProgramEncoding::from_program(&program, input.len(), aux_start);
        (program, enc, circuit)
    }

    fn setup(circuit: Circuit<Fr>, seed: u64) -> (crate::preprocess::ProvingKey<Bn254>, VerifierKey<Bn254>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let (kzg_pp, kzg_vp) = Mkzg::<Bn254>::gen_srs(5, &mut rng);
        Preprocessor::build(kzg_pp, kzg_vp, circuit)
    }

    #[test]
    fn registration_round_trips() {
        let (_program, enc, circuit) = fixture();
        let (pk, vk) = setup(circuit, 101);
        let mut rng = StdRng::seed_from_u64(7);
        let mut ro = util::util::RandomOracle::new(&mut rng);
        let (key, proof) = Registrar::register(&pk, &vk, &enc, &mut ro).unwrap();
        println!("registration proof size {} bytes", proof.size());
        Registrar::verify(&vk, &key, proof, &mut ro);
    }

    /// A matrix multiplication followed by a requantizing lookup: the shape
    /// every fully connected layer compiles to.
    fn matmult_fixture() -> (ProgramEncoding, Circuit<Fr>) {
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
            Instruction::Div {
                input1: vec![(weights.len() + input.len(), 1)],
                input2: vec![(0, 1)],
                divisor: DIVISOR,
            },
            Instruction::Lookup {
                input: vec![(weights.len() + input.len() + 2, 1)],
                tp: LookupType::Relu,
            },
        ];
        let program = Program::<Fr>::new(instructions, weights);
        let aux_start = program.execute(input.clone()).len();
        let circuit = program.compile(input.len(), aux_start, table()).unwrap();
        let enc = ProgramEncoding::from_program(&program, input.len(), aux_start);
        (enc, circuit)
    }

    /// A two-channel convolution followed by ReLU lookups.
    fn conv_fixture() -> (ProgramEncoding, Circuit<Fr>) {
        let raw_weights = vec![1, 2, 3, 4, 5, 6, 7, 8, 1, -1, 2, -2, 3, -3, 4, -4];
        let input = vec![1, 2, 3, 4, -1, 0, 2, 1];
        let mut weights = vec![1];
        weights.extend(raw_weights.iter().copied());
        let conv_out = weights.len() + input.len();
        let instructions = vec![
            Instruction::Conv {
                n: 2,
                m: 2,
                in_channels: 2,
                out_channels: 2,
                start1: weights.len(),
                start2: 1,
            },
            Instruction::Lookup {
                input: vec![(conv_out, 1)],
                tp: LookupType::Relu,
            },
            Instruction::Lookup {
                input: vec![(conv_out + 1, 1)],
                tp: LookupType::Relu,
            },
        ];
        let program = Program::<Fr>::new(instructions, weights);
        let aux_start = program.execute(input.clone()).len();
        let circuit = program.compile(input.len(), aux_start, table()).unwrap();
        let enc = ProgramEncoding::from_program(&program, input.len(), aux_start);
        (enc, circuit)
    }

    #[test]
    fn registration_round_trips_with_matmult() {
        let (enc, circuit) = matmult_fixture();
        let (pk, vk) = setup(circuit, 111);
        let mut rng = StdRng::seed_from_u64(19);
        let mut ro = util::util::RandomOracle::new(&mut rng);
        let (key, proof) = Registrar::register(&pk, &vk, &enc, &mut ro).unwrap();
        println!("matmult registration proof size {} bytes", proof.size());
        Registrar::verify(&vk, &key, proof, &mut ro);
    }

    #[test]
    fn registration_round_trips_with_conv() {
        let (enc, circuit) = conv_fixture();
        let (pk, vk) = setup(circuit, 113);
        let mut rng = StdRng::seed_from_u64(23);
        let mut ro = util::util::RandomOracle::new(&mut rng);
        let (key, proof) = Registrar::register(&pk, &vk, &enc, &mut ro).unwrap();
        println!("conv registration proof size {} bytes", proof.size());
        Registrar::verify(&vk, &key, proof, &mut ro);
    }

    /// Reading the wrong input cell in a MatMult row cannot be certified, even
    /// though the circuit stays perfectly satisfiable.
    #[test]
    fn registration_rejects_shifted_matmult_operand() {
        let (enc, mut circuit) = matmult_fixture();
        circuit.a.rows[enc.cr[0]].elems[0].0 += 1;
        let (pk, vk) = setup(circuit, 115);
        let profile = RegistrationProfile::new(&vk, &enc);
        let witness = ProgramWitness::build(&enc, &pk.circuit, &profile);
        let key = RegistrationKey {
            profile,
            commits: witness.commit(&pk.kzg_pp),
        };
        let mut rng = StdRng::seed_from_u64(29);
        let mut ro = util::util::RandomOracle::new(&mut rng);
        let failed = std::panic::catch_unwind(AssertUnwindSafe(|| {
            Registrar::prove(&pk, &key, &witness, &mut ro);
        }));
        assert!(failed.is_err(), "prover certified a shifted operand");
    }

    /// Nor can a convolution whose kernel is read in the wrong order, which is
    /// what separates a convolution from a correlation.
    #[test]
    fn registration_rejects_reordered_conv_kernel() {
        let (enc, mut circuit) = conv_fixture();
        let row = &mut circuit.b.rows[enc.cr[0]];
        let first = row.elems[0].2;
        row.elems[0].2 = row.elems[1].2;
        row.elems[1].2 = first;
        let (pk, vk) = setup(circuit, 117);
        let profile = RegistrationProfile::new(&vk, &enc);
        let witness = ProgramWitness::build(&enc, &pk.circuit, &profile);
        let key = RegistrationKey {
            profile,
            commits: witness.commit(&pk.kzg_pp),
        };
        let mut rng = StdRng::seed_from_u64(31);
        let mut ro = util::util::RandomOracle::new(&mut rng);
        let failed = std::panic::catch_unwind(AssertUnwindSafe(|| {
            Registrar::prove(&pk, &key, &witness, &mut ro);
        }));
        assert!(failed.is_err(), "prover certified a reordered kernel");
    }

    /// A circuit whose output cell was moved cannot be registered: the rule
    /// set fails before a certificate exists.
    #[test]
    fn registration_rejects_relabelled_output_cell() {
        let (_program, enc, mut circuit) = fixture();
        circuit.c.rows[enc.cr[0]].elems[0].0 += 1;
        let (pk, vk) = setup(circuit, 105);
        let mut rng = StdRng::seed_from_u64(11);
        let mut ro = util::util::RandomOracle::new(&mut rng);

        // The clear-text rules reject it, ...
        let err = match Registrar::register(&pk, &vk, &enc, &mut ro) {
            Ok(_) => panic!("a relabelled output cell should not be registrable"),
            Err(err) => err,
        };
        assert!(matches!(err, RegistrationError::NotCompiled(_)), "{}", err);

        // ... and so does the argument itself, with the compilation check
        // skipped so the prover has to try.
        let profile = RegistrationProfile::new(&vk, &enc);
        let witness = ProgramWitness::build(&enc, &pk.circuit, &profile);
        let key = RegistrationKey {
            profile,
            commits: witness.commit(&pk.kzg_pp),
        };
        let failed = std::panic::catch_unwind(AssertUnwindSafe(|| {
            Registrar::prove(&pk, &key, &witness, &mut ro);
        }));
        assert!(failed.is_err(), "prover produced a certificate anyway");
    }

    #[test]
    fn registration_rejects_swapped_lookup_type() {
        let (_program, enc, mut circuit) = fixture();
        circuit.tp[enc.lr[1] + 1] = Fr::from(LookupType::Relu.tag());
        let (pk, vk) = setup(circuit, 107);
        let profile = RegistrationProfile::new(&vk, &enc);
        let witness = ProgramWitness::build(&enc, &pk.circuit, &profile);
        let key = RegistrationKey {
            profile,
            commits: witness.commit(&pk.kzg_pp),
        };
        let mut rng = StdRng::seed_from_u64(13);
        let mut ro = util::util::RandomOracle::new(&mut rng);
        let failed = std::panic::catch_unwind(AssertUnwindSafe(|| {
            Registrar::prove(&pk, &key, &witness, &mut ro);
        }));
        assert!(failed.is_err(), "prover certified the wrong lookup type");
    }

    /// The certificate and the inference proofs are checked against the same
    /// verifier key, which is what ties a registered program to the outputs it
    /// later proves.
    #[test]
    fn registration_and_inference_share_a_verifier_key() {
        use ark_ff::UniformRand;

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
            Instruction::Div {
                input1: vec![(weights.len() + input.len(), 1)],
                input2: vec![(0, 1)],
                divisor: DIVISOR,
            },
            Instruction::Lookup {
                input: vec![(weights.len() + input.len() + 2, 1)],
                tp: LookupType::Relu,
            },
        ];
        let program = Program::<Fr>::new(instructions, weights);
        let trace = program.execute(input.clone());
        let aux_start = trace.len();
        let circuit = program.compile(input.len(), aux_start, table()).unwrap();
        let enc = ProgramEncoding::from_program(&program, input.len(), aux_start);
        let (pk, vk) = setup(circuit, 119);

        let mut rng = StdRng::seed_from_u64(37);
        let mut reg_ro = util::util::RandomOracle::new(&mut rng);
        let (key, cert) = Registrar::register(&pk, &vk, &enc, &mut reg_ro).unwrap();
        Registrar::verify(&vk, &key, cert, &mut reg_ro);

        // The same key then verifies an inference against the registered
        // circuit.
        let gamma = Fr::rand(&mut rng);
        let z = program.gen_z(enc.weight_len + enc.input_len, trace, gamma);
        let mut ro = util::util::RandomOracle::new(&mut rng);
        let proof = crate::prover::Prover::new(pk).prove(z, gamma, &mut ro);
        crate::verifier::Verifier::new(vk).verify(proof, gamma, &mut ro);
    }

    /// A certificate for one circuit must not verify against another.
    #[test]
    fn registration_rejects_foreign_circuit() {
        let (_program, enc, circuit) = fixture();
        let (pk, vk) = setup(circuit.clone(), 109);
        let mut rng = StdRng::seed_from_u64(17);
        let mut ro = util::util::RandomOracle::new(&mut rng);
        let (key, proof) = Registrar::register(&pk, &vk, &enc, &mut ro).unwrap();

        let mut other = circuit;
        other.a.rows[enc.cr[0]].elems[0].1 = Fr::from(9u64);
        let (_pk2, vk2) = setup(other, 109);
        let failed = std::panic::catch_unwind(AssertUnwindSafe(|| {
            Registrar::verify(&vk2, &key, proof, &mut ro);
        }));
        assert!(failed.is_err(), "certificate verified against a foreign circuit");
    }
}
