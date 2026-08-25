//! Sub-arguments the registration certificate is built from.
//!
//! Every rule of the compilation relation reduces to one of two shapes:
//!
//! * a *zero check*, that some low-degree expression over committed vectors
//!   vanishes on the hypercube, proved with an `eq`-weighted sumcheck; and
//! * a *fractional sum*, that two multisets agree, proved with the logup
//!   identity `sum_i num_i / (alpha + fp_i)` over the GKR product circuit that
//!   [`crate::sparse`] already uses for its indexed lookups.
//!
//! Both leave behind evaluation claims on the underlying committed vectors,
//! which [`ClaimAcc`] collects and a single batched multilinear KZG opening
//! discharges. Fingerprints are kept *affine* in the committed vectors on
//! purpose: an affine expression's multilinear extension is the same affine
//! combination of the parts' extensions, so the verifier can rebuild the
//! fractional-sum leaf claims without another sumcheck.

use ark_ec::pairing::Pairing;
use ark_ff::{AdditiveGroup, Field, PrimeField};

use util::{
    kzg::{Mkzg, MkzgCommit, MkzgProof, MkzgProveParams, MkzgVerParams, SumcheckProof},
    poly::MlPoly,
    util::{batch_inverse, Proof, RandomOracle},
};

use crate::sparse::{gkr_fractional_sum_prove, gkr_fractional_sum_verify, verifier_sumcheck};

/// Rounds `n` up to a power of two, with a floor of 2 because the fractional
/// sum circuit needs at least one layer.
pub(crate) fn pad_len(n: usize) -> usize {
    let mut len = 2usize;
    while len < n {
        len <<= 1;
    }
    len
}

pub(crate) fn pad_to<F: Field>(mut v: Vec<F>, len: usize, filler: F) -> Vec<F> {
    assert!(v.len() <= len);
    v.resize(len, filler);
    v
}

/// Proves `sum_x eq(eq_point, x) * g(tables(x)) = claim`, where `degree` is the
/// total degree of the summand, `eq` included.
///
/// Returns the sumcheck challenge point and the value of every table there;
/// both are also written to the transcript for the verifier.
pub(crate) fn eq_sumcheck_prove<E: Pairing, G>(
    eq_point: &[E::ScalarField],
    tables: Vec<Vec<E::ScalarField>>,
    degree: usize,
    g: G,
    proof: &mut Proof<E>,
    ro: &mut RandomOracle<E::ScalarField>,
) -> (Vec<E::ScalarField>, Vec<E::ScalarField>)
where
    G: Fn(&[E::ScalarField]) -> E::ScalarField,
{
    let nv = eq_point.len();
    let len = 1usize << nv;
    let mut folded: Vec<Vec<E::ScalarField>> = Vec::with_capacity(tables.len() + 1);
    folded.push(MlPoly::new_eq(&eq_point.to_vec()).0);
    for table in tables {
        assert_eq!(table.len(), len, "table length must match the eq point");
        folded.push(table);
    }

    let mut point = Vec::with_capacity(nv);
    let mut scratch = vec![E::ScalarField::ZERO; folded.len()];
    for _ in 0..nv {
        let m = folded[0].len();
        let mut sums = vec![E::ScalarField::ZERO; degree + 1];
        for pair in (0..m).step_by(2) {
            for (e, sum) in sums.iter_mut().enumerate() {
                let step = E::ScalarField::from(e as u64);
                for (slot, table) in folded.iter().enumerate() {
                    let lo = table[pair];
                    scratch[slot] = lo + (table[pair + 1] - lo) * step;
                }
                *sum += scratch[0] * g(&scratch[1..]);
            }
        }
        proof.push_f(&sums);
        let challenge = ro.next_field();
        point.push(challenge);
        for table in folded.iter_mut() {
            for i in 0..m / 2 {
                table[i] = table[i * 2] + (table[i * 2 + 1] - table[i * 2]) * challenge;
            }
            table.truncate(m / 2);
        }
    }

    let values: Vec<E::ScalarField> = folded[1..].iter().map(|t| t[0]).collect();
    proof.push_f(&values);
    (point, values)
}

/// Verifier half of [`eq_sumcheck_prove`].
pub(crate) fn eq_sumcheck_verify<E: Pairing, G>(
    eq_point: &[E::ScalarField],
    claim: E::ScalarField,
    degree: usize,
    num_tables: usize,
    g: G,
    proof: &mut Proof<E>,
    ro: &mut RandomOracle<E::ScalarField>,
) -> (Vec<E::ScalarField>, Vec<E::ScalarField>)
where
    G: Fn(&[E::ScalarField]) -> E::ScalarField,
{
    let nv = eq_point.len();
    let (point, y) = verifier_sumcheck::<E>(claim, nv, degree, proof, ro);
    let values = proof.next_n_fs(num_tables);
    let eq = MlPoly::<E::ScalarField>::eval_eq(&eq_point.to_vec(), &point);
    assert_eq!(y, eq * g(&values), "eq-sumcheck final check failed");
    (point, values)
}

/// Proves `sum_i num[i] / den[i]`, writing the sum to the transcript.
///
/// Returns the sum, the point the fractional sum reduces to, and the leaf
/// evaluations `num~(point)`, `den~(point)` there.
pub(crate) fn frac_sum_prove<E: Pairing>(
    num: Vec<E::ScalarField>,
    den: Vec<E::ScalarField>,
    proof: &mut Proof<E>,
    ro: &mut RandomOracle<E::ScalarField>,
) -> (
    E::ScalarField,
    Vec<E::ScalarField>,
    E::ScalarField,
    E::ScalarField,
) {
    assert_eq!(num.len(), den.len());
    assert!(num.len().is_power_of_two());
    let mut inv = den.clone();
    batch_inverse(&mut inv);
    let sum = num
        .iter()
        .zip(inv.iter())
        .fold(E::ScalarField::ZERO, |acc, (n, i)| acc + *n * *i);
    proof.push_f(&[sum]);
    let point = gkr_fractional_sum_prove::<E>(num.clone(), den.clone(), sum, proof, ro);
    let num_eval = MlPoly::new(num).eval(&point);
    let den_eval = MlPoly::new(den).eval(&point);
    (sum, point, num_eval, den_eval)
}

/// Verifier half of [`frac_sum_prove`].
pub(crate) fn frac_sum_verify<E: Pairing>(
    log_len: usize,
    proof: &mut Proof<E>,
    ro: &mut RandomOracle<E::ScalarField>,
) -> (
    E::ScalarField,
    Vec<E::ScalarField>,
    E::ScalarField,
    E::ScalarField,
) {
    let sum = proof.next_f();
    let (point, num_eval, den_eval) = gkr_fractional_sum_verify::<E>(sum, log_len, proof, ro);
    (sum, point, num_eval, den_eval)
}

/// Evaluation claims gathered during a proof, discharged by one batched KZG
/// opening. The prover and verifier build the same list in the same order.
pub(crate) struct ProverClaims<F: PrimeField> {
    polys: Vec<MlPoly<F>>,
    points: Vec<Vec<F>>,
    values: Vec<F>,
}

impl<F: PrimeField> ProverClaims<F> {
    pub(crate) fn new() -> Self {
        ProverClaims {
            polys: vec![],
            points: vec![],
            values: vec![],
        }
    }

    pub(crate) fn push(&mut self, poly: &MlPoly<F>, point: Vec<F>, value: F) {
        debug_assert_eq!(poly.clone().eval(&point), value, "claim does not hold");
        self.polys.push(poly.clone());
        self.points.push(point);
        self.values.push(value);
    }

    pub(crate) fn open<E: Pairing<ScalarField = F>>(
        self,
        kzg_pp: &MkzgProveParams<E>,
        proof: &mut Proof<E>,
        ro: &mut RandomOracle<F>,
    ) {
        let (kzg_proof, sumcheck_proof) =
            Mkzg::<E>::batch_open(kzg_pp, &self.polys, &self.points, ro);
        proof.push_u(kzg_proof.0.len());
        proof.push_gs(&kzg_proof.0);
        proof.push_u(sumcheck_proof.0.len());
        proof.push_f(&sumcheck_proof.0);
    }
}

pub(crate) struct VerifierClaims<E: Pairing> {
    commits: Vec<MkzgCommit<E>>,
    points: Vec<Vec<E::ScalarField>>,
    values: Vec<E::ScalarField>,
}

impl<E: Pairing> VerifierClaims<E> {
    pub(crate) fn new() -> Self {
        VerifierClaims {
            commits: vec![],
            points: vec![],
            values: vec![],
        }
    }

    pub(crate) fn push(
        &mut self,
        commit: &MkzgCommit<E>,
        point: Vec<E::ScalarField>,
        value: E::ScalarField,
    ) {
        self.commits.push(commit.clone());
        self.points.push(point);
        self.values.push(value);
    }

    pub(crate) fn verify(
        self,
        kzg_vp: &MkzgVerParams<E>,
        proof: &mut Proof<E>,
        ro: &mut RandomOracle<E::ScalarField>,
    ) -> bool {
        let kzg_proof_len = proof.next_u();
        let kzg_proof = MkzgProof(proof.next_n_gs(kzg_proof_len));
        let sumcheck_len = proof.next_u();
        let sumcheck_proof = SumcheckProof(proof.next_n_fs(sumcheck_len));
        Mkzg::batch_verify(
            kzg_vp,
            &self.points,
            &self.commits,
            &self.values,
            (kzg_proof, sumcheck_proof),
            ro,
        )
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::{Bn254, Fr};
    use ark_ff::UniformRand;
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;

    fn rand_vec(n: usize, rng: &mut StdRng) -> Vec<Fr> {
        (0..n).map(|_| Fr::rand(rng)).collect()
    }

    #[test]
    fn eq_sumcheck_round_trips() {
        let mut rng = StdRng::seed_from_u64(1);
        let nv = 4;
        let len = 1 << nv;
        let a = rand_vec(len, &mut rng);
        let b = rand_vec(len, &mut rng);
        let eq_point = rand_vec(nv, &mut rng);

        // sum_x eq(r, x) * (a(x) * b(x) - a(x)) = the same expression's MLE at r
        let g = |v: &[Fr]| v[0] * v[1] - v[0];
        let eq = MlPoly::new_eq(&eq_point).0;
        let claim = (0..len).fold(Fr::from(0u64), |acc, i| acc + eq[i] * g(&[a[i], b[i]]));

        let mut proof = Proof::<Bn254>::new();
        let mut ro = RandomOracle::new(&mut rng);
        let (point, values) =
            eq_sumcheck_prove::<Bn254, _>(&eq_point, vec![a.clone(), b.clone()], 3, g, &mut proof, &mut ro);
        assert_eq!(values[0], MlPoly::new(a).eval(&point));
        assert_eq!(values[1], MlPoly::new(b).eval(&point));

        ro.restart();
        let (point_v, values_v) =
            eq_sumcheck_verify::<Bn254, _>(&eq_point, claim, 3, 2, g, &mut proof, &mut ro);
        assert_eq!(point_v, point);
        assert_eq!(values_v, values);
    }

    #[test]
    #[should_panic]
    fn eq_sumcheck_rejects_wrong_claim() {
        let mut rng = StdRng::seed_from_u64(2);
        let nv = 3;
        let len = 1 << nv;
        let a = rand_vec(len, &mut rng);
        let eq_point = rand_vec(nv, &mut rng);
        let g = |v: &[Fr]| v[0];

        let mut proof = Proof::<Bn254>::new();
        let mut ro = RandomOracle::new(&mut rng);
        eq_sumcheck_prove::<Bn254, _>(&eq_point, vec![a], 2, g, &mut proof, &mut ro);
        ro.restart();
        eq_sumcheck_verify::<Bn254, _>(&eq_point, Fr::from(7u64), 2, 1, g, &mut proof, &mut ro);
    }

    /// Two multisets agree exactly when their logup sums do, which is the only
    /// property the rule set relies on.
    #[test]
    fn frac_sum_detects_multiset_difference() {
        let mut rng = StdRng::seed_from_u64(3);
        let alpha = Fr::rand(&mut rng);
        let left = vec![1i64, 4, 9, 16];
        let equal_right = vec![9i64, 1, 16, 4];
        let unequal_right = vec![9i64, 1, 16, 5];

        let leaves = |values: &[i64]| {
            let num = vec![Fr::from(1u64); values.len()];
            let den = values.iter().map(|&v| alpha + Fr::from(v)).collect();
            (num, den)
        };

        let sum_of = |values: &[i64]| {
            let (num, den) = leaves(values);
            let mut proof = Proof::<Bn254>::new();
            let mut ro = RandomOracle::new(&mut StdRng::seed_from_u64(4));
            let (sum, _, _, _) = frac_sum_prove::<Bn254>(num, den, &mut proof, &mut ro);
            sum
        };

        assert_eq!(sum_of(&left), sum_of(&equal_right));
        assert_ne!(sum_of(&left), sum_of(&unequal_right));
    }

    #[test]
    fn frac_sum_round_trips() {
        let mut rng = StdRng::seed_from_u64(5);
        let len = 8;
        let num = rand_vec(len, &mut rng);
        let den = rand_vec(len, &mut rng);

        let mut proof = Proof::<Bn254>::new();
        let mut ro = RandomOracle::new(&mut rng);
        let (sum, point, num_eval, den_eval) =
            frac_sum_prove::<Bn254>(num.clone(), den.clone(), &mut proof, &mut ro);

        ro.restart();
        let (sum_v, point_v, num_v, den_v) = frac_sum_verify::<Bn254>(3, &mut proof, &mut ro);
        assert_eq!(sum_v, sum);
        assert_eq!(point_v, point);
        assert_eq!(num_v, num_eval);
        assert_eq!(den_v, den_eval);
        assert_eq!(num_eval, MlPoly::new(num).eval(&point));
        assert_eq!(den_eval, MlPoly::new(den).eval(&point));
    }
}
