use ark_ec::{CurveGroup, VariableBaseMSM, pairing::Pairing};
use ark_ff::{Field, One, UniformRand, Zero};
use rand::Rng;
use std::marker::PhantomData;

use crate::{oracle::RandomOracle, poly::MlPoly};

pub const LOG_CHUNK_NUM: usize = 4;
pub struct Mkzg<E: Pairing>(PhantomData<E>);
#[derive(Debug, Clone)]
pub struct MkzgCommit<E: Pairing>([E::G1; 1 << LOG_CHUNK_NUM]);
#[derive(Debug, Clone)]
pub struct MkzgProof<E: Pairing>(Vec<E::G1>);
pub struct SumcheckProof<F: Field>(Vec<F>);
#[derive(Debug, Clone)]
pub struct MkzgVerParams<E: Pairing> {
    pub g: E::G1,
    pub h: E::G2,
    params: Vec<E::G2>,
}

impl<E: Pairing> MkzgVerParams<E> {
    pub fn trim(&self, log_len: usize) -> Self {
        let mut params = self.params.clone();
        params.truncate(log_len);
        MkzgVerParams {
            g: self.g,
            h: self.h,
            params,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MkzgProveParams<E: Pairing>(Vec<E::G1Affine>);
impl<E: Pairing> MkzgProveParams<E> {
    pub fn trim(&self, log_len: usize) -> Self {
        let mut params = self.0.clone();
        params.truncate(1 << log_len);
        MkzgProveParams(params)
    }
}

impl<E: Pairing> Mkzg<E> {
    pub fn gen_srs(len: usize, rng: &mut impl Rng) -> (MkzgProveParams<E>, MkzgVerParams<E>) {
        let g = <E::G1 as UniformRand>::rand(rng);
        let h = <E::G2 as UniformRand>::rand(rng);
        let tau = (0..len)
            .map(|_| E::ScalarField::rand(rng))
            .collect::<Vec<_>>();
        let vp = MkzgVerParams {
            g,
            h,
            params: (0..len).map(|x| h * tau[x]).collect(),
        };
        let mut power_of_g = vec![E::ScalarField::one()];
        for i in 0..len {
            let cur_len = power_of_g.len();
            for j in 0..cur_len {
                power_of_g.push(power_of_g[j] * tau[i]);
            }
        }
        (
            MkzgProveParams(<E::G1 as CurveGroup>::normalize_batch(
                &power_of_g.iter().map(|x| g * x).collect::<Vec<_>>(),
            )),
            vp,
        )
    }

    pub fn commit(srs: &MkzgProveParams<E>, poly: &MlPoly<E::ScalarField>) -> MkzgCommit<E> {
        if srs.0.len() != poly.0.len() >> LOG_CHUNK_NUM {
            panic!("{} {}", file!(), line!());
        }
        let polies = poly.split(1 << LOG_CHUNK_NUM);
        let mut commits = vec![];
        for p in polies {
            commits.push(E::G1::msm_unchecked(&srs.0, &p.0));
        }
        MkzgCommit(commits.try_into().unwrap())
    }

    pub fn open(
        srs: &MkzgProveParams<E>,
        mut poly: MlPoly<E::ScalarField>,
        mut point: Vec<E::ScalarField>,
    ) -> (MkzgProof<E>, E::ScalarField) {
        assert_eq!(srs.0.len() << LOG_CHUNK_NUM, 1 << point.len());
        poly.fold(&point[0..LOG_CHUNK_NUM]);
        point = point[LOG_CHUNK_NUM..].to_vec();

        let one = E::ScalarField::one();
        let mul = point
            .iter()
            .fold(E::ScalarField::from(1), |acc, x| acc * (one - x));
        let point = point
            .iter()
            .map(|&x| x * Field::inverse(&(one - x)).unwrap())
            .collect::<Vec<_>>();
        let mut proofs = vec![];
        let mut cur_len = poly.0.len() >> 1;
        let mut poly = poly.0;
        assert_eq!(poly.len(), srs.0.len());
        let mut bases = srs.0.clone();
        for p in point.iter() {
            let mut scalars = vec![];
            for i in 0..cur_len {
                bases[i] = bases[i * 2];
                scalars.push(poly[i * 2 + 1]);
            }
            bases.truncate(cur_len);
            proofs.push(E::G1::msm_unchecked(&bases, &scalars));
            for i in 0..cur_len {
                poly[i] = poly[i * 2] + poly[i * 2 + 1] * p;
            }
            poly.truncate(cur_len);
            cur_len >>= 1;
        }
        (MkzgProof(proofs), poly[0] * mul)
    }

    pub fn batch_open(
        srs: &MkzgProveParams<E>,
        poly: MlPoly<E::ScalarField>,
        point: Vec<Vec<E::ScalarField>>,
        oracle: &mut RandomOracle<E::ScalarField>,
    ) -> (MkzgProof<E>, SumcheckProof<E::ScalarField>) {
        let mut sumcheck_proof = vec![];
        let r = oracle.next_field();
        let nv = poly.0.len().ilog2() as usize;
        let mut eq_acc = vec![E::ScalarField::zero(); 1 << nv];
        for i in point.iter() {
            let eq = MlPoly::new_eq(i);
            for (j, k) in eq_acc.iter_mut().zip(eq.0.into_iter()) {
                *j *= r;
                *j += k
            }
        }
        let mut r_point = vec![];

        let mut poly_evals = poly.0.clone();
        for i in 0..nv {
            let m = 1usize << (nv - i);
            let sums =
                (0..m)
                    .step_by(2)
                    .fold([<E::ScalarField as Zero>::zero(); 3], |mut acc, x| {
                        let v00 = poly_evals[x];
                        let v01 = poly_evals[x + 1];
                        let v02 = v01 + v01 - v00;
                        let v10 = eq_acc[x];
                        let v11 = eq_acc[x + 1];
                        let v12 = v11 + v11 - v10;
                        acc[0] += v00 * v10;
                        acc[1] += v01 * v11;
                        acc[2] += v02 * v12;
                        acc
                    });
            for j in 0..3 {
                sumcheck_proof.push(sums[j]);
            }
            let challenge = oracle.next_field();
            r_point.push(challenge);
            let m = m >> 1;
            for j in 0..m {
                poly_evals[j] =
                    poly_evals[j * 2] + (poly_evals[j * 2 + 1] - poly_evals[j * 2]) * challenge;
                eq_acc[j] = eq_acc[j * 2] + (eq_acc[j * 2 + 1] - eq_acc[j * 2]) * challenge;
            }
            poly_evals.truncate(m);
            eq_acc.truncate(m);
        }
        let (proof, _) = Self::open(srs, poly, r_point);
        (proof, SumcheckProof(sumcheck_proof))
    }

    pub fn batch_verify(
        vp: &MkzgVerParams<E>,
        point: Vec<Vec<E::ScalarField>>,
        comm: &MkzgCommit<E>,
        value: Vec<E::ScalarField>,
        proof: (MkzgProof<E>, SumcheckProof<E::ScalarField>),
        oracle: &mut RandomOracle<E::ScalarField>,
    ) -> bool {
        let nv = point[0].len();
        let r = oracle.next_field();
        let mut y = E::ScalarField::zero();
        for i in value.into_iter() {
            y *= r;
            y += i;
        }
        let mut r_point = vec![];
        let (kzg_proof, sumcheck_proof) = proof;
        let one_over_two = E::ScalarField::from(2).inverse().unwrap();
        let three_over_two = one_over_two * E::ScalarField::from(3);
        for i in 0..nv {
            let sum = [
                sumcheck_proof.0[i * 3],
                sumcheck_proof.0[i * 3 + 1],
                sumcheck_proof.0[i * 3 + 2],
            ];
            assert_eq!(sum[0] + sum[1], y);

            let challenge = oracle.next_field();
            r_point.push(challenge);
            y = sum[0]
                + challenge * (-three_over_two * sum[0] + sum[1] + sum[1] - one_over_two * sum[2])
                + challenge * challenge * (one_over_two * sum[0] - sum[1] + one_over_two * sum[2]);
        }
        let mut eq_r = E::ScalarField::zero();
        for i in point.into_iter() {
            eq_r *= r;
            eq_r += MlPoly::eval_eq(&i, &r_point);
        }
        Self::verify(vp, r_point, comm, y * eq_r.inverse().unwrap(), kzg_proof)
    }

    pub fn verify(
        vp: &MkzgVerParams<E>,
        mut point: Vec<E::ScalarField>,
        comm: &MkzgCommit<E>,
        value: E::ScalarField,
        proof: MkzgProof<E>,
    ) -> bool {
        let point2 = point.split_off(LOG_CHUNK_NUM);
        let one = E::ScalarField::one();
        let mul = point2
            .iter()
            .fold(E::ScalarField::from(1), |acc, x| acc * (one - x));
        let point2 = point2
            .iter()
            .map(|&x| x * Field::inverse(&(one - x)).unwrap())
            .collect::<Vec<_>>();
        let value = value * mul.inverse().unwrap();
        let MkzgVerParams { g, h, mut params } = vp.clone();
        for i in 0..point2.len() {
            params[i] -= h * point2[i];
        }
        let mut proof = proof.0;
        proof.push(g);
        params.push(h * value);

        let mut commits = comm.0;
        let mut cur_len = commits.len() >> 1;
        for r in point.iter() {
            for i in 0..cur_len {
                commits[i] = commits[i * 2] + (commits[i * 2 + 1] - commits[i * 2]) * (*r);
            }
            cur_len >>= 1;
        }

        E::multi_pairing(proof, params) == E::pairing(commits[0], h)
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::{Bn254, Fr};
    use ark_ff::UniformRand;
    use rand::thread_rng;

    use crate::{
        kzg::{LOG_CHUNK_NUM, Mkzg},
        oracle::RandomOracle,
        poly::MlPoly,
    };

    #[test]
    fn it_works() {
        let log_len = 8;
        let mut rng = thread_rng();
        let poly = MlPoly(
            (0..1 << log_len)
                .map(|_| <Fr as UniformRand>::rand(&mut rng))
                .collect(),
        );
        let (pp, vp) = Mkzg::<Bn254>::gen_srs(10, &mut rng);
        let pp = pp.trim(log_len - LOG_CHUNK_NUM);
        let vp = vp.trim(log_len - LOG_CHUNK_NUM);
        let commit = Mkzg::commit(&pp, &poly);
        let point = (0..10)
            .map(|_| {
                (0..log_len)
                    .map(|_| <Fr as UniformRand>::rand(&mut rng))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut random_oracle = RandomOracle::new(&mut rng);
        let proof = Mkzg::batch_open(&pp, poly.clone(), point.clone(), &mut random_oracle);

        random_oracle.restart();
        assert!(Mkzg::batch_verify(
            &vp,
            point.clone(),
            &commit,
            point.iter().map(|p| poly.clone().eval(p)).collect(),
            proof,
            &mut random_oracle
        ));
    }
}
