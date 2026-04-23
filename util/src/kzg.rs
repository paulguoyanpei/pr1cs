use ark_ec::{CurveGroup, VariableBaseMSM, pairing::Pairing};
use ark_ff::{Field, One, UniformRand, Zero};
use rand::Rng;
use std::marker::PhantomData;

use crate::{poly::MlPoly, util::RandomOracle};

pub const LOG_CHUNK_SIZE: usize = 18;
pub struct Mkzg<E: Pairing>(PhantomData<E>);
#[derive(Debug, Clone)]
pub struct MkzgCommit<E: Pairing>(pub Vec<E::G1>);
#[derive(Debug, Clone)]
pub struct MkzgProof<E: Pairing>(pub Vec<E::G1>);
pub struct SumcheckProof<F: Field>(pub Vec<F>);
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
pub struct MkzgProveParams<E: Pairing>(pub Vec<E::G1Affine>);
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
        let chunk_size = srs.0.len();
        let polies = poly.split(chunk_size);
        let mut commits = vec![];
        for p in polies {
            commits.push(E::G1::msm_unchecked(&srs.0, &p.0));
        }
        MkzgCommit(commits)
    }

    pub fn open(
        srs: &MkzgProveParams<E>,
        mut poly: MlPoly<E::ScalarField>,
        point: Vec<E::ScalarField>,
    ) -> (MkzgProof<E>, E::ScalarField) {
        let chunk_size = srs.0.len();
        let log_chunk_size = chunk_size.ilog2() as usize;

        // First log_chunk_size variables are KZG (within-chunk),
        // remaining variables are chunk selectors (upper bits).
        let kzg_point = point[..log_chunk_size].to_vec();
        let chunk_point = point[log_chunk_size..].to_vec();

        // Fold chunk dimension (strided fold over upper variables)
        for r in chunk_point.iter() {
            let num_chunks = (poly.0.len() + chunk_size - 1) / chunk_size;
            let new_num = (num_chunks + 1) / 2;
            for i in 0..new_num {
                for j in 0..chunk_size {
                    let idx0 = i * 2 * chunk_size + j;
                    let idx1 = (i * 2 + 1) * chunk_size + j;
                    let v0 = if idx0 < poly.0.len() {
                        poly.0[idx0]
                    } else {
                        E::ScalarField::zero()
                    };
                    let v1 = if idx1 < poly.0.len() {
                        poly.0[idx1]
                    } else {
                        E::ScalarField::zero()
                    };
                    poly.0[i * chunk_size + j] = v0 + (v1 - v0) * (*r);
                }
            }
            poly.0.truncate(new_num * chunk_size);
        }

        // Pad to chunk_size if needed
        poly.0.resize(chunk_size, E::ScalarField::zero());

        let one = E::ScalarField::one();
        let mul = kzg_point
            .iter()
            .fold(E::ScalarField::from(1), |acc, x| acc * (one - x));
        let kzg_point = kzg_point
            .iter()
            .map(|&x| x * Field::inverse(&(one - x)).unwrap())
            .collect::<Vec<_>>();
        let mut proofs = vec![];
        let mut cur_len = poly.0.len() >> 1;
        let mut poly = poly.0;
        assert_eq!(poly.len(), srs.0.len());
        let mut bases = srs.0.clone();
        for p in kzg_point.iter() {
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
        polys: &[MlPoly<E::ScalarField>],
        points: &[Vec<E::ScalarField>],
        oracle: &mut RandomOracle<E::ScalarField>,
    ) -> (MkzgProof<E>, SumcheckProof<E::ScalarField>) {
        let k = polys.len();
        assert_eq!(k, points.len());
        assert!(k > 0);
        let chunk_size = srs.0.len();
        let log_chunk_size = chunk_size.ilog2() as usize;

        let mut sumcheck_proof = vec![];
        let r = oracle.next_field();

        // r_powers[i] = r^{k-1-i} (Horner ordering)
        let mut r_powers = vec![E::ScalarField::one(); k];
        {
            let mut rp = E::ScalarField::one();
            for i in (0..k).rev() {
                r_powers[i] = rp;
                rp *= r;
            }
        }

        // For each poly, fold chunk-selector vars (point[log_chunk_size..]) so that
        // the resulting g_i has exactly chunk_size evals and g_i(lower_i) = f_i(point_i),
        // where lower_i = point_i[..log_chunk_size] padded with zeros to log_chunk_size.
        let zero = E::ScalarField::zero();
        let mut folded_polys: Vec<Vec<E::ScalarField>> = Vec::with_capacity(k);
        let mut lower_points: Vec<Vec<E::ScalarField>> = Vec::with_capacity(k);
        for i in 0..k {
            let pt = &points[i];
            let split = pt.len().min(log_chunk_size);
            let mut lower = pt[..split].to_vec();
            lower.resize(log_chunk_size, zero);
            let upper: &[E::ScalarField] = if pt.len() > log_chunk_size {
                &pt[log_chunk_size..]
            } else {
                &[]
            };

            let mut poly = polys[i].0.clone();
            for rv in upper.iter() {
                let num_chunks = (poly.len() + chunk_size - 1) / chunk_size;
                let new_num = (num_chunks + 1) / 2;
                for ii in 0..new_num {
                    for j in 0..chunk_size {
                        let idx0 = ii * 2 * chunk_size + j;
                        let idx1 = (ii * 2 + 1) * chunk_size + j;
                        let v0 = if idx0 < poly.len() { poly[idx0] } else { zero };
                        let v1 = if idx1 < poly.len() { poly[idx1] } else { zero };
                        poly[ii * chunk_size + j] = v0 + (v1 - v0) * (*rv);
                    }
                }
                poly.truncate(new_num * chunk_size);
            }
            poly.resize(chunk_size, zero);

            folded_polys.push(poly);
            lower_points.push(lower);
        }

        // Sumcheck state
        let mut eq_evals: Vec<Vec<E::ScalarField>> =
            (0..k).map(|i| MlPoly::new_eq(&lower_points[i]).0).collect();
        let mut poly_evals_arr: Vec<Vec<E::ScalarField>> = folded_polys.iter().cloned().collect();

        let mut r_point = vec![];
        for round in 0..log_chunk_size {
            let m = 1usize << (log_chunk_size - round);
            let sums =
                (0..m)
                    .step_by(2)
                    .fold([<E::ScalarField as Zero>::zero(); 3], |mut acc, x| {
                        for i in 0..k {
                            let v00 = poly_evals_arr[i][x];
                            let v01 = poly_evals_arr[i][x + 1];
                            let v02 = v01 + v01 - v00;
                            let v10 = eq_evals[i][x];
                            let v11 = eq_evals[i][x + 1];
                            let v12 = v11 + v11 - v10;
                            acc[0] += r_powers[i] * v00 * v10;
                            acc[1] += r_powers[i] * v01 * v11;
                            acc[2] += r_powers[i] * v02 * v12;
                        }
                        acc
                    });
            for j in 0..3 {
                sumcheck_proof.push(sums[j]);
            }
            let challenge = oracle.next_field();
            r_point.push(challenge);
            let m = m >> 1;
            for j in 0..m {
                for i in 0..k {
                    poly_evals_arr[i][j] = poly_evals_arr[i][j * 2]
                        + (poly_evals_arr[i][j * 2 + 1] - poly_evals_arr[i][j * 2]) * challenge;
                    eq_evals[i][j] = eq_evals[i][j * 2]
                        + (eq_evals[i][j * 2 + 1] - eq_evals[i][j * 2]) * challenge;
                }
            }
            for i in 0..k {
                poly_evals_arr[i].truncate(m);
                eq_evals[i].truncate(m);
            }
        }

        // Combined polynomial H(x) = Σ_i w_i * g_i(x), size = chunk_size
        // where w_i = r_powers[i] * eq(lower_i, r_point)
        let mut combined = vec![E::ScalarField::zero(); chunk_size];
        for i in 0..k {
            let w = r_powers[i] * eq_evals[i][0];
            for j in 0..chunk_size {
                combined[j] += w * folded_polys[i][j];
            }
        }

        let (proof, _) = Self::open(srs, MlPoly(combined), r_point);
        (proof, SumcheckProof(sumcheck_proof))
    }

    pub fn batch_verify(
        vp: &MkzgVerParams<E>,
        points: &[Vec<E::ScalarField>],
        comms: &[MkzgCommit<E>],
        values: &[E::ScalarField],
        proof: (MkzgProof<E>, SumcheckProof<E::ScalarField>),
        oracle: &mut RandomOracle<E::ScalarField>,
    ) -> bool {
        let k = points.len();
        assert_eq!(k, comms.len());
        assert_eq!(k, values.len());

        let log_chunk_size = vp.params.len();
        let one = E::ScalarField::one();
        let zero = E::ScalarField::zero();
        let r = oracle.next_field();

        let mut y = E::ScalarField::zero();
        for &v in values.iter() {
            y *= r;
            y += v;
        }

        let mut r_powers = vec![E::ScalarField::one(); k];
        {
            let mut rp = E::ScalarField::one();
            for i in (0..k).rev() {
                r_powers[i] = rp;
                rp *= r;
            }
        }

        let mut r_point = vec![];
        let (kzg_proof, sumcheck_proof) = proof;
        let one_over_two = E::ScalarField::from(2).inverse().unwrap();
        let three_over_two = one_over_two * E::ScalarField::from(3);
        for i in 0..log_chunk_size {
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

        // weights[i] = r_powers[i] * eq(lower_i, r_point)
        // where lower_i = points[i][..log_chunk_size] padded with zeros.
        let weights: Vec<_> = (0..k)
            .map(|i| {
                let split = points[i].len().min(log_chunk_size);
                let mut lower = points[i][..split].to_vec();
                lower.resize(log_chunk_size, zero);
                r_powers[i] * MlPoly::eval_eq(&lower, &r_point)
            })
            .collect();

        // For each poly, fold comm_i by upper_i (chunk-selector) to a single G1,
        // then combine with weights[i] into a single-chunk commitment.
        let mut combined: E::G1 = Zero::zero();
        for i in 0..k {
            let upper: &[E::ScalarField] = if points[i].len() > log_chunk_size {
                &points[i][log_chunk_size..]
            } else {
                &[]
            };
            let mut cs = comms[i].0.clone();
            for rv in upper.iter() {
                let cur_len = cs.len();
                let new_len = (cur_len + 1) / 2;
                for ii in 0..new_len {
                    if ii * 2 + 1 < cur_len {
                        cs[ii] = cs[ii * 2] + (cs[ii * 2 + 1] - cs[ii * 2]) * (*rv);
                    } else {
                        cs[ii] = cs[ii * 2] * (one - *rv);
                    }
                }
                cs.truncate(new_len);
            }
            if !cs.is_empty() {
                combined += cs[0] * weights[i];
            }
        }

        Self::verify(vp, r_point, &MkzgCommit(vec![combined]), y, kzg_proof)
    }

    pub fn verify(
        vp: &MkzgVerParams<E>,
        point: Vec<E::ScalarField>,
        comm: &MkzgCommit<E>,
        value: E::ScalarField,
        proof: MkzgProof<E>,
    ) -> bool {
        let log_chunk_size = vp.params.len();

        // First log_chunk_size variables are KZG (within-chunk),
        // remaining are chunk selectors.
        let kzg_point = &point[..log_chunk_size];
        let chunk_point = &point[log_chunk_size..];

        let one = E::ScalarField::one();
        let mul = kzg_point
            .iter()
            .fold(E::ScalarField::from(1), |acc, x| acc * (one - x));
        let kzg_point_transformed: Vec<_> = kzg_point
            .iter()
            .map(|&x| x * Field::inverse(&(one - x)).unwrap())
            .collect();
        let value = value * mul.inverse().unwrap();
        let MkzgVerParams { g, h, mut params } = vp.clone();
        for i in 0..kzg_point_transformed.len() {
            params[i] -= h * kzg_point_transformed[i];
        }
        let mut proof = proof.0;
        proof.push(g);
        params.push(h * value);

        // Fold commits using chunk-selector variables
        let mut commits = comm.0.clone();
        for r in chunk_point.iter() {
            let cur_len = commits.len();
            let new_len = (cur_len + 1) / 2;
            for i in 0..new_len {
                if i * 2 + 1 < cur_len {
                    commits[i] = commits[i * 2] + (commits[i * 2 + 1] - commits[i * 2]) * (*r);
                } else {
                    commits[i] = commits[i * 2] * (one - *r);
                }
            }
            commits.truncate(new_len);
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
        kzg::{LOG_CHUNK_SIZE, Mkzg},
        poly::MlPoly,
        util::RandomOracle,
    };

    #[test]
    fn it_works() {
        let log_len = 12;
        let mut rng = thread_rng();
        let poly = MlPoly(
            (0..(1 << log_len) - 100)
                .map(|_| <Fr as UniformRand>::rand(&mut rng))
                .collect(),
        );
        let (pp, vp) = Mkzg::<Bn254>::gen_srs(LOG_CHUNK_SIZE, &mut rng);
        let commit = Mkzg::commit(&pp, &poly);
        let points: Vec<Vec<Fr>> = (0..10)
            .map(|_| {
                (0..log_len)
                    .map(|_| <Fr as UniformRand>::rand(&mut rng))
                    .collect()
            })
            .collect();
        let polys: Vec<_> = (0..10).map(|_| poly.clone()).collect();
        let comms: Vec<_> = (0..10).map(|_| commit.clone()).collect();
        let values: Vec<Fr> = points.iter().map(|p| poly.clone().eval(p)).collect();

        let mut random_oracle = RandomOracle::new(&mut rng);
        let proof = Mkzg::batch_open(&pp, &polys, &points, &mut random_oracle);

        random_oracle.restart();
        assert!(Mkzg::batch_verify(
            &vp,
            &points,
            &comms,
            &values,
            proof,
            &mut random_oracle
        ));
    }

    #[test]
    fn test_batch_multi_poly() {
        let mut rng = thread_rng();
        let (pp, vp) = Mkzg::<Bn254>::gen_srs(LOG_CHUNK_SIZE, &mut rng);

        let poly1 = MlPoly((0..500).map(|_| Fr::rand(&mut rng)).collect());
        let poly2 = MlPoly((0..2000).map(|_| Fr::rand(&mut rng)).collect());
        let poly3 = MlPoly((0..(1 << 12)).map(|_| Fr::rand(&mut rng)).collect());

        let commit1 = Mkzg::commit(&pp, &poly1);
        let commit2 = Mkzg::commit(&pp, &poly2);
        let commit3 = Mkzg::commit(&pp, &poly3);

        let point1: Vec<Fr> = (0..9).map(|_| Fr::rand(&mut rng)).collect();
        let point2: Vec<Fr> = (0..11).map(|_| Fr::rand(&mut rng)).collect();
        let point3: Vec<Fr> = (0..12).map(|_| Fr::rand(&mut rng)).collect();

        let value1 = poly1.clone().eval(&point1);
        let value2 = poly2.clone().eval(&point2);
        let value3 = poly3.clone().eval(&point3);

        let polys = vec![poly1, poly2, poly3];
        let points = vec![point1, point2, point3];
        let comms = vec![commit1, commit2, commit3];
        let values = vec![value1, value2, value3];

        let mut random_oracle = RandomOracle::new(&mut rng);
        let proof = Mkzg::batch_open(&pp, &polys, &points, &mut random_oracle);

        random_oracle.restart();
        assert!(Mkzg::batch_verify(
            &vp,
            &points,
            &comms,
            &values,
            proof,
            &mut random_oracle
        ));
    }
}
