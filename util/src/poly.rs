use ark_ff::Field;

#[derive(Debug, Clone)]
pub struct MlPoly<F: Field>(pub Vec<F>);

impl<F: Field> MlPoly<F> {
    pub fn new(v: Vec<F>) -> MlPoly<F> {
        MlPoly(v)
    }

    pub fn eval(self, point: &[F]) -> F {
        let mut scratch = self.0;
        assert!(scratch.len() <= 1 << point.len());
        for r in point.iter() {
            let cur_len = scratch.len();
            let new_len = (cur_len + 1) / 2;
            for i in 0..new_len {
                if i * 2 + 1 < cur_len {
                    scratch[i] = scratch[i * 2] + (scratch[i * 2 + 1] - scratch[i * 2]) * (*r);
                } else {
                    scratch[i] = scratch[i * 2] * (F::ONE - *r);
                }
            }
            scratch.truncate(new_len);
        }
        scratch[0]
    }

    pub fn split(&self, n: usize) -> Vec<MlPoly<F>> {
        let len = self.0.len();
        let num_chunks = (len + n - 1) / n;
        let mut polies = Vec::with_capacity(num_chunks);
        for i in 0..num_chunks {
            let start = i * n;
            let end = std::cmp::min(start + n, len);
            let mut chunk = self.0[start..end].to_vec();
            chunk.resize(n, F::ZERO);
            polies.push(MlPoly(chunk));
        }
        polies
    }

    pub fn fold(&mut self, point: &[F]) {
        for r in point.iter() {
            let cur_len = self.0.len();
            let new_len = (cur_len + 1) / 2;
            for i in 0..new_len {
                if i * 2 + 1 < cur_len {
                    self.0[i] = self.0[i * 2] + (self.0[i * 2 + 1] - self.0[i * 2]) * (*r);
                } else {
                    self.0[i] = self.0[i * 2] * (F::ONE - *r);
                }
            }
            self.0.truncate(new_len);
        }
    }

    pub fn new_eq(point: &Vec<F>) -> Self {
        let mut evals = vec![F::one()];
        for &i in point.iter().rev() {
            evals = evals
                .iter()
                .flat_map(|&x| [(F::one() - i) * x, i * x])
                .collect()
        }
        MlPoly(evals)
    }

    pub fn eval_eq(r: &Vec<F>, point: &Vec<F>) -> F {
        let mut res = F::one();
        for (&i, &j) in r.iter().zip(point.iter()) {
            res *= i * j + (F::one() - i) * (F::one() - j);
        }
        res
    }

    pub fn eval_eq_pref(r: &Vec<F>, point: &Vec<F>, n: usize) -> F {
        let k = r.len();
        let m = point.len();
        let one = F::one();

        // Suffix: prod(1 - r[i]) for i = m..k, from bits that are always 0
        let mut suffix = one;
        for i in m..k {
            suffix *= one - r[i];
        }

        // If n >= 2^m, we sum over all 2^m indices => full eval_eq on first m vars
        if n >= (1usize << m) {
            let mut res = one;
            for i in 0..m {
                res *= r[i] * point[i] + (one - r[i]) * (one - point[i]);
            }
            return res * suffix;
        }

        // Precompute prefix products: P[i] = prod(eq_rp[0..i])
        let mut prefix_prod = vec![one; m + 1];
        for i in 0..m {
            let eq_rp = r[i] * point[i] + (one - r[i]) * (one - point[i]);
            prefix_prod[i + 1] = prefix_prod[i] * eq_rp;
        }

        // Process bits of n from MSB to LSB
        let mut result = F::ZERO;
        let mut carry = one;
        for bit in (0..m).rev() {
            let phi_0 = (one - r[bit]) * (one - point[bit]);
            if (n >> bit) & 1 == 1 {
                result += carry * phi_0 * prefix_prod[bit];
                carry *= r[bit] * point[bit];
            } else {
                carry *= phi_0;
            }
        }

        result * suffix
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_ff::{AdditiveGroup, UniformRand};
    use rand::thread_rng;

    use crate::poly::MlPoly;

    #[test]
    fn it_works() {
        let nv = 10;
        let mut rng = thread_rng();
        let poly = MlPoly((0..(1 << nv)).map(|_| Fr::rand(&mut rng)).collect());
        let point = (0..nv).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();
        let v = poly.clone().eval(&point);
        let eq_poly = MlPoly::new_eq(&point);
        let v2 = poly
            .0
            .iter()
            .zip(eq_poly.0.iter())
            .fold(Fr::ZERO, |acc, (&i, &j)| acc + i * j);
        assert_eq!(v, v2);
        let point2 = (0..nv).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();
        assert_eq!(eq_poly.eval(&point2), MlPoly::eval_eq(&point, &point2));
    }

    #[test]
    fn test_eval_eq_pref() {
        let k = 10;
        let mut rng = thread_rng();
        let r: Vec<Fr> = (0..k).map(|_| Fr::rand(&mut rng)).collect();
        let eq_poly = MlPoly::new_eq(&r);

        for n in [1, 2, 3, 5, 7, 8, 13, 100, 511, 512, 1000, 1024] {
            let m = (u32::BITS - (n as u32 - 1).leading_zeros()) as usize;
            let point: Vec<Fr> = (0..m).map(|_| Fr::rand(&mut rng)).collect();

            // Naive: materialize first n evals, pad to 2^m, evaluate
            let naive = MlPoly::new(eq_poly.0[..n].to_vec()).eval(&point);

            // Fast
            let fast = MlPoly::<Fr>::eval_eq_pref(&r, &point, n);

            assert_eq!(naive, fast, "mismatch at n={}", n);
        }

        // Test n=0
        let point1: Vec<Fr> = (0..1).map(|_| Fr::rand(&mut rng)).collect();
        assert_eq!(MlPoly::<Fr>::eval_eq_pref(&r, &point1, 0), Fr::ZERO);
    }
}
