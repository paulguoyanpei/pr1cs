use ark_ff::{FftField, Field, One};
use ark_serialize::CanonicalSerialize;

#[derive(Debug, Clone)]
pub struct MlPoly<F: Field>(pub Vec<F>);

impl<F: Field> MlPoly<F> {
    pub fn new(mut v: Vec<F>) -> MlPoly<F> {
        let log_len = (v.len() - 1).ilog2() + 1;
        let len = 1usize << log_len;
        while v.len() < len {
            v.push(F::ZERO);
        }
        MlPoly(v)
    }

    pub fn eval(self, point: &[F]) -> F {
        let mut scratch = self.0;
        let mut cur_len = scratch.len() >> 1;
        assert_eq!(1 << point.len(), scratch.len());
        for r in point.iter() {
            for i in 0..cur_len {
                scratch[i] = scratch[i * 2] + (scratch[i * 2 + 1] - scratch[i * 2]) * (*r);
            }
            cur_len >>= 1;
        }
        scratch[0]
    }

    pub fn split(&self, n: usize) -> Vec<MlPoly<F>> {
        assert_eq!(n & (n - 1), 0);
        let mut polies = (0..n).map(|_| vec![]).collect::<Vec<_>>();
        let len = self.0.len();
        for i in (0..len).step_by(n) {
            for j in 0..n {
                polies[j].push(self.0[i + j]);
            }
        }
        polies.into_iter().map(|x| MlPoly(x)).collect()
    }

    pub fn fold(&mut self, point: &[F]) {
        let mut cur_len = self.0.len();
        for r in point.iter() {
            cur_len >>= 1;
            for i in 0..cur_len {
                self.0[i] = self.0[i * 2] + (self.0[i * 2 + 1] - self.0[i * 2]) * (*r);
            }
        }
        self.0.truncate(cur_len);
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

    pub fn eval_eq(point1: &Vec<F>, point2: &Vec<F>) -> F {
        let mut res = F::one();
        for (&i, &j) in point1.iter().zip(point2.iter()) {
            res *= i * j + (F::one() - i) * (F::one() - j);
        }
        res
    }
}

#[derive(Debug, Clone)]
pub struct UniVarPoly<F: Field>(Vec<F>);

impl<F: Field> UniVarPoly<F> {
    pub fn new(coeff: Vec<F>) -> Self {
        UniVarPoly(coeff)
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut bytes = vec![];
        self.0
            .iter()
            .for_each(|x| <F as CanonicalSerialize>::serialize_compressed(&x, &mut bytes).unwrap());
        bytes
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn eval(&self, point: &F) -> F {
        let mut res = self.0.last().unwrap().clone();
        for i in self.0.iter().rev().skip(1) {
            res *= point;
            res += i;
        }
        res
    }
}

#[derive(Debug, Clone)]
pub struct UniPolyEvals<F: Field> {
    evals: Vec<F>,
    offset_inv: F,
}

impl<F: FftField> UniPolyEvals<F> {
    pub fn new(evals: Vec<F>, offset_inv: F) -> UniPolyEvals<F> {
        UniPolyEvals { evals, offset_inv }
    }

    pub fn len(&self) -> usize {
        self.evals.len()
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut bytes = vec![];
        self.evals
            .iter()
            .for_each(|x| <F as CanonicalSerialize>::serialize_compressed(&x, &mut bytes).unwrap());
        bytes
    }

    pub fn n_th_eval(&self, n: usize) -> F {
        self.evals[n & (self.evals.len() - 1)]
    }

    pub fn eval(self, mut point: F, mut root_inv: F, inv_2: F) -> F {
        let UniPolyEvals {
            mut evals,
            mut offset_inv,
        } = self;
        let mut len = evals.len();
        let mut inv = <F as One>::one();
        for _ in 0..evals.len().ilog2() {
            len >>= 1;
            let mut w = offset_inv;
            for j in 0..len {
                let t = (evals[j] - evals[j + len]) * point * w;
                evals[j] = evals[j] + evals[j + len] + t;
                w *= root_inv;
            }
            offset_inv *= offset_inv;
            inv *= inv_2;
            point *= point;
            root_inv *= root_inv;
        }
        evals[0] * inv
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
}
