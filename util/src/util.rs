use ark_ec::pairing::Pairing;
use ark_ff::{Field, PrimeField, UniformRand};
use rand::Rng;

pub struct Proof<E: Pairing> {
    u: Vec<usize>,
    f: Vec<E::ScalarField>,
    g: Vec<E::G1>,
    u_idx: usize,
    f_idx: usize,
    g_idx: usize,
}

impl<E: Pairing> Proof<E> {
    pub fn new() -> Proof<E> {
        Proof {
            u: vec![],
            f: vec![],
            g: vec![],
            u_idx: 0,
            f_idx: 0,
            g_idx: 0,
        }
    }

    pub fn push_u(&mut self, u: usize) {
        self.u.push(u);
    }

    pub fn push_f(&mut self, f: &[E::ScalarField]) {
        for i in f {
            self.f.push(*i);
        }
    }

    pub fn push_g(&mut self, g: E::G1) {
        self.g.push(g);
    }

    pub fn push_gs(&mut self, gs: &[E::G1]) {
        for i in gs {
            self.g.push(*i);
        }
    }

    pub fn next_f(&mut self) -> E::ScalarField {
        let ret = self.f[self.f_idx];
        self.f_idx += 1;
        ret
    }

    pub fn next_n_fs(&mut self, n: usize) -> Vec<E::ScalarField> {
        let ret = self.f[self.f_idx..self.f_idx + n].to_vec();
        self.f_idx += n;
        ret
    }

    pub fn next_u(&mut self) -> usize {
        let ret = self.u[self.u_idx];
        self.u_idx += 1;
        ret
    }

    pub fn next_g(&mut self) -> E::G1 {
        let ret = self.g[self.g_idx];
        self.g_idx += 1;
        ret
    }

    pub fn next_n_gs(&mut self, n: usize) -> Vec<E::G1> {
        let ret = self.g[self.g_idx..self.g_idx + n].to_vec();
        self.g_idx += n;
        ret
    }
}

pub fn batch_inverse<F: PrimeField>(v: &mut [F]) {
    let mut aux = vec![v[0]];
    let len = v.len();
    for i in 1..len {
        aux.push(aux[i - 1] * v[i]);
    }
    let mut prod = aux[len - 1].inverse().unwrap();
    for i in (1..len).rev() {
        (prod, v[i]) = (prod * v[i], prod * aux[i - 1]);
    }
    v[0] = prod;
}

pub struct RandomOracle<F: Field> {
    fields: Vec<F>,
    ints: Vec<usize>,
    fields_idx: usize,
    ints_idx: usize,
}

impl<F: Field> RandomOracle<F> {
    pub fn new<R: Rng>(rng: &mut R) -> Self {
        RandomOracle {
            fields: (0..1000).map(|_| <F as UniformRand>::rand(rng)).collect(),
            ints: (0..1000).map(|_| usize::rand(rng)).collect(),
            fields_idx: 0,
            ints_idx: 0,
        }
    }

    pub fn next_field(&mut self) -> F {
        let res = self.fields[self.fields_idx];
        self.fields_idx += 1;
        res
    }

    pub fn next_n_fields(&mut self, n: usize) -> Vec<F> {
        let res = self.fields[self.fields_idx..(self.fields_idx + n)].to_vec();
        self.fields_idx += n;
        res
    }

    pub fn next_int(&mut self) -> usize {
        let res = self.ints[self.ints_idx];
        self.ints_idx += 1;
        res
    }

    pub fn next_n_ints(&mut self, n: usize) -> Vec<usize> {
        let res = self.ints[self.ints_idx..(self.ints_idx + n)].to_vec();
        self.ints_idx += n;
        res
    }

    pub fn restart(&mut self) {
        self.fields_idx = 0;
        self.ints_idx = 0;
    }
}
