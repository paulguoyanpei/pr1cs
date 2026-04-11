use ark_ff::{Field, UniformRand};
use rand::Rng;

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
