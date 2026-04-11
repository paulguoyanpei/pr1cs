use ark_ff::FftField;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};

#[derive(Debug, Clone)]
pub struct Radix2Group<F: FftField> {
    domain: Radix2EvaluationDomain<F>,
    elements: Vec<F>,
}

impl<F: FftField> Radix2Group<F> {
    pub fn new(size: usize) -> Self {
        let domain = Radix2EvaluationDomain::new(size).unwrap();
        let elements = domain.elements().collect::<Vec<_>>();
        Radix2Group { domain, elements }
    }

    pub fn fft(&self, v: &[F]) -> Vec<F> {
        self.domain.fft(v)
    }

    pub fn ifft(&self, v: &[F]) -> Vec<F> {
        self.domain.ifft(v)
    }

    pub fn element_at(&self, idx: usize) -> F {
        self.elements[idx]
    }

    pub fn element_inv_at(&self, idx: usize) -> F {
        if idx == 0 {
            self.elements[0]
        } else {
            self.elements[self.domain.size() - idx]
        }
    }

    pub fn size(&self) -> usize {
        self.domain.size()
    }
}
