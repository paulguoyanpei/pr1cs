use ark_ff::PrimeField;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub enum LookupType {
    Ge0,
    Relu,
}

#[derive(Debug, Clone)]
pub struct SparseRow<F: PrimeField> {
    pub elems: Vec<(usize, F, Option<usize>)>,
}

impl<F: PrimeField> SparseRow<F> {
    pub fn from_row(row: &Vec<(usize, i64)>) -> SparseRow<F> {
        let mut elems = vec![];
        for elem in row {
            elems.push((elem.0, F::from(elem.1), None));
        }
        return SparseRow { elems };
    }

    pub fn from_row_with_idx(row: &Vec<(usize, i64, Option<usize>)>) -> SparseRow<F> {
        let mut elems = vec![];
        for elem in row {
            elems.push((elem.0, F::from(elem.1), elem.2));
        }
        return SparseRow { elems };
    }

    pub fn mult_vec(&self, z: &Vec<F>, gamma: F) -> F {
        let mut s = F::zero();
        for elem in self.elems.clone() {
            match elem.2 {
                Some(p) => s += z[elem.0] * elem.1 * gamma.pow(vec![p as u64]),
                None => s += z[elem.0] * elem.1,
            }
        }
        s
    }
}

#[derive(Debug, Clone)]
pub struct SparseMatrix<F: PrimeField> {
    pub rows: Vec<SparseRow<F>>,
}

impl<F: PrimeField> SparseMatrix<F> {
    pub fn new() -> SparseMatrix<F> {
        SparseMatrix {
            rows: vec![],
        }
    }

    pub fn append(&mut self, row: &Vec<(usize, i64)>) {
        let new_row = SparseRow::from_row(row);
        self.rows.push(new_row);
    }

    pub fn append_with_idx(&mut self, row: &Vec<(usize, i64, Option<usize>)>) {
        let new_row = SparseRow::from_row_with_idx(row);
        self.rows.push(new_row);
    }

    pub fn len(&self) -> usize {
        return self.rows.len();
    }

    pub fn mult_vec_at(&self, idx: usize, z: &Vec<F>, gamma: F) -> F {
        return self.rows[idx].mult_vec(z, gamma);
    }

    pub fn vec_mult(&self, v: &Vec<F>, col_num: usize, gamma: F) -> Vec<F> {
        assert!(v.len() >= self.rows.len());
        let mut res = Vec::new();
        res.resize(col_num, F::ZERO);
        for i in 0..(self.rows.len()) {
            for (idx, val, pow) in self.rows[i].elems.iter() {
                let p = match *pow {
                    Some(p) => gamma.pow([p as u64]),
                    None => F::ONE,
                };
                res[*idx] += v[i] * val.clone() * p;
            }
        }
        res
    }
}

#[derive(Debug, Clone)]
pub struct Circuit<F: PrimeField> {
    pub a: SparseMatrix<F>,
    pub b: SparseMatrix<F>,
    pub c: SparseMatrix<F>,
    pub d: SparseMatrix<F>,
    pub e: SparseMatrix<F>,
    pub tp: Vec<LookupType>,
    pub weight_len: usize,
}

impl<F: PrimeField> Circuit<F> {
    pub fn new(
        a: SparseMatrix<F>,
        b: SparseMatrix<F>,
        c: SparseMatrix<F>,
        d: SparseMatrix<F>,
        e: SparseMatrix<F>,
        tp: Vec<LookupType>,
        weight_len: usize,
    ) -> Circuit<F> {
        Circuit {
            a,
            b,
            c,
            d,
            e,
            tp,
            weight_len,
        }
    }

    // testing function
    pub fn check(&self, z: Vec<F>, gamma: F, set: HashSet<(F, F, F)>) -> bool {
        assert_eq!(self.a.len(), self.b.len());
        assert_eq!(self.b.len(), self.c.len());

        for cr in 0..self.a.len() {
            assert_eq!(
                self.a.mult_vec_at(cr, &z, gamma) * self.b.mult_vec_at(cr, &z, gamma),
                self.c.mult_vec_at(cr, &z, gamma)
            );
        }

        assert_eq!(self.d.len(), self.e.len());
        assert_eq!(self.e.len(), self.tp.len());

        for cr in 0..self.d.len() {
            match self.tp[cr] {
                LookupType::Relu => assert!(set.contains(&(
                    self.d.mult_vec_at(cr, &z, gamma),
                    self.e.mult_vec_at(cr, &z, gamma),
                    F::ZERO
                ))),
                LookupType::Ge0 => assert!(set.contains(&(
                    self.d.mult_vec_at(cr, &z, gamma),
                    self.e.mult_vec_at(cr, &z, gamma),
                    F::ZERO
                ))),
            }
        }
        return true;
    }
}
