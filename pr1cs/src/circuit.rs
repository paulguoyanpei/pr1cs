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
        SparseMatrix { rows: vec![] }
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

    pub fn vec_mult_suf(&self, v: &Vec<F>, col_num: usize, pos: usize, gamma: F) -> Vec<F> {
        assert!(v.len() >= self.rows.len());
        let mut res = Vec::new();
        res.resize(col_num - pos, F::ZERO);
        for i in 0..(self.rows.len()) {
            for &(idx, val, pow) in self.rows[i].elems.iter() {
                if idx < pos {
                    continue;
                }
                let p = match pow {
                    Some(p) => gamma.pow([p as u64]),
                    None => F::ONE,
                };
                res[idx - pos] += v[i] * val.clone() * p;
            }
        }
        res
    }

    pub fn vec_mult_pre(&self, v: &Vec<F>, pos: usize, gamma: F) -> Vec<F> {
        assert!(v.len() >= self.rows.len());
        let mut res = Vec::new();
        res.resize(pos, F::ZERO);
        for i in 0..(self.rows.len()) {
            for &(idx, val, pow) in self.rows[i].elems.iter() {
                if idx >= pos {
                    continue;
                }
                let p = match pow {
                    Some(p) => gamma.pow([p as u64]),
                    None => F::ONE,
                };
                res[idx] += v[i] * val.clone() * p;
            }
        }
        res
    }

    // Evaluates MLE of the sparse matrix restricted to column window
    // [col_offset, col_limit), at (row_point, col_point), with row_eq and
    // col_eq being the full eq-vectors for those points. Returns
    //   sum_{(cr, col, val, pow) in entries, col_offset <= col < col_limit}
    //       row_eq[cr] * col_eq[col - col_offset] * val * gamma^pow
    pub fn mle(
        &self,
        row_eq: &[F],
        col_eq: &[F],
        col_offset: usize,
        col_limit: usize,
        gamma: F,
    ) -> F {
        let mut result = F::ZERO;
        for cr in 0..self.rows.len() {
            let re = row_eq[cr];
            for (col, val, pow) in self.rows[cr].elems.iter() {
                if *col >= col_offset && *col < col_limit {
                    let p = match *pow {
                        Some(p) => gamma.pow([p as u64]),
                        None => F::ONE,
                    };
                    result += re * col_eq[col - col_offset] * *val * p;
                }
            }
        }
        result
    }
}

#[derive(Debug, Clone)]
pub struct Circuit<F: PrimeField> {
    pub a: SparseMatrix<F>,
    pub b: SparseMatrix<F>,
    pub c: SparseMatrix<F>,
    pub d: SparseMatrix<F>,
    pub e: SparseMatrix<F>,
    pub tp: Vec<F>,
    pub weights: Vec<F>,
    pub weight_len: usize,
    pub table: Vec<(F, F, F)>,
}

impl<F: PrimeField> Circuit<F> {
    pub fn new(
        a: SparseMatrix<F>,
        b: SparseMatrix<F>,
        c: SparseMatrix<F>,
        d: SparseMatrix<F>,
        e: SparseMatrix<F>,
        tp: Vec<LookupType>,
        weights: Vec<F>,
        weight_len: usize,
        table: Vec<(F, F, F)>,
    ) -> Circuit<F> {
        Circuit {
            a,
            b,
            c,
            d,
            e,
            tp: tp
                .into_iter()
                .map(|x| match x {
                    LookupType::Ge0 => F::from(1),
                    LookupType::Relu => F::from(2),
                })
                .collect(),
            weights,
            weight_len,
            table,
        }
    }

    // testing function
    pub fn check(&self, z: Vec<F>, gamma: F) -> bool {
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

        let set = self.table.clone().into_iter().collect::<HashSet<_>>();

        for cr in 0..self.d.len() {
            if self.tp[cr] == F::from(1) {
                assert!(set.contains(&(
                    self.d.mult_vec_at(cr, &z, gamma),
                    self.e.mult_vec_at(cr, &z, gamma),
                    F::from(1)
                )))
            } else if self.tp[cr] == F::from(2) {
                assert!(set.contains(&(
                    self.d.mult_vec_at(cr, &z, gamma),
                    self.e.mult_vec_at(cr, &z, gamma),
                    F::from(2)
                )))
            } else {
                panic!("undefined lookup type")
            }
        }
        return true;
    }
}
