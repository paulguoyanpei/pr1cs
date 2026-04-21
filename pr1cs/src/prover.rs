use std::collections::HashMap;

use ark_ec::pairing::Pairing;
use ark_ff::{AdditiveGroup, One, PrimeField, Zero};
use util::{
    kzg::{Mkzg, MkzgProveParams},
    poly::MlPoly,
    util::{batch_inverse, Proof, RandomOracle},
};

use crate::circuit::Circuit;

pub struct Prover<E: Pairing> {
    kzg_pp: MkzgProveParams<E>,
    circuit: Circuit<E::ScalarField>,
}

impl<E: Pairing> Prover<E> {
    // prove a*b*eq = c
    fn sumcheck_1(
        mut a: Vec<E::ScalarField>,
        mut b: Vec<E::ScalarField>,
        mut c: Vec<E::ScalarField>,
        proof: &mut Proof<E>,
        ro: &mut RandomOracle<E::ScalarField>,
    ) -> Vec<E::ScalarField> {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), c.len());
        proof.push_u(a.len());
        let log_len = (a.len() - 1).ilog2() as usize + 1;
        let mut eq = MlPoly::new_eq(&ro.next_n_fields(log_len)).0;
        eq.truncate(a.len());
        let mut new_point = vec![];

        for _ in 0..log_len {
            if a.len() % 2 == 1 {
                a.push(E::ScalarField::zero());
                b.push(E::ScalarField::zero());
                c.push(E::ScalarField::zero());
                eq.push(E::ScalarField::zero());
            }
            let m = a.len();
            let mut sums = [0; 4].map(|_| E::ScalarField::zero());

            for j in (0..m).step_by(2) {
                let diff_a = a[j + 1] - a[j];
                let diff_b = b[j + 1] - b[j];
                let diff_c = c[j + 1] - c[j];
                let diff_eq = eq[j + 1] - eq[j];
                sums[0] += (a[j] * b[j] - c[j]) * eq[j];
                sums[1] += (a[j + 1] * b[j + 1] - c[j + 1]) * eq[j + 1];
                sums[2] += ((a[j + 1] + diff_a) * (b[j + 1] + diff_b) - (c[j + 1] + diff_c))
                    * (eq[j + 1] + diff_eq);
                sums[3] += ((a[j + 1] + diff_a + diff_a) * (b[j + 1] + diff_b + diff_b)
                    - (c[j + 1] + diff_c + diff_c))
                    * (eq[j + 1] + diff_eq + diff_eq);
            }

            proof.push_f(&sums);

            let challenge = ro.next_field();
            new_point.push(challenge);

            for i in 0..m / 2 {
                a[i] = a[i * 2] + (a[i * 2 + 1] - a[i * 2]) * challenge;
                b[i] = b[i * 2] + (b[i * 2 + 1] - b[i * 2]) * challenge;
                c[i] = c[i * 2] + (c[i * 2 + 1] - c[i * 2]) * challenge;
                eq[i] = eq[i * 2] + (eq[i * 2 + 1] - eq[i * 2]) * challenge;
            }
            a.truncate(m / 2);
            b.truncate(m / 2);
            c.truncate(m / 2);
            eq.truncate(m / 2);
        }

        assert_eq!(a.len(), 1);
        proof.push_f(&[a[0], b[0], c[0]]);
        new_point
    }

    // prove sum m[i] * z[i]
    fn sumcheck_2(
        mut m: Vec<E::ScalarField>,
        mut z: Vec<E::ScalarField>,
        proof: &mut Proof<E>,
        ro: &mut RandomOracle<E::ScalarField>,
    ) -> Vec<E::ScalarField> {
        assert_eq!(m.len(), z.len());
        let log_len = (m.len() - 1).ilog2() as usize + 1;
        proof.push_u(log_len);
        let mut new_point = vec![];

        for _ in 0..log_len {
            if m.len() % 2 == 1 {
                m.push(E::ScalarField::zero());
                z.push(E::ScalarField::zero());
            }
            let l = m.len();
            let mut sums = [0; 3].map(|_| E::ScalarField::zero());

            for j in (0..l).step_by(2) {
                let diff_m = m[j + 1] - m[j];
                let diff_z = z[j + 1] - z[j];
                sums[0] += m[j] * z[j];
                sums[1] += m[j + 1] * z[j + 1];
                sums[2] += (m[j + 1] + diff_m) * (z[j + 1] + diff_z);
            }

            proof.push_f(&sums);

            let challenge = ro.next_field();
            new_point.push(challenge);

            for i in 0..l / 2 {
                m[i] = m[i * 2] + (m[i * 2 + 1] - m[i * 2]) * challenge;
                z[i] = z[i * 2] + (z[i * 2 + 1] - z[i * 2]) * challenge;
            }
            m.truncate(l / 2);
            z.truncate(l / 2);
        }

        assert_eq!(m.len(), 1);
        proof.push_f(&[m[0], z[0]]);
        new_point
    }

    pub fn new(pp: MkzgProveParams<E>, c: Circuit<E::ScalarField>) -> Self {
        Prover {
            kzg_pp: pp,
            circuit: c,
        }
    }

    fn count(ele: &Vec<E::ScalarField>, tab: &Vec<E::ScalarField>) -> Vec<E::ScalarField> {
        let mut freq = HashMap::new();
        for &e in ele {
            *freq.entry(e).or_insert(0u64) += 1;
        }
        tab.iter()
            .map(|t| E::ScalarField::from(freq.get(t).copied().unwrap_or(0)))
            .collect()
    }

    fn logup_sumcheck_left(
        mut ele: Vec<E::ScalarField>,
        mut ele_inv: Vec<E::ScalarField>,
        alpha: E::ScalarField,
        proof: &mut Proof<E>,
        ro: &mut RandomOracle<E::ScalarField>,
    ) -> Vec<E::ScalarField> {
        let log_len = (ele.len() - 1).ilog2() as usize + 1;
        ele.iter_mut().for_each(|x| *x += alpha);
        proof.push_u(ele.len());
        let mut eq = MlPoly::new_eq(&ro.next_n_fields(log_len)).0;
        eq.truncate(ele.len());
        let r = ro.next_field();
        let mut new_point = vec![];

        for _ in 0..log_len {
            if ele.len() % 2 == 1 {
                ele.push(E::ScalarField::zero());
                ele_inv.push(E::ScalarField::zero());
                eq.push(E::ScalarField::zero());
            }
            let m = ele.len();
            let mut sums = [0; 4].map(|_| E::ScalarField::zero());

            for j in (0..m).step_by(2) {
                let diff_ele = ele[j + 1] - ele[j];
                let diff_ele_inv = ele_inv[j + 1] - ele_inv[j];
                let diff_eq = eq[j + 1] - eq[j];
                sums[0] += (ele[j] * ele_inv[j] - E::ScalarField::one()) * eq[j] + r * ele_inv[j];
                sums[1] += (ele[j + 1] * ele_inv[j + 1] - E::ScalarField::one()) * eq[j + 1]
                    + r * ele_inv[j + 1];
                sums[2] += ((ele[j + 1] + diff_ele) * (ele_inv[j + 1] + diff_ele_inv)
                    - E::ScalarField::one())
                    * (eq[j + 1] + diff_eq)
                    + r * (ele_inv[j + 1] + diff_ele_inv);
                sums[3] += ((ele[j + 1] + diff_ele + diff_ele)
                    * (ele_inv[j + 1] + diff_ele_inv + diff_ele_inv)
                    - E::ScalarField::one())
                    * (eq[j + 1] + diff_eq + diff_eq)
                    + r * (ele_inv[j + 1] + diff_ele_inv + diff_ele_inv);
            }

            proof.push_f(&sums);

            let challenge = ro.next_field();
            new_point.push(challenge);

            for i in 0..m / 2 {
                ele[i] = ele[i * 2] + (ele[i * 2 + 1] - ele[i * 2]) * challenge;
                ele_inv[i] = ele_inv[i * 2] + (ele_inv[i * 2 + 1] - ele_inv[i * 2]) * challenge;
                eq[i] = eq[i * 2] + (eq[i * 2 + 1] - eq[i * 2]) * challenge;
            }
            ele.truncate(m / 2);
            ele_inv.truncate(m / 2);
            eq.truncate(m / 2);
        }

        assert_eq!(ele.len(), 1);
        proof.push_f(&[ele[0], ele_inv[0]]);
        new_point
    }

    fn logup_sumcheck_right(
        mut tab: Vec<E::ScalarField>,
        mut tab_inv: Vec<E::ScalarField>,
        mut count: Vec<E::ScalarField>,
        alpha: E::ScalarField,
        proof: &mut Proof<E>,
        ro: &mut RandomOracle<E::ScalarField>,
    ) -> Vec<E::ScalarField> {
        let log_len = (tab.len() - 1).ilog2() as usize + 1;
        tab.iter_mut().for_each(|x| *x += alpha);
        proof.push_u(tab.len());
        let mut eq = MlPoly::new_eq(&ro.next_n_fields(log_len)).0;
        eq.truncate(tab.len());
        let r = ro.next_field();
        let mut new_point = vec![];

        for _ in 0..log_len {
            if tab.len() % 2 == 1 {
                tab.push(E::ScalarField::zero());
                tab_inv.push(E::ScalarField::zero());
                count.push(E::ScalarField::zero());
                eq.push(E::ScalarField::zero());
            }
            let m = tab.len();
            let mut sums = [0; 4].map(|_| E::ScalarField::zero());

            for j in (0..m).step_by(2) {
                let diff_tab = tab[j + 1] - tab[j];
                let diff_tab_inv = tab_inv[j + 1] - tab_inv[j];
                let diff_count = count[j + 1] - count[j];
                let diff_eq = eq[j + 1] - eq[j];
                sums[0] += (tab[j] * tab_inv[j] - E::ScalarField::one()) * eq[j]
                    + r * tab_inv[j] * count[j];
                sums[1] += (tab[j + 1] * tab_inv[j + 1] - E::ScalarField::one()) * eq[j + 1]
                    + r * tab_inv[j + 1] * count[j + 1];
                sums[2] += ((tab[j + 1] + diff_tab) * (tab_inv[j + 1] + diff_tab_inv)
                    - E::ScalarField::one())
                    * (eq[j + 1] + diff_eq)
                    + r * (tab_inv[j + 1] + diff_tab_inv) * (count[j + 1] + diff_count);
                sums[3] += ((tab[j + 1] + diff_tab + diff_tab)
                    * (tab_inv[j + 1] + diff_tab_inv + diff_tab_inv)
                    - E::ScalarField::one())
                    * (eq[j + 1] + diff_eq + diff_eq)
                    + r * (tab_inv[j + 1] + diff_tab_inv.double())
                        * (count[j + 1] + diff_count + diff_count);
            }

            proof.push_f(&sums);

            let challenge = ro.next_field();
            new_point.push(challenge);

            for i in 0..m / 2 {
                tab[i] = tab[i * 2] + (tab[i * 2 + 1] - tab[i * 2]) * challenge;
                tab_inv[i] = tab_inv[i * 2] + (tab_inv[i * 2 + 1] - tab_inv[i * 2]) * challenge;
                count[i] = count[i * 2] + (count[i * 2 + 1] - count[i * 2]) * challenge;
                eq[i] = eq[i * 2] + (eq[i * 2 + 1] - eq[i * 2]) * challenge;
            }
            tab.truncate(m / 2);
            tab_inv.truncate(m / 2);
            count.truncate(m / 2);
            eq.truncate(m / 2);
        }

        assert_eq!(tab.len(), 1);
        proof.push_f(&[tab[0], tab_inv[0], count[0]]);
        new_point
    }

    pub fn prove(
        &self,
        z: Vec<E::ScalarField>,
        gamma: E::ScalarField,
        ro: &mut RandomOracle<E::ScalarField>,
    ) -> Proof<E> {
        let mut proof = Proof::new();
        let wei_len = self.circuit.weight_len;

        // Commit z_suf up-front (witness, independent of any challenge).
        let z_suf_poly = MlPoly::new(z[wei_len..].to_vec());
        let z_suf_commit = Mkzg::<E>::commit(&self.kzg_pp, &z_suf_poly);
        proof.push_u(z_suf_commit.0.len());
        proof.push_gs(&z_suf_commit.0);

        let cons_line = self.circuit.a.len();
        let lu_line = self.circuit.d.len();
        let mut a = vec![];
        let mut b = vec![];
        let mut c = vec![];
        for cr in 0..cons_line {
            a.push(self.circuit.a.mult_vec_at(cr, &z, gamma));
            b.push(self.circuit.b.mult_vec_at(cr, &z, gamma));
            c.push(self.circuit.c.mult_vec_at(cr, &z, gamma));
        }
        let mut d = vec![];
        let mut e = vec![];
        for i in 0..lu_line {
            d.push(self.circuit.d.mult_vec_at(i, &z, gamma));
            e.push(self.circuit.e.mult_vec_at(i, &z, gamma));
        }

        let point1 = Self::sumcheck_1(a, b, c, &mut proof, ro);

        let r = ro.next_field();
        let alpha = ro.next_field();
        let mut ele = vec![];
        for i in 0..lu_line {
            ele.push(((self.circuit.tp[i] * r) + e[i]) * r + d[i]);
        }
        let mut tab = vec![];
        for &(i, j, k) in &self.circuit.table {
            tab.push(((k * r) + j) * r + i);
        }
        let count = Self::count(&ele, &tab);

        let mut ele_inv = ele.iter().map(|&x| x + alpha).collect::<Vec<_>>();
        batch_inverse(&mut ele_inv);
        let mut tab_inv = tab.iter().map(|&x| x + alpha).collect::<Vec<_>>();
        batch_inverse(&mut tab_inv);

        // Commit count, ele_inv, tab_inv (depend on challenges r, alpha).
        let count_poly = MlPoly::new(count.clone());
        let count_commit = Mkzg::<E>::commit(&self.kzg_pp, &count_poly);
        proof.push_u(count_commit.0.len());
        proof.push_gs(&count_commit.0);
        let ele_inv_poly = MlPoly::new(ele_inv.clone());
        let ele_inv_commit = Mkzg::<E>::commit(&self.kzg_pp, &ele_inv_poly);
        proof.push_u(ele_inv_commit.0.len());
        proof.push_gs(&ele_inv_commit.0);
        let tab_inv_poly = MlPoly::new(tab_inv.clone());
        let tab_inv_commit = Mkzg::<E>::commit(&self.kzg_pp, &tab_inv_poly);
        proof.push_u(tab_inv_commit.0.len());
        proof.push_gs(&tab_inv_commit.0);

        let sum: E::ScalarField = ele_inv.iter().sum();
        proof.push_f(&[sum]);
        let point_logup_left = Self::logup_sumcheck_left(ele, ele_inv, alpha, &mut proof, ro);
        proof.push_f(&[
            MlPoly::new(d).eval(&point_logup_left),
            MlPoly::new(e).eval(&point_logup_left),
            MlPoly::new(self.circuit.tp.clone()).eval(&point_logup_left),
        ]);
        let point_logup_right =
            Self::logup_sumcheck_right(tab, tab_inv, count, alpha, &mut proof, ro);

        let eq_v = MlPoly::new_eq(&point1).0;
        let a = self.circuit.a.vec_mult(&eq_v, z.len(), gamma);
        let b = self.circuit.b.vec_mult(&eq_v, z.len(), gamma);
        let c = self.circuit.c.vec_mult(&eq_v, z.len(), gamma);
        let eq_v = MlPoly::new_eq(&point_logup_left).0;
        let d = self.circuit.d.vec_mult(&eq_v, z.len(), gamma);
        let e = self.circuit.e.vec_mult(&eq_v, z.len(), gamma);

        let a_suf = a[wei_len..].to_vec();
        let b_suf = b[wei_len..].to_vec();
        let c_suf = c[wei_len..].to_vec();
        let d_suf = d[wei_len..].to_vec();
        let e_suf = e[wei_len..].to_vec();
        let z_suf = z[wei_len..].to_vec();
        let mut m_suf = e_suf.clone();
        let mut a_suf_sum = E::ScalarField::zero();
        let mut b_suf_sum = E::ScalarField::zero();
        let mut c_suf_sum = E::ScalarField::zero();
        let mut d_suf_sum = E::ScalarField::zero();
        let mut e_suf_sum = E::ScalarField::zero();
        for i in 0..a_suf.len() {
            a_suf_sum += a_suf[i] * z_suf[i];
            b_suf_sum += b_suf[i] * z_suf[i];
            c_suf_sum += c_suf[i] * z_suf[i];
            d_suf_sum += d_suf[i] * z_suf[i];
            e_suf_sum += e_suf[i] * z_suf[i];
        }
        proof.push_f(&[a_suf_sum, b_suf_sum, c_suf_sum, d_suf_sum, e_suf_sum]);
        let r = ro.next_field();
        for i in 0..m_suf.len() {
            m_suf[i] *= r;
            m_suf[i] += d_suf[i];
            m_suf[i] *= r;
            m_suf[i] += c_suf[i];
            m_suf[i] *= r;
            m_suf[i] += b_suf[i];
            m_suf[i] *= r;
            m_suf[i] += a_suf[i];
        }
        let point2 = Self::sumcheck_2(m_suf, z_suf, &mut proof, ro);

        let a_pre = a[..wei_len].to_vec();
        let b_pre = b[..wei_len].to_vec();
        let c_pre = c[..wei_len].to_vec();
        let d_pre = d[..wei_len].to_vec();
        let e_pre = e[..wei_len].to_vec();
        let z_pre = z[..wei_len].to_vec();
        let mut m_pre = e_pre.clone();
        let mut a_pre_sum = E::ScalarField::zero();
        let mut b_pre_sum = E::ScalarField::zero();
        let mut c_pre_sum = E::ScalarField::zero();
        let mut d_pre_sum = E::ScalarField::zero();
        let mut e_pre_sum = E::ScalarField::zero();
        for i in 0..wei_len {
            a_pre_sum += a_pre[i] * z_pre[i];
            b_pre_sum += b_pre[i] * z_pre[i];
            c_pre_sum += c_pre[i] * z_pre[i];
            d_pre_sum += d_pre[i] * z_pre[i];
            e_pre_sum += e_pre[i] * z_pre[i];
        }
        proof.push_f(&[a_pre_sum, b_pre_sum, c_pre_sum, d_pre_sum, e_pre_sum]);
        let r = ro.next_field();
        for i in 0..m_pre.len() {
            m_pre[i] *= r;
            m_pre[i] += d_pre[i];
            m_pre[i] *= r;
            m_pre[i] += c_pre[i];
            m_pre[i] *= r;
            m_pre[i] += b_pre[i];
            m_pre[i] *= r;
            m_pre[i] += a_pre[i];
        }

        let _point3 = Self::sumcheck_2(m_pre, z_pre, &mut proof, ro);

        // Batch-open the four committed polynomials at their respective points.
        let polys = vec![z_suf_poly, count_poly, ele_inv_poly, tab_inv_poly];
        let points = vec![
            point2,
            point_logup_right.clone(),
            point_logup_left,
            point_logup_right,
        ];
        let (kzg_proof, sumcheck_proof) = Mkzg::<E>::batch_open(&self.kzg_pp, &polys, &points, ro);
        proof.push_u(kzg_proof.0.len());
        proof.push_gs(&kzg_proof.0);
        proof.push_u(sumcheck_proof.0.len());
        proof.push_f(&sumcheck_proof.0);

        proof
    }
}
