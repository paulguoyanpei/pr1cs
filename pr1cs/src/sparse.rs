use ark_ec::pairing::Pairing;
use ark_ff::{Field, One, PrimeField, Zero};
use rayon::prelude::*;
use std::{env, time::Instant};

use util::{
    kzg::{Mkzg, MkzgCommit, MkzgProof, MkzgProveParams, MkzgVerParams, SumcheckProof},
    poly::MlPoly,
    util::{Proof, RandomOracle, batch_inverse},
};

use crate::circuit::{Circuit, SparseMatrix};

// Claim ordering matches SparseEvals::as_array:
//   a_suf, b_suf, c_suf, d_suf, e_suf, a_pre, b_pre, c_pre, d_pre, e_pre.
//
// The sparse proof proves the five suffix claims as one batch and proves b_pre
// as a separate opening. The remaining prefix claims must be zero.
pub const NUM_CLAIMS: usize = 10;
pub const NUM_SUPERGROUPS: usize = 1;

const SUF_CLAIMS: &[usize] = &[0, 1, 2, 3, 4];

struct DebugTimer {
    name: &'static str,
    enabled: bool,
    start: Instant,
    last: Instant,
}

impl DebugTimer {
    fn new(name: &'static str) -> Self {
        let enabled = env::var_os("PR1CS_DEBUG_SPARSE_OPEN").is_some();
        let now = Instant::now();
        DebugTimer {
            name,
            enabled,
            start: now,
            last: now,
        }
    }

    fn log(&mut self, label: &str) {
        if !self.enabled {
            return;
        }
        let now = Instant::now();
        eprintln!(
            "[debug][{}] {}: {:?} (total {:?})",
            self.name,
            label,
            now.duration_since(self.last),
            now.duration_since(self.start)
        );
        self.last = now;
    }
}

#[derive(Clone)]
pub struct SparseSupergroup<F: PrimeField> {
    pub len: usize,
    pub log_row: usize,
    pub log_col: usize,
    pub log_pow: usize,
    pub row: MlPoly<F>,
    pub col: MlPoly<F>,
    pub val: MlPoly<F>,
    pub pow: MlPoly<F>,
    pub count_row: MlPoly<F>,
    pub count_col: MlPoly<F>,
    pub count_pow: MlPoly<F>,
    pub row_idx: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub pow_idx: Vec<usize>,
    // Length of each constituent matrix's entry slice within this supergroup.
    pub matrix_lens: Vec<usize>,
}

#[derive(Clone)]
pub struct BPreGroup<F: PrimeField> {
    pub len: usize,
    pub log_row: usize,
    pub log_col: usize,
    pub log_pow: usize,
    pub row: MlPoly<F>,
    pub col: MlPoly<F>,
    pub val: MlPoly<F>,
    pub pow: MlPoly<F>,
    pub count_row: MlPoly<F>,
    pub count_col: MlPoly<F>,
    pub count_pow: MlPoly<F>,
    pub row_idx: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub pow_idx: Vec<usize>,
}

#[derive(Clone)]
pub struct SparsePolys<F: PrimeField> {
    pub supergroups: Vec<SparseSupergroup<F>>,
    pub b_pre: BPreGroup<F>,
}

#[derive(Clone)]
pub struct SparseSupergroupCommit<E: Pairing> {
    pub len: usize,
    pub log_row: usize,
    pub log_col: usize,
    pub log_pow: usize,
    pub row: MkzgCommit<E>,
    pub col: MkzgCommit<E>,
    pub val: MkzgCommit<E>,
    pub pow: MkzgCommit<E>,
    pub count_row: MkzgCommit<E>,
    pub count_col: MkzgCommit<E>,
    pub count_pow: MkzgCommit<E>,
    pub matrix_lens: Vec<usize>,
}

#[derive(Clone)]
pub struct BPreCommit<E: Pairing> {
    pub len: usize,
    pub log_row: usize,
    pub log_col: usize,
    pub log_pow: usize,
    pub row: MkzgCommit<E>,
    pub col: MkzgCommit<E>,
    pub val: MkzgCommit<E>,
    pub pow: MkzgCommit<E>,
    pub count_row: MkzgCommit<E>,
    pub count_col: MkzgCommit<E>,
    pub count_pow: MkzgCommit<E>,
}

#[derive(Clone)]
pub struct SparseCommits<E: Pairing> {
    pub supergroups: Vec<SparseSupergroupCommit<E>>,
    pub b_pre: BPreCommit<E>,
}

#[derive(Debug, Clone)]
pub struct SparseEvals<F: PrimeField> {
    pub a_suf: F,
    pub b_suf: F,
    pub c_suf: F,
    pub d_suf: F,
    pub e_suf: F,
    pub a_pre: F,
    pub b_pre: F,
    pub c_pre: F,
    pub d_pre: F,
    pub e_pre: F,
}

impl<F: PrimeField> SparseEvals<F> {
    fn as_array(&self) -> [F; NUM_CLAIMS] {
        [
            self.a_suf, self.b_suf, self.c_suf, self.d_suf, self.e_suf, self.a_pre, self.b_pre,
            self.c_pre, self.d_pre, self.e_pre,
        ]
    }

    fn from_array(arr: [F; NUM_CLAIMS]) -> Self {
        SparseEvals {
            a_suf: arr[0],
            b_suf: arr[1],
            c_suf: arr[2],
            d_suf: arr[3],
            e_suf: arr[4],
            a_pre: arr[5],
            b_pre: arr[6],
            c_pre: arr[7],
            d_pre: arr[8],
            e_pre: arr[9],
        }
    }
}

fn log2_ceil(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        (n - 1).ilog2() as usize + 1
    }
}

fn identity_mle<F: Field>(point: &[F]) -> F {
    let mut acc = F::zero();
    let mut pow = F::one();
    let two = F::from(2u8);
    for &x in point {
        acc += x * pow;
        pow *= two;
    }
    acc
}

fn gamma_mle<F: Field>(point: &[F], gamma: F) -> F {
    let mut acc = F::one();
    let mut pow_2j = gamma;
    for &x in point {
        acc *= F::one() + x * (pow_2j - F::one());
        pow_2j = pow_2j * pow_2j;
    }
    acc
}

fn pad_point<F: Field>(point: &[F], len: usize) -> Vec<F> {
    let mut out = point.to_vec();
    out.resize(len, F::zero());
    out
}

fn combined_row_table<F: PrimeField>(
    point_abc: &[F],
    point_de: &[F],
    row_base_log: usize,
) -> Vec<F> {
    let mut abc = MlPoly::new_eq(&pad_point(point_abc, row_base_log)).0;
    let mut de = MlPoly::new_eq(&pad_point(point_de, row_base_log)).0;
    abc.append(&mut de);
    abc
}

fn combined_row_mle<F: PrimeField>(point_abc: &[F], point_de: &[F], point: &[F]) -> F {
    assert!(!point.is_empty());
    let row_base_log = point.len() - 1;
    let lower = point[..row_base_log].to_vec();
    let flag = point[row_base_log];
    let abc = MlPoly::<F>::eval_eq(&pad_point(point_abc, row_base_log), &lower);
    let de = MlPoly::<F>::eval_eq(&pad_point(point_de, row_base_log), &lower);
    abc * (F::one() - flag) + de * flag
}

// MLE of indicator [i < n] at `point`.
fn prefix_mle<F: Field>(point: &[F], n: usize) -> F {
    let nv = point.len();
    if n == 0 {
        return F::zero();
    }
    if n >= (1usize << nv) {
        return F::one();
    }
    let one = F::one();
    let mut result = F::zero();
    let mut carry = one;
    for bit in (0..nv).rev() {
        let b = (n >> bit) & 1;
        if b == 1 {
            result += carry * (one - point[bit]);
            carry *= point[bit];
        } else {
            carry *= one - point[bit];
        }
    }
    result
}

// MLE of the piecewise-constant weight polynomial used by the RLC main
// sumcheck: slice i (cumulative length cum[i]..cum[i+1]) has value r^i.
fn weight_mle<F: Field>(point: &[F], cum: &[usize], r: F) -> F {
    let mut acc = F::zero();
    let mut r_pow = F::one();
    for i in 0..cum.len() - 1 {
        let lo = prefix_mle::<F>(point, cum[i]);
        let hi = prefix_mle::<F>(point, cum[i + 1]);
        acc += r_pow * (hi - lo);
        r_pow *= r;
    }
    acc
}

fn collect_entries<F: PrimeField>(
    mat: &SparseMatrix<F>,
    col_lo: usize,
    col_hi: usize,
) -> Vec<(usize, usize, F, usize)> {
    let mut entries = vec![];
    for cr in 0..mat.rows.len() {
        for &(col, val, pow) in &mat.rows[cr].elems {
            if col >= col_lo && col < col_hi {
                let p = pow.unwrap_or(0);
                entries.push((cr, col - col_lo, val, p));
            }
        }
    }
    entries
}

fn build_supergroup<F: PrimeField>(
    per_matrix: Vec<Vec<(usize, usize, F, usize)>>,
    row_base_log: usize,
    log_col: usize,
    log_pow: usize,
) -> SparseSupergroup<F> {
    let matrix_lens: Vec<usize> = per_matrix.iter().map(|v| v.len()).collect();
    let log_row = row_base_log + 1;
    let mut entries = Vec::new();
    for (matrix_idx, matrix_entries) in per_matrix.into_iter().enumerate() {
        let row_flag = if matrix_idx < 3 {
            0usize
        } else {
            1usize << row_base_log
        };
        entries.extend(
            matrix_entries
                .into_iter()
                .map(|(r, c, v, p)| (row_flag + r, c, v, p)),
        );
    }
    if entries.is_empty() {
        entries.push((0, 0, F::zero(), 0));
    }
    let len = entries.len();
    let mut row = Vec::with_capacity(len);
    let mut col = Vec::with_capacity(len);
    let mut val = Vec::with_capacity(len);
    let mut pow = Vec::with_capacity(len);
    let mut row_idx = Vec::with_capacity(len);
    let mut col_idx = Vec::with_capacity(len);
    let mut pow_idx = Vec::with_capacity(len);
    let mut count_row = vec![F::zero(); 1 << log_row];
    let mut count_col = vec![F::zero(); 1 << log_col];
    let mut count_pow = vec![F::zero(); 1 << log_pow];
    for (r, c, v, p) in entries {
        row.push(F::from(r as u64));
        col.push(F::from(c as u64));
        val.push(v);
        pow.push(F::from(p as u64));
        row_idx.push(r);
        col_idx.push(c);
        pow_idx.push(p);
        count_row[r] += F::one();
        count_col[c] += F::one();
        count_pow[p] += F::one();
    }
    SparseSupergroup {
        len,
        log_row,
        log_col,
        log_pow,
        row: MlPoly::new(row),
        col: MlPoly::new(col),
        val: MlPoly::new(val),
        pow: MlPoly::new(pow),
        count_row: MlPoly::new(count_row),
        count_col: MlPoly::new(count_col),
        count_pow: MlPoly::new(count_pow),
        row_idx,
        col_idx,
        pow_idx,
        matrix_lens,
    }
}

fn build_b_pre_group<F: PrimeField>(
    entries: Vec<(usize, usize, F, usize)>,
    log_row: usize,
    log_col: usize,
    log_pow: usize,
) -> BPreGroup<F> {
    let mut entries = entries;
    if entries.is_empty() {
        entries.push((0, 0, F::zero(), 0));
    }

    let len = entries.len();
    let mut row = Vec::with_capacity(len);
    let mut col = Vec::with_capacity(len);
    let mut val = Vec::with_capacity(len);
    let mut pow = Vec::with_capacity(len);
    let mut row_idx = Vec::with_capacity(len);
    let mut col_idx = Vec::with_capacity(len);
    let mut pow_idx = Vec::with_capacity(len);
    let mut count_row = vec![F::zero(); 1 << log_row];
    let mut count_col = vec![F::zero(); 1 << log_col];
    let mut count_pow = vec![F::zero(); 1 << log_pow];

    for (r, c, v, p) in entries {
        row.push(F::from(r as u64));
        col.push(F::from(c as u64));
        val.push(v);
        pow.push(F::from(p as u64));
        row_idx.push(r);
        col_idx.push(c);
        pow_idx.push(p);
        count_row[r] += F::one();
        count_col[c] += F::one();
        count_pow[p] += F::one();
    }

    BPreGroup {
        len,
        log_row,
        log_col,
        log_pow,
        row: MlPoly::new(row),
        col: MlPoly::new(col),
        val: MlPoly::new(val),
        pow: MlPoly::new(pow),
        count_row: MlPoly::new(count_row),
        count_col: MlPoly::new(count_col),
        count_pow: MlPoly::new(count_pow),
        row_idx,
        col_idx,
        pow_idx,
    }
}

pub fn sparse_commit<E: Pairing>(
    kzg_pp: &MkzgProveParams<E>,
    circuit: &Circuit<E::ScalarField>,
) -> (SparsePolys<E::ScalarField>, SparseCommits<E>) {
    let wei_len = circuit.weight_len;
    let z_len = circuit.z_len;
    let cons_line = circuit.a.len();
    let lu_line = circuit.d.len();

    let log_row_abc = log2_ceil(cons_line);
    let log_row_de = log2_ceil(lu_line);
    let row_base_log = log_row_abc.max(log_row_de);
    let log_col_suf = log2_ceil(z_len - wei_len);
    let log_col_pre = log2_ceil(wei_len);

    let mats = [&circuit.a, &circuit.b, &circuit.c, &circuit.d, &circuit.e];
    let max_pow = mats
        .iter()
        .flat_map(|m| m.rows.iter())
        .flat_map(|r| r.elems.iter())
        .filter_map(|(_, _, p)| *p)
        .max()
        .unwrap_or(0);
    let log_pow = log2_ceil(max_pow + 1);

    let per_matrix: Vec<Vec<(usize, usize, _, usize)>> = (0..5)
        .map(|m| collect_entries(mats[m], wei_len, usize::MAX))
        .collect();
    let supergroups: Vec<SparseSupergroup<_>> = vec![build_supergroup(
        per_matrix,
        row_base_log,
        log_col_suf,
        log_pow,
    )];
    let b_pre = build_b_pre_group(
        collect_entries(&circuit.b, 0, wei_len),
        log_row_abc,
        log_col_pre,
        log_pow,
    );

    // Commit all static preprocessed polys for the single suffix batch group.
    let commits: Vec<SparseSupergroupCommit<E>> = supergroups
        .par_iter()
        .map(|g| SparseSupergroupCommit {
            len: g.len,
            log_row: g.log_row,
            log_col: g.log_col,
            log_pow: g.log_pow,
            row: Mkzg::<E>::commit(kzg_pp, &g.row),
            col: Mkzg::<E>::commit(kzg_pp, &g.col),
            val: Mkzg::<E>::commit(kzg_pp, &g.val),
            pow: Mkzg::<E>::commit(kzg_pp, &g.pow),
            count_row: Mkzg::<E>::commit(kzg_pp, &g.count_row),
            count_col: Mkzg::<E>::commit(kzg_pp, &g.count_col),
            count_pow: Mkzg::<E>::commit(kzg_pp, &g.count_pow),
            matrix_lens: g.matrix_lens.clone(),
        })
        .collect();
    let b_pre_commit = BPreCommit {
        len: b_pre.len,
        log_row: b_pre.log_row,
        log_col: b_pre.log_col,
        log_pow: b_pre.log_pow,
        row: Mkzg::<E>::commit(kzg_pp, &b_pre.row),
        col: Mkzg::<E>::commit(kzg_pp, &b_pre.col),
        val: Mkzg::<E>::commit(kzg_pp, &b_pre.val),
        pow: Mkzg::<E>::commit(kzg_pp, &b_pre.pow),
        count_row: Mkzg::<E>::commit(kzg_pp, &b_pre.count_row),
        count_col: Mkzg::<E>::commit(kzg_pp, &b_pre.count_col),
        count_pow: Mkzg::<E>::commit(kzg_pp, &b_pre.count_pow),
    };

    (
        SparsePolys { supergroups, b_pre },
        SparseCommits {
            supergroups: commits,
            b_pre: b_pre_commit,
        },
    )
}

// ============================================================
// Sumcheck helpers: prover side.
// ============================================================

// Degree-4 product sumcheck:
//     sum_t a[t] * b[t] * c[t] * d[t].
fn sumcheck_main4<E: Pairing>(
    mut a: Vec<E::ScalarField>,
    mut b: Vec<E::ScalarField>,
    mut c: Vec<E::ScalarField>,
    mut d: Vec<E::ScalarField>,
    proof: &mut Proof<E>,
    ro: &mut RandomOracle<E::ScalarField>,
) -> Vec<E::ScalarField> {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), c.len());
    assert_eq!(a.len(), d.len());
    proof.push_u(a.len());
    let log_len = log2_ceil(a.len());
    let mut new_point = vec![];
    for _ in 0..log_len {
        if a.len() % 2 == 1 {
            a.push(E::ScalarField::zero());
            b.push(E::ScalarField::zero());
            c.push(E::ScalarField::zero());
            d.push(E::ScalarField::zero());
        }
        let m = a.len();
        // Degree 4 => 5 evaluations at X = 0, 1, 2, 3, 4.
        let mut sums = [E::ScalarField::zero(); 5];
        for j in (0..m).step_by(2) {
            let da = a[j + 1] - a[j];
            let db = b[j + 1] - b[j];
            let dc = c[j + 1] - c[j];
            let dd = d[j + 1] - d[j];
            sums[0] += a[j] * b[j] * c[j] * d[j];
            sums[1] += a[j + 1] * b[j + 1] * c[j + 1] * d[j + 1];
            let mut av = a[j + 1];
            let mut bv = b[j + 1];
            let mut cv = c[j + 1];
            let mut dv = d[j + 1];
            for k in 2..=4 {
                av += da;
                bv += db;
                cv += dc;
                dv += dd;
                sums[k] += av * bv * cv * dv;
            }
        }
        proof.push_f(&sums);
        let challenge = ro.next_field();
        new_point.push(challenge);
        for i in 0..m / 2 {
            a[i] = a[i * 2] + (a[i * 2 + 1] - a[i * 2]) * challenge;
            b[i] = b[i * 2] + (b[i * 2 + 1] - b[i * 2]) * challenge;
            c[i] = c[i * 2] + (c[i * 2 + 1] - c[i * 2]) * challenge;
            d[i] = d[i * 2] + (d[i * 2 + 1] - d[i * 2]) * challenge;
        }
        a.truncate(m / 2);
        b.truncate(m / 2);
        c.truncate(m / 2);
        d.truncate(m / 2);
    }
    assert_eq!(a.len(), 1);
    proof.push_f(&[a[0], b[0], c[0], d[0]]);
    new_point
}

fn sumcheck_vrho_consistency<E: Pairing>(
    point: &[E::ScalarField],
    mut w: Vec<E::ScalarField>,
    mut val: Vec<E::ScalarField>,
    proof: &mut Proof<E>,
    ro: &mut RandomOracle<E::ScalarField>,
) -> Vec<E::ScalarField> {
    assert_eq!(w.len(), val.len());
    proof.push_u(w.len());
    let log_len = log2_ceil(w.len());
    let mut eq = MlPoly::new_eq(&point.to_vec()).0;
    eq.truncate(w.len());
    let mut new_point = vec![];
    for _ in 0..log_len {
        if w.len() % 2 == 1 {
            eq.push(E::ScalarField::zero());
            w.push(E::ScalarField::zero());
            val.push(E::ScalarField::zero());
        }
        let m = w.len();
        let mut sums = [E::ScalarField::zero(); 4];
        for j in (0..m).step_by(2) {
            let deq = eq[j + 1] - eq[j];
            let dw = w[j + 1] - w[j];
            let dval = val[j + 1] - val[j];
            sums[0] += eq[j] * w[j] * val[j];
            sums[1] += eq[j + 1] * w[j + 1] * val[j + 1];
            sums[2] += (eq[j + 1] + deq) * (w[j + 1] + dw) * (val[j + 1] + dval);
            sums[3] += (eq[j + 1] + deq + deq) * (w[j + 1] + dw + dw) * (val[j + 1] + dval + dval);
        }
        proof.push_f(&sums);
        let challenge = ro.next_field();
        new_point.push(challenge);
        for i in 0..m / 2 {
            eq[i] = eq[i * 2] + (eq[i * 2 + 1] - eq[i * 2]) * challenge;
            w[i] = w[i * 2] + (w[i * 2 + 1] - w[i * 2]) * challenge;
            val[i] = val[i * 2] + (val[i * 2 + 1] - val[i * 2]) * challenge;
        }
        eq.truncate(m / 2);
        w.truncate(m / 2);
        val.truncate(m / 2);
    }
    proof.push_f(&[eq[0], w[0], val[0]]);
    new_point
}

// RLC-combined logup-left sumcheck for 3 independent lookups sharing the same
// alpha: prove
//     sum_t [(er[t]*eir[t]-1) + rho*(ec[t]*eic[t]-1) + rho^2*(eg[t]*eig[t]-1)] * eq[t]
//         + r_sum * (eir[t] + rho*eic[t] + rho^2*eig[t])
//   = r_sum * (S_row + rho*S_col + rho^2*S_pow).
// Degree 3 per round.
#[allow(clippy::too_many_arguments)]
fn combined_logup_left<E: Pairing>(
    mut er: Vec<E::ScalarField>,
    mut eir: Vec<E::ScalarField>,
    mut ec: Vec<E::ScalarField>,
    mut eic: Vec<E::ScalarField>,
    mut eg: Vec<E::ScalarField>,
    mut eig: Vec<E::ScalarField>,
    alpha: E::ScalarField,
    rho: E::ScalarField,
    proof: &mut Proof<E>,
    ro: &mut RandomOracle<E::ScalarField>,
) -> Vec<E::ScalarField> {
    let log_len = log2_ceil(er.len());
    er.iter_mut().for_each(|x| *x += alpha);
    ec.iter_mut().for_each(|x| *x += alpha);
    eg.iter_mut().for_each(|x| *x += alpha);
    proof.push_u(er.len());
    let mut eq = MlPoly::new_eq(&ro.next_n_fields(log_len)).0;
    eq.truncate(er.len());
    let r_sum = ro.next_field();
    let rho2 = rho * rho;
    let one = E::ScalarField::one();

    let contrib = |er: E::ScalarField,
                   eir: E::ScalarField,
                   ec: E::ScalarField,
                   eic: E::ScalarField,
                   eg: E::ScalarField,
                   eig: E::ScalarField,
                   eq: E::ScalarField|
     -> E::ScalarField {
        let prod_row = er * eir - one;
        let prod_col = ec * eic - one;
        let prod_pow = eg * eig - one;
        let inv_comb = eir + rho * eic + rho2 * eig;
        (prod_row + rho * prod_col + rho2 * prod_pow) * eq + r_sum * inv_comb
    };

    let mut new_point = vec![];
    for _ in 0..log_len {
        if er.len() % 2 == 1 {
            er.push(E::ScalarField::zero());
            eir.push(E::ScalarField::zero());
            ec.push(E::ScalarField::zero());
            eic.push(E::ScalarField::zero());
            eg.push(E::ScalarField::zero());
            eig.push(E::ScalarField::zero());
            eq.push(E::ScalarField::zero());
        }
        let m = er.len();
        // Degree 3 (triple products ele * ele_inv * eq).
        let mut sums = [E::ScalarField::zero(); 4];
        for j in (0..m).step_by(2) {
            let der = er[j + 1] - er[j];
            let deir = eir[j + 1] - eir[j];
            let dec = ec[j + 1] - ec[j];
            let deic = eic[j + 1] - eic[j];
            let deg = eg[j + 1] - eg[j];
            let deig = eig[j + 1] - eig[j];
            let deq = eq[j + 1] - eq[j];

            sums[0] += contrib(er[j], eir[j], ec[j], eic[j], eg[j], eig[j], eq[j]);
            sums[1] += contrib(
                er[j + 1],
                eir[j + 1],
                ec[j + 1],
                eic[j + 1],
                eg[j + 1],
                eig[j + 1],
                eq[j + 1],
            );
            let mut ver = er[j + 1];
            let mut veir = eir[j + 1];
            let mut vec_ = ec[j + 1];
            let mut veic = eic[j + 1];
            let mut veg = eg[j + 1];
            let mut veig = eig[j + 1];
            let mut veq = eq[j + 1];
            for k in 2..=3 {
                ver += der;
                veir += deir;
                vec_ += dec;
                veic += deic;
                veg += deg;
                veig += deig;
                veq += deq;
                sums[k] += contrib(ver, veir, vec_, veic, veg, veig, veq);
            }
        }
        proof.push_f(&sums);
        let ch = ro.next_field();
        new_point.push(ch);
        for i in 0..m / 2 {
            er[i] = er[i * 2] + (er[i * 2 + 1] - er[i * 2]) * ch;
            eir[i] = eir[i * 2] + (eir[i * 2 + 1] - eir[i * 2]) * ch;
            ec[i] = ec[i * 2] + (ec[i * 2 + 1] - ec[i * 2]) * ch;
            eic[i] = eic[i * 2] + (eic[i * 2 + 1] - eic[i * 2]) * ch;
            eg[i] = eg[i * 2] + (eg[i * 2 + 1] - eg[i * 2]) * ch;
            eig[i] = eig[i * 2] + (eig[i * 2 + 1] - eig[i * 2]) * ch;
            eq[i] = eq[i * 2] + (eq[i * 2 + 1] - eq[i * 2]) * ch;
        }
        er.truncate(m / 2);
        eir.truncate(m / 2);
        ec.truncate(m / 2);
        eic.truncate(m / 2);
        eg.truncate(m / 2);
        eig.truncate(m / 2);
        eq.truncate(m / 2);
    }
    assert_eq!(er.len(), 1);
    proof.push_f(&[er[0], eir[0], ec[0], eic[0], eg[0], eig[0]]);
    new_point
}

fn logup_sumcheck_right<E: Pairing>(
    mut tab: Vec<E::ScalarField>,
    mut tab_inv: Vec<E::ScalarField>,
    mut count: Vec<E::ScalarField>,
    alpha: E::ScalarField,
    proof: &mut Proof<E>,
    ro: &mut RandomOracle<E::ScalarField>,
) -> Vec<E::ScalarField> {
    let log_len = log2_ceil(tab.len());
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
        let mut sums = [E::ScalarField::zero(); 4];

        for j in (0..m).step_by(2) {
            let diff_tab = tab[j + 1] - tab[j];
            let diff_tab_inv = tab_inv[j + 1] - tab_inv[j];
            let diff_count = count[j + 1] - count[j];
            let diff_eq = eq[j + 1] - eq[j];
            sums[0] +=
                (tab[j] * tab_inv[j] - E::ScalarField::one()) * eq[j] + r * tab_inv[j] * count[j];
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
                + r * (tab_inv[j + 1] + diff_tab_inv + diff_tab_inv)
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

// ============================================================
// Sumcheck helpers: verifier side.
// ============================================================

fn init_base<F: PrimeField>(n: usize) -> Vec<F> {
    let mut res = vec![];
    for i in 0..n + 1 {
        let mut prod = F::one();
        for j in 0..n + 1 {
            if i != j {
                prod *= F::from(i as u32) - F::from(j as u32);
            }
        }
        res.push(prod);
    }
    batch_inverse(&mut res);
    res
}

fn uni_extrapolate<F: PrimeField>(base: &Vec<F>, v: &Vec<F>, x: F) -> F {
    let n = base.len() - 1;
    let mut prod = x;
    for i in 1..n + 1 {
        prod *= x - F::from(i as u32);
    }
    let mut numerator = (0..n + 1)
        .map(|y| x - F::from(y as u32))
        .collect::<Vec<_>>();
    batch_inverse(&mut numerator);
    let mut res = F::zero();
    for i in 0..n + 1 {
        res += numerator[i] * base[i] * v[i];
    }
    res * prod
}

fn verifier_sumcheck<E: Pairing>(
    mut y: E::ScalarField,
    nv: usize,
    degree: usize,
    proof: &mut Proof<E>,
    ro: &mut RandomOracle<E::ScalarField>,
) -> (Vec<E::ScalarField>, E::ScalarField) {
    let base = init_base::<E::ScalarField>(degree);
    let mut new_point = vec![];
    for _ in 0..nv {
        let sums = proof.next_n_fs(degree + 1);
        assert_eq!(sums[0] + sums[1], y);
        let challenge = ro.next_field();
        new_point.push(challenge);
        y = uni_extrapolate(&base, &sums, challenge);
    }
    (new_point, y)
}

// ============================================================
// Prover / verifier accumulators for the final batch KZG open.
// ============================================================

struct ProverAcc<F: PrimeField> {
    polys: Vec<MlPoly<F>>,
    points: Vec<Vec<F>>,
}

impl<F: PrimeField> ProverAcc<F> {
    fn new() -> Self {
        ProverAcc {
            polys: vec![],
            points: vec![],
        }
    }
    fn push(&mut self, poly: MlPoly<F>, point: Vec<F>) {
        self.polys.push(poly);
        self.points.push(point);
    }
}

struct VerifierAcc<E: Pairing> {
    commits: Vec<MkzgCommit<E>>,
    points: Vec<Vec<E::ScalarField>>,
    values: Vec<E::ScalarField>,
}

impl<E: Pairing> VerifierAcc<E> {
    fn new() -> Self {
        VerifierAcc {
            commits: vec![],
            points: vec![],
            values: vec![],
        }
    }
    fn push(&mut self, commit: MkzgCommit<E>, point: Vec<E::ScalarField>, value: E::ScalarField) {
        self.commits.push(commit);
        self.points.push(point);
        self.values.push(value);
    }
}

// ============================================================
// Prover: per-supergroup Lasso.
// ============================================================

#[allow(clippy::too_many_arguments)]
fn lasso_prove_supergroup<E: Pairing>(
    kzg_pp: &MkzgProveParams<E>,
    sg: &SparseSupergroup<E::ScalarField>,
    point_abc: &[E::ScalarField],
    point_de: &[E::ScalarField],
    col_eq_point: &[E::ScalarField],
    gamma: E::ScalarField,
    claims: &[E::ScalarField],
    proof: &mut Proof<E>,
    ro: &mut RandomOracle<E::ScalarField>,
    acc: &mut ProverAcc<E::ScalarField>,
) {
    assert_eq!(col_eq_point.len(), sg.log_col);
    assert_eq!(claims.len(), sg.matrix_lens.len());

    let row_base_log = sg.log_row - 1;
    let row_eq = combined_row_table(point_abc, point_de, row_base_log);
    let col_eq = MlPoly::new_eq(&col_eq_point.to_vec()).0;

    let gamma_table: Vec<E::ScalarField> = {
        let mut out = Vec::with_capacity(1usize << sg.log_pow);
        let mut cur = E::ScalarField::one();
        for _ in 0..(1usize << sg.log_pow) {
            out.push(cur);
            cur *= gamma;
        }
        out
    };

    // Table reads — parallel over entries.
    let e_row: Vec<E::ScalarField> = sg.row_idx.par_iter().map(|&i| row_eq[i]).collect();
    let e_col: Vec<E::ScalarField> = sg.col_idx.par_iter().map(|&i| col_eq[i]).collect();
    let e_gamma: Vec<E::ScalarField> = sg.pow_idx.par_iter().map(|&i| gamma_table[i]).collect();

    let e_row_poly = MlPoly::new(e_row.clone());
    let e_col_poly = MlPoly::new(e_col.clone());
    let e_gamma_poly = MlPoly::new(e_gamma.clone());
    // Commit 3 table-read polys (parallel MSMs).
    let e_commits: Vec<MkzgCommit<E>> = [&e_row_poly, &e_col_poly, &e_gamma_poly]
        .par_iter()
        .map(|p| Mkzg::<E>::commit(kzg_pp, p))
        .collect();
    for c in &e_commits {
        proof.push_u(c.0.len());
        proof.push_gs(&c.0);
    }

    // Main sumcheck: RLC the `k` matrix claims into a single sumcheck.
    // The RLC weight is multiplied into a dynamic V_rho polynomial, reducing
    // the product degree from 5 to 4.
    let r_m = ro.next_field();
    let mut cum = vec![0usize];
    for &ml in &sg.matrix_lens {
        cum.push(cum.last().unwrap() + ml);
    }
    let mut w_vec: Vec<E::ScalarField> = Vec::with_capacity(sg.len);
    let mut v_rho_vec: Vec<E::ScalarField> = Vec::with_capacity(sg.len);
    {
        let mut r_pow = E::ScalarField::one();
        for (i, &ml) in sg.matrix_lens.iter().enumerate() {
            for j in 0..ml {
                w_vec.push(r_pow);
                v_rho_vec.push(sg.val.0[cum[i] + j] * r_pow);
            }
            if i + 1 < sg.matrix_lens.len() {
                r_pow *= r_m;
            }
        }
        // Pad with zeros to match sg.len (for the dummy-zero case).
        while v_rho_vec.len() < sg.len {
            w_vec.push(E::ScalarField::zero());
            v_rho_vec.push(E::ScalarField::zero());
        }
    }
    let v_rho_poly = MlPoly::new(v_rho_vec.clone());
    let v_rho_commit = Mkzg::<E>::commit(kzg_pp, &v_rho_poly);
    proof.push_u(v_rho_commit.0.len());
    proof.push_gs(&v_rho_commit.0);

    let point_main = sumcheck_main4::<E>(
        e_row.clone(),
        e_col.clone(),
        e_gamma.clone(),
        v_rho_vec,
        proof,
        ro,
    );
    let point_vrho =
        sumcheck_vrho_consistency::<E>(&point_main, w_vec, sg.val.0.clone(), proof, ro);

    acc.push(e_row_poly.clone(), point_main.clone());
    acc.push(e_col_poly.clone(), point_main.clone());
    acc.push(e_gamma_poly.clone(), point_main.clone());
    acc.push(v_rho_poly, point_main);
    acc.push(sg.val.clone(), point_vrho);

    // Memory-check setup: sample three betas, then alpha, then rho for RLC.
    let beta_row = ro.next_field();
    let beta_col = ro.next_field();
    let beta_pow = ro.next_field();
    let alpha = ro.next_field();
    let rho = ro.next_field();

    // Compute ele_X and ele_inv_X.
    let ele_row: Vec<E::ScalarField> = (0..sg.len)
        .into_par_iter()
        .map(|k| sg.row.0[k] + beta_row * e_row[k])
        .collect();
    let ele_col: Vec<E::ScalarField> = (0..sg.len)
        .into_par_iter()
        .map(|k| sg.col.0[k] + beta_col * e_col[k])
        .collect();
    let ele_pow: Vec<E::ScalarField> = (0..sg.len)
        .into_par_iter()
        .map(|k| sg.pow.0[k] + beta_pow * e_gamma[k])
        .collect();

    let mut eir: Vec<E::ScalarField> = ele_row.iter().map(|&x| x + alpha).collect();
    batch_inverse(&mut eir);
    let mut eic: Vec<E::ScalarField> = ele_col.iter().map(|&x| x + alpha).collect();
    batch_inverse(&mut eic);
    let mut eig: Vec<E::ScalarField> = ele_pow.iter().map(|&x| x + alpha).collect();
    batch_inverse(&mut eig);

    let eir_poly = MlPoly::new(eir.clone());
    let eic_poly = MlPoly::new(eic.clone());
    let eig_poly = MlPoly::new(eig.clone());

    // Commit the three ele_inv polys (parallel MSMs).
    let ele_inv_commits: Vec<MkzgCommit<E>> = [&eir_poly, &eic_poly, &eig_poly]
        .par_iter()
        .map(|p| Mkzg::<E>::commit(kzg_pp, p))
        .collect();
    for c in &ele_inv_commits {
        proof.push_u(c.0.len());
        proof.push_gs(&c.0);
    }

    let s_row: E::ScalarField = eir.iter().copied().sum();
    let s_col: E::ScalarField = eic.iter().copied().sum();
    let s_pow: E::ScalarField = eig.iter().copied().sum();
    proof.push_f(&[s_row, s_col, s_pow]);

    // Combined logup-left sumcheck.
    let point_l = combined_logup_left::<E>(
        ele_row,
        eir.clone(),
        ele_col,
        eic.clone(),
        ele_pow,
        eig.clone(),
        alpha,
        rho,
        proof,
        ro,
    );

    // Open idx and e polys at point_l.
    let row_idx_v = sg.row.clone().eval(&point_l);
    let col_idx_v = sg.col.clone().eval(&point_l);
    let pow_idx_v = sg.pow.clone().eval(&point_l);
    let e_row_v_l = e_row_poly.clone().eval(&point_l);
    let e_col_v_l = e_col_poly.clone().eval(&point_l);
    let e_gamma_v_l = e_gamma_poly.clone().eval(&point_l);
    proof.push_f(&[
        row_idx_v,
        col_idx_v,
        pow_idx_v,
        e_row_v_l,
        e_col_v_l,
        e_gamma_v_l,
    ]);

    acc.push(sg.row.clone(), point_l.clone());
    acc.push(sg.col.clone(), point_l.clone());
    acc.push(sg.pow.clone(), point_l.clone());
    acc.push(e_row_poly, point_l.clone());
    acc.push(e_col_poly, point_l.clone());
    acc.push(e_gamma_poly, point_l.clone());
    acc.push(eir_poly, point_l.clone());
    acc.push(eic_poly, point_l.clone());
    acc.push(eig_poly, point_l);

    // Three memory-right sumchecks (one per table). Work in parallel on tab and tab_inv
    // construction, but each sumcheck itself must run sequentially (shared RO & proof).
    let build_right = |table: &[E::ScalarField],
                       beta: E::ScalarField|
     -> (Vec<E::ScalarField>, Vec<E::ScalarField>) {
        let tab: Vec<E::ScalarField> = (0..table.len())
            .into_par_iter()
            .map(|i| E::ScalarField::from(i as u64) + beta * table[i])
            .collect();
        let mut tab_inv: Vec<E::ScalarField> = tab.iter().map(|&x| x + alpha).collect();
        batch_inverse(&mut tab_inv);
        (tab, tab_inv)
    };

    let right_inputs = vec![
        (&row_eq[..], beta_row, &sg.count_row),
        (&col_eq[..], beta_col, &sg.count_col),
        (&gamma_table[..], beta_pow, &sg.count_pow),
    ];
    // Precompute tab and tab_inv for all 3 tables in parallel.
    let right_tabs: Vec<(Vec<E::ScalarField>, Vec<E::ScalarField>)> = right_inputs
        .par_iter()
        .map(|(t, b, _)| build_right(t, *b))
        .collect();

    // Commit tab_inv polys in parallel.
    let tab_inv_polys: Vec<MlPoly<E::ScalarField>> = right_tabs
        .iter()
        .map(|(_, ti)| MlPoly::new(ti.clone()))
        .collect();
    let tab_inv_commits: Vec<MkzgCommit<E>> = tab_inv_polys
        .par_iter()
        .map(|p| Mkzg::<E>::commit(kzg_pp, p))
        .collect();
    for c in &tab_inv_commits {
        proof.push_u(c.0.len());
        proof.push_gs(&c.0);
    }

    // Now run the 3 right-side sumchecks sequentially, interleaving
    // tab_inv/count opens to match the verifier's push order.
    for ((tab, tab_inv), (tip, (_, _, count_poly))) in right_tabs
        .into_iter()
        .zip(tab_inv_polys.into_iter().zip(right_inputs.iter()))
    {
        let point_r =
            logup_sumcheck_right::<E>(tab, tab_inv, count_poly.0.clone(), alpha, proof, ro);
        acc.push(tip, point_r.clone());
        acc.push((*count_poly).clone(), point_r);
    }
}

#[allow(clippy::too_many_arguments)]
fn prove_b_pre<E: Pairing>(
    kzg_pp: &MkzgProveParams<E>,
    group: &BPreGroup<E::ScalarField>,
    row_eq_point: &[E::ScalarField],
    col_eq_point: &[E::ScalarField],
    gamma: E::ScalarField,
    claim: E::ScalarField,
    proof: &mut Proof<E>,
    ro: &mut RandomOracle<E::ScalarField>,
    acc: &mut ProverAcc<E::ScalarField>,
) {
    let mut timer = DebugTimer::new("sparse_open::b_pre");
    assert_eq!(row_eq_point.len(), group.log_row);
    assert_eq!(col_eq_point.len(), group.log_col);

    let row_eq = MlPoly::new_eq(&row_eq_point.to_vec()).0;
    let col_eq = MlPoly::new_eq(&col_eq_point.to_vec()).0;
    let gamma_table: Vec<E::ScalarField> = {
        let mut out = Vec::with_capacity(1usize << group.log_pow);
        let mut cur = E::ScalarField::one();
        for _ in 0..(1usize << group.log_pow) {
            out.push(cur);
            cur *= gamma;
        }
        out
    };
    timer.log("build eq tables");

    let e_row: Vec<E::ScalarField> = group.row_idx.par_iter().map(|&i| row_eq[i]).collect();
    let e_col: Vec<E::ScalarField> = group.col_idx.par_iter().map(|&i| col_eq[i]).collect();
    let e_gamma: Vec<E::ScalarField> = group.pow_idx.par_iter().map(|&i| gamma_table[i]).collect();
    timer.log("materialize table reads");

    let e_row_poly = MlPoly::new(e_row.clone());
    let e_col_poly = MlPoly::new(e_col.clone());
    let e_gamma_poly = MlPoly::new(e_gamma.clone());
    let e_commits: Vec<MkzgCommit<E>> = [&e_row_poly, &e_col_poly, &e_gamma_poly]
        .par_iter()
        .map(|p| Mkzg::<E>::commit(kzg_pp, p))
        .collect();
    for c in &e_commits {
        proof.push_u(c.0.len());
        proof.push_gs(&c.0);
    }
    timer.log("commit read polys");
    let direct_claim: E::ScalarField = (0..group.len)
        .into_par_iter()
        .map(|i| e_row[i] * e_col[i] * e_gamma[i] * group.val.0[i])
        .sum();
    assert_eq!(direct_claim, claim);
    timer.log("compute direct claim");

    let point_main = sumcheck_main4::<E>(
        e_row.clone(),
        e_col.clone(),
        e_gamma.clone(),
        group.val.0.clone(),
        proof,
        ro,
    );
    timer.log("main sumcheck");
    acc.push(e_row_poly.clone(), point_main.clone());
    acc.push(e_col_poly.clone(), point_main.clone());
    acc.push(e_gamma_poly.clone(), point_main.clone());
    acc.push(group.val.clone(), point_main);

    let beta_row = ro.next_field();
    let beta_col = ro.next_field();
    let beta_pow = ro.next_field();
    let alpha = ro.next_field();
    let rho = ro.next_field();
    timer.log("sample logup challenges");

    let ele_row: Vec<E::ScalarField> = (0..group.len)
        .into_par_iter()
        .map(|k| group.row.0[k] + beta_row * e_row[k])
        .collect();
    let ele_col: Vec<E::ScalarField> = (0..group.len)
        .into_par_iter()
        .map(|k| group.col.0[k] + beta_col * e_col[k])
        .collect();
    let ele_pow: Vec<E::ScalarField> = (0..group.len)
        .into_par_iter()
        .map(|k| group.pow.0[k] + beta_pow * e_gamma[k])
        .collect();
    timer.log("build left logup inputs");

    let mut ei_row: Vec<E::ScalarField> = ele_row.iter().map(|&x| x + alpha).collect();
    batch_inverse(&mut ei_row);
    let mut ei_col: Vec<E::ScalarField> = ele_col.iter().map(|&x| x + alpha).collect();
    batch_inverse(&mut ei_col);
    let mut ei_pow: Vec<E::ScalarField> = ele_pow.iter().map(|&x| x + alpha).collect();
    batch_inverse(&mut ei_pow);
    timer.log("batch invert left inputs");

    let ei_row_poly = MlPoly::new(ei_row.clone());
    let ei_col_poly = MlPoly::new(ei_col.clone());
    let ei_pow_poly = MlPoly::new(ei_pow.clone());
    let inv_commits: Vec<MkzgCommit<E>> = [&ei_row_poly, &ei_col_poly, &ei_pow_poly]
        .par_iter()
        .map(|p| Mkzg::<E>::commit(kzg_pp, p))
        .collect();
    for c in &inv_commits {
        proof.push_u(c.0.len());
        proof.push_gs(&c.0);
    }
    timer.log("commit inverse polys");

    let s_row: E::ScalarField = ei_row.iter().copied().sum();
    let s_col: E::ScalarField = ei_col.iter().copied().sum();
    let s_pow: E::ScalarField = ei_pow.iter().copied().sum();
    proof.push_f(&[s_row, s_col, s_pow]);
    timer.log("accumulate inverse sums");

    let point_l = combined_logup_left::<E>(
        ele_row,
        ei_row.clone(),
        ele_col,
        ei_col.clone(),
        ele_pow,
        ei_pow.clone(),
        alpha,
        rho,
        proof,
        ro,
    );
    timer.log("left logup sumcheck");

    let row_idx_v = group.row.clone().eval(&point_l);
    let col_idx_v = group.col.clone().eval(&point_l);
    let pow_idx_v = group.pow.clone().eval(&point_l);
    let e_row_v_l = e_row_poly.clone().eval(&point_l);
    let e_col_v_l = e_col_poly.clone().eval(&point_l);
    let e_gamma_v_l = e_gamma_poly.clone().eval(&point_l);
    proof.push_f(&[
        row_idx_v,
        col_idx_v,
        pow_idx_v,
        e_row_v_l,
        e_col_v_l,
        e_gamma_v_l,
    ]);
    timer.log("evaluate left point");

    acc.push(group.row.clone(), point_l.clone());
    acc.push(group.col.clone(), point_l.clone());
    acc.push(group.pow.clone(), point_l.clone());
    acc.push(e_row_poly, point_l.clone());
    acc.push(e_col_poly, point_l.clone());
    acc.push(e_gamma_poly, point_l.clone());
    acc.push(ei_row_poly, point_l.clone());
    acc.push(ei_col_poly, point_l.clone());
    acc.push(ei_pow_poly, point_l);
    timer.log("queue left batch opens");

    let build_right = |table: &[E::ScalarField],
                       beta: E::ScalarField|
     -> (Vec<E::ScalarField>, Vec<E::ScalarField>) {
        let tab: Vec<E::ScalarField> = (0..table.len())
            .into_par_iter()
            .map(|i| E::ScalarField::from(i as u64) + beta * table[i])
            .collect();
        let mut tab_inv: Vec<E::ScalarField> = tab.iter().map(|&x| x + alpha).collect();
        batch_inverse(&mut tab_inv);
        (tab, tab_inv)
    };
    let right_inputs = vec![
        (&row_eq[..], beta_row, &group.count_row),
        (&col_eq[..], beta_col, &group.count_col),
        (&gamma_table[..], beta_pow, &group.count_pow),
    ];
    let right_tabs: Vec<(Vec<E::ScalarField>, Vec<E::ScalarField>)> = right_inputs
        .par_iter()
        .map(|(t, b, _)| build_right(t, *b))
        .collect();
    timer.log("build right logup tables");
    let tab_inv_polys: Vec<MlPoly<E::ScalarField>> = right_tabs
        .iter()
        .map(|(_, ti)| MlPoly::new(ti.clone()))
        .collect();
    let tab_inv_commits: Vec<MkzgCommit<E>> = tab_inv_polys
        .par_iter()
        .map(|p| Mkzg::<E>::commit(kzg_pp, p))
        .collect();
    for c in &tab_inv_commits {
        proof.push_u(c.0.len());
        proof.push_gs(&c.0);
    }
    timer.log("commit right inverse polys");
    for ((tab, tab_inv), (tip, (_, _, count_poly))) in right_tabs
        .into_iter()
        .zip(tab_inv_polys.into_iter().zip(right_inputs.iter()))
    {
        let point_r =
            logup_sumcheck_right::<E>(tab, tab_inv, count_poly.0.clone(), alpha, proof, ro);
        acc.push(tip, point_r.clone());
        acc.push((*count_poly).clone(), point_r);
    }
    timer.log("right logup sumchecks");
}

pub fn sparse_open<E: Pairing>(
    kzg_pp: &MkzgProveParams<E>,
    circuit: &Circuit<E::ScalarField>,
    polys: &SparsePolys<E::ScalarField>,
    point_abc: &[E::ScalarField],
    point_de: &[E::ScalarField],
    point_suf: &[E::ScalarField],
    point_pre: &[E::ScalarField],
    gamma: E::ScalarField,
    proof: &mut Proof<E>,
    ro: &mut RandomOracle<E::ScalarField>,
) -> SparseEvals<E::ScalarField> {
    let mut timer = DebugTimer::new("sparse_open");
    // Compute the 10 claimed MLE values directly.
    let row_eq_abc = MlPoly::new_eq(&point_abc.to_vec()).0;
    let row_eq_de = MlPoly::new_eq(&point_de.to_vec()).0;
    let col_eq_suf = MlPoly::new_eq(&point_suf.to_vec()).0;
    let col_eq_pre = MlPoly::new_eq(&point_pre.to_vec()).0;
    let wei_len = circuit.weight_len;

    let sparse_evals = SparseEvals {
        a_suf: circuit
            .a
            .mle(&row_eq_abc, &col_eq_suf, wei_len, usize::MAX, gamma),
        b_suf: circuit
            .b
            .mle(&row_eq_abc, &col_eq_suf, wei_len, usize::MAX, gamma),
        c_suf: circuit
            .c
            .mle(&row_eq_abc, &col_eq_suf, wei_len, usize::MAX, gamma),
        d_suf: circuit
            .d
            .mle(&row_eq_de, &col_eq_suf, wei_len, usize::MAX, gamma),
        e_suf: circuit
            .e
            .mle(&row_eq_de, &col_eq_suf, wei_len, usize::MAX, gamma),
        a_pre: circuit.a.mle(&row_eq_abc, &col_eq_pre, 0, wei_len, gamma),
        b_pre: circuit.b.mle(&row_eq_abc, &col_eq_pre, 0, wei_len, gamma),
        c_pre: circuit.c.mle(&row_eq_abc, &col_eq_pre, 0, wei_len, gamma),
        d_pre: circuit.d.mle(&row_eq_de, &col_eq_pre, 0, wei_len, gamma),
        e_pre: circuit.e.mle(&row_eq_de, &col_eq_pre, 0, wei_len, gamma),
    };
    timer.log("compute direct evals");

    // Push the 10 claimed values first.
    proof.push_f(&sparse_evals.as_array());
    timer.log("push direct evals");

    let claim_array = sparse_evals.as_array();
    let mut acc = ProverAcc::<E::ScalarField>::new();

    let sg = &polys.supergroups[0];
    let claims: Vec<E::ScalarField> = SUF_CLAIMS.iter().map(|&i| claim_array[i]).collect();
    lasso_prove_supergroup::<E>(
        kzg_pp, sg, point_abc, point_de, point_suf, gamma, &claims, proof, ro, &mut acc,
    );
    timer.log("prove suffix supergroup");
    prove_b_pre::<E>(
        kzg_pp,
        &polys.b_pre,
        point_abc,
        point_pre,
        gamma,
        sparse_evals.b_pre,
        proof,
        ro,
        &mut acc,
    );
    timer.log("prove b_pre");

    let (kzg_proof, sumcheck_proof) = Mkzg::<E>::batch_open(kzg_pp, &acc.polys, &acc.points, ro);
    proof.push_u(kzg_proof.0.len());
    proof.push_gs(&kzg_proof.0);
    proof.push_u(sumcheck_proof.0.len());
    proof.push_f(&sumcheck_proof.0);
    timer.log("final batch open");

    sparse_evals
}

// ============================================================
// Verifier: per-supergroup Lasso.
// ============================================================

#[allow(clippy::too_many_arguments)]
fn lasso_verify_supergroup<E: Pairing>(
    sg_commit: &SparseSupergroupCommit<E>,
    point_abc: &[E::ScalarField],
    point_de: &[E::ScalarField],
    col_eq_point: &[E::ScalarField],
    gamma: E::ScalarField,
    claims: &[E::ScalarField],
    proof: &mut Proof<E>,
    ro: &mut RandomOracle<E::ScalarField>,
    acc: &mut VerifierAcc<E>,
) {
    assert_eq!(col_eq_point.len(), sg_commit.log_col);
    assert_eq!(claims.len(), sg_commit.matrix_lens.len());

    // Read e_row, e_col, e_gamma commits.
    let read_commit = |proof: &mut Proof<E>| {
        let l = proof.next_u();
        MkzgCommit(proof.next_n_gs(l))
    };
    let e_row_commit = read_commit(proof);
    let e_col_commit = read_commit(proof);
    let e_gamma_commit = read_commit(proof);

    // RLC challenge for matrix claims. The prover commits V_rho after rho is
    // sampled, then proves V_rho = W_rho * val at the main sumcheck point.
    let r_m = ro.next_field();
    let v_rho_commit = read_commit(proof);
    let mut cum = vec![0usize];
    for &ml in &sg_commit.matrix_lens {
        cum.push(cum.last().unwrap() + ml);
    }
    let mut combined_claim = E::ScalarField::zero();
    {
        let mut r_pow = E::ScalarField::one();
        for &c in claims {
            combined_claim += r_pow * c;
            r_pow *= r_m;
        }
    }

    let len_main = proof.next_u();
    let nv_main = log2_ceil(len_main);
    let (point_main, y_main) = verifier_sumcheck::<E>(combined_claim, nv_main, 4, proof, ro);
    let e_row_v = proof.next_f();
    let e_col_v = proof.next_f();
    let e_gamma_v = proof.next_f();
    let v_rho_v = proof.next_f();
    assert_eq!(y_main, e_row_v * e_col_v * e_gamma_v * v_rho_v);

    let len_vrho = proof.next_u();
    let nv_vrho = log2_ceil(len_vrho);
    assert_eq!(nv_vrho, nv_main);
    let (point_vrho, y_vrho) = verifier_sumcheck::<E>(v_rho_v, nv_vrho, 3, proof, ro);
    let eq_v = proof.next_f();
    let w_v = proof.next_f();
    let val_v = proof.next_f();
    assert_eq!(
        eq_v,
        MlPoly::eval_eq_pref(&point_main, &point_vrho, len_vrho)
    );
    assert_eq!(w_v, weight_mle::<E::ScalarField>(&point_vrho, &cum, r_m));
    assert_eq!(y_vrho, eq_v * w_v * val_v);

    acc.push(e_row_commit.clone(), point_main.clone(), e_row_v);
    acc.push(e_col_commit.clone(), point_main.clone(), e_col_v);
    acc.push(e_gamma_commit.clone(), point_main.clone(), e_gamma_v);
    acc.push(v_rho_commit, point_main, v_rho_v);
    acc.push(sg_commit.val.clone(), point_vrho, val_v);

    // Memory-check setup — mirror prover order.
    let beta_row = ro.next_field();
    let beta_col = ro.next_field();
    let beta_pow = ro.next_field();
    let alpha = ro.next_field();
    let rho = ro.next_field();

    // Read ele_inv commits.
    let eir_commit = read_commit(proof);
    let eic_commit = read_commit(proof);
    let eig_commit = read_commit(proof);

    let s_row = proof.next_f();
    let s_col = proof.next_f();
    let s_pow = proof.next_f();

    // Combined logup-left verification.
    let len_l = proof.next_u();
    let nv_l = log2_ceil(len_l);
    let r_eq_l = ro.next_n_fields(nv_l);
    let r_sum_l = ro.next_field();
    let rho2 = rho * rho;
    let y0 = r_sum_l * (s_row + rho * s_col + rho2 * s_pow);
    let (point_l, y_l) = verifier_sumcheck::<E>(y0, nv_l, 3, proof, ro);
    let er_v = proof.next_f();
    let eir_v = proof.next_f();
    let ec_v = proof.next_f();
    let eic_v = proof.next_f();
    let eg_v = proof.next_f();
    let eig_v = proof.next_f();
    let eq_pref = MlPoly::eval_eq_pref(&r_eq_l, &point_l, len_l);
    let one = E::ScalarField::one();
    let combined_rhs =
        ((er_v * eir_v - one) + rho * (ec_v * eic_v - one) + rho2 * (eg_v * eig_v - one)) * eq_pref
            + r_sum_l * (eir_v + rho * eic_v + rho2 * eig_v);
    assert_eq!(y_l, combined_rhs);

    let row_idx_v = proof.next_f();
    let col_idx_v = proof.next_f();
    let pow_idx_v = proof.next_f();
    let e_row_v_l = proof.next_f();
    let e_col_v_l = proof.next_f();
    let e_gamma_v_l = proof.next_f();

    // ele_X = alpha * prefix_mle(T) + idx_X + beta_X * e_X.
    let prefix_v = prefix_mle::<E::ScalarField>(&point_l, len_l);
    assert_eq!(er_v, alpha * prefix_v + row_idx_v + beta_row * e_row_v_l);
    assert_eq!(ec_v, alpha * prefix_v + col_idx_v + beta_col * e_col_v_l);
    assert_eq!(eg_v, alpha * prefix_v + pow_idx_v + beta_pow * e_gamma_v_l);

    acc.push(sg_commit.row.clone(), point_l.clone(), row_idx_v);
    acc.push(sg_commit.col.clone(), point_l.clone(), col_idx_v);
    acc.push(sg_commit.pow.clone(), point_l.clone(), pow_idx_v);
    acc.push(e_row_commit, point_l.clone(), e_row_v_l);
    acc.push(e_col_commit, point_l.clone(), e_col_v_l);
    acc.push(e_gamma_commit, point_l.clone(), e_gamma_v_l);
    acc.push(eir_commit, point_l.clone(), eir_v);
    acc.push(eic_commit, point_l.clone(), eic_v);
    acc.push(eig_commit, point_l, eig_v);

    // Three memory-right verifications.
    let tab_inv_row_commit = read_commit(proof);
    let tab_inv_col_commit = read_commit(proof);
    let tab_inv_gamma_commit = read_commit(proof);

    // Row.
    verify_memory_right::<E>(
        proof,
        ro,
        alpha,
        beta_row,
        s_row,
        sg_commit.log_row,
        |pt| combined_row_mle::<E::ScalarField>(point_abc, point_de, pt),
        tab_inv_row_commit,
        sg_commit.count_row.clone(),
        acc,
    );
    // Col.
    verify_memory_right::<E>(
        proof,
        ro,
        alpha,
        beta_col,
        s_col,
        sg_commit.log_col,
        |pt| MlPoly::<E::ScalarField>::eval_eq(&col_eq_point.to_vec(), &pt.to_vec()),
        tab_inv_col_commit,
        sg_commit.count_col.clone(),
        acc,
    );
    // Gamma.
    verify_memory_right::<E>(
        proof,
        ro,
        alpha,
        beta_pow,
        s_pow,
        sg_commit.log_pow,
        |pt| gamma_mle::<E::ScalarField>(pt, gamma),
        tab_inv_gamma_commit,
        sg_commit.count_pow.clone(),
        acc,
    );
}

#[allow(clippy::too_many_arguments)]
fn verify_b_pre<E: Pairing>(
    commit: &BPreCommit<E>,
    row_eq_point: &[E::ScalarField],
    col_eq_point: &[E::ScalarField],
    gamma: E::ScalarField,
    claim: E::ScalarField,
    proof: &mut Proof<E>,
    ro: &mut RandomOracle<E::ScalarField>,
    acc: &mut VerifierAcc<E>,
) {
    assert_eq!(row_eq_point.len(), commit.log_row);
    assert_eq!(col_eq_point.len(), commit.log_col);

    let read_commit = |proof: &mut Proof<E>| {
        let l = proof.next_u();
        MkzgCommit(proof.next_n_gs(l))
    };
    let e_row_commit = read_commit(proof);
    let e_col_commit = read_commit(proof);
    let e_gamma_commit = read_commit(proof);

    let len_main = proof.next_u();
    let nv_main = log2_ceil(len_main);
    let (point_main, y_main) = verifier_sumcheck::<E>(claim, nv_main, 4, proof, ro);
    let e_row_v = proof.next_f();
    let e_col_v = proof.next_f();
    let e_gamma_v = proof.next_f();
    let val_v = proof.next_f();
    assert_eq!(y_main, e_row_v * e_col_v * e_gamma_v * val_v);

    acc.push(e_row_commit.clone(), point_main.clone(), e_row_v);
    acc.push(e_col_commit.clone(), point_main.clone(), e_col_v);
    acc.push(e_gamma_commit.clone(), point_main.clone(), e_gamma_v);
    acc.push(commit.val.clone(), point_main, val_v);

    let beta_row = ro.next_field();
    let beta_col = ro.next_field();
    let beta_pow = ro.next_field();
    let alpha = ro.next_field();
    let rho = ro.next_field();

    let ei_row_commit = read_commit(proof);
    let ei_col_commit = read_commit(proof);
    let ei_pow_commit = read_commit(proof);

    let s_row = proof.next_f();
    let s_col = proof.next_f();
    let s_pow = proof.next_f();

    let len_l = proof.next_u();
    let nv_l = log2_ceil(len_l);
    let r_eq_l = ro.next_n_fields(nv_l);
    let r_sum_l = ro.next_field();
    let rho2 = rho * rho;
    let y0 = r_sum_l * (s_row + rho * s_col + rho2 * s_pow);
    let (point_l, y_l) = verifier_sumcheck::<E>(y0, nv_l, 3, proof, ro);
    let e0_v = proof.next_f();
    let ei0_v = proof.next_f();
    let e1_v = proof.next_f();
    let ei1_v = proof.next_f();
    let e2_v = proof.next_f();
    let ei2_v = proof.next_f();
    let eq_pref = MlPoly::eval_eq_pref(&r_eq_l, &point_l, len_l);
    let one = E::ScalarField::one();
    let rhs = ((e0_v * ei0_v - one) + rho * (e1_v * ei1_v - one) + rho2 * (e2_v * ei2_v - one))
        * eq_pref
        + r_sum_l * (ei0_v + rho * ei1_v + rho2 * ei2_v);
    assert_eq!(y_l, rhs);

    let row_idx_v = proof.next_f();
    let col_idx_v = proof.next_f();
    let pow_idx_v = proof.next_f();
    let e_row_v_l = proof.next_f();
    let e_col_v_l = proof.next_f();
    let e_gamma_v_l = proof.next_f();

    let prefix_v = prefix_mle::<E::ScalarField>(&point_l, len_l);
    assert_eq!(e0_v, alpha * prefix_v + row_idx_v + beta_row * e_row_v_l);
    assert_eq!(e1_v, alpha * prefix_v + col_idx_v + beta_col * e_col_v_l);
    assert_eq!(e2_v, alpha * prefix_v + pow_idx_v + beta_pow * e_gamma_v_l);

    acc.push(commit.row.clone(), point_l.clone(), row_idx_v);
    acc.push(commit.col.clone(), point_l.clone(), col_idx_v);
    acc.push(commit.pow.clone(), point_l.clone(), pow_idx_v);
    acc.push(e_row_commit, point_l.clone(), e_row_v_l);
    acc.push(e_col_commit, point_l.clone(), e_col_v_l);
    acc.push(e_gamma_commit, point_l.clone(), e_gamma_v_l);
    acc.push(ei_row_commit, point_l.clone(), ei0_v);
    acc.push(ei_col_commit, point_l.clone(), ei1_v);
    acc.push(ei_pow_commit, point_l, ei2_v);

    let tab_inv_row_commit = read_commit(proof);
    let tab_inv_col_commit = read_commit(proof);
    let tab_inv_pow_commit = read_commit(proof);

    verify_memory_right::<E>(
        proof,
        ro,
        alpha,
        beta_row,
        s_row,
        commit.log_row,
        |pt| MlPoly::<E::ScalarField>::eval_eq(&row_eq_point.to_vec(), &pt.to_vec()),
        tab_inv_row_commit,
        commit.count_row.clone(),
        acc,
    );
    verify_memory_right::<E>(
        proof,
        ro,
        alpha,
        beta_col,
        s_col,
        commit.log_col,
        |pt| MlPoly::<E::ScalarField>::eval_eq(&col_eq_point.to_vec(), &pt.to_vec()),
        tab_inv_col_commit,
        commit.count_col.clone(),
        acc,
    );
    verify_memory_right::<E>(
        proof,
        ro,
        alpha,
        beta_pow,
        s_pow,
        commit.log_pow,
        |pt| gamma_mle::<E::ScalarField>(pt, gamma),
        tab_inv_pow_commit,
        commit.count_pow.clone(),
        acc,
    );
}

#[allow(clippy::too_many_arguments)]
fn verify_memory_right<E: Pairing>(
    proof: &mut Proof<E>,
    ro: &mut RandomOracle<E::ScalarField>,
    alpha: E::ScalarField,
    beta: E::ScalarField,
    claim_s: E::ScalarField,
    log_n: usize,
    table_mle_at: impl Fn(&[E::ScalarField]) -> E::ScalarField,
    tab_inv_commit: MkzgCommit<E>,
    count_commit: MkzgCommit<E>,
    acc: &mut VerifierAcc<E>,
) {
    let len_r = proof.next_u();
    let nv_r = log2_ceil(len_r);
    assert_eq!(nv_r, log_n);
    let r_eq = ro.next_n_fields(nv_r);
    let r_sum = ro.next_field();
    let (point_r, y_r) = verifier_sumcheck::<E>(claim_s * r_sum, nv_r, 3, proof, ro);
    let tab_v = proof.next_f();
    let tab_inv_v = proof.next_f();
    let count_v = proof.next_f();
    let eq_pref = MlPoly::eval_eq_pref(&r_eq, &point_r, len_r);
    assert_eq!(
        y_r,
        (tab_v * tab_inv_v - E::ScalarField::one()) * eq_pref + r_sum * tab_inv_v * count_v
    );
    let expected_tab_v =
        alpha + identity_mle::<E::ScalarField>(&point_r) + beta * table_mle_at(&point_r);
    assert_eq!(tab_v, expected_tab_v);

    acc.push(tab_inv_commit, point_r.clone(), tab_inv_v);
    acc.push(count_commit, point_r, count_v);
}

pub fn sparse_verify<E: Pairing>(
    kzg_vp: &MkzgVerParams<E>,
    commits: &SparseCommits<E>,
    point_abc: &[E::ScalarField],
    point_de: &[E::ScalarField],
    point_suf: &[E::ScalarField],
    point_pre: &[E::ScalarField],
    gamma: E::ScalarField,
    proof: &mut Proof<E>,
    ro: &mut RandomOracle<E::ScalarField>,
) -> SparseEvals<E::ScalarField> {
    let arr = proof.next_n_fs(NUM_CLAIMS);
    let mut claims_arr = [E::ScalarField::zero(); NUM_CLAIMS];
    claims_arr.copy_from_slice(&arr);
    let evals = SparseEvals::from_array(claims_arr);
    let claim_array = evals.as_array();
    assert_eq!(evals.a_pre, E::ScalarField::zero());
    assert_eq!(evals.c_pre, E::ScalarField::zero());
    assert_eq!(evals.d_pre, E::ScalarField::zero());
    assert_eq!(evals.e_pre, E::ScalarField::zero());

    let mut acc = VerifierAcc::<E>::new();
    let sg_commit = &commits.supergroups[0];
    let claims: Vec<E::ScalarField> = SUF_CLAIMS.iter().map(|&i| claim_array[i]).collect();
    lasso_verify_supergroup::<E>(
        sg_commit, point_abc, point_de, point_suf, gamma, &claims, proof, ro, &mut acc,
    );
    verify_b_pre::<E>(
        &commits.b_pre,
        point_abc,
        point_pre,
        gamma,
        evals.b_pre,
        proof,
        ro,
        &mut acc,
    );

    let kzg_len = proof.next_u();
    let kzg_proof = MkzgProof(proof.next_n_gs(kzg_len));
    let sumcheck_len = proof.next_u();
    let sumcheck_proof = SumcheckProof(proof.next_n_fs(sumcheck_len));
    assert!(Mkzg::<E>::batch_verify(
        kzg_vp,
        &acc.points,
        &acc.commits,
        &acc.values,
        (kzg_proof, sumcheck_proof),
        ro,
    ));

    evals
}
