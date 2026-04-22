use ark_ec::pairing::Pairing;
use ark_ff::{Field, One, PrimeField, Zero};

use util::{
    kzg::{Mkzg, MkzgCommit, MkzgProof, MkzgProveParams, MkzgVerParams, SumcheckProof},
    poly::MlPoly,
    util::{batch_inverse, Proof, RandomOracle},
};

use crate::circuit::{Circuit, SparseMatrix};

// 10 groups: index 0..=4 = {a,b,c,d,e}_suf, 5..=9 = {a,b,c,d,e}_pre.
const NUM_GROUPS: usize = 10;

#[derive(Clone)]
pub struct SparseGroup<F: PrimeField> {
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
    pub groups: Vec<SparseGroup<F>>,
}

#[derive(Clone)]
pub struct SparseGroupCommit<E: Pairing> {
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
    pub groups: Vec<SparseGroupCommit<E>>,
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
    fn as_array(&self) -> [F; NUM_GROUPS] {
        [
            self.a_suf, self.b_suf, self.c_suf, self.d_suf, self.e_suf, self.a_pre, self.b_pre,
            self.c_pre, self.d_pre, self.e_pre,
        ]
    }

    fn from_array(arr: [F; NUM_GROUPS]) -> Self {
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

// MLE of [0, 1, 2, ..., 2^k - 1] at `point` (length k).
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

// MLE of [gamma^0, gamma^1, ..., gamma^{2^k - 1}] at `point` (length k).
fn gamma_mle<F: Field>(point: &[F], gamma: F) -> F {
    let mut acc = F::one();
    let mut pow_2j = gamma;
    for &x in point {
        acc *= F::one() + x * (pow_2j - F::one());
        pow_2j = pow_2j * pow_2j;
    }
    acc
}

// MLE of indicator [i < n] at `point` (length nv). Matches the implicit zero
// padding of MlPoly::eval when a length-n poly is evaluated at a point of
// length nv = log2_ceil(n).
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

// Collect entries (row, col_shifted, val, pow) with col in [col_lo, col_hi).
// col_shifted = col - col_lo.
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

fn build_group<F: PrimeField>(
    mut entries: Vec<(usize, usize, F, usize)>,
    log_row: usize,
    log_col: usize,
    log_pow: usize,
) -> SparseGroup<F> {
    if entries.is_empty() {
        // Avoid empty polys; contribute a single zero entry which contributes 0 to any MLE.
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
    SparseGroup {
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
    let log_col_suf = log2_ceil(z_len - wei_len);
    let log_col_pre = log2_ceil(wei_len);

    let mats = [
        &circuit.a,
        &circuit.b,
        &circuit.c,
        &circuit.d,
        &circuit.e,
    ];
    let mut max_pow = 0usize;
    for mat in mats.iter() {
        for row in &mat.rows {
            for &(_, _, pow) in &row.elems {
                if let Some(p) = pow {
                    if p > max_pow {
                        max_pow = p;
                    }
                }
            }
        }
    }
    let log_pow = log2_ceil(max_pow + 1);

    let mut groups = Vec::with_capacity(NUM_GROUPS);
    for (i, mat) in mats.iter().enumerate() {
        let log_row = if i < 3 { log_row_abc } else { log_row_de };
        let entries = collect_entries(mat, wei_len, usize::MAX);
        groups.push(build_group(entries, log_row, log_col_suf, log_pow));
    }
    for (i, mat) in mats.iter().enumerate() {
        let log_row = if i < 3 { log_row_abc } else { log_row_de };
        let entries = collect_entries(mat, 0, wei_len);
        groups.push(build_group(entries, log_row, log_col_pre, log_pow));
    }

    let mut commits = Vec::with_capacity(NUM_GROUPS);
    for g in &groups {
        commits.push(SparseGroupCommit {
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
        });
    }

    (SparsePolys { groups }, SparseCommits { groups: commits })
}

// ============================================================
// Low-level sumcheck helpers (prover side).
// ============================================================

// Degree-4 product sumcheck: proves sum_t a[t]*b[t]*c[t]*d[t].
fn sumcheck_main<E: Pairing>(
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
        let mut sums = [E::ScalarField::zero(); 5];
        for j in (0..m).step_by(2) {
            let da = a[j + 1] - a[j];
            let db = b[j + 1] - b[j];
            let dc = c[j + 1] - c[j];
            let dd = d[j + 1] - d[j];
            sums[0] += a[j] * b[j] * c[j] * d[j];
            sums[1] += a[j + 1] * b[j + 1] * c[j + 1] * d[j + 1];
            let ax = a[j + 1] + da;
            let bx = b[j + 1] + db;
            let cx = c[j + 1] + dc;
            let dx = d[j + 1] + dd;
            sums[2] += ax * bx * cx * dx;
            let ax = ax + da;
            let bx = bx + db;
            let cx = cx + dc;
            let dx = dx + dd;
            sums[3] += ax * bx * cx * dx;
            let ax = ax + da;
            let bx = bx + db;
            let cx = cx + dc;
            let dx = dx + dd;
            sums[4] += ax * bx * cx * dx;
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

// Logup left-side sumcheck (duplicated from prover.rs). Proves
//     sum_t 1/(alpha + ele[t]) = S,
// structured as (ele * ele_inv - 1) * eq + r * ele_inv summing to r * S.
fn logup_sumcheck_left<E: Pairing>(
    mut ele: Vec<E::ScalarField>,
    mut ele_inv: Vec<E::ScalarField>,
    alpha: E::ScalarField,
    proof: &mut Proof<E>,
    ro: &mut RandomOracle<E::ScalarField>,
) -> Vec<E::ScalarField> {
    let log_len = log2_ceil(ele.len());
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
        let mut sums = [E::ScalarField::zero(); 4];

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

// Logup right-side sumcheck. Proves
//     sum_i count[i] / (alpha + tab[i]) = S.
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
// Low-level sumcheck helpers (verifier side).
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
    let mut numerator = (0..n + 1).map(|y| x - F::from(y as u32)).collect::<Vec<_>>();
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
// Prover: per-group Lasso.
// ============================================================

// Prover accumulator for the final batch KZG open.
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

// Verifier accumulator for the final batch KZG verify.
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

// Prover half of the memory check that e[t] = table[idx[t]].
fn memory_check_prover<E: Pairing>(
    kzg_pp: &MkzgProveParams<E>,
    idx_vec: &[E::ScalarField],
    idx_poly: &MlPoly<E::ScalarField>,
    e_poly: &MlPoly<E::ScalarField>,
    e_vec: &[E::ScalarField],
    table: &[E::ScalarField],
    count_poly: &MlPoly<E::ScalarField>,
    proof: &mut Proof<E>,
    ro: &mut RandomOracle<E::ScalarField>,
    acc: &mut ProverAcc<E::ScalarField>,
) {
    let t = idx_vec.len();
    let n = table.len();

    let beta = ro.next_field();
    let alpha = ro.next_field();

    // ele[t] = idx[t] + beta * e[t].  Alpha will be added internally by logup_sumcheck_left.
    let ele: Vec<E::ScalarField> = (0..t).map(|k| idx_vec[k] + beta * e_vec[k]).collect();
    // ele_inv[t] = 1 / (alpha + ele[t]).
    let mut ele_inv: Vec<E::ScalarField> = ele.iter().map(|&x| x + alpha).collect();
    batch_inverse(&mut ele_inv);

    let tab: Vec<E::ScalarField> = (0..n)
        .map(|i| E::ScalarField::from(i as u64) + beta * table[i])
        .collect();
    let mut tab_inv: Vec<E::ScalarField> = tab.iter().map(|&x| x + alpha).collect();
    batch_inverse(&mut tab_inv);

    let ele_inv_poly = MlPoly::new(ele_inv.clone());
    let tab_inv_poly = MlPoly::new(tab_inv.clone());
    let ele_inv_commit = Mkzg::<E>::commit(kzg_pp, &ele_inv_poly);
    let tab_inv_commit = Mkzg::<E>::commit(kzg_pp, &tab_inv_poly);
    proof.push_u(ele_inv_commit.0.len());
    proof.push_gs(&ele_inv_commit.0);
    proof.push_u(tab_inv_commit.0.len());
    proof.push_gs(&tab_inv_commit.0);

    let sum_s: E::ScalarField = ele_inv.iter().sum();
    proof.push_f(&[sum_s]);

    let point_l = logup_sumcheck_left::<E>(ele, ele_inv, alpha, proof, ro);
    let point_r =
        logup_sumcheck_right::<E>(tab, tab_inv, count_poly.0.clone(), alpha, proof, ro);

    // Openings: idx and e at point_l (for the verifier to reconstruct ele_v).
    let idx_v = idx_poly.clone().eval(&point_l);
    let e_v = e_poly.clone().eval(&point_l);
    proof.push_f(&[idx_v, e_v]);

    acc.push(idx_poly.clone(), point_l.clone());
    acc.push(e_poly.clone(), point_l.clone());
    acc.push(ele_inv_poly, point_l);
    acc.push(tab_inv_poly, point_r.clone());
    acc.push(count_poly.clone(), point_r);
}

fn lasso_prove_group<E: Pairing>(
    kzg_pp: &MkzgProveParams<E>,
    group: &SparseGroup<E::ScalarField>,
    row_eq_point: &[E::ScalarField],
    col_eq_point: &[E::ScalarField],
    gamma: E::ScalarField,
    proof: &mut Proof<E>,
    ro: &mut RandomOracle<E::ScalarField>,
    acc: &mut ProverAcc<E::ScalarField>,
) {
    assert_eq!(row_eq_point.len(), group.log_row);
    assert_eq!(col_eq_point.len(), group.log_col);

    let row_eq = MlPoly::new_eq(&row_eq_point.to_vec()).0;
    let col_eq = MlPoly::new_eq(&col_eq_point.to_vec()).0;

    let mut gamma_table = Vec::with_capacity(1usize << group.log_pow);
    {
        let mut cur = E::ScalarField::one();
        for _ in 0..(1usize << group.log_pow) {
            gamma_table.push(cur);
            cur *= gamma;
        }
    }

    let e_row: Vec<E::ScalarField> = group.row_idx.iter().map(|&i| row_eq[i]).collect();
    let e_col: Vec<E::ScalarField> = group.col_idx.iter().map(|&i| col_eq[i]).collect();
    let e_gamma: Vec<E::ScalarField> = group.pow_idx.iter().map(|&i| gamma_table[i]).collect();

    let e_row_poly = MlPoly::new(e_row.clone());
    let e_col_poly = MlPoly::new(e_col.clone());
    let e_gamma_poly = MlPoly::new(e_gamma.clone());
    let e_row_commit = Mkzg::<E>::commit(kzg_pp, &e_row_poly);
    let e_col_commit = Mkzg::<E>::commit(kzg_pp, &e_col_poly);
    let e_gamma_commit = Mkzg::<E>::commit(kzg_pp, &e_gamma_poly);
    proof.push_u(e_row_commit.0.len());
    proof.push_gs(&e_row_commit.0);
    proof.push_u(e_col_commit.0.len());
    proof.push_gs(&e_col_commit.0);
    proof.push_u(e_gamma_commit.0.len());
    proof.push_gs(&e_gamma_commit.0);

    // Main degree-4 product sumcheck proving the claimed MLE value.
    let point_main = sumcheck_main::<E>(
        e_row.clone(),
        e_col.clone(),
        group.val.0.clone(),
        e_gamma.clone(),
        proof,
        ro,
    );

    acc.push(e_row_poly.clone(), point_main.clone());
    acc.push(e_col_poly.clone(), point_main.clone());
    acc.push(group.val.clone(), point_main.clone());
    acc.push(e_gamma_poly.clone(), point_main);

    // 3 memory checks: row, col, gamma.
    memory_check_prover::<E>(
        kzg_pp,
        &group.row.0,
        &group.row,
        &e_row_poly,
        &e_row,
        &row_eq,
        &group.count_row,
        proof,
        ro,
        acc,
    );
    memory_check_prover::<E>(
        kzg_pp,
        &group.col.0,
        &group.col,
        &e_col_poly,
        &e_col,
        &col_eq,
        &group.count_col,
        proof,
        ro,
        acc,
    );
    memory_check_prover::<E>(
        kzg_pp,
        &group.pow.0,
        &group.pow,
        &e_gamma_poly,
        &e_gamma,
        &gamma_table,
        &group.count_pow,
        proof,
        ro,
        acc,
    );
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
    // Compute the 10 claimed MLE values directly (cheap in total non-zeros).
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

    // Push the 10 claimed values first so the verifier has the claims before Lasso proof.
    proof.push_f(&sparse_evals.as_array());

    let mut acc = ProverAcc::<E::ScalarField>::new();

    // Run Lasso per group.
    for g_idx in 0..NUM_GROUPS {
        let group = &polys.groups[g_idx];
        let row_eq_point = if g_idx % 5 < 3 { point_abc } else { point_de };
        let col_eq_point = if g_idx < 5 { point_suf } else { point_pre };
        lasso_prove_group::<E>(
            kzg_pp,
            group,
            row_eq_point,
            col_eq_point,
            gamma,
            proof,
            ro,
            &mut acc,
        );
    }

    // Final batch open.
    let (kzg_proof, sumcheck_proof) =
        Mkzg::<E>::batch_open(kzg_pp, &acc.polys, &acc.points, ro);
    proof.push_u(kzg_proof.0.len());
    proof.push_gs(&kzg_proof.0);
    proof.push_u(sumcheck_proof.0.len());
    proof.push_f(&sumcheck_proof.0);

    sparse_evals
}

// ============================================================
// Verifier: per-group Lasso.
// ============================================================

// Returns the opened values (idx_v, e_v) at point_l and point_r.
// `table_mle_at` must compute the MLE of the table (row_eq/col_eq/gamma) at a given point.
fn memory_check_verifier<E: Pairing>(
    idx_commit: &MkzgCommit<E>,
    e_commit: &MkzgCommit<E>,
    count_commit: &MkzgCommit<E>,
    table_mle_at: impl Fn(&[E::ScalarField]) -> E::ScalarField,
    log_n: usize,
    proof: &mut Proof<E>,
    ro: &mut RandomOracle<E::ScalarField>,
    acc: &mut VerifierAcc<E>,
) {
    let beta = ro.next_field();
    let alpha = ro.next_field();

    // Read ele_inv and tab_inv commits.
    let ele_inv_commit_len = proof.next_u();
    let ele_inv_commit = MkzgCommit(proof.next_n_gs(ele_inv_commit_len));
    let tab_inv_commit_len = proof.next_u();
    let tab_inv_commit = MkzgCommit(proof.next_n_gs(tab_inv_commit_len));

    let sum_s = proof.next_f();

    // Logup-left sumcheck.
    let len_l = proof.next_u();
    let nv_l = log2_ceil(len_l);
    let r_eq_l = ro.next_n_fields(nv_l);
    let r_sum_l = ro.next_field();
    let (point_l, y_l) = verifier_sumcheck::<E>(sum_s * r_sum_l, nv_l, 3, proof, ro);
    let ele_v = proof.next_f();
    let ele_inv_v = proof.next_f();
    assert_eq!(
        y_l,
        (ele_v * ele_inv_v - E::ScalarField::one())
            * MlPoly::eval_eq_pref(&r_eq_l, &point_l, len_l)
            + r_sum_l * ele_inv_v
    );

    // Logup-right sumcheck.
    let len_r = proof.next_u();
    let nv_r = log2_ceil(len_r);
    assert_eq!(nv_r, log_n);
    let r_eq_r = ro.next_n_fields(nv_r);
    let r_sum_r = ro.next_field();
    let (point_r, y_r) = verifier_sumcheck::<E>(sum_s * r_sum_r, nv_r, 3, proof, ro);
    let tab_v = proof.next_f();
    let tab_inv_v = proof.next_f();
    let count_v = proof.next_f();
    assert_eq!(
        y_r,
        (tab_v * tab_inv_v - E::ScalarField::one())
            * MlPoly::eval_eq_pref(&r_eq_r, &point_r, len_r)
            + r_sum_r * tab_inv_v * count_v
    );

    // Expected tab_v: alpha + identity_mle(point_r) + beta * table_mle(point_r).
    let expected_tab_v =
        alpha + identity_mle::<E::ScalarField>(&point_r) + beta * table_mle_at(&point_r);
    assert_eq!(tab_v, expected_tab_v);

    // Opened values at point_l: idx_v and e_v.
    let idx_v = proof.next_f();
    let e_v = proof.next_f();
    // Alpha was added as a constant to the first len_l positions of ele (the
    // rest is zero padding). Hence MLE(ele) = alpha * prefix_mle(len_l) + ...
    assert_eq!(
        ele_v,
        alpha * prefix_mle::<E::ScalarField>(&point_l, len_l) + idx_v + beta * e_v
    );

    // Accumulate opens for the batch verifier.
    acc.push(idx_commit.clone(), point_l.clone(), idx_v);
    acc.push(e_commit.clone(), point_l.clone(), e_v);
    acc.push(ele_inv_commit, point_l, ele_inv_v);
    acc.push(tab_inv_commit, point_r.clone(), tab_inv_v);
    acc.push(count_commit.clone(), point_r, count_v);
}

fn lasso_verify_group<E: Pairing>(
    group_commit: &SparseGroupCommit<E>,
    row_eq_point: &[E::ScalarField],
    col_eq_point: &[E::ScalarField],
    gamma: E::ScalarField,
    claim: E::ScalarField,
    proof: &mut Proof<E>,
    ro: &mut RandomOracle<E::ScalarField>,
    acc: &mut VerifierAcc<E>,
) {
    assert_eq!(row_eq_point.len(), group_commit.log_row);
    assert_eq!(col_eq_point.len(), group_commit.log_col);

    // Read e_row, e_col, e_gamma commits.
    let e_row_len = proof.next_u();
    let e_row_commit = MkzgCommit(proof.next_n_gs(e_row_len));
    let e_col_len = proof.next_u();
    let e_col_commit = MkzgCommit(proof.next_n_gs(e_col_len));
    let e_gamma_len = proof.next_u();
    let e_gamma_commit = MkzgCommit(proof.next_n_gs(e_gamma_len));

    // Main sumcheck: degree 4, log_T rounds, proving claim.
    let len = proof.next_u();
    let nv = log2_ceil(len);
    let (point_main, y_main) = verifier_sumcheck::<E>(claim, nv, 4, proof, ro);
    let e_row_v = proof.next_f();
    let e_col_v = proof.next_f();
    let val_v = proof.next_f();
    let e_gamma_v = proof.next_f();
    assert_eq!(y_main, e_row_v * e_col_v * val_v * e_gamma_v);

    acc.push(e_row_commit.clone(), point_main.clone(), e_row_v);
    acc.push(e_col_commit.clone(), point_main.clone(), e_col_v);
    acc.push(group_commit.val.clone(), point_main.clone(), val_v);
    acc.push(e_gamma_commit.clone(), point_main, e_gamma_v);

    // Row memory check: table MLE is eq(row_eq_point, .).
    let row_eq_point_vec = row_eq_point.to_vec();
    memory_check_verifier::<E>(
        &group_commit.row,
        &e_row_commit,
        &group_commit.count_row,
        |pt| MlPoly::<E::ScalarField>::eval_eq(&row_eq_point_vec, &pt.to_vec()),
        group_commit.log_row,
        proof,
        ro,
        acc,
    );

    // Col memory check: table MLE is eq(col_eq_point, .).
    let col_eq_point_vec = col_eq_point.to_vec();
    memory_check_verifier::<E>(
        &group_commit.col,
        &e_col_commit,
        &group_commit.count_col,
        |pt| MlPoly::<E::ScalarField>::eval_eq(&col_eq_point_vec, &pt.to_vec()),
        group_commit.log_col,
        proof,
        ro,
        acc,
    );

    // Gamma memory check: table MLE is gamma_mle(., gamma).
    memory_check_verifier::<E>(
        &group_commit.pow,
        &e_gamma_commit,
        &group_commit.count_pow,
        |pt| gamma_mle::<E::ScalarField>(pt, gamma),
        group_commit.log_pow,
        proof,
        ro,
        acc,
    );
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
    // Read 10 claimed MLE values first.
    let arr = proof.next_n_fs(NUM_GROUPS);
    let mut claims_arr = [E::ScalarField::zero(); NUM_GROUPS];
    claims_arr.copy_from_slice(&arr);
    let evals = SparseEvals::from_array(claims_arr);
    let claims = evals.as_array();

    let mut acc = VerifierAcc::<E>::new();
    for g_idx in 0..NUM_GROUPS {
        let group_commit = &commits.groups[g_idx];
        let row_eq_point = if g_idx % 5 < 3 { point_abc } else { point_de };
        let col_eq_point = if g_idx < 5 { point_suf } else { point_pre };
        lasso_verify_group::<E>(
            group_commit,
            row_eq_point,
            col_eq_point,
            gamma,
            claims[g_idx],
            proof,
            ro,
            &mut acc,
        );
    }

    // Batch KZG verify.
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
