//! The vectors a registration prover commits to, and the public size profile
//! the verifier needs to state the relation.
//!
//! Besides the program itself (selectors, operands and the four counters) the
//! prover commits a few *derived* vectors. They exist so that every leaf of a
//! fractional-sum check stays affine in committed vectors, which is what lets
//! the verifier rebuild the leaf claims without an extra sumcheck:
//!
//! * `row_out` / `lu_out`: the read bound of each constraint and lookup row,
//!   i.e. the output cell of the instruction that owns it;
//! * `max_read`: the same bound, but fetched per descriptor entry;
//! * `diff`: the slack `max_read - col - 1` a range lookup checks.

use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;

use util::{
    kzg::{Mkzg, MkzgCommit, MkzgProveParams},
    poly::MlPoly,
};

use crate::{
    circuit::Circuit,
    registration::family::{families, PowTerm, MAX_DIGITS},
    preprocess::VerifierKey,
    registration::{
        args::{pad_len, pad_to},
        encode::{Opcode, ProgramEncoding, NUM_OPCODES, NUM_OPERANDS},
    },
    sparse::log2_ceil,
};

/// Counter slots, in the order they are committed.
pub(crate) const CTR_OUT: usize = 0;
pub(crate) const CTR_AUX: usize = 1;
pub(crate) const CTR_CR: usize = 2;
pub(crate) const CTR_LR: usize = 3;
pub(crate) const NUM_COUNTERS: usize = 4;

/// The public size profile of a registered model: everything the verifier has
/// to know about the shape of the circuit and the program. The paper allows
/// this much to leak, so none of it is hidden.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegistrationProfile {
    pub weight_len: usize,
    pub input_len: usize,
    pub aux_start: usize,
    pub z_len: usize,
    pub cons_line: usize,
    pub lu_line: usize,
    pub instr_count: usize,
    /// Entry counts of A, B, C, D, E inside the sparse supergroup.
    pub matrix_lens: Vec<usize>,
    pub entry_len: usize,
    pub b_pre_len: usize,
    /// Row-space offset the supergroup adds to D and E rows.
    pub row_flag: usize,
    /// Range-check width: every read slack has to land in `[0, 2^log_range)`.
    pub log_range: usize,
    /// Entry count of each expansion family, in [`family::families`] order.
    pub family_lens: Vec<usize>,
    /// Total length of the expansion table.
    pub x_len: usize,
}

impl RegistrationProfile {
    pub fn new<E: Pairing>(vk: &VerifierKey<E>, enc: &ProgramEncoding) -> Self {
        let sg = &vk.sparse_commits.supergroups[0];
        let fams = families();
        let mut family_lens = vec![0usize; fams.len()];
        for i in 0..enc.len() {
            let ops = enc.operands_i64(i);
            for (f, family) in fams.iter().enumerate() {
                if family.opcode == enc.opcode[i] {
                    family_lens[f] += family.size(&ops);
                }
            }
        }
        let x_len = family_lens.iter().sum();
        RegistrationProfile {
            weight_len: enc.weight_len,
            input_len: enc.input_len,
            aux_start: enc.aux_start,
            z_len: enc.z_len,
            cons_line: enc.cons_line,
            lu_line: enc.lu_line,
            instr_count: enc.len(),
            matrix_lens: sg.matrix_lens.clone(),
            entry_len: sg.len,
            b_pre_len: vk.sparse_commits.b_pre.len,
            row_flag: 1usize << (sg.log_row - 1),
            log_range: log2_ceil(enc.z_len + 2),
            family_lens,
            x_len,
        }
    }

    /// `[lo, hi)` range of family `f` inside the expansion table.
    pub(crate) fn family_range(&self, f: usize) -> (usize, usize) {
        let lo: usize = self.family_lens[..f].iter().sum();
        (lo, lo + self.family_lens[f])
    }

    pub(crate) fn log_x(&self) -> usize {
        log2_ceil(self.x_len)
    }

    /// Widest domain the certificate touches. Commitments and openings are
    /// done against an SRS trimmed to this size: the multilinear KZG here pads
    /// every polynomial to the chunk size, so using the full proving SRS for
    /// the short vectors a registration commits would cost orders of magnitude
    /// more than the vectors themselves.
    pub(crate) fn log_srs(&self) -> usize {
        self.log_instr()
            .max(self.log_cons())
            .max(self.log_lu())
            .max(self.log_entry())
            .max(self.log_b_pre())
            .max(self.log_x())
            .max(self.log_range)
    }

    /// `[lo, hi)` entry range of matrix `m` inside the supergroup.
    pub(crate) fn segment(&self, m: usize) -> (usize, usize) {
        let lo: usize = self.matrix_lens[..m].iter().sum();
        (lo, lo + self.matrix_lens[m])
    }

    pub(crate) fn log_instr(&self) -> usize {
        log2_ceil(self.instr_count)
    }

    pub(crate) fn log_cons(&self) -> usize {
        log2_ceil(self.cons_line)
    }

    pub(crate) fn log_lu(&self) -> usize {
        log2_ceil(self.lu_line)
    }

    pub(crate) fn log_entry(&self) -> usize {
        log2_ceil(self.entry_len)
    }

    pub(crate) fn log_b_pre(&self) -> usize {
        log2_ceil(self.b_pre_len)
    }
}

/// Committed vectors of a registration, prover side.
#[derive(Clone)]
pub struct ProgramWitness<F: PrimeField> {
    // Program domain: one entry per instruction.
    pub(crate) sel: Vec<MlPoly<F>>,
    pub(crate) opn: Vec<MlPoly<F>>,
    pub(crate) ctr: Vec<MlPoly<F>>,
    pub(crate) dctr: Vec<MlPoly<F>>,
    /// Start of each instruction's run inside its expansion family.
    pub(crate) base: Vec<MlPoly<F>>,
    /// Number of entries each instruction contributes to each family.
    pub(crate) mult: Vec<MlPoly<F>>,
    // Constraint-row domain.
    pub(crate) row_out: MlPoly<F>,
    pub(crate) row_ptr: MlPoly<F>,
    pub(crate) row_cr: MlPoly<F>,
    pub(crate) row_dcr: MlPoly<F>,
    pub(crate) is_lin: MlPoly<F>,
    pub(crate) row_slack_lo: MlPoly<F>,
    pub(crate) row_slack_hi: MlPoly<F>,
    pub(crate) cnt_c: MlPoly<F>,
    pub(crate) b_cnt: MlPoly<F>,
    // Lookup-row domain.
    pub(crate) lu_out: MlPoly<F>,
    pub(crate) is_lookup: MlPoly<F>,
    pub(crate) cnt_l: MlPoly<F>,
    // Descriptor entry domain.
    pub(crate) max_read: MlPoly<F>,
    pub(crate) diff: MlPoly<F>,
    pub(crate) ent_lin: MlPoly<F>,
    pub(crate) mask_a: MlPoly<F>,
    pub(crate) mask_b: MlPoly<F>,
    pub(crate) mask_c: MlPoly<F>,
    // Prefix descriptor domain.
    pub(crate) diff_pre: MlPoly<F>,
    pub(crate) b_ent_lin: MlPoly<F>,
    pub(crate) mask_pre: MlPoly<F>,
    // Expansion table.
    pub(crate) x_ptr: MlPoly<F>,
    pub(crate) x_base: MlPoly<F>,
    pub(crate) x_cr: MlPoly<F>,
    pub(crate) x_out: MlPoly<F>,
    pub(crate) x_aux: MlPoly<F>,
    pub(crate) x_op: Vec<MlPoly<F>>,
    pub(crate) x_digit: Vec<MlPoly<F>>,
    pub(crate) x_digit_slack: Vec<MlPoly<F>>,
    pub(crate) x_pow: MlPoly<F>,
    // Range table multiplicities.
    pub(crate) range_cnt: MlPoly<F>,
}

/// The same vectors as commitments; this is what a registered model publishes
/// alongside the circuit commitment.
#[derive(Clone)]
pub struct ProgramCommits<E: Pairing> {
    pub sel: Vec<MkzgCommit<E>>,
    pub opn: Vec<MkzgCommit<E>>,
    pub ctr: Vec<MkzgCommit<E>>,
    pub dctr: Vec<MkzgCommit<E>>,
    pub base: Vec<MkzgCommit<E>>,
    pub mult: Vec<MkzgCommit<E>>,
    pub row_out: MkzgCommit<E>,
    pub row_ptr: MkzgCommit<E>,
    pub row_cr: MkzgCommit<E>,
    pub row_dcr: MkzgCommit<E>,
    pub is_lin: MkzgCommit<E>,
    pub row_slack_lo: MkzgCommit<E>,
    pub row_slack_hi: MkzgCommit<E>,
    pub cnt_c: MkzgCommit<E>,
    pub b_cnt: MkzgCommit<E>,
    pub lu_out: MkzgCommit<E>,
    pub is_lookup: MkzgCommit<E>,
    pub cnt_l: MkzgCommit<E>,
    pub max_read: MkzgCommit<E>,
    pub diff: MkzgCommit<E>,
    pub ent_lin: MkzgCommit<E>,
    pub mask_a: MkzgCommit<E>,
    pub mask_b: MkzgCommit<E>,
    pub mask_c: MkzgCommit<E>,
    pub diff_pre: MkzgCommit<E>,
    pub b_ent_lin: MkzgCommit<E>,
    pub mask_pre: MkzgCommit<E>,
    pub x_ptr: MkzgCommit<E>,
    pub x_base: MkzgCommit<E>,
    pub x_cr: MkzgCommit<E>,
    pub x_out: MkzgCommit<E>,
    pub x_aux: MkzgCommit<E>,
    pub x_op: Vec<MkzgCommit<E>>,
    pub x_digit: Vec<MkzgCommit<E>>,
    pub x_digit_slack: Vec<MkzgCommit<E>>,
    pub x_pow: MkzgCommit<E>,
    pub range_cnt: MkzgCommit<E>,
}

fn from_i64<F: PrimeField>(values: &[i64], len: usize) -> MlPoly<F> {
    let mut out: Vec<F> = values.iter().map(|&v| signed::<F>(v)).collect();
    out.resize(len, F::ZERO);
    MlPoly::new(out)
}

pub(crate) fn signed<F: PrimeField>(value: i64) -> F {
    if value < 0 {
        -F::from(value.unsigned_abs())
    } else {
        F::from(value as u64)
    }
}

fn from_usize<F: PrimeField>(values: &[usize], len: usize, filler: F) -> MlPoly<F> {
    let out: Vec<F> = values.iter().map(|&v| F::from(v as u64)).collect();
    MlPoly::new(pad_to(out, len, filler))
}

impl<F: PrimeField> ProgramWitness<F> {
    /// Builds every committed vector from the program encoding and the circuit
    /// its compilation produced.
    pub fn build(
        enc: &ProgramEncoding,
        circuit: &Circuit<F>,
        profile: &RegistrationProfile,
    ) -> Self {
        let q_len = pad_len(profile.instr_count);
        let cons_len = pad_len(profile.cons_line);
        let lu_len = pad_len(profile.lu_line);
        let entry_len = pad_len(profile.entry_len);
        let b_pre_len = pad_len(profile.b_pre_len);
        let x_len = pad_len(profile.x_len);
        let range_len = 1usize << profile.log_range;
        let q = profile.instr_count;
        let fams = families();

        let sel = (0..NUM_OPCODES)
            .map(|j| from_i64::<F>(&enc.sel[j], q_len))
            .collect::<Vec<_>>();
        let opn = (0..NUM_OPERANDS)
            .map(|slot| from_i64::<F>(&enc.opn[slot], q_len))
            .collect::<Vec<_>>();

        // Counters are padded with their final value, so the recurrence
        // `c[i + 1] = c[i] + delta[i]` also holds across the padding, where
        // every delta is zero.
        let counters = [&enc.out, &enc.aux, &enc.cr, &enc.lr];
        let ends = [
            profile.aux_start,
            profile.z_len,
            profile.cons_line,
            profile.lu_line,
        ];
        let ctr = (0..NUM_COUNTERS)
            .map(|c| from_usize::<F>(counters[c], q_len, F::from(ends[c] as u64)))
            .collect::<Vec<_>>();

        let mut dctr = vec![vec![F::ZERO; q_len]; NUM_COUNTERS];
        for i in 0..q {
            let d = ProgramEncoding::deltas(enc.opcode[i], &enc.operands_i64(i));
            dctr[CTR_OUT][i] = F::from(d.out as u64);
            dctr[CTR_AUX][i] = F::from(d.aux as u64);
            dctr[CTR_CR][i] = F::from(d.cr as u64);
            dctr[CTR_LR][i] = F::from(d.lr as u64);
        }
        let dctr = dctr.into_iter().map(MlPoly::new).collect::<Vec<_>>();

        // Expansion layout: each family owns a public range of the table, and
        // inside it every instruction owns a contiguous run.
        let mut mult_raw = vec![vec![0usize; q_len]; fams.len()];
        let mut base_raw = vec![vec![0usize; q_len]; fams.len()];
        for (f, family) in fams.iter().enumerate() {
            let mut running = 0usize;
            for i in 0..q_len {
                base_raw[f][i] = running;
                if i < q && family.opcode == enc.opcode[i] {
                    let size = family.size(&enc.operands_i64(i));
                    mult_raw[f][i] = size;
                    running += size;
                }
            }
        }
        let base = (0..fams.len())
            .map(|f| from_usize::<F>(&base_raw[f], q_len, F::ZERO))
            .collect::<Vec<_>>();
        let mult = (0..fams.len())
            .map(|f| from_usize::<F>(&mult_raw[f], q_len, F::ZERO))
            .collect::<Vec<_>>();

        // Row-level data: which instruction owns each constraint row, and how
        // far into that instruction's block the row sits.
        let mut row_out = vec![F::ZERO; cons_len];
        let mut row_ptr = vec![F::ZERO; cons_len];
        let mut row_cr = vec![F::ZERO; cons_len];
        let mut row_dcr = vec![F::ZERO; cons_len];
        let mut is_lin = vec![F::ZERO; cons_len];
        let mut row_slack_lo = vec![F::ZERO; cons_len];
        let mut row_slack_hi = vec![F::ZERO; cons_len];
        let mut lu_out = vec![F::ZERO; lu_len];
        let mut is_lookup = vec![F::ZERO; lu_len];
        for i in 0..q {
            let out = F::from(enc.out[i] as u64);
            let d = ProgramEncoding::deltas(enc.opcode[i], &enc.operands_i64(i));
            let lin = matches!(enc.opcode[i], Opcode::MatMult | Opcode::Conv);
            for r in 0..d.cr {
                let row = enc.cr[i] + r;
                row_out[row] = out;
                row_ptr[row] = F::from(i as u64);
                row_cr[row] = F::from(enc.cr[i] as u64);
                row_dcr[row] = F::from(d.cr as u64);
                is_lin[row] = if lin { F::ONE } else { F::ZERO };
                row_slack_lo[row] = F::from(r as u64);
                row_slack_hi[row] = F::from((d.cr - 1 - r) as u64);
            }
            for l in 0..d.lr {
                lu_out[enc.lr[i] + l] = out;
                // Only a `Lookup` row may carry entries on the D side; the
                // rows a `Div` emits are range checks with an empty input.
                is_lookup[enc.lr[i] + l] = if enc.opcode[i] == Opcode::Lookup {
                    F::ONE
                } else {
                    F::ZERO
                };
            }
        }

        // Per-entry data: the bound that applies, whether the row belongs to a
        // linear-algebra instruction, and the slack the range check sees.
        let (a_lo, a_hi) = profile.segment(0);
        let (b_lo, b_hi) = profile.segment(1);
        let (_c_lo, c_hi) = profile.segment(2);
        let (d_lo, d_hi) = profile.segment(3);
        let weight_len = F::from(profile.weight_len as u64);
        let mut cnt_c = vec![F::ZERO; cons_len];
        let mut cnt_l = vec![F::ZERO; lu_len];
        let mut b_cnt = vec![F::ZERO; cons_len];
        let mut max_read = vec![F::ZERO; entry_len];
        let mut diff = vec![F::ZERO; entry_len];
        let mut ent_lin = vec![F::ZERO; entry_len];
        let mut mask_a = vec![F::ZERO; entry_len];
        let mut mask_b = vec![F::ZERO; entry_len];
        let mut mask_c = vec![F::ZERO; entry_len];
        let view = SupergroupView::new(circuit, profile);
        for t in 0..profile.entry_len {
            let row = view.row_idx[t];
            let col = view.col[t];
            if (a_lo..c_hi).contains(&t) {
                cnt_c[row] += F::ONE;
                max_read[t] = row_out[row];
                ent_lin[t] = is_lin[row];
                let mask = if (a_lo..a_hi).contains(&t) {
                    &mut mask_a
                } else if (b_lo..b_hi).contains(&t) {
                    &mut mask_b
                } else {
                    &mut mask_c
                };
                mask[t] = is_lin[row];
                // Rows a MatMult or Conv owns are pinned entry by entry and
                // legitimately read advice cells, so they carry no slack.
                if is_lin[row].is_zero() && t < b_hi {
                    diff[t] = max_read[t] - weight_len - col - F::ONE;
                }
            } else if (d_lo..d_hi).contains(&t) {
                let l = row - profile.row_flag;
                cnt_l[l] += F::ONE;
                max_read[t] = lu_out[l];
                diff[t] = max_read[t] - weight_len - col - F::ONE;
            }
        }

        // Columns in the prefix group address the model parameters, which
        // every instruction may read; all they need is to stay in range.
        let mut diff_pre = vec![F::ZERO; b_pre_len];
        let mut b_ent_lin = vec![F::ZERO; b_pre_len];
        let mut mask_pre = vec![F::ZERO; b_pre_len];
        for t in 0..profile.b_pre_len {
            let row = view.b_row_idx[t];
            diff_pre[t] = weight_len - F::ONE - view.b_col[t];
            b_cnt[row] += F::ONE;
            b_ent_lin[t] = is_lin[row];
            mask_pre[t] = is_lin[row];
        }

        // The expansion table itself.
        let mut x_ptr = vec![F::ZERO; x_len];
        let mut x_base = vec![F::ZERO; x_len];
        let mut x_cr = vec![F::ZERO; x_len];
        let mut x_out = vec![F::ZERO; x_len];
        let mut x_aux = vec![F::ZERO; x_len];
        let mut x_op = vec![vec![F::ZERO; x_len]; NUM_OPERANDS];
        let mut x_digit = vec![vec![F::ZERO; x_len]; MAX_DIGITS];
        let mut x_digit_slack = vec![vec![F::ZERO; x_len]; MAX_DIGITS];
        let mut x_pow = vec![F::ZERO; x_len];
        for (f, family) in fams.iter().enumerate() {
            let (lo, _) = profile.family_range(f);
            for i in 0..q {
                if family.opcode != enc.opcode[i] {
                    continue;
                }
                let ops = enc.operands_i64(i);
                let radices: Vec<i64> = family.radices.iter().map(|r| r.eval_i64(&ops)).collect();
                let run = lo + base_raw[f][i];
                for flat in 0..mult_raw[f][i] {
                    let t = run + flat;
                    x_ptr[t] = F::from(i as u64);
                    x_base[t] = F::from(base_raw[f][i] as u64);
                    x_cr[t] = F::from(enc.cr[i] as u64);
                    x_out[t] = F::from(enc.out[i] as u64);
                    x_aux[t] = F::from(enc.aux[i] as u64);
                    for slot in 0..NUM_OPERANDS {
                        x_op[slot][t] = signed::<F>(ops[slot]);
                    }

                    // Mixed-radix digits of the position inside the run.
                    let mut digits = vec![0i64; radices.len()];
                    let mut rest = flat as i64;
                    for j in (0..radices.len()).rev() {
                        digits[j] = rest % radices[j];
                        rest /= radices[j];
                    }
                    for j in 0..radices.len() {
                        x_digit[j][t] = signed::<F>(digits[j]);
                        x_digit_slack[j][t] = signed::<F>(radices[j] - 1 - digits[j]);
                    }

                    let mut pow = family.pow_const.eval_i64(&ops);
                    for (coeff, term) in &family.pow {
                        let value = match term {
                            PowTerm::Digit(j) => digits[*j],
                            PowTerm::Flat => flat as i64,
                        };
                        pow += coeff.eval_i64(&ops) * value;
                    }
                    x_pow[t] = signed::<F>(pow);
                }
            }
        }

        let mut range_cnt = vec![F::ZERO; range_len];
        let mut tally = |value: F| {
            let index = field_to_index(value)
                .unwrap_or_else(|| panic!("slack {} escapes the range table", value));
            assert!(index < range_len, "slack {} escapes the range table", index);
            range_cnt[index] += F::ONE;
        };
        for value in diff
            .iter()
            .chain(diff_pre.iter())
            .chain(row_slack_lo.iter())
            .chain(row_slack_hi.iter())
            .chain(x_digit.iter().flatten())
            .chain(x_digit_slack.iter().flatten())
        {
            tally(*value);
        }

        ProgramWitness {
            sel,
            opn,
            ctr,
            dctr,
            base,
            mult,
            row_out: MlPoly::new(row_out),
            row_ptr: MlPoly::new(row_ptr),
            row_cr: MlPoly::new(row_cr),
            row_dcr: MlPoly::new(row_dcr),
            is_lin: MlPoly::new(is_lin),
            row_slack_lo: MlPoly::new(row_slack_lo),
            row_slack_hi: MlPoly::new(row_slack_hi),
            cnt_c: MlPoly::new(cnt_c),
            b_cnt: MlPoly::new(b_cnt),
            lu_out: MlPoly::new(lu_out),
            is_lookup: MlPoly::new(is_lookup),
            cnt_l: MlPoly::new(cnt_l),
            max_read: MlPoly::new(max_read),
            diff: MlPoly::new(diff),
            ent_lin: MlPoly::new(ent_lin),
            mask_a: MlPoly::new(mask_a),
            mask_b: MlPoly::new(mask_b),
            mask_c: MlPoly::new(mask_c),
            diff_pre: MlPoly::new(diff_pre),
            b_ent_lin: MlPoly::new(b_ent_lin),
            mask_pre: MlPoly::new(mask_pre),
            x_ptr: MlPoly::new(x_ptr),
            x_base: MlPoly::new(x_base),
            x_cr: MlPoly::new(x_cr),
            x_out: MlPoly::new(x_out),
            x_aux: MlPoly::new(x_aux),
            x_op: x_op.into_iter().map(MlPoly::new).collect(),
            x_digit: x_digit.into_iter().map(MlPoly::new).collect(),
            x_digit_slack: x_digit_slack.into_iter().map(MlPoly::new).collect(),
            x_pow: MlPoly::new(x_pow),
            range_cnt: MlPoly::new(range_cnt),
        }
    }

    pub fn commit<E: Pairing<ScalarField = F>>(
        &self,
        kzg_pp: &MkzgProveParams<E>,
    ) -> ProgramCommits<E> {
        let commit = |poly: &MlPoly<F>| Mkzg::<E>::commit(kzg_pp, poly);
        let commit_all =
            |polys: &Vec<MlPoly<F>>| polys.iter().map(|p| Mkzg::<E>::commit(kzg_pp, p)).collect();
        ProgramCommits {
            sel: commit_all(&self.sel),
            opn: commit_all(&self.opn),
            ctr: commit_all(&self.ctr),
            dctr: commit_all(&self.dctr),
            base: commit_all(&self.base),
            mult: commit_all(&self.mult),
            row_out: commit(&self.row_out),
            row_ptr: commit(&self.row_ptr),
            row_cr: commit(&self.row_cr),
            row_dcr: commit(&self.row_dcr),
            is_lin: commit(&self.is_lin),
            row_slack_lo: commit(&self.row_slack_lo),
            row_slack_hi: commit(&self.row_slack_hi),
            cnt_c: commit(&self.cnt_c),
            b_cnt: commit(&self.b_cnt),
            lu_out: commit(&self.lu_out),
            is_lookup: commit(&self.is_lookup),
            cnt_l: commit(&self.cnt_l),
            max_read: commit(&self.max_read),
            diff: commit(&self.diff),
            ent_lin: commit(&self.ent_lin),
            mask_a: commit(&self.mask_a),
            mask_b: commit(&self.mask_b),
            mask_c: commit(&self.mask_c),
            diff_pre: commit(&self.diff_pre),
            b_ent_lin: commit(&self.b_ent_lin),
            mask_pre: commit(&self.mask_pre),
            x_ptr: commit(&self.x_ptr),
            x_base: commit(&self.x_base),
            x_cr: commit(&self.x_cr),
            x_out: commit(&self.x_out),
            x_aux: commit(&self.x_aux),
            x_op: commit_all(&self.x_op),
            x_digit: commit_all(&self.x_digit),
            x_digit_slack: commit_all(&self.x_digit_slack),
            x_pow: commit(&self.x_pow),
            range_cnt: commit(&self.range_cnt),
        }
    }
}

/// Flattened view of the committed sparse descriptor, in the same entry order
/// [`crate::sparse::sparse_commit`] uses.
pub(crate) struct SupergroupView<F: PrimeField> {
    pub(crate) row_idx: Vec<usize>,
    pub(crate) col: Vec<F>,
    pub(crate) b_row_idx: Vec<usize>,
    pub(crate) b_col: Vec<F>,
}

impl<F: PrimeField> SupergroupView<F> {
    pub(crate) fn new(circuit: &Circuit<F>, profile: &RegistrationProfile) -> Self {
        let wl = profile.weight_len;
        let mats = [&circuit.a, &circuit.b, &circuit.c, &circuit.d, &circuit.e];
        let mut row_idx = vec![];
        let mut col = vec![];
        for (m, mat) in mats.iter().enumerate() {
            let flag = if m < 3 { 0 } else { profile.row_flag };
            for (r, row) in mat.rows.iter().enumerate() {
                for &(c, _, _) in &row.elems {
                    if c >= wl {
                        row_idx.push(flag + r);
                        col.push(F::from((c - wl) as u64));
                    }
                }
            }
        }
        let mut b_row_idx = vec![];
        let mut b_col = vec![];
        for (r, row) in circuit.b.rows.iter().enumerate() {
            for &(c, _, _) in &row.elems {
                if c < wl {
                    b_row_idx.push(r);
                    b_col.push(F::from(c as u64));
                }
            }
        }
        SupergroupView {
            row_idx,
            col,
            b_row_idx,
            b_col,
        }
    }
}

/// Reads a field element back as a small non-negative integer, or `None` when
/// it is not one.
pub(crate) fn field_to_index<F: PrimeField>(value: F) -> Option<usize> {
    let repr = value.into_bigint();
    let limbs = repr.as_ref();
    if limbs.iter().skip(1).any(|&l| l != 0) {
        return None;
    }
    usize::try_from(limbs[0]).ok()
}
