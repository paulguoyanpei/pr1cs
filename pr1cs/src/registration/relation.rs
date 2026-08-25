//! The compilation relation, written as a list of checks over committed
//! vectors.
//!
//! The list is built from public data only, so the prover and the verifier
//! walk exactly the same rules in exactly the same order. Two shapes appear:
//!
//! * [`ZeroCheck`]: a low-degree expression vanishes on the hypercube;
//! * [`Equality`]: two multisets agree, written as fractional sums whose leaf
//!   numerators and fingerprint components are *affine* in the committed
//!   vectors. Affine is what lets the verifier rebuild the leaf claims from
//!   the component openings alone.
//!
//! Together they state the rules [`super::encode::check_compilation`] checks
//! in the clear.

use ark_ff::PrimeField;

use crate::{
    circuit::{LookupType, QUOTIENT_BOUND},
    registration::{
        encode::{Opcode, NUM_OPCODES, NUM_OPERANDS},
        family::{families, ColBase, Ex, PowTerm, RowSpec, MAX_DIGITS},
        witness::{signed, RegistrationProfile, CTR_AUX, CTR_CR, CTR_LR, CTR_OUT, NUM_COUNTERS},
    },
};

/// A vector the checks read from. Most are committed; the last three are
/// public, so the verifier evaluates them itself instead of opening them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Tab {
    Sel(usize),
    Opn(usize),
    Ctr(usize),
    DCtr(usize),
    /// Start of an instruction's run inside expansion family `f`.
    Base(usize),
    /// Entries an instruction contributes to expansion family `f`.
    Mult(usize),
    RowOut,
    RowPtr,
    RowCr,
    RowDcr,
    IsLin,
    RowSlackLo,
    RowSlackHi,
    BCnt,
    EntLin,
    MaskA,
    MaskB,
    MaskC,
    BEntLin,
    MaskPre,
    XPtr,
    XBase,
    XCr,
    XOut,
    XAux,
    XOp(usize),
    XDigit(usize),
    XDigitSlack(usize),
    XPow,
    CntC,
    LuOut,
    IsLookup,
    CntL,
    MaxRead,
    Diff,
    DiffPre,
    RangeCnt,
    /// Descriptor vectors, from the circuit commitment.
    Row,
    Col,
    Val,
    Pow,
    BRow,
    BCol,
    BVal,
    BPow,
    Tp,
    /// Public: the constant one.
    One,
    /// Public: the index itself.
    Identity,
    /// Public: the indicator of `[lo, hi)`.
    Ind(usize, usize),
}

impl Tab {
    pub(crate) fn is_committed(self) -> bool {
        !matches!(self, Tab::One | Tab::Identity | Tab::Ind(_, _))
    }
}

/// An affine expression `constant + sum coeff * table`.
#[derive(Clone, Debug)]
pub(crate) struct Lin {
    pub(crate) constant: i64,
    pub(crate) terms: Vec<(i64, Tab)>,
}

impl Lin {
    pub(crate) fn constant(value: i64) -> Self {
        Lin {
            constant: value,
            terms: vec![],
        }
    }

    pub(crate) fn of(tab: Tab) -> Self {
        Lin {
            constant: 0,
            terms: vec![(1, tab)],
        }
    }

    pub(crate) fn plus(mut self, coeff: i64, tab: Tab) -> Self {
        self.terms.push((coeff, tab));
        self
    }

    pub(crate) fn shift(mut self, value: i64) -> Self {
        self.constant += value;
        self
    }

    pub(crate) fn eval<F: PrimeField>(&self, values: &[(Tab, F)]) -> F {
        let mut acc = signed::<F>(self.constant);
        for &(coeff, tab) in &self.terms {
            let value = values
                .iter()
                .find(|(t, _)| *t == tab)
                .map(|(_, v)| *v)
                .unwrap_or_else(|| panic!("missing value for {:?}", tab));
            acc += signed::<F>(coeff) * value;
        }
        acc
    }
}

/// One side of a multiset equality: `sum_x num(x) / (alpha + sum beta^i c_i(x))`.
#[derive(Clone, Debug)]
pub(crate) struct Block {
    pub(crate) log_len: usize,
    pub(crate) num: Lin,
    pub(crate) comps: Vec<Lin>,
}

/// `sum of left blocks == sum of right blocks`, i.e. the two multisets agree.
pub(crate) struct Equality {
    pub(crate) name: &'static str,
    pub(crate) left: Vec<Block>,
    pub(crate) right: Vec<Block>,
}

/// `sum_x eq(r, x) * g(tables(x)) == 0`.
pub(crate) struct ZeroCheck<F: PrimeField> {
    pub(crate) name: &'static str,
    pub(crate) log_len: usize,
    pub(crate) degree: usize,
    pub(crate) tables: Vec<Tab>,
    #[allow(clippy::type_complexity)]
    pub(crate) g: Box<dyn Fn(&[F], &[F]) -> F>,
}

pub(crate) struct Rules<F: PrimeField> {
    pub(crate) zero_checks: Vec<ZeroCheck<F>>,
    pub(crate) equalities: Vec<Equality>,
}

/// Number of aggregation challenges the zero checks share.
pub(crate) const NUM_CHALLENGES: usize = 8;

const SEL_ADD: usize = Opcode::AddMult as usize;
const SEL_LOOKUP: usize = Opcode::Lookup as usize;
const SEL_DIV: usize = Opcode::Div as usize;
const SEL_MATMULT: usize = Opcode::MatMult as usize;
const SEL_CONV: usize = Opcode::Conv as usize;

/// Builds the rule list for a given public profile.
pub(crate) fn build_rules<F: PrimeField>(profile: &RegistrationProfile) -> Rules<F> {
    let q_log = profile.log_instr();
    let cons_log = profile.log_cons();
    let lu_log = profile.log_lu();
    let entry_log = profile.log_entry();
    let b_pre_log = profile.log_b_pre();
    let x_log = profile.log_x();
    let q = profile.instr_count;
    let weight_len = profile.weight_len as i64;
    let flag = profile.row_flag as i64;
    let (a_lo, a_hi) = profile.segment(0);
    let (b_lo, b_hi) = profile.segment(1);
    let (c_lo, c_hi) = profile.segment(2);
    let (d_lo, d_hi) = profile.segment(3);
    let (e_lo, e_hi) = profile.segment(4);
    let bound_tag = LookupType::Bound(QUOTIENT_BOUND).tag();
    let fams = families();
    let nf = fams.len();
    let fam_ind: Vec<Tab> = (0..nf)
        .map(|f| {
            let (lo, hi) = profile.family_range(f);
            Tab::Ind(lo, hi)
        })
        .collect();
    let fam_lo: Vec<i64> = (0..nf).map(|f| profile.family_range(f).0 as i64).collect();

    let mut zero_checks: Vec<ZeroCheck<F>> = vec![];

    // Every instruction picks exactly one opcode, and the padding past the
    // program picks none.
    zero_checks.push(ZeroCheck {
        name: "selectors are one-hot",
        log_len: q_log,
        degree: 3,
        tables: (0..NUM_OPCODES)
            .map(Tab::Sel)
            .chain(std::iter::once(Tab::Ind(0, q)))
            .collect(),
        g: Box::new(|v: &[F], ch: &[F]| {
            let mut acc = F::ZERO;
            let mut weight = ch[0];
            let mut total = F::ZERO;
            for j in 0..NUM_OPCODES {
                acc += weight * v[j] * (v[j] - F::ONE);
                total += v[j];
                weight *= ch[0];
            }
            acc + weight * (total - v[NUM_OPCODES])
        }),
    });

    // The counter increments are the ones the compiler applies. Keeping them
    // in a committed vector is what makes the recurrence below affine.
    let mut increment_tables: Vec<Tab> = (0..NUM_OPCODES).map(Tab::Sel).collect();
    increment_tables.extend((0..NUM_OPERANDS).map(Tab::Opn));
    increment_tables.extend((0..NUM_COUNTERS).map(Tab::DCtr));
    zero_checks.push(ZeroCheck {
        name: "counter increments match the opcode",
        log_len: q_log,
        degree: 5,
        tables: increment_tables,
        g: Box::new(|v: &[F], ch: &[F]| {
            let sel = |j: usize| v[j];
            let opn = |j: usize| v[NUM_OPCODES + j];
            let dctr = |j: usize| v[NUM_OPCODES + NUM_OPERANDS + j];
            let side = opn(2) + opn(3) - F::ONE;
            let d_out = sel(SEL_ADD)
                + sel(SEL_LOOKUP)
                + sel(SEL_DIV)
                + sel(SEL_MATMULT) * opn(0) * opn(2)
                + sel(SEL_CONV) * opn(1) * side * side;
            let d_aux = sel(SEL_DIV) + sel(SEL_MATMULT) * opn(1) + sel(SEL_CONV) * opn(0);
            let d_cr = sel(SEL_ADD)
                + sel(SEL_DIV)
                + sel(SEL_MATMULT) * (opn(1) + F::ONE)
                + sel(SEL_CONV) * (opn(0) + F::ONE);
            let d_lr = sel(SEL_LOOKUP) + sel(SEL_DIV).double();
            ch[1] * (dctr(CTR_OUT) - d_out)
                + ch[2] * (dctr(CTR_AUX) - d_aux)
                + ch[3] * (dctr(CTR_CR) - d_cr)
                + ch[4] * (dctr(CTR_LR) - d_lr)
        }),
    });

    // Number of entries each instruction contributes to each expansion family.
    let mut mult_tables: Vec<Tab> = (0..NUM_OPCODES).map(Tab::Sel).collect();
    mult_tables.extend((0..NUM_OPERANDS).map(Tab::Opn));
    mult_tables.extend((0..nf).map(Tab::Mult));
    let mult_families = families();
    zero_checks.push(ZeroCheck {
        name: "family sizes match the operands",
        log_len: q_log,
        degree: 6,
        tables: mult_tables,
        g: Box::new(move |v: &[F], ch: &[F]| {
            let ops: Vec<F> = (0..NUM_OPERANDS).map(|j| v[NUM_OPCODES + j]).collect();
            let mut acc = F::ZERO;
            let mut weight = ch[6];
            for (f, family) in mult_families.iter().enumerate() {
                let size = family
                    .radices
                    .iter()
                    .fold(F::ONE, |acc, r| acc * r.eval::<F>(&ops));
                let sel = v[family.opcode.index()];
                acc += weight * (v[NUM_OPCODES + NUM_OPERANDS + f] - sel * size);
                weight *= ch[6];
            }
            acc
        }),
    });

    // The slack a read leaves against its bound, which the range check below
    // forces to be a non-negative integer. Entries a linear-algebra
    // instruction owns are pinned entry by entry instead, and C entries are
    // pinned by their own equality, so neither carries slack.
    zero_checks.push(ZeroCheck {
        name: "read slack is well formed",
        log_len: entry_log,
        degree: 3,
        tables: vec![
            Tab::Diff,
            Tab::MaxRead,
            Tab::Col,
            Tab::Ind(a_lo, a_hi),
            Tab::Ind(b_lo, b_hi),
            Tab::Ind(d_lo, d_hi),
            Tab::MaskA,
            Tab::MaskB,
        ],
        g: Box::new(move |v: &[F], _ch: &[F]| {
            let plain = v[3] + v[4] + v[5] - v[6] - v[7];
            v[0] - plain * (v[1] - F::from(weight_len as u64) - v[2] - F::ONE)
        }),
    });

    // Prefix columns address the model parameters, which every instruction may
    // read; they only have to stay inside the parameter block.
    zero_checks.push(ZeroCheck {
        name: "prefix slack is well formed",
        log_len: b_pre_log,
        degree: 3,
        tables: vec![Tab::DiffPre, Tab::BCol, Tab::Ind(0, profile.b_pre_len)],
        g: Box::new(move |v: &[F], _ch: &[F]| {
            v[0] - v[2] * (F::from(weight_len as u64) - F::ONE - v[1])
        }),
    });

    // A `Div` emits its range checks with an empty input row, so only rows a
    // `Lookup` owns may carry entries on the D side.
    zero_checks.push(ZeroCheck {
        name: "only lookup rows read inputs",
        log_len: lu_log,
        degree: 3,
        tables: vec![Tab::CntL, Tab::IsLookup],
        g: Box::new(|v: &[F], _ch: &[F]| v[0] * (F::ONE - v[1])),
    });

    // Only the linear-algebra instructions randomize their rows; the rows an
    // AddMult, Div or Lookup owns carry no powers of gamma.
    zero_checks.push(ZeroCheck {
        name: "plain rows carry no powers",
        log_len: entry_log,
        degree: 3,
        tables: vec![
            Tab::Pow,
            Tab::Ind(a_lo, a_hi),
            Tab::Ind(b_lo, b_hi),
            Tab::Ind(d_lo, d_hi),
            Tab::MaskA,
            Tab::MaskB,
        ],
        g: Box::new(|v: &[F], _ch: &[F]| (v[1] + v[2] + v[3] - v[4] - v[5]) * v[0]),
    });
    zero_checks.push(ZeroCheck {
        name: "prefix rows carry no powers",
        log_len: b_pre_log,
        degree: 3,
        tables: vec![
            Tab::BPow,
            Tab::Ind(0, profile.b_pre_len),
            Tab::MaskPre,
        ],
        g: Box::new(|v: &[F], _ch: &[F]| (v[1] - v[2]) * v[0]),
    });

    // Masks split each descriptor segment into the entries a linear-algebra
    // instruction owns and the rest.
    zero_checks.push(ZeroCheck {
        name: "segment masks follow the rows",
        log_len: entry_log,
        degree: 3,
        tables: vec![
            Tab::MaskA,
            Tab::MaskB,
            Tab::MaskC,
            Tab::EntLin,
            Tab::Ind(a_lo, a_hi),
            Tab::Ind(b_lo, b_hi),
            Tab::Ind(c_lo, c_hi),
        ],
        g: Box::new(|v: &[F], ch: &[F]| {
            ch[5] * (v[0] - v[4] * v[3])
                + ch[5] * ch[5] * (v[1] - v[5] * v[3])
                + ch[5] * ch[5] * ch[5] * (v[2] - v[6] * v[3])
        }),
    });
    zero_checks.push(ZeroCheck {
        name: "prefix mask follows the rows",
        log_len: b_pre_log,
        degree: 3,
        tables: vec![Tab::MaskPre, Tab::BEntLin, Tab::Ind(0, profile.b_pre_len)],
        g: Box::new(|v: &[F], _ch: &[F]| v[0] - v[2] * v[1]),
    });

    // Each constraint row sits inside its owner's block of rows.
    zero_checks.push(ZeroCheck {
        name: "rows sit inside their block",
        log_len: cons_log,
        degree: 3,
        tables: vec![
            Tab::RowSlackLo,
            Tab::RowSlackHi,
            Tab::RowCr,
            Tab::RowDcr,
            Tab::Identity,
            Tab::Ind(0, profile.cons_line),
        ],
        g: Box::new(|v: &[F], ch: &[F]| {
            ch[5] * (v[0] - v[5] * (v[4] - v[2]))
                + ch[5] * ch[5] * (v[1] - v[5] * (v[2] + v[3] - F::ONE - v[4]))
        }),
    });

    // --- the expansion table ---
    //
    // Inside a family the entries of one instruction form a contiguous run, so
    // the flat position is `index - family start - run start`, and the loop
    // coordinates are its mixed-radix digits.
    let mut x_tables = vec![Tab::Identity, Tab::XBase];
    x_tables.extend((0..MAX_DIGITS).map(Tab::XDigit));
    x_tables.extend((0..NUM_OPERANDS).map(Tab::XOp));
    x_tables.extend(fam_ind.iter().copied());
    let digit_families = families();
    let digit_lo = fam_lo.clone();
    zero_checks.push(ZeroCheck {
        name: "expansion digits decode the position",
        log_len: x_log,
        degree: 6,
        tables: x_tables,
        g: Box::new(move |v: &[F], ch: &[F]| {
            let ident = v[0];
            let base = v[1];
            let digit = |j: usize| v[2 + j];
            let ops: Vec<F> = (0..NUM_OPERANDS).map(|j| v[2 + MAX_DIGITS + j]).collect();
            let ind = |f: usize| v[2 + MAX_DIGITS + NUM_OPERANDS + f];
            let mut acc = F::ZERO;
            let mut weight = ch[7];
            for (f, family) in digit_families.iter().enumerate() {
                let flat = ident - signed::<F>(digit_lo[f]) - base;
                let mut combined = F::ZERO;
                for j in 0..family.digits() {
                    let radix = family.radices[j].eval::<F>(&ops);
                    combined = if j == 0 {
                        digit(0)
                    } else {
                        combined * radix + digit(j)
                    };
                }
                // Unused coordinates stay zero.
                let mut unused = F::ZERO;
                for j in family.digits()..MAX_DIGITS {
                    unused += digit(j);
                }
                acc += weight * ind(f) * (flat - combined + unused);
                weight *= ch[7];
            }
            acc
        }),
    });

    let mut slack_tables: Vec<Tab> = (0..MAX_DIGITS).map(Tab::XDigitSlack).collect();
    slack_tables.extend((0..MAX_DIGITS).map(Tab::XDigit));
    slack_tables.extend((0..NUM_OPERANDS).map(Tab::XOp));
    slack_tables.extend(fam_ind.iter().copied());
    let slack_families = families();
    zero_checks.push(ZeroCheck {
        name: "expansion digits stay inside their loop",
        log_len: x_log,
        degree: 4,
        tables: slack_tables,
        g: Box::new(move |v: &[F], ch: &[F]| {
            let slack = |j: usize| v[j];
            let digit = |j: usize| v[MAX_DIGITS + j];
            let ops: Vec<F> = (0..NUM_OPERANDS).map(|j| v[2 * MAX_DIGITS + j]).collect();
            let ind = |f: usize| v[2 * MAX_DIGITS + NUM_OPERANDS + f];
            let mut acc = F::ZERO;
            let mut weight = ch[7];
            for (f, family) in slack_families.iter().enumerate() {
                for j in 0..MAX_DIGITS {
                    let expected = if j < family.digits() {
                        family.radices[j].eval::<F>(&ops) - F::ONE - digit(j)
                    } else {
                        F::ZERO
                    };
                    acc += weight * ind(f) * (slack(j) - expected);
                    weight *= ch[7];
                }
            }
            acc
        }),
    });

    let mut pow_tables = vec![Tab::XPow, Tab::Identity, Tab::XBase];
    pow_tables.extend((0..MAX_DIGITS).map(Tab::XDigit));
    pow_tables.extend((0..NUM_OPERANDS).map(Tab::XOp));
    pow_tables.extend(fam_ind.iter().copied());
    let pow_families = families();
    let pow_lo = fam_lo.clone();
    zero_checks.push(ZeroCheck {
        name: "expansion powers follow the family",
        log_len: x_log,
        degree: 5,
        tables: pow_tables,
        g: Box::new(move |v: &[F], ch: &[F]| {
            let x_pow = v[0];
            let ident = v[1];
            let base = v[2];
            let digit = |j: usize| v[3 + j];
            let ops: Vec<F> = (0..NUM_OPERANDS).map(|j| v[3 + MAX_DIGITS + j]).collect();
            let ind = |f: usize| v[3 + MAX_DIGITS + NUM_OPERANDS + f];
            let mut acc = F::ZERO;
            let mut weight = ch[7];
            for (f, family) in pow_families.iter().enumerate() {
                let flat = ident - signed::<F>(pow_lo[f]) - base;
                let mut expected = family.pow_const.eval::<F>(&ops);
                for (coeff, term) in &family.pow {
                    let value = match term {
                        PowTerm::Digit(j) => digit(*j),
                        PowTerm::Flat => flat,
                    };
                    expected += coeff.eval::<F>(&ops) * value;
                }
                acc += weight * ind(f) * (x_pow - expected);
                weight *= ch[7];
            }
            acc
        }),
    });

    let mut equalities = vec![];

    // Counters advance by their increment, checked as a multiset equality
    // between the shifted and the unshifted sequence. The expansion bases ride
    // along, since they are prefix sums of the family sizes.
    let top = (1usize << q_log) - 1;
    let mut shifted = vec![Lin::of(Tab::Identity).shift(1)];
    let mut plain = vec![Lin::of(Tab::Identity)];
    for c in 0..NUM_COUNTERS {
        shifted.push(Lin::of(Tab::Ctr(c)).plus(1, Tab::DCtr(c)));
        plain.push(Lin::of(Tab::Ctr(c)));
    }
    for f in 0..nf {
        shifted.push(Lin::of(Tab::Base(f)).plus(1, Tab::Mult(f)));
        plain.push(Lin::of(Tab::Base(f)));
    }
    equalities.push(Equality {
        name: "counters advance",
        left: vec![Block {
            log_len: q_log,
            num: Lin::constant(1).plus(-1, Tab::Ind(top, top + 1)),
            comps: shifted,
        }],
        right: vec![Block {
            log_len: q_log,
            num: Lin::constant(1).plus(-1, Tab::Ind(0, 1)),
            comps: plain,
        }],
    });

    // Each constraint row belongs to one instruction and inherits its read
    // bound; the slacks above pin which rows of the block it may be.
    equalities.push(Equality {
        name: "constraint rows belong to their instruction",
        left: vec![Block {
            log_len: cons_log,
            num: Lin::of(Tab::Ind(0, profile.cons_line)),
            comps: vec![
                Lin::of(Tab::RowPtr),
                Lin::of(Tab::RowCr),
                Lin::of(Tab::RowOut),
                Lin::of(Tab::RowDcr),
                Lin::of(Tab::IsLin),
            ],
        }],
        right: vec![Block {
            log_len: q_log,
            num: Lin::of(Tab::DCtr(CTR_CR)),
            comps: vec![
                Lin::of(Tab::Identity),
                Lin::of(Tab::Ctr(CTR_CR)),
                Lin::of(Tab::Ctr(CTR_OUT)),
                Lin::of(Tab::DCtr(CTR_CR)),
                Lin::of(Tab::Sel(SEL_MATMULT)).plus(1, Tab::Sel(SEL_CONV)),
            ],
        }],
    });

    // Same for lookup rows, which additionally record whether they came from a
    // `Lookup` (and so may read an input) or from a `Div` range check.
    equalities.push(Equality {
        name: "lookup rows carry their bound",
        left: vec![Block {
            log_len: lu_log,
            num: Lin::of(Tab::Ind(0, profile.lu_line)),
            comps: vec![
                Lin::of(Tab::Identity),
                Lin::of(Tab::LuOut),
                Lin::of(Tab::IsLookup),
            ],
        }],
        right: vec![
            Block {
                log_len: q_log,
                num: Lin::of(Tab::Sel(SEL_LOOKUP)).plus(1, Tab::Sel(SEL_DIV)),
                comps: vec![
                    Lin::of(Tab::Ctr(CTR_LR)),
                    Lin::of(Tab::Ctr(CTR_OUT)),
                    Lin::of(Tab::Sel(SEL_LOOKUP)),
                ],
            },
            Block {
                log_len: q_log,
                num: Lin::of(Tab::Sel(SEL_DIV)),
                comps: vec![
                    Lin::of(Tab::Ctr(CTR_LR)).shift(1),
                    Lin::of(Tab::Ctr(CTR_OUT)),
                    Lin::constant(0),
                ],
            },
        ],
    });

    // Every A, B and C entry looks up the bound of the row it sits in, and
    // whether that row belongs to a linear-algebra instruction.
    equalities.push(Equality {
        name: "constraint entries fetch their row",
        left: vec![Block {
            log_len: entry_log,
            num: Lin::of(Tab::Ind(a_lo, a_hi))
                .plus(1, Tab::Ind(b_lo, b_hi))
                .plus(1, Tab::Ind(c_lo, c_hi)),
            comps: vec![
                Lin::of(Tab::Row),
                Lin::of(Tab::MaxRead),
                Lin::of(Tab::EntLin),
            ],
        }],
        right: vec![Block {
            log_len: cons_log,
            num: Lin::of(Tab::CntC),
            comps: vec![
                Lin::of(Tab::Identity),
                Lin::of(Tab::RowOut),
                Lin::of(Tab::IsLin),
            ],
        }],
    });

    // ... and so does every D entry, against the lookup rows.
    equalities.push(Equality {
        name: "lookup entries fetch their bound",
        left: vec![Block {
            log_len: entry_log,
            num: Lin::of(Tab::Ind(d_lo, d_hi)),
            comps: vec![Lin::of(Tab::Row).shift(-flag), Lin::of(Tab::MaxRead)],
        }],
        right: vec![Block {
            log_len: lu_log,
            num: Lin::of(Tab::CntL),
            comps: vec![Lin::of(Tab::Identity), Lin::of(Tab::LuOut)],
        }],
    });

    // Prefix entries do the same, so the mask that exempts them from the
    // plain-row rules cannot be set freely.
    equalities.push(Equality {
        name: "prefix entries fetch their row",
        left: vec![Block {
            log_len: b_pre_log,
            num: Lin::of(Tab::Ind(0, profile.b_pre_len)),
            comps: vec![Lin::of(Tab::BRow), Lin::of(Tab::BEntLin)],
        }],
        right: vec![Block {
            log_len: cons_log,
            num: Lin::of(Tab::BCnt),
            comps: vec![Lin::of(Tab::Identity), Lin::of(Tab::IsLin)],
        }],
    });

    // Every slack and coordinate lands in `[0, 2^log_range)`, which turns the
    // affine identities above into the inequalities they stand for.
    let mut range_left = vec![
        Block {
            log_len: entry_log,
            num: Lin::of(Tab::One),
            comps: vec![Lin::of(Tab::Diff)],
        },
        Block {
            log_len: b_pre_log,
            num: Lin::of(Tab::One),
            comps: vec![Lin::of(Tab::DiffPre)],
        },
        Block {
            log_len: cons_log,
            num: Lin::of(Tab::One),
            comps: vec![Lin::of(Tab::RowSlackLo)],
        },
        Block {
            log_len: cons_log,
            num: Lin::of(Tab::One),
            comps: vec![Lin::of(Tab::RowSlackHi)],
        },
    ];
    for j in 0..MAX_DIGITS {
        range_left.push(Block {
            log_len: x_log,
            num: Lin::of(Tab::One),
            comps: vec![Lin::of(Tab::XDigit(j))],
        });
        range_left.push(Block {
            log_len: x_log,
            num: Lin::of(Tab::One),
            comps: vec![Lin::of(Tab::XDigitSlack(j))],
        });
    }
    equalities.push(Equality {
        name: "slacks are in range",
        left: range_left,
        right: vec![Block {
            log_len: profile.log_range,
            num: Lin::of(Tab::RangeCnt),
            comps: vec![Lin::of(Tab::Identity)],
        }],
    });

    // The output side of plain instructions is pinned exactly: these two
    // equalities are what make the committed circuit compute their outputs.
    equalities.push(Equality {
        name: "product rows write the compiled cells",
        left: vec![Block {
            log_len: entry_log,
            num: Lin::of(Tab::Ind(c_lo, c_hi)).plus(-1, Tab::MaskC),
            comps: vec![
                Lin::of(Tab::Row),
                Lin::of(Tab::Col).shift(weight_len),
                Lin::of(Tab::Val),
                Lin::of(Tab::Pow),
            ],
        }],
        right: vec![
            Block {
                log_len: q_log,
                num: Lin::of(Tab::Sel(SEL_ADD)),
                comps: vec![
                    Lin::of(Tab::Ctr(CTR_CR)),
                    Lin::of(Tab::Ctr(CTR_OUT)),
                    Lin::constant(1),
                    Lin::constant(0),
                ],
            },
            // <A, z> * <B, z> = divisor * z[out] + z[aux]
            Block {
                log_len: q_log,
                num: Lin::of(Tab::Sel(SEL_DIV)),
                comps: vec![
                    Lin::of(Tab::Ctr(CTR_CR)),
                    Lin::of(Tab::Ctr(CTR_OUT)),
                    Lin::of(Tab::Opn(0)),
                    Lin::constant(0),
                ],
            },
            Block {
                log_len: q_log,
                num: Lin::of(Tab::Sel(SEL_DIV)),
                comps: vec![
                    Lin::of(Tab::Ctr(CTR_CR)),
                    Lin::of(Tab::Ctr(CTR_AUX)),
                    Lin::constant(1),
                    Lin::constant(0),
                ],
            },
        ],
    });

    equalities.push(Equality {
        name: "lookup rows write the compiled cells",
        left: vec![Block {
            log_len: entry_log,
            num: Lin::of(Tab::Ind(e_lo, e_hi)),
            comps: vec![
                Lin::of(Tab::Row),
                Lin::of(Tab::Col).shift(weight_len),
                Lin::of(Tab::Val),
                Lin::of(Tab::Pow),
            ],
        }],
        right: vec![
            Block {
                log_len: q_log,
                num: Lin::of(Tab::Sel(SEL_LOOKUP)),
                comps: vec![
                    Lin::of(Tab::Ctr(CTR_LR)).shift(flag),
                    Lin::of(Tab::Ctr(CTR_OUT)),
                    Lin::constant(1),
                    Lin::constant(0),
                ],
            },
            // The remainder sits in the advice cell, ...
            Block {
                log_len: q_log,
                num: Lin::of(Tab::Sel(SEL_DIV)),
                comps: vec![
                    Lin::of(Tab::Ctr(CTR_LR)).shift(flag),
                    Lin::of(Tab::Ctr(CTR_AUX)),
                    Lin::constant(1),
                    Lin::constant(0),
                ],
            },
            // ... and the quotient carries its own range check.
            Block {
                log_len: q_log,
                num: Lin::of(Tab::Sel(SEL_DIV)),
                comps: vec![
                    Lin::of(Tab::Ctr(CTR_LR)).shift(flag + 1),
                    Lin::of(Tab::Ctr(CTR_OUT)),
                    Lin::constant(1),
                    Lin::constant(0),
                ],
            },
        ],
    });

    equalities.push(Equality {
        name: "lookup types are the compiled ones",
        left: vec![Block {
            log_len: lu_log,
            num: Lin::of(Tab::Ind(0, profile.lu_line)),
            comps: vec![Lin::of(Tab::Identity), Lin::of(Tab::Tp)],
        }],
        right: vec![
            Block {
                log_len: q_log,
                num: Lin::of(Tab::Sel(SEL_LOOKUP)),
                comps: vec![Lin::of(Tab::Ctr(CTR_LR)), Lin::of(Tab::Opn(0))],
            },
            // `Range(divisor)` tags as `-divisor`.
            Block {
                log_len: q_log,
                num: Lin::of(Tab::Sel(SEL_DIV)),
                comps: vec![
                    Lin::of(Tab::Ctr(CTR_LR)),
                    Lin {
                        constant: 0,
                        terms: vec![(-1, Tab::Opn(0))],
                    },
                ],
            },
            Block {
                log_len: q_log,
                num: Lin::of(Tab::Sel(SEL_DIV)),
                comps: vec![
                    Lin::of(Tab::Ctr(CTR_LR)).shift(1),
                    Lin::constant(bound_tag),
                ],
            },
        ],
    });

    // Each expansion entry carries its instruction's operands, fetched once
    // per family so that the run it belongs to is the one its base names.
    for (f, family) in fams.iter().enumerate() {
        let mut left_comps = vec![
            Lin::of(Tab::XPtr),
            Lin::of(Tab::XBase),
            Lin::of(Tab::XCr),
            Lin::of(Tab::XOut),
            Lin::of(Tab::XAux),
        ];
        left_comps.extend((0..NUM_OPERANDS).map(|j| Lin::of(Tab::XOp(j))));
        let mut right_comps = vec![
            Lin::of(Tab::Identity),
            Lin::of(Tab::Base(f)),
            Lin::of(Tab::Ctr(CTR_CR)),
            Lin::of(Tab::Ctr(CTR_OUT)),
            Lin::of(Tab::Ctr(CTR_AUX)),
        ];
        right_comps.extend((0..NUM_OPERANDS).map(|j| Lin::of(Tab::Opn(j))));
        equalities.push(Equality {
            name: family.name,
            left: vec![Block {
                log_len: x_log,
                num: Lin::of(fam_ind[f]),
                comps: left_comps,
            }],
            right: vec![Block {
                log_len: q_log,
                num: Lin::of(Tab::Mult(f)),
                comps: right_comps,
            }],
        });
    }

    // Finally, the entries a MatMult or Conv owns are exactly the expansion.
    let descriptor_entry = |mask: Tab, matrix: i64| Block {
        log_len: entry_log,
        num: Lin::of(mask),
        comps: vec![
            Lin::constant(matrix),
            Lin::of(Tab::Row),
            Lin::of(Tab::Col).shift(weight_len),
            Lin::of(Tab::Val),
            Lin::of(Tab::Pow),
        ],
    };
    let mut expansion = vec![];
    for (f, family) in fams.iter().enumerate() {
        let row = match &family.row {
            RowSpec::Digit(j) => Lin::of(Tab::XCr).plus(1, Tab::XDigit(*j)),
            RowSpec::Offset(ex) => match ex {
                Ex::Op(slot) => Lin::of(Tab::XCr).plus(1, Tab::XOp(*slot)),
                _ => panic!("row offsets have to be a single operand"),
            },
        };
        // Every family lays its columns out contiguously from a base, so the
        // column is that base plus the flat position in the run.
        let col_base = match family.col_base {
            ColBase::Operand(slot) => Lin::of(Tab::XOp(slot)),
            ColBase::Advice => Lin::of(Tab::XAux),
            ColBase::Output => Lin::of(Tab::XOut),
            ColBase::Unit => Lin::constant(0),
        };
        let col = col_base
            .plus(1, Tab::Identity)
            .plus(-1, Tab::XBase)
            .shift(-fam_lo[f]);
        expansion.push(Block {
            log_len: x_log,
            num: Lin::of(fam_ind[f]),
            comps: vec![
                Lin::constant(family.matrix as i64),
                row,
                col,
                Lin::constant(1),
                Lin::of(Tab::XPow),
            ],
        });
    }
    equalities.push(Equality {
        name: "linear-algebra rows are the expansion",
        left: vec![
            descriptor_entry(Tab::MaskA, 0),
            descriptor_entry(Tab::MaskB, 1),
            descriptor_entry(Tab::MaskC, 2),
            Block {
                log_len: b_pre_log,
                num: Lin::of(Tab::MaskPre),
                comps: vec![
                    Lin::constant(1),
                    Lin::of(Tab::BRow),
                    Lin::of(Tab::BCol),
                    Lin::of(Tab::BVal),
                    Lin::of(Tab::BPow),
                ],
            },
        ],
        right: expansion,
    });

    Rules {
        zero_checks,
        equalities,
    }
}
