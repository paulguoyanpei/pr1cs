//! Canonical encoding of a private VM program, and the in-the-clear rules that
//! say a compiled circuit is that program's compilation.
//!
//! The compiler in [`crate::program`] is deterministic and sequential: every
//! instruction appends its constraint rows in a fixed order and advances four
//! implicit counters (output cell, advice cell, constraint row, lookup row).
//! A program is therefore fully described by, per instruction, a one-hot
//! opcode selector and a handful of operands; the counters are prefix sums of
//! per-instruction increments that depend only on those two.
//!
//! [`check_compilation`] states every rule the registration argument has to
//! prove, but over the plain vectors instead of their commitments. It is the
//! reference the zero-knowledge rule set is tested against.

use std::fmt;

use ark_ff::PrimeField;

use crate::{
    circuit::{Circuit, LookupType, SparseRow, QUOTIENT_BOUND},
    instruction::Instruction,
    program::Program,
};

/// Number of instruction types, i.e. of one-hot selector vectors.
pub const NUM_OPCODES: usize = 5;
/// Width of the operand table. See [`ProgramEncoding`] for the layout.
pub const NUM_OPERANDS: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opcode {
    AddMult = 0,
    Lookup = 1,
    Div = 2,
    MatMult = 3,
    Conv = 4,
}

impl Opcode {
    pub fn index(self) -> usize {
        self as usize
    }
}

/// Per-instruction increments of the four implicit counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Deltas {
    pub out: usize,
    pub aux: usize,
    pub cr: usize,
    pub lr: usize,
}

/// The committed form of a private program.
///
/// Operand layout, one column per instruction:
///
/// | opcode    | opn0        | opn1         | opn2 | opn3 | opn4 | opn5 |
/// |-----------|-------------|--------------|------|------|------|------|
/// | `AddMult` | –           | –            | –    | –    | –    | –    |
/// | `Lookup`  | type tag    | –            | –    | –    | –    | –    |
/// | `Div`     | divisor     | –            | –    | –    | –    | –    |
/// | `MatMult` | m           | n            | k    | s1   | s2   | –    |
/// | `Conv`    | in_channels | out_channels | n    | m    | s1   | s2   |
///
/// Unused operand slots are zero. The sparse rows an `AddMult`, `Lookup` or
/// `Div` reads from are not operands: they live in the committed circuit
/// itself, and the rules below only bound which cells they may touch.
#[derive(Debug, Clone)]
pub struct ProgramEncoding {
    pub opcode: Vec<Opcode>,
    /// `sel[j][i] = 1` iff instruction `i` has opcode `j`.
    pub sel: Vec<Vec<i64>>,
    pub opn: Vec<Vec<i64>>,
    /// Start of each instruction's output block in the trace.
    pub out: Vec<usize>,
    /// Start of each instruction's advice block.
    pub aux: Vec<usize>,
    /// First constraint row each instruction emits.
    pub cr: Vec<usize>,
    /// First lookup row each instruction emits.
    pub lr: Vec<usize>,
    pub weight_len: usize,
    pub input_len: usize,
    pub aux_start: usize,
    pub z_len: usize,
    pub cons_line: usize,
    pub lu_line: usize,
}

impl ProgramEncoding {
    pub fn len(&self) -> usize {
        self.opcode.len()
    }

    pub fn is_empty(&self) -> bool {
        self.opcode.is_empty()
    }

    /// Increments of the four counters, as a function of opcode and operands.
    /// The registration argument proves the committed counters advance by
    /// exactly this much, so the formula lives in one place.
    pub fn deltas(opcode: Opcode, opn: &[i64]) -> Deltas {
        let o = |i: usize| opn[i] as usize;
        match opcode {
            Opcode::AddMult => Deltas {
                out: 1,
                aux: 0,
                cr: 1,
                lr: 0,
            },
            Opcode::Lookup => Deltas {
                out: 1,
                aux: 0,
                cr: 0,
                lr: 1,
            },
            Opcode::Div => Deltas {
                out: 1,
                aux: 1,
                cr: 1,
                lr: 2,
            },
            Opcode::MatMult => Deltas {
                // m * k outputs, one advice cell per shared dimension.
                out: o(0) * o(2),
                aux: o(1),
                cr: o(1) + 1,
                lr: 0,
            },
            Opcode::Conv => {
                let side = o(2) + o(3) - 1;
                Deltas {
                    out: o(1) * side * side,
                    aux: o(0),
                    cr: o(0) + 1,
                    lr: 0,
                }
            }
        }
    }

    /// Reads the encoding straight off a program. `aux_start` and `input_len`
    /// have to match the ones the circuit was compiled with.
    pub fn from_program<F: PrimeField>(
        program: &Program<F>,
        input_len: usize,
        aux_start: usize,
    ) -> Self {
        let weight_len = program.weight_len();
        let instructions = program.instructions();
        let q = instructions.len();

        let mut opcode = Vec::with_capacity(q);
        let mut sel = vec![vec![0i64; q]; NUM_OPCODES];
        let mut opn = vec![vec![0i64; q]; NUM_OPERANDS];
        let mut out = Vec::with_capacity(q);
        let mut aux = Vec::with_capacity(q);
        let mut cr = Vec::with_capacity(q);
        let mut lr = Vec::with_capacity(q);

        let mut out_cur = weight_len + input_len;
        let mut aux_cur = aux_start;
        let mut cr_cur = 0usize;
        let mut lr_cur = 0usize;

        for (i, instr) in instructions.iter().enumerate() {
            let (op, operands): (Opcode, [i64; NUM_OPERANDS]) = match instr {
                Instruction::AddMult { .. } => (Opcode::AddMult, [0; NUM_OPERANDS]),
                Instruction::Lookup { tp, .. } => {
                    (Opcode::Lookup, [tp.tag(), 0, 0, 0, 0, 0])
                }
                Instruction::Div { divisor, .. } => (Opcode::Div, [*divisor, 0, 0, 0, 0, 0]),
                Instruction::MatMult {
                    m,
                    n,
                    k,
                    start1,
                    start2,
                } => (
                    Opcode::MatMult,
                    [
                        *m as i64,
                        *n as i64,
                        *k as i64,
                        *start1 as i64,
                        *start2 as i64,
                        0,
                    ],
                ),
                Instruction::Conv {
                    n,
                    m,
                    in_channels,
                    out_channels,
                    start1,
                    start2,
                } => (
                    Opcode::Conv,
                    [
                        *in_channels as i64,
                        *out_channels as i64,
                        *n as i64,
                        *m as i64,
                        *start1 as i64,
                        *start2 as i64,
                    ],
                ),
            };

            opcode.push(op);
            sel[op.index()][i] = 1;
            for (slot, value) in operands.iter().enumerate() {
                opn[slot][i] = *value;
            }
            out.push(out_cur);
            aux.push(aux_cur);
            cr.push(cr_cur);
            lr.push(lr_cur);

            let d = Self::deltas(op, &operands);
            out_cur += d.out;
            aux_cur += d.aux;
            cr_cur += d.cr;
            lr_cur += d.lr;
        }

        ProgramEncoding {
            opcode,
            sel,
            opn,
            out,
            aux,
            cr,
            lr,
            weight_len,
            input_len,
            aux_start,
            z_len: aux_cur,
            cons_line: cr_cur,
            lu_line: lr_cur,
        }
    }

    /// Operand column of instruction `i`.
    pub(crate) fn operands_i64(&self, i: usize) -> Vec<i64> {
        self.operands(i)
    }

    fn operands(&self, i: usize) -> Vec<i64> {
        (0..NUM_OPERANDS).map(|slot| self.opn[slot][i]).collect()
    }
}

/// A rule of the compilation relation that the circuit does not satisfy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleViolation {
    pub instr: Option<usize>,
    pub rule: &'static str,
    pub detail: String,
}

impl fmt::Display for RuleViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.instr {
            Some(i) => write!(f, "instruction {}: {} ({})", i, self.rule, self.detail),
            None => write!(f, "{} ({})", self.rule, self.detail),
        }
    }
}

fn fail<T>(instr: Option<usize>, rule: &'static str, detail: String) -> Result<T, RuleViolation> {
    Err(RuleViolation {
        instr,
        rule,
        detail,
    })
}

/// Sorted `(col, val, pow)` view of a sparse row, so rows compare as multisets.
fn row_entries<F: PrimeField>(row: &SparseRow<F>) -> Vec<(usize, F, usize)> {
    let mut entries: Vec<(usize, F, usize)> = row
        .elems
        .iter()
        .map(|&(col, val, pow)| (col, val, pow.unwrap_or(0)))
        .collect();
    entries.sort_by_key(|&(col, _, pow)| (col, pow));
    entries
}

fn expect_row<F: PrimeField>(
    instr: usize,
    rule: &'static str,
    row: &SparseRow<F>,
    expected: &[(usize, i64, usize)],
) -> Result<(), RuleViolation> {
    let got = row_entries(row);
    let mut want: Vec<(usize, F, usize)> = expected
        .iter()
        .map(|&(col, val, pow)| (col, F::from(val), pow))
        .collect();
    want.sort_by_key(|&(col, _, pow)| (col, pow));
    if got == want {
        Ok(())
    } else {
        fail(
            Some(instr),
            rule,
            format!("expected {} entries, found {}", want.len(), got.len()),
        )
    }
}

/// Checks that a row only reads cells an instruction is allowed to read: the
/// parameters, the public input and the trace written before `limit`.
fn check_reads<F: PrimeField>(
    instr: usize,
    rule: &'static str,
    row: &SparseRow<F>,
    limit: usize,
) -> Result<(), RuleViolation> {
    for &(col, _, _) in &row.elems {
        if col >= limit {
            return fail(
                Some(instr),
                rule,
                format!("reads z[{}] with only z[..{}] available", col, limit),
            );
        }
    }
    Ok(())
}

/// Verifies that `circuit` is the compilation of the program `enc` encodes.
///
/// Every rule here is one the registration argument proves in zero knowledge
/// over the committed descriptor vectors; keeping the clear-text version means
/// the two can be cross-checked on the same fixtures.
pub fn check_compilation<F: PrimeField>(
    enc: &ProgramEncoding,
    circuit: &Circuit<F>,
) -> Result<(), RuleViolation> {
    check_shape(enc, circuit)?;
    check_counters(enc)?;
    for i in 0..enc.len() {
        check_instruction(enc, circuit, i)?;
    }
    Ok(())
}

fn check_shape<F: PrimeField>(
    enc: &ProgramEncoding,
    circuit: &Circuit<F>,
) -> Result<(), RuleViolation> {
    if circuit.weight_len != enc.weight_len {
        return fail(
            None,
            "weight length",
            format!("circuit {} vs program {}", circuit.weight_len, enc.weight_len),
        );
    }
    if circuit.a.len() != enc.cons_line
        || circuit.b.len() != enc.cons_line
        || circuit.c.len() != enc.cons_line
    {
        return fail(
            None,
            "constraint row count",
            format!(
                "a={} b={} c={} expected {}",
                circuit.a.len(),
                circuit.b.len(),
                circuit.c.len(),
                enc.cons_line
            ),
        );
    }
    if circuit.d.len() != enc.lu_line
        || circuit.e.len() != enc.lu_line
        || circuit.tp.len() != enc.lu_line
    {
        return fail(
            None,
            "lookup row count",
            format!(
                "d={} e={} tp={} expected {}",
                circuit.d.len(),
                circuit.e.len(),
                circuit.tp.len(),
                enc.lu_line
            ),
        );
    }
    if circuit.z_len != enc.z_len {
        return fail(
            None,
            "witness length",
            format!("circuit {} vs program {}", circuit.z_len, enc.z_len),
        );
    }
    Ok(())
}

fn check_counters(enc: &ProgramEncoding) -> Result<(), RuleViolation> {
    let mut out = enc.weight_len + enc.input_len;
    let mut aux = enc.aux_start;
    let mut cr = 0usize;
    let mut lr = 0usize;
    for i in 0..enc.len() {
        for (name, got, want) in [
            ("output counter", enc.out[i], out),
            ("advice counter", enc.aux[i], aux),
            ("constraint counter", enc.cr[i], cr),
            ("lookup counter", enc.lr[i], lr),
        ] {
            if got != want {
                return fail(Some(i), name, format!("{} != {}", got, want));
            }
        }
        for (name, value) in [("selector", enc.sel.len())] {
            if value != NUM_OPCODES {
                return fail(Some(i), name, format!("{} vectors", value));
            }
        }
        let ones: i64 = (0..NUM_OPCODES).map(|j| enc.sel[j][i]).sum();
        let one_hot = (0..NUM_OPCODES).all(|j| enc.sel[j][i] == 0 || enc.sel[j][i] == 1);
        if ones != 1 || !one_hot || enc.sel[enc.opcode[i].index()][i] != 1 {
            return fail(Some(i), "selector one-hot", format!("sum {}", ones));
        }
        let d = ProgramEncoding::deltas(enc.opcode[i], &enc.operands(i));
        out += d.out;
        aux += d.aux;
        cr += d.cr;
        lr += d.lr;
    }
    if out != enc.aux_start {
        return fail(
            None,
            "trace end",
            format!("trace ends at {}, advice starts at {}", out, enc.aux_start),
        );
    }
    if aux != enc.z_len || cr != enc.cons_line || lr != enc.lu_line {
        return fail(
            None,
            "counter totals",
            format!("aux {} cr {} lr {}", aux, cr, lr),
        );
    }
    Ok(())
}

fn check_instruction<F: PrimeField>(
    enc: &ProgramEncoding,
    circuit: &Circuit<F>,
    i: usize,
) -> Result<(), RuleViolation> {
    let opn = enc.operands(i);
    let out = enc.out[i];
    let aux = enc.aux[i];
    let cr = enc.cr[i];
    let lr = enc.lr[i];

    match enc.opcode[i] {
        Opcode::AddMult => {
            check_reads(i, "AddMult reads", &circuit.a.rows[cr], out)?;
            check_reads(i, "AddMult reads", &circuit.b.rows[cr], out)?;
            expect_row(i, "AddMult output", &circuit.c.rows[cr], &[(out, 1, 0)])?;
        }
        Opcode::Lookup => {
            check_reads(i, "Lookup reads", &circuit.d.rows[lr], out)?;
            expect_row(i, "Lookup output", &circuit.e.rows[lr], &[(out, 1, 0)])?;
            if circuit.tp[lr] != F::from(opn[0]) {
                return fail(Some(i), "Lookup type", format!("tag {}", opn[0]));
            }
        }
        Opcode::Div => {
            let divisor = opn[0];
            if divisor <= 0 {
                return fail(Some(i), "Div divisor", format!("{}", divisor));
            }
            check_reads(i, "Div reads", &circuit.a.rows[cr], out)?;
            check_reads(i, "Div reads", &circuit.b.rows[cr], out)?;
            // <A[cr], z> * <B[cr], z> = divisor * z[out] + z[aux]
            expect_row(
                i,
                "Div output",
                &circuit.c.rows[cr],
                &[(out, divisor, 0), (aux, 1, 0)],
            )?;
            // Remainder in [0, divisor), read out of the advice cell.
            expect_row(i, "Div remainder input", &circuit.d.rows[lr], &[])?;
            expect_row(i, "Div remainder", &circuit.e.rows[lr], &[(aux, 1, 0)])?;
            if circuit.tp[lr] != F::from(LookupType::Range(divisor).tag()) {
                return fail(Some(i), "Div remainder type", format!("row {}", lr));
            }
            // The remainder check alone leaves `divisor` valid quotients, so
            // the quotient carries its own range check.
            expect_row(i, "Div quotient input", &circuit.d.rows[lr + 1], &[])?;
            expect_row(i, "Div quotient", &circuit.e.rows[lr + 1], &[(out, 1, 0)])?;
            if circuit.tp[lr + 1] != F::from(LookupType::Bound(QUOTIENT_BOUND).tag()) {
                return fail(Some(i), "Div quotient type", format!("row {}", lr + 1));
            }
        }
        Opcode::MatMult => {
            let (m, n, k) = (opn[0] as usize, opn[1] as usize, opn[2] as usize);
            let (s1, s2) = (opn[3] as usize, opn[4] as usize);
            if m == 0 || n == 0 || k == 0 {
                return fail(Some(i), "MatMult dimensions", format!("{}x{}x{}", m, n, k));
            }
            if s1 + n * m > out || s2 + n * k > out {
                return fail(
                    Some(i),
                    "MatMult operand range",
                    format!("s1={} s2={} out={}", s1, s2, out),
                );
            }
            for row in 0..n {
                let a_expected: Vec<_> = (0..m).map(|j| (s1 + row * m + j, 1, j * k)).collect();
                expect_row(i, "MatMult X row", &circuit.a.rows[cr + row], &a_expected)?;
                let b_expected: Vec<_> = (0..k).map(|j| (s2 + row * k + j, 1, j)).collect();
                expect_row(i, "MatMult Y row", &circuit.b.rows[cr + row], &b_expected)?;
                expect_row(
                    i,
                    "MatMult advice",
                    &circuit.c.rows[cr + row],
                    &[(aux + row, 1, 0)],
                )?;
            }
            // Freivalds check: the advice products sum to the randomized
            // combination of the output block.
            let sum_expected: Vec<_> = (0..n).map(|j| (aux + j, 1, 0)).collect();
            expect_row(i, "MatMult advice sum", &circuit.a.rows[cr + n], &sum_expected)?;
            expect_row(i, "MatMult sum unit", &circuit.b.rows[cr + n], &[(0, 1, 0)])?;
            let out_expected: Vec<_> = (0..m)
                .flat_map(|r| (0..k).map(move |j| (r * k + j, r * k + j)))
                .map(|(offset, pow)| (out + offset, 1, pow))
                .collect();
            expect_row(i, "MatMult output", &circuit.c.rows[cr + n], &out_expected)?;
        }
        Opcode::Conv => {
            let (channels, filters) = (opn[0] as usize, opn[1] as usize);
            let (n, m) = (opn[2] as usize, opn[3] as usize);
            let (s1, s2) = (opn[4] as usize, opn[5] as usize);
            if channels == 0 || filters == 0 || n == 0 || m == 0 {
                return fail(
                    Some(i),
                    "Conv dimensions",
                    format!("C={} D={} n={} m={}", channels, filters, n, m),
                );
            }
            let side = n + m - 1;
            let plane = side * side;
            let input_plane = n * n;
            let kernel_plane = m * m;
            if s1 + channels * input_plane > out || s2 + channels * filters * kernel_plane > out {
                return fail(
                    Some(i),
                    "Conv operand range",
                    format!("s1={} s2={} out={}", s1, s2, out),
                );
            }
            for ch in 0..channels {
                let a_expected: Vec<_> = (0..n)
                    .flat_map(|r| (0..n).map(move |c| (r, c)))
                    .map(|(r, c)| (s1 + ch * input_plane + r * n + c, 1, r * side + c))
                    .collect();
                expect_row(i, "Conv input plane", &circuit.a.rows[cr + ch], &a_expected)?;
                let b_expected: Vec<_> = (0..filters)
                    .flat_map(|f| (0..m).flat_map(move |u| (0..m).map(move |v| (f, u, v))))
                    .map(|(f, u, v)| {
                        (
                            s2 + (ch * filters + f) * kernel_plane + u * m + v,
                            1,
                            f * plane + (m - 1 - u) * side + (m - 1 - v),
                        )
                    })
                    .collect();
                expect_row(i, "Conv kernel", &circuit.b.rows[cr + ch], &b_expected)?;
                expect_row(
                    i,
                    "Conv advice",
                    &circuit.c.rows[cr + ch],
                    &[(aux + ch, 1, 0)],
                )?;
            }
            let sum_expected: Vec<_> = (0..channels).map(|ch| (aux + ch, 1, 0)).collect();
            expect_row(
                i,
                "Conv advice sum",
                &circuit.a.rows[cr + channels],
                &sum_expected,
            )?;
            expect_row(i, "Conv sum unit", &circuit.b.rows[cr + channels], &[(0, 1, 0)])?;
            let out_expected: Vec<_> = (0..filters)
                .flat_map(|f| {
                    (0..side).flat_map(move |alpha| {
                        (0..side).map(move |beta| f * plane + alpha * side + beta)
                    })
                })
                .map(|offset| (out + offset, 1, offset))
                .collect();
            expect_row(
                i,
                "Conv output",
                &circuit.c.rows[cr + channels],
                &out_expected,
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cmp;

    use ark_bn254::Fr;

    use super::*;
    use crate::{circuit::LookupType, program::CompileError};

    const DIVISOR: i64 = 64;

    fn table() -> Vec<(Fr, Fr, Fr)> {
        let mut rows = (0i64..DIVISOR)
            .map(|i| {
                (
                    Fr::from(0),
                    Fr::from(i),
                    Fr::from(LookupType::Range(DIVISOR).tag()),
                )
            })
            .collect::<Vec<_>>();
        rows.extend(
            (-64i64..=64).map(|i| (Fr::from(i), Fr::from(cmp::max(0, i)), Fr::from(2u64))),
        );
        rows.extend((-64i64..=64).map(|i| {
            (
                Fr::from(0),
                Fr::from(i),
                Fr::from(LookupType::Bound(QUOTIENT_BOUND).tag()),
            )
        }));
        rows
    }

    /// MatMult, then Div, then a ReLU lookup: covers three of the five
    /// instructions plus every counter.
    fn matmult_program() -> (Program<Fr>, usize, usize) {
        let weights = vec![1, 1, 0, 0, 1];
        let input = vec![1, 2];
        let instructions = vec![
            Instruction::MatMult {
                m: 1,
                n: 2,
                k: 2,
                start1: weights.len(),
                start2: 1,
            },
            Instruction::Div {
                input1: vec![(weights.len() + input.len(), 1)],
                input2: vec![(0, 1)],
                divisor: DIVISOR,
            },
            Instruction::Lookup {
                input: vec![(weights.len() + input.len() + 2, 1)],
                tp: LookupType::Relu,
            },
        ];
        let input_len = input.len();
        let program = Program::<Fr>::new(instructions, weights);
        let aux_start = program.execute(input).len();
        (program, input_len, aux_start)
    }

    /// Conv, then an AddMult over two of its outputs, then a lookup.
    fn conv_program() -> (Program<Fr>, usize, usize) {
        let raw_weights = vec![1, 2, 3, 4, 5, 6, 7, 8, 1, -1, 2, -2, 3, -3, 4, -4];
        let input = vec![1, 2, 3, 4, -1, 0, 2, 1];
        let mut weights = vec![1];
        weights.extend(raw_weights.iter().copied());
        let conv_out = weights.len() + input.len();
        let instructions = vec![
            Instruction::Conv {
                n: 2,
                m: 2,
                in_channels: 2,
                out_channels: 2,
                start1: weights.len(),
                start2: 1,
            },
            Instruction::AddMult {
                input1: vec![(conv_out, 1)],
                input2: vec![(conv_out + 1, 1)],
            },
            Instruction::Lookup {
                input: vec![(conv_out, 1)],
                tp: LookupType::Relu,
            },
        ];
        let input_len = input.len();
        let program = Program::<Fr>::new(instructions, weights);
        let aux_start = program.execute(input).len();
        (program, input_len, aux_start)
    }

    #[test]
    fn encoding_matches_compiled_circuit() {
        for (program, input_len, aux_start) in [matmult_program(), conv_program()] {
            let circuit = program.compile(input_len, aux_start, table()).unwrap();
            let enc = ProgramEncoding::from_program(&program, input_len, aux_start);
            assert_eq!(enc.len(), program.instructions().len());
            assert_eq!(enc.cons_line, circuit.a.len());
            assert_eq!(enc.lu_line, circuit.d.len());
            assert_eq!(enc.z_len, circuit.z_len);
            check_compilation(&enc, &circuit).unwrap();
        }
    }

    #[test]
    fn rejects_read_of_future_trace() {
        let (_, input_len, aux_start) = matmult_program();
        let weights = vec![1, 1, 0, 0, 1];
        // The lookup reads the cell it is about to write.
        let program = Program::<Fr>::new(
            vec![Instruction::Lookup {
                input: vec![(weights.len() + input_len, 1)],
                tp: LookupType::Relu,
            }],
            weights,
        );
        let err = program.compile(input_len, aux_start, table()).unwrap_err();
        assert!(matches!(err, CompileError::ReadsFutureTrace { .. }), "{}", err);
    }

    #[test]
    fn rejects_read_of_advice_region() {
        let (program, input_len, aux_start) = matmult_program();
        // Cut the advice region short so the last lookup reads into it.
        let err = program
            .compile(input_len, aux_start - 3, table())
            .unwrap_err();
        assert!(matches!(err, CompileError::ReadsAdvice { .. }), "{}", err);
    }

    #[test]
    fn rejects_trace_length_mismatch() {
        let (program, input_len, aux_start) = matmult_program();
        let err = program
            .compile(input_len, aux_start + 1, table())
            .unwrap_err();
        assert!(
            matches!(err, CompileError::TraceLengthMismatch { .. }),
            "{}",
            err
        );
    }

    #[test]
    fn rejects_non_positive_divisor() {
        let weights = vec![1i64];
        let program = Program::<Fr>::new(
            vec![Instruction::Div {
                input1: vec![(1, 1)],
                input2: vec![(0, 1)],
                divisor: 0,
            }],
            weights,
        );
        let err = program.compile(1, 3, table()).unwrap_err();
        assert!(
            matches!(err, CompileError::NonPositiveDivisor { .. }),
            "{}",
            err
        );
    }

    #[test]
    fn rejects_range_check_used_as_standalone_lookup() {
        let weights = vec![1i64];
        let program = Program::<Fr>::new(
            vec![Instruction::Lookup {
                input: vec![(1, 1)],
                tp: LookupType::Range(DIVISOR),
            }],
            weights,
        );
        let err = program.compile(1, 3, table()).unwrap_err();
        assert!(
            matches!(err, CompileError::ReservedLookupType { .. }),
            "{}",
            err
        );
    }

    #[test]
    fn rejects_lookup_type_absent_from_table() {
        let (program, input_len, aux_start) = matmult_program();
        // Drop the ReLU rows the program's lookup needs.
        let rows = table()
            .into_iter()
            .filter(|&(_, _, tag)| tag != Fr::from(LookupType::Relu.tag()))
            .collect::<Vec<_>>();
        let err = program.compile(input_len, aux_start, rows).unwrap_err();
        assert!(
            matches!(err, CompileError::MissingLookupRows { .. }),
            "{}",
            err
        );
    }

    #[test]
    fn rejects_table_that_is_not_a_function() {
        let (program, input_len, aux_start) = matmult_program();
        let mut rows = table();
        // A second ReLU output for input 1 makes the lookup non-deterministic.
        rows.push((Fr::from(1), Fr::from(7), Fr::from(LookupType::Relu.tag())));
        let err = program.compile(input_len, aux_start, rows).unwrap_err();
        assert!(matches!(err, CompileError::AmbiguousTable { .. }), "{}", err);
    }

    #[test]
    fn catches_relabelled_output_cell() {
        let (program, input_len, aux_start) = matmult_program();
        let mut circuit = program.compile(input_len, aux_start, table()).unwrap();
        let enc = ProgramEncoding::from_program(&program, input_len, aux_start);
        // Point the MatMult output constraint at a different trace cell.
        circuit.c.rows[enc.cr[0] + 2].elems[0].0 += 1;
        let violation = check_compilation(&enc, &circuit).unwrap_err();
        assert_eq!(violation.rule, "MatMult output");
    }

    #[test]
    fn catches_tampered_randomization_exponent() {
        let (program, input_len, aux_start) = matmult_program();
        let mut circuit = program.compile(input_len, aux_start, table()).unwrap();
        let enc = ProgramEncoding::from_program(&program, input_len, aux_start);
        circuit.a.rows[enc.cr[0]].elems[0].2 = Some(3);
        let violation = check_compilation(&enc, &circuit).unwrap_err();
        assert_eq!(violation.rule, "MatMult X row");
    }

    #[test]
    fn catches_read_outside_the_trace_prefix() {
        let (program, input_len, aux_start) = matmult_program();
        let mut circuit = program.compile(input_len, aux_start, table()).unwrap();
        let enc = ProgramEncoding::from_program(&program, input_len, aux_start);
        // Let the Div read a cell that only later instructions write.
        circuit.a.rows[enc.cr[1]]
            .elems
            .push((enc.out[2], Fr::from(1u64), None));
        let violation = check_compilation(&enc, &circuit).unwrap_err();
        assert_eq!(violation.rule, "Div reads");
    }

    #[test]
    fn catches_swapped_lookup_type() {
        let (program, input_len, aux_start) = matmult_program();
        let mut circuit = program.compile(input_len, aux_start, table()).unwrap();
        let enc = ProgramEncoding::from_program(&program, input_len, aux_start);
        // Turn the quotient range check into a plain ReLU row.
        circuit.tp[enc.lr[1] + 1] = Fr::from(LookupType::Relu.tag());
        let violation = check_compilation(&enc, &circuit).unwrap_err();
        assert_eq!(violation.rule, "Div quotient type");
    }

    #[test]
    fn catches_dropped_freivalds_sum_row() {
        let (program, input_len, aux_start) = matmult_program();
        let mut circuit = program.compile(input_len, aux_start, table()).unwrap();
        let enc = ProgramEncoding::from_program(&program, input_len, aux_start);
        // Drop one advice cell from the sum that Freivalds' check relies on.
        circuit.a.rows[enc.cr[0] + 2].elems.pop();
        let violation = check_compilation(&enc, &circuit).unwrap_err();
        assert_eq!(violation.rule, "MatMult advice sum");
    }
}
