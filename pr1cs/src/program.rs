use std::{
    collections::{HashMap, HashSet},
    fmt,
    marker::PhantomData,
};

use ark_ff::PrimeField;

use crate::{
    circuit::{Circuit, LookupType, SparseMatrix, QUOTIENT_BOUND},
    instruction::Instruction,
};

/// Why `Program::compile` rejected a program, i.e. the cases in which the
/// paper's `Compile(P)` returns bottom.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileError {
    /// An instruction reads a trace cell that no earlier instruction wrote.
    ReadsFutureTrace {
        instr: usize,
        col: usize,
        limit: usize,
    },
    /// An instruction reads the advice region, which the VM never exposes to
    /// programs because the prover picks its contents.
    ReadsAdvice {
        instr: usize,
        col: usize,
        aux_start: usize,
    },
    /// `Div` divisors have to be positive for the quotient to be well defined.
    NonPositiveDivisor { instr: usize, divisor: i64 },
    /// A `MatMult` or `Conv` operand has a zero dimension.
    ZeroDimension { instr: usize },
    /// The public table has no rows for a lookup type the program uses.
    MissingLookupRows { instr: usize, tag: i64 },
    /// The public table sends one `(input, type)` pair to two outputs, so the
    /// lookup it defines is not a function.
    AmbiguousTable { row: usize },
    /// A range-check type appeared as a standalone `Lookup`. Those types
    /// constrain rather than compute, so they leave their output cell free.
    ReservedLookupType { instr: usize, tag: i64 },
    /// A matrix other than `B` reads a model parameter. Only `B` keeps its
    /// parameter columns in the committed descriptor, so such a read would be
    /// dropped during preprocessing instead of constraining anything.
    ParameterReadOutsideB { instr: usize, col: usize },
    /// The semantic trace does not end where the advice region starts.
    TraceLengthMismatch { trace_end: usize, aux_start: usize },
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReadsFutureTrace { instr, col, limit } => write!(
                f,
                "instruction {} reads z[{}], past the trace prefix z[..{}]",
                instr, col, limit
            ),
            Self::ReadsAdvice {
                instr,
                col,
                aux_start,
            } => write!(
                f,
                "instruction {} reads z[{}], inside the advice region z[{}..]",
                instr, col, aux_start
            ),
            Self::NonPositiveDivisor { instr, divisor } => {
                write!(f, "instruction {} divides by {}", instr, divisor)
            }
            Self::ZeroDimension { instr } => {
                write!(f, "instruction {} has a zero dimension", instr)
            }
            Self::MissingLookupRows { instr, tag } => write!(
                f,
                "instruction {} uses lookup tag {}, which the table does not cover",
                instr, tag
            ),
            Self::AmbiguousTable { row } => write!(
                f,
                "table row {} repeats an earlier (input, type) pair with a different output",
                row
            ),
            Self::ReservedLookupType { instr, tag } => write!(
                f,
                "instruction {} uses range-check tag {} as a standalone lookup",
                instr, tag
            ),
            Self::ParameterReadOutsideB { instr, col } => write!(
                f,
                "instruction {} reads parameter z[{}] on a side other than B",
                instr, col
            ),
            Self::TraceLengthMismatch {
                trace_end,
                aux_start,
            } => write!(
                f,
                "trace ends at {} but the advice region starts at {}",
                trace_end, aux_start
            ),
        }
    }
}

pub struct Program<F: PrimeField> {
    instructions: Vec<Instruction>,
    weights: Vec<i64>,
    lookup_tables: HashMap<LookupType, LookupTable>,
    _pd: PhantomData<F>,
}

#[derive(Clone)]
pub struct LookupTable {
    min: i64,
    values: Vec<i64>,
}

impl LookupTable {
    pub fn new(min: i64, values: Vec<i64>) -> Self {
        assert!(!values.is_empty());
        Self { min, values }
    }

    pub fn lookup(&self, input: i64) -> i64 {
        let max = self.min + self.values.len() as i64 - 1;
        let clamped = input.clamp(self.min, max);
        self.values[(clamped - self.min) as usize]
    }
}

impl<F: PrimeField> Program<F> {
    pub fn new(instructions: Vec<Instruction>, weights: Vec<i64>) -> Self {
        Program {
            instructions,
            weights,
            lookup_tables: HashMap::new(),
            _pd: PhantomData::default(),
        }
    }

    pub fn new_with_lookup_tables(
        instructions: Vec<Instruction>,
        weights: Vec<i64>,
        lookup_tables: HashMap<LookupType, LookupTable>,
    ) -> Self {
        Program {
            instructions,
            weights,
            lookup_tables,
            _pd: PhantomData::default(),
        }
    }

    /// Compiles the program into its canonical pR1CS circuit, or reports why
    /// the program is not in `P`, the set of valid VM programs. This is the
    /// paper's `Compile`, including the cases where it returns bottom.
    ///
    /// The validity conditions are what make the compiled circuit
    /// output-binding: every instruction may read only the model parameters,
    /// the public input and a prefix of the semantic trace, and no instruction
    /// may read the advice region, whose contents the prover picks freely.
    pub fn compile(
        &self,
        input_size: usize,
        aux_start: usize,
        table: Vec<(F, F, F)>,
    ) -> Result<Circuit<F>, CompileError> {
        let tags = self.table_tags(&table)?;
        let mut a = SparseMatrix::<F>::new();
        let mut b = SparseMatrix::<F>::new();
        let mut c = SparseMatrix::<F>::new();
        let mut d = SparseMatrix::<F>::new();
        let mut e = SparseMatrix::<F>::new();
        let mut tp = vec![];
        let mut output_index = self.weights.len() + input_size;
        let mut auxiliary_index = aux_start;

        for (instr_idx, instr) in self.instructions.iter().enumerate() {
            match &instr {
                &Instruction::AddMult { input1, input2 } => {
                    Self::check_no_parameter_read(instr_idx, input1, self.weights.len())?;
                    Self::check_reads(instr_idx, input1, output_index, aux_start)?;
                    Self::check_reads(instr_idx, input2, output_index, aux_start)?;
                    a.append(input1);
                    b.append(input2);
                    c.append(&vec![(output_index, 1)]);
                    output_index += 1
                }
                &Instruction::Conv {
                    n,
                    m,
                    in_channels,
                    out_channels,
                    start1,
                    start2,
                } => {
                    let n = *n;
                    let m = *m;
                    let in_channels = *in_channels;
                    let out_channels = *out_channels;
                    let start1 = *start1;
                    let start2 = *start2;
                    if n == 0 || m == 0 || in_channels == 0 || out_channels == 0 {
                        return Err(CompileError::ZeroDimension {
                            instr: instr_idx,
                        });
                    }
                    let side = n + m - 1;
                    let plane = side * side;
                    let input_plane = n * n;
                    let kernel_plane = m * m;
                    if start1 < self.weights.len() {
                        return Err(CompileError::ParameterReadOutsideB {
                            instr: instr_idx,
                            col: start1,
                        });
                    }
                    Self::check_block(
                        instr_idx,
                        start1,
                        in_channels * input_plane,
                        output_index,
                        aux_start,
                    )?;
                    Self::check_block(
                        instr_idx,
                        start2,
                        in_channels * out_channels * kernel_plane,
                        output_index,
                        aux_start,
                    )?;

                    // One product gate per input channel, then one final sum gate.
                    for ch in 0..in_channels {
                        let mut row = vec![];
                        for i in 0..n {
                            for j in 0..n {
                                row.push((
                                    start1 + ch * input_plane + i * n + j,
                                    1,
                                    Some(i * side + j),
                                ));
                            }
                        }
                        a.append_with_idx(&row);
                    }

                    for ch in 0..in_channels {
                        let mut row = vec![];
                        for out in 0..out_channels {
                            for u in 0..m {
                                for v in 0..m {
                                    row.push((
                                        start2
                                            + (ch * out_channels + out) * kernel_plane
                                            + u * m
                                            + v,
                                        1,
                                        Some(out * plane + (m - 1 - u) * side + (m - 1 - v)),
                                    ));
                                }
                            }
                        }
                        b.append_with_idx(&row);
                    }

                    for ch in 0..in_channels {
                        c.append(&vec![(auxiliary_index + ch, 1)]);
                    }

                    let mut row = vec![];
                    for ch in 0..in_channels {
                        row.push((auxiliary_index + ch, 1));
                    }
                    a.append(&row);
                    b.append(&vec![(0, 1)]);

                    let mut row = vec![];
                    for out in 0..out_channels {
                        for alpha in 0..side {
                            for beta in 0..side {
                                row.push((
                                    output_index + out * plane + alpha * side + beta,
                                    1,
                                    Some(out * plane + alpha * side + beta),
                                ));
                            }
                        }
                    }
                    c.append_with_idx(&row);

                    auxiliary_index += in_channels;
                    output_index += out_channels * plane;
                }
                &Instruction::Lookup {
                    input,
                    tp: lookup_type,
                } => {
                    let tag = lookup_type.tag();
                    match lookup_type {
                        LookupType::Range(_) | LookupType::Bound(_) => {
                            // Emitted only as part of `Div`; as a standalone
                            // instruction they would constrain rather than
                            // compute, leaving the output cell unpinned.
                            return Err(CompileError::ReservedLookupType {
                                instr: instr_idx,
                                tag,
                            });
                        }
                        _ => {}
                    }
                    Self::check_no_parameter_read(instr_idx, input, self.weights.len())?;
                    Self::check_reads(instr_idx, input, output_index, aux_start)?;
                    Self::check_tag(instr_idx, tag, &tags)?;
                    d.append(input);
                    e.append(&vec![(output_index, 1)]);
                    tp.push(lookup_type.clone());
                    output_index += 1
                }
                &Instruction::Div {
                    input1,
                    input2,
                    divisor,
                } => {
                    let divisor = *divisor;
                    if divisor <= 0 {
                        return Err(CompileError::NonPositiveDivisor {
                            instr: instr_idx,
                            divisor,
                        });
                    }
                    Self::check_no_parameter_read(instr_idx, input1, self.weights.len())?;
                    Self::check_reads(instr_idx, input1, output_index, aux_start)?;
                    Self::check_reads(instr_idx, input2, output_index, aux_start)?;
                    Self::check_tag(instr_idx, LookupType::Range(divisor).tag(), &tags)?;
                    Self::check_tag(
                        instr_idx,
                        LookupType::Bound(QUOTIENT_BOUND).tag(),
                        &tags,
                    )?;
                    a.append(input1);
                    b.append(input2);
                    c.append(&vec![(output_index, divisor), (auxiliary_index, 1)]);
                    d.append(&vec![]);
                    e.append(&vec![(auxiliary_index, 1)]);
                    tp.push(LookupType::Range(divisor));

                    // The remainder check alone leaves `divisor` valid choices
                    // of quotient, so pin the quotient down too.
                    d.append(&vec![]);
                    e.append(&vec![(output_index, 1)]);
                    tp.push(LookupType::Bound(QUOTIENT_BOUND));

                    output_index += 1;
                    auxiliary_index += 1
                }
                &Instruction::MatMult {
                    m,
                    n,
                    k,
                    start1,
                    start2,
                } => {
                    let m = *m;
                    let n = *n;
                    let k = *k;
                    let start1 = *start1;
                    let start2 = *start2;
                    if m == 0 || n == 0 || k == 0 {
                        return Err(CompileError::ZeroDimension {
                            instr: instr_idx,
                        });
                    }
                    if start1 < self.weights.len() {
                        return Err(CompileError::ParameterReadOutsideB {
                            instr: instr_idx,
                            col: start1,
                        });
                    }
                    Self::check_block(instr_idx, start1, n * m, output_index, aux_start)?;
                    Self::check_block(instr_idx, start2, n * k, output_index, aux_start)?;

                    // Compute P^T * gamma_1
                    // Select n * m values of P^T
                    for i in 0..n {
                        let mut row = vec![];
                        for j in 0..m {
                            // Select element of P^T with multiplier of 1, and idx[j]
                            row.push((start1 + i * m + j, 1, Some(j * k)));
                        }
                        a.append_with_idx(&row);
                    }

                    // Compute Q * gamma_2
                    // Select n * k values of Q
                    for i in 0..n {
                        let mut row = vec![];
                        for j in 0..k {
                            row.push((start2 + i * k + j, 1, Some(j)));
                        }
                        b.append_with_idx(&row);
                    }

                    // Constraint v[i]s
                    for i in 0..n {
                        c.append(&vec![(auxiliary_index + i, 1)]);
                    }

                    // Constraint sum of v[i]s
                    let mut row = vec![];
                    for i in 0..n {
                        row.push((auxiliary_index + i, 1));
                    }
                    a.append(&row);
                    b.append(&vec![(0, 1)]);

                    let mut row = vec![];
                    for i in 0..m {
                        for j in 0..k {
                            row.push((output_index + i * k + j, 1, Some(i * k + j)));
                        }
                    }
                    c.append_with_idx(&row);

                    auxiliary_index += n;
                    output_index += m * k;
                }
            }
        }

        // The semantic trace has to end exactly where the advice region
        // begins; otherwise the two regions overlap and an instruction could
        // read prover-chosen advice through a trace index.
        if output_index != aux_start {
            return Err(CompileError::TraceLengthMismatch {
                trace_end: output_index,
                aux_start,
            });
        }

        let z_len = auxiliary_index;
        Ok(Circuit::<F>::new(
            a,
            b,
            c,
            d,
            e,
            tp,
            self.weights(),
            self.weights.len(),
            z_len,
            table,
        ))
    }

    /// Panicking wrapper around [`Program::compile`], kept for callers that
    /// build their programs statically.
    pub fn to_circuit(
        &self,
        input_size: usize,
        aux_start: usize,
        table: Vec<(F, F, F)>,
    ) -> Circuit<F> {
        self.compile(input_size, aux_start, table)
            .unwrap_or_else(|err| panic!("invalid program: {}", err))
    }

    fn check_reads(
        instr: usize,
        row: &Vec<(usize, i64)>,
        limit: usize,
        aux_start: usize,
    ) -> Result<(), CompileError> {
        for &(col, _) in row {
            Self::check_read(instr, col, limit, aux_start)?;
        }
        Ok(())
    }

    fn check_read(
        instr: usize,
        col: usize,
        limit: usize,
        aux_start: usize,
    ) -> Result<(), CompileError> {
        if col >= aux_start {
            return Err(CompileError::ReadsAdvice {
                instr,
                col,
                aux_start,
            });
        }
        if col >= limit {
            return Err(CompileError::ReadsFutureTrace { instr, col, limit });
        }
        Ok(())
    }

    /// Checks a contiguous operand block `[start, start + len)`.
    fn check_block(
        instr: usize,
        start: usize,
        len: usize,
        limit: usize,
        aux_start: usize,
    ) -> Result<(), CompileError> {
        if len == 0 {
            return Err(CompileError::ZeroDimension { instr });
        }
        Self::check_read(instr, start, limit, aux_start)?;
        Self::check_read(instr, start + len - 1, limit, aux_start)
    }

    /// Only matrix `B` keeps its parameter columns through preprocessing, so
    /// every other side has to read parameters indirectly.
    fn check_no_parameter_read(
        instr: usize,
        row: &Vec<(usize, i64)>,
        weight_len: usize,
    ) -> Result<(), CompileError> {
        for &(col, _) in row {
            if col < weight_len {
                return Err(CompileError::ParameterReadOutsideB { instr, col });
            }
        }
        Ok(())
    }

    fn check_tag(instr: usize, tag: i64, tags: &HashSet<F>) -> Result<(), CompileError> {
        if tags.contains(&F::from(tag)) {
            Ok(())
        } else {
            Err(CompileError::MissingLookupRows { instr, tag })
        }
    }

    /// Collects the lookup-type tags the public table covers.
    ///
    /// For every type a `Lookup` instruction reads a value from, the table has
    /// to send each input to at most one output; that uniqueness is what makes
    /// the instruction deterministic. The range-check types `Range` and
    /// `Bound` are exempt: they pin a value that already sits in the witness
    /// rather than producing one, so their rows deliberately repeat the same
    /// input with different outputs.
    fn table_tags(&self, table: &[(F, F, F)]) -> Result<HashSet<F>, CompileError> {
        let deterministic = self
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                // `Range` and `Bound` are rejected below as standalone
                // lookups, so their rows are never required to be a function.
                Instruction::Lookup {
                    tp: LookupType::Range(_) | LookupType::Bound(_),
                    ..
                } => None,
                Instruction::Lookup { tp, .. } => Some(F::from(tp.tag())),
                _ => None,
            })
            .collect::<HashSet<_>>();

        let mut outputs = HashMap::new();
        let mut tags = HashSet::new();
        for (row, &(i, j, k)) in table.iter().enumerate() {
            if deterministic.contains(&k) {
                if let Some(&seen) = outputs.get(&(i, k)) {
                    if seen != j {
                        return Err(CompileError::AmbiguousTable { row });
                    }
                } else {
                    outputs.insert((i, k), j);
                }
            }
            tags.insert(k);
        }
        Ok(tags)
    }

    pub fn instructions(&self) -> &[Instruction] {
        &self.instructions
    }

    pub fn weight_len(&self) -> usize {
        self.weights.len()
    }

    pub fn weights(&self) -> Vec<F> {
        self.weights.iter().map(|&x| F::from(x)).collect()
    }

    pub fn execute_i64(&self, mut input: Vec<i64>) -> Vec<i64> {
        // execute the prgram, generate the whole traces
        let mut z = self.weights.clone();
        z.append(&mut input);

        for instr in &self.instructions {
            match &instr {
                &Instruction::AddMult { input1, input2 } => {
                    let mut a = 0;
                    for i in input1 {
                        a += z[i.0] * i.1
                    }
                    let mut b = 0;
                    for i in input2 {
                        b += z[i.0] * i.1
                    }
                    z.push(a * b);
                }
                &Instruction::Conv {
                    n,
                    m,
                    in_channels,
                    out_channels,
                    start1,
                    start2,
                } => {
                    let n = *n;
                    let m = *m;
                    let in_channels = *in_channels;
                    let out_channels = *out_channels;
                    let start1 = *start1;
                    let start2 = *start2;
                    let side = n + m - 1;
                    let input_plane = n * n;
                    let kernel_plane = m * m;

                    for out in 0..out_channels {
                        for alpha in 0..side {
                            for beta in 0..side {
                                let mut total = 0;
                                for ch in 0..in_channels {
                                    for u in 0..m {
                                        for v in 0..m {
                                            let i = alpha as isize - (m as isize - 1) + u as isize;
                                            let j = beta as isize - (m as isize - 1) + v as isize;
                                            if i >= 0 && i < n as isize && j >= 0 && j < n as isize
                                            {
                                                let input_idx = start1
                                                    + ch * input_plane
                                                    + i as usize * n
                                                    + j as usize;
                                                let weight_idx = start2
                                                    + (ch * out_channels + out) * kernel_plane
                                                    + u * m
                                                    + v;
                                                total += z[input_idx] * z[weight_idx];
                                            }
                                        }
                                    }
                                }
                                z.push(total);
                            }
                        }
                    }
                }
                &Instruction::Lookup { input, tp } => {
                    let mut a = 0;
                    for i in input {
                        a += z[i.0] * i.1
                    }
                    if let Some(v) = tp.eval(a) {
                        z.push(v);
                    } else {
                        match tp {
                            // Emitted only by `Div`, never as a standalone Lookup.
                            LookupType::Range(_) | LookupType::Bound(_) => panic!(),
                            _ => {
                                let table = self
                                    .lookup_tables
                                    .get(tp)
                                    .unwrap_or_else(|| panic!("missing lookup table for {:?}", tp));
                                z.push(table.lookup(a));
                            }
                        }
                    }
                }
                &Instruction::Div {
                    input1,
                    input2,
                    divisor,
                } => {
                    let divisor = *divisor;
                    assert!(divisor > 0);
                    let mut a = 0;
                    for i in input1 {
                        a += z[i.0] * i.1
                    }
                    let mut b = 0;
                    for i in input2 {
                        b += z[i.0] * i.1
                    }
                    let c = a * b;
                    let q = c.div_euclid(divisor);
                    assert!(
                        (-QUOTIENT_BOUND..QUOTIENT_BOUND).contains(&q),
                        "Div quotient {} escapes the range check [{}, {})",
                        q,
                        -QUOTIENT_BOUND,
                        QUOTIENT_BOUND
                    );
                    z.push(q);
                }
                &Instruction::MatMult {
                    m,
                    n,
                    k,
                    start1,
                    start2,
                } => {
                    let m = *m;
                    let n = *n;
                    let k = *k;
                    let start1 = *start1;
                    let start2 = *start2;
                    let mut mat_1 = vec![];
                    for i in 0..m {
                        let mut row = vec![];
                        for j in 0..n {
                            row.push(z[start1 + j * m + i]);
                        }
                        mat_1.push(row);
                    }
                    let mut mat_2 = vec![];
                    for i in 0..k {
                        let mut col = vec![];
                        for j in 0..n {
                            col.push(z[start2 + j * k + i]);
                        }
                        mat_2.push(col);
                    }
                    for i in 0..m {
                        for j in 0..k {
                            let mut v = 0;
                            for l in 0..n {
                                v += mat_1[i][l] * mat_2[j][l]
                            }
                            z.push(v);
                        }
                    }
                }
            }
        }

        z
    }

    pub fn execute(&self, input: Vec<i64>) -> Vec<F> {
        self.execute_i64(input)
            .iter()
            .map(|&x| F::from(x))
            .collect::<Vec<_>>()
    }

    pub fn gen_z(&self, output_start: usize, trace: Vec<F>, gamma: F) -> Vec<F> {
        let mut z = trace.clone();
        let mut aux = vec![];

        let mut output_index = output_start;
        for instr in &self.instructions {
            match &instr {
                &Instruction::AddMult {
                    input1: _,
                    input2: _,
                } => {
                    output_index += 1;
                }
                &Instruction::Conv {
                    n,
                    m,
                    in_channels,
                    out_channels,
                    start1,
                    start2,
                } => {
                    let n = *n;
                    let m = *m;
                    let in_channels = *in_channels;
                    let out_channels = *out_channels;
                    let start1 = *start1;
                    let start2 = *start2;
                    let side = n + m - 1;
                    let plane = side * side;
                    let input_plane = n * n;
                    let kernel_plane = m * m;

                    for ch in 0..in_channels {
                        let mut a = F::zero();
                        for i in 0..n {
                            for j in 0..n {
                                let exp = (i * side + j) as u64;
                                a += trace[start1 + ch * input_plane + i * n + j]
                                    * gamma.pow(vec![exp]);
                            }
                        }

                        let mut b = F::zero();
                        for out in 0..out_channels {
                            for u in 0..m {
                                for v in 0..m {
                                    let exp =
                                        (out * plane + (m - 1 - u) * side + (m - 1 - v)) as u64;
                                    b += trace[start2
                                        + (ch * out_channels + out) * kernel_plane
                                        + u * m
                                        + v]
                                        * gamma.pow(vec![exp]);
                                }
                            }
                        }

                        aux.push(a * b);
                    }
                    output_index += out_channels * plane;
                }
                &Instruction::Lookup { input: _, tp: _ } => {
                    output_index += 1;
                }
                &Instruction::Div {
                    input1,
                    input2,
                    divisor,
                } => {
                    let divisor = *divisor;
                    assert!(divisor > 0);
                    // add the remnant r st <A[cr],z> * <B[cr],z> = divisor * z[or] + r
                    let mut a = F::zero();
                    for i in input1 {
                        a += trace[i.0] * F::from(i.1);
                    }
                    let mut b = F::zero();
                    for i in input2 {
                        b += trace[i.0] * F::from(i.1);
                    }
                    let c = a * b;
                    let r = c - F::from(divisor) * trace[output_index];

                    aux.push(r);
                    output_index += 1;
                }
                &Instruction::MatMult {
                    m,
                    n,
                    k,
                    start1,
                    start2,
                } => {
                    // add v[i]s
                    let m = *m;
                    let n = *n;
                    let k = *k;
                    let start1 = *start1;
                    let start2 = *start2;

                    for i in 0..n {
                        let mut a = F::zero();
                        for j in 0..m {
                            let exp = (j * k) as u64;
                            a += trace[start1 + i * m + j] * gamma.pow(vec![exp]);
                        }

                        let mut b = F::zero();
                        for j in 0..k {
                            let exp = j as u64;
                            b += trace[start2 + i * k + j] * gamma.pow(vec![exp]);
                        }

                        aux.push(a * b);
                    }
                    output_index += m * k;
                }
            }
        }

        z.append(&mut aux);
        z
    }
}

#[cfg(test)]
mod tests {
    use std::cmp;

    use ark_bn254::{Bn254, Fr};
    use ark_ff::UniformRand;
    use rand::{rngs::StdRng, SeedableRng};
    use util::{kzg::Mkzg, util::RandomOracle};

    use super::Program;
    use crate::{
        circuit::LookupType, instruction::Instruction, preprocess::Preprocessor, prover::Prover,
        verifier::Verifier,
    };

    fn full_cross_correlation(
        input: &[i64],
        weights: &[i64],
        n: usize,
        m: usize,
        in_channels: usize,
        out_channels: usize,
    ) -> Vec<i64> {
        let side = n + m - 1;
        let input_plane = n * n;
        let kernel_plane = m * m;
        let mut out = vec![0; out_channels * side * side];

        for d in 0..out_channels {
            for alpha in 0..side {
                for beta in 0..side {
                    let mut total = 0;
                    for c in 0..in_channels {
                        for u in 0..m {
                            for v in 0..m {
                                let i = alpha as isize - (m as isize - 1) + u as isize;
                                let j = beta as isize - (m as isize - 1) + v as isize;
                                if i >= 0 && i < n as isize && j >= 0 && j < n as isize {
                                    let x = input[c * input_plane + i as usize * n + j as usize];
                                    let w =
                                        weights[(c * out_channels + d) * kernel_plane + u * m + v];
                                    total += x * w;
                                }
                            }
                        }
                    }
                    out[d * side * side + alpha * side + beta] = total;
                }
            }
        }

        out
    }

    #[test]
    fn div_instruction_matches_floor_division() {
        let weights = vec![1];
        let input = vec![-65, 65, 8191, 8192, 383, 384];
        let instructions = vec![
            Instruction::Div {
                input1: vec![(weights.len(), 1)],
                input2: vec![(0, 1)],
                divisor: 64,
            },
            Instruction::Div {
                input1: vec![(weights.len() + 2, 1)],
                input2: vec![(0, 1)],
                divisor: 4096,
            },
            Instruction::Div {
                input1: vec![(weights.len() + 4, 1)],
                input2: vec![(0, 1)],
                divisor: 192,
            },
        ];
        let program = Program::<Fr>::new(instructions, weights.clone());
        let trace = program.execute(input.clone());
        let out = weights.len() + input.len();
        assert_eq!(trace[out], Fr::from((-65i64).div_euclid(64)));
        assert_eq!(trace[out + 1], Fr::from(8191i64.div_euclid(4096)));
        assert_eq!(trace[out + 2], Fr::from(383i64.div_euclid(192)));
    }

    #[test]
    fn div_relu_lookup_fuses_div_and_relu() {
        let weights = vec![1];
        // Negative inputs must clamp to 0 regardless of how the division
        // rounds, which is what lets DivRelu replace a Div followed by a Relu.
        let input = vec![-65, -1, 0, 63, 64, 8191];
        let instructions: Vec<Instruction> = (0..input.len())
            .map(|i| Instruction::Lookup {
                input: vec![(weights.len() + i, 1)],
                tp: LookupType::DivRelu(64),
            })
            .collect();
        let program = Program::<Fr>::new(instructions, weights.clone());
        let trace = program.execute_i64(input.clone());
        let out = weights.len() + input.len();
        assert_eq!(&trace[out..], &[0, 0, 0, 0, 1, 127]);

        // Scaling the input by the divisor turns the same table into a plain
        // ReLU, which is how max-pooling reuses it.
        let scaled: Vec<Instruction> = (0..input.len())
            .map(|i| Instruction::Lookup {
                input: vec![(weights.len() + i, 64)],
                tp: LookupType::DivRelu(64),
            })
            .collect();
        let program = Program::<Fr>::new(scaled, weights.clone());
        let trace = program.execute_i64(input.clone());
        assert_eq!(&trace[out..], &[0, 0, 0, 63, 64, 8191]);
    }

    #[test]
    fn conv_instruction_matches_direct_evaluation() {
        let n = 2;
        let m = 2;
        let in_channels = 2;
        let out_channels = 2;

        let raw_weights = vec![1, 2, 3, 4, 5, 6, 7, 8, 1, -1, 2, -2, 3, -3, 4, -4];
        let input = vec![1, 2, 3, 4, -1, 0, 2, 1];

        let mut weights = vec![1];
        weights.extend(raw_weights.iter().copied());

        let instruction = Instruction::Conv {
            n,
            m,
            in_channels,
            out_channels,
            start1: weights.len(),
            start2: 1,
        };
        let program = Program::<Fr>::new(vec![instruction], weights.clone());

        let trace = program.execute(input.clone());
        let output_start = weights.len() + input.len();
        let expected =
            full_cross_correlation(&input, &raw_weights, n, m, in_channels, out_channels);
        let expected_fr = expected.into_iter().map(Fr::from).collect::<Vec<_>>();
        assert_eq!(
            &trace[output_start..output_start + expected_fr.len()],
            expected_fr.as_slice()
        );

        let aux_start = trace.len();
        let gamma = Fr::from(7u64);
        let z = program.gen_z(output_start, trace.clone(), gamma);
        let circuit = program.to_circuit(input.len(), aux_start, vec![]);
        assert!(circuit.check(z, gamma));
    }

    #[test]
    fn conv_instruction_proves_and_verifies() {
        let n = 2;
        let m = 2;
        let in_channels = 2;
        let out_channels = 2;

        let raw_weights = vec![1, 2, 3, 4, 5, 6, 7, 8, 1, -1, 2, -2, 3, -3, 4, -4];
        let input = vec![1, 2, 3, 4, -1, 0, 2, 1];

        let mut weights = vec![1];
        weights.extend(raw_weights.iter().copied());

        let conv_output_start = weights.len() + input.len();
        let instructions = vec![
            Instruction::Conv {
                n,
                m,
                in_channels,
                out_channels,
                start1: weights.len(),
                start2: 1,
            },
            Instruction::Lookup {
                input: vec![(conv_output_start, 1)],
                tp: LookupType::Relu,
            },
            Instruction::Lookup {
                input: vec![(conv_output_start + 1, 1)],
                tp: LookupType::Relu,
            },
        ];
        let program = Program::<Fr>::new(instructions, weights);

        let trace = program.execute(input.clone());
        let aux_start = trace.len();
        let mut rng = StdRng::seed_from_u64(7);
        let gamma = Fr::rand(&mut rng);
        let z = program.gen_z(conv_output_start, trace.clone(), gamma);

        let table = (-64i64..=64)
            .map(|i| (Fr::from(i), Fr::from(cmp::max(0, i)), Fr::from(2u64)))
            .collect::<Vec<_>>();
        let circuit = program.to_circuit(input.len(), aux_start, table);
        assert!(circuit.check(z.clone(), gamma));

        let (kzg_pp, kzg_vp) = Mkzg::<Bn254>::gen_srs(5, &mut rng);
        let (pk, vk) = Preprocessor::build(kzg_pp, kzg_vp, circuit);
        let prover = Prover::new(pk);
        let mut ro = RandomOracle::new(&mut rng);
        let proof = prover.prove(z, gamma, &mut ro);
        let verifier = Verifier::new(vk);
        verifier.verify(proof, gamma, &mut ro);
    }
}
