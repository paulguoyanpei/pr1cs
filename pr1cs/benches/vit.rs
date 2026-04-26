use ark_bn254::{Bn254, Fr};
use ark_ff::UniformRand;
use pr1cs::preprocess::Preprocessor;
use pr1cs::prover::Prover;
use pr1cs::verifier::Verifier;
use pr1cs::{
    circuit::LookupType,
    instruction::Instruction,
    program::{LookupTable, Program},
};
use rand::thread_rng;
use std::cmp;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use util::kzg::{Mkzg, LOG_CHUNK_SIZE};
use util::util::RandomOracle;

const Q: i64 = 12;
const SCALE: i64 = 1 << Q;
const N_PATCH: usize = 196;
const N: usize = 197;
const C: usize = 192;
const HEADS: usize = 3;
const D: usize = 64;
const MLP: usize = 768;
const BLOCKS: usize = 12;
const CLASSES: usize = 10;
const EPS_INT: i64 = 0;
const RELU_MIN: i64 = -(1 << 20);
const RELU_MAX: i64 = 1 << 20;
const GELU_STATIC_MIN: i64 = -(1 << 20);
const GELU_STATIC_MAX: i64 = 1 << 20;
const EXP_STATIC_MIN: i64 = -(1 << 20);

#[derive(Clone)]
struct Linear {
    w: usize,
    b: usize,
}

#[derive(Clone)]
struct Block {
    n1_g: usize,
    n1_b: usize,
    qkv: Linear,
    proj: Linear,
    n2_g: usize,
    n2_b: usize,
    fc1: Linear,
    fc2: Linear,
}

struct Weights {
    buf: Vec<i64>,
    cls: usize,
    pos: usize,
    blocks: Vec<Block>,
    norm_g: usize,
    norm_b: usize,
    head: Linear,
}

struct Builder {
    instructions: Vec<Instruction>,
    next: usize,
    marks: Vec<(String, usize, usize)>,
}

impl Builder {
    fn new(weight_len: usize, input_len: usize) -> Self {
        Self {
            instructions: Vec::new(),
            next: weight_len + input_len,
            marks: Vec::new(),
        }
    }

    fn mark(&mut self, name: impl Into<String>, start: usize, len: usize) {
        self.marks.push((name.into(), start, len));
    }

    fn add(&mut self, input1: Vec<(usize, i64)>) -> usize {
        let out = self.next;
        self.instructions.push(Instruction::AddMult {
            input1,
            input2: vec![(0, 1)],
        });
        self.next += 1;
        out
    }

    fn div(&mut self, input1: Vec<(usize, i64)>, input2: Vec<(usize, i64)>, divisor: i64) -> usize {
        let out = self.next;
        self.instructions.push(Instruction::Div {
            input1,
            input2,
            divisor,
        });
        self.next += 1;
        out
    }

    fn lookup(&mut self, input: Vec<(usize, i64)>, tp: LookupType) -> usize {
        let out = self.next;
        self.instructions.push(Instruction::Lookup { input, tp });
        self.next += 1;
        out
    }

    fn matmult(&mut self, m: usize, n: usize, k: usize, start1: usize, start2: usize) -> usize {
        let out = self.next;
        self.instructions.push(Instruction::MatMult {
            m,
            n,
            k,
            start1,
            start2,
        });
        self.next += m * k;
        out
    }

    fn linear(
        &mut self,
        input: usize,
        tokens: usize,
        in_dim: usize,
        out_dim: usize,
        lin: &Linear,
    ) -> usize {
        let raw = self.matmult(tokens, in_dim, out_dim, input, lin.w);
        let out = self.next;
        for f in 0..out_dim {
            for t in 0..tokens {
                self.div(
                    vec![(raw + t * out_dim + f, 1), (lin.b + f, SCALE)],
                    vec![(0, 1)],
                    SCALE,
                );
            }
        }
        out
    }

    fn qkv_linear(&mut self, input: usize, lin: &Linear) -> usize {
        let raw = self.matmult(N, C, 3 * C, input, lin.w);
        let out = self.next;

        for h in 0..HEADS {
            for d in 0..D {
                let f = h * D + d;
                for t in 0..N {
                    self.div(
                        vec![(raw + t * 3 * C + f, 1), (lin.b + f, SCALE)],
                        vec![(0, 1)],
                        SCALE,
                    );
                }
            }
        }

        for h in 0..HEADS {
            for d in 0..D {
                let f = C + h * D + d;
                for t in 0..N {
                    self.div(
                        vec![(raw + t * 3 * C + f, 1), (lin.b + f, SCALE)],
                        vec![(0, 1)],
                        SCALE,
                    );
                }
            }
        }

        for h in 0..HEADS {
            for t in 0..N {
                for d in 0..D {
                    let f = 2 * C + h * D + d;
                    self.div(
                        vec![(raw + t * 3 * C + f, 1), (lin.b + f, SCALE)],
                        vec![(0, 1)],
                        SCALE,
                    );
                }
            }
        }

        out
    }

    fn layer_norm_expr<F>(&mut self, tokens: usize, gamma: usize, beta: usize, mut elem: F) -> usize
    where
        F: FnMut(usize, usize) -> Vec<(usize, i64)>,
    {
        let mean = self.next;
        for t in 0..tokens {
            let terms = (0..C).flat_map(|f| elem(f, t)).collect();
            self.div(terms, vec![(0, 1)], C as i64);
        }

        let squares = self.next;
        for f in 0..C {
            for t in 0..tokens {
                let mut centered = elem(f, t);
                centered.push((mean + t, -1));
                self.instructions.push(Instruction::AddMult {
                    input1: centered.clone(),
                    input2: centered,
                });
                self.next += 1;
            }
        }

        let var_2q = self.next;
        for t in 0..tokens {
            let terms = (0..C).map(|f| (squares + f * tokens + t, 1)).collect();
            self.div(terms, vec![(0, 1)], C as i64);
        }

        let var_q = self.next;
        for t in 0..tokens {
            self.div(vec![(var_2q + t, 1)], vec![(0, 1)], SCALE);
        }

        let rsqrt = self.next;
        for t in 0..tokens {
            self.lookup(vec![(var_q + t, 1), (0, EPS_INT)], LookupType::Rsqrt);
        }

        let normed = self.next;
        for f in 0..C {
            for t in 0..tokens {
                let mut centered = elem(f, t);
                centered.push((mean + t, -1));
                self.div(centered, vec![(rsqrt + t, 1)], SCALE);
            }
        }

        let scaled = self.next;
        for f in 0..C {
            for t in 0..tokens {
                self.div(
                    vec![(normed + f * tokens + t, 1)],
                    vec![(gamma + f, 1)],
                    SCALE,
                );
            }
        }

        let out = self.next;
        for f in 0..C {
            for t in 0..tokens {
                self.add(vec![(scaled + f * tokens + t, 1), (beta + f, 1)]);
            }
        }
        out
    }

    fn max_rows(&mut self, scores: usize) -> Vec<usize> {
        let mut maxes = Vec::with_capacity(HEADS * N);
        for h in 0..HEADS {
            for q in 0..N {
                let base = scores + h * N * N + q * N;
                let mut cur = base;
                for k in 1..N {
                    let r = self.lookup(vec![(base + k, 1), (cur, -1)], LookupType::Relu);
                    cur = self.add(vec![(cur, 1), (r, 1)]);
                }
                maxes.push(cur);
            }
        }
        maxes
    }

    fn attention(&mut self, x: usize, block: &Block, debug: bool) -> usize {
        let qkv = self.qkv_linear(x, &block.qkv);
        if debug {
            self.mark("b0_qkv", qkv, 3 * C * N);
        }
        let scale_int = ((1f64 / (D as f64).sqrt()) * SCALE as f64).round() as i64;

        let mut qk_raw = Vec::with_capacity(HEADS);
        for h in 0..HEADS {
            let q = qkv + h * D * N;
            let k = qkv + (C + h * D) * N;
            qk_raw.push(self.matmult(N, D, N, q, k));
        }
        let scaled_2q = self.next;
        for &raw in &qk_raw {
            for qidx in 0..N {
                for kidx in 0..N {
                    self.div(
                        vec![(raw + qidx * N + kidx, 1)],
                        vec![(0, scale_int)],
                        SCALE,
                    );
                }
            }
        }
        let scores = self.next;
        for i in 0..HEADS * N * N {
            self.div(vec![(scaled_2q + i, 1)], vec![(0, 1)], SCALE);
        }
        if debug {
            self.mark("b0_scores", scores, HEADS * N * N);
        }

        let maxes = self.max_rows(scores);
        let exp = self.next;
        for h in 0..HEADS {
            for qidx in 0..N {
                let row_max = maxes[h * N + qidx];
                for kidx in 0..N {
                    self.lookup(
                        vec![(scores + h * N * N + qidx * N + kidx, 1), (row_max, -1)],
                        LookupType::Exp,
                    );
                }
            }
        }

        let recip = self.next;
        for h in 0..HEADS {
            for qidx in 0..N {
                let terms = (0..N)
                    .map(|kidx| (exp + h * N * N + qidx * N + kidx, 1))
                    .collect();
                self.lookup(terms, LookupType::Recip);
            }
        }

        let softmax_col_major = self.next;
        for h in 0..HEADS {
            for kidx in 0..N {
                for qidx in 0..N {
                    self.div(
                        vec![(exp + h * N * N + qidx * N + kidx, 1)],
                        vec![(recip + h * N + qidx, 1)],
                        SCALE,
                    );
                }
            }
        }

        let mut ctx_raw = Vec::with_capacity(HEADS);
        for h in 0..HEADS {
            ctx_raw.push(self.matmult(
                N,
                N,
                D,
                softmax_col_major + h * N * N,
                qkv + 2 * C * N + h * N * D,
            ));
        }
        let attn_out = self.next;
        for &raw in &ctx_raw {
            for d in 0..D {
                for t in 0..N {
                    self.div(vec![(raw + t * D + d, 1)], vec![(0, 1)], SCALE);
                }
            }
        }

        self.linear(attn_out, N, C, C, &block.proj)
    }

    fn block_expr<F>(&mut self, mut x_elem: F, block: &Block, debug: bool) -> usize
    where
        F: FnMut(usize, usize) -> Vec<(usize, i64)>,
    {
        let n1 = self.layer_norm_expr(N, block.n1_g, block.n1_b, |f, t| x_elem(f, t));
        if debug {
            self.mark("b0_n1", n1, C * N);
        }
        let attn = self.attention(n1, block, debug);
        if debug {
            self.mark("b0_attn", attn, C * N);
        }

        let n2 = self.layer_norm_expr(N, block.n2_g, block.n2_b, |f, t| {
            let mut terms = x_elem(f, t);
            terms.push((attn + f * N + t, 1));
            terms
        });
        let fc1 = self.linear(n2, N, C, MLP, &block.fc1);
        if debug {
            self.mark("b0_fc1", fc1, MLP * N);
        }
        let gelu = self.next;
        for f in 0..MLP {
            for t in 0..N {
                self.lookup(vec![(fc1 + f * N + t, 1)], LookupType::Gelu);
            }
        }
        let fc2 = self.linear(gelu, N, MLP, C, &block.fc2);
        if debug {
            self.mark("b0_fc2", fc2, C * N);
        }
        let out = self.next;
        for f in 0..C {
            for t in 0..N {
                let mut terms = x_elem(f, t);
                terms.push((attn + f * N + t, 1));
                terms.push((fc2 + f * N + t, 1));
                self.add(terms);
            }
        }
        out
    }

    fn block(&mut self, x: usize, block: &Block, debug: bool) -> usize {
        self.block_expr(|f, t| vec![(x + f * N + t, 1)], block, debug)
    }

    fn final_norm_cls(&mut self, x: usize, gamma: usize, beta: usize) -> usize {
        let mean = self.div(
            (0..C).map(|f| (x + f * N, 1)).collect(),
            vec![(0, 1)],
            C as i64,
        );
        let squares = self.next;
        for f in 0..C {
            let centered = vec![(x + f * N, 1), (mean, -1)];
            self.instructions.push(Instruction::AddMult {
                input1: centered.clone(),
                input2: centered,
            });
            self.next += 1;
        }
        let var_2q = self.div(
            (0..C).map(|f| (squares + f, 1)).collect(),
            vec![(0, 1)],
            C as i64,
        );
        let var_q = self.div(vec![(var_2q, 1)], vec![(0, 1)], SCALE);
        let rsqrt = self.lookup(vec![(var_q, 1), (0, EPS_INT)], LookupType::Rsqrt);

        let normed = self.next;
        for f in 0..C {
            self.div(vec![(x + f * N, 1), (mean, -1)], vec![(rsqrt, 1)], SCALE);
        }

        let scaled = self.next;
        for f in 0..C {
            self.div(vec![(normed + f, 1)], vec![(gamma + f, 1)], SCALE);
        }

        let out = self.next;
        for f in 0..C {
            self.add(vec![(scaled + f, 1), (beta + f, 1)]);
        }
        out
    }
}

fn data_path(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push(name);
    p
}

fn read_i64_bin(path: &Path) -> Vec<i64> {
    let mut buf = Vec::new();
    File::open(path)
        .unwrap_or_else(|e| panic!("open {}: {}", path.display(), e))
        .read_to_end(&mut buf)
        .unwrap();
    assert_eq!(
        buf.len() % 8,
        0,
        "{} byte length must be divisible by 8",
        path.display()
    );
    buf.chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn write_i64_bin(path: &Path, xs: &[i64]) {
    let mut file =
        File::create(path).unwrap_or_else(|e| panic!("create {}: {}", path.display(), e));
    for &x in xs {
        file.write_all(&x.to_le_bytes()).unwrap();
    }
}

fn take(buf: &mut Vec<i64>, raw: &[i64], off: &mut usize, len: usize) -> usize {
    let start = buf.len();
    buf.extend_from_slice(&raw[*off..*off + len]);
    *off += len;
    start
}

fn take_linear(
    buf: &mut Vec<i64>,
    raw: &[i64],
    off: &mut usize,
    in_dim: usize,
    out_dim: usize,
) -> Linear {
    let w = take(buf, raw, off, in_dim * out_dim);
    let b = take(buf, raw, off, out_dim);
    Linear { w, b }
}

fn load_weights() -> Weights {
    let raw = read_i64_bin(&data_path("vit_params.bin"));
    let mut buf = vec![1];
    let mut off = 0usize;
    let cls = take(&mut buf, &raw, &mut off, C);
    let pos = take(&mut buf, &raw, &mut off, N * C);
    let mut blocks = Vec::with_capacity(BLOCKS);
    for _ in 0..BLOCKS {
        let n1_g = take(&mut buf, &raw, &mut off, C);
        let n1_b = take(&mut buf, &raw, &mut off, C);
        let qkv = take_linear(&mut buf, &raw, &mut off, C, 3 * C);
        let proj = take_linear(&mut buf, &raw, &mut off, C, C);
        let n2_g = take(&mut buf, &raw, &mut off, C);
        let n2_b = take(&mut buf, &raw, &mut off, C);
        let fc1 = take_linear(&mut buf, &raw, &mut off, C, MLP);
        let fc2 = take_linear(&mut buf, &raw, &mut off, MLP, C);
        blocks.push(Block {
            n1_g,
            n1_b,
            qkv,
            proj,
            n2_g,
            n2_b,
            fc1,
            fc2,
        });
    }
    let norm_g = take(&mut buf, &raw, &mut off, C);
    let norm_b = take(&mut buf, &raw, &mut off, C);
    let head = take_linear(&mut buf, &raw, &mut off, C, CLASSES);
    assert_eq!(off, raw.len());
    Weights {
        buf,
        cls,
        pos,
        blocks,
        norm_g,
        norm_b,
        head,
    }
}

fn lookup_table(name: &str, min_int: i64, tp: LookupType) -> (LookupType, LookupTable) {
    let values = read_i64_bin(&data_path(name));
    (tp, LookupTable::new(min_int, values))
}

fn load_lookup_tables() -> HashMap<LookupType, LookupTable> {
    [
        lookup_table("vit_lut_gelu.bin", -8 * SCALE, LookupType::Gelu),
        lookup_table("vit_lut_exp.bin", -16 * SCALE, LookupType::Exp),
        lookup_table("vit_lut_recip.bin", 41, LookupType::Recip),
        lookup_table("vit_lut_rsqrt.bin", 4, LookupType::Rsqrt),
    ]
    .into_iter()
    .collect()
}

fn build_program(
    weights: &Weights,
    input_len: usize,
) -> (Vec<Instruction>, usize, usize, Vec<(String, usize, usize)>) {
    let mut b = Builder::new(weights.buf.len(), input_len);

    let input_start = weights.buf.len();
    let mut cur = 0;
    for (idx, block) in weights.blocks.iter().enumerate() {
        cur = if idx == 0 {
            b.block_expr(
                |f, t| {
                    if t == 0 {
                        vec![(weights.cls + f, 1), (weights.pos + f, 1)]
                    } else {
                        vec![
                            (input_start + (t - 1) * C + f, 1),
                            (weights.pos + t * C + f, 1),
                        ]
                    }
                },
                block,
                true,
            )
        } else {
            b.block(cur, block, false)
        };
        b.mark(format!("b{}_out", idx), cur, C * N);
    }
    let cls = b.final_norm_cls(cur, weights.norm_g, weights.norm_b);
    b.mark("final_cls", cls, C);
    let raw = b.matmult(1, C, CLASSES, cls, weights.head.w);
    b.mark("head_raw", raw, CLASSES);
    let logits = b.next;
    for j in 0..CLASSES {
        b.div(
            vec![(raw + j, 1), (weights.head.b + j, SCALE)],
            vec![(0, 1)],
            SCALE,
        );
    }
    b.mark("logits", logits, CLASSES);

    (b.instructions, logits, b.next, b.marks)
}

fn push_range_table(rows: &mut Vec<(Fr, Fr, Fr)>, divisor: i64) {
    for r in 0..divisor {
        rows.push((
            Fr::from(0),
            Fr::from(r),
            Fr::from(LookupType::Range(divisor).tag()),
        ));
    }
}

fn push_relu_table(rows: &mut Vec<(Fr, Fr, Fr)>) {
    for i in RELU_MIN..=RELU_MAX {
        rows.push((
            Fr::from(i),
            Fr::from(cmp::max(0, i)),
            Fr::from(LookupType::Relu.tag()),
        ));
    }
}

fn push_lut_table(rows: &mut Vec<(Fr, Fr, Fr)>, name: &str, min_int: i64, tp: LookupType) {
    let values = read_i64_bin(&data_path(name));
    for (offset, output) in values.into_iter().enumerate() {
        rows.push((
            Fr::from(min_int + offset as i64),
            Fr::from(output),
            Fr::from(tp.tag()),
        ));
    }
}

fn push_clamped_lut_table(
    rows: &mut Vec<(Fr, Fr, Fr)>,
    name: &str,
    lut_min_int: i64,
    static_min_int: i64,
    static_max_int: i64,
    tp: LookupType,
) {
    let values = read_i64_bin(&data_path(name));
    let lut_max_int = lut_min_int + values.len() as i64 - 1;
    for input in static_min_int..=static_max_int {
        let clamped = input.clamp(lut_min_int, lut_max_int);
        let output = values[(clamped - lut_min_int) as usize];
        rows.push((Fr::from(input), Fr::from(output), Fr::from(tp.tag())));
    }
}

fn build_lookup_table() -> Vec<(Fr, Fr, Fr)> {
    let mut rows = Vec::new();
    push_range_table(&mut rows, C as i64);
    push_range_table(&mut rows, SCALE);
    push_relu_table(&mut rows);
    push_clamped_lut_table(
        &mut rows,
        "vit_lut_gelu.bin",
        -8 * SCALE,
        GELU_STATIC_MIN,
        GELU_STATIC_MAX,
        LookupType::Gelu,
    );
    push_clamped_lut_table(
        &mut rows,
        "vit_lut_exp.bin",
        -16 * SCALE,
        EXP_STATIC_MIN,
        0,
        LookupType::Exp,
    );
    push_lut_table(&mut rows, "vit_lut_recip.bin", 41, LookupType::Recip);
    push_lut_table(&mut rows, "vit_lut_rsqrt.bin", 4, LookupType::Rsqrt);
    rows
}

fn eval_terms(trace: &[i64], terms: &[(usize, i64)]) -> i64 {
    terms.iter().map(|&(idx, coeff)| trace[idx] * coeff).sum()
}

fn check_lookup_coverage(
    table: &[(Fr, Fr, Fr)],
    instructions: &[Instruction],
    trace: &[i64],
    weight_len: usize,
    input_len: usize,
) {
    let set = table.iter().copied().collect::<HashSet<_>>();
    let mut output_index = weight_len + input_len;
    for (idx, instr) in instructions.iter().enumerate() {
        if let Instruction::Lookup { input, tp } = instr {
            let lookup_input = eval_terms(trace, input);
            let lookup_output = trace[output_index];
            let row = (
                Fr::from(lookup_input),
                Fr::from(lookup_output),
                Fr::from(tp.tag()),
            );
            if !set.contains(&row) {
                panic!(
                    "lookup table missing at instruction {} output {} type {:?}: input={} output={} tag={}",
                    idx,
                    output_index,
                    tp,
                    lookup_input,
                    lookup_output,
                    tp.tag()
                );
            }
        }
        output_index += instr.output_len();
    }
}

fn main() {
    let weights = load_weights();
    let input = read_i64_bin(&data_path("vit_input_embedded.bin"));
    assert_eq!(input.len(), N_PATCH * C);

    let start = Instant::now();
    let (instructions, logits_start, trace_len, marks) = build_program(&weights, input.len());
    let weight_len = weights.buf.len();
    let input_len = input.len();
    println!(
        "vit instructions: {}, weights: {}, expected trace: {}, build: {:?}",
        instructions.len(),
        weight_len,
        trace_len,
        start.elapsed()
    );

    let program = Program::<Fr>::new_with_lookup_tables(
        instructions.clone(),
        weights.buf,
        load_lookup_tables(),
    );
    let trace = program.execute_i64(input);
    assert_eq!(trace.len(), trace_len);

    let table = build_lookup_table();

    let trace_fr = trace.iter().map(|&x| Fr::from(x)).collect::<Vec<_>>();
    let aux_start = trace_fr.len();
    let mut rng = thread_rng();
    let gamma = <Fr as UniformRand>::rand(&mut rng);
    let z = program.gen_z(weight_len + input_len, trace_fr, gamma);

    let circuit_start = Instant::now();
    let circuit = program.to_circuit(input_len, aux_start, table);
    circuit.check(z.clone(), gamma);
    println!(
        "circuit.check ok, build/check: {:?}",
        circuit_start.elapsed()
    );

    let srs_start = Instant::now();
    let (kzg_pp, kzg_vp) = Mkzg::<Bn254>::gen_srs(LOG_CHUNK_SIZE, &mut rng);
    println!("srs_time = {} ms", srs_start.elapsed().as_millis());

    let preprocess_start = Instant::now();
    let (pk, vk) = Preprocessor::build(kzg_pp, kzg_vp, circuit);
    println!(
        "preprocess_time = {} ms",
        preprocess_start.elapsed().as_millis()
    );

    let prover = Prover::new(pk);
    let mut ro = RandomOracle::new(&mut rng);
    let proof = prover.prove(z, gamma, &mut ro);

    let verifier = Verifier::new(vk);
    let verifier_start = Instant::now();
    verifier.verify(proof, gamma, &mut ro);
    let verifier_time = verifier_start.elapsed().as_millis();
    println!("verifier_time = {} ms", verifier_time);
    println!("proof verified");
}
