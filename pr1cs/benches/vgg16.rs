use std::cmp;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use ark_bn254::{Bn254, Fr};
use ark_ff::{AdditiveGroup, UniformRand};
use pr1cs::preprocess::Preprocessor;
use pr1cs::prover::Prover;
use pr1cs::verifier::Verifier;
use pr1cs::{circuit::LookupType, instruction::Instruction, program::Program};
use rand::thread_rng;
use util::kzg::{LOG_CHUNK_SIZE, Mkzg};
use util::util::RandomOracle;

const KERNEL: usize = 3; // 3x3 conv
const LAYER_WEIGHT_LEN: [usize; 14] = [
    1728, 36864, 73728, 147456, 294912, 589824, 589824, 1179648, 2359296, 2359296, 2359296,
    2359296, 2359296, 5120,
];

#[derive(Copy, Clone, Debug)]
enum Layer {
    Conv(usize),
    Pool,
    Linear(usize),
}

const VGG16: &[Layer] = &[
    Layer::Conv(64),
    Layer::Conv(64),
    Layer::Pool,
    Layer::Conv(128),
    Layer::Conv(128),
    Layer::Pool,
    Layer::Conv(256),
    Layer::Conv(256),
    Layer::Conv(256),
    Layer::Pool,
    Layer::Conv(512),
    Layer::Conv(512),
    Layer::Conv(512),
    Layer::Pool,
    Layer::Conv(512),
    Layer::Conv(512),
    Layer::Conv(512),
    Layer::Pool,
    Layer::Linear(10),
];

fn data_path(name: &str) -> PathBuf {
    // `cargo bench` runs with CWD at the crate root (pr1cs/pr1cs).
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push(name);
    p
}

fn read_i32_bin(path: &Path) -> Vec<i32> {
    let mut buf = Vec::new();
    File::open(path)
        .unwrap_or_else(|e| panic!("open {}: {}", path.display(), e))
        .read_to_end(&mut buf)
        .unwrap();
    assert_eq!(
        buf.len() % 4,
        0,
        "{} is not a multiple of 4 bytes",
        path.display()
    );
    buf.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Layout the whole pr1cs `weights` vector: [1] || conv weights (transposed to
/// pr1cs (Cin, Cout, kH, kW)) || linear weights (Kin, Mout).
struct WeightPlan {
    buf: Vec<i64>,
    /// Start index (within `buf`) of the weights for each of the 14 learned
    /// layers (13 convs + 1 linear).
    layer_start: Vec<usize>,
}

fn build_weight_plan(raw: &[i32]) -> WeightPlan {
    let mut buf = vec![1i64];
    let mut layer_start = Vec::with_capacity(14);
    let mut off = 0usize;

    // Reconstruct the layout on the fly while scanning VGG16.
    let mut channels_in = 3usize;
    let mut learned_idx = 0usize;

    for layer in VGG16 {
        match layer {
            Layer::Conv(cout) => {
                let cout = *cout;
                let len = LAYER_WEIGHT_LEN[learned_idx];
                assert_eq!(len, cout * channels_in * KERNEL * KERNEL);
                layer_start.push(buf.len());

                // VerfCNN layout: (Cout, Cin, kH, kW) - pr1cs layout: (Cin, Cout, kH, kW)
                let slab = &raw[off..off + len];
                buf.resize(buf.len() + len, 0);
                let dest_start = *layer_start.last().unwrap();
                for d in 0..cout {
                    for c in 0..channels_in {
                        for u in 0..KERNEL {
                            for v in 0..KERNEL {
                                let src = d * channels_in * KERNEL * KERNEL
                                    + c * KERNEL * KERNEL
                                    + u * KERNEL
                                    + v;
                                let dst =
                                    dest_start + (c * cout + d) * KERNEL * KERNEL + u * KERNEL + v;
                                buf[dst] = slab[src] as i64;
                            }
                        }
                    }
                }
                off += len;
                learned_idx += 1;
                channels_in = cout;
            }
            Layer::Pool => { /* no weights */ }
            Layer::Linear(mout) => {
                let len = LAYER_WEIGHT_LEN[learned_idx];
                // Linear weight layout (Kin, Mout) identical between VerfCNN
                // and pr1cs MatMult; copy verbatim.
                layer_start.push(buf.len());
                buf.extend(raw[off..off + len].iter().map(|&x| x as i64));
                let _ = mout;
                off += len;
                learned_idx += 1;
            }
        }
    }

    assert_eq!(off, raw.len(), "raw weights not fully consumed");
    assert_eq!(learned_idx, 14);
    WeightPlan { buf, layer_start }
}

struct ProgramPlan {
    instructions: Vec<Instruction>,
    /// Slot (in the full trace) where the 10 logits are written.
    logits_start: usize,
    /// First auxiliary index past the trace (feeds `gen_z`).
    aux_start_hint: usize,
    /// Start of input block (right after weights).
    input_start: usize,
}

/// Build the entire VGG-16 instruction list and return bookkeeping pointers.
fn build_program(weight_len: usize, input_len: usize, plan: &WeightPlan) -> ProgramPlan {
    let mut instructions = Vec::new();

    // Trace index of the next value produced by an instruction.
    let mut next_slot = weight_len + input_len;
    // Start slot of the current feature map (contiguous (C, H, W) block).
    let mut fm_start = weight_len;
    let mut side = 32usize;
    let mut channels = 3usize;

    let mut learned_idx = 0usize;

    for layer in VGG16 {
        match layer {
            Layer::Conv(cout) => {
                let cout = *cout;
                let conv_start = next_slot;
                let conv_side = side + KERNEL - 1;
                let conv_plane = conv_side * conv_side;

                instructions.push(Instruction::Conv {
                    n: side,
                    m: KERNEL,
                    in_channels: channels,
                    out_channels: cout,
                    start1: fm_start,
                    start2: plan.layer_start[learned_idx],
                });
                next_slot += cout * conv_plane;

                // Quant the center n x n of each output channel (discard the
                // (n+2)x(n+2) border). Order them (d, i, j) so that the Quant
                // block is a contiguous (cout, side, side) feature map.
                let quant_start = next_slot;
                for d in 0..cout {
                    for i in 0..side {
                        for j in 0..side {
                            let src = conv_start + d * conv_plane + (i + 1) * conv_side + (j + 1);
                            instructions.push(Instruction::Quant {
                                input1: vec![(src, 1)],
                                input2: vec![(0, 1)],
                            });
                            next_slot += 1;
                        }
                    }
                }

                // ReLU (table lookup) on the Quant outputs; same (d, i, j)
                // ordering so the ReLU block is a contiguous feature map.
                let relu_start = next_slot;
                for d in 0..cout {
                    for i in 0..side {
                        for j in 0..side {
                            let src = quant_start + d * side * side + i * side + j;
                            instructions.push(Instruction::Lookup {
                                input: vec![(src, 1)],
                                tp: LookupType::Relu,
                            });
                            next_slot += 1;
                        }
                    }
                }

                fm_start = relu_start;
                channels = cout;
                learned_idx += 1;
            }
            Layer::Pool => {
                // 2x2 max pool with stride 2 using `max(a, b) = a + ReLU(b - a)`.
                let out_side = side / 2;
                let in_plane = side * side;
                let out_plane = out_side * out_side;

                // Three ReLUs per output element, then one AddMult to
                // materialize the pooled value. Emit the ReLUs in a single
                // 3-per-output block, and the AddMults in a contiguous block
                // afterwards so the next layer can consume a (C, out, out)
                // feature map at a single start address.
                let relu_block_start = next_slot;
                for c in 0..channels {
                    for i in 0..out_side {
                        for j in 0..out_side {
                            let base = fm_start + c * in_plane;
                            let a = base + (2 * i) * side + (2 * j);
                            let b = base + (2 * i) * side + (2 * j + 1);
                            let cc = base + (2 * i + 1) * side + (2 * j);
                            let dd = base + (2 * i + 1) * side + (2 * j + 1);

                            let idx = (c * out_side + i) * out_side + j;
                            let r1_slot = relu_block_start + 3 * idx;
                            let r2_slot = relu_block_start + 3 * idx + 1;

                            // r1 = ReLU(b - a)
                            instructions.push(Instruction::Lookup {
                                input: vec![(b, 1), (a, -1)],
                                tp: LookupType::Relu,
                            });
                            next_slot += 1;
                            // r2 = ReLU(c - (a + r1))
                            instructions.push(Instruction::Lookup {
                                input: vec![(cc, 1), (a, -1), (r1_slot, -1)],
                                tp: LookupType::Relu,
                            });
                            next_slot += 1;
                            // r3 = ReLU(d - (a + r1 + r2))
                            instructions.push(Instruction::Lookup {
                                input: vec![(dd, 1), (a, -1), (r1_slot, -1), (r2_slot, -1)],
                                tp: LookupType::Relu,
                            });
                            next_slot += 1;
                        }
                    }
                }

                // Materialize the max via AddMult: max = (a + r1 + r2 + r3) * 1.
                let pool_out_start = next_slot;
                for c in 0..channels {
                    for i in 0..out_side {
                        for j in 0..out_side {
                            let base = fm_start + c * in_plane;
                            let a = base + (2 * i) * side + (2 * j);
                            let idx = (c * out_side + i) * out_side + j;
                            let r1 = relu_block_start + 3 * idx;
                            let r2 = relu_block_start + 3 * idx + 1;
                            let r3 = relu_block_start + 3 * idx + 2;
                            instructions.push(Instruction::AddMult {
                                input1: vec![(a, 1), (r1, 1), (r2, 1), (r3, 1)],
                                input2: vec![(0, 1)],
                            });
                            next_slot += 1;
                        }
                    }
                }
                let _ = out_plane;

                fm_start = pool_out_start;
                side = out_side;
            }
            Layer::Linear(mout) => {
                // Linear layer: input is (channels, side, side) contiguous; we
                // treat it as a flat K-vector. MatMult(m=1, n=K, k=M) computes
                // output[0, j] = sum_l A[0, l] * B[l, j], matching VerfCNN's
                // `mat_mult(x, w, o, K, M)`.
                let mout = *mout;
                let k_in = channels * side * side;

                let logits_start = next_slot;
                instructions.push(Instruction::MatMult {
                    m: 1,
                    n: k_in,
                    k: mout,
                    start1: fm_start,
                    start2: plan.layer_start[learned_idx],
                });
                next_slot += mout;
                let _ = learned_idx;
                return ProgramPlan {
                    instructions,
                    logits_start,
                    aux_start_hint: next_slot,
                    input_start: weight_len,
                };
            }
        }
    }

    unreachable!("VGG16 must end in a Linear layer")
}

fn main() {
    let raw_weights = read_i32_bin(&data_path("vgg16_weights.bin"));
    let raw_input = read_i32_bin(&data_path("vgg16_input.bin"));
    assert_eq!(raw_input.len(), 3 * 32 * 32);

    let plan = build_weight_plan(&raw_weights);
    let weight_len = plan.buf.len();
    let input_len = raw_input.len();
    let program_plan = build_program(weight_len, input_len, &plan);

    let ProgramPlan {
        instructions,
        logits_start,
        aux_start_hint,
        input_start,
    } = program_plan;

    println!(
        "instr count: {}, weight_len: {}, logits_start: {}, expected trace: {}",
        instructions.len(),
        weight_len,
        logits_start,
        aux_start_hint,
    );
    let _ = input_start;

    let program = Program::<Fr>::new(instructions, plan.buf);
    let input_i64: Vec<i64> = raw_input.iter().map(|&x| x as i64).collect();
    let trace = program.execute(input_i64);
    assert_eq!(trace.len(), aux_start_hint);

    let mut rng = thread_rng();
    let gamma = <Fr as UniformRand>::rand(&mut rng);
    let z = program.gen_z(weight_len + input_len, trace, gamma);

    let mut table = vec![];
    for i in 0..(1 << 6) {
        table.push((Fr::ZERO, Fr::from(i), Fr::from(1)));
    }
    for i in (-(1 << 16) + 1)..(1 << 16) {
        table.push((Fr::from(i), Fr::from(cmp::max(0, i)), Fr::from(2)));
    }

    let circuit = program.to_circuit(input_len, aux_start_hint, table);
    circuit.check(z.clone(), gamma);
    println!("circuit.check ok");

    let mut rng = thread_rng();
    let (kzg_pp, kzg_vp) = Mkzg::<Bn254>::gen_srs(LOG_CHUNK_SIZE, &mut rng);
    let (pk, vk) = Preprocessor::build(kzg_pp, kzg_vp, circuit);
    let prover = Prover::new(pk);
    let mut ro = RandomOracle::new(&mut rng);
    let proof = prover.prove(z, gamma, &mut ro);
    let verifier = Verifier::new(vk);
    verifier.verify(proof, gamma, &mut ro);
    println!("proof verified");
}
