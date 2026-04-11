use ark_bn254::Fr;
use ark_ff::{AdditiveGroup, UniformRand};
use pr1cs::{circuit::LookupType, instruction::Instruction, program::Program};
use rand::thread_rng;
use std::cmp;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
const KERNEL_SIZE: usize = 3;

enum Layer {
    Conv(usize), // output channel
    Pooling,
    Linear(usize), // output dim
}

struct Vgg16 {
    layers: Vec<Layer>,
}

const WEIGHT_ALIGN: usize = 15 << 20;
const INPUT_ALIGN: usize = 1 << 12;
const TRACE_ALIGN: usize = 3 << 20;
impl Vgg16 {
    fn weight_length(
        &self,
        mut input_width: usize,
        mut input_height: usize,
        mut input_channel: usize,
    ) -> usize {
        let mut sum = 0;
        for i in &self.layers {
            match i {
                &Layer::Conv(output_channel) => {
                    println!(
                        "{}",
                        input_channel * output_channel * KERNEL_SIZE * KERNEL_SIZE
                    );
                    sum += input_channel * output_channel * KERNEL_SIZE * KERNEL_SIZE;
                    input_channel = output_channel;
                    input_height = input_height + 2 - KERNEL_SIZE + 1;
                    input_width = input_width + 2 - KERNEL_SIZE + 1;
                }
                &Layer::Pooling => {
                    input_height /= 2;
                    input_width /= 2;
                }
                &Layer::Linear(output_dim) => {
                    println!(
                        "{}",
                        output_dim * (input_channel * input_height * input_width)
                    );
                    sum += output_dim * (input_channel * input_height * input_width)
                }
            }
        }
        sum
    }

    fn instructions(
        &self,
        mut input_width: usize,
        mut input_height: usize,
        mut input_channel: usize,
    ) -> Vec<Instruction> {
        // let weight_length = self.weight_length(input_width, input_height, input_channel);
        let mut instructions = vec![];
        let mut weight_start = 1;
        let mut input_start = WEIGHT_ALIGN;
        let input_size = input_channel * input_height * input_width;
        for (idx, layer) in self.layers.iter().enumerate() {
            match layer {
                &Layer::Conv(output_channel) => {
                    let compute_value = |i: usize, j: usize, c: usize| {
                        if i == 0 || j == 0 || i == input_height + 1 || j == input_width + 1 {
                            vec![]
                        } else {
                            vec![(
                                input_start
                                    + c * input_height * input_width
                                    + (i - 1) * input_width
                                    + (j - 1),
                                1i64,
                            )]
                        }
                    };
                    for c in 0..input_channel {
                        for k1 in 0..KERNEL_SIZE {
                            for k2 in 0..KERNEL_SIZE {
                                for i in 0..input_height {
                                    for j in 0..input_width {
                                        let input1 = compute_value(i + k1, j + k2, c);
                                        instructions.push(Instruction::AddMult {
                                            input1,
                                            input2: vec![(0, 1)],
                                        });
                                    }
                                }
                            }
                        }
                    }
                    if idx == 0 {
                        input_start += INPUT_ALIGN;
                    } else {
                        input_start += input_height * input_width * input_channel;
                    }
                    instructions.push(Instruction::MatMult {
                        m: input_height * input_width,
                        n: input_channel * KERNEL_SIZE * KERNEL_SIZE,
                        k: output_channel,
                        start1: input_start,
                        start2: weight_start,
                    });
                    input_start +=
                        input_channel * KERNEL_SIZE * KERNEL_SIZE * input_height * input_width;
                    let compute_value = |i: usize, j: usize, c: usize| {
                        input_start + c + (i * input_width + j) * output_channel
                    };
                    for c in 0..output_channel {
                        for i in 0..input_height {
                            for j in 0..input_width {
                                instructions.push(Instruction::Quant {
                                    input1: vec![(compute_value(i, j, c), 1)],
                                    input2: vec![(0, 1)],
                                });
                            }
                        }
                    }
                    input_start += output_channel * input_height * input_width;
                    let compute_value = |i: usize, j: usize, c: usize| {
                        input_start + c * input_height * input_width + i * input_width + j
                    };
                    for c in 0..output_channel {
                        for i in 0..input_height {
                            for j in 0..input_width {
                                instructions.push(Instruction::Lookup {
                                    input: vec![(compute_value(i, j, c), 1)],
                                    tp: LookupType::Relu,
                                });
                            }
                        }
                    }
                    input_start += output_channel * input_height * input_width;

                    weight_start += input_channel * output_channel * KERNEL_SIZE * KERNEL_SIZE;
                    input_channel = output_channel;
                }
                &Layer::Pooling => {
                    let compute_input_index = |i: usize, j: usize, c: usize| {
                        input_start + c * input_height * input_width + i * input_width + j
                    };
                    let compute_output_index = |i: usize, j: usize, c: usize, round: usize| {
                        let h = input_height / 2;
                        let w = input_width / 2;
                        input_start
                            + input_channel * input_height * input_width
                            + round * input_channel * h * w
                            + c * h * w
                            + i * w
                            + j
                    };
                    for c in 0..input_channel {
                        for i in 0..(input_width / 2) {
                            for j in 0..(input_height / 2) {
                                instructions.push(Instruction::Lookup {
                                    input: vec![
                                        (compute_input_index(i * 2, j * 2, c), 1),
                                        (compute_input_index(i * 2 + 1, j * 2, c), -1),
                                    ],
                                    tp: LookupType::Relu,
                                });
                            }
                        }
                    }
                    for c in 0..input_channel {
                        for i in 0..(input_width / 2) {
                            for j in 0..(input_height / 2) {
                                instructions.push(Instruction::Lookup {
                                    input: vec![
                                        (compute_input_index(i * 2 + 1, j * 2, c), 1),
                                        (compute_output_index(i, j, c, 0), 1),
                                        (compute_input_index(i * 2, j * 2 + 1, c), -1),
                                    ],
                                    tp: LookupType::Relu,
                                });
                            }
                        }
                    }
                    for c in 0..input_channel {
                        for i in 0..(input_width / 2) {
                            for j in 0..(input_height / 2) {
                                instructions.push(Instruction::Lookup {
                                    input: vec![
                                        (compute_input_index(i * 2, j * 2 + 1, c), 1),
                                        (compute_output_index(i, j, c, 1), 1),
                                        (compute_input_index(i * 2 + 1, j * 2 + 1, c), -1),
                                    ],
                                    tp: LookupType::Relu,
                                });
                            }
                        }
                    }
                    for c in 0..input_channel {
                        for i in 0..(input_width / 2) {
                            for j in 0..(input_height / 2) {
                                instructions.push(Instruction::AddMult {
                                    input1: vec![
                                        (compute_input_index(i * 2 + 1, j * 2 + 1, c), 1),
                                        (compute_output_index(i, j, c, 2), 1),
                                    ],
                                    input2: vec![(0, 1)],
                                });
                            }
                        }
                    }
                    input_start += input_channel * input_height * input_width
                        + 3 * input_channel * (input_height / 2) * (input_width / 2);
                    input_height /= 2;
                    input_width /= 2;
                }
                &Layer::Linear(output_dim) => {
                    instructions.push(Instruction::MatMult {
                        m: 1,
                        n: input_channel * input_height * input_width,
                        k: output_dim,
                        start1: input_start,
                        start2: weight_start,
                    });
                    input_start += input_channel * input_height * input_width + output_dim;
                    weight_start += output_dim * (input_channel * input_height * input_width)
                }
            }
        }
        assert!(input_start < (18 << 20));
        assert!(weight_start < (15 << 20));
        instructions
    }
}

fn main() {
    let vgg16 = Vgg16 {
        layers: vec![
            Layer::Conv(64),
            Layer::Conv(64),
            Layer::Pooling,
            Layer::Conv(128),
            Layer::Conv(128),
            Layer::Pooling,
            Layer::Conv(256),
            Layer::Conv(256),
            Layer::Conv(256),
            Layer::Pooling,
            Layer::Conv(512),
            Layer::Conv(512),
            Layer::Conv(512),
            Layer::Pooling,
            Layer::Conv(512),
            Layer::Conv(512),
            Layer::Conv(512),
            Layer::Pooling,
            Layer::Linear(10),
        ],
    };
    let input_width = 32;
    let input_height = 32;
    let input_channel = 3;
    let file = File::open("weight.txt").unwrap();
    let reader = BufReader::new(file);

    let mut weights = vec![1];
    // Read and parse each line
    for line in reader.lines() {
        let int_value: i64 = line.unwrap().parse().unwrap();
        weights.push(int_value << 1);
    }
    while weights.len() < WEIGHT_ALIGN {
        weights.push(0);
    }

    // commit weights
    let instructions = vgg16.instructions(input_width, input_height, input_channel);
    let program = Program::<Fr>::new(instructions, weights);

    let file = File::open("input_file.txt").unwrap();
    let reader = BufReader::new(file);
    let mut input = vec![];
    // Read and parse each line
    for line in reader.lines() {
        let int_value: i64 = line.unwrap().parse().unwrap();
        input.push(int_value << 3);
    }
    while input.len() < INPUT_ALIGN {
        input.push(0);
    }

    let trace = program.execute(input);

    let mut rng = thread_rng();
    let gamma = <Fr as UniformRand>::rand(&mut rng);
    let z = program.gen_z(trace, gamma);

    let circuit = program.to_circuit();

    let mut set = HashSet::new();
    for i in (-(1 << 16) + 1)..(1 << 16) {
        set.insert((Fr::from(i), Fr::from(cmp::max(0, i)), Fr::ZERO));
    }
    for i in 0..(1 << 6) {
        set.insert((Fr::ZERO, Fr::from(i), Fr::ZERO));
    }
    circuit.check(z, gamma, set);
}
