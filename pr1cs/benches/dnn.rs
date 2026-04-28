use ark_bn254::{Bn254, Fr};
use ark_ff::{AdditiveGroup, UniformRand};
use pr1cs::preprocess::Preprocessor;
use pr1cs::prover::Prover;
use pr1cs::verifier::Verifier;
use pr1cs::{circuit::LookupType, instruction::Instruction, program::Program};
use rand::thread_rng;
use std::cmp;
use std::time::Instant;
use util::kzg::{Mkzg, LOG_CHUNK_SIZE};
use util::util::RandomOracle;

const LAYER_COUNT: usize = 16;
const HIDDEN_DIM: usize = 128;
const SEQ_LEN: usize = 1;
const INPUT_ALIGN: usize = HIDDEN_DIM * SEQ_LEN;

fn instructions(weight_len: usize) -> Vec<Instruction> {
    let mut instructions = vec![];
    let mut weight_start = 1;
    let mut input_start = weight_len;

    for _ in 0..LAYER_COUNT {
        instructions.push(Instruction::MatMult {
            m: SEQ_LEN,
            n: HIDDEN_DIM,
            k: HIDDEN_DIM,
            start1: input_start,
            start2: weight_start,
        });

        let matmul_start = input_start + INPUT_ALIGN;
        for offset in 0..INPUT_ALIGN {
            instructions.push(Instruction::Div {
                input1: vec![(matmul_start + offset, 1)],
                input2: vec![(0, 1)],
                divisor: 64,
            });
        }

        let quant_start = matmul_start + INPUT_ALIGN;
        for offset in 0..INPUT_ALIGN {
            instructions.push(Instruction::Lookup {
                input: vec![(quant_start + offset, 1)],
                tp: LookupType::Relu,
            });
        }

        input_start = quant_start + INPUT_ALIGN;
        weight_start += HIDDEN_DIM * HIDDEN_DIM;
    }

    assert_eq!(weight_start, weight_len);
    instructions
}

fn main() {
    let mut rng = thread_rng();
    let mut weights = vec![1];
    for _ in 0..LAYER_COUNT {
        for l in 0..HIDDEN_DIM {
            for j in 0..HIDDEN_DIM {
                weights.push(if l == j { 1 } else { 0 });
            }
        }
    }
    let weight_len = weights.len();

    let instructions = instructions(weights.len());
    let program = Program::<Fr>::new(instructions, weights);

    let mut input = vec![];
    while input.len() < INPUT_ALIGN {
        input.push(1 << 6);
    }

    let trace = program.execute(input);
    assert_eq!(
        trace.len(),
        1 + LAYER_COUNT * HIDDEN_DIM * HIDDEN_DIM + INPUT_ALIGN + LAYER_COUNT * INPUT_ALIGN * 3
    );
    let aux_start = trace.len();

    let gamma = <Fr as UniformRand>::rand(&mut rng);
    let z = program.gen_z(weight_len + INPUT_ALIGN, trace, gamma);

    let mut table = vec![];
    for i in 0..(1 << 6) {
        table.push((Fr::ZERO, Fr::from(i), Fr::from(LookupType::Range(64).tag())));
    }
    for i in (-(1 << 16) + 1)..(1 << 16) {
        table.push((Fr::from(i), Fr::from(cmp::max(0, i)), Fr::from(2)));
    }
    let circuit = program.to_circuit(INPUT_ALIGN, aux_start, table);
    circuit.check(z.clone(), gamma);

    let mut rng = thread_rng();
    let (kzg_pp, kzg_vp) = Mkzg::<Bn254>::gen_srs(LOG_CHUNK_SIZE, &mut rng);
    let (pk, vk) = Preprocessor::build(kzg_pp, kzg_vp, circuit);
    let prover = Prover::new(pk);
    let mut ro = RandomOracle::new(&mut rng);
    let proof = prover.prove(z, gamma, &mut ro);
    println!("proof size {} bytes", proof.size());
    let verifier = Verifier::new(vk);
    let _ = weight_len;
    let verifier_start = Instant::now();
    verifier.verify(proof, gamma, &mut ro);
    let verifier_time = verifier_start.elapsed().as_millis();
    println!("verifier_time = {} ms", verifier_time);
    println!("finish DNN!")
}
