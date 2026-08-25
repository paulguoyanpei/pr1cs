//! One-time model registration on a fully connected DNN.
//!
//! Registration is paid once per model, not per inference, so this driver
//! reports it separately from the online prover in `dnn.rs`. Set `LAYERS` and
//! `HIDDEN` to change the model size.

use ark_bn254::{Bn254, Fr};
use pr1cs::preprocess::Preprocessor;
use pr1cs::registration::{ProgramEncoding, Registrar};
use pr1cs::{
    circuit::{divrelu_table, LookupType},
    instruction::Instruction,
    program::Program,
};
use rand::thread_rng;
use std::time::Instant;
use util::kzg::{Mkzg, LOG_CHUNK_SIZE};
use util::util::RandomOracle;

/// Requantization divisor applied after every matmul.
const QUANT: i64 = 64;
const LOG_TABLE_HALF: usize = 12;

fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let layers = env("LAYERS", 4);
    let hidden = env("HIDDEN", 64);

    let mut weights = vec![1];
    for _ in 0..layers {
        for l in 0..hidden {
            for j in 0..hidden {
                weights.push(if l == j { 1 } else { 0 });
            }
        }
    }
    let weight_len = weights.len();

    let mut instructions = vec![];
    let mut weight_start = 1;
    let mut input_start = weight_len;
    for _ in 0..layers {
        instructions.push(Instruction::MatMult {
            m: 1,
            n: hidden,
            k: hidden,
            start1: input_start,
            start2: weight_start,
        });
        let matmul_start = input_start + hidden;
        for offset in 0..hidden {
            instructions.push(Instruction::Lookup {
                input: vec![(matmul_start + offset, 1)],
                tp: LookupType::DivRelu(QUANT),
            });
        }
        input_start = matmul_start + hidden;
        weight_start += hidden * hidden;
    }

    let program = Program::<Fr>::new(instructions, weights);
    let input = vec![1 << 6; hidden];
    let trace = program.execute(input.clone());
    let aux_start = trace.len();

    let table = divrelu_table::<Fr>(QUANT, 1 << LOG_TABLE_HALF);
    let compile_start = Instant::now();
    let circuit = program
        .compile(input.len(), aux_start, table)
        .expect("program is not valid");
    println!(
        "{} layers x {} hidden: {} constraint rows, {} lookup rows, |z| = {} (compile {} ms)",
        layers,
        hidden,
        circuit.a.len(),
        circuit.d.len(),
        circuit.z_len,
        compile_start.elapsed().as_millis()
    );

    let mut rng = thread_rng();
    let (kzg_pp, kzg_vp) = Mkzg::<Bn254>::gen_srs(LOG_CHUNK_SIZE, &mut rng);
    let (pk, vk) = Preprocessor::build(kzg_pp, kzg_vp, circuit);

    let enc = ProgramEncoding::from_program(&program, input.len(), aux_start);
    let mut ro = RandomOracle::new(&mut rng);

    let register_start = Instant::now();
    let (key, proof) = Registrar::register(&pk, &vk, &enc, &mut ro).expect("registration failed");
    let register_time = register_start.elapsed().as_millis();
    println!(
        "expansion entries: {}, certificate {} bytes, prover {} ms",
        key.profile.x_len,
        proof.size(),
        register_time
    );

    let verify_start = Instant::now();
    Registrar::verify(&vk, &key, proof, &mut ro);
    println!("verifier {} ms", verify_start.elapsed().as_millis());
}
