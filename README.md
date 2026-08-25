# PR1CS

PR1CS is a Rust workspace for building and verifying lookup-enabled arithmetic
circuits over prime fields. The main crate turns a compact instruction program
into sparse circuit constraints, preprocesses those constraints, and runs a
prover/verifier protocol backed by multilinear KZG commitments.

The repository currently focuses on integer neural-network style workloads:
matrix multiplication, convolution, division/quantization, and lookup-table
activations such as ReLU, GELU, exp, reciprocal, and reciprocal square root.

## Workspace Layout

```text
.
├── Cargo.toml          # workspace manifest
├── pr1cs/              # circuit model, preprocessing, prover, verifier
│   ├── src/
│   ├── benches/        # DNN, VGG16, and ViT benchmark drivers
│   └── tools/          # data export/reference helpers
└── util/               # multilinear polynomials, KZG, random oracle helpers
```

## Crates

- `pr1cs`: exposes the circuit, instruction, program, preprocessing, prover,
  verifier, and sparse proof modules.
- `util`: provides multilinear polynomial utilities, a random oracle, and the
  `Mkzg` multilinear KZG commitment implementation used by `pr1cs`.

## Requirements

- Rust and Cargo. The workspace uses Rust 2021 for `pr1cs` and Rust 2024 for
  `util`, so use a recent stable toolchain.
- Optional Python dependencies for the helper scripts in `pr1cs/tools`.
  `reference.py` needs NumPy. `export_vit.py` additionally expects PyTorch and
  the sibling `../int-vit` project with its checkpoint/data setup.
- Optional C++ toolchain if rebuilding `pr1cs/tools/dump_weights.cpp`.

## Quick Start

Build the workspace:

```sh
cargo build
```

Run the test suite:

```sh
cargo test
```

Run a benchmark target:

```sh
cargo bench -p pr1cs --bench dnn
cargo bench -p pr1cs --bench vgg16
cargo bench -p pr1cs --bench vit
cargo bench -p util --bench kzg
```

## Core Flow

A typical proof pipeline is:

1. Build a `Program` from `Instruction` values and integer weights.
2. Execute the program on integer inputs to produce a trace.
3. Convert the program into a `Circuit`.
4. Generate KZG parameters and preprocess the circuit.
5. Produce a proof with `Prover`.
6. Verify it with `Verifier`.

The `pr1cs/benches/dnn.rs` benchmark is the smallest end-to-end example. It
constructs a layered integer network, creates range and ReLU lookup tables,
checks the resulting circuit, proves the execution trace, and verifies the
proof.

## Supported Instructions

`pr1cs::instruction::Instruction` currently supports:

- `AddMult`: multiply two linear combinations and constrain the output.
- `MatMult`: matrix multiplication with compressed random-linear-combination
  constraints.
- `Conv`: full convolution constraints for image-like tensors.
- `Div`: integer division with a range lookup for the remainder.
- `Lookup`: table lookup constraints for nonlinear or quantized operations.

Lookup types are defined in `pr1cs::circuit::LookupType`.

## Data Tools

The `pr1cs/tools` directory contains helpers for benchmark data:

- `reference.py`: runs a NumPy reference pass for the shipped VGG16 integer
  data and writes `vgg16_logits.txt`.
- `dump_weights.cpp`: helper for converting VerfCNN-style weights into the
  binary layout consumed by the VGG16 benchmark. Build it with
  `build_dump_weights.sh`, which expects a VerfCNN checkout as a sibling of the
  workspace root (override with `VERFCNN_DIR`).
- `export_vgg16_input.py`: quantizes one CIFAR-10 test image into
  `vgg16_input.bin` / `vgg16_label.txt`, mirroring VerfCNN's `main.py`
  preprocessing. Needs NumPy, PyArrow, and Pillow.
- `export_vit.py`: exports fixed-point DeiT-Tiny inputs, parameters, lookup
  tables, and reference logits for the ViT benchmark.
- `check_vit_logits.py`: compares PR1CS ViT integer logits with exported
  reference logits.

Some generated benchmark data is expected to live under `pr1cs/` next to the
benchmark crate files.

## Development Notes

- Keep generated or large benchmark artifacts out of version control unless
  they are intentionally part of a reproducible benchmark fixture.
- Prefer adding focused tests near the module being changed. Existing examples
  live in `pr1cs/src/program.rs`, `pr1cs/src/preprocess.rs`, and
  `util/src/kzg.rs`.
- The circuit code is generic over `ark_ff::PrimeField`; benchmarks currently
  use `ark_bn254::Fr` with `ark_bn254::Bn254` commitments.
