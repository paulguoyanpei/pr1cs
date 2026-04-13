use std::marker::PhantomData;

use ark_ff::PrimeField;

use crate::{
    circuit::{Circuit, LookupType, SparseMatrix},
    instruction::Instruction,
};

pub struct Program<F: PrimeField> {
    instructions: Vec<Instruction>,
    weights: Vec<i64>,
    _pd: PhantomData<F>,
}

impl<F: PrimeField> Program<F> {
    pub fn new(instructions: Vec<Instruction>, weights: Vec<i64>) -> Self {
        Program {
            instructions,
            weights,
            _pd: PhantomData::default(),
        }
    }

    pub fn to_circuit(&self, input_size: usize, aux_start: usize, table: Vec<(F, F, F)>) -> Circuit<F> {
        let mut a = SparseMatrix::<F>::new();
        let mut b = SparseMatrix::<F>::new();
        let mut c = SparseMatrix::<F>::new();
        let mut d = SparseMatrix::<F>::new();
        let mut e = SparseMatrix::<F>::new();
        let mut tp = vec![];
        let mut output_index = self.weights.len() + input_size;
        let mut auxiliary_index = aux_start;

        for instr in &self.instructions {
            match &instr {
                &Instruction::AddMult { input1, input2 } => {
                    a.append(input1);
                    b.append(input2);
                    c.append(&vec![(output_index, 1)]);
                    output_index += 1
                }
                &Instruction::Lookup {
                    input,
                    tp: lookup_type,
                } => {
                    d.append(input);
                    e.append(&vec![(output_index, 1)]);
                    tp.push(lookup_type.clone());
                    output_index += 1
                }
                &Instruction::Quant { input1, input2 } => {
                    a.append(input1);
                    b.append(input2);
                    c.append(&vec![(output_index, (1 << 6)), (auxiliary_index, 1)]);
                    d.append(&vec![]);
                    e.append(&vec![(auxiliary_index, 1)]);
                    tp.push(LookupType::Ge0);

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

        return Circuit::<F>::new(a, b, c, d, e, tp, self.weights.len(), table);
    }

    pub fn weights(&self) -> Vec<F> {
        self.weights.iter().map(|&x| F::from(x)).collect()
    }

    pub fn execute(&self, mut input: Vec<i64>) -> Vec<F> {
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
                &Instruction::Lookup { input, tp } => {
                    let mut a = 0;
                    for i in input {
                        a += z[i.0] * i.1
                    }
                    match tp {
                        LookupType::Ge0 => panic!(),
                        LookupType::Relu => {
                            assert!(a < (1 << 16));
                            assert!(a > -(1 << 16));
                            if a >= 0 {
                                z.push(a);
                            } else {
                                z.push(0);
                            }
                        }
                    }
                }
                &Instruction::Quant { input1, input2 } => {
                    let mut a = 0;
                    for i in input1 {
                        a += z[i.0] * i.1
                    }
                    let mut b = 0;
                    for i in input2 {
                        b += z[i.0] * i.1
                    }
                    let c = a * b;
                    z.push(c >> 6);
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

        z.iter().map(|&x| F::from(x)).collect::<Vec<_>>()
    }

    pub fn gen_z(&self, output_start: usize, trace: Vec<F>, gamma: F) -> Vec<F> {
        let mut z = trace.clone();
        let mut aux = vec![];

        let mut output_index = output_start;
        for instr in &self.instructions {
            match &instr {
                &Instruction::AddMult { input1, input2 } => {
                    output_index += 1;
                }
                &Instruction::Lookup { input, tp } => {
                    output_index += 1;
                }
                &Instruction::Quant { input1, input2 } => {
                    // add the remnant r st <A[cr],z> * <B[cr],z> = 2**6 * z[or] + r
                    let mut a = F::zero();
                    for i in input1 {
                        a += trace[i.0] * F::from(i.1);
                    }
                    let mut b = F::zero();
                    for i in input2 {
                        b += trace[i.0] * F::from(i.1);
                    }
                    let c = a * b;
                    let r = c - F::from(1 << 6) * trace[output_index];

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
