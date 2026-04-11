use crate::circuit::LookupType;

pub enum Instruction {
    AddMult {
        input1: Vec<(usize, i64)>,
        input2: Vec<(usize, i64)>,
    },
    Lookup {
        input: Vec<(usize, i64)>,
        tp: LookupType,
    },
    Quant {
        input1: Vec<(usize, i64)>,
        input2: Vec<(usize, i64)>,
    },
    MatMult {
        m: usize,
        n: usize,
        k: usize,
        start1: usize,
        start2: usize,
    },
}

impl Instruction {
    pub fn output_len(&self) -> usize {
        match self {
            &Self::MatMult {
                m,
                n: _,
                k,
                start1: _,
                start2: _,
            } => m * k,
            _ => 1,
        }
    }
}
