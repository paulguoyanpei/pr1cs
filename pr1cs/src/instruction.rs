use crate::circuit::LookupType;

#[derive(Clone)]
pub enum Instruction {
    AddMult {
        input1: Vec<(usize, i64)>,
        input2: Vec<(usize, i64)>,
    },
    Conv {
        n: usize,
        m: usize,
        in_channels: usize,
        out_channels: usize,
        start1: usize,
        start2: usize,
    },
    Lookup {
        input: Vec<(usize, i64)>,
        tp: LookupType,
    },
    Div {
        input1: Vec<(usize, i64)>,
        input2: Vec<(usize, i64)>,
        divisor: i64,
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
            &Self::Conv {
                n,
                m,
                in_channels: _,
                out_channels,
                start1: _,
                start2: _,
            } => {
                let side = n + m - 1;
                out_channels * side * side
            }
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
