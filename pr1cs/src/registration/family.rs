//! How a `MatMult` or a `Conv` expands into descriptor entries.
//!
//! Both instructions emit whole blocks of rows, so their entries cannot be
//! enumerated slot by slot the way an `AddMult` can. Instead every block is
//! described here as a *family*: a grid of loop coordinates whose entry
//! `(row, col, val, pow)` is a fixed formula in the instruction's operands and
//! the coordinates.
//!
//! Two properties make the grids cheap to prove. Within one instruction a
//! family occupies a contiguous run of the expansion table, so its flat index
//! is `position - run start` and needs no ordering argument; and in every
//! family the column is `base + flat index`, so only the exponent of gamma
//! needs the digits back.

use ark_ff::PrimeField;

use crate::registration::encode::Opcode;

/// A small arithmetic expression over an instruction's operands.
#[derive(Clone, Debug)]
pub(crate) enum Ex {
    Const(i64),
    Op(usize),
    Add(Box<Ex>, Box<Ex>),
    Sub(Box<Ex>, Box<Ex>),
    Mul(Box<Ex>, Box<Ex>),
}

pub(crate) fn c(value: i64) -> Ex {
    Ex::Const(value)
}

pub(crate) fn op(slot: usize) -> Ex {
    Ex::Op(slot)
}

pub(crate) fn add(a: Ex, b: Ex) -> Ex {
    Ex::Add(Box::new(a), Box::new(b))
}

pub(crate) fn sub(a: Ex, b: Ex) -> Ex {
    Ex::Sub(Box::new(a), Box::new(b))
}

pub(crate) fn mul(a: Ex, b: Ex) -> Ex {
    Ex::Mul(Box::new(a), Box::new(b))
}

impl Ex {
    pub(crate) fn eval_i64(&self, ops: &[i64]) -> i64 {
        match self {
            Ex::Const(v) => *v,
            Ex::Op(slot) => ops[*slot],
            Ex::Add(a, b) => a.eval_i64(ops) + b.eval_i64(ops),
            Ex::Sub(a, b) => a.eval_i64(ops) - b.eval_i64(ops),
            Ex::Mul(a, b) => a.eval_i64(ops) * b.eval_i64(ops),
        }
    }

    pub(crate) fn eval<F: PrimeField>(&self, ops: &[F]) -> F {
        match self {
            Ex::Const(v) => {
                if *v < 0 {
                    -F::from(v.unsigned_abs())
                } else {
                    F::from(*v as u64)
                }
            }
            Ex::Op(slot) => ops[*slot],
            Ex::Add(a, b) => a.eval(ops) + b.eval(ops),
            Ex::Sub(a, b) => a.eval(ops) - b.eval(ops),
            Ex::Mul(a, b) => a.eval(ops) * b.eval(ops),
        }
    }
}

/// Where a family's columns start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ColBase {
    /// An operand: the start of one of the two operand blocks.
    Operand(usize),
    /// The instruction's advice block.
    Advice,
    /// The instruction's output block.
    Output,
    /// The constant cell `z[0]`, which the sum row multiplies by.
    Unit,
}

/// A term of the gamma exponent: a coefficient times a digit or the flat index.
#[derive(Clone, Debug)]
pub(crate) enum PowTerm {
    Digit(usize),
    Flat,
}

/// Which row of the block an entry lands in.
#[derive(Clone, Debug)]
pub(crate) enum RowSpec {
    /// `cr + digit`, i.e. one row per value of the outermost coordinate.
    Digit(usize),
    /// `cr + expr`: the trailing row that sums the advice cells.
    Offset(Ex),
}

pub(crate) struct Family {
    pub(crate) name: &'static str,
    pub(crate) opcode: Opcode,
    /// 0 = A, 1 = B, 2 = C.
    pub(crate) matrix: usize,
    /// Loop bounds, outermost first.
    pub(crate) radices: Vec<Ex>,
    pub(crate) row: RowSpec,
    pub(crate) col_base: ColBase,
    pub(crate) pow: Vec<(Ex, PowTerm)>,
    pub(crate) pow_const: Ex,
}

impl Family {
    pub(crate) fn digits(&self) -> usize {
        self.radices.len()
    }

    /// Number of entries this family emits for one instruction.
    pub(crate) fn size(&self, ops: &[i64]) -> usize {
        self.radices
            .iter()
            .map(|r| r.eval_i64(ops))
            .product::<i64>() as usize
    }
}

/// Widest grid any family uses; `Conv`'s kernel block needs all four.
pub(crate) const MAX_DIGITS: usize = 4;

/// `MatMult` operands are `m, n, k, s1, s2`; `Conv`'s are
/// `in_channels, out_channels, n, m, s1, s2`.
pub(crate) fn families() -> Vec<Family> {
    let side = || sub(add(op(2), op(3)), c(1));
    let plane = || mul(side(), side());

    vec![
        // --- MatMult: Z = X * Y, checked with n randomized products ---
        // One row per shared-dimension index, selecting a column of X.
        Family {
            name: "matmult.x",
            opcode: Opcode::MatMult,
            matrix: 0,
            radices: vec![op(1), op(0)],
            row: RowSpec::Digit(0),
            col_base: ColBase::Operand(3),
            pow: vec![(op(2), PowTerm::Digit(1))],
            pow_const: c(0),
        },
        // ... and the matching row of Y.
        Family {
            name: "matmult.y",
            opcode: Opcode::MatMult,
            matrix: 1,
            radices: vec![op(1), op(2)],
            row: RowSpec::Digit(0),
            col_base: ColBase::Operand(4),
            pow: vec![(c(1), PowTerm::Digit(1))],
            pow_const: c(0),
        },
        // Each product lands in its own advice cell.
        Family {
            name: "matmult.product",
            opcode: Opcode::MatMult,
            matrix: 2,
            radices: vec![op(1)],
            row: RowSpec::Digit(0),
            col_base: ColBase::Advice,
            pow: vec![],
            pow_const: c(0),
        },
        // The trailing row adds the advice cells up, ...
        Family {
            name: "matmult.sum",
            opcode: Opcode::MatMult,
            matrix: 0,
            radices: vec![op(1)],
            row: RowSpec::Offset(op(1)),
            col_base: ColBase::Advice,
            pow: vec![],
            pow_const: c(0),
        },
        Family {
            name: "matmult.unit",
            opcode: Opcode::MatMult,
            matrix: 1,
            radices: vec![c(1)],
            row: RowSpec::Offset(op(1)),
            col_base: ColBase::Unit,
            pow: vec![],
            pow_const: c(0),
        },
        // ... and compares them with the randomized output block.
        Family {
            name: "matmult.out",
            opcode: Opcode::MatMult,
            matrix: 2,
            radices: vec![op(0), op(2)],
            row: RowSpec::Offset(op(1)),
            col_base: ColBase::Output,
            pow: vec![(c(1), PowTerm::Flat)],
            pow_const: c(0),
        },
        // --- Conv: one product per input channel, VerfCNN style ---
        Family {
            name: "conv.input",
            opcode: Opcode::Conv,
            matrix: 0,
            radices: vec![op(0), op(2), op(2)],
            row: RowSpec::Digit(0),
            col_base: ColBase::Operand(4),
            pow: vec![(side(), PowTerm::Digit(1)), (c(1), PowTerm::Digit(2))],
            pow_const: c(0),
        },
        // The kernel is read in reverse, which turns the product into a
        // convolution rather than a correlation.
        Family {
            name: "conv.kernel",
            opcode: Opcode::Conv,
            matrix: 1,
            radices: vec![op(0), op(1), op(3), op(3)],
            row: RowSpec::Digit(0),
            col_base: ColBase::Operand(5),
            pow: vec![
                (plane(), PowTerm::Digit(1)),
                (sub(c(0), side()), PowTerm::Digit(2)),
                (c(-1), PowTerm::Digit(3)),
            ],
            pow_const: mul(sub(op(3), c(1)), add(side(), c(1))),
        },
        Family {
            name: "conv.product",
            opcode: Opcode::Conv,
            matrix: 2,
            radices: vec![op(0)],
            row: RowSpec::Digit(0),
            col_base: ColBase::Advice,
            pow: vec![],
            pow_const: c(0),
        },
        Family {
            name: "conv.sum",
            opcode: Opcode::Conv,
            matrix: 0,
            radices: vec![op(0)],
            row: RowSpec::Offset(op(0)),
            col_base: ColBase::Advice,
            pow: vec![],
            pow_const: c(0),
        },
        Family {
            name: "conv.unit",
            opcode: Opcode::Conv,
            matrix: 1,
            radices: vec![c(1)],
            row: RowSpec::Offset(op(0)),
            col_base: ColBase::Unit,
            pow: vec![],
            pow_const: c(0),
        },
        Family {
            name: "conv.out",
            opcode: Opcode::Conv,
            matrix: 2,
            radices: vec![op(1), side(), side()],
            row: RowSpec::Offset(op(0)),
            col_base: ColBase::Output,
            pow: vec![(c(1), PowTerm::Flat)],
            pow_const: c(0),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The family sizes have to add up to what the compiler emits per
    /// instruction: `n + 1` constraint rows for a MatMult, `C + 1` for a Conv.
    #[test]
    fn family_sizes_match_the_compiler() {
        let fams = families();
        // m = 2, n = 3, k = 4
        let mm = [2i64, 3, 4, 10, 20, 0];
        let sizes: Vec<usize> = fams
            .iter()
            .filter(|f| f.opcode == Opcode::MatMult)
            .map(|f| f.size(&mm))
            .collect();
        assert_eq!(sizes, vec![3 * 2, 3 * 4, 3, 3, 1, 2 * 4]);

        // C = 2, D = 3, n = 4, m = 2 => side = 5
        let conv = [2i64, 3, 4, 2, 10, 20];
        let sizes: Vec<usize> = fams
            .iter()
            .filter(|f| f.opcode == Opcode::Conv)
            .map(|f| f.size(&conv))
            .collect();
        assert_eq!(sizes, vec![2 * 4 * 4, 2 * 3 * 2 * 2, 2, 2, 1, 3 * 5 * 5]);
    }
}
