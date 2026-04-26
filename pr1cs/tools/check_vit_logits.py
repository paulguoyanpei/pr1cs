"""Compare PR1CS ViT integer logits with the exported reference logits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
Q = 12


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--actual", type=Path, default=None)
    args = parser.parse_args()

    actual_path = args.actual or (args.root / "vit_logits_pr1cs.bin")
    expected = np.fromfile(args.root / "vit_logits_int.bin", dtype=np.int64)
    actual = np.fromfile(actual_path, dtype=np.int64)
    if expected.shape != actual.shape:
        raise AssertionError(f"shape mismatch: expected {expected.shape}, actual {actual.shape}")
    if not np.array_equal(expected, actual):
        diff = actual - expected
        raise AssertionError(
            "logits mismatch\n"
            f"expected={expected.tolist()}\n"
            f"actual={actual.tolist()}\n"
            f"diff={diff.tolist()}"
        )
    print("integer logits match")
    print((actual.astype(np.float32) / (1 << Q)).tolist())


if __name__ == "__main__":
    main()
