"""Export a CIFAR-10 test image as the fixed-point input for the VGG16 bench.

Mirrors VerfCNN's `main.py` preprocessing exactly:

    ToTensor()                                  # uint8 HWC -> float32 CHW in [0, 1]
    Normalize((0.4914, 0.4822, 0.4465),
              (0.247, 0.243, 0.261))
    (x * 2 ** FXP_VALUE).round().astype(intc)   # FXP_VALUE = 6

VerfCNN iterates the torchvision CIFAR-10 test set with `shuffle=False` and
exits after the first batch, so index 0 is the sample it actually proves.

Writes `vgg16_input.bin` (3*32*32 little-endian int32, C-H-W order) and
`vgg16_label.txt` next to the benchmark crate files.

The image source is the HuggingFace mirror of the CIFAR-10 test split, whose
row order matches the original `test_batch`:

    curl -L -o cifar10_test.parquet \
      https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet
"""
import argparse
import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

FXP_VALUE = 6
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD = np.array([0.247, 0.243, 0.261], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path, help="CIFAR-10 test split parquet")
    parser.add_argument("--index", type=int, default=0, help="test-set index to export")
    args = parser.parse_args()

    table = pq.read_table(args.parquet)
    row = table.slice(args.index, 1).to_pylist()[0]
    label = int(row["label"])

    img = Image.open(io.BytesIO(row["img"]["bytes"])).convert("RGB")
    hwc = np.asarray(img, dtype=np.uint8)
    assert hwc.shape == (32, 32, 3), hwc.shape

    # ToTensor: HWC uint8 -> CHW float32 in [0, 1].
    x = hwc.transpose(2, 0, 1).astype(np.float32) / np.float32(255.0)
    # Normalize, then quantize with round-half-to-even (torch.round semantics).
    x = (x - MEAN[:, None, None]) / STD[:, None, None]
    q = np.round(x * np.float32(2**FXP_VALUE)).astype(np.int32)

    (ROOT / "vgg16_input.bin").write_bytes(q.tobytes())
    (ROOT / "vgg16_label.txt").write_text(f"{label}\n")
    print(f"index {args.index}: label={label} range={q.min()}..{q.max()}")
    print(f"wrote {ROOT / 'vgg16_input.bin'} ({q.size} int32) and vgg16_label.txt")


if __name__ == "__main__":
    main()
