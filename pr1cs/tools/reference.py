"""Reference CIFAR-10 VGG-16 inference that mirrors VerfCNN's integer flow.

Reads the weights/scales/input binaries shipped with this crate and writes the
final logits to vgg16_logits.txt.  The computation matches convnet.cpp:
    for each conv layer:
        y = conv2d(x, w)               # integer, 3x3, stride 1, pad 1
        y = relu(y >> 6)               # trunc toward 0, then max(_, 0)
    for each max-pool layer:
        y = max over 2x2 windows (stride 2)
    linear:
        logits = x @ w.reshape(Kin, Mout)  # no shift
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

LAYERS = [
    ("C", 64), ("C", 64), ("M", 0),
    ("C", 128), ("C", 128), ("M", 0),
    ("C", 256), ("C", 256), ("C", 256), ("M", 0),
    ("C", 512), ("C", 512), ("C", 512), ("M", 0),
    ("C", 512), ("C", 512), ("C", 512), ("M", 0),
    ("L", 10),
]
LAYER_SIZES = [
    1728, 36864, 73728, 147456, 294912, 589824, 589824,
    1179648, 2359296, 2359296, 2359296, 2359296, 2359296, 5120,
]


def load_weights():
    all_w = np.fromfile(ROOT / "vgg16_weights.bin", dtype=np.int32)
    segments = []
    off = 0
    for sz in LAYER_SIZES:
        segments.append(all_w[off:off + sz].astype(np.int64))
        off += sz
    assert off == all_w.size
    return segments


def conv2d_int(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """x: (C_in, H, W) int64, w: (C_out, C_in, 3, 3) int64 -> (C_out, H, W)."""
    C_in, H, W = x.shape
    C_out = w.shape[0]
    pad = np.zeros((C_in, H + 2, W + 2), dtype=np.int64)
    pad[:, 1:1 + H, 1:1 + W] = x
    out = np.zeros((C_out, H, W), dtype=np.int64)
    for d in range(C_out):
        for c in range(C_in):
            wv = w[d, c]
            for u in range(3):
                for v in range(3):
                    out[d] += wv[u, v] * pad[c, u:u + H, v:v + W]
    return out


def relu_shift(y: np.ndarray) -> np.ndarray:
    # Truncate toward zero (C int division semantics), then max(_, 0).
    shifted = np.where(y >= 0, y // 64, -((-y) // 64))
    return np.maximum(shifted, 0)


def maxpool2(x: np.ndarray) -> np.ndarray:
    C, H, W = x.shape
    x = x.reshape(C, H // 2, 2, W // 2, 2)
    return x.max(axis=(2, 4))


def main():
    weights = load_weights()
    x = np.fromfile(ROOT / "vgg16_input.bin", dtype=np.int32).astype(np.int64)
    x = x.reshape(3, 32, 32)
    label = int((ROOT / "vgg16_label.txt").read_text().strip())
    print(f"input range: {x.min()}..{x.max()} label: {label}")

    wi = 0
    for idx, (kind, param) in enumerate(LAYERS):
        if kind == "C":
            C_in = x.shape[0]
            C_out = param
            w = weights[wi].reshape(C_out, C_in, 3, 3)
            y = conv2d_int(x, w)
            x = relu_shift(y)
            wi += 1
            print(f"layer {idx} C{C_out}: shape={x.shape} range={x.min()}..{x.max()}")
        elif kind == "M":
            x = maxpool2(x)
            print(f"layer {idx} M:   shape={x.shape} range={x.min()}..{x.max()}")
        else:  # 'L'
            C, H, W = x.shape
            flat = x.reshape(-1)
            w = weights[wi].reshape(C * H * W, param)
            logits = flat @ w
            wi += 1
            print(f"logits: {logits.tolist()}")
            print(f"predicted: {int(np.argmax(logits))} (label {label})")
            (ROOT / "vgg16_logits.txt").write_text(
                "\n".join(str(int(v)) for v in logits) + "\n"
            )
    assert wi == len(LAYER_SIZES)


if __name__ == "__main__":
    main()
