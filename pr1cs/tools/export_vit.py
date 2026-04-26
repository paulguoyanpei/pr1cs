"""Export fixed-point DeiT-Tiny data for the PR1CS ViT benchmark.

Run from this crate or repo root after installing the ../int-vit Python deps:

    python pr1cs/tools/export_vit.py

The PR1CS benchmark consumes the generated *.bin files from pr1cs/pr1cs/.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
INT_VIT = (ROOT / "../../int-vit").resolve()


def write_i64(path: Path, arr) -> None:
    np.asarray(arr, dtype=np.int64).reshape(-1).tofile(path)


def export_linear(buf: list[np.ndarray], linear) -> None:
    # PR1CS MatMult stores weights as (in_features, out_features).
    buf.append(linear.W_int.t().contiguous().cpu().numpy())
    buf.append(linear.b_int.contiguous().cpu().numpy())


def logits_int(fxp, x_int: torch.Tensor) -> torch.Tensor:
    bsz = x_int.shape[0]
    cls_tokens = fxp.cls_token_int.expand(bsz, -1, -1)
    x_int = torch.cat([cls_tokens, x_int], dim=1)
    x_int = x_int + fxp.pos_embed_int
    for block in fxp.blocks:
        x_int = block(x_int)
    x_int = fxp.norm(x_int)
    return fxp.head(x_int[:, 0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()

    os.chdir(INT_VIT)
    sys.path.insert(0, str(INT_VIT))
    import config  # noqa: WPS433
    from data import get_dataloaders  # noqa: WPS433
    from fixed_point_ops import quantize  # noqa: WPS433
    from fixed_point_vit import FixedPointViT  # noqa: WPS433
    from model import get_model  # noqa: WPS433

    ckpt_path = INT_VIT / config.CHECKPOINT_DIR / "deit_tiny_cifar10_best.pth"
    model = get_model(pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()

    fxp = FixedPointViT(model, config.DEFAULT_Q)

    _, val_loader = get_dataloaders(batch_size=1, num_workers=0)
    image = None
    label = None
    seen = 0
    for images, labels in val_loader:
        if seen + images.shape[0] > args.sample_index:
            offset = args.sample_index - seen
            image = images[offset : offset + 1]
            label = labels[offset : offset + 1]
            break
        seen += images.shape[0]
    if image is None:
        raise ValueError(f"sample-index {args.sample_index} is outside validation set")

    image_int = quantize(image, config.DEFAULT_Q)
    embedded = fxp.patch_embed(image_int)
    ref_logits_int = logits_int(fxp, embedded)

    buf: list[np.ndarray] = []
    buf.append(fxp.cls_token_int[0, 0].contiguous().cpu().numpy())
    buf.append(fxp.pos_embed_int[0].contiguous().cpu().numpy())
    for block in fxp.blocks:
        buf.append(block.norm1.gamma_int.contiguous().cpu().numpy())
        buf.append(block.norm1.beta_int.contiguous().cpu().numpy())
        export_linear(buf, block.attn.qkv)
        export_linear(buf, block.attn.proj)
        buf.append(block.norm2.gamma_int.contiguous().cpu().numpy())
        buf.append(block.norm2.beta_int.contiguous().cpu().numpy())
        export_linear(buf, block.mlp.fc1)
        export_linear(buf, block.mlp.fc2)
    buf.append(fxp.norm.gamma_int.contiguous().cpu().numpy())
    buf.append(fxp.norm.beta_int.contiguous().cpu().numpy())
    export_linear(buf, fxp.head)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_i64(args.out_dir / "vit_params.bin", np.concatenate([x.reshape(-1) for x in buf]))
    write_i64(args.out_dir / "vit_input_embedded.bin", embedded[0].contiguous().cpu().numpy())
    write_i64(args.out_dir / "vit_logits_int.bin", ref_logits_int[0].contiguous().cpu().numpy())
    write_i64(args.out_dir / "vit_lut_gelu.bin", fxp.gelu_lut.table.cpu().numpy())
    write_i64(args.out_dir / "vit_lut_exp.bin", fxp.exp_lut.table.cpu().numpy())
    write_i64(args.out_dir / "vit_lut_recip.bin", fxp.recip_lut.table.cpu().numpy())
    write_i64(args.out_dir / "vit_lut_rsqrt.bin", fxp.rsqrt_lut.table.cpu().numpy())

    (args.out_dir / "vit_label.txt").write_text(f"{int(label.item())}\n")
    (args.out_dir / "vit_logits_float.txt").write_text(
        "\n".join(str(float(x)) for x in (ref_logits_int[0].float() / (1 << config.DEFAULT_Q)))
        + "\n"
    )
    print(f"wrote ViT PR1CS files to {args.out_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
