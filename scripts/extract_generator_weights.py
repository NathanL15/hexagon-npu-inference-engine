"""
Neural Canvas - Generator Weight Extractor
==========================================
Reads weights/generator.pth and writes each tensor as a flat little-endian
float32 binary file that the C++ engine loads directly with fread():

    weights/fc1_weight.bin   shape [128, 16]   -> 2048 floats
    weights/fc1_bias.bin     shape [128]        ->  128 floats
    weights/fc2_weight.bin   shape [512, 128]   -> 65536 floats
    weights/fc2_bias.bin     shape [512]        ->  512 floats
    weights/fc3_weight.bin   shape [784, 512]   -> 401408 floats
    weights/fc3_bias.bin     shape [784]        ->  784 floats

Binary layout: raw IEEE-754 float32 values, no header, row-major (C order).
"""

import os
import struct
import sys

import torch


# Map from PyTorch state-dict keys to output filenames
_KEY_MAP = {
    "net.0.weight": "fc1_weight.bin",
    "net.0.bias":   "fc1_bias.bin",
    "net.2.weight": "fc2_weight.bin",
    "net.2.bias":   "fc2_bias.bin",
    "net.4.weight": "fc3_weight.bin",
    "net.4.bias":   "fc3_bias.bin",
}


def extract(
    model_path: str = "weights/generator.pth",
    output_dir: str = "weights",
):
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        print("Run scripts/train_generator.py first.")
        sys.exit(1)

    state_dict = torch.load(model_path, map_location="cpu")

    os.makedirs(output_dir, exist_ok=True)

    for key, filename in _KEY_MAP.items():
        if key not in state_dict:
            print(f"[ERROR] Key '{key}' not found in state dict.")
            print(f"Available keys: {list(state_dict.keys())}")
            sys.exit(1)

        tensor = state_dict[key].detach().cpu().float().contiguous()
        flat = tensor.numpy().flatten()
        out_path = os.path.join(output_dir, filename)

        with open(out_path, "wb") as f:
            f.write(struct.pack(f"{len(flat)}f", *flat))

        print(f"  Wrote {out_path:40s}  ({len(flat):>7} floats, {len(flat) * 4:>10} bytes)")

    print(f"\n[DONE] 6 weight files written to '{output_dir}/'")
    print("Next step: build and run the C++ engine.")


if __name__ == "__main__":
    extract()
