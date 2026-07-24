#!/usr/bin/env python3
"""One-time safe conversion of VoxCPM2's AudioVAE checkpoint.

Chew never loads pickle/PyTorch checkpoints at runtime. This preparation tool
uses torch's weights-only loader and writes plain Safetensors next to the
official audiovae.pth.
"""

from pathlib import Path
import argparse

import torch
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    source = args.model_dir / "audiovae.pth"
    target = args.model_dir / "audiovae.safetensors"
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    tensors = {
        name: tensor.detach().contiguous()
        for name, tensor in state.items()
    }
    save_file(
        tensors,
        target,
        metadata={"format": "pt", "source": "OpenBMB/VoxCPM2 audiovae.pth"},
    )
    print(f"wrote {target} ({len(tensors)} tensors)")


if __name__ == "__main__":
    main()
