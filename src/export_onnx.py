"""Export TinyTransformerEncoder to ONNX with dynamic batch/sequence axes, then verify it."""
import argparse
import hashlib

import onnx
import torch

from src.contract import HIDDEN, INPUT_NAMES, OUTPUT_NAME, PROFILE_OPT, VOCAB_SIZE
from src.model import TinyTransformerEncoder


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def export(onnx_path: str = "model.onnx", seed: int = 0) -> dict:
    model = TinyTransformerEncoder(seed=seed).eval()
    batch, seq = PROFILE_OPT
    input_ids = torch.randint(0, VOCAB_SIZE, (batch, seq), dtype=torch.int32)
    attention_mask = torch.ones(batch, seq, dtype=torch.int32)

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        onnx_path,
        input_names=list(INPUT_NAMES),
        output_names=[OUTPUT_NAME],
        dynamic_axes={
            INPUT_NAMES[0]: {0: "batch", 1: "sequence"},
            INPUT_NAMES[1]: {0: "batch", 1: "sequence"},
            OUTPUT_NAME: {0: "batch", 1: "sequence"},
        },
        opset_version=17,
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    return {"onnx_sha256": sha256_of_file(onnx_path), "seed": seed, "hidden": HIDDEN}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="model.onnx")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(export(args.out, args.seed))
