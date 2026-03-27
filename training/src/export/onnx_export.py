"""Export trained PyTorch models to ONNX format for Rust inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.models.code_encoder import CodeEncoder
from src.models.memory_encoder import MemoryEncoder


def export_to_onnx(
    model: torch.nn.Module,
    output_path: Path,
    max_seq_len: int = 512,
    hidden_dim: int = 384,
    opset_version: int = 17,
) -> None:
    """Export a model to ONNX format.

    The exported model takes (input_ids, attention_mask) and returns embeddings.
    """
    model.eval()
    device = next(model.parameters()).device

    # Dummy inputs
    batch_size = 1
    dummy_input_ids = torch.randint(0, 1000, (batch_size, max_seq_len), device=device)
    dummy_attention_mask = torch.ones(batch_size, max_seq_len, dtype=torch.long, device=device)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        str(output_path),
        opset_version=opset_version,
        input_names=["input_ids", "attention_mask"],
        output_names=["embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "embeddings": {0: "batch_size"},
        },
    )
    print(f"Exported to {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-type", choices=["code", "memory"], required=True)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--max-seq-len", type=int, default=512)
    args = parser.parse_args()

    if args.model_type == "code":
        model = CodeEncoder(hidden_dim=args.hidden_dim, max_seq_len=args.max_seq_len)
    else:
        model = MemoryEncoder(
            hidden_dim=args.hidden_dim, max_seq_len=min(args.max_seq_len, 256)
        )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    export_to_onnx(model, args.output, max_seq_len=args.max_seq_len)


if __name__ == "__main__":
    main()
