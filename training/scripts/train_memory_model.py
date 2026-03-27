#!/usr/bin/env python3
"""Train the memory embedding model.

Usage:
    python scripts/train_memory_model.py --data ../dataset/memory/generated/ --epochs 20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(__file__ and os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.distill.dataset import MemoryQueryDataset
from src.distill.trainer import Trainer
from src.models.memory_encoder import MemoryEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Train memory embedding model")
    parser.add_argument("--data", type=Path, required=True, help="Path to generated/ directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/memory"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    print("Loading tokenizer...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Load datasets
    queries_path = args.data / "memory_queries.jsonl"

    if not queries_path.exists():
        print(f"No training data found at {queries_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading memory queries from {queries_path}", file=sys.stderr)
    dataset = MemoryQueryDataset(queries_path, tokenizer)
    print(f"  {len(dataset)} training samples", file=sys.stderr)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Initialize model
    model = MemoryEncoder(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=args.hidden_dim,
        max_seq_len=256,
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count / 1e6:.1f}M", file=sys.stderr)

    trainer = Trainer(
        model=model,
        train_loader=loader,
        lr=args.lr,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )

    for epoch in range(args.epochs):
        loss = trainer.train_epoch_triplet(epoch)
        trainer.log(f"  Epoch {epoch + 1}/{args.epochs} — loss: {loss:.4f}")

        if (epoch + 1) % 5 == 0:
            path = trainer.save_checkpoint(f"memory_epoch{epoch + 1}")
            trainer.log(f"  Saved: {path}")

    final_path = trainer.save_checkpoint("memory_latest")
    print(f"\nTraining complete. Final checkpoint: {final_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
