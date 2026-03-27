#!/usr/bin/env python3
"""Train the code embedding model.

Usage:
    python scripts/train_code_model.py --data ../dataset/code/generated/ --epochs 20
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

from src.distill.dataset import CodePairDataset, CodeSimilarityDataset
from src.distill.trainer import Trainer
from src.models.code_encoder import CodeEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Train code embedding model")
    parser.add_argument("--data", type=Path, required=True, help="Path to generated/ directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/code"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    print("Loading tokenizer...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Load datasets
    pairs_path = args.data / "code_pairs.jsonl"
    sim_path = args.data / "code_similarity.jsonl"

    datasets = []
    if pairs_path.exists():
        print(f"Loading code pairs from {pairs_path}", file=sys.stderr)
        datasets.append(("pairs", CodePairDataset(pairs_path, tokenizer)))
    if sim_path.exists():
        print(f"Loading code similarity from {sim_path}", file=sys.stderr)
        datasets.append(("similarity", CodeSimilarityDataset(sim_path, tokenizer)))

    if not datasets:
        print("No training data found!", file=sys.stderr)
        sys.exit(1)

    # Initialize model
    model = CodeEncoder(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=args.hidden_dim,
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count / 1e6:.1f}M", file=sys.stderr)

    # Train on each dataset type
    for ds_name, dataset in datasets:
        print(f"\nTraining on {ds_name} ({len(dataset)} samples)...", file=sys.stderr)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        trainer = Trainer(
            model=model,
            train_loader=loader,
            lr=args.lr,
            epochs=args.epochs,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
        )

        for epoch in range(args.epochs):
            if ds_name == "pairs":
                loss = trainer.train_epoch_code_pairs(epoch)
            else:
                loss = trainer.train_epoch_similarity(epoch)

            trainer.log(f"  Epoch {epoch + 1}/{args.epochs} — loss: {loss:.4f}")

            if (epoch + 1) % 5 == 0:
                path = trainer.save_checkpoint(f"code_{ds_name}_epoch{epoch + 1}")
                trainer.log(f"  Saved: {path}")

    # Final checkpoint
    final_path = trainer.save_checkpoint("code_latest")
    print(f"\nTraining complete. Final checkpoint: {final_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
