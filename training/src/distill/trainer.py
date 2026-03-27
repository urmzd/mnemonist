"""Training loop for distillation of embedding models."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.distill.loss import DistillationLoss


class Trainer:
    """Distillation trainer for code and memory embedding models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        lr: float = 2e-4,
        epochs: int = 20,
        warmup_steps: int = 100,
        checkpoint_dir: Path = Path("checkpoints"),
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.loss_fn = DistillationLoss()
        self.best_val_loss = float("inf")

    def train_epoch_code_pairs(self, epoch: int) -> float:
        """Train one epoch on code-description pairs (InfoNCE)."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            code_ids = batch["code_input_ids"].to(self.device)
            code_mask = batch["code_attention_mask"].to(self.device)
            desc_ids = batch["desc_input_ids"].to(self.device)
            desc_mask = batch["desc_attention_mask"].to(self.device)

            code_emb = self.model(code_ids, code_mask)
            desc_emb = self.model(desc_ids, desc_mask)

            loss = self.loss_fn.infonce_loss(code_emb, desc_emb)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        self.scheduler.step()
        return total_loss / max(n_batches, 1)

    def train_epoch_similarity(self, epoch: int) -> float:
        """Train one epoch on similarity triples (cosine loss)."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            a_ids = batch["a_input_ids"].to(self.device)
            a_mask = batch["a_attention_mask"].to(self.device)
            b_ids = batch["b_input_ids"].to(self.device)
            b_mask = batch["b_attention_mask"].to(self.device)
            sim = batch["similarity"].to(self.device)

            emb_a = self.model(a_ids, a_mask)
            emb_b = self.model(b_ids, b_mask)

            loss = self.loss_fn.cosine_loss(emb_a, emb_b, sim)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        self.scheduler.step()
        return total_loss / max(n_batches, 1)

    def train_epoch_triplet(self, epoch: int) -> float:
        """Train one epoch on query-positive-negative triples."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            q_ids = batch["query_input_ids"].to(self.device)
            q_mask = batch["query_attention_mask"].to(self.device)
            p_ids = batch["positive_input_ids"].to(self.device)
            p_mask = batch["positive_attention_mask"].to(self.device)
            n_ids = batch["negative_input_ids"].to(self.device)
            n_mask = batch["negative_attention_mask"].to(self.device)

            q_emb = self.model(q_ids, q_mask)
            p_emb = self.model(p_ids, p_mask)
            n_emb = self.model(n_ids, n_mask)

            loss = self.loss_fn.triplet_loss(q_emb, p_emb, n_emb)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        self.scheduler.step()
        return total_loss / max(n_batches, 1)

    def save_checkpoint(self, name: str) -> Path:
        """Save model checkpoint."""
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        return path

    def log(self, msg: str) -> None:
        print(msg, file=sys.stderr)
