"""Loss functions for distillation training.

Three loss components:
1. Cosine similarity loss — student similarity should match target similarity
2. Contrastive loss — anchor should be closer to positive than negative
3. Ranking loss — preserve relative ordering of similarities
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineSimilarityLoss(nn.Module):
    """MSE between student cosine similarity and target similarity."""

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        target_similarity: torch.Tensor,
    ) -> torch.Tensor:
        student_sim = F.cosine_similarity(emb_a, emb_b, dim=-1)
        return F.mse_loss(student_sim, target_similarity)


class TripletContrastiveLoss(nn.Module):
    """Triplet loss: anchor should be closer to positive than negative."""

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        pos_sim = F.cosine_similarity(anchor, positive, dim=-1)
        neg_sim = F.cosine_similarity(anchor, negative, dim=-1)
        return F.relu(neg_sim - pos_sim + self.margin).mean()


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss with in-batch negatives."""

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
    ) -> torch.Tensor:
        # Normalize
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)

        # Similarity matrix (batch x batch)
        sim_matrix = torch.mm(anchor, positive.t()) / self.temperature

        # Labels: diagonal entries are positive pairs
        labels = torch.arange(sim_matrix.shape[0], device=sim_matrix.device)

        return F.cross_entropy(sim_matrix, labels)


class DistillationLoss(nn.Module):
    """Combined distillation loss.

    L = alpha * L_cosine + beta * L_contrastive + gamma * L_infonce
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.5,
        margin: float = 0.2,
        temperature: float = 0.05,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.cosine_loss = CosineSimilarityLoss()
        self.triplet_loss = TripletContrastiveLoss(margin=margin)
        self.infonce_loss = InfoNCELoss(temperature=temperature)
