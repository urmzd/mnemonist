"""Temporal feature computation and sinusoidal encoding.

This module implements the query-time temporal scoring used by the memory
embedding model. Temporal features are NOT baked into embeddings — they
modulate search scores at retrieval time.

Temporal features:
- age_days: (now - created_at) in days
- recency_days: (now - last_accessed_at) in days
- access_frequency: access_count / age_days
- type_weight: memory type durability (feedback=1.0, project=0.8, user=0.6, reference=0.4)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TemporalFeatures:
    """Temporal features for a memory entry."""

    age_days: float
    recency_days: float
    access_frequency: float
    type_weight: float

    @classmethod
    def from_metadata(
        cls,
        created_at: datetime,
        last_accessed_at: datetime,
        access_count: int,
        memory_type: str,
        now: datetime | None = None,
    ) -> TemporalFeatures:
        """Compute temporal features from memory metadata."""
        if now is None:
            now = datetime.now()

        age_days = max((now - created_at).total_seconds() / 86400, 0.01)
        recency_days = max((now - last_accessed_at).total_seconds() / 86400, 0.0)
        access_frequency = access_count / age_days

        type_weight = {
            "feedback": 1.0,
            "project": 0.8,
            "user": 0.6,
            "reference": 0.4,
        }.get(memory_type, 0.5)

        return cls(
            age_days=age_days,
            recency_days=recency_days,
            access_frequency=access_frequency,
            type_weight=type_weight,
        )


def temporal_score(features: TemporalFeatures) -> float:
    """Compute a temporal relevance score in [0, 1].

    score = exp(-decay * recency) * ln(1 + freq) * type_weight

    This score is combined with cosine similarity at query time:
        final_score = (1 - lambda) * cosine_sim + lambda * temporal_score
    """
    recency = math.exp(-0.01 * features.recency_days)
    frequency_boost = math.log(1.0 + features.access_frequency)
    # Clamp frequency_boost to [0, 1] range for normalization
    frequency_boost = min(frequency_boost / 3.0, 1.0)
    return recency * frequency_boost * features.type_weight


def temporal_sinusoidal_encoding(features: TemporalFeatures, dim: int = 64) -> list[float]:
    """Encode temporal features as a sinusoidal position vector.

    Similar to transformer positional encoding but over temporal dimensions.
    Each of the 4 features gets dim/4 encoding dimensions using sin/cos pairs.

    This encoding is used during training to teach the model temporal patterns.
    At inference time, the simpler `temporal_score()` function is used instead.
    """
    emb = [0.0] * dim
    feature_values = [
        features.age_days,
        features.recency_days,
        features.access_frequency,
        features.type_weight,
    ]

    dims_per_feature = dim // 4
    for f_idx, feat in enumerate(feature_values):
        offset = f_idx * dims_per_feature
        for i in range(dims_per_feature // 2):
            freq = 1.0 / (10000.0 ** (i / (dims_per_feature / 2)))
            emb[offset + 2 * i] = math.sin(feat * freq)
            emb[offset + 2 * i + 1] = math.cos(feat * freq)

    return emb
