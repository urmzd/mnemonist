//! Temporal scoring for memory retrieval.
//!
//! Ported from `training/src/models/temporal.py`.
//!
//! Score = exp(-0.01 * recency_days) * ln(1 + access_frequency) * type_weight
//!
//! Combined with cosine similarity at query time:
//!   final_score = (1 - λ) * cosine_sim + λ * temporal_score

use crate::MemoryType;

/// Type durability weights — how long each type naturally persists.
fn type_weight(mt: MemoryType) -> f64 {
    match mt {
        MemoryType::Feedback => 1.0,
        MemoryType::Project => 0.8,
        MemoryType::User => 0.6,
        MemoryType::Reference => 0.4,
    }
}

/// Compute temporal relevance score in [0, 1].
///
/// - `recency_days`: days since last access (0 = just accessed)
/// - `age_days`: days since creation (must be > 0)
/// - `access_count`: number of times retrieved
/// - `memory_type`: memory type for durability weighting
pub fn temporal_score(
    recency_days: f64,
    age_days: f64,
    access_count: u32,
    memory_type: MemoryType,
) -> f64 {
    let recency = (-0.01 * recency_days).exp();
    let frequency = (access_count as f64) / age_days.max(0.01);
    let frequency_boost = ((1.0 + frequency).ln() / 3.0).min(1.0);
    recency * frequency_boost * type_weight(memory_type)
}

/// Blend cosine similarity with temporal score.
///
/// `lambda` ∈ [0, 1]: 0 = pure cosine, 1 = pure temporal.
pub fn blend(cosine_sim: f64, temporal: f64, lambda: f64) -> f64 {
    (1.0 - lambda) * cosine_sim + lambda * temporal
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recent_high_frequency_scores_high() {
        // Just accessed, accessed 10x in 5 days
        let score = temporal_score(0.0, 5.0, 10, MemoryType::Feedback);
        // frequency=2.0, boost=ln(3)/3≈0.37, recency=1.0, type=1.0 → ~0.37
        assert!(score > 0.3, "expected > 0.3, got {score}");
        // Much higher than old/infrequent
        let old_score = temporal_score(200.0, 200.0, 0, MemoryType::Feedback);
        assert!(score > old_score * 2.0, "recent should dominate old");
    }

    #[test]
    fn old_never_accessed_scores_low() {
        // 200 days old, never accessed beyond creation
        let score = temporal_score(200.0, 200.0, 0, MemoryType::Reference);
        assert!(score < 0.1, "expected < 0.1, got {score}");
    }

    #[test]
    fn type_weight_ordering() {
        let base = |mt| temporal_score(1.0, 10.0, 5, mt);
        assert!(base(MemoryType::Feedback) > base(MemoryType::Project));
        assert!(base(MemoryType::Project) > base(MemoryType::User));
        assert!(base(MemoryType::User) > base(MemoryType::Reference));
    }

    #[test]
    fn blend_pure_cosine() {
        assert!((blend(0.9, 0.1, 0.0) - 0.9).abs() < 1e-6);
    }

    #[test]
    fn blend_pure_temporal() {
        assert!((blend(0.9, 0.1, 1.0) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn blend_mixed() {
        // λ=0.2: 0.8 * 0.9 + 0.2 * 0.5 = 0.72 + 0.10 = 0.82
        let result = blend(0.9, 0.5, 0.2);
        assert!((result - 0.82).abs() < 1e-6, "got {result}");
    }
}
