//! Temporal decay for memory retrieval.
//!
//! One signal survives here: **content freshness**. Between two candidates the
//! embedder cannot separate (same content evolved over time — a codebase that
//! changes often, superseded notes), the newer version should win.
//!
//!   freshness(age_days) = exp(-0.01 * age_days)     // ~half-life 70 days
//!
//! Freshness decays on *content age* (`created_at`, rewritten on upsert), not
//! on access time: a stale memory that keeps getting recalled must not look
//! fresh, and newly written content must not need prior accesses to score.
//!
//! The previous formula — `recency * ln(1 + access/age) * type_weight` — was
//! multiplicative in access frequency, so a never-accessed memory scored zero
//! regardless of freshness, and the reinforcement factor measurably never
//! changed a retrieval outcome (Exp 4: 0/0 discordant pairs, p = 1.0).
//! Access counts still matter for decay *protection* during consolidation;
//! they no longer influence ranking.

/// Decay rate per day. exp(-0.01 * 70) ≈ 0.5 — freshness halves in ~70 days.
pub const DECAY_RATE: f64 = 0.01;

/// Content freshness in (0, 1]: 1.0 for brand-new content, ~0.5 at 70 days,
/// ~0.13 at 200 days. `age_days` below zero (clock skew) clamps to 1.0.
pub fn freshness(age_days: f64) -> f64 {
    (-DECAY_RATE * age_days.max(0.0)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_content_scores_one() {
        assert!((freshness(0.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn halves_around_seventy_days() {
        let f = freshness(70.0);
        assert!((0.45..0.55).contains(&f), "expected ~0.5, got {f}");
    }

    #[test]
    fn old_content_decays_but_never_reaches_zero() {
        let f = freshness(365.0);
        assert!(f > 0.0 && f < 0.05, "expected small positive, got {f}");
    }

    #[test]
    fn monotonically_decreasing() {
        assert!(freshness(1.0) > freshness(30.0));
        assert!(freshness(30.0) > freshness(180.0));
    }

    #[test]
    fn negative_age_clamps_to_fresh() {
        assert!((freshness(-5.0) - 1.0).abs() < 1e-9);
    }
}
