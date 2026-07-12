//! Multi-signal reranker with auto-calibration.
//!
//! Replaces the simple `(1-λ)*cosine + λ*temporal` blend with a normalised
//! weighted combination of all available signals, plus a dynamic similarity
//! threshold derived from the embedding space's own statistics.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;

// ---------------------------------------------------------------------------
// Recall profile — per-project calibration stored during `learn`
// ---------------------------------------------------------------------------

/// Embedding-space statistics captured during `learn`.
/// Stored as `.recall-profile.json` in the memory directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallProfile {
    /// Average pairwise cosine similarity (0 = dispersed, 1 = collapsed).
    pub anisotropy: f32,
    /// Max - min pairwise similarity (higher = better use of the space).
    pub similarity_range: f32,
    /// Number of chunks the profile was computed from.
    pub sample_size: usize,
    /// Dynamic similarity floor: results below this are noise.
    pub similarity_floor: f32,
    /// How much to trust cosine vs metadata (auto-tuned from anisotropy).
    pub semantic_confidence: f32,
}

const PROFILE_FILE: &str = ".recall-profile.json";

impl RecallProfile {
    /// Calibrate from embedding quality metrics.
    ///
    /// - Low anisotropy → high semantic confidence (embeddings spread well)
    /// - High anisotropy → low confidence (everything looks similar, lean on metadata)
    /// - Similarity floor derived from the range: if the space is tight,
    ///   the floor must be higher to filter noise.
    pub fn calibrate(anisotropy: f32, similarity_range: f32, sample_size: usize) -> Self {
        // Semantic confidence: inversely proportional to anisotropy.
        // aniso=0 → confidence=0.9, aniso=0.5 → confidence=0.4, aniso≥1 → confidence=0.1
        let semantic_confidence = (0.9 - anisotropy * 1.0).clamp(0.1, 0.9);

        // Similarity floor: baseline 0.25, raised if the range is narrow
        // (narrow range → scores cluster → need higher floor to separate signal from noise).
        let floor = if similarity_range > 0.01 {
            // The floor sits at ~40% of the way up from the bottom of the range.
            // With typical aniso=0.36 and range=1.06 this gives ~0.25 (lenient).
            // With aniso=0.49 and range=1.02 this gives ~0.30 (tighter).
            (anisotropy * 0.5 + (1.0 - similarity_range) * 0.3).clamp(0.15, 0.50)
        } else {
            0.25
        };

        Self {
            anisotropy,
            similarity_range,
            sample_size,
            similarity_floor: floor,
            semantic_confidence,
        }
    }

    /// Load from a memory directory. Returns `None` if no profile exists.
    pub fn load(mem_dir: &Path) -> Option<Self> {
        let path = mem_dir.join(PROFILE_FILE);
        let content = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&content).ok()
    }

    /// Save to a memory directory.
    pub fn save(&self, mem_dir: &Path) -> Result<(), crate::Error> {
        let path = mem_dir.join(PROFILE_FILE);
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| crate::Error::ConfigFormat(e.to_string()))?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// A sensible default for projects that haven't been calibrated yet.
    pub fn uncalibrated() -> Self {
        Self {
            anisotropy: 0.4,
            similarity_range: 1.0,
            sample_size: 0,
            similarity_floor: 0.30,
            semantic_confidence: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Candidate — a search result with all available signals
// ---------------------------------------------------------------------------

/// A candidate result to be reranked.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// Unique identifier (filename or chunk ID).
    pub id: String,
    /// Raw cosine similarity from HNSW search.
    pub cosine_score: f32,
    /// Memory metadata signals (None for code chunks).
    pub memory_signals: Option<MemorySignals>,
    /// Source file path (for diversity tracking).
    pub source_file: String,
}

/// Signals available from memory file frontmatter.
///
/// Access counts and last-accessed timestamps deliberately do not appear:
/// they protect memories from decay during consolidation but measurably never
/// changed a ranking outcome (Exp 4), so ranking ignores them.
#[derive(Debug, Clone)]
pub struct MemorySignals {
    pub strength: f32,
    /// Days since `created_at` — content age. `created_at` is rewritten on
    /// upsert, so this tracks how fresh the content is, not access history.
    pub age_days: f64,
    pub source: Option<String>,
    pub ref_count: usize,
}

// ---------------------------------------------------------------------------
// Reranker
// ---------------------------------------------------------------------------

/// Reranked result with final score and component breakdown.
#[derive(Debug, Clone)]
pub struct RankedResult {
    pub id: String,
    pub final_score: f32,
    pub cosine_score: f32,
    pub source_file: String,
}

/// Absolute cosine below which a candidate is junk regardless of calibration.
/// The fallback path never resurrects candidates under this.
const JUNK_FLOOR: f32 = 0.25;

/// How many candidates the fallback path keeps when the calibrated floor
/// eliminates everything.
const FALLBACK_K: usize = 3;

/// Rerank candidates using normalised multi-signal scoring.
///
/// 1. Filter below similarity floor (falling back to the top few above the
///    junk floor when calibration eliminates everything)
/// 2. Normalise cosine scores within the batch (min-max)
/// 3. Compute weighted signal combination
/// 4. Apply diversity penalty (max 2 results per source file)
/// 5. Sort descending
pub fn rerank(candidates: &[Candidate], profile: &RecallProfile) -> Vec<RankedResult> {
    if candidates.is_empty() {
        return Vec::new();
    }

    // Step 1: filter below floor
    let mut above_floor: Vec<&Candidate> = candidates
        .iter()
        .filter(|c| c.cosine_score >= profile.similarity_floor)
        .collect();

    // A calibrated floor that eliminates every candidate is miscalibrated for
    // this query — recall that returns nothing while plausible matches exist
    // defeats the tool. Fall back to the best few by raw cosine, still
    // refusing outright junk.
    if above_floor.is_empty() {
        let mut plausible: Vec<&Candidate> = candidates
            .iter()
            .filter(|c| c.cosine_score >= JUNK_FLOOR)
            .collect();
        plausible.sort_by(|a, b| {
            b.cosine_score
                .partial_cmp(&a.cosine_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        plausible.truncate(FALLBACK_K);
        above_floor = plausible;
    }

    if above_floor.is_empty() {
        return Vec::new();
    }

    // Step 2: normalise cosine scores within this batch
    let min_cos = above_floor
        .iter()
        .map(|c| c.cosine_score)
        .fold(f32::INFINITY, f32::min);
    let max_cos = above_floor
        .iter()
        .map(|c| c.cosine_score)
        .fold(f32::NEG_INFINITY, f32::max);
    let cos_range = (max_cos - min_cos).max(1e-6);

    let sc = profile.semantic_confidence;

    // Step 3: score each candidate
    let mut scored: Vec<RankedResult> = above_floor
        .iter()
        .map(|c| {
            let norm_cosine = (c.cosine_score - min_cos) / cos_range;

            let metadata_score = match &c.memory_signals {
                Some(ms) => {
                    // Freshness: between candidates the embedder cannot separate
                    // (same content evolved over time), newer content wins.
                    let freshness_bonus = 0.2 * crate::temporal::freshness(ms.age_days) as f32;

                    // Strength bonus: memories that survived consolidation matter more
                    let strength_bonus = (ms.strength / 2.0).min(0.3);

                    // Source bonus: user-created memories > auto-learned.
                    // "memorize" and "note" are legacy values from pre-0.11 memory files.
                    let source_bonus = match ms.source.as_deref() {
                        Some("remember") | Some("memorize") => 0.15,
                        Some("note") => 0.10,
                        Some("consolidation") => 0.05,
                        _ => 0.0,
                    };

                    // Connectivity: memories with refs are better anchored
                    let ref_bonus = (ms.ref_count as f32 * 0.05).min(0.15);

                    freshness_bonus + strength_bonus + source_bonus + ref_bonus
                }
                // Code chunks: no metadata, neutral score
                None => 0.3,
            };

            // Weighted combination: semantic_confidence controls the mix
            let final_score = sc * norm_cosine + (1.0 - sc) * metadata_score;

            RankedResult {
                id: c.id.clone(),
                final_score,
                cosine_score: c.cosine_score,
                source_file: c.source_file.clone(),
            }
        })
        .collect();

    // Step 4: sort descending by score
    scored.sort_by(|a, b| {
        b.final_score
            .partial_cmp(&a.final_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Step 5: diversity — allow max 2 results per source file
    let mut file_counts: HashSet<String> = HashSet::new();
    let mut file_hit_count = std::collections::HashMap::<String, usize>::new();
    let mut diverse: Vec<RankedResult> = Vec::with_capacity(scored.len());

    for r in scored {
        let count = file_hit_count.entry(r.source_file.clone()).or_insert(0);
        if *count < 2 {
            *count += 1;
            file_counts.insert(r.source_file.clone());
            diverse.push(r);
        }
    }

    diverse
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> RecallProfile {
        RecallProfile::calibrate(0.36, 1.06, 100)
    }

    fn code_candidate(id: &str, cosine: f32, file: &str) -> Candidate {
        Candidate {
            id: id.to_string(),
            cosine_score: cosine,
            memory_signals: None,
            source_file: file.to_string(),
        }
    }

    fn memory_candidate(id: &str, cosine: f32, age_days: f64) -> Candidate {
        Candidate {
            id: id.to_string(),
            cosine_score: cosine,
            memory_signals: Some(MemorySignals {
                strength: 1.0,
                age_days,
                source: Some("remember".to_string()),
                ref_count: 2,
            }),
            source_file: id.to_string(),
        }
    }

    #[test]
    fn filters_below_floor() {
        let p = profile();
        let candidates = vec![
            code_candidate("license", 0.10, "LICENSE"),
            code_candidate("good", 0.60, "src/lib.rs"),
        ];
        let results = rerank(&candidates, &p);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "good");
    }

    #[test]
    fn empty_input_returns_empty() {
        let results = rerank(&[], &profile());
        assert!(results.is_empty());
    }

    #[test]
    fn all_below_floor_returns_empty() {
        let p = profile();
        let candidates = vec![
            code_candidate("a", 0.10, "a.rs"),
            code_candidate("b", 0.15, "b.rs"),
        ];
        let results = rerank(&candidates, &p);
        assert!(results.is_empty());
    }

    #[test]
    fn below_calibrated_floor_falls_back_to_plausible() {
        // Strict calibrated floor (collapsed embedding space) must not blank
        // out plausible matches — the fallback keeps the best above JUNK_FLOOR.
        let p = RecallProfile::calibrate(0.48, 0.34, 7);
        let candidates = vec![
            code_candidate("junk", 0.10, "LICENSE"),
            code_candidate("plausible-low", 0.30, "a.rs"),
            code_candidate("plausible-high", 0.40, "b.rs"),
        ];
        assert!(
            candidates
                .iter()
                .all(|c| c.cosine_score < p.similarity_floor)
        );
        let results = rerank(&candidates, &p);
        assert_eq!(results.len(), 2, "junk stays filtered, plausible survive");
        assert_eq!(results[0].id, "plausible-high");
    }

    #[test]
    fn memory_with_signals_ranks_higher_than_bare_code() {
        let p = profile();
        let candidates = vec![
            code_candidate("code", 0.50, "src/lib.rs"),
            memory_candidate("mem", 0.50, 30.0),
        ];
        let results = rerank(&candidates, &p);
        assert_eq!(results.len(), 2);
        // Memory should rank higher due to freshness + strength + source bonuses
        assert_eq!(results[0].id, "mem");
    }

    /// The staleness scenario: the same content exists in several versions
    /// over time (a codebase that changes often, superseded notes). Cosine
    /// cannot separate them — the fresh version must rank first, and access
    /// history must not be able to rescue the stale one (it no longer feeds
    /// ranking at all).
    #[test]
    fn fresh_version_wins_cosine_tie_against_stale() {
        let p = profile();
        let candidates = vec![
            // Unrelated context so cosine normalisation has a realistic range
            code_candidate("noise-low", 0.30, "a.rs"),
            code_candidate("noise-mid", 0.55, "b.rs"),
            // Same evolved content: identical cosine, different content age
            memory_candidate("v-stale", 0.80, 180.0),
            memory_candidate("v-fresh", 0.80, 1.0),
        ];
        let results = rerank(&candidates, &p);
        let stale_rank = results.iter().position(|r| r.id == "v-stale").unwrap();
        let fresh_rank = results.iter().position(|r| r.id == "v-fresh").unwrap();
        assert!(
            fresh_rank < stale_rank,
            "fresh version must outrank stale near-tie (fresh={fresh_rank}, stale={stale_rank})"
        );
        assert_eq!(results[0].id, "v-fresh");
    }

    /// Freshness breaks ties; it must not override semantics. Old content
    /// that is clearly the better semantic match still wins.
    #[test]
    fn freshness_does_not_override_clear_semantic_gap() {
        let p = profile();
        let candidates = vec![
            code_candidate("noise", 0.30, "a.rs"),
            memory_candidate("old-relevant", 0.85, 300.0),
            memory_candidate("new-irrelevant", 0.45, 1.0),
        ];
        let results = rerank(&candidates, &p);
        assert_eq!(results[0].id, "old-relevant");
    }

    #[test]
    fn diversity_limits_per_file() {
        let p = profile();
        let candidates = vec![
            code_candidate("a:1:10", 0.70, "same.rs"),
            code_candidate("a:11:20", 0.65, "same.rs"),
            code_candidate("a:21:30", 0.60, "same.rs"),
            code_candidate("b:1:10", 0.55, "other.rs"),
        ];
        let results = rerank(&candidates, &p);
        let same_count = results
            .iter()
            .filter(|r| r.source_file == "same.rs")
            .count();
        assert!(
            same_count <= 2,
            "should limit to 2 per file, got {same_count}"
        );
        // other.rs should also be present
        assert!(results.iter().any(|r| r.source_file == "other.rs"));
    }

    #[test]
    fn calibrate_low_anisotropy() {
        // Great embedding space: low aniso, wide range
        let p = RecallProfile::calibrate(0.15, 1.2, 200);
        assert!(
            p.semantic_confidence > 0.7,
            "should trust cosine: {}",
            p.semantic_confidence
        );
        assert!(
            p.similarity_floor < 0.25,
            "floor should be lenient: {}",
            p.similarity_floor
        );
    }

    #[test]
    fn calibrate_high_anisotropy() {
        // Collapsed embedding space: high aniso, narrow range
        let p = RecallProfile::calibrate(0.55, 0.5, 200);
        assert!(
            p.semantic_confidence < 0.5,
            "should distrust cosine: {}",
            p.semantic_confidence
        );
        assert!(
            p.similarity_floor > 0.25,
            "floor should be strict: {}",
            p.similarity_floor
        );
    }

    #[test]
    fn profile_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let p = RecallProfile::calibrate(0.36, 1.06, 100);
        p.save(dir.path()).unwrap();
        let loaded = RecallProfile::load(dir.path()).unwrap();
        assert!((loaded.anisotropy - p.anisotropy).abs() < 1e-6);
        assert!((loaded.similarity_floor - p.similarity_floor).abs() < 1e-6);
    }
}
