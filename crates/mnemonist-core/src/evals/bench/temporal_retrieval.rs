//! Experiment 4: Temporal Staleness Disambiguation
//!
//! The scenario freshness decay exists for: the same content lives in the
//! index in several versions over time (a codebase that changes often,
//! superseded notes), and the embedder cannot reliably separate them. Between
//! near-tie versions, the *fresh* one should rank first.
//!
//! Construct: for each sampled topic (a LongMemEval session with a single
//! gold query), index three versions — the original ("fresh", age 1 day) and
//! two deterministically drifted variants (words dropped, ages 90 and 180
//! days). Query with the topic's question and score whether the fresh version
//! outranks both stale ones among retrieved gold versions.
//!
//! Both arms run the identical retrieve-20 → `rerank()` → score path; the
//! control assigns every version the same age, so the only difference is the
//! freshness signal itself. Paired exact McNemar on per-query outcomes.
//!
//! This replaces the earlier Hebbian-reinforcement experiment, which measured
//! +0.0 pp (0/0 discordant pairs, p = 1.0): access-count reinforcement never
//! changed a retrieval outcome and was removed from ranking.

use std::collections::HashMap;

use serde::Serialize;

use crate::ann::AnnIndex;
use crate::ann::hnsw::{HnswConfig, HnswIndex};
use crate::embed::Embedder;
use crate::rerank::{Candidate, MemorySignals, RecallProfile, rerank};

use crate::evals::EvalError;
use crate::evals::longmemeval::LongMemEvalDataset;
use crate::evals::search::{discordant_pairs, mcnemar_exact_p};

/// Ages (days) assigned to the three versions of each topic.
const FRESH_AGE: f64 = 1.0;
const STALE_AGES: [f64; 2] = [90.0, 180.0];

/// Max topics sampled from the dataset (keeps the experiment fast on CPU).
const MAX_TOPICS: usize = 150;

/// Results from the staleness disambiguation experiment.
#[derive(Debug, Clone, Serialize)]
pub struct TemporalResult {
    /// Fraction of queries where the fresh version outranks both stale
    /// versions, with real version ages (freshness signal active).
    pub fresh_first_with_freshness: f64,
    /// Same path, all versions assigned equal age — cosine order decides.
    pub fresh_first_without_freshness: f64,
    /// `with - without`.
    pub delta: f64,
    pub n_eval_queries: usize,
    pub n_versions_per_topic: usize,
    pub stale_ages_days: [f64; 2],
    /// (with hit ∧ without miss, with miss ∧ without hit).
    pub n_discordant: [usize; 2],
    /// Exact two-sided McNemar p-value on the paired per-query outcomes.
    pub mcnemar_p: f64,
    pub n_documents: usize,
}

/// Deterministic content drift: drop every `drop_every`-th word. Simulates an
/// older draft of the same content — semantically near-identical, not
/// embedding-identical.
fn drift(text: &str, drop_every: usize) -> String {
    text.split_whitespace()
        .enumerate()
        .filter(|(i, _)| (i + 1) % drop_every != 0)
        .map(|(_, w)| w)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Run Experiment 4: staleness disambiguation via freshness decay.
pub fn run(
    dataset: &LongMemEvalDataset,
    embedder: &dyn Embedder,
) -> Result<TemporalResult, EvalError> {
    // Topics: queries with exactly one gold session, deterministic order.
    let topics: Vec<(&str, &str)> = dataset
        .queries
        .iter()
        .filter(|q| q.gold_session_ids.len() == 1)
        .filter_map(|q| {
            let sid = q.gold_session_ids[0].as_str();
            dataset
                .sessions
                .get(sid)
                .map(|text| (q.question.as_str(), text.as_str()))
        })
        .take(MAX_TOPICS)
        .collect();

    if topics.len() < 5 {
        return Err(EvalError::InsufficientData {
            min: 5,
            got: topics.len(),
        });
    }

    // Build the versioned corpus: topic i → fresh + two drifted stale versions.
    let mut ids: Vec<String> = Vec::with_capacity(topics.len() * 3);
    let mut texts: Vec<String> = Vec::with_capacity(topics.len() * 3);
    let mut ages: HashMap<String, f64> = HashMap::new();
    for (i, (_, text)) in topics.iter().enumerate() {
        for (suffix, age, body) in [
            ("fresh", FRESH_AGE, text.to_string()),
            ("mid", STALE_AGES[0], drift(text, 20)),
            ("old", STALE_AGES[1], drift(text, 10)),
        ] {
            let id = format!("t{i}#{suffix}");
            ages.insert(id.clone(), age);
            ids.push(id);
            texts.push(body);
        }
    }

    eprintln!(
        "  Embedding {} versioned documents ({} topics x 3)...",
        ids.len(),
        topics.len()
    );
    let doc_refs: Vec<&str> = texts.iter().map(|t| t.as_str()).collect();
    let doc_embeddings = embed_batched(embedder, &doc_refs)?;

    let dim = doc_embeddings
        .first()
        .ok_or(EvalError::InsufficientData { min: 1, got: 0 })?
        .len();

    let mut index = HnswIndex::new(
        dim,
        HnswConfig {
            ef_search: 100,
            ..HnswConfig::default()
        },
    );
    for (id, emb) in ids.iter().zip(&doc_embeddings) {
        index
            .insert(id, emb)
            .map_err(|e| EvalError::Other(e.to_string()))?;
    }

    eprintln!("  Embedding {} queries...", topics.len());
    let query_refs: Vec<&str> = topics.iter().map(|(q, _)| *q).collect();
    let query_embeddings = embed_batched(embedder, &query_refs)?;

    let profile = RecallProfile::uncalibrated();

    let mut with_hits = Vec::with_capacity(topics.len());
    let mut without_hits = Vec::with_capacity(topics.len());

    for (i, q_emb) in query_embeddings.iter().enumerate() {
        let hits = index.search(q_emb, 20).unwrap_or_default();

        let to_candidates = |real_ages: bool| -> Vec<Candidate> {
            hits.iter()
                .map(|h| Candidate {
                    id: h.id.clone(),
                    cosine_score: h.score,
                    memory_signals: Some(MemorySignals {
                        strength: 1.0,
                        age_days: if real_ages {
                            *ages.get(&h.id).unwrap_or(&FRESH_AGE)
                        } else {
                            FRESH_AGE
                        },
                        source: None,
                        ref_count: 0,
                    }),
                    // Distinct per version so rerank's per-file diversity
                    // cap never drops a version.
                    source_file: h.id.clone(),
                })
                .collect()
        };

        let fresh_id = format!("t{i}#fresh");
        let gold_prefix = format!("t{i}#");

        let fresh_first = |candidates: Vec<Candidate>| -> bool {
            let ranked = rerank(&candidates, &profile);
            ranked
                .iter()
                .find(|r| r.id.starts_with(&gold_prefix))
                .is_some_and(|first_gold| first_gold.id == fresh_id)
        };

        with_hits.push(fresh_first(to_candidates(true)));
        without_hits.push(fresh_first(to_candidates(false)));
    }

    let rate = |hits: &[bool]| -> f64 {
        if hits.is_empty() {
            return 0.0;
        }
        hits.iter().filter(|h| **h).count() as f64 / hits.len() as f64
    };

    let with_rate = rate(&with_hits);
    let without_rate = rate(&without_hits);
    let (b, c) = discordant_pairs(&with_hits, &without_hits);

    Ok(TemporalResult {
        fresh_first_with_freshness: with_rate,
        fresh_first_without_freshness: without_rate,
        delta: with_rate - without_rate,
        n_eval_queries: topics.len(),
        n_versions_per_topic: 3,
        stale_ages_days: STALE_AGES,
        n_discordant: [b, c],
        mcnemar_p: mcnemar_exact_p(b, c),
        n_documents: ids.len(),
    })
}

fn embed_batched(embedder: &dyn Embedder, texts: &[&str]) -> Result<Vec<Vec<f32>>, EvalError> {
    const BATCH_SIZE: usize = 256;
    let mut out = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(BATCH_SIZE) {
        let batch = embedder
            .embed_batch(chunk)
            .map_err(|e| EvalError::Other(e.to_string()))?;
        out.extend(batch);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drift_drops_words_deterministically() {
        let text = "a b c d e f g h i j";
        let d = drift(text, 5);
        // Every 5th word (e, j) dropped
        assert_eq!(d, "a b c d f g h i");
        assert_eq!(drift(text, 5), d, "must be deterministic");
    }

    #[test]
    fn drift_preserves_most_content() {
        let text = "word ".repeat(100);
        let d = drift(&text, 10);
        let kept = d.split_whitespace().count();
        assert_eq!(kept, 90);
    }
}
