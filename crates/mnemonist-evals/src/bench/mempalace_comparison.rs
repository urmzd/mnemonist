//! Experiment 5: MemPalace Comparison (Retrieval Only — NOT QA)
//!
//! Apples-to-apples retrieval comparison with MemPalace's methodology:
//! per-question haystack, raw vector retrieval with k=5.
//!
//! IMPORTANT: MemPalace reported recall@5 as a "LongMemEval score",
//! but this is NOT a LongMemEval QA accuracy score. It is raw session retrieval
//! recall — whether the correct session appears in the top-5 vector search results.
//! The actual LongMemEval benchmark measures end-to-end QA accuracy (see Experiment 6).

use std::collections::HashSet;

use serde::Serialize;

use mnemonist_core::embed::Embedder;
use mnemonist_index::AnnIndex;
use mnemonist_index::hnsw::{HnswConfig, HnswIndex};

use crate::EvalError;
use crate::longmemeval::LongMemEvalDataset;

/// Results from the MemPalace retrieval comparison.
#[derive(Debug, Clone, Serialize)]
pub struct MemPalaceComparisonResult {
    pub mnemonist_recall_any_at_5: f64,
    pub mnemonist_recall_all_at_5: f64,
    pub n_sessions: usize,
    pub n_queries: usize,
    pub avg_haystack_size: f64,
    pub note: String,
}

/// Run Experiment 5: MemPalace apples-to-apples retrieval comparison.
///
/// Uses per-question haystacks with k=5, ef_search=100.
pub fn run(
    dataset: &LongMemEvalDataset,
    embedder: &dyn Embedder,
) -> Result<MemPalaceComparisonResult, EvalError> {
    let dim = embedder
        .dimension()
        .map_err(|e| EvalError::Other(e.to_string()))?;

    let total = dataset.queries.len();
    if total == 0 {
        return Err(EvalError::InsufficientData { min: 1, got: 0 });
    }

    let mut any_hits = 0usize;
    let mut all_hits = 0usize;
    let mut total_haystack = 0usize;

    for (i, q) in dataset.queries.iter().enumerate() {
        let haystack_ids: Vec<&str> = if !q.haystack_session_ids.is_empty() {
            q.haystack_session_ids.iter().map(|s| s.as_str()).collect()
        } else {
            dataset.sessions.keys().map(|s| s.as_str()).collect()
        };

        let haystack_texts: Vec<&str> = haystack_ids
            .iter()
            .filter_map(|id| dataset.sessions.get(*id).map(|s| s.as_str()))
            .collect();

        let valid_ids: Vec<&str> = haystack_ids
            .iter()
            .filter(|id| dataset.sessions.contains_key(**id))
            .copied()
            .collect();

        total_haystack += valid_ids.len();

        if haystack_texts.is_empty() {
            continue;
        }

        let session_embeddings = embedder
            .embed_batch(&haystack_texts)
            .map_err(|e| EvalError::Other(e.to_string()))?;

        let mut index = HnswIndex::new(
            dim,
            HnswConfig {
                ef_search: 100,
                ..HnswConfig::default()
            },
        );

        for (j, emb) in session_embeddings.iter().enumerate() {
            index
                .insert(valid_ids[j], emb)
                .map_err(|e| EvalError::Other(e.to_string()))?;
        }

        let q_embedding = embedder
            .embed(&q.question)
            .map_err(|e| EvalError::Other(e.to_string()))?;

        let hits = index
            .search(&q_embedding, 5)
            .map_err(|e| EvalError::Other(e.to_string()))?;

        let top_k: HashSet<&str> = hits.iter().map(|h| h.id.as_str()).collect();
        let gold: HashSet<&str> = q.gold_session_ids.iter().map(|s| s.as_str()).collect();

        if gold.iter().any(|g| top_k.contains(g)) {
            any_hits += 1;
        }
        if gold.iter().all(|g| top_k.contains(g)) {
            all_hits += 1;
        }

        if (i + 1) % 50 == 0 || i + 1 == total {
            eprintln!("  [{}/{total}]", i + 1);
        }
    }

    Ok(MemPalaceComparisonResult {
        mnemonist_recall_any_at_5: any_hits as f64 / total as f64,
        mnemonist_recall_all_at_5: all_hits as f64 / total as f64,
        n_sessions: dataset.sessions.len(),
        n_queries: total,
        avg_haystack_size: total_haystack as f64 / total as f64,
        note: "MemPalace reported 82.8% recall@5 as a 'LongMemEval score'. \
               This is NOT a QA accuracy score — it is raw vector retrieval recall. \
               Their 96.6% number was never reproduced independently. \
               See Experiment 6 for actual LongMemEval QA evaluation."
            .to_string(),
    })
}
