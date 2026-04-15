//! Experiment 1: Vector Retrieval Quality
//!
//! Measures raw session retrieval recall — can the vector index find the right
//! session given a question embedding? Uses per-question haystacks (each question
//! searches only its own ~48 sessions, not all 19K).
//!
//! **This is NOT LongMemEval QA accuracy.** For actual end-to-end question
//! answering, see Experiment 6 (`qa.rs`).

use std::collections::HashMap;
use std::time::Instant;

use serde::Serialize;

use crate::embed::Embedder;
use crate::ann::AnnIndex;
use crate::ann::hnsw::{HnswConfig, HnswIndex};

use crate::evals::EvalError;
use crate::evals::longmemeval::LongMemEvalDataset;
use crate::evals::search::{QueryEval, evaluate_search, recall_all_at_k, recall_any_at_k};

/// Results from the vector retrieval experiment.
///
/// NOTE: This measures **session retrieval recall** (can the index find the right
/// session?), NOT LongMemEval QA accuracy.
#[derive(Debug, Clone, Serialize)]
pub struct RetrievalResult {
    pub recall_any_at_5: f64,
    pub recall_all_at_5: f64,
    pub recall_any_at_10: f64,
    pub recall_all_at_10: f64,
    pub mrr: f64,
    pub precision_at_5: f64,
    pub n_sessions: usize,
    pub n_queries: usize,
    pub avg_haystack_size: f64,
    pub embed_time_ms: u64,
    pub index_build_time_ms: u64,
    pub total_query_time_ms: u64,
    pub avg_query_time_us: f64,
}

/// Run Experiment 1: Vector retrieval quality (per-question haystacks).
///
/// For each question, builds an HNSW index from only that question's haystack
/// sessions (~48 on average), then queries it. This matches the LongMemEval
/// protocol where each question has its own distractor set.
pub fn run(
    dataset: &LongMemEvalDataset,
    embedder: &dyn Embedder,
) -> Result<RetrievalResult, EvalError> {
    let dim = embedder
        .dimension()
        .map_err(|e| EvalError::Other(e.to_string()))?;

    let total = dataset.queries.len();
    if total == 0 {
        return Err(EvalError::InsufficientData { min: 1, got: 0 });
    }

    let embed_start = Instant::now();
    let mut query_evals = Vec::with_capacity(total);
    let mut total_haystack = 0usize;
    let mut total_build_ms = 0u64;
    let mut total_query_us = 0u64;

    for (i, q) in dataset.queries.iter().enumerate() {
        // Collect this question's haystack sessions
        let haystack_ids: Vec<&str> = if !q.haystack_session_ids.is_empty() {
            q.haystack_session_ids.iter().map(|s| s.as_str()).collect()
        } else {
            // Fallback: use all sessions
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

        // Embed this question's haystack
        let session_embeddings = embedder
            .embed_batch(&haystack_texts)
            .map_err(|e| EvalError::Other(e.to_string()))?;

        // Build per-question HNSW
        let build_start = Instant::now();
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
        total_build_ms += build_start.elapsed().as_millis() as u64;

        // Embed and search
        let query_start = Instant::now();
        let q_embedding = embedder
            .embed(&q.question)
            .map_err(|e| EvalError::Other(e.to_string()))?;

        let hits = index
            .search(&q_embedding, 10)
            .map_err(|e| EvalError::Other(e.to_string()))?;
        total_query_us += query_start.elapsed().as_micros() as u64;

        let retrieved: Vec<String> = hits.into_iter().map(|h| h.id).collect();

        let mut judgments = HashMap::new();
        for gold_id in &q.gold_session_ids {
            judgments.insert(gold_id.clone(), 1);
        }

        query_evals.push(QueryEval {
            query_id: q.question.clone(),
            retrieved,
            judgments,
        });

        if (i + 1) % 50 == 0 || i + 1 == total {
            eprintln!("  [{}/{total}]", i + 1);
        }
    }

    let embed_time = embed_start.elapsed();
    let search_metrics = evaluate_search(&query_evals, 5);

    Ok(RetrievalResult {
        recall_any_at_5: recall_any_at_k(&query_evals, 5),
        recall_all_at_5: recall_all_at_k(&query_evals, 5),
        recall_any_at_10: recall_any_at_k(&query_evals, 10),
        recall_all_at_10: recall_all_at_k(&query_evals, 10),
        mrr: search_metrics.mrr,
        precision_at_5: search_metrics.precision_at_k,
        n_sessions: dataset.sessions.len(),
        n_queries: total,
        avg_haystack_size: total_haystack as f64 / total as f64,
        embed_time_ms: embed_time.as_millis() as u64,
        index_build_time_ms: total_build_ms,
        total_query_time_ms: total_query_us / 1000,
        avg_query_time_us: total_query_us as f64 / total as f64,
    })
}
