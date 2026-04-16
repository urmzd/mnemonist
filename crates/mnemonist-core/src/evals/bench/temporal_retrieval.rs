//! Experiment 4: Temporal Retrieval
//!
//! Demonstrates retrieval quality that improves over time through Hebbian
//! access-pattern reinforcement — something static vector stores cannot do.

use std::collections::{HashMap, HashSet};

use serde::Serialize;

use crate::MemoryType;
use crate::ann::AnnIndex;
use crate::ann::hnsw::{HnswConfig, HnswIndex};
use crate::embed::Embedder;
use crate::temporal::{blend, temporal_score};

use crate::evals::EvalError;
use crate::evals::longmemeval::LongMemEvalDataset;

/// Results from the temporal retrieval experiment.
#[derive(Debug, Clone, Serialize)]
pub struct TemporalResult {
    pub baseline_recall_any_at_5: f64,
    pub reinforced_recall_any_at_5: f64,
    pub recall_delta: f64,
    pub n_consolidation_cycles: usize,
    pub n_queries_per_cycle: usize,
    pub n_documents: usize,
}

/// Run Experiment 4: temporal retrieval with Hebbian reinforcement.
///
/// Embeds all sessions and queries, then:
/// 1. Builds HNSW index from all embeddings
/// 2. Splits queries: first half for "training" access patterns, second half for eval
/// 3. Simulates repeated query cycles on the training set, tracking access counts
/// 4. Eval: reranks with temporal signals (access frequency -> strength boost)
/// 5. Compares reinforced recall vs pure-cosine baseline
pub fn run(
    dataset: &LongMemEvalDataset,
    embedder: &dyn Embedder,
    n_cycles: usize,
) -> Result<TemporalResult, EvalError> {
    let (embeddings, query_embeddings, query_gold) = embed_dataset(dataset, embedder)?;

    let dim = embeddings
        .first()
        .ok_or(EvalError::InsufficientData { min: 1, got: 0 })?
        .1
        .len();

    let mut index = HnswIndex::new(
        dim,
        HnswConfig {
            ef_search: 100,
            ..HnswConfig::default()
        },
    );
    for (id, emb) in &embeddings {
        index
            .insert(id, emb)
            .map_err(|e| EvalError::Other(e.to_string()))?;
    }

    // Baseline: pure cosine retrieval on ALL queries
    let baseline_recall = compute_recall_any(&index, &query_embeddings, &query_gold, 5);

    // Split queries: first half trains access patterns, second half evaluates
    let train_end = query_embeddings.len() / 2;
    let eval_queries = &query_embeddings[train_end..];
    let eval_gold = &query_gold[train_end..];

    // Simulate access patterns over multiple consolidation cycles
    let mut access_counts: HashMap<String, u32> = HashMap::new();
    let lambda = 0.15;

    for cycle in 0..n_cycles {
        let _recency_base = ((n_cycles - cycle) * 5) as f64;

        for (q_emb, gold_ids) in query_embeddings[..train_end]
            .iter()
            .zip(&query_gold[..train_end])
        {
            let hits = index.search(q_emb, 10).unwrap_or_default();

            let gold_set: HashSet<&str> = gold_ids.iter().map(|s| s.as_str()).collect();
            for hit in &hits {
                if gold_set.contains(hit.id.as_str()) {
                    *access_counts.entry(hit.id.clone()).or_insert(0) += 1;
                }
            }
        }
    }

    // Evaluate: temporal-reranked retrieval on held-out queries
    let mut reinforced_hits = 0;
    let eval_count = eval_queries.len();

    for (q_emb, gold_ids) in eval_queries.iter().zip(eval_gold) {
        let gold: HashSet<&str> = gold_ids.iter().map(|s| s.as_str()).collect();

        let hits = index.search(q_emb, 20).unwrap_or_default();

        let mut reranked: Vec<(String, f64)> = hits
            .iter()
            .map(|hit| {
                let ac = *access_counts.get(&hit.id).unwrap_or(&0);
                let ts = temporal_score(1.0, 30.0, ac, MemoryType::Reference);
                let final_score = blend(hit.score as f64, ts, lambda);
                (hit.id.clone(), final_score)
            })
            .collect();

        reranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_5: HashSet<&str> = reranked.iter().take(5).map(|(id, _)| id.as_str()).collect();
        if gold.iter().any(|g| top_5.contains(g)) {
            reinforced_hits += 1;
        }
    }

    let reinforced_recall = if eval_count == 0 {
        0.0
    } else {
        reinforced_hits as f64 / eval_count as f64
    };

    Ok(TemporalResult {
        baseline_recall_any_at_5: baseline_recall,
        reinforced_recall_any_at_5: reinforced_recall,
        recall_delta: reinforced_recall - baseline_recall,
        n_consolidation_cycles: n_cycles,
        n_queries_per_cycle: train_end,
        n_documents: embeddings.len(),
    })
}

/// Pure-cosine recall_any@k on pre-built HNSW index.
fn compute_recall_any(
    index: &HnswIndex,
    query_embeddings: &[Vec<f32>],
    query_gold: &[Vec<String>],
    k: usize,
) -> f64 {
    if query_embeddings.is_empty() {
        return 0.0;
    }

    let mut hits = 0;
    for (q_emb, gold_ids) in query_embeddings.iter().zip(query_gold) {
        let gold: HashSet<&str> = gold_ids.iter().map(|s| s.as_str()).collect();
        let results = index.search(q_emb, k).unwrap_or_default();
        let top_k: HashSet<&str> = results.iter().map(|h| h.id.as_str()).collect();
        if gold.iter().any(|g| top_k.contains(g)) {
            hits += 1;
        }
    }
    hits as f64 / query_embeddings.len() as f64
}

/// Embed all sessions and queries from the dataset.
fn embed_dataset(
    dataset: &LongMemEvalDataset,
    embedder: &dyn Embedder,
) -> Result<(Vec<(String, Vec<f32>)>, Vec<Vec<f32>>, Vec<Vec<String>>), EvalError> {
    const BATCH_SIZE: usize = 256;

    let total_sessions = dataset.sessions.len();
    eprintln!("  Embedding {total_sessions} sessions...");
    let session_ids: Vec<String> = dataset.sessions.keys().cloned().collect();
    let session_texts: Vec<&str> = session_ids
        .iter()
        .map(|id| dataset.sessions[id].as_str())
        .collect();

    let mut session_embeddings: Vec<Vec<f32>> = Vec::with_capacity(total_sessions);
    for chunk in session_texts.chunks(BATCH_SIZE) {
        let batch = embedder
            .embed_batch(chunk)
            .map_err(|e| EvalError::Other(e.to_string()))?;
        session_embeddings.extend(batch);
    }

    let embeddings: Vec<(String, Vec<f32>)> =
        session_ids.into_iter().zip(session_embeddings).collect();

    let total_queries = dataset.queries.len();
    eprintln!("  Embedding {total_queries} queries...");
    let query_texts: Vec<&str> = dataset
        .queries
        .iter()
        .map(|q| q.question.as_str())
        .collect();

    let mut query_embeddings: Vec<Vec<f32>> = Vec::with_capacity(total_queries);
    for chunk in query_texts.chunks(BATCH_SIZE) {
        let batch = embedder
            .embed_batch(chunk)
            .map_err(|e| EvalError::Other(e.to_string()))?;
        query_embeddings.extend(batch);
    }

    let query_gold: Vec<Vec<String>> = dataset
        .queries
        .iter()
        .map(|q| q.gold_session_ids.clone())
        .collect();

    Ok((embeddings, query_embeddings, query_gold))
}
