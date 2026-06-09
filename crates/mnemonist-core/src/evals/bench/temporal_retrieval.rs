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
use crate::evals::search::{discordant_pairs, mcnemar_exact_p};

/// Results from the temporal retrieval experiment.
#[derive(Debug, Clone, Serialize)]
pub struct TemporalResult {
    pub baseline_recall_any_at_5: f64,
    /// Same retrieve-20/rerank/take-5 path as `reinforced`, but with zero
    /// access counts — isolates the rerank pipeline from the reinforcement signal.
    pub control_recall_any_at_5: f64,
    pub reinforced_recall_any_at_5: f64,
    /// `reinforced - control`. Both arms share the retrieve-20/rerank/take-5
    /// path, so this isolates the reinforcement signal from the reranker's own
    /// contribution (which `reinforced - baseline` would confound).
    pub recall_delta: f64,
    /// Held-out queries every arm is scored on (the shared denominator).
    pub n_eval_queries: usize,
    /// Discordant pairs between the reinforced and control arms:
    /// (reinforced hit ∧ control miss, reinforced miss ∧ control hit).
    pub n_discordant_reinforced_vs_control: [usize; 2],
    /// Exact two-sided McNemar p-value, reinforced vs control on paired
    /// per-query hits. The control shares the rerank path, so this tests the
    /// reinforcement signal itself.
    pub mcnemar_p_reinforced_vs_control: f64,
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
/// 5. Compares reinforced recall vs a no-reinforcement control (identical rerank
///    path, zero access counts) and vs the pure-cosine baseline — all three on
///    the same held-out eval slice
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

    // Split queries: first half trains access patterns, second half evaluates
    let train_end = query_embeddings.len() / 2;
    let eval_queries = &query_embeddings[train_end..];
    let eval_gold = &query_gold[train_end..];

    // Baseline: pure cosine retrieval on the same held-out eval queries
    let baseline_recall = compute_recall_any(&index, eval_queries, eval_gold, 5);

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

    // Control: identical retrieve-20/rerank/take-5 path with zero access counts
    let control_hits =
        compute_reranked_hits(&index, eval_queries, eval_gold, &HashMap::new(), lambda);
    let control_recall = recall_from_hits(&control_hits);

    // Evaluate: temporal-reranked retrieval on held-out queries
    let reinforced_hits =
        compute_reranked_hits(&index, eval_queries, eval_gold, &access_counts, lambda);
    let reinforced_recall = recall_from_hits(&reinforced_hits);

    // Paired test on per-query hits: does reinforcement change outcomes beyond
    // what the shared rerank path produces on its own?
    let (b, c) = discordant_pairs(&reinforced_hits, &control_hits);
    let mcnemar_p = mcnemar_exact_p(b, c);

    Ok(TemporalResult {
        baseline_recall_any_at_5: baseline_recall,
        control_recall_any_at_5: control_recall,
        reinforced_recall_any_at_5: reinforced_recall,
        recall_delta: reinforced_recall - control_recall,
        n_eval_queries: eval_queries.len(),
        n_discordant_reinforced_vs_control: [b, c],
        mcnemar_p_reinforced_vs_control: mcnemar_p,
        n_consolidation_cycles: n_cycles,
        n_queries_per_cycle: train_end,
        n_documents: embeddings.len(),
    })
}

/// Fraction of hits in a per-query hit vector (0.0 for an empty vector).
fn recall_from_hits(hits: &[bool]) -> f64 {
    if hits.is_empty() {
        return 0.0;
    }
    hits.iter().filter(|h| **h).count() as f64 / hits.len() as f64
}

/// Per-query hits through the retrieve-20 / temporal-rerank / take-5 path.
fn compute_reranked_hits(
    index: &HnswIndex,
    query_embeddings: &[Vec<f32>],
    query_gold: &[Vec<String>],
    access_counts: &HashMap<String, u32>,
    lambda: f64,
) -> Vec<bool> {
    let mut per_query = Vec::with_capacity(query_embeddings.len());
    for (q_emb, gold_ids) in query_embeddings.iter().zip(query_gold) {
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
        per_query.push(gold.iter().any(|g| top_5.contains(g)));
    }
    per_query
}

/// recall_any@5 through the retrieve-20 / temporal-rerank / take-5 path.
#[cfg(test)]
fn compute_reranked_recall_any(
    index: &HnswIndex,
    query_embeddings: &[Vec<f32>],
    query_gold: &[Vec<String>],
    access_counts: &HashMap<String, u32>,
    lambda: f64,
) -> f64 {
    recall_from_hits(&compute_reranked_hits(
        index,
        query_embeddings,
        query_gold,
        access_counts,
        lambda,
    ))
}

/// Pure-cosine recall_any@k on pre-built HNSW index.
fn compute_recall_any(
    index: &HnswIndex,
    query_embeddings: &[Vec<f32>],
    query_gold: &[Vec<String>],
    k: usize,
) -> f64 {
    let mut per_query = Vec::with_capacity(query_embeddings.len());
    for (q_emb, gold_ids) in query_embeddings.iter().zip(query_gold) {
        let gold: HashSet<&str> = gold_ids.iter().map(|s| s.as_str()).collect();
        let results = index.search(q_emb, k).unwrap_or_default();
        let top_k: HashSet<&str> = results.iter().map(|h| h.id.as_str()).collect();
        per_query.push(gold.iter().any(|g| top_k.contains(g)));
    }
    recall_from_hits(&per_query)
}

type SessionEmbeddings = Vec<(String, Vec<f32>)>;
type QueryEmbeddings = Vec<Vec<f32>>;
type QueryGold = Vec<Vec<String>>;

/// Embed all sessions and queries from the dataset.
fn embed_dataset(
    dataset: &LongMemEvalDataset,
    embedder: &dyn Embedder,
) -> Result<(SessionEmbeddings, QueryEmbeddings, QueryGold), EvalError> {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// With zero access counts the temporal blend is order-preserving, so the
    /// no-reinforcement control must equal the pure-cosine baseline on the
    /// same query population. Guards against the arms diverging for any
    /// reason other than the reinforcement signal.
    #[test]
    fn control_with_no_access_counts_matches_baseline() {
        let n_docs = 8;
        let mut index = HnswIndex::new(
            4,
            HnswConfig {
                ef_search: 100,
                ..HnswConfig::default()
            },
        );
        let mut doc_vecs = Vec::new();
        for j in 0..n_docs {
            let theta = j as f32 * 0.15;
            let v = vec![theta.cos(), theta.sin(), 0.0, 0.0];
            index.insert(&format!("doc{j}"), &v).unwrap();
            doc_vecs.push(v);
        }

        let mut queries = Vec::new();
        let mut gold = Vec::new();
        for (i, v) in doc_vecs.iter().enumerate() {
            queries.push(v.clone());
            // Even queries are answerable (gold = nearest doc); odd queries
            // are not (gold = farthest doc, always outside the top 5 of 8).
            let g = if i % 2 == 0 {
                format!("doc{i}")
            } else if i < n_docs / 2 {
                format!("doc{}", n_docs - 1)
            } else {
                "doc0".to_string()
            };
            gold.push(vec![g]);
        }

        let baseline = compute_recall_any(&index, &queries, &gold, 5);
        let control = compute_reranked_recall_any(&index, &queries, &gold, &HashMap::new(), 0.15);
        assert_eq!(baseline, control);
        assert!(baseline > 0.0 && baseline < 1.0);
    }
}
