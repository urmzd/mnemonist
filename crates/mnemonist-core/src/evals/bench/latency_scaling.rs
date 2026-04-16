//! Experiment 2: Latency Scaling
//!
//! Builds HNSW indices at increasing document counts and measures index build
//! time plus p50/p95/p99 query latency.

use std::time::Instant;

use serde::Serialize;

use crate::ann::AnnIndex;
use crate::ann::hnsw::{HnswConfig, HnswIndex};
use crate::embed::Embedder;

use crate::evals::EvalError;
use crate::evals::longmemeval::LongMemEvalDataset;

/// Results from the latency scaling experiment.
#[derive(Debug, Clone, Serialize)]
pub struct LatencyResult {
    pub scale_points: Vec<ScalePoint>,
}

/// Measurements at a single scale point (document count).
#[derive(Debug, Clone, Serialize)]
pub struct ScalePoint {
    pub n_documents: usize,
    pub index_build_time_ms: u64,
    pub query_latency_p50_us: f64,
    pub query_latency_p95_us: f64,
    pub query_latency_p99_us: f64,
    pub avg_query_time_us: f64,
    pub n_queries: usize,
}

/// Run Experiment 2: latency at scale.
///
/// Embeds all sessions and queries, then builds HNSW indices at each scale size
/// and measures build time and query latency percentiles.
pub fn run(
    dataset: &LongMemEvalDataset,
    embedder: &dyn Embedder,
    scale_sizes: &[usize],
) -> Result<LatencyResult, EvalError> {
    // Embed sessions
    let (embeddings, query_embeddings) = embed_dataset(dataset, embedder)?;

    let dim = embeddings
        .first()
        .ok_or(EvalError::InsufficientData { min: 1, got: 0 })?
        .1
        .len();

    let mut scale_points = Vec::new();

    for &n in scale_sizes {
        let n = n.min(embeddings.len());

        let build_start = Instant::now();
        let mut index = HnswIndex::new(
            dim,
            HnswConfig {
                ef_search: 50,
                ..HnswConfig::default()
            },
        );

        for (id, emb) in embeddings.iter().take(n) {
            index
                .insert(id, emb)
                .map_err(|e| EvalError::Other(e.to_string()))?;
        }
        let build_time = build_start.elapsed();

        let mut latencies_us: Vec<f64> = Vec::with_capacity(query_embeddings.len());

        for q in &query_embeddings {
            let start = Instant::now();
            let _ = index.search(q, 5);
            latencies_us.push(start.elapsed().as_nanos() as f64 / 1000.0);
        }

        latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let percentile = |p: f64| -> f64 {
            if latencies_us.is_empty() {
                return 0.0;
            }
            let idx = ((p / 100.0) * (latencies_us.len() - 1) as f64).round() as usize;
            latencies_us[idx.min(latencies_us.len() - 1)]
        };

        let avg = if latencies_us.is_empty() {
            0.0
        } else {
            latencies_us.iter().sum::<f64>() / latencies_us.len() as f64
        };

        scale_points.push(ScalePoint {
            n_documents: n,
            index_build_time_ms: build_time.as_millis() as u64,
            query_latency_p50_us: percentile(50.0),
            query_latency_p95_us: percentile(95.0),
            query_latency_p99_us: percentile(99.0),
            avg_query_time_us: avg,
            n_queries: query_embeddings.len(),
        });
    }

    Ok(LatencyResult { scale_points })
}

/// Embed all sessions and queries from the dataset.
fn embed_dataset(
    dataset: &LongMemEvalDataset,
    embedder: &dyn Embedder,
) -> Result<(Vec<(String, Vec<f32>)>, Vec<Vec<f32>>), EvalError> {
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

    Ok((embeddings, query_embeddings))
}
