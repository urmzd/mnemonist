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
///
/// Percentiles and averages are the **median across `n_runs` repeated passes**
/// over the full query set, taken after a warmup pass — not a single cold run.
#[derive(Debug, Clone, Serialize)]
pub struct ScalePoint {
    pub n_documents: usize,
    pub index_build_time_ms: u64,
    pub query_latency_p50_us: f64,
    pub query_latency_p95_us: f64,
    pub query_latency_p99_us: f64,
    pub avg_query_time_us: f64,
    pub n_queries: usize,
    pub n_runs: usize,
    pub n_warmup_queries: usize,
}

/// Queries issued before measurement begins, to populate caches and branch
/// predictors so percentiles reflect steady state rather than cold start.
const WARMUP_QUERIES: usize = 50;
/// Measured passes over the query set per scale point; reported percentiles
/// are the median across passes.
const MEASUREMENT_RUNS: usize = 5;

/// Run Experiment 2: latency at scale.
///
/// Embeds all sessions and queries, then builds an HNSW index at each scale
/// size and measures build time plus steady-state query latency percentiles
/// (warmup pass, then median of repeated measurement passes).
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

        let n_warmup = WARMUP_QUERIES.min(query_embeddings.len());
        for q in query_embeddings.iter().take(n_warmup) {
            index
                .search(q, 5)
                .map_err(|e| EvalError::Other(e.to_string()))?;
        }

        let mut p50_runs = Vec::with_capacity(MEASUREMENT_RUNS);
        let mut p95_runs = Vec::with_capacity(MEASUREMENT_RUNS);
        let mut p99_runs = Vec::with_capacity(MEASUREMENT_RUNS);
        let mut avg_runs = Vec::with_capacity(MEASUREMENT_RUNS);

        for _ in 0..MEASUREMENT_RUNS {
            let mut latencies_us: Vec<f64> = Vec::with_capacity(query_embeddings.len());

            for q in &query_embeddings {
                let start = Instant::now();
                index
                    .search(q, 5)
                    .map_err(|e| EvalError::Other(e.to_string()))?;
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

            p50_runs.push(percentile(50.0));
            p95_runs.push(percentile(95.0));
            p99_runs.push(percentile(99.0));
            avg_runs.push(avg);
        }

        scale_points.push(ScalePoint {
            n_documents: n,
            index_build_time_ms: build_time.as_millis() as u64,
            query_latency_p50_us: median(&mut p50_runs),
            query_latency_p95_us: median(&mut p95_runs),
            query_latency_p99_us: median(&mut p99_runs),
            avg_query_time_us: median(&mut avg_runs),
            n_queries: query_embeddings.len(),
            n_runs: MEASUREMENT_RUNS,
            n_warmup_queries: n_warmup,
        });
    }

    Ok(LatencyResult { scale_points })
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    }
}

type SessionEmbeddings = Vec<(String, Vec<f32>)>;
type QueryEmbeddings = Vec<Vec<f32>>;

/// Embed all sessions and queries from the dataset.
fn embed_dataset(
    dataset: &LongMemEvalDataset,
    embedder: &dyn Embedder,
) -> Result<(SessionEmbeddings, QueryEmbeddings), EvalError> {
    const BATCH_SIZE: usize = 256;

    let total_sessions = dataset.sessions.len();
    eprintln!("  Embedding {total_sessions} sessions...");
    // Sorted so `take(n)` subsets are the same documents on every run.
    let mut session_ids: Vec<String> = dataset.sessions.keys().cloned().collect();
    session_ids.sort_unstable();
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
