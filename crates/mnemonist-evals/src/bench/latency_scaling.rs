//! Experiment 2: Latency Scaling
//!
//! Builds HNSW indices at increasing document counts and measures index build
//! time plus p50/p95/p99 query latency.

use std::time::Instant;

use serde::Serialize;

use mnemonist_index::AnnIndex;
use mnemonist_index::hnsw::{HnswConfig, HnswIndex};

use crate::EvalError;

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
pub fn run(
    embeddings: &[(String, Vec<f32>)],
    query_embeddings: &[Vec<f32>],
    scale_sizes: &[usize],
) -> Result<LatencyResult, EvalError> {
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

        for q in query_embeddings {
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
