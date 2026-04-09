//! Benchmark suite: 4 experiments comparing mnemonist retrieval infrastructure.
//!
//! 1. **Retrieval quality** — LongMemEval recall\_any@5, recall\_all@5 with timing
//! 2. **Latency scaling** — index build + p50/p95/p99 query latency at 100–10k docs
//! 3. **Storage footprint** — raw vs TurboQuant compressed at 1–4 bits, with recall impact
//! 4. **Temporal retrieval** — dynamic recall improvement via Hebbian access patterns

use std::collections::HashMap;
use std::time::Instant;

use serde::Serialize;

use mnemonist_core::distance::cosine_similarity;
use mnemonist_core::embed::Embedder;
use mnemonist_index::AnnIndex;
use mnemonist_index::hnsw::{HnswConfig, HnswIndex};

use crate::longmemeval::LongMemEvalDataset;
use crate::search::{QueryEval, evaluate_search, recall_all_at_k, recall_any_at_k};

// ═══════════════════════════════════════════════════════════════════════════
// Experiment 1: LongMemEval Retrieval Quality
// ═══════════════════════════════════════════════════════════════════════════

/// Results from the LongMemEval retrieval experiment.
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
    pub embed_time_ms: u64,
    pub index_build_time_ms: u64,
    pub total_query_time_ms: u64,
    pub avg_query_time_us: f64,
}

/// Run Experiment 1: LongMemEval retrieval quality.
///
/// Embeds all sessions, builds an HNSW index, queries it, and computes
/// recall\_any@5, recall\_all@5, MRR, and precision@5 — the same methodology
/// MemPalace uses, but with mnemonist's native HNSW instead of ChromaDB.
pub fn run_retrieval(
    dataset: &LongMemEvalDataset,
    embedder: &dyn Embedder,
) -> Result<RetrievalResult, crate::EvalError> {
    let dim = embedder
        .dimension()
        .map_err(|e| crate::EvalError::Other(e.to_string()))?;

    // Embed all sessions
    let session_ids: Vec<&String> = dataset.sessions.keys().collect();
    let session_texts: Vec<&str> = session_ids
        .iter()
        .map(|id| dataset.sessions[*id].as_str())
        .collect();

    let embed_start = Instant::now();
    let session_embeddings = embedder
        .embed_batch(&session_texts)
        .map_err(|e| crate::EvalError::Other(e.to_string()))?;
    let embed_time = embed_start.elapsed();

    // Build HNSW index
    let build_start = Instant::now();
    let mut index = HnswIndex::new(
        dim,
        HnswConfig {
            ef_search: 100, // Higher ef for better recall in eval
            ..HnswConfig::default()
        },
    );

    for (i, embedding) in session_embeddings.iter().enumerate() {
        index
            .insert(session_ids[i], embedding)
            .map_err(|e| crate::EvalError::Other(e.to_string()))?;
    }
    let build_time = build_start.elapsed();

    // Embed queries and search
    let query_start = Instant::now();
    let mut query_evals = Vec::with_capacity(dataset.queries.len());

    for q in &dataset.queries {
        let q_embedding = embedder
            .embed(&q.question)
            .map_err(|e| crate::EvalError::Other(e.to_string()))?;

        let hits = index
            .search(&q_embedding, 10)
            .map_err(|e| crate::EvalError::Other(e.to_string()))?;

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
    }
    let query_time = query_start.elapsed();

    // Compute metrics
    let search_metrics = evaluate_search(&query_evals, 5);

    Ok(RetrievalResult {
        recall_any_at_5: recall_any_at_k(&query_evals, 5),
        recall_all_at_5: recall_all_at_k(&query_evals, 5),
        recall_any_at_10: recall_any_at_k(&query_evals, 10),
        recall_all_at_10: recall_all_at_k(&query_evals, 10),
        mrr: search_metrics.mrr,
        precision_at_5: search_metrics.precision_at_k,
        n_sessions: dataset.sessions.len(),
        n_queries: dataset.queries.len(),
        embed_time_ms: embed_time.as_millis() as u64,
        index_build_time_ms: build_time.as_millis() as u64,
        total_query_time_ms: query_time.as_millis() as u64,
        avg_query_time_us: if dataset.queries.is_empty() {
            0.0
        } else {
            query_time.as_micros() as f64 / dataset.queries.len() as f64
        },
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Experiment 2: Latency Scaling
// ═══════════════════════════════════════════════════════════════════════════

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
/// Builds HNSW indices at increasing document counts and measures index build
/// time plus p50/p95/p99 query latency. The gap vs ChromaDB should widen with
/// scale since HNSW is O(log n) in native Rust vs Python + SQLite.
pub fn run_latency_scaling(
    embeddings: &[(String, Vec<f32>)],
    query_embeddings: &[Vec<f32>],
    scale_sizes: &[usize],
) -> Result<LatencyResult, crate::EvalError> {
    let dim = embeddings
        .first()
        .ok_or(crate::EvalError::InsufficientData { min: 1, got: 0 })?
        .1
        .len();

    let mut scale_points = Vec::new();

    for &n in scale_sizes {
        let n = n.min(embeddings.len());

        // Build index
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
                .map_err(|e| crate::EvalError::Other(e.to_string()))?;
        }
        let build_time = build_start.elapsed();

        // Measure query latencies
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

// ═══════════════════════════════════════════════════════════════════════════
// Experiment 3: Storage Footprint
// ═══════════════════════════════════════════════════════════════════════════

/// Results from the storage footprint experiment.
#[derive(Debug, Clone, Serialize)]
pub struct StorageResult {
    pub n_vectors: usize,
    pub dimension: usize,
    pub raw_embedding_bytes: usize,
    pub hnsw_index_bytes: usize,
    pub quantized: Vec<QuantizedStoragePoint>,
}

/// Storage and quality metrics at a single quantization bit-width.
#[derive(Debug, Clone, Serialize)]
pub struct QuantizedStoragePoint {
    pub bits: u8,
    pub compressed_bytes: usize,
    pub compression_ratio: f64,
    pub recall_any_at_5: f64,
    pub recall_any_at_10: f64,
    pub cosine_distortion: f64,
}

/// Run Experiment 3: storage footprint comparison.
///
/// Measures raw embedding size, HNSW index size, and TurboQuant compressed
/// sizes at each bit-width. Also measures recall degradation — if 2-bit
/// mnemonist still matches MemPalace's raw recall, that's a clean win.
#[cfg(feature = "quant")]
pub fn run_storage_footprint(
    embeddings: &[(String, Vec<f32>)],
    query_embeddings: &[Vec<f32>],
    query_gold: &[Vec<String>],
    bits_range: &[u8],
    seed: u64,
) -> Result<StorageResult, crate::EvalError> {
    use mnemonist_quant::TurboQuantMse;

    let dim = embeddings
        .first()
        .ok_or(crate::EvalError::InsufficientData { min: 1, got: 0 })?
        .1
        .len();

    // Raw embedding storage
    let raw_bytes = embeddings.len() * dim * 4; // f32 = 4 bytes per coordinate

    // HNSW index: build and measure serialized size
    let mut index = HnswIndex::with_defaults(dim);
    for (id, emb) in embeddings {
        index
            .insert(id, emb)
            .map_err(|e| crate::EvalError::Other(e.to_string()))?;
    }
    let tmp_path = std::env::temp_dir().join("mnemonist_bench_hnsw.bin");
    index
        .save(&tmp_path)
        .map_err(|e| crate::EvalError::Other(e.to_string()))?;
    let hnsw_bytes = std::fs::metadata(&tmp_path)
        .map(|m| m.len() as usize)
        .unwrap_or(0);
    let _ = std::fs::remove_file(&tmp_path);

    // Quantized storage at each bit width
    let mut quantized = Vec::new();

    for &bits in bits_range {
        let quant = TurboQuantMse::new(dim, bits, seed)?;

        // Dequantize all vectors
        let dequantized: Vec<Vec<f32>> = embeddings
            .iter()
            .map(|(_, v)| {
                let q = quant.quantize(v).unwrap();
                quant.dequantize(&q).unwrap()
            })
            .collect();

        // Cosine distortion
        let total_dist: f64 = embeddings
            .iter()
            .zip(dequantized.iter())
            .map(|((_, orig), deq)| (1.0 - cosine_similarity(orig, deq)) as f64)
            .sum();
        let cos_dist = total_dist / embeddings.len() as f64;

        // Compressed size: packed indices + norm per vector
        let bytes_per_vec = (dim * bits as usize).div_ceil(8) + 4; // ceil(dim*bits/8) + 4B norm
        let compressed_bytes = embeddings.len() * bytes_per_vec;

        // Recall on dequantized vectors via brute-force ranking
        let recall_5 =
            compute_quantized_recall_any(embeddings, &dequantized, query_embeddings, query_gold, 5);
        let recall_10 = compute_quantized_recall_any(
            embeddings,
            &dequantized,
            query_embeddings,
            query_gold,
            10,
        );

        quantized.push(QuantizedStoragePoint {
            bits,
            compressed_bytes,
            compression_ratio: raw_bytes as f64 / compressed_bytes as f64,
            recall_any_at_5: recall_5,
            recall_any_at_10: recall_10,
            cosine_distortion: cos_dist,
        });
    }

    Ok(StorageResult {
        n_vectors: embeddings.len(),
        dimension: dim,
        raw_embedding_bytes: raw_bytes,
        hnsw_index_bytes: hnsw_bytes,
        quantized,
    })
}

/// Brute-force recall\_any on dequantized vectors.
#[cfg(feature = "quant")]
fn compute_quantized_recall_any(
    embeddings: &[(String, Vec<f32>)],
    dequantized: &[Vec<f32>],
    query_embeddings: &[Vec<f32>],
    query_gold: &[Vec<String>],
    k: usize,
) -> f64 {
    use std::collections::HashSet;

    if query_embeddings.is_empty() {
        return 0.0;
    }

    let mut total_hits = 0;

    for (q_emb, gold_ids) in query_embeddings.iter().zip(query_gold) {
        let gold: HashSet<&str> = gold_ids.iter().map(|s| s.as_str()).collect();

        // Rank by cosine similarity to dequantized vectors
        let mut scored: Vec<(&str, f32)> = dequantized
            .iter()
            .enumerate()
            .map(|(i, deq)| (embeddings[i].0.as_str(), cosine_similarity(q_emb, deq)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_k: HashSet<&str> = scored.iter().take(k).map(|(id, _)| *id).collect();
        if gold.iter().any(|g| top_k.contains(g)) {
            total_hits += 1;
        }
    }

    total_hits as f64 / query_embeddings.len() as f64
}

// ═══════════════════════════════════════════════════════════════════════════
// Experiment 4: Temporal Retrieval
// ═══════════════════════════════════════════════════════════════════════════

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
/// Demonstrates what MemPalace structurally can't do: retrieval quality that
/// improves over time through access-pattern reinforcement.
///
/// Protocol:
/// 1. Build HNSW index from all embeddings
/// 2. Split queries: first half for "training" access patterns, second half for eval
/// 3. Simulate repeated query cycles on the training set, tracking access counts
/// 4. Eval: rerank with temporal signals (access frequency → strength boost)
/// 5. Compare reinforced recall vs pure-cosine baseline
pub fn run_temporal_retrieval(
    embeddings: &[(String, Vec<f32>)],
    query_embeddings: &[Vec<f32>],
    query_gold: &[Vec<String>],
    n_cycles: usize,
) -> Result<TemporalResult, crate::EvalError> {
    use mnemonist_core::MemoryType;
    use mnemonist_core::temporal::{blend, temporal_score};
    use std::collections::HashSet;

    let dim = embeddings
        .first()
        .ok_or(crate::EvalError::InsufficientData { min: 1, got: 0 })?
        .1
        .len();

    // Build HNSW
    let mut index = HnswIndex::new(
        dim,
        HnswConfig {
            ef_search: 100,
            ..HnswConfig::default()
        },
    );
    for (id, emb) in embeddings {
        index
            .insert(id, emb)
            .map_err(|e| crate::EvalError::Other(e.to_string()))?;
    }

    // Baseline: pure cosine retrieval on ALL queries
    let baseline_recall = compute_recall_any(&index, query_embeddings, query_gold, 5);

    // Split queries: first half trains access patterns, second half evaluates
    let train_end = query_embeddings.len() / 2;
    let eval_queries = &query_embeddings[train_end..];
    let eval_gold = &query_gold[train_end..];

    // Simulate access patterns over multiple consolidation cycles
    let mut access_counts: HashMap<String, u32> = HashMap::new();
    let lambda = 0.15; // Temporal blend weight

    for cycle in 0..n_cycles {
        let _recency_base = ((n_cycles - cycle) * 5) as f64;

        for (q_emb, gold_ids) in query_embeddings[..train_end]
            .iter()
            .zip(&query_gold[..train_end])
        {
            let hits = index.search(q_emb, 10).unwrap_or_default();

            // Record accesses for gold documents that appear in results
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

        // Retrieve more candidates for reranking
        let hits = index.search(q_emb, 20).unwrap_or_default();

        // Rerank with temporal signal
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

/// Pure-cosine recall\_any@k on pre-built HNSW index.
fn compute_recall_any(
    index: &HnswIndex,
    query_embeddings: &[Vec<f32>],
    query_gold: &[Vec<String>],
    k: usize,
) -> f64 {
    use std::collections::HashSet;

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

// ═══════════════════════════════════════════════════════════════════════════
// Report
// ═══════════════════════════════════════════════════════════════════════════

/// Complete benchmark report across all experiments.
#[derive(Debug, Clone, Serialize)]
pub struct BenchReport {
    pub timestamp: String,
    pub retrieval: Option<RetrievalResult>,
    pub latency: Option<LatencyResult>,
    #[cfg(feature = "quant")]
    pub storage: Option<StorageResult>,
    pub temporal: Option<TemporalResult>,
}

impl BenchReport {
    /// Render as pretty-printed JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Render as a concise human-readable summary.
    pub fn to_summary(&self) -> String {
        let mut lines = Vec::new();

        if let Some(ref r) = self.retrieval {
            lines.push("═══ Experiment 1: LongMemEval Retrieval ═══".to_string());
            lines.push(format!(
                "  sessions: {}  queries: {}",
                r.n_sessions, r.n_queries
            ));
            lines.push(format!(
                "  recall_any@5:  {:.4} ({:.1}%)",
                r.recall_any_at_5,
                r.recall_any_at_5 * 100.0
            ));
            lines.push(format!(
                "  recall_all@5:  {:.4} ({:.1}%)",
                r.recall_all_at_5,
                r.recall_all_at_5 * 100.0
            ));
            lines.push(format!(
                "  recall_any@10: {:.4} ({:.1}%)",
                r.recall_any_at_10,
                r.recall_any_at_10 * 100.0
            ));
            lines.push(format!(
                "  recall_all@10: {:.4} ({:.1}%)",
                r.recall_all_at_10,
                r.recall_all_at_10 * 100.0
            ));
            lines.push(format!(
                "  mrr: {:.4}  precision@5: {:.4}",
                r.mrr, r.precision_at_5
            ));
            lines.push(format!(
                "  embed: {}ms  build: {}ms  query: {}ms (avg {:.0}µs)",
                r.embed_time_ms, r.index_build_time_ms, r.total_query_time_ms, r.avg_query_time_us
            ));
        }

        if let Some(ref l) = self.latency {
            lines.push(String::new());
            lines.push("═══ Experiment 2: Latency Scaling ═══".to_string());
            lines.push(format!(
                "  {:>8}  {:>10}  {:>10}  {:>10}  {:>10}",
                "n_docs", "build_ms", "p50_µs", "p95_µs", "p99_µs"
            ));
            for p in &l.scale_points {
                lines.push(format!(
                    "  {:>8}  {:>10}  {:>10.0}  {:>10.0}  {:>10.0}",
                    p.n_documents,
                    p.index_build_time_ms,
                    p.query_latency_p50_us,
                    p.query_latency_p95_us,
                    p.query_latency_p99_us
                ));
            }
        }

        #[cfg(feature = "quant")]
        if let Some(ref s) = self.storage {
            lines.push(String::new());
            lines.push("═══ Experiment 3: Storage Footprint ═══".to_string());
            lines.push(format!("  vectors: {}  dim: {}", s.n_vectors, s.dimension));
            lines.push(format!(
                "  raw embeddings: {} bytes ({:.1} KB)",
                s.raw_embedding_bytes,
                s.raw_embedding_bytes as f64 / 1024.0
            ));
            lines.push(format!(
                "  HNSW index:     {} bytes ({:.1} KB)",
                s.hnsw_index_bytes,
                s.hnsw_index_bytes as f64 / 1024.0
            ));
            lines.push(format!(
                "  {:>4}  {:>10}  {:>6}  {:>12}  {:>10}  {:>11}",
                "bits", "bytes", "ratio", "cos_dist", "recall@5", "recall@10"
            ));
            for q in &s.quantized {
                lines.push(format!(
                    "  {:>4}  {:>10}  {:>5.1}x  {:>12.6}  {:>9.1}%  {:>10.1}%",
                    q.bits,
                    q.compressed_bytes,
                    q.compression_ratio,
                    q.cosine_distortion,
                    q.recall_any_at_5 * 100.0,
                    q.recall_any_at_10 * 100.0
                ));
            }
        }

        if let Some(ref t) = self.temporal {
            lines.push(String::new());
            lines.push("═══ Experiment 4: Temporal Retrieval ═══".to_string());
            lines.push(format!(
                "  documents: {}  cycles: {}  queries/cycle: {}",
                t.n_documents, t.n_consolidation_cycles, t.n_queries_per_cycle
            ));
            lines.push(format!(
                "  baseline recall_any@5:    {:.4} ({:.1}%)",
                t.baseline_recall_any_at_5,
                t.baseline_recall_any_at_5 * 100.0
            ));
            lines.push(format!(
                "  reinforced recall_any@5:  {:.4} ({:.1}%)",
                t.reinforced_recall_any_at_5,
                t.reinforced_recall_any_at_5 * 100.0
            ));
            lines.push(format!(
                "  delta: {:+.4} ({:+.1}%)",
                t.recall_delta,
                t.recall_delta * 100.0
            ));
        }

        if lines.is_empty() {
            "no experiments run".to_string()
        } else {
            lines.join("\n")
        }
    }
}
