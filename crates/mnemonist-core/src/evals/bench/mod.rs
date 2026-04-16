//! Benchmark suite: 6 experiments comparing mnemonist retrieval infrastructure.
//!
//! Each experiment lives in its own module and is fully self-contained:
//!
//! 1. [`vector_retrieval`] — session retrieval recall@k (what MemPalace measured — NOT QA)
//! 2. [`latency_scaling`] — index build + p50/p95/p99 query latency at 100–10k docs
//! 3. [`storage_footprint`] — raw vs TurboQuant compressed at 1–4 bits, with recall impact
//! 4. [`temporal_retrieval`] — dynamic recall improvement via Hebbian access patterns
//! 5. [`mempalace_comparison`] — apples-to-apples retrieval parity (NOT a LongMemEval QA score)
//! 6. [`longmemeval_qa`] — real end-to-end QA (retrieval + LLM scoring)

pub mod latency_scaling;
pub mod longmemeval_qa;
pub mod mempalace_comparison;
#[cfg(feature = "quant")]
pub mod storage_footprint;
pub mod temporal_retrieval;
pub mod vector_retrieval;

use serde::Serialize;

// Re-export result types for convenience.
pub use latency_scaling::{LatencyResult, ScalePoint};
pub use longmemeval_qa::QaExperimentResult;
pub use mempalace_comparison::MemPalaceComparisonResult;
#[cfg(feature = "quant")]
pub use storage_footprint::{QuantizedStoragePoint, StorageResult};
pub use temporal_retrieval::TemporalResult;
pub use vector_retrieval::RetrievalResult;

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
    pub mempalace: Option<MemPalaceComparisonResult>,
    pub qa: Option<QaExperimentResult>,
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
            lines.push("═══ Experiment 1: Vector Retrieval (NOT LongMemEval QA) ═══".to_string());
            lines.push(format!(
                "  sessions: {}  queries: {}  avg haystack: {:.0}",
                r.n_sessions, r.n_queries, r.avg_haystack_size
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

        if let Some(ref m) = self.mempalace {
            lines.push(String::new());
            lines.push(
                "═══ Experiment 5: MemPalace Comparison (Retrieval Only — NOT QA) ═══".to_string(),
            );
            lines.push(format!(
                "  sessions: {}  queries: {}  avg haystack: {:.0}",
                m.n_sessions, m.n_queries, m.avg_haystack_size
            ));
            lines.push(format!(
                "  mnemonist recall_any@5:  {:.4} ({:.1}%)",
                m.mnemonist_recall_any_at_5,
                m.mnemonist_recall_any_at_5 * 100.0
            ));
            lines.push(format!(
                "  mnemonist recall_all@5:  {:.4} ({:.1}%)",
                m.mnemonist_recall_all_at_5,
                m.mnemonist_recall_all_at_5 * 100.0
            ));
            lines.push(format!(
                "  time: {:.1}s ({:.2}s per question)",
                m.total_time_s, m.per_question_time_s
            ));
            lines.push(format!("  note: {}", m.note));
        }

        if let Some(ref q) = self.qa {
            lines.push(String::new());
            lines.push("═══ Experiment 6: LongMemEval QA ═══".to_string());
            lines.push(format!("  mode: {}", q.mode));
            if let Some(recall) = q.retrieval_recall_any_at_k {
                lines.push(format!("  retrieval recall_any@k: {:.1}%", recall * 100.0));
            }
            if let Some(avg_ms) = q.avg_time_per_question_ms {
                lines.push(format!("  avg time/question: {:.0}ms", avg_ms));
            }
            if let Some(acc) = q.overall_accuracy {
                lines.push(format!("  overall accuracy: {:.1}%", acc * 100.0));
            }
            if let Some(n) = q.n_questions {
                lines.push(format!("  questions: {}", n));
            }
        }

        if lines.is_empty() {
            "no experiments run".to_string()
        } else {
            lines.join("\n")
        }
    }
}
