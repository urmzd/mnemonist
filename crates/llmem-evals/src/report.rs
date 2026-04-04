//! Structured evaluation report.

use serde::{Deserialize, Serialize};

use crate::embedding::EmbeddingMetrics;
use crate::search::SearchMetrics;

/// Complete evaluation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    pub timestamp: String,
    pub embedding: Option<EmbeddingMetrics>,
    pub search: Option<SearchMetrics>,
    #[cfg(feature = "quant")]
    pub quantization: Option<Vec<crate::quantization::QuantMetrics>>,
    pub dataset_info: Option<DatasetInfo>,
}

/// Metadata about the dataset used for evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    pub n_documents: usize,
    pub n_queries: usize,
    pub dimension: usize,
    pub source: String,
}

impl EvalReport {
    /// Render as pretty-printed JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Render as a concise human-readable summary.
    pub fn to_summary(&self) -> String {
        let mut lines = Vec::new();

        if let Some(ref info) = self.dataset_info {
            lines.push(format!(
                "dataset: {} ({} docs, {} queries, dim={})",
                info.source, info.n_documents, info.n_queries, info.dimension
            ));
        }

        if let Some(ref e) = self.embedding {
            lines.push(format!(
                "embedding: anisotropy={:.4} range={:.4} id={:.1} (n={})",
                e.anisotropy, e.similarity_range, e.intrinsic_dimensionality, e.sample_size
            ));
            if let Some(gap) = e.discrimination_gap {
                lines.push(format!("  discrimination_gap={gap:.4}"));
            }
        }

        if let Some(ref s) = self.search {
            lines.push(format!(
                "search@{}: p={:.4} r={:.4} mrr={:.4} ndcg={:.4} (n={})",
                s.k, s.precision_at_k, s.recall_at_k, s.mrr, s.ndcg_at_k, s.n_queries
            ));
        }

        #[cfg(feature = "quant")]
        if let Some(ref qs) = self.quantization {
            for q in qs {
                lines.push(format!(
                    "quant@{}bit: mse={:.6} cos_dist={:.6} ratio={:.1}x (n={})",
                    q.bits, q.mean_mse, q.cosine_distortion, q.compression_ratio, q.n_vectors
                ));
            }
        }

        if lines.is_empty() {
            "no metrics computed".to_string()
        } else {
            lines.join("\n")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_json_roundtrip() {
        let report = EvalReport {
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            embedding: Some(EmbeddingMetrics {
                anisotropy: 0.15,
                similarity_range: 0.8,
                discrimination_gap: Some(0.12),
                intrinsic_dimensionality: 24.0,
                sample_size: 100,
            }),
            search: None,
            #[cfg(feature = "quant")]
            quantization: None,
            dataset_info: None,
        };

        let json = report.to_json();
        let parsed: EvalReport = serde_json::from_str(&json).unwrap();
        assert!((parsed.embedding.unwrap().anisotropy - 0.15).abs() < 1e-6);
    }

    #[test]
    fn summary_output() {
        let report = EvalReport {
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            embedding: Some(EmbeddingMetrics {
                anisotropy: 0.15,
                similarity_range: 0.8,
                discrimination_gap: None,
                intrinsic_dimensionality: 24.0,
                sample_size: 100,
            }),
            search: Some(SearchMetrics {
                precision_at_k: 0.8,
                recall_at_k: 0.6,
                mrr: 0.9,
                ndcg_at_k: 0.85,
                k: 10,
                n_queries: 50,
            }),
            #[cfg(feature = "quant")]
            quantization: None,
            dataset_info: Some(DatasetInfo {
                n_documents: 500,
                n_queries: 50,
                dimension: 32,
                source: "synthetic".to_string(),
            }),
        };

        let s = report.to_summary();
        assert!(s.contains("anisotropy=0.1500"));
        assert!(s.contains("mrr=0.9000"));
        assert!(s.contains("synthetic"));
    }
}
