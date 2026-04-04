//! End-to-end eval harness. Orchestrates dataset generation, index construction,
//! and metric computation into a single [`EvalReport`].

use crate::EvalError;
use crate::dataset::{self, BenchmarkDataset, SyntheticConfig, brute_force_top_k};
use crate::embedding::{self, EmbeddingMetrics};
use crate::report::{DatasetInfo, EvalReport};
use crate::search::{self, QueryEval, SearchMetrics};

/// Configuration for an eval run.
pub struct EvalConfig {
    pub embedding_eval: bool,
    pub search_eval: bool,
    #[cfg(feature = "quant")]
    pub quantization_eval: bool,
    pub search_k: usize,
    #[cfg(feature = "quant")]
    pub quant_bits: Vec<u8>,
    pub synthetic: SyntheticConfig,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            embedding_eval: true,
            search_eval: true,
            #[cfg(feature = "quant")]
            quantization_eval: true,
            search_k: 10,
            #[cfg(feature = "quant")]
            quant_bits: vec![1, 2, 3, 4],
            synthetic: SyntheticConfig::default(),
        }
    }
}

/// Run the full evaluation suite on a synthetic dataset.
pub fn run(config: &EvalConfig) -> Result<EvalReport, EvalError> {
    let ds = dataset::generate_synthetic(&config.synthetic);
    let timestamp = chrono_stub();

    let emb_metrics = if config.embedding_eval {
        let embeddings: Vec<Vec<f32>> = ds.documents.iter().map(|d| d.embedding.clone()).collect();
        let groups: Vec<usize> = ds.documents.iter().map(|d| d.group).collect();
        Some(run_embedding_eval(&embeddings, Some(&groups)))
    } else {
        None
    };

    let search_metrics = if config.search_eval {
        Some(run_search_eval_bruteforce(&ds, config.search_k))
    } else {
        None
    };

    #[cfg(feature = "quant")]
    let quant_metrics = if config.quantization_eval {
        let embeddings: Vec<Vec<f32>> = ds.documents.iter().map(|d| d.embedding.clone()).collect();
        Some(crate::quantization::evaluate_mse_quantizer(
            &embeddings,
            &config.quant_bits,
            config.synthetic.seed,
        )?)
    } else {
        None
    };

    Ok(EvalReport {
        timestamp,
        embedding: emb_metrics,
        search: search_metrics,
        #[cfg(feature = "quant")]
        quantization: quant_metrics,
        dataset_info: Some(DatasetInfo {
            n_documents: ds.documents.len(),
            n_queries: ds.queries.len(),
            dimension: config.synthetic.dimension,
            source: "synthetic".to_string(),
        }),
    })
}

/// Run embedding quality eval on pre-computed embeddings.
pub fn run_embedding_eval(embeddings: &[Vec<f32>], groups: Option<&[usize]>) -> EmbeddingMetrics {
    embedding::evaluate_embeddings(embeddings, groups)
}

/// Run search quality eval using brute-force retrieval on a benchmark dataset.
pub fn run_search_eval_bruteforce(ds: &BenchmarkDataset, k: usize) -> SearchMetrics {
    let queries: Vec<QueryEval> = ds
        .queries
        .iter()
        .map(|q| {
            let top_indices = brute_force_top_k(&q.embedding, &ds.documents, k);
            let retrieved: Vec<String> = top_indices
                .iter()
                .map(|&i| ds.documents[i].id.clone())
                .collect();

            QueryEval {
                query_id: q.id.clone(),
                retrieved,
                judgments: q.judgments.clone(),
            }
        })
        .collect();

    search::evaluate_search(&queries, k)
}

/// Run search quality eval using an ANN index on a benchmark dataset.
#[cfg(feature = "index")]
pub fn run_search_eval_ann(
    index: &dyn llmem_index::AnnIndex,
    ds: &BenchmarkDataset,
    k: usize,
) -> SearchMetrics {
    let queries: Vec<QueryEval> = ds
        .queries
        .iter()
        .map(|q| {
            let hits = index.search(&q.embedding, k).unwrap_or_default();
            let retrieved: Vec<String> = hits.into_iter().map(|h| h.id).collect();

            // Translate judgments to use the same ID format the index stores
            QueryEval {
                query_id: q.id.clone(),
                retrieved,
                judgments: q.judgments.clone(),
            }
        })
        .collect();

    search::evaluate_search(&queries, k)
}

/// Evaluate an ANN index: build it from the dataset, then measure search quality.
#[cfg(feature = "index")]
pub fn eval_ann_index(
    index: &mut dyn llmem_index::AnnIndex,
    ds: &BenchmarkDataset,
    k: usize,
) -> Result<(SearchMetrics, SearchMetrics), EvalError> {
    // Insert all documents
    for doc in &ds.documents {
        index
            .insert(&doc.id, &doc.embedding)
            .map_err(|e| EvalError::Other(e.to_string()))?;
    }

    // Brute-force baseline
    let baseline = run_search_eval_bruteforce(ds, k);
    // ANN results
    let ann = run_search_eval_ann(index, ds, k);

    Ok((baseline, ann))
}

fn chrono_stub() -> String {
    // Avoid pulling in chrono — use a simple timestamp
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}s", d.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_default_config() {
        let config = EvalConfig {
            synthetic: SyntheticConfig {
                n_documents: 100,
                n_clusters: 5,
                n_queries: 10,
                dimension: 16,
                noise: 0.3,
                seed: 42,
            },
            ..Default::default()
        };

        let report = run(&config).unwrap();
        assert!(report.embedding.is_some());
        assert!(report.search.is_some());
        assert!(report.dataset_info.is_some());

        let emb = report.embedding.unwrap();
        assert!(emb.anisotropy < 0.5);
        assert!(emb.intrinsic_dimensionality > 1.0);

        let search = report.search.unwrap();
        // Brute-force on clustered data should have decent recall
        assert!(search.recall_at_k > 0.3);
    }

    #[cfg(feature = "index")]
    #[test]
    fn eval_hnsw_index() {
        let config = SyntheticConfig {
            n_documents: 200,
            n_clusters: 5,
            n_queries: 20,
            dimension: 16,
            noise: 0.3,
            seed: 42,
        };
        let ds = dataset::generate_synthetic(&config);

        let mut hnsw = llmem_index::hnsw::HnswIndex::with_defaults(config.dimension);
        let (baseline, ann) = eval_ann_index(&mut hnsw, &ds, 10).unwrap();

        // ANN recall should be close to brute-force
        assert!(
            ann.recall_at_k >= baseline.recall_at_k * 0.8,
            "ann recall {} too far below baseline {}",
            ann.recall_at_k,
            baseline.recall_at_k
        );
    }
}
