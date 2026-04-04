//! Synthetic benchmark dataset generation for retrieval evaluation.

use std::collections::HashMap;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, StandardNormal};
use serde::{Deserialize, Serialize};

use llmem_core::distance::normalize;

/// A benchmark dataset with documents and queries with known ground truth.
#[derive(Debug, Clone)]
pub struct BenchmarkDataset {
    pub documents: Vec<Document>,
    pub queries: Vec<BenchmarkQuery>,
}

/// A document in the benchmark dataset.
#[derive(Debug, Clone)]
pub struct Document {
    pub id: String,
    pub embedding: Vec<f32>,
    pub group: usize,
}

/// A query with known relevant document IDs and graded relevance.
#[derive(Debug, Clone)]
pub struct BenchmarkQuery {
    pub id: String,
    pub embedding: Vec<f32>,
    /// Map from doc ID to relevance grade (higher = more relevant).
    pub judgments: HashMap<String, u32>,
}

/// Configuration for synthetic dataset generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticConfig {
    pub n_documents: usize,
    pub n_clusters: usize,
    pub n_queries: usize,
    pub dimension: usize,
    pub noise: f32,
    pub seed: u64,
}

impl Default for SyntheticConfig {
    fn default() -> Self {
        Self {
            n_documents: 500,
            n_clusters: 10,
            n_queries: 50,
            dimension: 32,
            noise: 0.3,
            seed: 42,
        }
    }
}

/// Generate a synthetic dataset using clustered random vectors.
///
/// Each cluster has a random centroid. Documents are noisy perturbations of their
/// cluster centroid. Queries are generated near centroids — documents in the same
/// cluster are relevant (grade 2), nearest-neighbor cluster docs get grade 1.
pub fn generate_synthetic(config: &SyntheticConfig) -> BenchmarkDataset {
    let mut rng = StdRng::seed_from_u64(config.seed);
    let normal = Normal::new(0.0f32, config.noise).unwrap();

    // Generate cluster centroids
    let centroids: Vec<Vec<f32>> = (0..config.n_clusters)
        .map(|_| {
            let mut v: Vec<f32> = (0..config.dimension)
                .map(|_| StandardNormal.sample(&mut rng))
                .collect();
            normalize(&mut v);
            v
        })
        .collect();

    // Generate documents around centroids
    let docs_per_cluster = config.n_documents / config.n_clusters.max(1);
    let mut documents = Vec::with_capacity(config.n_documents);

    for (cluster_id, centroid) in centroids.iter().enumerate() {
        for j in 0..docs_per_cluster {
            let mut emb: Vec<f32> = centroid
                .iter()
                .map(|&c| c + normal.sample(&mut rng))
                .collect();
            normalize(&mut emb);

            documents.push(Document {
                id: format!("doc_{cluster_id}_{j}"),
                embedding: emb,
                group: cluster_id,
            });
        }
    }

    // Generate queries near centroids with ground truth
    let queries_per_cluster = config.n_queries / config.n_clusters.max(1);
    let query_noise = Normal::new(0.0f32, config.noise * 0.5).unwrap();
    let mut queries = Vec::with_capacity(config.n_queries);

    for (cluster_id, centroid) in centroids.iter().enumerate() {
        for j in 0..queries_per_cluster {
            let mut emb: Vec<f32> = centroid
                .iter()
                .map(|&c| c + query_noise.sample(&mut rng))
                .collect();
            normalize(&mut emb);

            // Ground truth: same-cluster docs are relevant (grade 2)
            let mut judgments = HashMap::new();
            for doc in &documents {
                if doc.group == cluster_id {
                    judgments.insert(doc.id.clone(), 2);
                }
            }

            queries.push(BenchmarkQuery {
                id: format!("query_{cluster_id}_{j}"),
                embedding: emb,
                judgments,
            });
        }
    }

    BenchmarkDataset { documents, queries }
}

/// Brute-force top-k nearest neighbors by cosine similarity.
///
/// Returns indices into `documents` sorted by descending similarity.
pub fn brute_force_top_k(query: &[f32], documents: &[Document], k: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = documents
        .iter()
        .enumerate()
        .map(|(i, doc)| {
            (
                i,
                llmem_core::distance::cosine_similarity(query, &doc.embedding),
            )
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_dataset_structure() {
        let config = SyntheticConfig {
            n_documents: 100,
            n_clusters: 5,
            n_queries: 10,
            dimension: 16,
            noise: 0.3,
            seed: 42,
        };
        let ds = generate_synthetic(&config);

        assert_eq!(ds.documents.len(), 100);
        assert_eq!(ds.queries.len(), 10);

        // Each query should have relevant docs
        for q in &ds.queries {
            assert!(!q.judgments.is_empty());
        }

        // Each doc should be unit-normalized
        for doc in &ds.documents {
            let norm: f32 = doc.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn brute_force_returns_nearest() {
        let config = SyntheticConfig {
            n_documents: 50,
            n_clusters: 5,
            n_queries: 5,
            dimension: 8,
            noise: 0.1,
            seed: 99,
        };
        let ds = generate_synthetic(&config);
        let q = &ds.queries[0];

        let top = brute_force_top_k(&q.embedding, &ds.documents, 5);
        assert_eq!(top.len(), 5);

        // Top results should mostly be from the query's cluster
        let query_cluster = q
            .judgments
            .keys()
            .next()
            .and_then(|id| id.split('_').nth(1))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap();

        let same_cluster = top
            .iter()
            .filter(|&&i| ds.documents[i].group == query_cluster)
            .count();
        assert!(
            same_cluster >= 3,
            "expected >= 3 same-cluster, got {same_cluster}"
        );
    }

    #[test]
    fn deterministic_output() {
        let config = SyntheticConfig::default();
        let ds1 = generate_synthetic(&config);
        let ds2 = generate_synthetic(&config);
        assert_eq!(ds1.documents.len(), ds2.documents.len());
        assert_eq!(ds1.documents[0].embedding, ds2.documents[0].embedding);
    }
}
