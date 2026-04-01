//! Quality validation tests for ANN indices and eval metrics.
//!
//! These tests assert thresholds that must hold across code changes,
//! serving as regression guards for recall and embedding quality.

use llmem_index::AnnIndex;
use llmem_index::distance::{cosine_similarity, normalize};
use llmem_index::eval::{anisotropy, discrimination_gap, mean_center, similarity_range};
use llmem_index::hnsw::HnswIndex;
use llmem_index::ivf::{IvfConfig, IvfFlatIndex};

fn make_vector(dim: usize, seed: f32) -> Vec<f32> {
    (0..dim).map(|i| ((i as f32 + seed) * 0.1).sin()).collect()
}

/// Brute-force top-k by cosine similarity for ground truth.
fn brute_force_top_k(query: &[f32], vectors: &[Vec<f32>], k: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i, cosine_similarity(query, v)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

// ── HNSW Recall ────────────────────────────────────────────────────────────

#[test]
fn hnsw_recall_at_10_gte_90() {
    let dim = 32;
    let n = 500;
    let k = 10;
    let n_queries = 50;

    let vectors: Vec<Vec<f32>> = (0..n).map(|i| make_vector(dim, i as f32)).collect();

    let mut index = HnswIndex::with_defaults(dim);
    for (i, v) in vectors.iter().enumerate() {
        index.insert(&format!("item_{i}"), v).unwrap();
    }

    let mut total_recall = 0.0;
    for q in 0..n_queries {
        let query = make_vector(dim, (n + q) as f32);
        let truth = brute_force_top_k(&query, &vectors, k);
        let results = index.search(&query, k).unwrap();
        let result_ids: Vec<usize> = results
            .iter()
            .filter_map(|h| h.id.strip_prefix("item_").and_then(|s| s.parse().ok()))
            .collect();
        let hits = truth.iter().filter(|t| result_ids.contains(t)).count();
        total_recall += hits as f64 / k as f64;
    }

    let avg_recall = total_recall / n_queries as f64;
    assert!(
        avg_recall >= 0.90,
        "HNSW recall@{k} = {avg_recall:.3}, expected >= 0.90"
    );
}

// ── IVF Recall ─────────────────────────────────────────────────────────────

#[test]
fn ivf_recall_at_10_gte_85() {
    let dim = 32;
    let n = 500;
    let k = 10;
    let n_queries = 50;

    let vectors: Vec<Vec<f32>> = (0..n).map(|i| make_vector(dim, i as f32)).collect();

    let mut index = IvfFlatIndex::new(
        dim,
        IvfConfig {
            n_lists: 16,
            n_probe: 10,
            kmeans_iters: 20,
        },
    );
    index.train(&vectors);
    for (i, v) in vectors.iter().enumerate() {
        index.insert(&format!("item_{i}"), v).unwrap();
    }

    let mut total_recall = 0.0;
    for q in 0..n_queries {
        let query = make_vector(dim, (n + q) as f32);
        let truth = brute_force_top_k(&query, &vectors, k);
        let results = index.search(&query, k).unwrap();
        let result_ids: Vec<usize> = results
            .iter()
            .filter_map(|h| h.id.strip_prefix("item_").and_then(|s| s.parse().ok()))
            .collect();
        let hits = truth.iter().filter(|t| result_ids.contains(t)).count();
        total_recall += hits as f64 / k as f64;
    }

    let avg_recall = total_recall / n_queries as f64;
    assert!(
        avg_recall >= 0.85,
        "IVF recall@{k} = {avg_recall:.3}, expected >= 0.85"
    );
}

// ── Eval Quality Thresholds ────────────────────────────────────────────────

#[test]
fn anisotropy_orthogonal_near_zero() {
    let dim = 8;
    let vecs: Vec<Vec<f32>> = (0..dim)
        .map(|i| {
            let mut v = vec![0.0f32; dim];
            v[i] = 1.0;
            v
        })
        .collect();

    let a = anisotropy(&vecs);
    assert!(
        a.abs() < 0.01,
        "anisotropy on orthogonal basis = {a}, expected < 0.01"
    );
}

#[test]
fn similarity_range_mixed_vectors_gt_03() {
    let vecs = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.7071, 0.7071, 0.0, 0.0],
    ];
    let range = similarity_range(&vecs);
    assert!(range > 0.3, "similarity_range = {range}, expected > 0.3");
}

#[test]
fn discrimination_gap_two_clusters_gt_005() {
    let mut embeddings = Vec::new();
    let mut groups = Vec::new();

    // Cluster 0: near [1, 0, 0, 0]
    for i in 0..5 {
        let mut v = vec![1.0, 0.0, 0.0, 0.0];
        v[1] = (i as f32) * 0.01;
        normalize(&mut v);
        embeddings.push(v);
        groups.push(0);
    }

    // Cluster 1: near [0, 0, 1, 0]
    for i in 0..5 {
        let mut v = vec![0.0, 0.0, 1.0, 0.0];
        v[3] = (i as f32) * 0.01;
        normalize(&mut v);
        embeddings.push(v);
        groups.push(1);
    }

    let gap = discrimination_gap(&embeddings, &groups);
    assert!(gap > 0.05, "discrimination_gap = {gap}, expected > 0.05");
}

#[test]
fn mean_center_reduces_anisotropy() {
    let base = vec![1.0f32, 0.0, 0.0, 0.0];
    let vecs: Vec<Vec<f32>> = (0..10)
        .map(|i| {
            let mut v = base.clone();
            v[1] = (i as f32) * 0.05;
            normalize(&mut v);
            v
        })
        .collect();

    let before = anisotropy(&vecs);
    let centered = mean_center(&vecs);
    let after = anisotropy(&centered);

    assert!(
        after < before,
        "mean_center should reduce anisotropy: {before} -> {after}"
    );
}
