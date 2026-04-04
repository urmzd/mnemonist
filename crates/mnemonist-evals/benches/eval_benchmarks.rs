use criterion::{Criterion, criterion_group, criterion_main};
use mnemonist_evals::dataset::{SyntheticConfig, generate_synthetic};
use mnemonist_evals::embedding;
use mnemonist_evals::search::{QueryEval, evaluate_search};
use std::collections::HashMap;

fn bench_anisotropy(c: &mut Criterion) {
    let config = SyntheticConfig {
        n_documents: 200,
        n_clusters: 10,
        n_queries: 0,
        dimension: 32,
        noise: 0.3,
        seed: 42,
    };
    let ds = generate_synthetic(&config);
    let embeddings: Vec<Vec<f32>> = ds.documents.iter().map(|d| d.embedding.clone()).collect();

    c.bench_function("anisotropy_200x32", |b| {
        b.iter(|| embedding::anisotropy(&embeddings))
    });
}

fn bench_intrinsic_dim(c: &mut Criterion) {
    let config = SyntheticConfig {
        n_documents: 100,
        n_clusters: 5,
        n_queries: 0,
        dimension: 32,
        noise: 0.3,
        seed: 42,
    };
    let ds = generate_synthetic(&config);
    let embeddings: Vec<Vec<f32>> = ds.documents.iter().map(|d| d.embedding.clone()).collect();

    c.bench_function("intrinsic_dim_100x32", |b| {
        b.iter(|| embedding::intrinsic_dimensionality(&embeddings))
    });
}

fn bench_search_metrics(c: &mut Criterion) {
    let queries: Vec<QueryEval> = (0..100)
        .map(|i| {
            let retrieved: Vec<String> = (0..20).map(|j| format!("doc_{j}")).collect();
            let mut judgments = HashMap::new();
            for j in 0..5 {
                judgments.insert(format!("doc_{}", j + i % 10), 1);
            }
            QueryEval {
                query_id: format!("q_{i}"),
                retrieved,
                judgments,
            }
        })
        .collect();

    c.bench_function("search_metrics_100q_k10", |b| {
        b.iter(|| evaluate_search(&queries, 10))
    });
}

criterion_group!(
    benches,
    bench_anisotropy,
    bench_intrinsic_dim,
    bench_search_metrics
);
criterion_main!(benches);
