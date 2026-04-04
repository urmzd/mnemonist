use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use mnemonist_index::AnnIndex;
use mnemonist_index::distance::{cosine_similarity, dot_product, l2_distance_squared, normalize};
use mnemonist_index::eval::{anisotropy, discrimination_gap, mean_center, similarity_range};
use mnemonist_index::hnsw::HnswIndex;
use mnemonist_index::ivf::{IvfConfig, IvfFlatIndex};

fn make_vector(dim: usize, seed: f32) -> Vec<f32> {
    (0..dim).map(|i| ((i as f32 + seed) * 0.1).sin()).collect()
}

fn bench_distance_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance");

    for dim in [32, 128, 384] {
        let a = make_vector(dim, 1.0);
        let b = make_vector(dim, 2.0);

        group.bench_with_input(BenchmarkId::new("cosine", dim), &dim, |bench, _| {
            bench.iter(|| cosine_similarity(&a, &b));
        });

        group.bench_with_input(BenchmarkId::new("dot_product", dim), &dim, |bench, _| {
            bench.iter(|| dot_product(&a, &b));
        });

        group.bench_with_input(BenchmarkId::new("l2_squared", dim), &dim, |bench, _| {
            bench.iter(|| l2_distance_squared(&a, &b));
        });

        group.bench_with_input(BenchmarkId::new("normalize", dim), &dim, |bench, _| {
            bench.iter(|| {
                let mut v = a.clone();
                normalize(&mut v);
                v
            });
        });
    }

    group.finish();
}

fn bench_hnsw(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw");
    let dim = 32;
    let n = 500;

    let vectors: Vec<Vec<f32>> = (0..n).map(|i| make_vector(dim, i as f32)).collect();

    // Benchmark insert (build from scratch)
    group.bench_function("insert_500", |bench| {
        bench.iter(|| {
            let mut index = HnswIndex::with_defaults(dim);
            for (i, v) in vectors.iter().enumerate() {
                index.insert(&format!("item_{i}"), v).unwrap();
            }
            index
        });
    });

    // Build index once for search benchmarks
    let mut index = HnswIndex::with_defaults(dim);
    for (i, v) in vectors.iter().enumerate() {
        index.insert(&format!("item_{i}"), v).unwrap();
    }

    let query = make_vector(dim, 42.0);

    for k in [1, 10, 50] {
        group.bench_with_input(BenchmarkId::new("search_top_k", k), &k, |bench, &k| {
            bench.iter(|| index.search(&query, k).unwrap());
        });
    }

    // Benchmark save/load
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("bench.hnsw");
    index.save(&path).unwrap();

    group.bench_function("save_500", |bench| {
        bench.iter(|| index.save(&path).unwrap());
    });

    group.bench_function("load_500", |bench| {
        bench.iter(|| HnswIndex::load_from(&path).unwrap());
    });

    group.finish();
}

fn bench_ivf(c: &mut Criterion) {
    let mut group = c.benchmark_group("ivf");
    let dim = 32;
    let n = 500;

    let vectors: Vec<Vec<f32>> = (0..n).map(|i| make_vector(dim, i as f32)).collect();

    // Benchmark training
    group.bench_function("train_500", |bench| {
        bench.iter(|| {
            let mut index = IvfFlatIndex::new(
                dim,
                IvfConfig {
                    n_lists: 16,
                    n_probe: 10,
                    kmeans_iters: 20,
                },
            );
            index.train(&vectors);
            index
        });
    });

    // Build index for search benchmarks
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

    let query = make_vector(dim, 42.0);

    for k in [1, 10, 50] {
        group.bench_with_input(BenchmarkId::new("search_top_k", k), &k, |bench, &k| {
            bench.iter(|| index.search(&query, k).unwrap());
        });
    }

    // Benchmark save/load
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("bench.ivf");
    index.save(&path).unwrap();

    group.bench_function("save_500", |bench| {
        bench.iter(|| index.save(&path).unwrap());
    });

    group.bench_function("load_500", |bench| {
        bench.iter(|| IvfFlatIndex::load_from(&path).unwrap());
    });

    group.finish();
}

fn bench_eval(c: &mut Criterion) {
    let mut group = c.benchmark_group("eval");

    for &(dim, count) in &[(32, 50), (128, 50), (384, 20)] {
        let label = format!("{dim}d_x{count}");
        let vectors: Vec<Vec<f32>> = (0..count).map(|i| make_vector(dim, i as f32)).collect();

        group.bench_with_input(BenchmarkId::new("anisotropy", &label), &dim, |bench, _| {
            bench.iter(|| anisotropy(&vectors));
        });

        group.bench_with_input(
            BenchmarkId::new("similarity_range", &label),
            &dim,
            |bench, _| {
                bench.iter(|| similarity_range(&vectors));
            },
        );

        group.bench_with_input(BenchmarkId::new("mean_center", &label), &dim, |bench, _| {
            bench.iter(|| mean_center(&vectors));
        });
    }

    // discrimination_gap with groups
    let dim = 32;
    let vectors: Vec<Vec<f32>> = (0..50).map(|i| make_vector(dim, i as f32)).collect();
    let groups: Vec<usize> = (0..50).map(|i| i / 10).collect(); // 5 groups of 10

    group.bench_function("discrimination_gap/32d_x50", |bench| {
        bench.iter(|| discrimination_gap(&vectors, &groups));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_distance_functions,
    bench_hnsw,
    bench_ivf,
    bench_eval
);
criterion_main!(benches);
