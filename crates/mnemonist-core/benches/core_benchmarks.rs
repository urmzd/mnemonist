use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use mnemonist_core::embed::EmbeddingStore;
use mnemonist_core::embed::content_hash;
use mnemonist_core::inbox::{Inbox, InboxItem};
use mnemonist_core::index::{IndexEntry, MemoryIndex};
use std::path::PathBuf;

fn make_vector(dim: usize, seed: f32) -> Vec<f32> {
    (0..dim).map(|i| ((i as f32 + seed) * 0.1).sin()).collect()
}

fn make_item(id: usize, score: f32) -> InboxItem {
    InboxItem {
        id: format!("item_{id}"),
        content: format!("content for item {id}"),
        source: "note".to_string(),
        attention_score: score,
        created_at: "2026-03-30T00:00:00Z".to_string(),
        file_source: None,
    }
}

fn make_entry(i: usize) -> IndexEntry {
    IndexEntry {
        title: format!("Memory {i}"),
        file: format!("feedback_mem-{i}.md"),
        summary: format!("summary for memory number {i}"),
    }
}

fn make_index(n: usize) -> MemoryIndex {
    MemoryIndex {
        entries: (0..n).map(make_entry).collect(),
        dir: PathBuf::from("/tmp/bench"),
    }
}

// ── Embedding Store ────────────────────────────────────────────────────────

fn bench_embedding_store(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding_store");

    for &(dim, count) in &[(128, 100), (384, 100), (384, 500)] {
        let label = format!("{dim}d_x{count}");

        // Benchmark upsert
        group.bench_with_input(BenchmarkId::new("upsert", &label), &count, |bench, &n| {
            bench.iter(|| {
                let mut store = EmbeddingStore::new(dim);
                for i in 0..n {
                    store.upsert(mnemonist_core::embed::EmbeddingEntry {
                        file: format!("mem_{i}.md"),
                        hash: i as u64,
                        embedding: make_vector(dim, i as f32),
                    });
                }
                store
            });
        });

        // Build store for get/remove/save/load benchmarks
        let mut store = EmbeddingStore::new(dim);
        for i in 0..count {
            store.upsert(mnemonist_core::embed::EmbeddingEntry {
                file: format!("mem_{i}.md"),
                hash: i as u64,
                embedding: make_vector(dim, i as f32),
            });
        }

        group.bench_with_input(BenchmarkId::new("get", &label), &count, |bench, _| {
            bench.iter(|| store.get("mem_50.md"));
        });

        group.bench_with_input(BenchmarkId::new("remove", &label), &count, |bench, _| {
            bench.iter(|| {
                let mut s = store.clone();
                s.remove("mem_50.md");
                s
            });
        });

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("bench.bin");
        store.save(&path).unwrap();

        group.bench_with_input(BenchmarkId::new("save", &label), &count, |bench, _| {
            bench.iter(|| store.save(&path).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("load", &label), &count, |bench, _| {
            bench.iter(|| EmbeddingStore::load(&path).unwrap());
        });
    }

    group.finish();
}

// ── Inbox ──────────────────────────────────────────────────────────────────

fn bench_inbox(c: &mut Criterion) {
    let mut group = c.benchmark_group("inbox");

    for &cap in &[7, 50] {
        // Push to capacity
        group.bench_with_input(
            BenchmarkId::new("push_to_capacity", cap),
            &cap,
            |bench, &cap| {
                bench.iter(|| {
                    let mut inbox = Inbox::new(cap);
                    for i in 0..cap {
                        inbox.push(make_item(i, (i as f32) / (cap as f32)));
                    }
                    inbox
                });
            },
        );

        // Push with eviction (push 2x capacity into a capacity-limited inbox)
        group.bench_with_input(
            BenchmarkId::new("push_with_eviction", cap),
            &cap,
            |bench, &cap| {
                bench.iter(|| {
                    let mut inbox = Inbox::new(cap);
                    for i in 0..(cap * 2) {
                        inbox.push(make_item(i, (i as f32) / (cap as f32 * 2.0)));
                    }
                    inbox
                });
            },
        );

        // Save/load
        let mut inbox = Inbox::new(cap);
        for i in 0..cap {
            inbox.push(make_item(i, (i as f32) / (cap as f32)));
        }

        let tmp = tempfile::tempdir().unwrap();
        inbox.save(tmp.path()).unwrap();

        group.bench_with_input(BenchmarkId::new("save", cap), &cap, |bench, _| {
            bench.iter(|| inbox.save(tmp.path()).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("load", cap), &cap, |bench, _| {
            bench.iter(|| Inbox::load(tmp.path(), cap).unwrap());
        });

        // Drain
        group.bench_with_input(BenchmarkId::new("drain", cap), &cap, |bench, &cap| {
            bench.iter(|| {
                let mut inbox = Inbox::new(cap);
                for i in 0..cap {
                    inbox.push(make_item(i, (i as f32) / (cap as f32)));
                }
                inbox.drain()
            });
        });
    }

    group.finish();
}

// ── Memory Index ───────────────────────────────────────────────────────────

fn bench_memory_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_index");

    // Parse / to_line on single entries
    let line =
        "- [Open Standards](feedback_open_standards.md) — use open specs, not proprietary tools";
    group.bench_function("parse_line", |bench| {
        bench.iter(|| IndexEntry::parse(line));
    });

    let entry = make_entry(0);
    group.bench_function("to_line", |bench| {
        bench.iter(|| entry.to_line());
    });

    // Search and upsert parameterized by count
    for &n in &[10, 100] {
        let index = make_index(n);

        group.bench_with_input(BenchmarkId::new("search", n), &n, |bench, _| {
            bench.iter(|| index.search("memory number 5"));
        });

        group.bench_with_input(BenchmarkId::new("upsert_new", n), &n, |bench, _| {
            bench.iter(|| {
                let mut idx = index.clone();
                idx.upsert(IndexEntry {
                    title: "New Entry".to_string(),
                    file: "feedback_new.md".to_string(),
                    summary: "a brand new entry".to_string(),
                });
                idx
            });
        });

        group.bench_with_input(BenchmarkId::new("upsert_existing", n), &n, |bench, _| {
            bench.iter(|| {
                let mut idx = index.clone();
                idx.upsert(IndexEntry {
                    title: "Updated".to_string(),
                    file: "feedback_mem-0.md".to_string(),
                    summary: "updated summary".to_string(),
                });
                idx
            });
        });
    }

    group.finish();
}

// ── Content Hash ───────────────────────────────────────────────────────────

fn bench_content_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("content_hash");

    for &len in &[100, 1_000, 10_000] {
        let text: String = "a".repeat(len);
        group.bench_with_input(BenchmarkId::new("hash", len), &len, |bench, _| {
            bench.iter(|| content_hash(&text));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_embedding_store,
    bench_inbox,
    bench_memory_index,
    bench_content_hash
);
criterion_main!(benches);
