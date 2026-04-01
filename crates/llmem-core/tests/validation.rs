//! Core storage fidelity validation tests.
//!
//! These tests assert that binary/JSON roundtrips preserve data exactly,
//! serving as regression guards for storage formats.

use llmem_core::embed::{EmbeddingEntry, EmbeddingStore};
use llmem_core::inbox::{Inbox, InboxItem};

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

// ── EmbeddingStore Binary Fidelity ─────────────────────────────────────────

#[test]
fn embedding_store_roundtrip_bit_exact() {
    let dim = 384;
    let count = 100;
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("embeddings.bin");

    let mut store = EmbeddingStore::new(dim);
    for i in 0..count {
        store.upsert(EmbeddingEntry {
            file: format!("mem_{i}.md"),
            hash: i as u64 * 12345,
            embedding: make_vector(dim, i as f32),
        });
    }

    store.save(&path).unwrap();
    let loaded = EmbeddingStore::load(&path).unwrap();

    assert_eq!(loaded.dimension, dim);
    assert_eq!(loaded.entries.len(), count);

    for (orig, loaded) in store.entries.iter().zip(loaded.entries.iter()) {
        assert_eq!(orig.file, loaded.file);
        assert_eq!(orig.hash, loaded.hash);
        assert_eq!(
            orig.embedding, loaded.embedding,
            "embedding mismatch for {}",
            orig.file
        );
    }
}

// ── Inbox Roundtrip with Eviction ──────────────────────────────────────────

#[test]
fn inbox_roundtrip_with_eviction() {
    let capacity = 7;
    let tmp = tempfile::tempdir().unwrap();

    let mut inbox = Inbox::new(capacity);
    for i in 0..20 {
        inbox.push(make_item(i, (i as f32) / 20.0));
    }

    assert_eq!(
        inbox.len(),
        capacity,
        "inbox should be at capacity after 20 pushes"
    );

    // Verify descending score order
    for w in inbox.items.windows(2) {
        assert!(
            w[0].attention_score >= w[1].attention_score,
            "inbox items not sorted: {} >= {} failed",
            w[0].attention_score,
            w[1].attention_score
        );
    }

    inbox.save(tmp.path()).unwrap();
    let loaded = Inbox::load(tmp.path(), capacity).unwrap();

    assert_eq!(loaded.len(), capacity);
    assert_eq!(loaded.capacity, capacity);

    for (orig, loaded) in inbox.items.iter().zip(loaded.items.iter()) {
        assert_eq!(orig.id, loaded.id);
        assert_eq!(orig.attention_score, loaded.attention_score);
        assert_eq!(orig.content, loaded.content);
    }
}
