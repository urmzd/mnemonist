//! Experiment 3: Storage Footprint
//!
//! Measures raw embedding size, HNSW index size, and TurboQuant compressed
//! sizes at each bit-width, with recall degradation tracking.

use serde::Serialize;

use crate::ann::AnnIndex;
use crate::ann::hnsw::HnswIndex;
use crate::distance::cosine_similarity;
use crate::embed::Embedder;

use crate::evals::EvalError;
use crate::evals::longmemeval::LongMemEvalDataset;
use crate::evals::search::{discordant_pairs, mcnemar_exact_p};

/// Results from the storage footprint experiment.
#[derive(Debug, Clone, Serialize)]
pub struct StorageResult {
    pub n_vectors: usize,
    pub dimension: usize,
    pub n_queries: usize,
    pub raw_embedding_bytes: usize,
    pub hnsw_index_bytes: usize,
    /// Raw (unquantized) recall baseline, so quantization degradation is
    /// measurable. NOTE: this is GLOBAL retrieval over every session (the
    /// hardest setting), not the per-question haystack used in Exp 1/5.
    pub raw_recall_any_at_5: f64,
    pub raw_recall_any_at_10: f64,
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
    /// Exact two-sided McNemar p-values vs the raw baseline on paired
    /// per-query hits — small recall deltas at this n are usually noise.
    pub mcnemar_p_vs_raw_at_5: f64,
    pub mcnemar_p_vs_raw_at_10: f64,
    pub cosine_distortion: f64,
}

/// Run Experiment 3: storage footprint comparison.
///
/// Embeds all sessions and queries, then measures raw vs quantized storage
/// and recall degradation at each bit-width.
#[cfg(feature = "quant")]
pub fn run(
    dataset: &LongMemEvalDataset,
    embedder: &dyn Embedder,
    bits_range: &[u8],
    seed: u64,
) -> Result<StorageResult, EvalError> {
    use crate::quant::TurboQuantMse;

    let (embeddings, query_embeddings, query_gold) = embed_dataset(dataset, embedder)?;

    let dim = embeddings
        .first()
        .ok_or(EvalError::InsufficientData { min: 1, got: 0 })?
        .1
        .len();

    let raw_bytes = embeddings.len() * dim * 4;

    let mut index = HnswIndex::with_defaults(dim);
    for (id, emb) in &embeddings {
        index
            .insert(id, emb)
            .map_err(|e| EvalError::Other(e.to_string()))?;
    }
    let tmp_path = std::env::temp_dir().join("mnemonist_bench_hnsw.bin");
    index
        .save(&tmp_path)
        .map_err(|e| EvalError::Other(e.to_string()))?;
    let hnsw_bytes = std::fs::metadata(&tmp_path)
        .map(|m| m.len() as usize)
        .unwrap_or(0);
    let _ = std::fs::remove_file(&tmp_path);

    // Raw (unquantized) recall baseline — the reference point for degradation.
    let raw_vecs: Vec<Vec<f32>> = embeddings.iter().map(|(_, v)| v.clone()).collect();
    let raw_hits_5 = compute_hits(&embeddings, &raw_vecs, &query_embeddings, &query_gold, 5);
    let raw_hits_10 = compute_hits(&embeddings, &raw_vecs, &query_embeddings, &query_gold, 10);
    let raw_recall_5 = recall_from_hits(&raw_hits_5);
    let raw_recall_10 = recall_from_hits(&raw_hits_10);

    let mut quantized = Vec::new();

    for &bits in bits_range {
        let quant = TurboQuantMse::new(dim, bits, seed)?;

        let dequantized: Vec<Vec<f32>> = embeddings
            .iter()
            .map(|(_, v)| {
                let q = quant.quantize(v).unwrap();
                quant.dequantize(&q).unwrap()
            })
            .collect();

        let total_dist: f64 = embeddings
            .iter()
            .zip(dequantized.iter())
            .map(|((_, orig), deq)| (1.0 - cosine_similarity(orig, deq)) as f64)
            .sum();
        let cos_dist = total_dist / embeddings.len() as f64;

        let bytes_per_vec = (dim * bits as usize).div_ceil(8) + 4;
        let compressed_bytes = embeddings.len() * bytes_per_vec;

        let hits_5 = compute_hits(&embeddings, &dequantized, &query_embeddings, &query_gold, 5);
        let hits_10 = compute_hits(
            &embeddings,
            &dequantized,
            &query_embeddings,
            &query_gold,
            10,
        );

        let (b5, c5) = discordant_pairs(&hits_5, &raw_hits_5);
        let (b10, c10) = discordant_pairs(&hits_10, &raw_hits_10);

        quantized.push(QuantizedStoragePoint {
            bits,
            compressed_bytes,
            compression_ratio: raw_bytes as f64 / compressed_bytes as f64,
            recall_any_at_5: recall_from_hits(&hits_5),
            recall_any_at_10: recall_from_hits(&hits_10),
            mcnemar_p_vs_raw_at_5: mcnemar_exact_p(b5, c5),
            mcnemar_p_vs_raw_at_10: mcnemar_exact_p(b10, c10),
            cosine_distortion: cos_dist,
        });
    }

    Ok(StorageResult {
        n_vectors: embeddings.len(),
        dimension: dim,
        n_queries: query_embeddings.len(),
        raw_embedding_bytes: raw_bytes,
        hnsw_index_bytes: hnsw_bytes,
        raw_recall_any_at_5: raw_recall_5,
        raw_recall_any_at_10: raw_recall_10,
        quantized,
    })
}

/// Fraction of hits in a per-query hit vector (0.0 for an empty vector).
#[cfg(feature = "quant")]
fn recall_from_hits(hits: &[bool]) -> f64 {
    if hits.is_empty() {
        return 0.0;
    }
    hits.iter().filter(|h| **h).count() as f64 / hits.len() as f64
}

/// Brute-force per-query recall_any hits on (de)quantized vectors.
#[cfg(feature = "quant")]
fn compute_hits(
    embeddings: &[(String, Vec<f32>)],
    dequantized: &[Vec<f32>],
    query_embeddings: &[Vec<f32>],
    query_gold: &[Vec<String>],
    k: usize,
) -> Vec<bool> {
    use std::collections::HashSet;

    let mut per_query = Vec::with_capacity(query_embeddings.len());

    for (q_emb, gold_ids) in query_embeddings.iter().zip(query_gold) {
        let gold: HashSet<&str> = gold_ids.iter().map(|s| s.as_str()).collect();

        let mut scored: Vec<(&str, f32)> = dequantized
            .iter()
            .enumerate()
            .map(|(i, deq)| (embeddings[i].0.as_str(), cosine_similarity(q_emb, deq)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_k: HashSet<&str> = scored.iter().take(k).map(|(id, _)| *id).collect();
        per_query.push(gold.iter().any(|g| top_k.contains(g)));
    }

    per_query
}

#[cfg(feature = "quant")]
type SessionEmbeddings = Vec<(String, Vec<f32>)>;
#[cfg(feature = "quant")]
type QueryEmbeddings = Vec<Vec<f32>>;
#[cfg(feature = "quant")]
type QueryGold = Vec<Vec<String>>;

/// Embed all sessions and queries from the dataset.
#[cfg(feature = "quant")]
fn embed_dataset(
    dataset: &LongMemEvalDataset,
    embedder: &dyn Embedder,
) -> Result<(SessionEmbeddings, QueryEmbeddings, QueryGold), EvalError> {
    const BATCH_SIZE: usize = 256;

    let total_sessions = dataset.sessions.len();
    eprintln!("  Embedding {total_sessions} sessions...");
    let session_ids: Vec<String> = dataset.sessions.keys().cloned().collect();
    let session_texts: Vec<&str> = session_ids
        .iter()
        .map(|id| dataset.sessions[id].as_str())
        .collect();

    let mut session_embeddings: Vec<Vec<f32>> = Vec::with_capacity(total_sessions);
    for chunk in session_texts.chunks(BATCH_SIZE) {
        let batch = embedder
            .embed_batch(chunk)
            .map_err(|e| EvalError::Other(e.to_string()))?;
        session_embeddings.extend(batch);
    }

    let embeddings: Vec<(String, Vec<f32>)> =
        session_ids.into_iter().zip(session_embeddings).collect();

    let total_queries = dataset.queries.len();
    eprintln!("  Embedding {total_queries} queries...");
    let query_texts: Vec<&str> = dataset
        .queries
        .iter()
        .map(|q| q.question.as_str())
        .collect();

    let mut query_embeddings: Vec<Vec<f32>> = Vec::with_capacity(total_queries);
    for chunk in query_texts.chunks(BATCH_SIZE) {
        let batch = embedder
            .embed_batch(chunk)
            .map_err(|e| EvalError::Other(e.to_string()))?;
        query_embeddings.extend(batch);
    }

    let query_gold: Vec<Vec<String>> = dataset
        .queries
        .iter()
        .map(|q| q.gold_session_ids.clone())
        .collect();

    Ok((embeddings, query_embeddings, query_gold))
}
