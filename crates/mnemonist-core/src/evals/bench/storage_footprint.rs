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

/// Results from the storage footprint experiment.
#[derive(Debug, Clone, Serialize)]
pub struct StorageResult {
    pub n_vectors: usize,
    pub dimension: usize,
    pub raw_embedding_bytes: usize,
    pub hnsw_index_bytes: usize,
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

        let recall_5 =
            compute_quantized_recall_any(&embeddings, &dequantized, &query_embeddings, &query_gold, 5);
        let recall_10 = compute_quantized_recall_any(
            &embeddings,
            &dequantized,
            &query_embeddings,
            &query_gold,
            10,
        );

        quantized.push(QuantizedStoragePoint {
            bits,
            compressed_bytes,
            compression_ratio: raw_bytes as f64 / compressed_bytes as f64,
            recall_any_at_5: recall_5,
            recall_any_at_10: recall_10,
            cosine_distortion: cos_dist,
        });
    }

    Ok(StorageResult {
        n_vectors: embeddings.len(),
        dimension: dim,
        raw_embedding_bytes: raw_bytes,
        hnsw_index_bytes: hnsw_bytes,
        quantized,
    })
}

/// Brute-force recall_any on dequantized vectors.
#[cfg(feature = "quant")]
fn compute_quantized_recall_any(
    embeddings: &[(String, Vec<f32>)],
    dequantized: &[Vec<f32>],
    query_embeddings: &[Vec<f32>],
    query_gold: &[Vec<String>],
    k: usize,
) -> f64 {
    use std::collections::HashSet;

    if query_embeddings.is_empty() {
        return 0.0;
    }

    let mut total_hits = 0;

    for (q_emb, gold_ids) in query_embeddings.iter().zip(query_gold) {
        let gold: HashSet<&str> = gold_ids.iter().map(|s| s.as_str()).collect();

        let mut scored: Vec<(&str, f32)> = dequantized
            .iter()
            .enumerate()
            .map(|(i, deq)| (embeddings[i].0.as_str(), cosine_similarity(q_emb, deq)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_k: HashSet<&str> = scored.iter().take(k).map(|(id, _)| *id).collect();
        if gold.iter().any(|g| top_k.contains(g)) {
            total_hits += 1;
        }
    }

    total_hits as f64 / query_embeddings.len() as f64
}

/// Embed all sessions and queries from the dataset.
#[cfg(feature = "quant")]
fn embed_dataset(
    dataset: &LongMemEvalDataset,
    embedder: &dyn Embedder,
) -> Result<(Vec<(String, Vec<f32>)>, Vec<Vec<f32>>, Vec<Vec<String>>), EvalError> {
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
