//! Quantization quality evaluation.
//!
//! Measures fidelity of TurboQuant quantizers at various bit-widths:
//! MSE, cosine distortion, inner-product bias, and ANN recall impact.

use mnemonist_core::distance::{cosine_similarity, dot_product};
use mnemonist_quant::{TurboQuantMse, TurboQuantProd};
use serde::{Deserialize, Serialize};

use crate::EvalError;

/// Quantization fidelity metrics at a given bit-width.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantMetrics {
    pub bits: u8,
    /// Mean squared error between original and dequantized vectors.
    pub mean_mse: f64,
    /// Maximum MSE across all test vectors.
    pub max_mse: f64,
    /// Average cosine distortion: 1 - cos(x, x_hat).
    pub cosine_distortion: f64,
    /// Compression ratio: bytes_original / bytes_quantized.
    pub compression_ratio: f64,
    pub n_vectors: usize,
}

/// Inner-product preservation metrics for the product quantizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProdMetrics {
    pub bits: u8,
    /// Average absolute inner-product error: |<y,x> - <y,x_hat>|.
    pub mean_ip_error: f64,
    /// Maximum inner-product error.
    pub max_ip_error: f64,
    /// Average bias: mean(<y,x_hat>) - <y,x>. Should be near 0 for unbiased quantizer.
    pub mean_ip_bias: f64,
    pub n_pairs: usize,
}

/// Recall comparison before and after quantization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallComparison {
    pub recall_original: f64,
    pub recall_quantized: f64,
    pub recall_delta: f64,
    pub bits: u8,
    pub k: usize,
}

/// Evaluate MSE quantizer across multiple bit-widths.
pub fn evaluate_mse_quantizer(
    vectors: &[Vec<f32>],
    bits_range: &[u8],
    seed: u64,
) -> Result<Vec<QuantMetrics>, EvalError> {
    if vectors.is_empty() {
        return Err(EvalError::InsufficientData { min: 1, got: 0 });
    }

    let dim = vectors[0].len();
    let bytes_original = (dim * 4) as f64; // f32 = 4 bytes

    let mut results = Vec::with_capacity(bits_range.len());
    for &bits in bits_range {
        let quant = TurboQuantMse::new(dim, bits, seed)?;

        let mut total_mse = 0.0f64;
        let mut max_mse = 0.0f64;
        let mut total_cos_dist = 0.0f64;

        for v in vectors {
            let q = quant.quantize(v)?;
            let v_hat = quant.dequantize(&q)?;

            let mse: f64 = v
                .iter()
                .zip(v_hat.iter())
                .map(|(a, b)| {
                    let d = *a as f64 - *b as f64;
                    d * d
                })
                .sum();
            total_mse += mse;
            max_mse = max_mse.max(mse);

            let cos = cosine_similarity(v, &v_hat);
            total_cos_dist += (1.0 - cos) as f64;
        }

        let n = vectors.len() as f64;
        // bits per coordinate → bytes per vector
        let bytes_quantized = (dim as f64 * bits as f64) / 8.0 + 4.0; // +4 for norm

        results.push(QuantMetrics {
            bits,
            mean_mse: total_mse / n,
            max_mse,
            cosine_distortion: total_cos_dist / n,
            compression_ratio: bytes_original / bytes_quantized,
            n_vectors: vectors.len(),
        });
    }

    Ok(results)
}

/// Evaluate product quantizer for inner-product preservation.
pub fn evaluate_prod_quantizer(
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    bits_range: &[u8],
    mse_seed: u64,
    qjl_seed: u64,
) -> Result<Vec<ProdMetrics>, EvalError> {
    if vectors.is_empty() || queries.is_empty() {
        return Err(EvalError::InsufficientData { min: 1, got: 0 });
    }

    let dim = vectors[0].len();
    let mut results = Vec::with_capacity(bits_range.len());

    for &bits in bits_range {
        let quant = TurboQuantProd::new(dim, bits, mse_seed, qjl_seed)?;

        let mut total_error = 0.0f64;
        let mut max_error = 0.0f64;
        let mut total_bias = 0.0f64;
        let mut count = 0usize;

        for v in vectors {
            let q = quant.quantize(v)?;
            let v_hat = quant.dequantize(&q)?;

            for query in queries {
                let true_ip = dot_product(v, query) as f64;
                let est_ip = dot_product(&v_hat, query) as f64;
                let error = (est_ip - true_ip).abs();
                let bias = est_ip - true_ip;

                total_error += error;
                max_error = max_error.max(error);
                total_bias += bias;
                count += 1;
            }
        }

        let n = count as f64;
        results.push(ProdMetrics {
            bits,
            mean_ip_error: total_error / n,
            max_ip_error: max_error,
            mean_ip_bias: total_bias / n,
            n_pairs: count,
        });
    }

    Ok(results)
}

/// Compare ANN recall before and after quantization.
///
/// Builds brute-force rankings on original vectors vs dequantized vectors
/// and measures recall degradation.
#[cfg(feature = "index")]
pub fn quantization_recall_impact(
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    bits: u8,
    k: usize,
    seed: u64,
) -> Result<RecallComparison, EvalError> {
    if vectors.is_empty() || queries.is_empty() {
        return Err(EvalError::InsufficientData { min: 1, got: 0 });
    }

    let dim = vectors[0].len();
    let quant = TurboQuantMse::new(dim, bits, seed)?;

    // Dequantize all vectors
    let dequantized: Vec<Vec<f32>> = vectors
        .iter()
        .map(|v| {
            let q = quant.quantize(v).unwrap();
            quant.dequantize(&q).unwrap()
        })
        .collect();

    let mut total_recall_orig = 0.0f64;
    let mut total_recall_quant = 0.0f64;

    for query in queries {
        // Ground truth: brute-force on original vectors
        let mut scored_orig: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, cosine_similarity(query, v)))
            .collect();
        scored_orig.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let truth: std::collections::HashSet<usize> =
            scored_orig.iter().take(k).map(|(i, _)| *i).collect();

        // Recall on original (= 1.0 by definition)
        total_recall_orig += 1.0;

        // Recall on dequantized: rank by similarity to dequantized, check overlap with truth
        let mut scored_quant: Vec<(usize, f32)> = dequantized
            .iter()
            .enumerate()
            .map(|(i, v)| (i, cosine_similarity(query, v)))
            .collect();
        scored_quant.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let quant_top: std::collections::HashSet<usize> =
            scored_quant.iter().take(k).map(|(i, _)| *i).collect();

        let hits = truth.intersection(&quant_top).count();
        total_recall_quant += hits as f64 / k as f64;
    }

    let n = queries.len() as f64;
    let recall_original = total_recall_orig / n;
    let recall_quantized = total_recall_quant / n;

    Ok(RecallComparison {
        recall_original,
        recall_quantized,
        recall_delta: recall_quantized - recall_original,
        bits,
        k,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_unit_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        use rand_distr::{Distribution, StandardNormal};

        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim).map(|_| StandardNormal.sample(&mut rng)).collect();
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                for x in &mut v {
                    *x /= norm;
                }
                v
            })
            .collect()
    }

    #[test]
    fn mse_decreases_with_bits() {
        let vecs = random_unit_vectors(50, 128, 42);
        let results = evaluate_mse_quantizer(&vecs, &[1, 2, 3, 4], 42).unwrap();

        for w in results.windows(2) {
            assert!(
                w[1].mean_mse < w[0].mean_mse,
                "{}-bit mse ({}) not less than {}-bit ({})",
                w[1].bits,
                w[1].mean_mse,
                w[0].bits,
                w[0].mean_mse
            );
        }
    }

    #[test]
    fn compression_ratio_increases_with_fewer_bits() {
        let vecs = random_unit_vectors(20, 64, 42);
        let results = evaluate_mse_quantizer(&vecs, &[1, 2, 4], 42).unwrap();

        // 1-bit should have higher compression ratio than 4-bit
        assert!(results[0].compression_ratio > results[2].compression_ratio);
    }

    #[test]
    fn cosine_distortion_decreases_with_bits() {
        let vecs = random_unit_vectors(50, 128, 42);
        let results = evaluate_mse_quantizer(&vecs, &[1, 2, 3, 4], 42).unwrap();
        for w in results.windows(2) {
            assert!(
                w[1].cosine_distortion < w[0].cosine_distortion,
                "{}-bit distortion ({}) not less than {}-bit ({})",
                w[1].bits,
                w[1].cosine_distortion,
                w[0].bits,
                w[0].cosine_distortion
            );
        }
        // 4-bit should have lower distortion than 1-bit
        assert!(results[3].cosine_distortion < results[0].cosine_distortion);
    }

    #[test]
    fn prod_bias_near_zero() {
        let vecs = random_unit_vectors(30, 128, 1);
        let queries = random_unit_vectors(30, 128, 2);
        let results = evaluate_prod_quantizer(&vecs, &queries, &[3], 42, 99).unwrap();
        // Bias should be small (unbiased estimator)
        assert!(
            results[0].mean_ip_bias.abs() < 0.1,
            "bias = {}",
            results[0].mean_ip_bias
        );
    }

    #[cfg(feature = "index")]
    #[test]
    fn recall_degrades_gracefully() {
        let vecs = random_unit_vectors(100, 32, 42);
        let queries = random_unit_vectors(20, 32, 99);
        let result = quantization_recall_impact(&vecs, &queries, 3, 10, 42).unwrap();
        // 3-bit quantization should preserve reasonable recall
        assert!(
            result.recall_quantized > 0.5,
            "recall = {}",
            result.recall_quantized
        );
        assert!(result.recall_delta <= 0.0); // can't be better than original
    }
}
