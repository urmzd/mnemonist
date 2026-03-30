//! Quantization fidelity validation tests.
//!
//! These tests assert quality thresholds for roundtrip accuracy,
//! serving as regression guards for quantization algorithms.

use llmem_quant::mse::TurboQuantMse;
use llmem_quant::pack;
use llmem_quant::prod::TurboQuantProd;

fn random_unit_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut v: Vec<f32> = (0..dim)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f32 / u64::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// ── MSE Roundtrip Fidelity ─────────────────────────────────────────────────

#[test]
fn mse_2bit_cosine_gte_090() {
    let dim = 128;
    let quant = TurboQuantMse::new(dim, 2, 42).unwrap();
    let x = random_unit_vector(dim, 7);
    let q = quant.quantize(&x).unwrap();
    let reconstructed = quant.dequantize(&q).unwrap();
    let sim = cosine_sim(&x, &reconstructed);
    assert!(sim >= 0.90, "MSE 2-bit cosine = {sim:.4}, expected >= 0.90");
}

#[test]
fn mse_4bit_cosine_gte_095() {
    let dim = 128;
    let quant = TurboQuantMse::new(dim, 4, 42).unwrap();
    let x = random_unit_vector(dim, 7);
    let q = quant.quantize(&x).unwrap();
    let reconstructed = quant.dequantize(&q).unwrap();
    let sim = cosine_sim(&x, &reconstructed);
    assert!(sim >= 0.95, "MSE 4-bit cosine = {sim:.4}, expected >= 0.95");
}

// ── Prod Inner Product Estimate ────────────────────────────────────────────

#[test]
fn prod_inner_product_estimate_bounded() {
    let dim = 128;

    // Tolerance scales with bit-width: low bits have higher quantization noise.
    let cases: &[(u8, f32)] = &[(2, 0.50), (3, 0.35), (4, 0.25)];

    for &(bits, max_error) in cases {
        let quant = TurboQuantProd::new(dim, bits, 42, 99).unwrap();
        let x = random_unit_vector(dim, 7);
        let query = random_unit_vector(dim, 99);
        let q = quant.quantize(&x).unwrap();

        let true_ip = dot_product(&x, &query);
        let est_ip = quant.inner_product_estimate(&query, &q).unwrap();

        let abs_error = (est_ip - true_ip).abs();

        assert!(
            abs_error < max_error,
            "Prod {bits}-bit IP estimate: true={true_ip:.4}, est={est_ip:.4}, abs_error={abs_error:.4}, expected < {max_error}"
        );
    }
}

// ── Pack/Unpack Exact Roundtrip ────────────────────────────────────────────

#[test]
fn pack_unpack_exact_roundtrip() {
    for &(dim, bits) in &[(128, 2u8), (384, 2), (384, 4)] {
        let indices: Vec<u8> = (0..dim).map(|i| (i % (1 << bits)) as u8).collect();
        let packed = pack::pack_indices(&indices, bits).unwrap();
        let unpacked = pack::unpack_indices(&packed, bits, dim).unwrap();
        assert_eq!(
            indices, unpacked,
            "pack/unpack roundtrip failed for {dim}x{bits}b"
        );
    }
}
