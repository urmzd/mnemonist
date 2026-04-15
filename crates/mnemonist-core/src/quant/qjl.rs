//! Quantized Johnson-Lindenstrauss (QJL) 1-bit transform.
//!
//! QJL maps a vector to sign bits of a random Gaussian projection:
//!   Q_qjl(x) = sign(S · x)  where S ∈ R^{d×d}, S_{i,j} ~ N(0,1)
//!
//! The inverse/dequantization map is:
//!   Q_qjl^{-1}(z) = (√(π/2) / d) · S^T · z
//!
//! Properties (Lemma 4 in the paper):
//! - Unbiased: E[⟨y, Q_qjl^{-1}(Q_qjl(x))⟩] = ⟨y, x⟩
//! - Variance: Var(⟨y, Q_qjl^{-1}(Q_qjl(x))⟩) ≤ (π / 2d) · ||y||²
//!
//! The S matrix is not stored explicitly — it is generated on-the-fly from
//! a seed using a PRNG, making this memory-efficient.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

use super::pack;

/// QJL 1-bit transform.
///
/// Uses a seeded PRNG to generate the projection matrix on-the-fly,
/// avoiding O(d²) storage.
#[derive(Debug, Clone)]
pub struct QjlTransform {
    dim: usize,
    seed: u64,
}

/// Result of a QJL quantization.
#[derive(Debug, Clone)]
pub struct QjlResult {
    /// Packed sign bits (1 bit per dimension).
    pub packed_signs: Vec<u8>,
    /// Dimension of the original vector.
    pub dimension: usize,
}

impl QjlTransform {
    /// Create a new QJL transform.
    pub fn new(dim: usize, seed: u64) -> Self {
        Self { dim, seed }
    }

    /// Quantize: compute sign(S · x).
    ///
    /// Returns packed sign bits (±1 as 0/1 bits).
    pub fn quantize(&self, x: &[f32]) -> QjlResult {
        debug_assert_eq!(x.len(), self.dim);

        let mut rng = StdRng::seed_from_u64(self.seed);
        let normal = StandardNormal;

        let mut signs = Vec::with_capacity(self.dim);

        // Compute sign(S_i · x) for each row i of S
        for _ in 0..self.dim {
            let mut dot = 0.0f32;
            for &xj in x {
                let s_ij: f32 = normal.sample(&mut rng);
                dot += s_ij * xj;
            }
            signs.push(if dot >= 0.0 { 1i8 } else { -1i8 });
        }

        QjlResult {
            packed_signs: pack::pack_signs(&signs),
            dimension: self.dim,
        }
    }

    /// Dequantize: compute (√(π/2) / d) · S^T · z.
    ///
    /// `gamma` is a scaling factor (typically the residual norm).
    pub fn dequantize(&self, qjl: &QjlResult, gamma: f32) -> Vec<f32> {
        debug_assert_eq!(qjl.dimension, self.dim);

        let signs = pack::unpack_signs(&qjl.packed_signs, self.dim);
        let scale = gamma * (std::f32::consts::FRAC_PI_2).sqrt() / self.dim as f32;

        let mut result = vec![0.0f32; self.dim];
        let mut rng = StdRng::seed_from_u64(self.seed);
        let normal = StandardNormal;

        // S^T · z: for each row i of S (= column i of S^T),
        // add z[i] * S[i,:] to the result
        for &sign in signs.iter().take(self.dim) {
            let z_i = sign as f32; // ±1
            for item in result.iter_mut().take(self.dim) {
                let s_ij: f32 = normal.sample(&mut rng);
                *item += z_i * s_ij;
            }
        }

        // Apply scale
        for val in &mut result {
            *val *= scale;
        }

        result
    }

    /// Estimate ⟨query, dequant(qjl_result)⟩ without materializing the dequantized vector.
    ///
    /// Computes: gamma · (√(π/2) / d) · ⟨query, S^T · z⟩
    ///         = gamma · (√(π/2) / d) · Σ_i z_i · ⟨query, S_{i,:}⟩
    pub fn inner_product_estimate(&self, query: &[f32], qjl: &QjlResult, gamma: f32) -> f32 {
        debug_assert_eq!(query.len(), self.dim);

        let signs = pack::unpack_signs(&qjl.packed_signs, self.dim);
        let scale = gamma * (std::f32::consts::FRAC_PI_2).sqrt() / self.dim as f32;

        let mut rng = StdRng::seed_from_u64(self.seed);
        let normal = StandardNormal;

        let mut estimate = 0.0f32;

        for &sign in signs.iter().take(self.dim) {
            let z_i = sign as f32;
            let mut dot_query_row = 0.0f32;
            for &qj in query {
                let s_ij: f32 = normal.sample(&mut rng);
                dot_query_row += qj * s_ij;
            }
            estimate += z_i * dot_query_row;
        }

        estimate * scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vector(dim: usize, seed: u64) -> Vec<f32> {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        use rand_distr::{Distribution, StandardNormal};

        let mut rng = StdRng::seed_from_u64(seed);
        let normal = StandardNormal;
        (0..dim).map(|_| normal.sample(&mut rng)).collect()
    }

    fn random_unit_vector(dim: usize, seed: u64) -> Vec<f32> {
        let mut v = random_vector(dim, seed);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= norm;
        }
        v
    }

    #[test]
    fn qjl_output_dimensions() {
        let dim = 64;
        let qjl = QjlTransform::new(dim, 42);
        let x = random_unit_vector(dim, 7);

        let result = qjl.quantize(&x);
        assert_eq!(result.dimension, dim);
        assert_eq!(result.packed_signs.len(), (dim + 7) / 8);
    }

    #[test]
    fn qjl_deterministic() {
        let dim = 32;
        let qjl = QjlTransform::new(dim, 42);
        let x = random_unit_vector(dim, 7);

        let r1 = qjl.quantize(&x);
        let r2 = qjl.quantize(&x);
        assert_eq!(r1.packed_signs, r2.packed_signs);
    }

    #[test]
    fn unbiased_inner_product() {
        // E[⟨y, dequant(quant(x))⟩] ≈ ⟨y, x⟩ over many trials
        let dim = 64;
        let x = random_unit_vector(dim, 1);
        let y = random_unit_vector(dim, 2);
        let true_ip: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

        let n_trials = 200;
        let mut total_estimated = 0.0f32;

        for trial in 0..n_trials {
            let qjl = QjlTransform::new(dim, trial + 100);
            let result = qjl.quantize(&x);
            let x_hat = qjl.dequantize(&result, 1.0);
            let estimated: f32 = y.iter().zip(x_hat.iter()).map(|(a, b)| a * b).sum();
            total_estimated += estimated;
        }

        let avg_estimated = total_estimated / n_trials as f32;
        let bias = (avg_estimated - true_ip).abs();
        assert!(
            bias < 0.15,
            "QJL bias too large: avg={avg_estimated}, true={true_ip}, bias={bias}"
        );
    }

    #[test]
    fn inner_product_estimate_matches_explicit() {
        let dim = 32;
        let qjl = QjlTransform::new(dim, 42);
        let x = random_unit_vector(dim, 1);
        let query = random_unit_vector(dim, 2);

        let result = qjl.quantize(&x);
        let gamma = 1.0;

        // Explicit dequantize then dot product
        let x_hat = qjl.dequantize(&result, gamma);
        let explicit: f32 = query.iter().zip(x_hat.iter()).map(|(a, b)| a * b).sum();

        // Fast estimate
        let estimated = qjl.inner_product_estimate(&query, &result, gamma);

        assert!(
            (explicit - estimated).abs() < 1e-4,
            "explicit={explicit}, estimated={estimated}"
        );
    }
}
