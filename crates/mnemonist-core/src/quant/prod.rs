//! TurboQuant_prod: unbiased inner-product quantizer (Algorithm 2).
//!
//! Combines TurboQuant_mse (at b-1 bits) with a 1-bit QJL transform on the
//! residual to produce an unbiased inner product estimator at b total bits.
//!
//! Properties (Theorem 2):
//! - Unbiased: E[⟨y, x̃⟩] = ⟨y, x⟩
//! - Inner-product distortion: D_prod ≤ (√3π²·||y||²/d) · 1/4^b
//!
//! The total bit budget is b bits per coordinate:
//! - b-1 bits for the MSE component
//! - 1 bit for the QJL residual sign

use super::QuantError;
use super::mse::{QuantizedVector, TurboQuantMse};
use super::qjl::{QjlResult, QjlTransform};

/// Unbiased inner-product TurboQuant quantizer.
pub struct TurboQuantProd {
    mse: TurboQuantMse,
    qjl: QjlTransform,
    /// Total bit-width (mse uses bits-1, qjl uses 1).
    bits: u8,
}

/// A quantized vector produced by TurboQuant_prod.
#[derive(Debug, Clone)]
pub struct QuantizedProdVector {
    /// The MSE-quantized component (b-1 bits per coordinate).
    pub mse_part: QuantizedVector,
    /// QJL sign bits of the residual (1 bit per coordinate).
    pub qjl_part: QjlResult,
    /// L2 norm of the residual vector (γ in the paper).
    pub residual_norm: f32,
}

impl TurboQuantProd {
    /// Create a new inner-product quantizer.
    ///
    /// - `dimension`: vector dimensionality
    /// - `bits`: total bit-width per coordinate (must be ≥ 2)
    /// - `mse_seed`: seed for the MSE rotation matrix
    /// - `qjl_seed`: seed for the QJL projection (must differ from mse_seed)
    pub fn new(
        dimension: usize,
        bits: u8,
        mse_seed: u64,
        qjl_seed: u64,
    ) -> Result<Self, QuantError> {
        if bits < 2 {
            return Err(QuantError::UnsupportedBitWidth(bits));
        }

        let mse = TurboQuantMse::new(dimension, bits - 1, mse_seed)?;
        let qjl = QjlTransform::new(dimension, qjl_seed);

        Ok(Self { mse, qjl, bits })
    }

    /// The dimension this quantizer operates on.
    pub fn dimension(&self) -> usize {
        self.mse.dimension()
    }

    /// The total bit-width per coordinate.
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Quantize a vector.
    pub fn quantize(&self, x: &[f32]) -> Result<QuantizedProdVector, QuantError> {
        // Step 1: MSE quantize at b-1 bits
        let mse_part = self.mse.quantize(x)?;

        // Step 2: Dequantize to get MSE approximation
        let x_mse = self.mse.dequantize(&mse_part)?;

        // Step 3: Compute residual r = x - x̃_mse
        let residual: Vec<f32> = x.iter().zip(x_mse.iter()).map(|(a, b)| a - b).collect();

        // Step 4: Compute residual norm γ = ||r||
        let residual_norm: f32 = residual.iter().map(|v| v * v).sum::<f32>().sqrt();

        // Step 5: Apply QJL to residual
        let qjl_part = self.qjl.quantize(&residual);

        Ok(QuantizedProdVector {
            mse_part,
            qjl_part,
            residual_norm,
        })
    }

    /// Dequantize a vector.
    ///
    /// Returns x̃ = x̃_mse + x̃_qjl where x̃_qjl = (√(π/2)/d) · γ · S^T · qjl
    pub fn dequantize(&self, q: &QuantizedProdVector) -> Result<Vec<f32>, QuantError> {
        // Dequantize MSE component
        let x_mse = self.mse.dequantize(&q.mse_part)?;

        // Dequantize QJL component
        let x_qjl = self.qjl.dequantize(&q.qjl_part, q.residual_norm);

        // Sum
        let result: Vec<f32> = x_mse.iter().zip(x_qjl.iter()).map(|(a, b)| a + b).collect();

        Ok(result)
    }

    /// Estimate ⟨query, quantized_x⟩ without full dequantization.
    ///
    /// Computes: ⟨query, x̃_mse⟩ + QJL_estimate(query, qjl_bits, γ)
    pub fn inner_product_estimate(
        &self,
        query: &[f32],
        q: &QuantizedProdVector,
    ) -> Result<f32, QuantError> {
        // MSE component inner product (requires dequantization)
        let x_mse = self.mse.dequantize(&q.mse_part)?;
        let ip_mse: f32 = query.iter().zip(x_mse.iter()).map(|(a, b)| a * b).sum();

        // QJL component inner product (fast estimate)
        let ip_qjl = self
            .qjl
            .inner_product_estimate(query, &q.qjl_part, q.residual_norm);

        Ok(ip_mse + ip_qjl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_unit_vector(dim: usize, seed: u64) -> Vec<f32> {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        use rand_distr::{Distribution, StandardNormal};

        let mut rng = StdRng::seed_from_u64(seed);
        let normal = StandardNormal;
        let mut v: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= norm;
        }
        v
    }

    #[test]
    fn prod_requires_minimum_2_bits() {
        assert!(TurboQuantProd::new(32, 1, 1, 2).is_err());
        assert!(TurboQuantProd::new(32, 2, 1, 2).is_ok());
    }

    #[test]
    fn prod_quantize_dequantize() {
        let dim = 128;
        let quant = TurboQuantProd::new(dim, 3, 42, 99).unwrap();
        let x = random_unit_vector(dim, 7);

        let q = quant.quantize(&x).unwrap();
        let x_hat = quant.dequantize(&q).unwrap();

        assert_eq!(x_hat.len(), dim);

        // MSE should be reasonable
        let mse: f32 = x
            .iter()
            .zip(x_hat.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>();
        assert!(mse < 1.0, "MSE too high: {mse}");
    }

    #[test]
    fn unbiased_inner_product() {
        // E[⟨y, x̃⟩] ≈ ⟨y, x⟩ over many QJL seeds
        let dim = 128;
        let x = random_unit_vector(dim, 1);
        let y = random_unit_vector(dim, 2);
        let true_ip: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

        let n_trials = 100;
        let mut total_estimated = 0.0f32;

        for trial in 0..n_trials {
            let quant = TurboQuantProd::new(dim, 3, 42, trial as u64 + 100).unwrap();
            let q = quant.quantize(&x).unwrap();
            let x_hat = quant.dequantize(&q).unwrap();
            let estimated: f32 = y.iter().zip(x_hat.iter()).map(|(a, b)| a * b).sum();
            total_estimated += estimated;
        }

        let avg_estimated = total_estimated / n_trials as f32;
        let bias = (avg_estimated - true_ip).abs();
        assert!(
            bias < 0.1,
            "prod bias too large: avg={avg_estimated}, true={true_ip}, bias={bias}"
        );
    }

    #[test]
    fn inner_product_estimate_close() {
        let dim = 64;
        let quant = TurboQuantProd::new(dim, 3, 42, 99).unwrap();
        let x = random_unit_vector(dim, 1);
        let query = random_unit_vector(dim, 2);

        let q = quant.quantize(&x).unwrap();

        // Full dequantize + dot
        let x_hat = quant.dequantize(&q).unwrap();
        let explicit: f32 = query.iter().zip(x_hat.iter()).map(|(a, b)| a * b).sum();

        // Fast estimate
        let estimated = quant.inner_product_estimate(&query, &q).unwrap();

        assert!(
            (explicit - estimated).abs() < 1e-3,
            "explicit={explicit}, estimated={estimated}"
        );
    }

    #[test]
    fn residual_norm_positive() {
        let dim = 64;
        let quant = TurboQuantProd::new(dim, 3, 42, 99).unwrap();
        let x = random_unit_vector(dim, 1);

        let q = quant.quantize(&x).unwrap();
        assert!(q.residual_norm > 0.0);
        assert!(q.residual_norm < 1.0); // residual of unit vector should be < 1
    }
}
