//! TurboQuant_mse: MSE-optimal vector quantizer (Algorithm 1).
//!
//! Quantizes d-dimensional vectors to b bits per coordinate by:
//! 1. Normalizing to unit norm (storing the original norm separately)
//! 2. Applying a random orthogonal rotation Π
//! 3. Scalar-quantizing each coordinate with a precomputed Lloyd-Max codebook
//!
//! Achieves MSE distortion D_mse ≤ (√3π/2) · 1/4^b.

use super::QuantError;
use super::codebook::Codebook;
use super::pack;
use super::rotation::Rotation;

/// MSE-optimal TurboQuant quantizer.
pub struct TurboQuantMse {
    rotation: Rotation,
    codebook: &'static Codebook,
    bits: u8,
    /// Scaling factor: codebook centroids are for N(0,1); after rotation,
    /// unit-sphere coordinates have variance ≈ 1/d. We scale coordinates
    /// by sqrt(d) before quantization so the codebook applies directly.
    scale: f32,
}

/// A quantized vector produced by TurboQuant_mse.
#[derive(Debug, Clone)]
pub struct QuantizedVector {
    /// Packed b-bit indices, one per coordinate.
    pub packed_indices: Vec<u8>,
    /// Original vector norm (for rescaling on dequantization).
    pub norm: f32,
    /// Bit-width used.
    pub bits: u8,
    /// Number of coordinates (dimension).
    pub dimension: usize,
}

impl TurboQuantMse {
    /// Create a new MSE-optimal quantizer.
    ///
    /// - `dimension`: vector dimensionality
    /// - `bits`: quantization bit-width (1-4)
    /// - `seed`: RNG seed for the rotation matrix (must match for quant/dequant)
    pub fn new(dimension: usize, bits: u8, seed: u64) -> Result<Self, QuantError> {
        let codebook = Codebook::for_bits(bits)?;
        let rotation = Rotation::new(dimension, seed);
        let scale = (dimension as f32).sqrt();

        Ok(Self {
            rotation,
            codebook,
            bits,
            scale,
        })
    }

    /// The dimension this quantizer operates on.
    pub fn dimension(&self) -> usize {
        self.rotation.dimension()
    }

    /// The bit-width per coordinate.
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// The rotation seed.
    pub fn seed(&self) -> u64 {
        self.rotation.seed()
    }

    /// Quantize a vector.
    pub fn quantize(&self, x: &[f32]) -> Result<QuantizedVector, QuantError> {
        let dim = self.rotation.dimension();
        if x.len() != dim {
            return Err(QuantError::DimensionMismatch {
                expected: dim,
                got: x.len(),
            });
        }

        // Compute norm and normalize
        let norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let mut y = if norm > 0.0 {
            x.iter().map(|v| v / norm).collect::<Vec<_>>()
        } else {
            vec![0.0; dim]
        };

        // Apply rotation: y = Π · x_normalized
        self.rotation.forward(&mut y);

        // Scale by sqrt(d) to match codebook domain
        for val in &mut y {
            *val *= self.scale;
        }

        // Scalar-quantize each coordinate
        let indices: Vec<u8> = y
            .iter()
            .map(|&v| self.codebook.quantize_scalar(v))
            .collect();

        let packed_indices = pack::pack_indices(&indices, self.bits)?;

        Ok(QuantizedVector {
            packed_indices,
            norm,
            bits: self.bits,
            dimension: dim,
        })
    }

    /// Dequantize a vector back to approximate floats.
    pub fn dequantize(&self, q: &QuantizedVector) -> Result<Vec<f32>, QuantError> {
        let dim = q.dimension;
        let indices = pack::unpack_indices(&q.packed_indices, q.bits, dim)?;

        // Look up centroids
        let mut y: Vec<f32> = indices
            .iter()
            .map(|&idx| self.codebook.dequantize_scalar(idx))
            .collect();

        // Unscale
        let inv_scale = 1.0 / self.scale;
        for val in &mut y {
            *val *= inv_scale;
        }

        // Apply inverse rotation: x = Π^T · y
        self.rotation.inverse(&mut y);

        // Rescale by original norm
        for val in &mut y {
            *val *= q.norm;
        }

        Ok(y)
    }

    /// Dequantize into a pre-allocated buffer (avoids allocation).
    pub fn dequantize_into(&self, q: &QuantizedVector, out: &mut [f32]) -> Result<(), QuantError> {
        let result = self.dequantize(q)?;
        out.copy_from_slice(&result);
        Ok(())
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
    fn quantize_dequantize_roundtrip() {
        let dim = 128;
        let quant = TurboQuantMse::new(dim, 2, 42).unwrap();

        let x = random_unit_vector(dim, 7);
        let q = quant.quantize(&x).unwrap();
        let x_hat = quant.dequantize(&q).unwrap();

        assert_eq!(x_hat.len(), dim);

        // Compute MSE
        let mse: f32 = x
            .iter()
            .zip(x_hat.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>();

        // For 2-bit, theoretical bound is D_mse ≤ (√3π/2) · 1/4^2 ≈ 0.170
        // But on unit vectors ||x||=1 so MSE = D_mse
        assert!(mse < 0.5, "MSE too high: {mse} (expected < 0.5 for 2-bit)");
    }

    #[test]
    fn mse_decreases_with_bits() {
        let dim = 256;
        let x = random_unit_vector(dim, 13);
        let mut prev_mse = f32::MAX;

        for bits in 1..=4 {
            let quant = TurboQuantMse::new(dim, bits, 42).unwrap();
            let q = quant.quantize(&x).unwrap();
            let x_hat = quant.dequantize(&q).unwrap();

            let mse: f32 = x
                .iter()
                .zip(x_hat.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>();

            assert!(
                mse < prev_mse,
                "{bits}-bit MSE ({mse}) not less than {}-bit ({prev_mse})",
                bits - 1
            );
            prev_mse = mse;
        }
    }

    #[test]
    fn preserves_norm() {
        let dim = 64;
        let quant = TurboQuantMse::new(dim, 3, 42).unwrap();

        // Non-unit vector
        let x: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let norm_orig: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();

        let q = quant.quantize(&x).unwrap();
        let x_hat = quant.dequantize(&q).unwrap();
        let norm_hat: f32 = x_hat.iter().map(|v| v * v).sum::<f32>().sqrt();

        // Norm should be approximately preserved (within quantization error)
        assert!(
            (norm_orig - norm_hat).abs() / norm_orig < 0.3,
            "norm diverged: {norm_orig} → {norm_hat}"
        );
    }

    #[test]
    fn zero_vector() {
        let dim = 32;
        let quant = TurboQuantMse::new(dim, 2, 42).unwrap();

        let x = vec![0.0f32; dim];
        let q = quant.quantize(&x).unwrap();
        assert_eq!(q.norm, 0.0);

        let x_hat = quant.dequantize(&q).unwrap();
        for v in &x_hat {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn dimension_mismatch() {
        let quant = TurboQuantMse::new(32, 2, 42).unwrap();
        let x = vec![1.0; 64];
        assert!(quant.quantize(&x).is_err());
    }

    #[test]
    fn average_mse_matches_theory() {
        // Test over many random vectors to verify empirical MSE
        // approaches theoretical bound D_mse ≈ 0.117 for b=2
        let dim = 256;
        let bits = 2;
        let quant = TurboQuantMse::new(dim, bits, 42).unwrap();
        let n_trials = 100;

        let total_mse: f32 = (0..n_trials)
            .map(|seed| {
                let x = random_unit_vector(dim, seed + 1000);
                let q = quant.quantize(&x).unwrap();
                let x_hat = quant.dequantize(&q).unwrap();
                x.iter()
                    .zip(x_hat.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>()
            })
            .sum();

        let avg_mse = total_mse / n_trials as f32;
        // Paper: D_mse(b=2) ≈ 0.117 for unit vectors
        // Allow generous margin for finite-d effects
        assert!(
            avg_mse < 0.35,
            "average MSE = {avg_mse}, expected < 0.35 for 2-bit"
        );
    }
}
