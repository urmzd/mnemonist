//! Random orthogonal rotation matrix for TurboQuant.
//!
//! After applying a random rotation Π to a unit vector x ∈ S^{d-1},
//! each coordinate of Π·x follows a Beta(d/2, d/2) distribution (Lemma 1),
//! and distinct coordinates become nearly independent in high dimensions.
//! This enables per-coordinate scalar quantization.
//!
//! We generate Π via QR decomposition of a random Gaussian matrix.
//! The rotation is seeded for reproducibility — the same seed must be used
//! for quantization and dequantization.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

/// A random orthogonal rotation matrix.
///
/// Stores the full d×d matrix in row-major order.
/// For typical embedding dimensions (384, 768, 1536), this is 0.6–9 MB.
#[derive(Debug, Clone)]
pub struct Rotation {
    dim: usize,
    /// Row-major d×d orthogonal matrix.
    matrix: Vec<f32>,
    seed: u64,
}

impl Rotation {
    /// Generate a random orthogonal rotation matrix of the given dimension.
    ///
    /// Uses QR decomposition of a random Gaussian matrix with the given seed.
    pub fn new(dim: usize, seed: u64) -> Self {
        let matrix = generate_orthogonal(dim, seed);
        Self { dim, matrix, seed }
    }

    /// The dimension of the rotation.
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// The seed used to generate this rotation.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Apply the forward rotation: y = Π · x (in-place).
    pub fn forward(&self, x: &mut [f32]) {
        debug_assert_eq!(x.len(), self.dim);
        let mut result = vec![0.0f32; self.dim];
        for (i, item) in result.iter_mut().enumerate().take(self.dim) {
            let row = &self.matrix[i * self.dim..(i + 1) * self.dim];
            *item = dot(row, x);
        }
        x.copy_from_slice(&result);
    }

    /// Apply the inverse rotation: x = Π^T · y (in-place).
    ///
    /// Since Π is orthogonal, Π^{-1} = Π^T.
    pub fn inverse(&self, y: &mut [f32]) {
        debug_assert_eq!(y.len(), self.dim);
        let mut result = vec![0.0f32; self.dim];
        // Π^T multiplication: result[j] = sum_i matrix[i][j] * y[i]
        for (i, &yi) in y.iter().enumerate().take(self.dim) {
            let row = &self.matrix[i * self.dim..(i + 1) * self.dim];
            for j in 0..self.dim {
                result[j] += row[j] * yi;
            }
        }
        y.copy_from_slice(&result);
    }
}

/// Dot product of two slices.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Generate a random orthogonal matrix via QR decomposition of a Gaussian matrix.
///
/// Uses Gram-Schmidt orthogonalization (numerically sufficient for our use case
/// since we only need the rotation to randomize coordinate distributions,
/// not for high-precision numerical linear algebra).
fn generate_orthogonal(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = StandardNormal;

    // Generate random Gaussian matrix
    let mut q = vec![0.0f32; dim * dim];
    for val in &mut q {
        *val = normal.sample(&mut rng);
    }

    // Modified Gram-Schmidt orthogonalization (row-wise)
    for i in 0..dim {
        let row_start = i * dim;

        // Subtract projections of all previous rows
        for j in 0..i {
            let prev_start = j * dim;
            let mut proj = 0.0f32;
            for k in 0..dim {
                proj += q[row_start + k] * q[prev_start + k];
            }
            for k in 0..dim {
                q[row_start + k] -= proj * q[prev_start + k];
            }
        }

        // Normalize
        let mut norm = 0.0f32;
        for k in 0..dim {
            norm += q[row_start + k] * q[row_start + k];
        }
        let norm = norm.sqrt();
        if norm > 0.0 {
            for k in 0..dim {
                q[row_start + k] /= norm;
            }
        }
    }

    q
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orthogonality() {
        let dim = 32;
        let rot = Rotation::new(dim, 42);

        // Check Q^T · Q ≈ I
        for i in 0..dim {
            for j in 0..dim {
                let row_i = &rot.matrix[i * dim..(i + 1) * dim];
                let row_j = &rot.matrix[j * dim..(j + 1) * dim];
                let d = dot(row_i, row_j);
                if i == j {
                    assert!(
                        (d - 1.0).abs() < 1e-4,
                        "diagonal [{i},{j}] = {d}, expected 1.0"
                    );
                } else {
                    assert!(d.abs() < 1e-4, "off-diagonal [{i},{j}] = {d}, expected 0");
                }
            }
        }
    }

    #[test]
    fn deterministic_with_seed() {
        let r1 = Rotation::new(16, 123);
        let r2 = Rotation::new(16, 123);
        assert_eq!(r1.matrix, r2.matrix);
    }

    #[test]
    fn different_seeds_differ() {
        let r1 = Rotation::new(16, 1);
        let r2 = Rotation::new(16, 2);
        assert_ne!(r1.matrix, r2.matrix);
    }

    #[test]
    fn forward_inverse_roundtrip() {
        let dim = 64;
        let rot = Rotation::new(dim, 99);

        let original: Vec<f32> = (0..dim).map(|i| (i as f32 + 0.5) / dim as f32).collect();
        let mut v = original.clone();

        rot.forward(&mut v);
        // Rotated vector should differ from original
        assert!(
            v.iter()
                .zip(original.iter())
                .any(|(a, b)| (a - b).abs() > 1e-4),
            "rotation had no effect"
        );

        rot.inverse(&mut v);
        // Should be back to original
        for (a, b) in v.iter().zip(original.iter()) {
            assert!(
                (a - b).abs() < 1e-3,
                "roundtrip failed: {a} vs {b} (diff={})",
                (a - b).abs()
            );
        }
    }

    #[test]
    fn preserves_norm() {
        let dim = 64;
        let rot = Rotation::new(dim, 7);

        let mut v: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        let norm_before: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();

        rot.forward(&mut v);
        let norm_after: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!(
            (norm_before - norm_after).abs() < 1e-3,
            "norm changed: {norm_before} → {norm_after}"
        );
    }
}
