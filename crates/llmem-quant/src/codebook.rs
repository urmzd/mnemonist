//! Precomputed Lloyd-Max optimal scalar quantizer codebooks.
//!
//! For high-dimensional vectors (d ≥ 64), after random rotation each coordinate
//! follows a Beta(d/2, d/2) distribution that closely approximates N(0, 1/d).
//! The codebooks partition [-1, 1] into 2^b buckets minimizing MSE.
//!
//! These are computed offline by solving the continuous 1-D k-means problem
//! (Eq. 4 in the TurboQuant paper) using the Max-Lloyd algorithm.

use crate::QuantError;

/// A scalar quantizer codebook for a given bit-width.
#[derive(Debug, Clone)]
pub struct Codebook {
    /// Bit-width (1-4).
    pub bits: u8,
    /// Reconstruction centroids in ascending order (2^b values).
    pub centroids: &'static [f32],
    /// Decision boundaries (2^b - 1 midpoints between consecutive centroids).
    pub boundaries: &'static [f32],
}

impl Codebook {
    /// Get the codebook for a given bit-width.
    pub fn for_bits(bits: u8) -> Result<&'static Codebook, QuantError> {
        match bits {
            1 => Ok(&CODEBOOK_1BIT),
            2 => Ok(&CODEBOOK_2BIT),
            3 => Ok(&CODEBOOK_3BIT),
            4 => Ok(&CODEBOOK_4BIT),
            _ => Err(QuantError::UnsupportedBitWidth(bits)),
        }
    }

    /// Find the index of the nearest centroid for a scalar value.
    #[inline]
    pub fn quantize_scalar(&self, x: f32) -> u8 {
        // Binary search on boundaries (they are sorted ascending).
        let mut idx = 0u8;
        for &b in self.boundaries {
            if x > b {
                idx += 1;
            } else {
                break;
            }
        }
        idx
    }

    /// Look up the centroid for a given index.
    #[inline]
    pub fn dequantize_scalar(&self, idx: u8) -> f32 {
        self.centroids[idx as usize]
    }
}

// ─── Precomputed codebooks ──────────────────────────────────────────────────
//
// These are optimal Lloyd-Max centroids for the Beta distribution that arises
// from randomly rotating unit-sphere vectors. In high dimensions, Beta(d/2, d/2)
// on [-1, 1] converges to N(0, 1/d). For moderate d, the paper uses numerical
// optimization. The values below are for the Gaussian approximation, scaled by
// 1/sqrt(d) at runtime. Since the rotation normalizes vectors to the unit sphere,
// coordinates land in roughly [-3/sqrt(d), 3/sqrt(d)], and the codebook operates
// on the pre-scaled domain [-1, 1].
//
// For b=1: optimal centroids are ±√(2/π) ≈ ±0.7979 (Gaussian quantizer)
// For b=2: ±0.4528, ±1.5104 (standard 2-bit Gaussian Lloyd-Max)
// For b=3,4: standard Lloyd-Max for N(0,1) scaled to [-1,1]
//
// We use the paper's high-d Gaussian approximation centroids, which give
// near-optimal MSE for d ≥ 64.

static CENTROIDS_1BIT: [f32; 2] = [-0.7979, 0.7979];
static BOUNDARIES_1BIT: [f32; 1] = [0.0];

static CENTROIDS_2BIT: [f32; 4] = [-1.5104, -0.4528, 0.4528, 1.5104];
static BOUNDARIES_2BIT: [f32; 3] = [-0.9816, 0.0, 0.9816];

static CENTROIDS_3BIT: [f32; 8] = [
    -2.1520, -1.3440, -0.7560, -0.2450, 0.2450, 0.7560, 1.3440, 2.1520,
];
static BOUNDARIES_3BIT: [f32; 7] = [-1.7480, -1.0500, -0.5005, 0.0, 0.5005, 1.0500, 1.7480];

static CENTROIDS_4BIT: [f32; 16] = [
    -2.7326, -2.0690, -1.6180, -1.2562, -0.9424, -0.6568, -0.3880, -0.1284, 0.1284, 0.3880, 0.6568,
    0.9424, 1.2562, 1.6180, 2.0690, 2.7326,
];
static BOUNDARIES_4BIT: [f32; 15] = [
    -2.4008, -1.8435, -1.4371, -1.0993, -0.7996, -0.5224, -0.2582, 0.0, 0.2582, 0.5224, 0.7996,
    1.0993, 1.4371, 1.8435, 2.4008,
];

static CODEBOOK_1BIT: Codebook = Codebook {
    bits: 1,
    centroids: &CENTROIDS_1BIT,
    boundaries: &BOUNDARIES_1BIT,
};

static CODEBOOK_2BIT: Codebook = Codebook {
    bits: 2,
    centroids: &CENTROIDS_2BIT,
    boundaries: &BOUNDARIES_2BIT,
};

static CODEBOOK_3BIT: Codebook = Codebook {
    bits: 3,
    centroids: &CENTROIDS_3BIT,
    boundaries: &BOUNDARIES_3BIT,
};

static CODEBOOK_4BIT: Codebook = Codebook {
    bits: 4,
    centroids: &CENTROIDS_4BIT,
    boundaries: &BOUNDARIES_4BIT,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebook_lookup_valid() {
        for bits in 1..=4 {
            let cb = Codebook::for_bits(bits).unwrap();
            assert_eq!(cb.centroids.len(), 1 << bits);
            assert_eq!(cb.boundaries.len(), (1 << bits) - 1);
        }
    }

    #[test]
    fn codebook_invalid_bits() {
        assert!(Codebook::for_bits(0).is_err());
        assert!(Codebook::for_bits(5).is_err());
    }

    #[test]
    fn quantize_scalar_1bit() {
        let cb = Codebook::for_bits(1).unwrap();
        assert_eq!(cb.quantize_scalar(-0.5), 0);
        assert_eq!(cb.quantize_scalar(0.5), 1);
        assert_eq!(cb.quantize_scalar(0.0), 0); // boundary → lower bucket
    }

    #[test]
    fn quantize_scalar_2bit() {
        let cb = Codebook::for_bits(2).unwrap();
        assert_eq!(cb.quantize_scalar(-2.0), 0);
        assert_eq!(cb.quantize_scalar(-0.5), 1);
        assert_eq!(cb.quantize_scalar(0.5), 2);
        assert_eq!(cb.quantize_scalar(2.0), 3);
    }

    #[test]
    fn centroids_are_sorted() {
        for bits in 1..=4 {
            let cb = Codebook::for_bits(bits).unwrap();
            for w in cb.centroids.windows(2) {
                assert!(w[0] < w[1], "centroids not sorted for {bits}-bit codebook");
            }
            for w in cb.boundaries.windows(2) {
                assert!(w[0] < w[1], "boundaries not sorted for {bits}-bit codebook");
            }
        }
    }

    #[test]
    fn boundaries_between_centroids() {
        for bits in 1..=4 {
            let cb = Codebook::for_bits(bits).unwrap();
            for (i, &b) in cb.boundaries.iter().enumerate() {
                assert!(
                    b > cb.centroids[i] && b < cb.centroids[i + 1],
                    "boundary {b} not between centroids {} and {} for {bits}-bit",
                    cb.centroids[i],
                    cb.centroids[i + 1]
                );
            }
        }
    }

    #[test]
    fn dequantize_roundtrip() {
        let cb = Codebook::for_bits(2).unwrap();
        for i in 0..4u8 {
            let val = cb.dequantize_scalar(i);
            let idx = cb.quantize_scalar(val);
            assert_eq!(idx, i, "dequantize({i}) = {val}, quantize({val}) = {idx}");
        }
    }
}
