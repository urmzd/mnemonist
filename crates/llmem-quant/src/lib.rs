//! TurboQuant: online vector quantization with near-optimal distortion rate.
//!
//! Implements the algorithms from [TurboQuant (arXiv:2504.19874)](https://arxiv.org/abs/2504.19874):
//!
//! - **`TurboQuantMse`** — MSE-optimal quantizer using random rotation + Lloyd-Max codebooks
//! - **`TurboQuantProd`** — unbiased inner-product quantizer (MSE + QJL residual)
//! - **`CompressedEmbeddingStore`** — binary storage format for quantized embeddings

pub mod codebook;
pub mod mse;
pub mod pack;
pub mod prod;
pub mod qjl;
pub mod rotation;
pub mod store;

pub use codebook::Codebook;
pub use mse::{QuantizedVector, TurboQuantMse};
pub use prod::{QuantizedProdVector, TurboQuantProd};
pub use qjl::QjlTransform;
pub use rotation::Rotation;
pub use store::CompressedEmbeddingStore;

/// Errors specific to quantization operations.
#[derive(Debug, thiserror::Error)]
pub enum QuantError {
    #[error("unsupported bit-width: {0} (must be 1-4)")]
    UnsupportedBitWidth(u8),

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("format error: {0}")]
    Format(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
