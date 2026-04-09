//! Evaluation metrics and harness for mnemonist.
//!
//! - [`search`] — retrieval quality (MRR, NDCG, precision@k, recall@k)
//! - [`embedding`] — embedding space quality (anisotropy, discrimination gap, intrinsic dimensionality)
//! - [`quantization`] — quantization fidelity (MSE, cosine distortion, recall impact)
//! - [`dataset`] — synthetic benchmark dataset generation
//! - [`harness`] — end-to-end eval runner producing structured reports

pub mod bench;
pub mod dataset;
pub mod embedding;
pub mod harness;
pub mod longmemeval;
pub mod report;
pub mod search;

#[cfg(feature = "quant")]
pub mod quantization;

pub use report::EvalReport;

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("insufficient data: need at least {min}, got {got}")]
    InsufficientData { min: usize, got: usize },

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("quantization error: {0}")]
    #[cfg(feature = "quant")]
    Quant(#[from] mnemonist_quant::QuantError),

    #[error("{0}")]
    Other(String),
}
