pub mod code;
pub mod distance;
pub mod eval;
pub mod hnsw;
pub mod ivf;

use mnemonist_core::Error;
use std::path::Path;

/// A single search result from an ANN index.
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub id: String,
    pub score: f32,
}

/// Trait for approximate nearest neighbor indices.
///
/// Implementations must support insert, remove, search, and serialization.
pub trait AnnIndex: Send + Sync {
    /// Insert a vector with an associated ID.
    fn insert(&mut self, id: &str, embedding: &[f32]) -> Result<(), Error>;

    /// Remove a vector by ID. Returns true if it existed.
    fn remove(&mut self, id: &str) -> Result<bool, Error>;

    /// Search for the top-k nearest neighbors to the query vector.
    /// Results are sorted by descending score (highest similarity first).
    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<SearchHit>, Error>;

    /// Number of vectors in the index.
    fn len(&self) -> usize;

    /// Whether the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Serialize the index to a file.
    fn save(&self, path: &Path) -> Result<(), Error>;
}
