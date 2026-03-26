pub mod file;

pub use file::FileBackend;

use crate::Error;
use crate::embed::Embedder;
use crate::index::IndexEntry;
use crate::memory::MemoryFile;

/// A search hit from semantic search.
#[derive(Debug, Clone)]
pub struct SemanticHit {
    pub name: String,
    pub score: f32,
}

/// Trait for memory storage backends.
///
/// File-based memory is the default implementation. Future backends
/// include knowledge graphs (e.g., via saige) and database-backed stores.
pub trait MemoryBackend: Send + Sync {
    /// Store a memory. Creates or updates.
    fn store(&mut self, memory: &MemoryFile) -> Result<(), Error>;

    /// Retrieve a memory by name.
    fn get(&self, name: &str) -> Result<Option<MemoryFile>, Error>;

    /// Remove a memory by name. Returns true if it existed.
    fn remove(&mut self, name: &str) -> Result<bool, Error>;

    /// List all memory entries (index only, no bodies).
    fn list(&self) -> Result<Vec<IndexEntry>, Error>;

    /// Search memories by text query (matches against index descriptions).
    fn search_text(&self, query: &str) -> Result<Vec<IndexEntry>, Error>;

    /// Search memories by embedding vector (semantic similarity).
    fn search_semantic(&self, embedding: &[f32], top_k: usize) -> Result<Vec<SemanticHit>, Error>;

    /// Sync embeddings for all memories. Returns count of re-embedded files.
    fn sync_embeddings(&mut self, embedder: &dyn Embedder) -> Result<usize, Error>;
}
