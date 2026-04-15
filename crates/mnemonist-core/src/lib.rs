pub mod backend;
pub mod chunk;
pub mod config;
pub mod distance;
pub mod embed;
mod error;
pub mod inbox;
pub mod index;
pub mod memory;
pub mod rerank;
pub mod temporal;

#[cfg(feature = "ann")]
pub mod ann;
#[cfg(feature = "quant")]
pub mod quant;
#[cfg(feature = "evals")]
pub mod evals;

pub use backend::{FileBackend, MemoryBackend, SemanticHit};
pub use chunk::{Chunk, ChunkingStrategy};
pub use config::Config;
pub use embed::{CandleEmbedder, Embedder, EmbeddingEntry, EmbeddingStore};
pub use error::Error;
pub use inbox::{FileSource, Inbox, InboxItem};
pub use index::{IndexEntry, MemoryIndex};
pub use memory::{Frontmatter, MemoryFile, MemoryLevel, MemoryType};
pub use rerank::{Candidate, MemorySignals, RankedResult, RecallProfile, rerank};

/// Default index file name within a memory directory.
pub const INDEX_FILE: &str = "MEMORY.md";

/// Returns the mnemonist root directory from config.
pub fn mnemonist_root() -> Option<std::path::PathBuf> {
    Some(Config::load().root())
}

/// Returns the global memory directory from config.
pub fn global_dir() -> Option<std::path::PathBuf> {
    Some(Config::load().global_dir())
}

/// Returns the project memory directory from config.
pub fn project_dir(root: &std::path::Path) -> std::path::PathBuf {
    Config::load().project_dir(root)
}
