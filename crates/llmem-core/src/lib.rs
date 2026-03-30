pub mod backend;
pub mod config;
pub mod embed;
mod error;
pub mod inbox;
pub mod index;
pub mod memory;

pub use backend::{FileBackend, MemoryBackend, SemanticHit};
pub use config::Config;
pub use embed::{Embedder, EmbeddingEntry, EmbeddingStore, FastEmbedder};
pub use error::Error;
pub use inbox::{FileSource, Inbox, InboxItem};
pub use index::{IndexEntry, MemoryIndex};
pub use memory::{Frontmatter, MemoryFile, MemoryLevel, MemoryType};

/// Default index file name within a memory directory.
pub const INDEX_FILE: &str = "MEMORY.md";

/// Returns the llmem root directory from config.
pub fn llmem_root() -> Option<std::path::PathBuf> {
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
