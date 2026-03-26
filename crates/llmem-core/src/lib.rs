pub mod backend;
pub mod embed;
mod error;
pub mod index;
pub mod memory;

pub use backend::{FileBackend, MemoryBackend, SemanticHit};
pub use embed::{Embedder, EmbeddingEntry, EmbeddingStore};
pub use error::Error;
pub use index::{IndexEntry, MemoryIndex};
pub use memory::{Frontmatter, MemoryFile, MemoryLevel, MemoryType};

/// Default project-level memory directory name.
pub const PROJECT_DIR: &str = ".llmem";

/// Default index file name within a memory directory.
pub const INDEX_FILE: &str = "MEMORY.md";

/// Returns the global memory directory path (`~/.config/llmem/`).
pub fn global_dir() -> Option<std::path::PathBuf> {
    dirs::config_dir().map(|d| d.join("llmem"))
}

/// Returns the project memory directory path (`.llmem/` in given root).
pub fn project_dir(root: &std::path::Path) -> std::path::PathBuf {
    root.join(PROJECT_DIR)
}
