#[cfg(feature = "candle")]
pub mod candle;

use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{Read as _, Write as _};
use std::path::Path;

#[cfg(feature = "candle")]
pub use candle::CandleEmbedder;

use crate::Error;

/// Pluggable embedding provider.
///
/// Implementations can use ONNX (local), Ollama, OpenAI, or a custom model.
pub trait Embedder: Send + Sync {
    /// Embed a single text into a vector.
    fn embed(&self, text: &str) -> Result<Vec<f32>, Error>;

    /// Embed a batch of texts. Default falls back to sequential single embeds.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, Error> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// The dimensionality of the output vectors.
    fn dimension(&self) -> Result<usize, Error>;
}

/// A single entry mapping a file to its embedding.
#[derive(Debug, Clone)]
pub struct EmbeddingEntry {
    pub file: String,
    pub hash: u64,
    pub embedding: Vec<f32>,
}

// Binary format:
// Header: [magic: 4 bytes "LMEM"] [version: u8] [dimension: u32] [count: u32]
// Entries: [file_len: u16] [file: bytes] [hash: u64] [embedding: f32 * dimension]

const MAGIC: &[u8; 4] = b"LMEM";
const FORMAT_VERSION: u8 = 1;

/// Stores embeddings alongside memory files.
#[derive(Debug, Clone)]
pub struct EmbeddingStore {
    pub dimension: usize,
    pub entries: Vec<EmbeddingEntry>,
}

impl EmbeddingStore {
    /// Create a new empty store.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            entries: Vec::new(),
        }
    }

    /// Load from a binary file.
    pub fn load(path: &Path) -> Result<Self, Error> {
        let data = fs::read(path)?;
        let mut cursor = &data[..];

        // Magic
        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(Error::EmbeddingFormat("invalid magic bytes".into()));
        }

        // Version
        let mut ver = [0u8; 1];
        cursor.read_exact(&mut ver)?;
        if ver[0] != FORMAT_VERSION {
            return Err(Error::EmbeddingFormat(format!(
                "unsupported version: {}",
                ver[0]
            )));
        }

        // Dimension
        let mut dim_bytes = [0u8; 4];
        cursor.read_exact(&mut dim_bytes)?;
        let dimension = u32::from_le_bytes(dim_bytes) as usize;

        // Count
        let mut count_bytes = [0u8; 4];
        cursor.read_exact(&mut count_bytes)?;
        let count = u32::from_le_bytes(count_bytes) as usize;

        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            // File name
            let mut file_len_bytes = [0u8; 2];
            cursor.read_exact(&mut file_len_bytes)?;
            let file_len = u16::from_le_bytes(file_len_bytes) as usize;
            let mut file_bytes = vec![0u8; file_len];
            cursor.read_exact(&mut file_bytes)?;
            let file = String::from_utf8(file_bytes)
                .map_err(|e| Error::EmbeddingFormat(format!("invalid UTF-8: {e}")))?;

            // Hash
            let mut hash_bytes = [0u8; 8];
            cursor.read_exact(&mut hash_bytes)?;
            let hash = u64::from_le_bytes(hash_bytes);

            // Embedding
            let mut embedding = vec![0f32; dimension];
            for val in &mut embedding {
                let mut f_bytes = [0u8; 4];
                cursor.read_exact(&mut f_bytes)?;
                *val = f32::from_le_bytes(f_bytes);
            }

            entries.push(EmbeddingEntry {
                file,
                hash,
                embedding,
            });
        }

        Ok(Self { dimension, entries })
    }

    /// Save to a binary file.
    pub fn save(&self, path: &Path) -> Result<(), Error> {
        let mut buf = Vec::new();

        // Header
        buf.write_all(MAGIC)?;
        buf.write_all(&[FORMAT_VERSION])?;
        buf.write_all(&(self.dimension as u32).to_le_bytes())?;
        buf.write_all(&(self.entries.len() as u32).to_le_bytes())?;

        // Entries
        for entry in &self.entries {
            let file_bytes = entry.file.as_bytes();
            buf.write_all(&(file_bytes.len() as u16).to_le_bytes())?;
            buf.write_all(file_bytes)?;
            buf.write_all(&entry.hash.to_le_bytes())?;
            for &val in &entry.embedding {
                buf.write_all(&val.to_le_bytes())?;
            }
        }

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, buf)?;
        Ok(())
    }

    /// Find an entry by filename.
    pub fn get(&self, file: &str) -> Option<&EmbeddingEntry> {
        self.entries.iter().find(|e| e.file == file)
    }

    /// Insert or update an entry.
    pub fn upsert(&mut self, entry: EmbeddingEntry) {
        if let Some(existing) = self.entries.iter_mut().find(|e| e.file == entry.file) {
            *existing = entry;
        } else {
            self.entries.push(entry);
        }
    }

    /// Remove an entry by filename.
    pub fn remove(&mut self, file: &str) -> bool {
        let len = self.entries.len();
        self.entries.retain(|e| e.file != file);
        self.entries.len() < len
    }

    /// Sync embeddings with memory files in a directory.
    /// Returns the number of files re-embedded.
    pub fn sync(&mut self, dir: &Path, embedder: &dyn Embedder) -> Result<usize, Error> {
        self.sync_with_progress(dir, embedder, None)
    }

    /// Sync embeddings with optional progress reporting.
    pub fn sync_with_progress(
        &mut self,
        dir: &Path,
        embedder: &dyn Embedder,
        reporter: Option<&dyn crate::progress::Progress>,
    ) -> Result<usize, Error> {
        let mut re_embedded = 0;

        // Find all memory files
        let mut current_files = Vec::new();
        if dir.exists() {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let name = entry.file_name().to_string_lossy().to_string();
                if name.ends_with(".md") && name != "MEMORY.md" {
                    current_files.push(name);
                }
            }
        }

        // Remove entries for deleted files
        self.entries.retain(|e| current_files.contains(&e.file));

        if let Some(r) = reporter {
            r.start("re-embedding", Some(current_files.len()));
        }

        // Embed new or changed files
        for (idx, file) in current_files.iter().enumerate() {
            let path = dir.join(file);
            let content = fs::read_to_string(&path)?;
            let hash = content_hash(&content);

            if let Some(existing) = self.entries.iter().find(|e| e.file == *file)
                && existing.hash == hash
            {
                if let Some(r) = reporter {
                    r.tick(idx + 1);
                }
                continue; // unchanged
            }

            let embedding = embedder.embed(&content)?;
            self.upsert(EmbeddingEntry {
                file: file.clone(),
                hash,
                embedding,
            });
            re_embedded += 1;
            if let Some(r) = reporter {
                r.tick(idx + 1);
            }
        }

        if let Some(r) = reporter {
            r.finish(Some(&format!("{re_embedded} re-embedded")));
        }

        Ok(re_embedded)
    }
}

/// Compute a content hash for change detection.
pub fn content_hash(content: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("embeddings.bin");

        let mut store = EmbeddingStore::new(3);
        store.upsert(EmbeddingEntry {
            file: "feedback_test.md".into(),
            hash: 12345,
            embedding: vec![0.1, 0.2, 0.3],
        });
        store.upsert(EmbeddingEntry {
            file: "user_prefs.md".into(),
            hash: 67890,
            embedding: vec![0.4, 0.5, 0.6],
        });

        store.save(&path).unwrap();
        let loaded = EmbeddingStore::load(&path).unwrap();

        assert_eq!(loaded.dimension, 3);
        assert_eq!(loaded.entries.len(), 2);
        assert_eq!(loaded.entries[0].file, "feedback_test.md");
        assert!((loaded.entries[0].embedding[0] - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn upsert_replaces() {
        let mut store = EmbeddingStore::new(2);
        store.upsert(EmbeddingEntry {
            file: "test.md".into(),
            hash: 1,
            embedding: vec![0.1, 0.2],
        });
        store.upsert(EmbeddingEntry {
            file: "test.md".into(),
            hash: 2,
            embedding: vec![0.3, 0.4],
        });

        assert_eq!(store.entries.len(), 1);
        assert_eq!(store.entries[0].hash, 2);
    }

    #[test]
    fn content_hash_deterministic() {
        let h1 = content_hash("hello world");
        let h2 = content_hash("hello world");
        let h3 = content_hash("different");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }
}
