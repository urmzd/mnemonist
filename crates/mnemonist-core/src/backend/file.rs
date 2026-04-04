use std::path::{Path, PathBuf};

use crate::Error;
use crate::backend::{MemoryBackend, SemanticHit};
use crate::embed::{Embedder, EmbeddingStore};
use crate::index::{IndexEntry, MemoryIndex};
use crate::memory::MemoryFile;

/// File-based memory backend.
///
/// Stores memories as markdown files in a directory with a MEMORY.md index
/// and optional .embeddings.bin for semantic search.
pub struct FileBackend {
    dir: PathBuf,
    index: MemoryIndex,
    embeddings: EmbeddingStore,
}

impl FileBackend {
    /// Open an existing memory directory.
    pub fn open(dir: &Path) -> Result<Self, Error> {
        let index = MemoryIndex::load(dir)?;
        let embeddings_path = dir.join(".embeddings.bin");
        let embeddings = if embeddings_path.exists() {
            EmbeddingStore::load(&embeddings_path)?
        } else {
            EmbeddingStore::new(0)
        };

        Ok(Self {
            dir: dir.to_path_buf(),
            index,
            embeddings,
        })
    }

    /// Initialize a new memory directory.
    pub fn init(dir: &Path) -> Result<Self, Error> {
        let index = MemoryIndex::init(dir)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            index,
            embeddings: EmbeddingStore::new(0),
        })
    }

    /// Get the directory path.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Save the index and embeddings to disk.
    pub fn save(&self) -> Result<(), Error> {
        self.index.save()?;
        if !self.embeddings.entries.is_empty() {
            self.embeddings.save(&self.dir.join(".embeddings.bin"))?;
        }
        Ok(())
    }
}

impl MemoryBackend for FileBackend {
    fn store(&mut self, memory: &MemoryFile) -> Result<(), Error> {
        let filename = memory.filename();
        let entry = IndexEntry {
            title: memory.frontmatter.name.replace('-', " "),
            file: filename.clone(),
            summary: memory.frontmatter.description.clone(),
        };

        // Remove existing if present (for updates)
        self.index.remove(&filename);
        self.index.add(entry)?;
        memory.write(&self.dir.join(&filename))?;
        self.save()?;
        Ok(())
    }

    fn get(&self, name: &str) -> Result<Option<MemoryFile>, Error> {
        // Try to find by name in index
        let entry = self.index.entries.iter().find(|e| e.file.contains(name));

        match entry {
            Some(e) => {
                let path = self.dir.join(&e.file);
                if path.exists() {
                    Ok(Some(MemoryFile::read(&path)?))
                } else {
                    Ok(None)
                }
            }
            None => Ok(None),
        }
    }

    fn remove(&mut self, name: &str) -> Result<bool, Error> {
        // Find the file matching this name
        let file = self
            .index
            .entries
            .iter()
            .find(|e| e.file.contains(name))
            .map(|e| e.file.clone());

        if let Some(file) = file {
            self.index.remove(&file);
            let path = self.dir.join(&file);
            if path.exists() {
                std::fs::remove_file(&path)?;
            }
            self.embeddings.remove(&file);
            self.save()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn list(&self) -> Result<Vec<IndexEntry>, Error> {
        Ok(self.index.entries.clone())
    }

    fn search_text(&self, query: &str) -> Result<Vec<IndexEntry>, Error> {
        Ok(self.index.search(query).into_iter().cloned().collect())
    }

    fn search_semantic(
        &self,
        _embedding: &[f32],
        _top_k: usize,
    ) -> Result<Vec<SemanticHit>, Error> {
        // Semantic search requires an ANN index, which lives in mnemonist-index.
        // FileBackend alone does text search only.
        // Full semantic search is composed at the server/CLI level
        // by combining FileBackend + AnnIndex.
        Ok(Vec::new())
    }

    fn sync_embeddings(&mut self, embedder: &dyn Embedder) -> Result<usize, Error> {
        let count = self.embeddings.sync(&self.dir, embedder)?;
        if count > 0 {
            self.embeddings.save(&self.dir.join(".embeddings.bin"))?;
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{Frontmatter, MemoryType};

    #[test]
    fn store_and_get() {
        let tmp = tempfile::tempdir().unwrap();
        let mut backend = FileBackend::init(tmp.path()).unwrap();

        let mem = MemoryFile {
            frontmatter: Frontmatter {
                name: "test-memory".into(),
                description: "A test".into(),
                memory_type: MemoryType::Feedback,
                ..Default::default()
            },
            body: "Test body.".into(),
        };

        backend.store(&mem).unwrap();

        let retrieved = backend.get("test-memory").unwrap().unwrap();
        assert_eq!(retrieved.frontmatter.name, "test-memory");
        assert_eq!(retrieved.body, "Test body.");
    }

    #[test]
    fn list_and_search() {
        let tmp = tempfile::tempdir().unwrap();
        let mut backend = FileBackend::init(tmp.path()).unwrap();

        for i in 0..3 {
            let mem = MemoryFile {
                frontmatter: Frontmatter {
                    name: format!("mem-{i}"),
                    description: format!("memory number {i}"),
                    memory_type: MemoryType::User,
                    ..Default::default()
                },
                body: String::new(),
            };
            backend.store(&mem).unwrap();
        }

        assert_eq!(backend.list().unwrap().len(), 3);

        let results = backend.search_text("number 1").unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn remove_works() {
        let tmp = tempfile::tempdir().unwrap();
        let mut backend = FileBackend::init(tmp.path()).unwrap();

        let mem = MemoryFile {
            frontmatter: Frontmatter {
                name: "to-remove".into(),
                description: "will be removed".into(),
                memory_type: MemoryType::Feedback,
                ..Default::default()
            },
            body: String::new(),
        };

        backend.store(&mem).unwrap();
        assert!(backend.remove("to-remove").unwrap());
        assert!(!backend.remove("to-remove").unwrap());
        assert_eq!(backend.list().unwrap().len(), 0);
    }
}
