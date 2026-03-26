use std::fs;
use std::path::{Path, PathBuf};

use crate::{Error, INDEX_FILE};

/// A single entry in the memory index.
#[derive(Debug, Clone)]
pub struct IndexEntry {
    pub title: String,
    pub file: String,
    pub summary: String,
}

impl IndexEntry {
    /// Format as a markdown list item.
    pub fn to_line(&self) -> String {
        format!("- [{}]({}) — {}", self.title, self.file, self.summary)
    }

    /// Parse from a markdown list item.
    pub fn parse(line: &str) -> Option<Self> {
        let line = line.trim();
        if !line.starts_with("- [") {
            return None;
        }

        let title_end = line.find("](").unwrap_or(0);
        let title = line[3..title_end].to_string();

        let file_start = title_end + 2;
        let file_end = line[file_start..].find(')').map(|i| i + file_start)?;
        let file = line[file_start..file_end].to_string();

        let summary = line[file_end + 1..]
            .trim()
            .strip_prefix("— ")
            .or_else(|| line[file_end + 1..].trim().strip_prefix("- "))
            .unwrap_or("")
            .to_string();

        Some(Self {
            title,
            file,
            summary,
        })
    }
}

const MAX_INDEX_LINES: usize = 200;

/// The MEMORY.md index.
#[derive(Debug, Clone)]
pub struct MemoryIndex {
    pub entries: Vec<IndexEntry>,
    pub dir: PathBuf,
}

impl MemoryIndex {
    /// Load the index from a memory directory.
    pub fn load(dir: &Path) -> Result<Self, Error> {
        let path = dir.join(INDEX_FILE);
        if !path.exists() {
            return Ok(Self {
                entries: vec![],
                dir: dir.to_path_buf(),
            });
        }

        let content = fs::read_to_string(&path)?;
        let entries: Vec<IndexEntry> = content.lines().filter_map(IndexEntry::parse).collect();

        Ok(Self {
            entries,
            dir: dir.to_path_buf(),
        })
    }

    /// Create a new empty index, writing MEMORY.md to disk.
    pub fn init(dir: &Path) -> Result<Self, Error> {
        fs::create_dir_all(dir)?;
        let path = dir.join(INDEX_FILE);
        if !path.exists() {
            fs::write(&path, "")?;
        }
        Ok(Self {
            entries: vec![],
            dir: dir.to_path_buf(),
        })
    }

    /// Add an entry to the index.
    pub fn add(&mut self, entry: IndexEntry) -> Result<(), Error> {
        if self.entries.iter().any(|e| e.file == entry.file) {
            return Err(Error::Duplicate {
                name: entry.file.clone(),
            });
        }
        self.entries.push(entry);
        if self.entries.len() > MAX_INDEX_LINES {
            return Err(Error::IndexTooLarge {
                lines: self.entries.len(),
                max: MAX_INDEX_LINES,
            });
        }
        Ok(())
    }

    /// Remove an entry by filename.
    pub fn remove(&mut self, file: &str) -> bool {
        let len = self.entries.len();
        self.entries.retain(|e| e.file != file);
        self.entries.len() < len
    }

    /// Search entries by matching query against titles and summaries.
    pub fn search(&self, query: &str) -> Vec<&IndexEntry> {
        let query_lower = query.to_lowercase();
        self.entries
            .iter()
            .filter(|e| {
                e.title.to_lowercase().contains(&query_lower)
                    || e.summary.to_lowercase().contains(&query_lower)
            })
            .collect()
    }

    /// Write the index to MEMORY.md.
    pub fn save(&self) -> Result<(), Error> {
        let content: String = self
            .entries
            .iter()
            .map(|e| e.to_line())
            .collect::<Vec<_>>()
            .join("\n");
        let path = self.dir.join(INDEX_FILE);
        fs::write(
            path,
            if content.is_empty() {
                String::new()
            } else {
                format!("{content}\n")
            },
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_index_entry() {
        let line = "- [Open Standards](feedback_open_standards.md) — use open specs, not proprietary tools";
        let entry = IndexEntry::parse(line).unwrap();
        assert_eq!(entry.title, "Open Standards");
        assert_eq!(entry.file, "feedback_open_standards.md");
        assert!(entry.summary.contains("open specs"));
    }

    #[test]
    fn roundtrip_entry() {
        let entry = IndexEntry {
            title: "Test".to_string(),
            file: "user_test.md".to_string(),
            summary: "A test entry".to_string(),
        };
        let line = entry.to_line();
        let parsed = IndexEntry::parse(&line).unwrap();
        assert_eq!(parsed.title, "Test");
        assert_eq!(parsed.file, "user_test.md");
    }

    #[test]
    fn search_entries() {
        let index = MemoryIndex {
            entries: vec![
                IndexEntry {
                    title: "Rust Expertise".to_string(),
                    file: "user_rust.md".to_string(),
                    summary: "deep Rust knowledge".to_string(),
                },
                IndexEntry {
                    title: "Open Standards".to_string(),
                    file: "feedback_standards.md".to_string(),
                    summary: "prefer open standards".to_string(),
                },
            ],
            dir: PathBuf::from("."),
        };
        let results = index.search("rust");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Rust Expertise");
    }

    #[test]
    fn init_and_save() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join(".ai-memory");
        let mut index = MemoryIndex::init(&dir).unwrap();
        index
            .add(IndexEntry {
                title: "Test".to_string(),
                file: "user_test.md".to_string(),
                summary: "test entry".to_string(),
            })
            .unwrap();
        index.save().unwrap();

        let reloaded = MemoryIndex::load(&dir).unwrap();
        assert_eq!(reloaded.entries.len(), 1);
    }
}
