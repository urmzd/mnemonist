use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::{Chunk, ChunkingStrategy, Error};
use ignore::WalkBuilder;

use super::AnnIndex;

/// Default max lines per chunk.
const DEFAULT_MAX_LINES: usize = 100;
/// Default min lines to bother creating a chunk.
const DEFAULT_MIN_LINES: usize = 3;
/// Default overlap in lines between consecutive chunks.
const DEFAULT_OVERLAP: usize = 10;

// ---------------------------------------------------------------------------
// Chunking strategies
// ---------------------------------------------------------------------------

/// Splits on blank-line boundaries (paragraphs / function gaps).
///
/// Natural for both prose (paragraphs) and code (functions separated by blank
/// lines). Adjacent small paragraphs are merged up to `max_lines`; oversized
/// paragraphs are split with `overlap`.
pub struct ParagraphChunking {
    pub max_lines: usize,
    pub min_lines: usize,
    pub overlap: usize,
}

impl Default for ParagraphChunking {
    fn default() -> Self {
        Self {
            max_lines: DEFAULT_MAX_LINES,
            min_lines: DEFAULT_MIN_LINES,
            overlap: DEFAULT_OVERLAP,
        }
    }
}

impl ChunkingStrategy for ParagraphChunking {
    fn chunk(&self, content: &str, file: &str) -> Vec<Chunk> {
        let lines: Vec<&str> = content.lines().collect();
        if lines.len() < self.min_lines {
            return Vec::new();
        }

        // Split into paragraphs at blank lines.
        let mut paragraphs: Vec<(usize, usize)> = Vec::new(); // (start, end) 0-indexed
        let mut start = 0;
        while start < lines.len() {
            // Skip leading blank lines.
            if lines[start].trim().is_empty() {
                start += 1;
                continue;
            }
            let mut end = start;
            while end + 1 < lines.len() && !lines[end + 1].trim().is_empty() {
                end += 1;
            }
            paragraphs.push((start, end));
            start = end + 1;
        }

        // Merge small paragraphs, split large ones.
        let mut chunks = Vec::new();
        let mut buf_start: Option<usize> = None;
        let mut buf_end: usize = 0;

        for &(ps, pe) in &paragraphs {
            let para_lines = pe - ps + 1;

            // If this paragraph alone exceeds max, flush buffer then split it.
            if para_lines > self.max_lines {
                // Flush accumulated buffer first.
                if let Some(bs) = buf_start {
                    push_chunk(&mut chunks, &lines, file, bs, buf_end);
                    buf_start = None;
                }
                // Split the oversized paragraph with overlap.
                let mut s = ps;
                while s <= pe {
                    let e = (s + self.max_lines - 1).min(pe);
                    push_chunk(&mut chunks, &lines, file, s, e);
                    if e == pe {
                        break;
                    }
                    s = (e + 1).saturating_sub(self.overlap);
                }
                continue;
            }

            match buf_start {
                None => {
                    buf_start = Some(ps);
                    buf_end = pe;
                }
                Some(bs) => {
                    let merged_lines = pe - bs + 1;
                    if merged_lines <= self.max_lines {
                        // Merge into buffer.
                        buf_end = pe;
                    } else {
                        // Flush buffer, start new.
                        push_chunk(&mut chunks, &lines, file, bs, buf_end);
                        buf_start = Some(ps);
                        buf_end = pe;
                    }
                }
            }
        }

        // Flush remaining buffer.
        if let Some(bs) = buf_start {
            let line_count = buf_end - bs + 1;
            if line_count >= self.min_lines {
                push_chunk(&mut chunks, &lines, file, bs, buf_end);
            }
        }

        chunks
    }
}

/// Fixed-size line windows with optional overlap.
pub struct FixedLineChunking {
    pub chunk_size: usize,
    pub overlap: usize,
    pub min_lines: usize,
}

impl Default for FixedLineChunking {
    fn default() -> Self {
        Self {
            chunk_size: DEFAULT_MAX_LINES,
            overlap: DEFAULT_OVERLAP,
            min_lines: DEFAULT_MIN_LINES,
        }
    }
}

impl ChunkingStrategy for FixedLineChunking {
    fn chunk(&self, content: &str, file: &str) -> Vec<Chunk> {
        let lines: Vec<&str> = content.lines().collect();
        if lines.len() < self.min_lines {
            return Vec::new();
        }
        let step = self.chunk_size.saturating_sub(self.overlap).max(1);
        let mut chunks = Vec::new();
        let mut start = 0;
        while start < lines.len() {
            let end = (start + self.chunk_size).min(lines.len());
            let content = lines[start..end].join("\n");
            if !content.trim().is_empty() {
                chunks.push(Chunk {
                    file: file.to_string(),
                    start_line: start + 1,
                    end_line: end,
                    content,
                });
            }
            let next = start + step;
            if next <= start || end == lines.len() {
                break;
            }
            start = next;
        }
        chunks
    }
}

fn push_chunk(chunks: &mut Vec<Chunk>, lines: &[&str], file: &str, start: usize, end: usize) {
    let content = lines[start..=end].join("\n");
    if !content.trim().is_empty() {
        chunks.push(Chunk {
            file: file.to_string(),
            start_line: start + 1,
            end_line: end + 1,
            content,
        });
    }
}

// ---------------------------------------------------------------------------
// CodeIndex — walks a project, chunks files, builds ANN
// ---------------------------------------------------------------------------

/// A search result from code search.
#[derive(Debug, Clone)]
pub struct CodeSearchHit {
    pub chunk: Chunk,
    pub score: f32,
}

/// Options controlling file discovery during indexing.
#[derive(Debug, Clone)]
pub struct IndexOptions {
    /// Include hidden files and directories (default: false — hidden files are skipped).
    pub hidden: bool,
    /// Respect `.gitignore` rules (default: true).
    pub git_ignore: bool,
    /// Additional glob patterns to exclude (e.g. `["dist/**", "*.min.js"]`).
    pub exclude_globs: Vec<String>,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            hidden: false,
            git_ignore: true,
            exclude_globs: Vec::new(),
        }
    }
}

/// Indexes source files using a pluggable [`ChunkingStrategy`].
pub struct CodeIndex<'a> {
    root: PathBuf,
    strategy: &'a dyn ChunkingStrategy,
    chunks: Vec<Chunk>,
    chunk_map: HashMap<String, usize>,
}

impl<'a> CodeIndex<'a> {
    /// Create a new code index for a project root with the given chunking strategy.
    pub fn new(root: &Path, strategy: &'a dyn ChunkingStrategy) -> Self {
        Self {
            root: root.to_path_buf(),
            strategy,
            chunks: Vec::new(),
            chunk_map: HashMap::new(),
        }
    }

    /// Walk the project and extract chunks from all text files.
    ///
    /// Files whose name starts with any pattern in `exclude_patterns` (case-insensitive)
    /// are skipped. Pass an empty slice to index everything.
    pub fn index(&mut self, exclude_patterns: &[String]) -> Result<usize, Error> {
        self.index_with_options(exclude_patterns, &IndexOptions::default())
    }

    /// Walk the project with explicit walker options.
    pub fn index_with_options(
        &mut self,
        exclude_patterns: &[String],
        opts: &IndexOptions,
    ) -> Result<usize, Error> {
        self.chunks.clear();
        self.chunk_map.clear();

        let mut builder = WalkBuilder::new(&self.root);
        builder
            .hidden(!opts.hidden)
            .git_ignore(opts.git_ignore)
            .git_global(opts.git_ignore);

        if !opts.exclude_globs.is_empty() {
            let mut ob = ignore::overrides::OverrideBuilder::new(&self.root);
            for pat in &opts.exclude_globs {
                ob.add(&format!("!{pat}"))
                    .map_err(|e| Error::Io(std::io::Error::other(e.to_string())))?;
            }
            let overrides = ob
                .build()
                .map_err(|e| Error::Io(std::io::Error::other(e.to_string())))?;
            builder.overrides(overrides);
        }

        let walker = builder.build();

        for entry in walker {
            let entry = entry.map_err(|e| Error::Io(std::io::Error::other(e.to_string())))?;
            let path = entry.path();

            if !path.is_file() {
                continue;
            }

            // Skip files matching exclude patterns (case-insensitive prefix on file name
            // or any path component)
            {
                let rel = path
                    .strip_prefix(&self.root)
                    .unwrap_or(path)
                    .to_string_lossy()
                    .to_lowercase();
                if exclude_patterns.iter().any(|p| {
                    let p_lower = p.to_lowercase();
                    // Match against filename prefix or any path component prefix
                    rel.split('/').any(|seg| seg.starts_with(&p_lower))
                }) {
                    continue;
                }
            }

            let content = match fs::read_to_string(path) {
                Ok(c) => c,
                Err(_) => continue, // skip binary files
            };

            let rel_path = path
                .strip_prefix(&self.root)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();

            for chunk in self.strategy.chunk(&content, &rel_path) {
                let id = chunk.id();
                let idx = self.chunks.len();
                self.chunks.push(chunk);
                self.chunk_map.insert(id, idx);
            }
        }

        Ok(self.chunks.len())
    }

    /// Get all chunks.
    pub fn chunks(&self) -> &[Chunk] {
        &self.chunks
    }

    /// Get a chunk by ID.
    pub fn get(&self, id: &str) -> Option<&Chunk> {
        self.chunk_map.get(id).map(|&idx| &self.chunks[idx])
    }

    /// Build an ANN index from chunks using the given embedder.
    pub fn build_ann(
        &self,
        embedder: &dyn crate::Embedder,
        ann: &mut dyn AnnIndex,
    ) -> Result<usize, Error> {
        let mut count = 0;
        for chunk in &self.chunks {
            let embedding = embedder.embed(&chunk.content)?;
            ann.insert(&chunk.id(), &embedding)?;
            count += 1;
        }
        Ok(count)
    }

    /// Search for code using an ANN index, resolving hits to chunks.
    pub fn search(
        &self,
        ann: &dyn AnnIndex,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<CodeSearchHit>, Error> {
        let hits = ann.search(query_embedding, top_k)?;
        Ok(hits
            .into_iter()
            .filter_map(|hit| {
                self.get(&hit.id).map(|chunk| CodeSearchHit {
                    chunk: chunk.clone(),
                    score: hit.score,
                })
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paragraph_chunking_splits_on_blank_lines() {
        let source = "\
fn hello() {
    println!(\"hello\");
}

fn world() {
    println!(\"world\");
}

struct Foo {
    bar: i32,
    baz: String,
}";
        let strategy = ParagraphChunking {
            max_lines: 100,
            min_lines: 2,
            ..Default::default()
        };
        let chunks = strategy.chunk(source, "test.rs");
        // Three paragraphs merged into one since total < max_lines
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("hello"));
        assert!(chunks[0].content.contains("Foo"));
    }

    #[test]
    fn paragraph_chunking_splits_large_blocks() {
        let strategy = ParagraphChunking {
            max_lines: 5,
            min_lines: 2,
            overlap: 1,
        };
        // 10 non-blank lines in a single paragraph
        let source = (0..10)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let chunks = strategy.chunk(&source, "big.txt");
        assert!(chunks.len() >= 2, "should split into multiple chunks");
    }

    #[test]
    fn fixed_line_chunking_with_overlap() {
        let source = (0..20)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let strategy = FixedLineChunking {
            chunk_size: 10,
            overlap: 2,
            min_lines: 1,
        };
        let chunks = strategy.chunk(&source, "test.txt");
        assert!(chunks.len() >= 2);
        // Second chunk should start before the end of the first
        assert!(chunks[1].start_line < chunks[0].end_line + 1);
    }

    #[test]
    fn chunk_id_format() {
        let chunk = Chunk {
            file: "src/main.rs".into(),
            start_line: 10,
            end_line: 25,
            content: String::new(),
        };
        assert_eq!(chunk.id(), "src/main.rs:10:25");
    }

    #[test]
    fn skips_tiny_files() {
        let strategy = ParagraphChunking::default();
        let chunks = strategy.chunk("hi", "tiny.txt");
        assert!(chunks.is_empty());
    }
}
