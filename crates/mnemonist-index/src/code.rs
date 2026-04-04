use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use ignore::WalkBuilder;
use mnemonist_core::Error;
use tree_sitter::{Language, Parser};

use crate::AnnIndex;

/// A chunk of source code extracted via tree-sitter.
#[derive(Debug, Clone)]
pub struct CodeChunk {
    /// Relative file path from project root.
    pub file: String,
    /// Start line (1-indexed).
    pub start_line: usize,
    /// End line (1-indexed).
    pub end_line: usize,
    /// The source code content.
    pub content: String,
    /// The kind of syntax node (e.g., "function_item", "class_definition").
    pub kind: String,
    /// Optional name (function/class name).
    pub name: Option<String>,
}

impl CodeChunk {
    /// A unique ID for this chunk.
    pub fn id(&self) -> String {
        format!("{}:{}:{}", self.file, self.start_line, self.end_line)
    }
}

/// A search result from code search.
#[derive(Debug, Clone)]
pub struct CodeSearchHit {
    pub chunk: CodeChunk,
    pub score: f32,
}

/// Indexes source code using tree-sitter for semantic chunking.
pub struct CodeIndex {
    root: PathBuf,
    chunks: Vec<CodeChunk>,
    chunk_map: HashMap<String, usize>, // id -> index into chunks
}

/// Tree-sitter node kinds that represent meaningful code boundaries.
/// These are the nodes we extract as chunks.
const CHUNK_NODE_KINDS: &[&str] = &[
    // Rust
    "function_item",
    "impl_item",
    "struct_item",
    "enum_item",
    "trait_item",
    "mod_item",
    "const_item",
    "static_item",
    "type_item",
    // Python
    "function_definition",
    "class_definition",
    "decorated_definition",
    // JavaScript/TypeScript
    "function_declaration",
    "class_declaration",
    "method_definition",
    "arrow_function",
    "export_statement",
    "lexical_declaration",
    // Go
    "function_declaration",
    "method_declaration",
    "type_declaration",
];

/// Max lines for a single chunk before splitting.
const MAX_CHUNK_LINES: usize = 100;

/// Min lines to bother creating a chunk.
const MIN_CHUNK_LINES: usize = 3;

impl CodeIndex {
    /// Create a new empty code index for a project root.
    pub fn new(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
            chunks: Vec::new(),
            chunk_map: HashMap::new(),
        }
    }

    /// Walk the project and extract chunks from all supported source files.
    pub fn index(&mut self) -> Result<usize, Error> {
        self.chunks.clear();
        self.chunk_map.clear();

        let walker = WalkBuilder::new(&self.root)
            .hidden(true)
            .git_ignore(true)
            .git_global(true)
            .build();

        for entry in walker {
            let entry = entry.map_err(|e| Error::Io(std::io::Error::other(e.to_string())))?;
            let path = entry.path();

            if !path.is_file() {
                continue;
            }

            let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
                continue;
            };

            let language = language_for_ext(ext);

            let content = match fs::read_to_string(path) {
                Ok(c) => c,
                Err(_) => continue, // skip binary files
            };

            let rel_path = path
                .strip_prefix(&self.root)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();

            let file_chunks = match language {
                Some(lang) => extract_chunks(&content, &rel_path, lang)?,
                None => extract_plain_chunks(&content, &rel_path),
            };
            for chunk in file_chunks {
                let id = chunk.id();
                let idx = self.chunks.len();
                self.chunks.push(chunk);
                self.chunk_map.insert(id, idx);
            }
        }

        Ok(self.chunks.len())
    }

    /// Get all chunks.
    pub fn chunks(&self) -> &[CodeChunk] {
        &self.chunks
    }

    /// Get a chunk by ID.
    pub fn get(&self, id: &str) -> Option<&CodeChunk> {
        self.chunk_map.get(id).map(|&idx| &self.chunks[idx])
    }

    /// Build an ANN index from chunks using the given embedder.
    pub fn build_ann(
        &self,
        embedder: &dyn mnemonist_core::Embedder,
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

/// Extract semantic chunks from source code using tree-sitter.
fn extract_chunks(source: &str, file: &str, language: Language) -> Result<Vec<CodeChunk>, Error> {
    let mut parser = Parser::new();
    parser
        .set_language(&language)
        .map_err(|e| Error::EmbeddingFormat(format!("tree-sitter language error: {e}")))?;

    let tree = parser
        .parse(source, None)
        .ok_or_else(|| Error::EmbeddingFormat(format!("failed to parse {file}")))?;

    let mut chunks = Vec::new();
    let lines: Vec<&str> = source.lines().collect();

    collect_chunks(tree.root_node(), source, file, &lines, &mut chunks);

    // If no semantic chunks found, fall back to line-based chunking
    if chunks.is_empty() && lines.len() >= MIN_CHUNK_LINES {
        for start in (0..lines.len()).step_by(MAX_CHUNK_LINES) {
            let end = (start + MAX_CHUNK_LINES).min(lines.len());
            let content = lines[start..end].join("\n");
            if content.trim().is_empty() {
                continue;
            }
            chunks.push(CodeChunk {
                file: file.to_string(),
                start_line: start + 1,
                end_line: end,
                content,
                kind: "block".to_string(),
                name: None,
            });
        }
    }

    Ok(chunks)
}

/// Recursively collect semantic chunks from the tree.
fn collect_chunks(
    node: tree_sitter::Node<'_>,
    source: &str,
    file: &str,
    lines: &[&str],
    chunks: &mut Vec<CodeChunk>,
) {
    let kind = node.kind();

    if CHUNK_NODE_KINDS.contains(&kind) {
        let start_line = node.start_position().row;
        let end_line = node.end_position().row;
        let line_count = end_line - start_line + 1;

        if line_count >= MIN_CHUNK_LINES {
            let content = if end_line < lines.len() {
                lines[start_line..=end_line].join("\n")
            } else {
                node.utf8_text(source.as_bytes()).unwrap_or("").to_string()
            };

            // Try to extract name from first named child
            let name = node
                .child_by_field_name("name")
                .and_then(|n| n.utf8_text(source.as_bytes()).ok())
                .map(|s| s.to_string());

            // If the chunk is too large, split it
            if line_count > MAX_CHUNK_LINES {
                for sub_start in (start_line..=end_line).step_by(MAX_CHUNK_LINES) {
                    let sub_end = (sub_start + MAX_CHUNK_LINES - 1).min(end_line);
                    let sub_content = if sub_end < lines.len() {
                        lines[sub_start..=sub_end].join("\n")
                    } else {
                        continue;
                    };
                    chunks.push(CodeChunk {
                        file: file.to_string(),
                        start_line: sub_start + 1,
                        end_line: sub_end + 1,
                        content: sub_content,
                        kind: kind.to_string(),
                        name: name.clone(),
                    });
                }
            } else {
                chunks.push(CodeChunk {
                    file: file.to_string(),
                    start_line: start_line + 1,
                    end_line: end_line + 1,
                    content,
                    kind: kind.to_string(),
                    name,
                });
            }

            return; // Don't recurse into children of a chunk node
        }
    }

    // Recurse into children
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_chunks(child, source, file, lines, chunks);
    }
}

/// Extract chunks from a plain-text file using line-based chunking.
/// Used as a fallback for file types without tree-sitter grammar support.
fn extract_plain_chunks(source: &str, file: &str) -> Vec<CodeChunk> {
    let lines: Vec<&str> = source.lines().collect();
    if lines.len() < MIN_CHUNK_LINES {
        return Vec::new();
    }
    let mut chunks = Vec::new();
    for start in (0..lines.len()).step_by(MAX_CHUNK_LINES) {
        let end = (start + MAX_CHUNK_LINES).min(lines.len());
        let content = lines[start..end].join("\n");
        if content.trim().is_empty() {
            continue;
        }
        chunks.push(CodeChunk {
            file: file.to_string(),
            start_line: start + 1,
            end_line: end,
            content,
            kind: "block".to_string(),
            name: None,
        });
    }
    chunks
}

/// Map file extension to tree-sitter language.
fn language_for_ext(ext: &str) -> Option<Language> {
    match ext {
        #[cfg(feature = "lang-rust")]
        "rs" => Some(tree_sitter_rust::LANGUAGE.into()),
        #[cfg(feature = "lang-python")]
        "py" => Some(tree_sitter_python::LANGUAGE.into()),
        #[cfg(feature = "lang-javascript")]
        "js" | "jsx" | "mjs" | "cjs" => Some(tree_sitter_javascript::LANGUAGE.into()),
        #[cfg(feature = "lang-javascript")]
        "ts" | "tsx" => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        #[cfg(feature = "lang-go")]
        "go" => Some(tree_sitter_go::LANGUAGE.into()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_rust_chunks() {
        let source = r#"
fn hello() {
    println!("hello");
}

fn world() {
    println!("world");
}

struct Foo {
    bar: i32,
    baz: String,
}
"#;
        let lang: Language = tree_sitter_rust::LANGUAGE.into();
        let chunks = extract_chunks(source, "test.rs", lang).unwrap();
        assert!(
            chunks.len() >= 2,
            "expected at least 2 chunks, got {}",
            chunks.len()
        );

        let hello = chunks.iter().find(|c| c.name.as_deref() == Some("hello"));
        assert!(hello.is_some(), "should find hello function");
    }

    #[test]
    fn chunk_id_format() {
        let chunk = CodeChunk {
            file: "src/main.rs".into(),
            start_line: 10,
            end_line: 25,
            content: String::new(),
            kind: "function_item".into(),
            name: Some("main".into()),
        };
        assert_eq!(chunk.id(), "src/main.rs:10:25");
    }

    #[test]
    fn language_mapping() {
        assert!(language_for_ext("rs").is_some());
        assert!(language_for_ext("py").is_some());
        assert!(language_for_ext("js").is_some());
        assert!(language_for_ext("ts").is_some());
        assert!(language_for_ext("go").is_some());
        assert!(language_for_ext("txt").is_none());
    }
}
