use thiserror::Error as ThisError;

#[derive(Debug, ThisError)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("YAML parse error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("missing frontmatter in {path}")]
    MissingFrontmatter { path: String },

    #[error("memory file not found: {path}")]
    NotFound { path: String },

    #[error("duplicate memory: {name}")]
    Duplicate { name: String },

    #[error("index too large ({lines} lines, max {max})")]
    IndexTooLarge { lines: usize, max: usize },

    #[error("embedding format error: {0}")]
    EmbeddingFormat(String),
}
