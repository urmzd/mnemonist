use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::Error;

/// Top-level llmem configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub storage: StorageConfig,
    pub embedding: EmbeddingConfig,
    pub recall: RecallConfig,
    pub index: IndexConfig,
    pub code: CodeConfig,
    pub quantization: QuantizationConfig,
    pub consolidation: ConsolidationConfig,
    pub inbox: InboxConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StorageConfig {
    /// Root directory for all llmem data (default: ~/.llmem).
    pub root: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    /// Embedding provider: "fastembed" or "none".
    pub provider: String,
    /// Embedding model name (fastembed model identifier).
    pub model: String,
}

/// Configuration for TurboQuant vector quantization.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct QuantizationConfig {
    /// Enable quantized embedding storage.
    pub enabled: bool,
    /// Bit-width per coordinate (1-4).
    pub bits: u8,
    /// Quantization algorithm: "mse" or "prod".
    pub algorithm: String,
    /// Temporal re-ranking weight λ ∈ [0, 1]. 0 = pure cosine, 1 = pure temporal.
    pub temporal_weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RecallConfig {
    /// Max output chars for recall.
    pub budget: usize,
    /// Type priority order for recall (highest first).
    pub priority: Vec<String>,
    /// Follow inter-layer refs edges during recall.
    #[serde(default = "default_true")]
    pub expand_refs: bool,
    /// Max code chunks to include via ref expansion per memory hit.
    #[serde(default = "default_max_ref_expansions")]
    pub max_ref_expansions: usize,
}

fn default_true() -> bool {
    true
}

fn default_max_ref_expansions() -> usize {
    3
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IndexConfig {
    /// Max entries in MEMORY.md.
    pub max_lines: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CodeConfig {
    /// Supported languages for code indexing.
    pub languages: Vec<String>,
    /// Max lines per code chunk.
    pub max_chunk_lines: usize,
}

/// Configuration for memory consolidation (sleep cycle).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ConsolidationConfig {
    /// Days since last access before a memory is considered for decay.
    pub decay_days: u64,
    /// Cosine similarity threshold for merging memories (0.0-1.0).
    pub merge_threshold: f32,
    /// Minimum access count to protect a memory from decay.
    pub protected_access_count: u32,
    /// Maximum memories per level before pruning is forced.
    pub max_memories: usize,
}

/// Configuration for the working memory inbox.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct InboxConfig {
    /// Maximum number of items in the inbox (default 7, like working memory capacity).
    pub capacity: usize,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            decay_days: 90,
            merge_threshold: 0.85,
            protected_access_count: 5,
            max_memories: 200,
        }
    }
}

impl Default for InboxConfig {
    fn default() -> Self {
        Self { capacity: 7 }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        let root = dirs::home_dir()
            .map(|d| d.join(".llmem").display().to_string())
            .unwrap_or_else(|| "~/.llmem".to_string());
        Self { root }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: "fastembed".to_string(),
            model: "all-MiniLM-L6-v2".to_string(),
        }
    }
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            bits: 2,
            algorithm: "mse".to_string(),
            temporal_weight: 0.2,
        }
    }
}

impl Default for RecallConfig {
    fn default() -> Self {
        Self {
            budget: 2000,
            priority: vec![
                "feedback".to_string(),
                "project".to_string(),
                "user".to_string(),
                "reference".to_string(),
            ],
            expand_refs: true,
            max_ref_expansions: 3,
        }
    }
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self { max_lines: 200 }
    }
}

impl Default for CodeConfig {
    fn default() -> Self {
        Self {
            languages: vec![
                "rust".to_string(),
                "python".to_string(),
                "javascript".to_string(),
                "go".to_string(),
            ],
            max_chunk_lines: 100,
        }
    }
}

impl Config {
    /// Load config from `~/.llmem/config.toml`. Falls back to defaults if missing.
    pub fn load() -> Self {
        let default_root = dirs::home_dir()
            .map(|d| d.join(".llmem"))
            .unwrap_or_else(|| PathBuf::from(".llmem"));
        let config_path = default_root.join("config.toml");

        if config_path.exists()
            && let Ok(content) = fs::read_to_string(&config_path)
            && let Ok(config) = toml::from_str::<Config>(&content)
        {
            return config;
        }

        Self::default()
    }

    /// Save config to `~/.llmem/config.toml`.
    pub fn save(&self) -> Result<(), Error> {
        let root = self.root();
        fs::create_dir_all(&root)?;
        let config_path = root.join("config.toml");
        let content =
            toml::to_string_pretty(self).map_err(|e| Error::ConfigFormat(e.to_string()))?;
        fs::write(config_path, content)?;
        Ok(())
    }

    /// Config file path.
    pub fn path(&self) -> PathBuf {
        self.root().join("config.toml")
    }

    /// Resolved root directory.
    pub fn root(&self) -> PathBuf {
        expand_tilde(&self.storage.root)
    }

    /// Global memory directory.
    pub fn global_dir(&self) -> PathBuf {
        self.root().join("global")
    }

    /// Project memory directory derived from project root basename.
    pub fn project_dir(&self, project_root: &Path) -> PathBuf {
        let project_name = project_root
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "default".to_string());
        self.root().join(project_name)
    }

    /// Get a config value by dot-notation key.
    pub fn get(&self, key: &str) -> Option<String> {
        let val = toml::Value::try_from(self).ok()?;
        let mut current = &val;
        for part in key.split('.') {
            current = current.get(part)?;
        }
        match current {
            toml::Value::String(s) => Some(s.clone()),
            toml::Value::Integer(n) => Some(n.to_string()),
            toml::Value::Boolean(b) => Some(b.to_string()),
            toml::Value::Array(arr) => {
                let items: Vec<String> = arr
                    .iter()
                    .map(|v| match v {
                        toml::Value::String(s) => s.clone(),
                        other => other.to_string(),
                    })
                    .collect();
                Some(items.join(", "))
            }
            other => Some(other.to_string()),
        }
    }

    /// Set a config value by dot-notation key.
    pub fn set(&mut self, key: &str, value: &str) -> Result<(), Error> {
        let mut val =
            toml::Value::try_from(&*self).map_err(|e| Error::ConfigFormat(e.to_string()))?;

        let parts: Vec<&str> = key.split('.').collect();
        let (last, path) = parts
            .split_last()
            .ok_or_else(|| Error::ConfigFormat("empty key".into()))?;

        let mut current = &mut val;
        for part in path {
            current = current
                .get_mut(*part)
                .ok_or_else(|| Error::ConfigFormat(format!("unknown key: {key}")))?;
        }

        let table = current
            .as_table_mut()
            .ok_or_else(|| Error::ConfigFormat(format!("{key} is not a table")))?;

        // Try to preserve the original type
        if let Some(existing) = table.get(*last) {
            match existing {
                toml::Value::Integer(_) => {
                    if let Ok(n) = value.parse::<i64>() {
                        table.insert((*last).to_string(), toml::Value::Integer(n));
                    } else {
                        return Err(Error::ConfigFormat(format!(
                            "{key} expects an integer, got: {value}"
                        )));
                    }
                }
                toml::Value::Boolean(_) => {
                    if let Ok(b) = value.parse::<bool>() {
                        table.insert((*last).to_string(), toml::Value::Boolean(b));
                    } else {
                        return Err(Error::ConfigFormat(format!(
                            "{key} expects a boolean, got: {value}"
                        )));
                    }
                }
                toml::Value::Array(_) => {
                    let items: Vec<toml::Value> = value
                        .split(',')
                        .map(|s| toml::Value::String(s.trim().to_string()))
                        .collect();
                    table.insert((*last).to_string(), toml::Value::Array(items));
                }
                _ => {
                    table.insert((*last).to_string(), toml::Value::String(value.to_string()));
                }
            }
        } else {
            return Err(Error::ConfigFormat(format!("unknown key: {key}")));
        }

        // Deserialize back into Config
        *self = val
            .try_into()
            .map_err(|e: toml::de::Error| Error::ConfigFormat(e.to_string()))?;
        Ok(())
    }
}

/// Expand `~` to home directory.
fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = dirs::home_dir()
    {
        return home.join(rest);
    }
    PathBuf::from(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_roundtrips() {
        let config = Config::default();
        let toml_str = toml::to_string_pretty(&config).unwrap();
        let parsed: Config = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.embedding.model, "all-MiniLM-L6-v2");
        assert_eq!(parsed.index.max_lines, 200);
        assert_eq!(parsed.recall.budget, 2000);
    }

    #[test]
    fn get_dot_notation() {
        let config = Config::default();
        assert_eq!(
            config.get("embedding.model"),
            Some("all-MiniLM-L6-v2".to_string())
        );
        assert_eq!(config.get("index.max_lines"), Some("200".to_string()));
        assert_eq!(config.get("nonexistent.key"), None);
    }

    #[test]
    fn set_dot_notation() {
        let mut config = Config::default();
        config.set("embedding.model", "all-minilm").unwrap();
        assert_eq!(config.embedding.model, "all-minilm");

        config.set("index.max_lines", "500").unwrap();
        assert_eq!(config.index.max_lines, 500);
    }

    #[test]
    fn expand_tilde_works() {
        let expanded = expand_tilde("~/.llmem");
        assert!(!expanded.to_string_lossy().starts_with("~"));
    }

    #[test]
    fn snapshot_default_config_toml() {
        let mut config = Config::default();
        // Use a fixed root so the snapshot is deterministic
        config.storage.root = "~/.llmem".to_string();
        let toml_str = toml::to_string_pretty(&config).unwrap();
        insta::assert_snapshot!(toml_str);
    }
}
