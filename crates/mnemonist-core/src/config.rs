use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::Error;

/// Top-level mnemonist configuration.
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
    pub output: OutputConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StorageConfig {
    /// Root directory for all mnemonist data (default: ~/.mnemonist).
    pub root: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    /// Embedding provider: "candle" or "none".
    pub provider: String,
    /// Embedding model name (HuggingFace model identifier).
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
    /// Minimum cosine similarity threshold (0.0-1.0). Results below this are discarded.
    #[serde(default = "default_min_similarity")]
    pub min_similarity: f32,
    /// Minimum number of results to return, even if budget is exceeded.
    #[serde(default = "default_min_results")]
    pub min_results: usize,
}

fn default_true() -> bool {
    true
}

fn default_max_ref_expansions() -> usize {
    3
}

fn default_min_similarity() -> f32 {
    0.35
}

fn default_min_results() -> usize {
    2
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
    /// File name patterns to exclude from code indexing (case-insensitive prefix match).
    /// Empty by default — rely on `recall.min_similarity` to filter irrelevant results.
    pub exclude_patterns: Vec<String>,
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
    /// Max tokens per memory body. Memories are cues, not copies.
    pub max_memory_tokens: usize,
}

/// Configuration for the working memory inbox.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct InboxConfig {
    /// Maximum number of items in the inbox (default 7, like working memory capacity).
    pub capacity: usize,
}

/// Configuration for CLI output behaviour.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputConfig {
    /// Suppress elapsed-time reporting on stderr.
    pub quiet: bool,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            decay_days: 90,
            merge_threshold: 0.85,
            protected_access_count: 5,
            max_memories: 200,
            max_memory_tokens: 120,
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
            .map(|d| d.join(".mnemonist").display().to_string())
            .unwrap_or_else(|| "~/.mnemonist".to_string());
        Self { root }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: "candle".to_string(),
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
            min_similarity: 0.35,
            min_results: 2,
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
            exclude_patterns: vec![
                // Build output / bundles
                "dist".to_string(),
                "build".to_string(),
                "out".to_string(),
                "_build".to_string(),
                // Dependencies
                "node_modules".to_string(),
                "vendor".to_string(),
                ".venv".to_string(),
                "venv".to_string(),
                // Caches / generated
                "target".to_string(),
                "__pycache__".to_string(),
                ".next".to_string(),
                ".nuxt".to_string(),
                ".turbo".to_string(),
                // Lock files (large, low signal)
                "package-lock".to_string(),
                "yarn.lock".to_string(),
                "pnpm-lock".to_string(),
                "cargo.lock".to_string(),
                "poetry.lock".to_string(),
                // Minified / compiled assets
                ".min.js".to_string(),
                ".min.css".to_string(),
                ".map".to_string(),
                ".chunk.".to_string(),
                ".bundle.".to_string(),
            ],
        }
    }
}

impl Config {
    /// Load global config from `~/.mnemonist/mnemonist.toml`. Falls back to defaults if missing.
    pub fn load() -> Self {
        Self::load_global()
    }

    /// Load global config, then overlay with `mnemonist.toml` from the project root (git root).
    ///
    /// Fields present in the project file override the global config; missing fields
    /// inherit from the global config (or defaults).
    pub fn load_with_project(project_root: &Path) -> Self {
        let base = Self::load_global();
        let project_path = project_root.join("mnemonist.toml");

        if !project_path.exists() {
            return base;
        }

        let Ok(project_content) = fs::read_to_string(&project_path) else {
            return base;
        };

        let Ok(project_table) = project_content.parse::<toml::Value>() else {
            return base;
        };

        // Serialize base config to a TOML Value, deep-merge project on top, deserialize back.
        let Ok(mut base_val) = toml::Value::try_from(&base) else {
            return base;
        };

        deep_merge(&mut base_val, &project_table);

        base_val.try_into().unwrap_or(base)
    }

    fn load_global() -> Self {
        let default_root = dirs::home_dir()
            .map(|d| d.join(".mnemonist"))
            .unwrap_or_else(|| PathBuf::from(".mnemonist"));

        // Prefer mnemonist.toml; fall back to legacy config.toml.
        let candidates = [
            default_root.join("mnemonist.toml"),
            default_root.join("config.toml"),
        ];

        for config_path in &candidates {
            if config_path.exists()
                && let Ok(content) = fs::read_to_string(config_path)
                && let Ok(config) = toml::from_str::<Config>(&content)
            {
                return config;
            }
        }

        Self::default()
    }

    /// Save config to `~/.mnemonist/mnemonist.toml`.
    pub fn save(&self) -> Result<(), Error> {
        let root = self.root();
        fs::create_dir_all(&root)?;
        let config_path = root.join("mnemonist.toml");
        let content =
            toml::to_string_pretty(self).map_err(|e| Error::ConfigFormat(e.to_string()))?;
        fs::write(config_path, content)?;
        Ok(())
    }

    /// Config file path.
    pub fn path(&self) -> PathBuf {
        self.root().join("mnemonist.toml")
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

/// Recursively merge `overlay` into `base`. Tables are merged key-by-key;
/// all other values in `overlay` replace those in `base`.
fn deep_merge(base: &mut toml::Value, overlay: &toml::Value) {
    if let (Some(base_table), Some(overlay_table)) = (base.as_table_mut(), overlay.as_table()) {
        for (key, overlay_val) in overlay_table {
            if let Some(base_val) = base_table.get_mut(key) {
                deep_merge(base_val, overlay_val);
            } else {
                base_table.insert(key.clone(), overlay_val.clone());
            }
        }
    } else {
        *base = overlay.clone();
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
        let expanded = expand_tilde("~/.mnemonist");
        assert!(!expanded.to_string_lossy().starts_with("~"));
    }

    #[test]
    fn snapshot_default_config_toml() {
        let mut config = Config::default();
        // Use a fixed root so the snapshot is deterministic
        config.storage.root = "~/.mnemonist".to_string();
        let toml_str = toml::to_string_pretty(&config).unwrap();
        insta::assert_snapshot!(toml_str);
    }
}
