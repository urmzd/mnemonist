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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RecallConfig {
    /// Max output chars for recall (default for `recall --budget`).
    pub budget: usize,
    /// Follow inter-layer refs edges during recall.
    #[serde(default = "default_true")]
    pub expand_refs: bool,
    /// Max code chunks to include via ref expansion per memory hit.
    #[serde(default = "default_max_ref_expansions")]
    pub max_ref_expansions: usize,
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
    /// File name patterns to exclude from code indexing (case-insensitive prefix match).
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
    /// Max tokens per memory body. Memories are cues, not copies.
    pub max_memory_tokens: usize,
    /// Run consolidation automatically as a detached background job after
    /// inbox writes (like `git gc --auto`). Triggers on inbox pressure
    /// (>= 80% full) or staleness. MNEMONIST_NO_AUTO_CONSOLIDATE=1 overrides.
    pub auto: bool,
    /// Days since the last consolidation before an inbox write also triggers
    /// an automatic background run.
    pub auto_stale_days: u64,
}

/// Configuration for the working memory inbox.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct InboxConfig {
    /// Maximum number of items in the inbox (default 10).
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
            max_memory_tokens: 120,
            auto: true,
            auto_stale_days: 7,
        }
    }
}

impl Default for InboxConfig {
    fn default() -> Self {
        Self { capacity: 10 }
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
            model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        }
    }
}

impl Default for RecallConfig {
    fn default() -> Self {
        Self {
            budget: 2000,
            expand_refs: true,
            max_ref_expansions: 3,
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

        let project_table = match project_content.parse::<toml::Value>() {
            Ok(table) => table,
            Err(e) => {
                warn_unparseable(&project_path, &e.to_string());
                return base;
            }
        };

        // Serialize base config to a TOML Value, deep-merge project on top, deserialize back.
        let Ok(mut base_val) = toml::Value::try_from(&base) else {
            return base;
        };

        deep_merge(&mut base_val, &project_table);

        match base_val.try_into() {
            Ok(merged) => merged,
            Err(e) => {
                warn_unparseable(&project_path, &e.to_string());
                base
            }
        }
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
            if !config_path.exists() {
                continue;
            }
            let Ok(content) = fs::read_to_string(config_path) else {
                continue;
            };
            match toml::from_str::<Config>(&content) {
                Ok(config) => return config,
                Err(e) => warn_unparseable(config_path, &e.to_string()),
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

/// Warn (once per process — `Config::load` runs several times per command)
/// when a config file exists but cannot be used, instead of silently falling
/// back to defaults.
fn warn_unparseable(path: &Path, err: &str) {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        eprintln!(
            "warning: ignoring unparseable config {}: {err}",
            path.display()
        );
    });
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
        assert_eq!(
            parsed.embedding.model,
            "sentence-transformers/all-MiniLM-L6-v2"
        );
        assert_eq!(parsed.index.max_lines, 200);
        assert_eq!(parsed.recall.budget, 2000);
    }

    #[test]
    fn get_dot_notation() {
        let config = Config::default();
        assert_eq!(
            config.get("embedding.model"),
            Some("sentence-transformers/all-MiniLM-L6-v2".to_string())
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

    /// Every key accepted by `config set` must be read somewhere in non-test
    /// code: a settable key nobody consumes is a lying API surface.
    /// Consumers are detected by the `.section.key` field-access pattern.
    #[test]
    fn every_config_key_has_a_consumer() {
        fn collect_keys(val: &toml::Value, prefix: &str, out: &mut Vec<String>) {
            if let Some(table) = val.as_table() {
                for (k, v) in table {
                    let path = if prefix.is_empty() {
                        k.clone()
                    } else {
                        format!("{prefix}.{k}")
                    };
                    if v.is_table() {
                        collect_keys(v, &path, out);
                    } else {
                        out.push(path);
                    }
                }
            }
        }

        fn collect_sources(dir: &Path, out: &mut String) {
            for entry in fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    collect_sources(&path, out);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    let content = fs::read_to_string(&path).unwrap();
                    // Everything from the first unit-test marker on is ignored:
                    // a consumer that only exists in tests doesn't count.
                    let code = content.split("#[cfg(test)]").next().unwrap_or("");
                    out.push_str(code);
                }
            }
        }

        let val = toml::Value::try_from(Config::default()).unwrap();
        let mut keys = Vec::new();
        collect_keys(&val, "", &mut keys);
        assert!(!keys.is_empty());

        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut sources = String::new();
        collect_sources(&manifest.join("src"), &mut sources);
        collect_sources(&manifest.join("../mnemonist-cli/src"), &mut sources);

        let missing: Vec<String> = keys
            .into_iter()
            .filter(|key| !sources.contains(&format!(".{key}")))
            .collect();

        assert!(
            missing.is_empty(),
            "config keys accepted by `config set` but never read: {missing:?} — wire them up or delete them"
        );
    }
}
