use std::sync::Mutex;

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};

use crate::Error;
use crate::embed::Embedder;

/// Local ONNX-based embedder using fastembed.
///
/// Downloads models on first use to `~/.cache/fastembed/`.
/// No external server required. Thread-safe via interior mutability.
pub struct FastEmbedder {
    model: Mutex<TextEmbedding>,
    dimension: usize,
}

impl FastEmbedder {
    /// Create a new embedder with the given model.
    pub fn new(model_name: EmbeddingModel) -> Result<Self, Error> {
        let mut model = TextEmbedding::try_new(TextInitOptions::new(model_name))
            .map_err(|e| Error::Embedding(format!("fastembed init failed: {e}")))?;

        // Probe dimension with a short text
        let probe = model
            .embed(["probe"], None)
            .map_err(|e| Error::Embedding(format!("fastembed probe failed: {e}")))?;
        let dimension = probe
            .first()
            .map(|v: &Vec<f32>| v.len())
            .ok_or_else(|| Error::Embedding("fastembed returned no embeddings".into()))?;

        Ok(Self {
            model: Mutex::new(model),
            dimension,
        })
    }

    /// Create with the default model (all-MiniLM-L6-v2, 384 dims, ~22MB).
    pub fn default_model() -> Result<Self, Error> {
        Self::new(EmbeddingModel::AllMiniLML6V2)
    }

    /// Create from config, mapping model name string to EmbeddingModel.
    pub fn from_config(config: &crate::Config) -> Result<Self, Error> {
        let model_name = match config.embedding.model.as_str() {
            "all-MiniLM-L6-v2" => EmbeddingModel::AllMiniLML6V2,
            "all-MiniLM-L6-v2-q" => EmbeddingModel::AllMiniLML6V2Q,
            "all-MiniLM-L12-v2" => EmbeddingModel::AllMiniLML12V2,
            "BGE-small-en-v1.5" => EmbeddingModel::BGESmallENV15,
            "BGE-small-en-v1.5-q" => EmbeddingModel::BGESmallENV15Q,
            _ => {
                return Err(Error::Embedding(format!(
                    "unknown fastembed model: {}",
                    config.embedding.model
                )));
            }
        };
        Self::new(model_name)
    }
}

impl Embedder for FastEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, Error> {
        let mut model = self
            .model
            .lock()
            .map_err(|e| Error::Embedding(format!("lock poisoned: {e}")))?;

        let results = model
            .embed([text], None)
            .map_err(|e| Error::Embedding(format!("fastembed embed failed: {e}")))?;

        results
            .into_iter()
            .next()
            .ok_or_else(|| Error::Embedding("fastembed returned no embeddings".into()))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, Error> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut model = self
            .model
            .lock()
            .map_err(|e| Error::Embedding(format!("lock poisoned: {e}")))?;

        model
            .embed(texts, None)
            .map_err(|e| Error::Embedding(format!("fastembed batch embed failed: {e}")))
    }

    fn dimension(&self) -> Result<usize, Error> {
        Ok(self.dimension)
    }
}
