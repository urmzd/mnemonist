use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::api::sync::Api;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

use crate::Error;
use crate::embed::Embedder;

const DEFAULT_MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
const MAX_SEQ_LEN: usize = 256;
const MICRO_BATCH: usize = 32;

/// Candle-based embedder with Accelerate/CUDA acceleration.
///
/// Both `BertModel::forward` and `Tokenizer::encode_batch` take `&self`,
/// so no Mutex is needed — this is freely `Sync` and can be shared across
/// rayon threads for parallel embedding.
pub struct CandleEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    dimension: usize,
}

// Tokenizer is Send+Sync but doesn't declare it; the inner data is thread-safe.
unsafe impl Send for CandleEmbedder {}
unsafe impl Sync for CandleEmbedder {}

impl CandleEmbedder {
    /// Create with default model (all-MiniLM-L6-v2) on the best available device.
    pub fn default_model() -> Result<Self, Error> {
        Self::from_model(DEFAULT_MODEL_ID)
    }

    /// Create with the given HuggingFace model id on the best available device.
    pub fn from_model(model_id: &str) -> Result<Self, Error> {
        let device = Self::best_device()?;
        Self::load(model_id, device)
    }

    /// Select the best available device (CUDA > CPU).
    fn best_device() -> Result<Device, Error> {
        #[cfg(feature = "cuda")]
        {
            if let Ok(dev) = Device::new_cuda(0) {
                return Ok(dev);
            }
        }
        Ok(Device::Cpu)
    }

    /// Load a sentence-transformer model from the HuggingFace Hub.
    pub fn load(model_id: &str, device: Device) -> Result<Self, Error> {
        let api = Api::new().map_err(|e| Error::Embedding(format!("hf-hub init: {e}")))?;
        let repo = api.model(model_id.to_string());

        let config_path = Self::get_file(&repo, "config.json")?;
        let tokenizer_path = Self::get_file(&repo, "tokenizer.json")?;
        let weights_path = Self::get_file(&repo, "model.safetensors")?;

        // Load config
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| Error::Embedding(format!("read config: {e}")))?;
        let config: BertConfig = serde_json::from_str(&config_str)
            .map_err(|e| Error::Embedding(format!("parse config: {e}")))?;

        let dimension = config.hidden_size;

        // Load weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                .map_err(|e| Error::Embedding(format!("load weights: {e}")))?
        };

        // Build model
        let model = BertModel::load(vb, &config)
            .map_err(|e| Error::Embedding(format!("build model: {e}")))?;

        // Setup tokenizer
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Embedding(format!("load tokenizer: {e}")))?;

        tokenizer.with_padding(Some(PaddingParams::default()));
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_SEQ_LEN,
                ..Default::default()
            }))
            .map_err(|e| Error::Embedding(format!("set truncation: {e}")))?;

        Ok(Self {
            model,
            tokenizer,
            device,
            dimension,
        })
    }

    fn get_file(repo: &hf_hub::api::sync::ApiRepo, filename: &str) -> Result<PathBuf, Error> {
        repo.get(filename)
            .map_err(|e| Error::Embedding(format!("download {filename}: {e}")))
    }

    /// Mean-pool token embeddings using the attention mask, then L2-normalize.
    fn mean_pool_normalize(embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor, Error> {
        let map_err = |e: candle_core::Error| Error::Embedding(format!("pool: {e}"));

        // attention_mask: [batch, seq_len] -> [batch, seq_len, 1]
        let mask = attention_mask
            .unsqueeze(2)
            .map_err(map_err)?
            .to_dtype(embeddings.dtype())
            .map_err(map_err)?;

        // Masked sum over seq_len dimension
        let masked = embeddings.broadcast_mul(&mask).map_err(map_err)?;
        let summed = masked.sum(1).map_err(map_err)?; // [batch, hidden]

        // Count of non-padding tokens
        let mask_sum = mask
            .sum(1)
            .map_err(map_err)?
            .clamp(1e-9, f64::MAX)
            .map_err(map_err)?; // [batch, 1]

        let mean = summed.broadcast_div(&mask_sum).map_err(map_err)?;

        // L2 normalize
        let norm = mean
            .sqr()
            .map_err(map_err)?
            .sum_keepdim(1)
            .map_err(map_err)?
            .sqrt()
            .map_err(map_err)?
            .clamp(1e-12, f64::MAX)
            .map_err(map_err)?;
        mean.broadcast_div(&norm).map_err(map_err)
    }
}

impl Embedder for CandleEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, Error> {
        let batch = self.embed_batch(&[text])?;
        batch
            .into_iter()
            .next()
            .ok_or_else(|| Error::Embedding("empty result".into()))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, Error> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_vecs = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(MICRO_BATCH) {
            let encodings = self
                .tokenizer
                .encode_batch(chunk.to_vec(), true)
                .map_err(|e| Error::Embedding(format!("tokenize: {e}")))?;

            let token_ids: Vec<&[u32]> = encodings.iter().map(|e| e.get_ids()).collect();
            let attention_masks: Vec<&[u32]> =
                encodings.iter().map(|e| e.get_attention_mask()).collect();

            let token_ids_tensor = Tensor::new(token_ids, &self.device)
                .map_err(|e| Error::Embedding(format!("token tensor: {e}")))?;
            let attention_mask_tensor = Tensor::new(attention_masks, &self.device)
                .map_err(|e| Error::Embedding(format!("mask tensor: {e}")))?;
            let token_type_ids = token_ids_tensor
                .zeros_like()
                .map_err(|e| Error::Embedding(format!("type ids: {e}")))?;

            let output = self
                .model
                .forward(
                    &token_ids_tensor,
                    &token_type_ids,
                    Some(&attention_mask_tensor),
                )
                .map_err(|e| Error::Embedding(format!("forward: {e}")))?;

            let pooled = Self::mean_pool_normalize(&output, &attention_mask_tensor)?;

            let vecs = pooled
                .to_vec2::<f32>()
                .map_err(|e| Error::Embedding(format!("to_vec2: {e}")))?;
            all_vecs.extend(vecs);
        }

        Ok(all_vecs)
    }

    fn dimension(&self) -> Result<usize, Error> {
        Ok(self.dimension)
    }
}
