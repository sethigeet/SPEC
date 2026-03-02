//! Model wrapper for candle-transformers Llama.
//!
//! Provides [`CandleLlama`] which encapsulates model loading from the
//! HuggingFace Hub and single-forward-pass inference.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama as llama_model;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

/// Configuration resolved at load time.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub config: llama_model::Config,
    pub eos_token_id: Option<llama_model::LlamaEosToks>,
    pub device: Device,
    pub dtype: DType,
}

/// Wrapper around `candle_transformers::models::llama::Llama`.
pub struct CandleLlama {
    model: llama_model::Llama,
    pub cache: llama_model::Cache,
    pub cfg: ModelConfig,
    /// Current position in the sequence (number of tokens processed so far).
    pub index_pos: usize,
}

impl CandleLlama {
    /// Load a Llama-family model from HuggingFace Hub.
    ///
    /// `model_id` — e.g. `"HuggingFaceTB/SmolLM2-135M"` or `"meta-llama/Llama-3.2-1B"`.
    /// `revision`  — branch/tag, typically `"main"`.
    pub fn from_hub(model_id: &str, revision: &str, device: &Device, dtype: DType) -> Result<Self> {
        let api = Api::new().context("failed to create HF Hub API")?;
        let repo = api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            revision.to_string(),
        ));

        // Config
        let config_path = repo.get("config.json").context("config.json not found")?;
        let raw = std::fs::read(&config_path)?;
        let llama_config: llama_model::LlamaConfig = serde_json::from_slice(&raw)?;
        let eos_token_id = llama_config.eos_token_id.clone();
        let config = llama_config.into_config(false); // no flash-attn on CPU

        // Weights — try single file first, fall back to sharded index
        let filenames = {
            let single = repo.get("model.safetensors");
            match single {
                Ok(path) => vec![path],
                Err(_) => {
                    // sharded
                    let index_path = repo
                        .get("model.safetensors.index.json")
                        .context("neither model.safetensors nor index found")?;
                    let index_raw = std::fs::read(&index_path)?;
                    let index: serde_json::Value = serde_json::from_slice(&index_raw)?;
                    let weight_map = index["weight_map"]
                        .as_object()
                        .context("invalid index format: no weight_map")?;
                    let mut files: Vec<String> = weight_map
                        .values()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect();
                    files.sort();
                    files.dedup();
                    files
                        .into_iter()
                        .map(|f| repo.get(&f).with_context(|| format!("failed to get {f}")))
                        .collect::<Result<Vec<_>>>()?
                }
            }
        };

        let cache = llama_model::Cache::new(true, dtype, &config, device)?;

        // SAFETY: memory-mapped safetensors — candle's standard pattern
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, device)? };
        let model = llama_model::Llama::load(vb, &config)?;

        let cfg = ModelConfig {
            config,
            eos_token_id,
            device: device.clone(),
            dtype,
        };

        Ok(Self {
            model,
            cache,
            cfg,
            index_pos: 0,
        })
    }

    /// Run a forward pass over `token_ids`, returning logits for the **last**
    /// token only. Shape: `(vocab_size,)`.
    ///
    /// Automatically advances `index_pos` by `token_ids.len()`.
    pub fn forward(&mut self, token_ids: &[u32]) -> Result<Tensor> {
        let dev = &self.cfg.device;
        let pos = self.index_pos;
        let input = Tensor::new(token_ids, dev)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, pos, &mut self.cache)?;
        self.index_pos += token_ids.len();
        Ok(logits.squeeze(0)?)
    }

    /// Run `n` individual forward passes, one token at a time, collecting
    /// logits at each position. Returns a `Vec<Tensor>` where each element has
    /// shape `(vocab_size,)`.
    ///
    /// This is the verification strategy for the target model: we feed the
    /// draft tokens one at a time so the KV cache accumulates correctly, and
    /// we get the target probability distribution at each position.
    pub fn forward_each(&mut self, token_ids: &[u32]) -> Result<Vec<Tensor>> {
        let mut all_logits = Vec::with_capacity(token_ids.len());
        for &tok in token_ids {
            let logits = self.forward(&[tok])?;
            all_logits.push(logits);
        }
        Ok(all_logits)
    }

    /// Reset the KV cache and position counter. Typically called after a
    /// rollback or when starting a fresh sequence.
    pub fn reset_cache(&mut self) -> Result<()> {
        self.cache = llama_model::Cache::new(
            true,
            self.cfg.dtype,
            &self.cfg.config,
            &self.cfg.device,
        )?;
        self.index_pos = 0;
        Ok(())
    }

    /// Truncate the KV cache to `new_pos` tokens. Resets the cache and
    /// re-processes `tokens[..new_pos]` to rebuild it.
    ///
    /// This is necessary after a partial rejection during speculative decoding.
    pub fn truncate_cache_to(&mut self, all_tokens: &[u32], new_pos: usize) -> Result<()> {
        self.reset_cache()?;
        if new_pos > 0 {
            // Re-process the prefix to rebuild the KV cache.
            // We do this in a single forward call (ignoring the logits).
            let _logits = self.forward(&all_tokens[..new_pos])?;
        }
        Ok(())
    }

    /// Check if a token is an EOS token.
    pub fn is_eos(&self, token: u32) -> bool {
        match &self.cfg.eos_token_id {
            Some(llama_model::LlamaEosToks::Single(id)) => token == *id,
            Some(llama_model::LlamaEosToks::Multiple(ids)) => ids.contains(&token),
            None => false,
        }
    }
}

/// Load a tokenizer from HuggingFace Hub.
pub fn load_tokenizer(model_id: &str, revision: &str) -> Result<Tokenizer> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));
    let tokenizer_path = repo
        .get("tokenizer.json")
        .context("tokenizer.json not found")?;
    Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))
}
