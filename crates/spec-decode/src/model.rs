//! Forked Llama model using paged KV cache.
//!
//! This is a fork of `candle_transformers::models::llama` that replaces the
//! default `Cache` with [`PagedKVCache`] from `spec-core`. The attention,
//! block, and model structs are kept as close to candle's originals as
//! possible, with only the cache access points changed.

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{
    embedding, linear, linear_no_bias, Embedding, Linear, Module, RmsNorm, VarBuilder,
};
use candle_transformers::models::llama as llama_model;
use hf_hub::{api::sync::Api, Repo, RepoType};
use spec_core::paged_kv_cache::{PagedCacheConfig, PagedKVCache, RopeScaling};
use tokenizers::Tokenizer;

// ─── Forked Internals ────────────────────────────────────────────────────────

fn repeat_kv(x: Tensor, n_rep: usize) -> candle_core::Result<Tensor> {
    candle_transformers::utils::repeat_kv(x, n_rep)
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> candle_core::Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

/// Some Llama-family checkpoints omit projection biases (e.g. SmolLM2).
/// Try loading a biased linear first, then fall back to bias-free tensors.
fn load_linear_with_optional_bias(
    in_features: usize,
    out_features: usize,
    vb: VarBuilder,
) -> candle_core::Result<Linear> {
    // Some checkpoints (e.g. SmolLM2) do not include `*.bias`.
    // So we fall back to a bias-free projection.
    let result = linear(in_features, out_features, vb.clone());
    if result.is_ok() {
        return result;
    }

    linear_no_bias(in_features, out_features, vb.clone())
}

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> candle_core::Result<Tensor> {
    candle_flash_attn_v3::flash_attn(
        q,
        k,
        v,
        softmax_scale,
        causal,
        /* use_gqa_packing */ false,
    )
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> candle_core::Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

/// Forked causal self-attention that uses [`PagedKVCache`].
struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    use_flash_attn: bool,
    max_position_embeddings: usize,
}

impl CausalSelfAttention {
    fn apply_rotary_emb(
        &self,
        x: &Tensor,
        index_pos: usize,
        cache: &PagedKVCache,
    ) -> candle_core::Result<Tensor> {
        let (_b_sz, _, seq_len, _hidden_size) = x.dims4()?;
        let (cos, sin) = cache.cos_sin(index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut PagedKVCache,
        epoch: usize,
    ) -> candle_core::Result<Tensor> {
        let (b_sz, seq_len, hidden_size) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.apply_rotary_emb(&q, index_pos, cache)?;
        let k = self.apply_rotary_emb(&k, index_pos, cache)?;

        // ── Paged KV cache: append and retrieve ──────────────────────
        let (k, v) = cache.append_and_get(block_idx, k, v, epoch)?;

        let k_seq_len = k.dim(2)?;
        let (k, v) = if k_seq_len > self.max_position_embeddings {
            let k = k
                .narrow(
                    2,
                    k_seq_len - self.max_position_embeddings,
                    self.max_position_embeddings,
                )?
                .contiguous()?;
            let v_seq_len = v.dim(2)?;
            let v = if v_seq_len > self.max_position_embeddings {
                v.narrow(
                    2,
                    v_seq_len - self.max_position_embeddings,
                    self.max_position_embeddings,
                )?
                .contiguous()?
            } else {
                v.contiguous()?
            };
            (k, v)
        } else {
            (k.contiguous()?, v.contiguous()?)
        };

        let y = if self.use_flash_attn {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            // It handles GQA natively, so no need for repeat_kv.
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            flash_attn(&q, &k, &v, softmax_scale, /* causal */ seq_len > 1)?.transpose(1, 2)?
        } else {
            let k = repeat_kv(k, self.num_attention_heads / self.num_key_value_heads)?;
            let v = repeat_kv(v, self.num_attention_heads / self.num_key_value_heads)?;

            let in_dtype = q.dtype();
            let q = q.to_dtype(DType::F32)?;
            let k = k.to_dtype(DType::F32)?;
            let v = v.to_dtype(DType::F32)?;
            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = if seq_len == 1 {
                att
            } else {
                let mask = cache.mask(seq_len)?.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, f32::NEG_INFINITY)?
            };

            let att = candle_nn::ops::softmax_last_dim(&att)?;
            att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?
        };
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;
        let y = self.o_proj.forward(&y)?;
        Ok(y)
    }

    fn load(vb: VarBuilder, cfg: &llama_model::Config) -> candle_core::Result<Self> {
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let q_proj = load_linear_with_optional_bias(size_in, size_q, vb.pp("q_proj"))?;
        let k_proj = load_linear_with_optional_bias(size_in, size_kv, vb.pp("k_proj"))?;
        let v_proj = load_linear_with_optional_bias(size_in, size_kv, vb.pp("v_proj"))?;
        let o_proj = load_linear_with_optional_bias(size_q, size_in, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            use_flash_attn: cfg.use_flash_attn,
            max_position_embeddings: cfg.max_position_embeddings,
        })
    }
}

struct Mlp {
    c_fc1: Linear,
    c_fc2: Linear,
    c_proj: Linear,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = (candle_nn::ops::silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x)
    }

    fn load(vb: VarBuilder, cfg: &llama_model::Config) -> candle_core::Result<Self> {
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let c_fc1 = load_linear_with_optional_bias(h_size, i_size, vb.pp("gate_proj"))?;
        let c_fc2 = load_linear_with_optional_bias(h_size, i_size, vb.pp("up_proj"))?;
        let c_proj = load_linear_with_optional_bias(i_size, h_size, vb.pp("down_proj"))?;
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
        })
    }
}

struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut PagedKVCache,
        epoch: usize,
    ) -> candle_core::Result<Tensor> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self.attn.forward(&x, index_pos, block_idx, cache, epoch)? + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }

    fn load(vb: VarBuilder, cfg: &llama_model::Config) -> candle_core::Result<Self> {
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg)?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg)?;
        let rms_1 =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
        })
    }
}

/// Forked Llama model using [`PagedKVCache`].
struct PagedLlama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
}

impl PagedLlama {
    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        cache: &mut PagedKVCache,
        epoch: usize,
    ) -> candle_core::Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mut x = self.wte.forward(x)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache, epoch)?;
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }

    fn load(vb: VarBuilder, cfg: &llama_model::Config) -> candle_core::Result<Self> {
        let wte = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(wte.embeddings().clone(), None)
        } else {
            linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        let ln_f = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(vb.pp(format!("model.layers.{i}")), cfg))
            .collect::<candle_core::Result<_>>()?;
        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
        })
    }
}

// ─── Public Wrapper ──────────────────────────────────────────────────────────

/// Configuration resolved at load time.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub config: llama_model::Config,
    pub eos_token_id: Option<llama_model::LlamaEosToks>,
    pub device: Device,
    pub dtype: DType,
}

/// Default number of KV blocks for the paged cache when no explicit value is
/// given. This should comfortably fit small-to-medium sequences on CPU.
const DEFAULT_MAX_KV_BLOCKS: usize = 4096;

/// Wrapper around the forked Llama model with paged KV cache.
pub struct CandleLlama {
    model: PagedLlama,
    pub cache: PagedKVCache,
    pub cfg: ModelConfig,
}

/// Convert a candle Llama config into `PagedCacheConfig`.
fn to_paged_cache_config(cfg: &llama_model::Config) -> PagedCacheConfig {
    let rope_scaling = cfg.rope_scaling.as_ref().map(|rs| RopeScaling {
        factor: rs.factor,
        low_freq_factor: rs.low_freq_factor,
        high_freq_factor: rs.high_freq_factor,
        original_max_position_embeddings: rs.original_max_position_embeddings,
    });
    PagedCacheConfig {
        num_hidden_layers: cfg.num_hidden_layers,
        num_attention_heads: cfg.num_attention_heads,
        hidden_size: cfg.hidden_size,
        rope_theta: cfg.rope_theta,
        max_position_embeddings: cfg.max_position_embeddings,
        rope_scaling,
    }
}

impl CandleLlama {
    /// Load a Llama-family model from HuggingFace Hub.
    ///
    /// `model_id` — e.g. `"HuggingFaceTB/SmolLM2-135M"` or `"meta-llama/Llama-3.2-1B"`.
    /// `revision`  — branch/tag, typically `"main"`.
    pub fn from_hub(model_id: &str, revision: &str, device: &Device, dtype: DType) -> Result<Self> {
        Self::from_hub_with_blocks(model_id, revision, device, dtype, DEFAULT_MAX_KV_BLOCKS)
    }

    /// Load with an explicit max KV block count.
    pub fn from_hub_with_blocks(
        model_id: &str,
        revision: &str,
        device: &Device,
        dtype: DType,
        max_kv_blocks: usize,
    ) -> Result<Self> {
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
        let use_flash_attn = cfg!(feature = "flash-attn");
        let config = llama_config.into_config(use_flash_attn);

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

        let paged_cfg = to_paged_cache_config(&config);
        let cache = PagedKVCache::new(max_kv_blocks, &paged_cfg, device, dtype)
            .map_err(|e| anyhow::anyhow!("failed to create paged cache: {e}"))?;

        // SAFETY: memory-mapped safetensors — candle's standard pattern
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, device)? };
        let model = PagedLlama::load(vb, &config)
            .map_err(|e| anyhow::anyhow!("failed to load model: {e}"))?;

        let cfg = ModelConfig {
            config,
            eos_token_id,
            device: device.clone(),
            dtype,
        };

        Ok(Self { model, cache, cfg })
    }

    /// Run a forward pass over `token_ids`, returning logits for the **last**
    /// token only. Shape: `(vocab_size,)`.
    ///
    /// `epoch` tags the KV cache entries for rollback support.
    pub fn forward(&mut self, token_ids: &[u32], epoch: usize) -> Result<Tensor> {
        let dev = &self.cfg.device;
        let pos = self.cache.seq_len();
        let input = Tensor::new(token_ids, dev)?.unsqueeze(0)?;
        let logits = self
            .model
            .forward(&input, pos, &mut self.cache, epoch)
            .map_err(|e| anyhow::anyhow!("forward failed: {e}"))?;
        Ok(logits.squeeze(0)?)
    }

    /// Run `n` individual forward passes, one token at a time, collecting
    /// logits at each position. Returns a `Vec<Tensor>` where each element has
    /// shape `(vocab_size,)`.
    ///
    /// This is the verification strategy for the target model: we feed the
    /// draft tokens one at a time so the KV cache accumulates correctly, and
    /// we get the target probability distribution at each position.
    pub fn forward_each(&mut self, token_ids: &[u32], epoch: usize) -> Result<Vec<Tensor>> {
        let mut all_logits = Vec::with_capacity(token_ids.len());
        for &tok in token_ids {
            let logits = self.forward(&[tok], epoch)?;
            all_logits.push(logits);
        }
        Ok(all_logits)
    }

    /// Reset the KV cache. Typically called when starting a fresh sequence.
    pub fn reset_cache(&mut self) {
        self.cache.reset();
    }

    /// Truncate the KV cache to `new_len` tokens.
    ///
    /// This is much faster than the old approach of resetting and replaying,
    /// because the paged cache can simply free blocks beyond `new_len`.
    pub fn truncate_cache_to(&mut self, new_len: usize) {
        self.cache.truncate_to(new_len);
    }

    /// Rollback all KV cache entries from `dead_epoch`.
    pub fn rollback_cache(&mut self, dead_epoch: usize) {
        self.cache.rollback(dead_epoch);
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
