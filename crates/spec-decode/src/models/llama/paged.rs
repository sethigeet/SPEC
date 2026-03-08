//! Forked Llama model using paged KV cache.
//!
//! This is a fork of `candle_transformers::models::llama` that replaces the
//! default `Cache` with [`PagedKVCache`] from `spec-core`.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{
    embedding, linear, linear_no_bias, Embedding, Linear, Module, RmsNorm, VarBuilder,
};
use candle_transformers::{models::llama as llama_model, utils::repeat_kv};
use hf_hub::{api::sync::Api, Repo, RepoType};
use spec_core::paged_kv_cache::{PagedCacheConfig, PagedKVCache, RopeScaling};

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> candle_core::Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

fn load_linear_with_optional_bias(
    in_features: usize,
    out_features: usize,
    vb: VarBuilder,
) -> candle_core::Result<Linear> {
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
    candle_flash_attn_v3::flash_attn(q, k, v, softmax_scale, causal, false)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> candle_core::Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

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
        let (_, _, seq_len, _) = x.dims4()?;
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
            .transpose(1, 2)?
            .contiguous()?;

        let q = self.apply_rotary_emb(&q, index_pos, cache)?;
        let k = self.apply_rotary_emb(&k, index_pos, cache)?;
        let (k, v) = cache.append_and_get(block_idx, k, v, epoch)?;

        let k_seq_len = k.dim(2)?;
        let (k, v) = if k_seq_len > self.max_position_embeddings {
            (
                k.narrow(
                    2,
                    k_seq_len - self.max_position_embeddings,
                    self.max_position_embeddings,
                )?
                .contiguous()?,
                v.narrow(
                    2,
                    k_seq_len - self.max_position_embeddings,
                    self.max_position_embeddings,
                )?
                .contiguous()?,
            )
        } else {
            (k.contiguous()?, v.contiguous()?)
        };

        let y = if self.use_flash_attn {
            let q = q.transpose(1, 2)?.contiguous()?;
            let k = k.transpose(1, 2)?.contiguous()?;
            let v = v.transpose(1, 2)?.contiguous()?;
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            flash_attn(&q, &k, &v, softmax_scale, seq_len > 1)?.transpose(1, 2)?
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
        self.o_proj.forward(&y)
    }

    fn load(vb: VarBuilder, cfg: &llama_model::Config) -> candle_core::Result<Self> {
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        Ok(Self {
            q_proj: load_linear_with_optional_bias(size_in, size_q, vb.pp("q_proj"))?,
            k_proj: load_linear_with_optional_bias(size_in, size_kv, vb.pp("k_proj"))?,
            v_proj: load_linear_with_optional_bias(size_in, size_kv, vb.pp("v_proj"))?,
            o_proj: load_linear_with_optional_bias(size_q, size_in, vb.pp("o_proj"))?,
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
        Ok(Self {
            c_fc1: load_linear_with_optional_bias(h_size, i_size, vb.pp("gate_proj"))?,
            c_fc2: load_linear_with_optional_bias(h_size, i_size, vb.pp("up_proj"))?,
            c_proj: load_linear_with_optional_bias(i_size, h_size, vb.pp("down_proj"))?,
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
        self.mlp.forward(&self.rms_2.forward(&x)?)? + residual
    }

    fn load(vb: VarBuilder, cfg: &llama_model::Config) -> candle_core::Result<Self> {
        Ok(Self {
            rms_1: candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            attn: CausalSelfAttention::load(vb.pp("self_attn"), cfg)?,
            rms_2: candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: Mlp::load(vb.pp("mlp"), cfg)?,
        })
    }
}

struct Model {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
}

impl Model {
    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        cache: &mut PagedKVCache,
        epoch: usize,
    ) -> candle_core::Result<Tensor> {
        let (_b_sz, _seq_len) = x.dims2()?;
        let mut x = self.wte.forward(x)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache, epoch)?;
        }
        let x = self.ln_f.forward(&x)?;
        self.lm_head.forward(&x)?.to_dtype(DType::F32)
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

#[derive(Debug, Clone)]
pub struct PagedLlamaConfig {
    pub config: llama_model::Config,
    pub eos_token_id: Option<llama_model::LlamaEosToks>,
    pub device: Device,
    pub dtype: DType,
}

const DEFAULT_MAX_KV_BLOCKS: usize = 4096;

pub struct PagedLlama {
    model: Model,
    pub cache: PagedKVCache,
    pub cfg: PagedLlamaConfig,
}

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

impl PagedLlama {
    pub fn from_hub(model_id: &str, revision: &str, device: &Device, dtype: DType) -> Result<Self> {
        Self::from_hub_with_blocks(model_id, revision, device, dtype, DEFAULT_MAX_KV_BLOCKS)
    }

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
        let config_path = repo.get("config.json").context("config.json not found")?;
        let raw = std::fs::read(&config_path)?;
        let llama_config: llama_model::LlamaConfig = serde_json::from_slice(&raw)?;
        let eos_token_id = llama_config.eos_token_id.clone();
        let config = llama_config.into_config(cfg!(feature = "flash-attn"));

        let filenames = {
            let single = repo.get("model.safetensors");
            match single {
                Ok(path) => vec![path],
                Err(_) => {
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
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, device)? };
        let model =
            Model::load(vb, &config).map_err(|e| anyhow::anyhow!("failed to load model: {e}"))?;

        Ok(Self {
            model,
            cache,
            cfg: PagedLlamaConfig {
                config,
                eos_token_id,
                device: device.clone(),
                dtype,
            },
        })
    }

    pub fn forward(&mut self, token_ids: &[u32], epoch: usize) -> Result<Tensor> {
        let input = Tensor::new(token_ids, &self.cfg.device)?.unsqueeze(0)?;
        let pos = self.cache.seq_len();
        let logits = self
            .model
            .forward(&input, pos, &mut self.cache, epoch)
            .map_err(|e| anyhow::anyhow!("forward failed: {e}"))?;
        Ok(logits.squeeze(0)?)
    }

    pub fn reset_cache(&mut self) {
        self.cache.reset();
    }

    pub fn truncate_cache_to(&mut self, new_len: usize) {
        self.cache.truncate_to(new_len);
    }

    pub fn rollback_cache(&mut self, dead_epoch: usize) {
        self.cache.rollback(dead_epoch);
    }

    pub fn is_eos(&self, token: u32) -> bool {
        match &self.cfg.eos_token_id {
            Some(llama_model::LlamaEosToks::Single(id)) => token == *id,
            Some(llama_model::LlamaEosToks::Multiple(ids)) => ids.contains(&token),
            None => false,
        }
    }
}
