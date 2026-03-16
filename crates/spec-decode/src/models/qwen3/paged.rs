//! Forked dense Qwen3 model using paged KV cache.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{
    embedding, linear_b, linear_no_bias, Embedding, Linear, Module, RmsNorm, VarBuilder,
};
use candle_transformers::{models::qwen3 as qwen3_model, utils::repeat_kv};
use spec_core::paged_kv_cache::{PagedCacheConfig, PagedKVCache};

use crate::models::utils::{load_model_files, parse_eos_token_ids};

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

struct Qwen3RotaryEmbedding;

impl Qwen3RotaryEmbedding {
    fn apply(
        &self,
        q: &Tensor,
        k: &Tensor,
        offset: usize,
        cache: &PagedKVCache,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let (cos, sin) = cache.cos_sin(offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

struct Qwen3Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: candle_nn::Activation,
}

impl Qwen3Mlp {
    fn load(vb: VarBuilder, cfg: &qwen3_model::Config) -> candle_core::Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
            act_fn: cfg.hidden_act,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let lhs = x.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = x.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

struct Qwen3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    use_flash_attn: bool,
    rotary_emb: Qwen3RotaryEmbedding,
}

impl Qwen3Attention {
    fn load(vb: VarBuilder, cfg: &qwen3_model::Config) -> candle_core::Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        Ok(Self {
            q_proj: linear_b(
                cfg.hidden_size,
                num_heads * head_dim,
                cfg.attention_bias,
                vb.pp("q_proj"),
            )?,
            k_proj: linear_b(
                cfg.hidden_size,
                num_kv_heads * head_dim,
                cfg.attention_bias,
                vb.pp("k_proj"),
            )?,
            v_proj: linear_b(
                cfg.hidden_size,
                num_kv_heads * head_dim,
                cfg.attention_bias,
                vb.pp("v_proj"),
            )?,
            o_proj: linear_b(
                num_heads * head_dim,
                cfg.hidden_size,
                cfg.attention_bias,
                vb.pp("o_proj"),
            )?,
            q_norm: candle_nn::rms_norm(cfg.head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: candle_nn::rms_norm(cfg.head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            hidden_size: head_dim * num_heads,
            use_flash_attn: cfg!(feature = "flash-attn"),
            rotary_emb: Qwen3RotaryEmbedding,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut PagedKVCache,
        epoch: usize,
    ) -> candle_core::Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q = self.q_norm.forward(&q_flat)?.reshape((
            b_sz,
            self.num_heads,
            seq_len,
            self.head_dim,
        ))?;
        let k = self.k_norm.forward(&k_flat)?.reshape((
            b_sz,
            self.num_kv_heads,
            seq_len,
            self.head_dim,
        ))?;

        let (q, k) = self.rotary_emb.apply(&q, &k, index_pos, cache)?;
        let (k, v) = cache.append_and_get(block_idx, k, v, epoch)?;
        let ctx = if self.use_flash_attn {
            let q = q.transpose(1, 2)?.contiguous()?;
            let k = k.transpose(1, 2)?.contiguous()?;
            let v = v.transpose(1, 2)?.contiguous()?;
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            flash_attn(&q, &k, &v, softmax_scale, seq_len > 1)?.transpose(1, 2)?
        } else {
            let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
            let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;
            let mut attn =
                (q.matmul(&k.transpose(2, 3)?)? * (1.0 / (self.head_dim as f64).sqrt()))?;
            if seq_len > 1 {
                let mask = cache
                    .mask(seq_len, k.dim(2)?, index_pos, 0)?
                    .broadcast_as(attn.shape())?;
                let neg_inf = Tensor::full(f32::NEG_INFINITY, attn.shape(), attn.device())?;
                attn = mask.where_cond(&neg_inf, &attn)?;
            }

            let probs = candle_nn::ops::softmax_last_dim(&attn)?;
            probs.matmul(&v)?
        };

        ctx.transpose(1, 2)?
            .reshape((b_sz, seq_len, self.hidden_size))?
            .apply(&self.o_proj)
    }
}

struct DecoderLayer {
    self_attn: Qwen3Attention,
    mlp: Qwen3Mlp,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl DecoderLayer {
    fn load(vb: VarBuilder, cfg: &qwen3_model::Config) -> candle_core::Result<Self> {
        Ok(Self {
            self_attn: Qwen3Attention::load(vb.pp("self_attn"), cfg)?,
            mlp: Qwen3Mlp::load(vb.pp("mlp"), cfg)?,
            ln1: candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            ln2: candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut PagedKVCache,
        epoch: usize,
    ) -> candle_core::Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let x = (x + self
            .self_attn
            .forward(&h, index_pos, block_idx, cache, epoch)?)?;
        let h2 = self.ln2.forward(&x)?;
        x + self.mlp.forward(&h2)?
    }
}

struct Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
}

impl Model {
    fn load(vb: VarBuilder, cfg: &qwen3_model::Config) -> candle_core::Result<Self> {
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| DecoderLayer::load(vb.pp(format!("model.layers.{i}")), cfg))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    fn forward(
        &self,
        input: &Tensor,
        index_pos: usize,
        cache: &mut PagedKVCache,
        epoch: usize,
    ) -> candle_core::Result<Tensor> {
        let mut h = self.embed_tokens.forward(input)?;
        for (block_idx, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, index_pos, block_idx, cache, epoch)?;
        }
        let h = self.norm.forward(&h)?;
        self.lm_head.forward(&h)?.to_dtype(DType::F32)
    }
}

#[derive(Debug, Clone)]
pub struct PagedQwen3Config {
    pub config: qwen3_model::Config,
    pub eos_token_ids: Vec<u32>,
    pub device: Device,
    pub dtype: DType,
}

const DEFAULT_MAX_KV_BLOCKS: usize = 4096;

pub struct PagedQwen3 {
    model: Model,
    pub cache: PagedKVCache,
    pub cfg: PagedQwen3Config,
}

fn to_paged_cache_config(cfg: &qwen3_model::Config) -> PagedCacheConfig {
    PagedCacheConfig {
        num_hidden_layers: cfg.num_hidden_layers,
        num_attention_heads: cfg.num_attention_heads,
        hidden_size: cfg.hidden_size,
        head_dim: cfg.head_dim,
        rope_theta: cfg.rope_theta as f32,
        max_position_embeddings: cfg.max_position_embeddings,
        rope_scaling: None,
    }
}

impl PagedQwen3 {
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
        let files = load_model_files(model_id, revision)?;
        let config: qwen3_model::Config = serde_json::from_slice(&files.raw_config)?;
        if config.use_sliding_window {
            anyhow::bail!("Qwen3 sliding-window attention is not supported");
        }
        let eos_token_ids = parse_eos_token_ids(&files.raw_config)?;

        let cache = PagedKVCache::new(
            max_kv_blocks,
            &to_paged_cache_config(&config),
            device,
            dtype,
        )
        .map_err(|e| anyhow::anyhow!("failed to create paged cache: {e}"))?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files.filenames, dtype, device)? };
        let model =
            Model::load(vb, &config).map_err(|e| anyhow::anyhow!("failed to load model: {e}"))?;

        Ok(Self {
            model,
            cache,
            cfg: PagedQwen3Config {
                config,
                eos_token_ids,
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
            .map_err(|e| anyhow::anyhow!("forward failed: {e:?}"))?;
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
        self.cfg.eos_token_ids.contains(&token)
    }
}
