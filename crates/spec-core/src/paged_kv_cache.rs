//! Paged KV cache backed by [`KVBlockAllocator`].
//!
//! Each token gets its own block. Blocks are allocated from the allocator and
//! tagged with an epoch so that rollbacks can bulk-free speculative entries.
//! The actual K/V tensors are stored in a flat `Vec` indexed by block ID.
//!
//! This module also owns the precomputed RoPE (cos/sin) tensors and the causal
//! mask cache, which were previously part of candle's `Cache`.

use candle_core::{DType, Device, Result, Tensor};

use crate::kv_metadata::KVBlockAllocator;

/// Configuration subset needed by the paged cache (RoPE + layer count).
///
/// Kept minimal so that callers can construct it from any model config.
#[derive(Debug, Clone)]
pub struct PagedCacheConfig {
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub hidden_size: usize,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    /// If set, applies Llama-3-style scaled RoPE.
    pub rope_scaling: Option<RopeScaling>,
}

/// Llama-3 RoPE scaling parameters.
#[derive(Debug, Clone)]
pub struct RopeScaling {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: usize,
}

/// Paged KV cache with epoch-aware allocation and rollback.
///
/// Stores actual K/V tensors in a flat array indexed by block IDs from
/// [`KVBlockAllocator`]. Each layer maintains an ordered list of block IDs
/// that form its key/value sequence.
pub struct PagedKVCache {
    allocator: KVBlockAllocator,
    /// Flat tensor store: `store[block_id] = Some((k, v))`.
    store: Vec<Option<(Tensor, Tensor)>>,
    /// Per-layer ordered list of `(block_id, epoch)`.
    layer_blocks: Vec<Vec<(usize, usize)>>,
    /// Precomputed RoPE cosine values, shape `(max_position_embeddings, head_dim/2)`.
    cos: Tensor,
    /// Precomputed RoPE sine values, shape `(max_position_embeddings, head_dim/2)`.
    sin: Tensor,
    /// Device for mask creation.
    device: Device,
    /// Number of layers.
    num_layers: usize,
}

fn calculate_default_inv_freq(hidden_size: usize, num_heads: usize, rope_theta: f32) -> Vec<f32> {
    let head_dim = hidden_size / num_heads;
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

impl PagedKVCache {
    /// Create a new paged KV cache.
    ///
    /// `max_blocks` — total number of KV blocks available for allocation.
    pub fn new(
        max_blocks: usize,
        cfg: &PagedCacheConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let num_layers = cfg.num_hidden_layers;

        // ── RoPE precomputation (matches candle's Cache::new) ─────────
        let theta = match &cfg.rope_scaling {
            None => calculate_default_inv_freq(cfg.hidden_size, cfg.num_attention_heads, cfg.rope_theta),
            Some(scaling) => {
                let low_freq_wavelen =
                    scaling.original_max_position_embeddings as f32 / scaling.low_freq_factor;
                let high_freq_wavelen =
                    scaling.original_max_position_embeddings as f32 / scaling.high_freq_factor;

                calculate_default_inv_freq(cfg.hidden_size, cfg.num_attention_heads, cfg.rope_theta)
                    .into_iter()
                    .map(|freq| {
                        let wavelen = 2.0 * std::f32::consts::PI / freq;
                        if wavelen < high_freq_wavelen {
                            freq
                        } else if wavelen > low_freq_wavelen {
                            freq / scaling.factor
                        } else {
                            let smooth = (scaling.original_max_position_embeddings as f32
                                / wavelen
                                - scaling.low_freq_factor)
                                / (scaling.high_freq_factor - scaling.low_freq_factor);
                            (1.0 - smooth) * freq / scaling.factor + smooth * freq
                        }
                    })
                    .collect::<Vec<_>>()
            }
        };

        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0, cfg.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;

        Ok(Self {
            allocator: KVBlockAllocator::new(max_blocks),
            store: (0..max_blocks).map(|_| None).collect(),
            layer_blocks: vec![Vec::new(); num_layers],
            cos,
            sin,
            device: device.clone(),
            num_layers,
        })
    }

    /// Append K/V tensors for `layer`, allocating one block per token tagged with `epoch`.
    ///
    /// Returns the concatenated (K, V) for the full sequence at this layer
    /// (including the newly appended entries). Returns `Err` if allocation fails.
    pub fn append_and_get(
        &mut self,
        layer: usize,
        k: Tensor,
        v: Tensor,
        epoch: usize,
    ) -> Result<(Tensor, Tensor)> {
        let k_seq_len = k.dim(2)?;
        let v_seq_len = v.dim(2)?;
        if k_seq_len != v_seq_len {
            return Err(candle_core::Error::Msg(
                "PagedKVCache: K/V sequence lengths differ".into(),
            ));
        }

        for token_idx in 0..k_seq_len {
            let block_id = self
                .allocator
                .alloc(epoch)
                .ok_or_else(|| candle_core::Error::Msg("PagedKVCache: out of blocks".into()))?;

            let k_tok = k.narrow(2, token_idx, 1)?.contiguous()?;
            let v_tok = v.narrow(2, token_idx, 1)?.contiguous()?;

            self.store[block_id] = Some((k_tok, v_tok));
            self.layer_blocks[layer].push((block_id, epoch));
        }

        self.get_kv(layer)
    }

    /// Get the concatenated (K, V) tensors for a layer's full sequence.
    pub fn get_kv(&self, layer: usize) -> Result<(Tensor, Tensor)> {
        let blocks = &self.layer_blocks[layer];
        if blocks.is_empty() {
            return Err(candle_core::Error::Msg(
                "PagedKVCache: no blocks for layer".into(),
            ));
        }

        let mut ks = Vec::with_capacity(blocks.len());
        let mut vs = Vec::with_capacity(blocks.len());
        for &(block_id, _) in blocks {
            let (k, v) = self.store[block_id]
                .as_ref()
                .ok_or_else(|| candle_core::Error::Msg("PagedKVCache: missing block tensor".into()))?;
            ks.push(k.clone());
            vs.push(v.clone());
        }

        let full_k = Tensor::cat(&ks, 2)?;
        let full_v = Tensor::cat(&vs, 2)?;
        Ok((full_k, full_v))
    }

    /// Rollback: free all blocks from `dead_epoch` across all layers.
    pub fn rollback(&mut self, dead_epoch: usize) {
        for layer_blocks in &mut self.layer_blocks {
            layer_blocks.retain(|&(block_id, epoch)| {
                if epoch == dead_epoch {
                    self.store[block_id] = None;
                    false
                } else {
                    true
                }
            });
        }
        self.allocator.rollback(dead_epoch);
    }

    /// Reset: free all blocks and clear all layer sequences.
    pub fn reset(&mut self) {
        for layer_blocks in &mut self.layer_blocks {
            for &(block_id, _) in layer_blocks.iter() {
                self.store[block_id] = None;
            }
            layer_blocks.clear();
        }
        // Re-create allocator to reset free stack
        let max_blocks = self.allocator.max_blocks();
        self.allocator = KVBlockAllocator::new(max_blocks);
    }

    /// Truncate all layers to at most `new_len` tokens.
    ///
    /// Blocks beyond `new_len` are freed back to the allocator.
    pub fn truncate_to(&mut self, new_len: usize) {
        for layer_blocks in &mut self.layer_blocks {
            while layer_blocks.len() > new_len {
                let (block_id, _epoch) = layer_blocks.pop().unwrap();
                self.store[block_id] = None;
                self.allocator.free(block_id);
            }
        }
    }

    /// Returns the current sequence length (number of cached tokens).
    pub fn seq_len(&self) -> usize {
        // All layers should have the same length; use layer 0.
        if self.num_layers == 0 {
            return 0;
        }
        self.layer_blocks[0].len()
    }

    /// RoPE: narrow cos/sin to `[index_pos .. index_pos + seq_len]`.
    pub fn cos_sin(&self, index_pos: usize, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        Ok((cos, sin))
    }

    /// Causal mask of size `(query_len, key_len)` for attention scores `QK^T`.
    ///
    /// `query_start` and `key_start` are absolute token positions of the first
    /// query and key rows. A mask value of `1` means "masked out".
    pub fn mask(
        &mut self,
        query_len: usize,
        key_len: usize,
        query_start: usize,
        key_start: usize,
    ) -> Result<Tensor> {
        let mask: Vec<u8> = (0..query_len)
            .flat_map(|i| {
                (0..key_len).map(move |j| {
                    let query_pos = query_start + i;
                    let key_pos = key_start + j;
                    u8::from(key_pos > query_pos)
                })
            })
            .collect();
        Tensor::from_slice(&mask, (query_len, key_len), &self.device)
    }

    /// Returns the number of free blocks available for allocation.
    pub fn available_blocks(&self) -> usize {
        self.allocator.available()
    }

    /// Returns the maximum number of blocks.
    pub fn max_blocks(&self) -> usize {
        self.allocator.max_blocks()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn test_config() -> PagedCacheConfig {
        PagedCacheConfig {
            num_hidden_layers: 2,
            num_attention_heads: 4,
            hidden_size: 64,
            rope_theta: 10000.0,
            max_position_embeddings: 128,
            rope_scaling: None,
        }
    }

    /// Helper: create a K or V tensor with shape (1, num_kv_heads, 1, head_dim).
    fn dummy_kv(num_kv_heads: usize, head_dim: usize, val: f32) -> Tensor {
        Tensor::full(val, (1, num_kv_heads, 1, head_dim), &Device::Cpu).unwrap()
    }

    #[test]
    fn append_and_get_kv() -> Result<()> {
        let cfg = test_config();
        let head_dim = cfg.hidden_size / cfg.num_attention_heads; // 16
        let mut cache = PagedKVCache::new(32, &cfg, &Device::Cpu, DType::F32)?;

        let k1 = dummy_kv(4, head_dim, 1.0);
        let v1 = dummy_kv(4, head_dim, 2.0);
        let (full_k, full_v) = cache.append_and_get(0, k1, v1, 0)?;

        // After 1 append: sequence length in dim 2 should be 1
        assert_eq!(full_k.dims(), &[1, 4, 1, head_dim]);
        assert_eq!(full_v.dims(), &[1, 4, 1, head_dim]);
        assert_eq!(cache.seq_len(), 1);

        // Append another
        let k2 = dummy_kv(4, head_dim, 3.0);
        let v2 = dummy_kv(4, head_dim, 4.0);
        let (full_k, full_v) = cache.append_and_get(0, k2, v2, 0)?;

        assert_eq!(full_k.dims(), &[1, 4, 2, head_dim]);
        assert_eq!(full_v.dims(), &[1, 4, 2, head_dim]);
        assert_eq!(cache.seq_len(), 2);
        Ok(())
    }

    #[test]
    fn append_multi_token_input_tracks_token_length() -> Result<()> {
        let cfg = test_config();
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let mut cache = PagedKVCache::new(32, &cfg, &Device::Cpu, DType::F32)?;

        let k = Tensor::full(1.0, (1, 4, 3, head_dim), &Device::Cpu)?;
        let v = Tensor::full(2.0, (1, 4, 3, head_dim), &Device::Cpu)?;
        let (full_k, full_v) = cache.append_and_get(0, k, v, 0)?;

        assert_eq!(full_k.dims(), &[1, 4, 3, head_dim]);
        assert_eq!(full_v.dims(), &[1, 4, 3, head_dim]);
        assert_eq!(cache.seq_len(), 3);
        assert_eq!(cache.available_blocks(), 32 - 3);
        Ok(())
    }

    #[test]
    fn rollback_clears_dead_epoch() -> Result<()> {
        let cfg = test_config();
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let mut cache = PagedKVCache::new(32, &cfg, &Device::Cpu, DType::F32)?;

        // Epoch 0: 2 tokens
        for _ in 0..2 {
            let k = dummy_kv(4, head_dim, 1.0);
            let v = dummy_kv(4, head_dim, 1.0);
            cache.append_and_get(0, k, v, 0)?;
        }
        // Also layer 1
        for _ in 0..2 {
            let k = dummy_kv(4, head_dim, 1.0);
            let v = dummy_kv(4, head_dim, 1.0);
            cache.append_and_get(1, k, v, 0)?;
        }

        // Epoch 1: 3 tokens
        for _ in 0..3 {
            let k = dummy_kv(4, head_dim, 2.0);
            let v = dummy_kv(4, head_dim, 2.0);
            cache.append_and_get(0, k, v, 1)?;
        }
        for _ in 0..3 {
            let k = dummy_kv(4, head_dim, 2.0);
            let v = dummy_kv(4, head_dim, 2.0);
            cache.append_and_get(1, k, v, 1)?;
        }

        assert_eq!(cache.layer_blocks[0].len(), 5);
        assert_eq!(cache.available_blocks(), 32 - 10); // 10 blocks used (5 per layer × 2 layers)

        // Rollback epoch 1
        cache.rollback(1);
        assert_eq!(cache.layer_blocks[0].len(), 2);
        assert_eq!(cache.layer_blocks[1].len(), 2);
        assert_eq!(cache.available_blocks(), 32 - 4); // 4 blocks remain (2 per layer × 2 layers)

        Ok(())
    }

    #[test]
    fn reset_clears_all() -> Result<()> {
        let cfg = test_config();
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let mut cache = PagedKVCache::new(16, &cfg, &Device::Cpu, DType::F32)?;

        for _ in 0..5 {
            let k = dummy_kv(4, head_dim, 1.0);
            let v = dummy_kv(4, head_dim, 1.0);
            cache.append_and_get(0, k, v, 0)?;
        }

        cache.reset();
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.available_blocks(), 16);
        Ok(())
    }

    #[test]
    fn truncate_to() -> Result<()> {
        let cfg = test_config();
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let mut cache = PagedKVCache::new(32, &cfg, &Device::Cpu, DType::F32)?;

        for i in 0..5 {
            let k = dummy_kv(4, head_dim, i as f32);
            let v = dummy_kv(4, head_dim, i as f32);
            cache.append_and_get(0, k, v, 0)?;
        }
        for i in 0..5 {
            let k = dummy_kv(4, head_dim, i as f32);
            let v = dummy_kv(4, head_dim, i as f32);
            cache.append_and_get(1, k, v, 0)?;
        }

        assert_eq!(cache.seq_len(), 5);
        cache.truncate_to(3);
        assert_eq!(cache.seq_len(), 3);
        assert_eq!(cache.layer_blocks[0].len(), 3);
        assert_eq!(cache.layer_blocks[1].len(), 3);
        // 4 blocks freed (2 per layer)
        assert_eq!(cache.available_blocks(), 32 - 6);
        Ok(())
    }

    #[test]
    fn cos_sin_lookup() -> Result<()> {
        let cfg = test_config();
        let cache = PagedKVCache::new(8, &cfg, &Device::Cpu, DType::F32)?;

        let (cos, sin) = cache.cos_sin(0, 4)?;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        assert_eq!(cos.dims(), &[4, head_dim / 2]);
        assert_eq!(sin.dims(), &[4, head_dim / 2]);
        Ok(())
    }

    #[test]
    fn mask_shape() -> Result<()> {
        let cfg = test_config();
        let mut cache = PagedKVCache::new(8, &cfg, &Device::Cpu, DType::F32)?;

        let mask = cache.mask(5, 5, 0, 0)?;
        assert_eq!(mask.dims(), &[5, 5]);

        // Upper triangular: (0,0) should be 0, (0,1) should be 1
        let vals = mask.to_vec2::<u8>()?;
        assert_eq!(vals[0][0], 0);
        assert_eq!(vals[0][1], 1);
        assert_eq!(vals[1][1], 0);
        Ok(())
    }

    #[test]
    fn mask_shape_matches_cached_attention_scores() -> Result<()> {
        let cfg = test_config();
        let mut cache = PagedKVCache::new(8, &cfg, &Device::Cpu, DType::F32)?;

        let mask = cache.mask(2, 5, 3, 0)?;
        assert_eq!(mask.dims(), &[2, 5]);

        let vals = mask.to_vec2::<u8>()?;
        assert_eq!(vals[0], vec![0, 0, 0, 0, 1]);
        assert_eq!(vals[1], vec![0, 0, 0, 0, 0]);
        Ok(())
    }
}
