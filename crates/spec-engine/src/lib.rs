use pyo3::prelude::*;

use spec_core::{DraftQueue, DraftToken, EngineState, KVBlockAllocator};
use spec_decode::{CandleLlama, Sampler, SamplerConfig, SpecDecoder};

/// Asynchronous speculative decoding engine.
///
/// Coordinates a draft model (producer) and a target model (consumer) via a
/// lock-free SPSC queue, epoch-based rollback state machine, and KV cache
/// block allocator.
///
/// # Usage from Python
///
/// ```python
/// from spec_engine import AsyncSpecEngine
///
/// engine = AsyncSpecEngine(queue_capacity=64, max_kv_blocks=128)
///
/// # Draft model loop (producer)
/// ok = engine.push_draft_token(token_id=42, kv_ptr=7)
/// if not ok:
///     # Rollback occurred — reset draft model, then:
///     engine.acknowledge_flush()
///
/// # Target model loop (consumer)
/// batch = engine.pull_draft_batch(max_k=8)
/// # ... verify batch with target model ...
/// if rejected:
///     engine.trigger_rollback(corrected_token=99)
/// ```
#[pyclass]
pub struct AsyncSpecEngine {
    queue: Box<DraftQueue>,
    state: Box<EngineState>,
    allocator: Box<KVBlockAllocator>,
}

#[pymethods]
impl AsyncSpecEngine {
    /// Creates a new engine.
    ///
    /// Args:
    ///     queue_capacity: Size of the draft token queue (must be a power of 2).
    ///     max_kv_blocks: Number of KV cache block slots to manage.
    #[new]
    fn new(queue_capacity: usize, max_kv_blocks: usize) -> Self {
        Self {
            queue: Box::new(DraftQueue::new(queue_capacity)),
            state: Box::new(EngineState::new()),
            allocator: Box::new(KVBlockAllocator::new(max_kv_blocks)),
        }
    }

    /// Push a draft token onto the queue.
    ///
    /// Returns `True` on success. Returns `False` if:
    /// - A rollback has occurred and the producer must flush (call
    ///   `acknowledge_flush()` after resetting draft model state).
    /// - The queue is full (back-pressure).
    ///
    /// Args:
    ///     token_id: The drafted token ID.
    ///     kv_ptr: Opaque KV cache pointer/index from the draft model.
    fn push_draft_token(&self, token_id: i64, kv_ptr: usize) -> bool {
        let _ = kv_ptr; // reserved for future use (e.g. storing draft model VRAM address)
        // Check rollback first
        if self.state.needs_flush() {
            return false;
        }

        let epoch = self.state.current_epoch();
        let kv_block_idx = match self.allocator.alloc(epoch) {
            Some(idx) => idx,
            None => return false, // No KV blocks available
        };

        let token = DraftToken {
            token_id,
            kv_block_idx,
        };

        if !self.queue.push(token) {
            // Queue full — free the block we just allocated
            self.allocator.free(kv_block_idx);
            return false;
        }

        true
    }

    /// Pull a batch of draft tokens for verification by the target model.
    ///
    /// Returns a list of `(token_id, kv_block_idx)` tuples, up to `max_k`.
    /// May return an empty list if no tokens are available.
    ///
    /// Args:
    ///     max_k: Maximum number of tokens to pull.
    fn pull_draft_batch(&self, max_k: usize) -> Vec<(i64, usize)> {
        self.queue
            .pop_batch(max_k)
            .into_iter()
            .map(|t| (t.token_id, t.kv_block_idx))
            .collect()
    }

    /// Trigger a rollback when the target model rejects a drafted token.
    ///
    /// This flushes the queue, increments the epoch, and frees all KV blocks
    /// from the dead epoch. The producer will see `push_draft_token` return
    /// `False` until it calls `acknowledge_flush()`.
    ///
    /// Args:
    ///     corrected_token: The correct token that the target model produced.
    fn trigger_rollback(&self, corrected_token: i64) {
        let old_epoch = self.state.current_epoch();
        self.state.trigger_rollback(corrected_token, &self.queue);
        self.allocator.rollback(old_epoch);
    }

    /// Acknowledge a rollback from the producer side.
    ///
    /// Call this after the draft model has reset its local state. This clears
    /// the flush flag so `push_draft_token` will succeed again.
    fn acknowledge_flush(&self) {
        self.state.acknowledge_flush();
    }

    /// Returns the current number of tokens in the queue.
    fn queue_len(&self) -> usize {
        self.queue.len()
    }

    /// Returns the number of free KV cache blocks.
    fn available_kv_blocks(&self) -> usize {
        self.allocator.available()
    }
}

// ─── Speculative Decoding Engine (candle-powered) ────────────────────────────

/// Full speculative decoding engine using candle-transformers.
///
/// Loads a draft and target Llama-family model from HuggingFace Hub, then
/// generates tokens using the speculative decoding algorithm.
///
/// # Usage from Python
///
/// ```python
/// from spec_engine import SpecDecodingEngine
///
/// engine = SpecDecodingEngine(
///     draft_model_id="HuggingFaceTB/SmolLM2-135M",
///     target_model_id="HuggingFaceTB/SmolLM2-360M",
///     gamma=5,
///     temperature=0.0,
///     seed=42,
/// )
///
/// output = engine.generate(prompt="My favorite theorem is ", max_tokens=100)
/// print(output)
/// ```
#[pyclass]
struct SpecDecodingEngine {
    decoder: SpecDecoder,
    tokenizer: tokenizers::Tokenizer,
}

#[pymethods]
impl SpecDecodingEngine {
    /// Create a new speculative decoding engine.
    ///
    /// Args:
    ///     draft_model_id: HuggingFace model ID for the small/draft model.
    ///     target_model_id: HuggingFace model ID for the large/target model.
    ///     gamma: Number of draft tokens per speculative step (default: 5).
    ///     temperature: Sampling temperature, 0.0 = greedy (default: 0.0).
    ///     top_p: Top-p / nucleus sampling (optional).
    ///     top_k: Top-k sampling (optional).
    ///     seed: Random seed (default: 42).
    ///     repeat_penalty: Penalty for repeated tokens (default: 1.0 = off).
    ///     repeat_last_n: Window for repeat penalty (default: 64).
    ///     revision: HF Hub revision (default: "main").
    #[new]
    #[pyo3(signature = (
        draft_model_id,
        target_model_id,
        gamma = 5,
        temperature = 0.0,
        top_p = None,
        top_k = None,
        seed = 42,
        repeat_penalty = 1.0,
        repeat_last_n = 64,
        revision = "main",
    ))]
    fn new(
        draft_model_id: &str,
        target_model_id: &str,
        gamma: usize,
        temperature: f64,
        top_p: Option<f64>,
        top_k: Option<usize>,
        seed: u64,
        repeat_penalty: f32,
        repeat_last_n: usize,
        revision: &str,
    ) -> PyResult<Self> {
        let device = candle_core::Device::Cpu;
        let dtype = candle_core::DType::F32;

        let draft = CandleLlama::from_hub(draft_model_id, revision, &device, dtype)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("draft model: {e}")))?;

        let target = CandleLlama::from_hub(target_model_id, revision, &device, dtype)
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("target model: {e}"))
            })?;

        let sampler_cfg = SamplerConfig {
            temperature,
            top_p,
            top_k,
            seed,
            repeat_penalty,
            repeat_last_n,
        };
        let sampler = Sampler::new(&sampler_cfg);

        let tokenizer = spec_decode::model::load_tokenizer(target_model_id, revision)
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("tokenizer: {e}"))
            })?;

        let decoder = SpecDecoder::new(draft, target, sampler, gamma, seed);

        Ok(Self { decoder, tokenizer })
    }

    /// Generate text from a prompt using speculative decoding.
    ///
    /// Args:
    ///     prompt: The input text prompt.
    ///     max_tokens: Maximum number of new tokens to generate.
    ///
    /// Returns:
    ///     The generated text (prompt + completion).
    #[pyo3(signature = (prompt, max_tokens = 100))]
    fn generate(&mut self, prompt: &str, max_tokens: usize) -> PyResult<String> {
        // Reset caches for a fresh generation
        self.decoder.draft.reset_cache();
        self.decoder.target.reset_cache();

        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("encode: {e}")))?;
        let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

        let all_tokens = self
            .decoder
            .generate(prompt_tokens, max_tokens)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("generate: {e}")))?;

        let text = self
            .tokenizer
            .decode(&all_tokens, true)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("decode: {e}")))?;

        Ok(text)
    }

    /// Generate token IDs from a prompt (no text decoding).
    ///
    /// Args:
    ///     prompt_tokens: List of input token IDs.
    ///     max_tokens: Maximum number of new tokens to generate.
    ///
    /// Returns:
    ///     List of all token IDs (prompt + generated).
    #[pyo3(signature = (prompt_tokens, max_tokens = 100))]
    fn generate_tokens(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
    ) -> PyResult<Vec<u32>> {
        self.decoder.draft.reset_cache();
        self.decoder.target.reset_cache();

        let all_tokens = self
            .decoder
            .generate(prompt_tokens, max_tokens)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("generate: {e}")))?;

        Ok(all_tokens)
    }
}

/// Python module definition.
#[pymodule]
fn spec_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AsyncSpecEngine>()?;
    m.add_class::<SpecDecodingEngine>()?;
    Ok(())
}

