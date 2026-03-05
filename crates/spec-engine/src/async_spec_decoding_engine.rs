use log::info;
use pyo3::prelude::*;
use spec_decode::{AsyncSpecPipeline, CandleLlama, Sampler, SamplerConfig};

/// Async continuous speculative decoding engine.
///
/// Like [`SpecDecodingEngine`], but runs the draft and target models on
/// **separate threads** for concurrent execution. The draft model continuously
/// generates speculative tokens via a lock-free SPSC queue while the target
/// model verifies them.
///
/// Acceptance statistics are logged automatically — set the ``SPEC_LOG``
/// environment variable to control verbosity (e.g. ``SPEC_LOG=info``).
///
/// # Usage from Python
///
/// ```python
/// from spec_engine import AsyncSpecDecodingEngine
///
/// engine = AsyncSpecDecodingEngine(
///     draft_model_id="HuggingFaceTB/SmolLM2-135M",
///     target_model_id="HuggingFaceTB/SmolLM2-360M",
///     gamma=5,
/// )
///
/// output = engine.generate("The meaning of life is", max_tokens=50)
/// print(output)
/// ```
#[pyclass]
pub struct AsyncSpecDecodingEngine {
    pipeline: AsyncSpecPipeline,
    tokenizer: tokenizers::Tokenizer,
}

#[pymethods]
impl AsyncSpecDecodingEngine {
    /// Create a new async speculative decoding engine.
    ///
    /// Args:
    ///     draft_model_id: HuggingFace model ID for the small/draft model.
    ///     target_model_id: HuggingFace model ID for the large/target model.
    ///     gamma: Number of draft tokens per verification batch (default: 5).
    ///     seed: Random seed (default: 42).
    ///     revision: HF Hub revision (default: "main").
    #[new]
    #[pyo3(signature = (
        draft_model_id,
        target_model_id,
        gamma = 5,
        seed = 42,
        revision = "main",
    ))]
    fn new(
        draft_model_id: &str,
        target_model_id: &str,
        gamma: usize,
        seed: u64,
        revision: &str,
    ) -> PyResult<Self> {
        let device = if cfg!(feature = "cuda") {
            candle_core::Device::cuda_if_available(0)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA init: {e}")))?
        } else {
            candle_core::Device::Cpu
        };
        let dtype = if device.is_cuda() {
            candle_core::DType::BF16
        } else {
            candle_core::DType::F32
        };

        info!(
            "loading draft model '{}' on {:?} ({:?})",
            draft_model_id, device, dtype
        );
        let draft = CandleLlama::from_hub(draft_model_id, revision, &device, dtype)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("draft model: {e}")))?;

        info!(
            "loading target model '{}' on {:?} ({:?})",
            target_model_id, device, dtype
        );
        let target = CandleLlama::from_hub(target_model_id, revision, &device, dtype)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("target model: {e}")))?;

        let sampler_cfg = SamplerConfig {
            temperature: 0.0,
            top_p: None,
            top_k: None,
            seed,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
        };
        let sampler = Sampler::new(&sampler_cfg);

        let tokenizer = spec_decode::model::load_tokenizer(target_model_id, revision)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("tokenizer: {e}")))?;

        let pipeline = AsyncSpecPipeline::new(draft, target, sampler, gamma, seed);

        info!(
            "AsyncSpecDecodingEngine ready: gamma={}, seed={}",
            gamma, seed
        );

        Ok(Self {
            pipeline,
            tokenizer,
        })
    }

    /// Generate text from a prompt using async speculative decoding.
    ///
    /// Args:
    ///     prompt: The input text prompt.
    ///     max_tokens: Maximum number of new tokens to generate.
    ///
    /// Returns:
    ///     The generated text (prompt + completion).
    #[pyo3(signature = (prompt, max_tokens = 100))]
    fn generate(&mut self, prompt: &str, max_tokens: usize) -> PyResult<String> {
        self.pipeline.reset_caches();

        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("encode: {e}")))?;
        let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

        let all_tokens = self
            .pipeline
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
        self.pipeline.reset_caches();

        let all_tokens = self
            .pipeline
            .generate(prompt_tokens, max_tokens)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("generate: {e}")))?;

        Ok(all_tokens)
    }
}
