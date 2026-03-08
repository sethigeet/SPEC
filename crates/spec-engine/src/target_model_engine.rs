use candle_core::IndexOp;
use log::info;
use pyo3::prelude::*;
use spec_decode::{PagedLlama, Sampler, SamplerConfig};

/// Full-size target model engine without speculative decoding.
///
/// This is primarily intended for end-to-end correctness tests that compare
/// speculative engines against the target model's direct output.
#[pyclass]
pub struct TargetModelEngine {
    model: PagedLlama,
    sampler: Sampler,
    tokenizer: tokenizers::Tokenizer,
}

#[pymethods]
impl TargetModelEngine {
    /// Create a new non-speculative target-only engine.
    #[new]
    #[pyo3(signature = (
        model_id,
        temperature = 0.0,
        top_p = None,
        top_k = None,
        seed = 42,
        repeat_penalty = 1.0,
        repeat_last_n = 64,
        revision = "main",
    ))]
    fn new(
        model_id: &str,
        temperature: f64,
        top_p: Option<f64>,
        top_k: Option<usize>,
        seed: u64,
        repeat_penalty: f32,
        repeat_last_n: usize,
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
            "loading target-only model '{}' on {:?} ({:?})",
            model_id, device, dtype
        );
        let model = PagedLlama::from_hub(model_id, revision, &device, dtype)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("model: {e}")))?;

        let sampler_cfg = SamplerConfig {
            temperature,
            top_p,
            top_k,
            seed,
            repeat_penalty,
            repeat_last_n,
        };
        let sampler = Sampler::new(&sampler_cfg);

        let tokenizer = spec_decode::models::load_tokenizer(model_id, revision)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("tokenizer: {e}")))?;

        Ok(Self {
            model,
            sampler,
            tokenizer,
        })
    }

    /// Generate text from a prompt using the target model alone.
    #[pyo3(signature = (prompt, max_tokens = 100))]
    fn generate(&mut self, prompt: &str, max_tokens: usize) -> PyResult<String> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("encode: {e}")))?;
        let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

        let all_tokens = self.generate_from_prompt_tokens(prompt_tokens, max_tokens)?;

        self.tokenizer
            .decode(&all_tokens, true)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("decode: {e}")))
    }

    /// Generate token IDs from a prompt using the target model alone.
    #[pyo3(signature = (prompt_tokens, max_tokens = 100))]
    fn generate_tokens(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
    ) -> PyResult<Vec<u32>> {
        self.generate_from_prompt_tokens(prompt_tokens, max_tokens)
    }
}

impl TargetModelEngine {
    fn generate_from_prompt_tokens(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
    ) -> PyResult<Vec<u32>> {
        if prompt_tokens.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "prompt must tokenize to at least one token",
            ));
        }

        self.model.reset_cache();

        let mut tokens = prompt_tokens;
        let mut logits = self
            .model
            .forward(&tokens, 0)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("generate: {e}")))?;

        let initial_len = tokens.len();
        info!(
            "starting target-only generation: prompt_len={}, max_new_tokens={}",
            initial_len, max_tokens
        );

        while tokens.len() - initial_len < max_tokens {
            let next_logits = logits
                .i(logits.dim(0).unwrap_or(1).saturating_sub(1))
                .and_then(|t| t.contiguous())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("logits: {e}")))?;

            let next_token = self
                .sampler
                .sample(&next_logits, &tokens)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("sample: {e}")))?;
            tokens.push(next_token);

            if self.model.is_eos(next_token) {
                break;
            }

            logits = self
                .model
                .forward(&[next_token], 0)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("generate: {e}")))?;
        }

        let max_len = initial_len + max_tokens;
        if tokens.len() > max_len {
            tokens.truncate(max_len);
        }

        Ok(tokens)
    }
}
