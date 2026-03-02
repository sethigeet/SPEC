//! Token sampling utilities wrapping candle's `LogitsProcessor`.

use anyhow::Result;
use candle_core::Tensor;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::utils::apply_repeat_penalty;

/// Configuration for the sampler.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub seed: u64,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0, // greedy by default
            top_p: None,
            top_k: None,
            seed: 42,
            repeat_penalty: 1.0, // no penalty
            repeat_last_n: 64,
        }
    }
}

/// Sampler wrapping candle's `LogitsProcessor` with repeat-penalty support.
pub struct Sampler {
    processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl Sampler {
    /// Create a new sampler from config.
    pub fn new(cfg: &SamplerConfig) -> Self {
        let sampling = if cfg.temperature <= 1e-7 {
            Sampling::ArgMax
        } else {
            match (cfg.top_k, cfg.top_p) {
                (None, None) => Sampling::All {
                    temperature: cfg.temperature,
                },
                (Some(k), None) => Sampling::TopK {
                    k,
                    temperature: cfg.temperature,
                },
                (None, Some(p)) => Sampling::TopP {
                    p,
                    temperature: cfg.temperature,
                },
                (Some(k), Some(p)) => Sampling::TopKThenTopP {
                    k,
                    p,
                    temperature: cfg.temperature,
                },
            }
        };
        let processor = LogitsProcessor::from_sampling(cfg.seed, sampling);

        Self {
            processor,
            repeat_penalty: cfg.repeat_penalty,
            repeat_last_n: cfg.repeat_last_n,
        }
    }

    /// Sample a single token from `logits` (shape `(vocab_size,)`), applying
    /// repeat penalty against `token_history`.
    pub fn sample(&mut self, logits: &Tensor, token_history: &[u32]) -> Result<u32> {
        let logits = if self.repeat_penalty != 1.0 && !token_history.is_empty() {
            let start_at = token_history.len().saturating_sub(self.repeat_last_n);
            apply_repeat_penalty(logits, self.repeat_penalty, &token_history[start_at..])?
        } else {
            logits.clone()
        };
        Ok(self.processor.sample(&logits)?)
    }

    /// Convert logits to a probability distribution (softmax). Returns a
    /// `Vec<f32>` of length `vocab_size`.
    pub fn logits_to_probs(logits: &Tensor) -> Result<Vec<f32>> {
        let probs = candle_nn::ops::softmax(logits, candle_core::D::Minus1)?;
        Ok(probs.to_vec1::<f32>()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_argmax_sampling() -> Result<()> {
        let cfg = SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(&cfg);

        // Token 3 has the highest logit
        let logits = Tensor::new(&[1.0_f32, 2.0, 0.5, 10.0, 0.1], &Device::Cpu)?;
        let token = sampler.sample(&logits, &[])?;
        assert_eq!(token, 3);
        Ok(())
    }

    #[test]
    fn test_logits_to_probs_sums_to_one() -> Result<()> {
        let logits = Tensor::new(&[1.0_f32, 2.0, 3.0], &Device::Cpu)?;
        let probs = Sampler::logits_to_probs(&logits)?;
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "probs should sum to 1, got {sum}");
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);
        Ok(())
    }
}
