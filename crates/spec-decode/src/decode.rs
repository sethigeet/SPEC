//! Core speculative decoding loop.
//!
//! Implements the algorithm from "Fast Inference from Transformers via
//! Speculative Decoding" (Leviathan et al., 2023).
//!
//! After generation, [`SpecStats`] provides acceptance metrics:
//! `acceptance_rate()` returns the fraction of draft tokens accepted by the
//! target model.
//!
//! # Algorithm
//!
//! ```text
//! repeat:
//!   1. Draft γ tokens with the small model.
//!   2. Score them all with the target model.
//!   3. Accept/reject via rejection sampling.
//!   4. On rejection, correct and roll back caches.
//!   5. If all accepted, sample a bonus token from the last target logits.
//! ```

use anyhow::Result;
use candle_core::Tensor;
use log::{debug, info, trace};
use rand::Rng;

use crate::model::CandleLlama;
use crate::sampler::Sampler;

/// Speculative decoder coordinating a draft and target model.
pub struct SpecDecoder {
    pub draft: CandleLlama,
    pub target: CandleLlama,
    pub sampler: Sampler,
    /// Number of draft tokens to generate per speculative step.
    pub gamma: usize,
    /// RNG for rejection sampling (separate from the sampler RNG).
    rng: rand::rngs::StdRng,
    /// Current epoch for KV cache tagging (incremented on rollback).
    epoch: usize,
}

impl SpecDecoder {
    /// Create a new speculative decoder.
    pub fn new(
        draft: CandleLlama,
        target: CandleLlama,
        sampler: Sampler,
        gamma: usize,
        seed: u64,
    ) -> Self {
        use rand::SeedableRng;
        Self {
            draft,
            target,
            sampler,
            gamma,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            epoch: 0,
        }
    }

    /// Process the prompt through both models to fill their KV caches.
    fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<()> {
        debug!("prefilling {} prompt tokens", prompt_tokens.len());
        let _draft_logits = self.draft.forward(prompt_tokens, self.epoch)?;
        let _target_logits = self.target.forward(prompt_tokens, self.epoch)?;
        debug!("prefill complete");
        Ok(())
    }

    /// Run one speculative decoding step.
    ///
    /// `tokens` is the full sequence generated so far (for repeat-penalty and
    /// cache rollback). Returns the newly accepted tokens to append.
    fn step(&mut self, tokens: &mut Vec<u32>) -> Result<StepResult> {
        // ── 1. Draft phase ───────────────────────────────────────────
        let mut draft_tokens = Vec::with_capacity(self.gamma);
        let mut draft_probs_at_pos = Vec::with_capacity(self.gamma);

        // The last token in `tokens` is the most recently accepted token.
        // The draft model's KV cache already contains everything up to that
        // point. We feed the last accepted token to get logits for the next
        // position, then auto-regressively draft γ tokens.
        let last_token = *tokens.last().unwrap();

        // First draft step: feed the last accepted token.
        let draft_logits = self.draft.forward(&[last_token], self.epoch)?;
        let draft_p = Sampler::logits_to_probs(&draft_logits)?;
        let draft_tok = self.sampler.sample(&draft_logits, tokens)?;
        draft_tokens.push(draft_tok);
        draft_probs_at_pos.push(draft_p);

        // Remaining γ-1 draft steps
        for _ in 1..self.gamma {
            let prev = *draft_tokens.last().unwrap();
            let draft_logits = self.draft.forward(&[prev], self.epoch)?;
            let draft_p = Sampler::logits_to_probs(&draft_logits)?;
            let draft_tok = self.sampler.sample(&draft_logits, tokens)?;
            draft_tokens.push(draft_tok);
            draft_probs_at_pos.push(draft_p);
        }

        // ── 2. Verify phase ─────────────────────────────────────────
        // Feed the last accepted token + all draft tokens through the target
        // model one at a time, collecting logits at each position.
        let verify_input: Vec<u32> = std::iter::once(last_token)
            .chain(draft_tokens.iter().copied())
            .collect();
        let target_logits_list = self.target.forward_each(&verify_input, self.epoch)?;

        // ── 3. Rejection sampling ────────────────────────────────────
        let mut draft_accepted = 0;

        for i in 0..self.gamma {
            let target_p = Sampler::logits_to_probs(&target_logits_list[i])?;
            let draft_p = &draft_probs_at_pos[i];
            let drafted = draft_tokens[i] as usize;

            let p_target = target_p[drafted];
            let p_draft = draft_p[drafted];

            // Accept with probability min(1, p_target / p_draft)
            let accept_prob = if p_draft > 0.0 {
                (p_target / p_draft).min(1.0)
            } else {
                // Draft assigned zero probability — always reject
                0.0
            };

            let r: f32 = self.rng.random();
            if r < accept_prob {
                // Accept this draft token
                tokens.push(draft_tokens[i]);
                draft_accepted += 1;
            } else {
                // Reject — sample correction token from
                // norm(max(0, p_target - p_draft))
                let correction_token = self.sample_correction(&target_p, draft_p)?;
                tokens.push(correction_token);

                // Rollback both model caches to the current position.
                // With paged cache, this is O(1) — just free blocks beyond
                // the accepted position.
                let new_len = tokens.len();
                self.draft.truncate_cache_to(new_len);
                self.target.truncate_cache_to(new_len);

                return Ok(StepResult {
                    draft_accepted,
                    draft_proposed: self.gamma,
                    hit_eos: self.draft.is_eos(correction_token)
                        || self.target.is_eos(correction_token),
                });
            }
        }

        // ── 4. All accepted — bonus token ────────────────────────────
        // Sample from the last target logits (position γ).
        debug!(
            "all {} draft tokens accepted, sampling bonus token",
            self.gamma
        );
        let bonus_logits = &target_logits_list[self.gamma];
        let bonus_token = self.sampler.sample(bonus_logits, tokens)?;
        tokens.push(bonus_token);

        // The draft model cache produced γ extra draft tokens that weren't
        // all verified. Truncate its cache to match the accepted length.
        let new_len = tokens.len();
        self.draft.truncate_cache_to(new_len);

        Ok(StepResult {
            draft_accepted,
            draft_proposed: self.gamma,
            hit_eos: self.draft.is_eos(bonus_token) || self.target.is_eos(bonus_token),
        })
    }

    /// Sample a correction token from the adjusted distribution:
    /// `norm(max(0, p_target - p_draft))`
    fn sample_correction(&mut self, target_p: &[f32], draft_p: &[f32]) -> Result<u32> {
        let mut adjusted: Vec<f32> = target_p
            .iter()
            .zip(draft_p.iter())
            .map(|(&t, &d)| (t - d).max(0.0))
            .collect();

        let sum: f32 = adjusted.iter().sum();
        if sum <= 0.0 {
            // Fallback: sample from target distribution directly
            let logits = Tensor::new(target_p, &self.target.cfg.device)?;
            return self.sampler.sample(&logits, &[]);
        }

        // Normalize
        for p in adjusted.iter_mut() {
            *p /= sum;
        }

        // Sample from the adjusted distribution
        let r: f32 = self.rng.random();
        let mut cumsum = 0.0_f32;
        for (i, &p) in adjusted.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return Ok(i as u32);
            }
        }

        // Floating-point edge case — return last token
        Ok((adjusted.len() - 1) as u32)
    }

    /// Generate tokens from a prompt using speculative decoding.
    ///
    /// Returns the full sequence (prompt + generated tokens).
    /// Acceptance statistics are logged at `info` level.
    pub fn generate(&mut self, prompt_tokens: Vec<u32>, max_new_tokens: usize) -> Result<Vec<u32>> {
        let mut tokens = prompt_tokens;

        // Prefill both models with the prompt
        self.prefill(&tokens)?;

        let initial_len = tokens.len();
        let mut total_generated = 0;
        let mut stats = SpecStats::new();

        info!(
            "starting speculative decoding: prompt_len={}, max_new_tokens={}, gamma={}",
            initial_len, max_new_tokens, self.gamma
        );

        while total_generated < max_new_tokens {
            let result = self.step(&mut tokens)?;
            total_generated = tokens.len() - initial_len;

            stats.num_steps += 1;
            stats.draft_proposed += result.draft_proposed;
            stats.draft_accepted += result.draft_accepted;

            trace!(
                "step {}: accepted {}/{} draft tokens, total_generated={}",
                stats.num_steps,
                result.draft_accepted,
                result.draft_proposed,
                total_generated
            );

            if result.hit_eos {
                debug!("EOS token encountered, stopping generation");
                break;
            }
        }

        // Trim to exact max_new_tokens if we overshot
        let max_len = initial_len + max_new_tokens;
        if tokens.len() > max_len {
            tokens.truncate(max_len);
        }

        stats.total_tokens = tokens.len();

        info!(
            "generation complete: steps={}, draft_proposed={}, draft_accepted={}, acceptance_rate={:.1}%, avg_accepted_per_step={:.2}, total_tokens={}",
            stats.num_steps,
            stats.draft_proposed,
            stats.draft_accepted,
            stats.acceptance_rate() * 100.0,
            stats.avg_accepted_per_step(),
            stats.total_tokens,
        );

        Ok(tokens)
    }
}

/// Result of a single speculative step.
struct StepResult {
    /// Number of draft tokens that were accepted by the target model.
    draft_accepted: usize,
    /// Number of draft tokens proposed in this step.
    draft_proposed: usize,
    /// Whether an EOS token was generated.
    hit_eos: bool,
}

/// Statistics from a speculative decoding run.
#[derive(Debug, Clone)]
pub struct SpecStats {
    /// Total number of draft tokens proposed across all steps.
    pub draft_proposed: usize,
    /// Number of draft tokens accepted by the target model.
    pub draft_accepted: usize,
    /// Total number of speculative decoding steps.
    pub num_steps: usize,
    /// Total tokens generated (including prompt).
    pub total_tokens: usize,
}

impl SpecStats {
    /// Create empty stats.
    pub fn new() -> Self {
        Self {
            draft_proposed: 0,
            draft_accepted: 0,
            num_steps: 0,
            total_tokens: 0,
        }
    }

    /// Draft token acceptance rate as a fraction in [0.0, 1.0].
    ///
    /// Returns 0.0 if no tokens were proposed.
    pub fn acceptance_rate(&self) -> f64 {
        if self.draft_proposed == 0 {
            0.0
        } else {
            self.draft_accepted as f64 / self.draft_proposed as f64
        }
    }

    /// Average number of draft tokens accepted per step.
    pub fn avg_accepted_per_step(&self) -> f64 {
        if self.num_steps == 0 {
            0.0
        } else {
            self.draft_accepted as f64 / self.num_steps as f64
        }
    }
}

impl Default for SpecStats {
    fn default() -> Self {
        Self::new()
    }
}
