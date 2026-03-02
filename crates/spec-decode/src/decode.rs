//! Core speculative decoding loop.
//!
//! Implements the algorithm from "Fast Inference from Transformers via
//! Speculative Decoding" (Leviathan et al., 2023).
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
        }
    }

    /// Process the prompt through both models to fill their KV caches.
    /// Returns the logits from the target model for the last prompt token.
    fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<()> {
        // Process entire prompt through both models.
        let _draft_logits = self.draft.forward(prompt_tokens)?;
        let _target_logits = self.target.forward(prompt_tokens)?;
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
        let draft_logits = self.draft.forward(&[last_token])?;
        let draft_p = Sampler::logits_to_probs(&draft_logits)?;
        let draft_tok = self.sampler.sample(&draft_logits, tokens)?;
        draft_tokens.push(draft_tok);
        draft_probs_at_pos.push(draft_p);

        // Remaining γ-1 draft steps
        for _ in 1..self.gamma {
            let prev = *draft_tokens.last().unwrap();
            let draft_logits = self.draft.forward(&[prev])?;
            let draft_p = Sampler::logits_to_probs(&draft_logits)?;
            let draft_tok = self.sampler.sample(&draft_logits, tokens)?;
            draft_tokens.push(draft_tok);
            draft_probs_at_pos.push(draft_p);
        }

        // ── 2. Verify phase ─────────────────────────────────────────
        // Feed the last accepted token + all draft tokens (except the last)
        // through the target model. We need target logits at positions
        // corresponding to each draft token.
        //
        // Position layout:
        //   target gets [last_token, draft[0], draft[1], ..., draft[γ-1]]
        //   target logits[i] = distribution for position *after* input[i]
        //   So target logits[0] corresponds to draft[0]'s position
        //   And target logits[γ] corresponds to the bonus position
        //
        // Since candle's forward only returns last-position logits, we use
        // forward_each to get logits at each position.
        let verify_input: Vec<u32> = std::iter::once(last_token)
            .chain(draft_tokens.iter().copied())
            .collect();
        let target_logits_list = self.target.forward_each(&verify_input)?;

        // target_logits_list[i] = logits after seeing verify_input[i]
        // target_logits_list[0] = logits at draft_tokens[0]'s position
        // target_logits_list[γ]   = bonus position logits

        // ── 3. Rejection sampling ────────────────────────────────────
        let mut accepted_count = 0;

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
                // Accept this token
                tokens.push(draft_tokens[i]);
                accepted_count += 1;
            } else {
                // Reject — sample correction token from
                // norm(max(0, p_target - p_draft))
                let correction_token = self.sample_correction(&target_p, draft_p)?;
                tokens.push(correction_token);
                accepted_count += 1; // the correction token counts

                // Rollback draft model cache to the current position
                let new_pos = tokens.len();
                self.draft.truncate_cache_to(tokens, new_pos)?;

                // Rollback target model cache — we've already processed up to
                // verify_input[i], but we need to be at `new_pos`.
                // The target cache already has the right state up to this point
                // since we processed token-by-token. We just need to truncate
                // any extra cache entries from positions beyond the rejection.
                // Since forward_each processes one at a time, the target cache
                // has entries for all γ+1 positions. We need to truncate.
                self.target.truncate_cache_to(tokens, new_pos)?;

                return Ok(StepResult {
                    accepted: accepted_count,
                    hit_eos: self.draft.is_eos(correction_token)
                        || self.target.is_eos(correction_token),
                });
            }
        }

        // ── 4. All accepted — bonus token ────────────────────────────
        // Sample from the last target logits (position γ).
        let bonus_logits = &target_logits_list[self.gamma];
        let bonus_token = self.sampler.sample(bonus_logits, tokens)?;
        tokens.push(bonus_token);
        accepted_count += 1;

        // The draft model cache is now γ positions ahead of the target's
        // last verified position. Reset it to match.
        let new_pos = tokens.len();
        self.draft.truncate_cache_to(tokens, new_pos)?;

        Ok(StepResult {
            accepted: accepted_count,
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
            let logits = Tensor::new(target_p, &candle_core::Device::Cpu)?;
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
    pub fn generate(&mut self, prompt_tokens: Vec<u32>, max_new_tokens: usize) -> Result<Vec<u32>> {
        let mut tokens = prompt_tokens;

        // Prefill both models with the prompt
        self.prefill(&tokens)?;

        let initial_len = tokens.len();
        let mut total_generated = 0;

        while total_generated < max_new_tokens {
            let result = self.step(&mut tokens)?;
            total_generated = tokens.len() - initial_len;

            if result.hit_eos {
                break;
            }
        }

        // Trim to exact max_new_tokens if we overshot
        let max_len = initial_len + max_new_tokens;
        if tokens.len() > max_len {
            tokens.truncate(max_len);
        }

        Ok(tokens)
    }
}

/// Result of a single speculative step.
#[allow(dead_code)]
struct StepResult {
    /// Number of tokens accepted (including correction/bonus).
    accepted: usize,
    /// Whether an EOS token was generated.
    hit_eos: bool,
}
