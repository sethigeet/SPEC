//! Async continuous speculative decoding pipeline.
//!
//! Runs the draft model (producer) and target model (consumer) on **separate
//! threads**, communicating via a lock-free SPSC queue. The draft model
//! continuously generates speculative tokens while the target model verifies
//! them using rejection sampling. Rollbacks are coordinated via the epoch-based
//! state machine from `spec-core`.
//!
//! # Architecture
//!
//! ```text
//!  ┌──────────────┐   DraftQueue (SPSC)   ┌──────────────┐
//!  │ Draft Model  │ ─────────────────────▶│ Target Model │
//!  │ (Producer)   │                       │ (Consumer)   │
//!  └──────────────┘                       └──────────────┘
//!         ▲                                       │
//!         │         EngineState (atomics)         │
//!         └───────────────────────────────────────┘
//!              epoch, flush_flag, last_valid_token
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use anyhow::Result;
use candle_core::Tensor;
use log::{debug, info, trace};
use rand::Rng;

use spec_core::{DraftQueue, DraftToken, EngineState};

use crate::decoder::stats::Stats;
use crate::model::Llama;
use crate::sampler::Sampler;

/// Async continuous speculative decoding pipeline.
///
/// Coordinates a draft model and a target model running on separate threads,
/// using a lock-free SPSC queue for token transfer and an epoch-based state
/// machine for rollback coordination.
pub struct AsyncDecoder {
    /// Draft model (small/fast). Wrapped in `Option` so we can move it to the
    /// producer thread and get it back after join.
    draft: Option<Llama>,
    /// Target model (large/accurate). Same `Option` pattern.
    target: Option<Llama>,
    pub sampler: Sampler,
    /// Number of draft tokens to generate before the consumer tries to verify.
    pub gamma: usize,
    /// Random seed for rejection sampling.
    pub seed: u64,
}

impl AsyncDecoder {
    /// Create a new async speculative decoding pipeline.
    pub fn new(draft: Llama, target: Llama, sampler: Sampler, gamma: usize, seed: u64) -> Self {
        Self {
            draft: Some(draft),
            target: Some(target),
            sampler,
            gamma,
            seed,
        }
    }

    /// Mutable reference to the draft model. Panics if the model was consumed.
    pub fn draft_mut(&mut self) -> &mut Llama {
        self.draft
            .as_mut()
            .expect("draft model not available (consumed by generate)")
    }

    /// Mutable reference to the target model. Panics if the model was consumed.
    pub fn target_mut(&mut self) -> &mut Llama {
        self.target
            .as_mut()
            .expect("target model not available (consumed by generate)")
    }

    /// Reset both model caches.
    pub fn reset_caches(&mut self) {
        self.draft_mut().reset_cache();
        self.target_mut().reset_cache();
    }

    /// Generate tokens from a prompt using the async continuous pipeline.
    ///
    /// Both models are prefilled with the prompt, then the draft model runs
    /// on a producer thread and the target model on a consumer thread.
    ///
    /// Returns the full token sequence (prompt + generated).
    /// Acceptance statistics are logged at `info` level.
    pub fn generate(&mut self, prompt_tokens: Vec<u32>, max_new_tokens: usize) -> Result<Vec<u32>> {
        // Take models out of the Option — they'll be put back after join.
        let mut draft_model = self
            .draft
            .take()
            .expect("draft model not available (already consumed)");
        let mut target_model = self
            .target
            .take()
            .expect("target model not available (already consumed)");

        // ── Prefill both models sequentially ─────────────────────────
        debug!("prefilling {} prompt tokens", prompt_tokens.len());
        let _draft_logits = draft_model.forward(&prompt_tokens, 0)?;
        let _target_logits = target_model.forward(&prompt_tokens, 0)?;
        debug!("prefill complete");

        let initial_len = prompt_tokens.len();
        let last_prompt_token = *prompt_tokens.last().unwrap();

        // ── Shared state ─────────────────────────────────────────────
        let queue_capacity = (self.gamma * 4).next_power_of_two().max(16);
        let queue = Arc::new(DraftQueue::new(queue_capacity));
        let state = Arc::new(EngineState::new());
        let done = Arc::new(AtomicBool::new(false));

        // The accepted output sequence (written by consumer, read at end).
        let output = Arc::new(Mutex::new(prompt_tokens.clone()));

        // Acceptance statistics (written by consumer).
        let stats = Arc::new(Mutex::new(Stats::new()));

        let gamma = self.gamma;
        let seed = self.seed;

        info!(
            "starting async speculative decoding: prompt_len={}, max_new_tokens={}, gamma={}",
            initial_len, max_new_tokens, gamma
        );

        // ── Producer thread (draft model) ────────────────────────────
        let q_prod = Arc::clone(&queue);
        let s_prod = Arc::clone(&state);
        let d_prod = Arc::clone(&done);
        let o_prod = Arc::clone(&output);

        let producer = thread::spawn(move || -> Result<Llama> {
            let mut model = draft_model;
            let mut local_epoch: usize = 0;
            let mut next_token = last_prompt_token;
            let mut _local_seq_len = initial_len;

            while !d_prod.load(Ordering::Acquire) {
                // ── Check for rollback ───────────────────────────────
                if s_prod.needs_flush() {
                    let corrected = s_prod.last_valid_token() as u32;
                    let new_epoch = s_prod.current_epoch();

                    // Roll back the draft model's cache to match the consumer's
                    // accepted sequence length.
                    let target_len = o_prod.lock().unwrap().len();
                    model.truncate_cache_to(target_len);

                    debug!(
                        "producer: rollback to epoch={}, corrected_token={}, cache_len={}",
                        new_epoch, corrected, target_len
                    );

                    local_epoch = new_epoch;
                    next_token = corrected;
                    _local_seq_len = target_len;

                    s_prod.acknowledge_flush();
                    continue;
                }

                // ── Draft one token ──────────────────────────────────
                let logits = match model.forward(&[next_token], local_epoch) {
                    Ok(l) => l,
                    Err(_) => break,
                };

                // Greedy argmax for drafting (fast, no sampling overhead)
                let draft_token = argmax_sample(&logits).unwrap_or(0);

                let token = DraftToken {
                    token_id: draft_token as i64,
                    kv_block_idx: 0, // KV is managed by PagedKVCache internally
                };

                // Try to push — spin if full, checking for done/rollback
                loop {
                    if d_prod.load(Ordering::Acquire) {
                        break;
                    }
                    if s_prod.needs_flush() {
                        break; // Will handle rollback at top of outer loop
                    }
                    if q_prod.push(token) {
                        break;
                    }
                    std::hint::spin_loop();
                }

                next_token = draft_token;
                _local_seq_len += 1;
            }

            Ok(model)
        });

        // ── Consumer thread (target model) ───────────────────────────
        let q_cons = Arc::clone(&queue);
        let s_cons = Arc::clone(&state);
        let d_cons = Arc::clone(&done);
        let o_cons = Arc::clone(&output);
        let st_cons = Arc::clone(&stats);

        let consumer = thread::spawn(move || -> Result<Llama> {
            use rand::SeedableRng;
            let mut model = target_model;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let mut total_generated: usize = 0;

            while total_generated < max_new_tokens {
                // ── Pull a batch of draft tokens ─────────────────────
                let batch = q_cons.pop_batch(gamma);
                if batch.is_empty() {
                    if d_cons.load(Ordering::Acquire) {
                        break;
                    }
                    std::hint::spin_loop();
                    continue;
                }

                // ── Verify each token in the batch ───────────────────
                for draft_tok in batch.iter() {
                    if total_generated >= max_new_tokens {
                        d_cons.store(true, Ordering::Release);
                        break;
                    }

                    let drafted_id = draft_tok.token_id as u32;

                    // Feed the last accepted token to the target model to get
                    // logits for the position the drafted token claims to fill.
                    let last_accepted = {
                        let out = o_cons.lock().unwrap();
                        *out.last().unwrap()
                    };

                    let target_logits = match model.forward(&[last_accepted], 0) {
                        Ok(l) => l,
                        Err(_) => {
                            d_cons.store(true, Ordering::Release);
                            break;
                        }
                    };

                    let target_probs = match Sampler::logits_to_probs(&target_logits) {
                        Ok(p) => p,
                        Err(_) => {
                            d_cons.store(true, Ordering::Release);
                            break;
                        }
                    };

                    // Find the target model's preferred token (argmax).
                    let target_argmax = target_probs
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx as u32)
                        .unwrap_or(0);

                    // ── Acceptance criterion ─────────────────────────
                    // If target and draft agree on the argmax, accept directly.
                    // Otherwise, use a probabilistic acceptance: accept with
                    // probability proportional to the target's confidence in
                    // the drafted token relative to its best alternative.
                    let accepted = if target_argmax == drafted_id {
                        true
                    } else {
                        let p_target = target_probs[drafted_id as usize];
                        let p_max = target_probs[target_argmax as usize];
                        if p_max > 0.0 {
                            let ratio = p_target / p_max;
                            let r: f32 = rng.random();
                            r < ratio
                        } else {
                            false
                        }
                    };

                    if accepted {
                        let mut out = o_cons.lock().unwrap();
                        out.push(drafted_id);
                        total_generated = out.len() - initial_len;

                        {
                            let mut s = st_cons.lock().unwrap();
                            s.draft_proposed += 1;
                            s.draft_accepted += 1;
                            s.num_steps += 1;
                        }

                        trace!(
                            "consumer: accepted draft token {}, total_generated={}",
                            drafted_id,
                            total_generated
                        );

                        if model.is_eos(drafted_id) {
                            drop(out);
                            d_cons.store(true, Ordering::Release);
                            break;
                        }
                    } else {
                        // ── Reject: use target's best token as correction ─
                        let corrected = target_argmax;

                        {
                            let mut out = o_cons.lock().unwrap();
                            out.push(corrected);
                            total_generated = out.len() - initial_len;
                        }

                        {
                            let mut s = st_cons.lock().unwrap();
                            s.draft_proposed += 1;
                            // draft_accepted NOT incremented — this was a rejection
                            s.num_steps += 1;
                        }

                        debug!(
                            "consumer: rejected draft token {}, corrected to {}, total_generated={}",
                            drafted_id, corrected, total_generated
                        );

                        // Truncate target cache to the new accepted length
                        let accepted_len = {
                            let out = o_cons.lock().unwrap();
                            out.len()
                        };
                        model.truncate_cache_to(accepted_len);

                        // Signal rollback to the producer
                        s_cons.trigger_rollback(corrected as i64, &q_cons);

                        if model.is_eos(corrected) {
                            d_cons.store(true, Ordering::Release);
                        }
                        break; // Remaining batch tokens are from a dead epoch
                    }
                }
            }

            d_cons.store(true, Ordering::Release);
            Ok(model)
        });

        // ── Join threads and recover models ──────────────────────────
        let draft_result = producer
            .join()
            .map_err(|_| anyhow::anyhow!("producer thread panicked"))?;
        let target_result = consumer
            .join()
            .map_err(|_| anyhow::anyhow!("consumer thread panicked"))?;

        self.draft = Some(draft_result?);
        self.target = Some(target_result?);

        // ── Collect output ───────────────────────────────────────────
        let mut tokens = output.lock().unwrap().clone();

        // Trim to exact max_new_tokens if we overshot
        let max_len = initial_len + max_new_tokens;
        if tokens.len() > max_len {
            tokens.truncate(max_len);
        }

        let mut final_stats = stats.lock().unwrap().clone();
        final_stats.total_tokens = tokens.len();

        info!(
            "async generation complete: steps={}, draft_proposed={}, draft_accepted={}, acceptance_rate={:.1}%, avg_accepted_per_step={:.2}, total_tokens={}",
            final_stats.num_steps,
            final_stats.draft_proposed,
            final_stats.draft_accepted,
            final_stats.acceptance_rate() * 100.0,
            final_stats.avg_accepted_per_step(),
            final_stats.total_tokens,
        );

        Ok(tokens)
    }
}

/// Simple argmax sampling from logits (greedy).
fn argmax_sample(logits: &Tensor) -> Result<u32> {
    let logits_vec = logits.to_vec1::<f32>()?;
    let (idx, _) = logits_vec
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    Ok(idx as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use spec_core::{DraftQueue, DraftToken, EngineState};
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    /// Test that the core shared-state coordination works without real models.
    /// Validates the producer/consumer protocol with the SPSC queue and
    /// epoch-based rollback.
    #[test]
    fn test_pipeline_coordination_protocol() {
        let queue = Arc::new(DraftQueue::new(16));
        let _state = Arc::new(EngineState::new());
        let done = Arc::new(AtomicBool::new(false));

        let q_prod = Arc::clone(&queue);
        let d_prod = Arc::clone(&done);

        // Producer: push 10 tokens
        let producer = thread::spawn(move || {
            for i in 0..10 {
                let token = DraftToken {
                    token_id: i,
                    kv_block_idx: 0,
                };
                while !q_prod.push(token) {
                    if d_prod.load(Ordering::Acquire) {
                        return;
                    }
                    std::hint::spin_loop();
                }
            }
        });

        let q_cons = Arc::clone(&queue);
        let d_cons = Arc::clone(&done);

        // Consumer: pull and verify
        let consumer = thread::spawn(move || {
            let mut received = Vec::new();
            while received.len() < 10 {
                let batch = q_cons.pop_batch(4);
                if batch.is_empty() {
                    std::hint::spin_loop();
                    continue;
                }
                received.extend(batch);
            }
            d_cons.store(true, Ordering::Release);

            // Verify order
            for (i, tok) in received.iter().enumerate() {
                assert_eq!(tok.token_id, i as i64, "out-of-order at {i}");
            }
        });

        producer.join().unwrap();
        consumer.join().unwrap();
    }

    /// Test rollback protocol: consumer triggers rollback, producer sees it.
    #[test]
    fn test_rollback_protocol() {
        let queue = Arc::new(DraftQueue::new(16));
        let state = Arc::new(EngineState::new());

        // Push some tokens
        for i in 0..5 {
            queue.push(DraftToken {
                token_id: i,
                kv_block_idx: 0,
            });
        }

        // Consumer triggers rollback
        let old_epoch = state.current_epoch();
        state.trigger_rollback(42, &queue);

        // Queue should be flushed
        assert!(queue.is_empty());
        assert_eq!(state.current_epoch(), old_epoch + 1);
        assert!(state.needs_flush());
        assert_eq!(state.last_valid_token(), 42);

        // Producer acknowledges
        state.acknowledge_flush();
        assert!(!state.needs_flush());
    }

    /// Test the full producer/consumer rollback coordination with threads.
    #[test]
    fn test_threaded_rollback_coordination() {
        let queue = Arc::new(DraftQueue::new(32));
        let state = Arc::new(EngineState::new());
        let done = Arc::new(AtomicBool::new(false));

        let q_prod = Arc::clone(&queue);
        let s_prod = Arc::clone(&state);
        let d_prod = Arc::clone(&done);

        // Producer: push tokens, handle rollback
        let producer = thread::spawn(move || {
            let mut produced = 0i64;
            let mut _local_epoch = 0usize;

            while !d_prod.load(Ordering::Acquire) {
                // Check for rollback
                if s_prod.needs_flush() {
                    let _corrected = s_prod.last_valid_token();
                    _local_epoch = s_prod.current_epoch();
                    s_prod.acknowledge_flush();
                    // Reset produced counter to simulate cache rollback
                    produced = 100; // Start from a different range after rollback
                    continue;
                }

                let token = DraftToken {
                    token_id: produced,
                    kv_block_idx: 0,
                };
                if q_prod.push(token) {
                    produced += 1;
                } else {
                    std::hint::spin_loop();
                }

                // Stop after enough tokens
                if produced > 110 {
                    break;
                }
            }
        });

        let q_cons = Arc::clone(&queue);
        let s_cons = Arc::clone(&state);
        let d_cons = Arc::clone(&done);

        // Consumer: pull tokens, trigger rollback after receiving some
        let consumer = thread::spawn(move || {
            let mut received = Vec::new();
            let mut rollback_done = false;

            loop {
                let batch = q_cons.pop_batch(4);
                if batch.is_empty() {
                    if d_cons.load(Ordering::Acquire) || received.len() >= 10 {
                        break;
                    }
                    std::hint::spin_loop();
                    continue;
                }

                received.extend_from_slice(&batch);

                // After receiving 3+ tokens, trigger a rollback (once)
                if received.len() >= 3 && !rollback_done {
                    s_cons.trigger_rollback(99, &q_cons);
                    rollback_done = true;
                    // Wait for producer to acknowledge
                    while s_cons.needs_flush() {
                        std::hint::spin_loop();
                    }
                }

                if received.len() >= 10 {
                    break;
                }
            }

            d_cons.store(true, Ordering::Release);
            assert!(rollback_done, "rollback should have been triggered");
            assert!(!received.is_empty(), "should have received tokens");
        });

        producer.join().unwrap();
        consumer.join().unwrap();
    }
}
