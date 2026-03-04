use std::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering};

use crossbeam_utils::CachePadded;
use log::{debug, trace};

use crate::spsc::DraftQueue;

/// Epoch-based state machine that coordinates the producer (draft model) and
/// consumer (target model), handling rollbacks when the target rejects a token.
///
/// # Protocol
///
/// **Consumer (target verifier):**
/// 1. Evaluates a batch of draft tokens.
/// 2. If token `N` is rejected, calls [`trigger_rollback`] with the corrected
///    token. This sets `flush_flag`, increments `epoch`, stores the corrected
///    token, and flushes the SPSC queue.
///
/// **Producer (draft generator):**
/// 1. Before each token generation, checks [`needs_flush`].
/// 2. If `true`, resets its local state (KV cache, etc.), syncs to
///    [`last_valid_token`], then calls [`acknowledge_flush`] to clear the flag.
pub struct EngineState {
    epoch: CachePadded<AtomicUsize>,
    last_valid_token: CachePadded<AtomicI64>,
    flush_flag: CachePadded<AtomicBool>,
}

impl EngineState {
    /// Creates a new engine state at epoch 0 with no pending flush.
    pub fn new() -> Self {
        Self {
            epoch: CachePadded::new(AtomicUsize::new(0)),
            last_valid_token: CachePadded::new(AtomicI64::new(0)),
            flush_flag: CachePadded::new(AtomicBool::new(false)),
        }
    }

    /// Returns the current epoch (generation timeline).
    pub fn current_epoch(&self) -> usize {
        self.epoch.load(Ordering::Acquire)
    }

    /// Returns the last valid token stored during the most recent rollback.
    pub fn last_valid_token(&self) -> i64 {
        self.last_valid_token.load(Ordering::Acquire)
    }

    /// Returns `true` if the producer must flush its local state.
    pub fn needs_flush(&self) -> bool {
        self.flush_flag.load(Ordering::Acquire)
    }

    /// Called by the **producer** after it has reset its local state in response
    /// to a flush. Clears the flush flag so the producer can resume drafting.
    pub fn acknowledge_flush(&self) {
        trace!("producer acknowledged flush, clearing flush flag");
        self.flush_flag.store(false, Ordering::Release);
    }

    /// Called by the **consumer** when it rejects a drafted token.
    ///
    /// This atomically:
    /// 1. Stores the corrected token.
    /// 2. Sets the flush flag.
    /// 3. Increments the epoch.
    /// 4. Flushes the SPSC queue (discarding all pending draft tokens).
    ///
    /// The consumer must ensure the producer is not concurrently pushing to the
    /// queue when this is called. In practice, the producer polls `flush_flag`
    /// before every push, so the queue quiesces naturally.
    pub fn trigger_rollback(&self, corrected_token: i64, queue: &DraftQueue) {
        let new_epoch = self.epoch.load(Ordering::Relaxed) + 1;
        debug!(
            "rollback triggered: corrected_token={}, new_epoch={}",
            corrected_token, new_epoch
        );
        self.last_valid_token.store(corrected_token, Ordering::Release);
        self.flush_flag.store(true, Ordering::Release);
        self.epoch.fetch_add(1, Ordering::Release);
        queue.flush();
    }
}

impl Default for EngineState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state() {
        let state = EngineState::new();
        assert_eq!(state.current_epoch(), 0);
        assert_eq!(state.last_valid_token(), 0);
        assert!(!state.needs_flush());
    }

    #[test]
    fn rollback_increments_epoch_and_sets_flag() {
        let state = EngineState::new();
        let queue = DraftQueue::new(8);

        state.trigger_rollback(42, &queue);

        assert_eq!(state.current_epoch(), 1);
        assert!(state.needs_flush());
        assert_eq!(state.last_valid_token(), 42);
    }

    #[test]
    fn acknowledge_clears_flush_flag() {
        let state = EngineState::new();
        let queue = DraftQueue::new(8);

        state.trigger_rollback(10, &queue);
        assert!(state.needs_flush());

        state.acknowledge_flush();
        assert!(!state.needs_flush());
    }

    #[test]
    fn multiple_rollbacks() {
        let state = EngineState::new();
        let queue = DraftQueue::new(8);

        for i in 0..5 {
            state.trigger_rollback(i * 10, &queue);
            assert_eq!(state.current_epoch(), i as usize + 1);
            assert_eq!(state.last_valid_token(), i * 10);
            state.acknowledge_flush();
        }

        assert_eq!(state.current_epoch(), 5);
        assert!(!state.needs_flush());
    }

    #[test]
    fn rollback_flushes_queue() {
        let state = EngineState::new();
        let queue = DraftQueue::new(8);

        use crate::spsc::DraftToken;
        for i in 0..5 {
            queue.push(DraftToken { token_id: i, kv_block_idx: 0 });
        }
        assert_eq!(queue.len(), 5);

        state.trigger_rollback(99, &queue);
        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());
    }
}
