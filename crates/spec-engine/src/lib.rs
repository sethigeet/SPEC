use pyo3::prelude::*;

use spec_core::{DraftQueue, DraftToken, EngineState, KVBlockAllocator};

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

/// Python module definition.
#[pymodule]
fn spec_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AsyncSpecEngine>()?;
    Ok(())
}
