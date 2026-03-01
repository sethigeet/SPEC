use std::sync::Mutex;

/// Manages indices into a pre-allocated VRAM KV cache.
///
/// Both the draft and target models share the same physical KV cache memory.
/// This allocator does **not** touch VRAM directly — it only hands out and
/// reclaims logical block indices. The actual VRAM addressing is handled on the
/// Python/CUDA side using these indices.
///
/// Each allocation is tagged with the current epoch so that on a rollback, all
/// blocks from the dead epoch can be bulk-freed.
pub struct KVBlockAllocator {
    free_stack: Mutex<Vec<usize>>,
    max_blocks: usize,
    /// Log of `(epoch, block_idx)` pairs for rollback-based freeing.
    epoch_log: Mutex<Vec<(usize, usize)>>,
}

impl KVBlockAllocator {
    /// Creates a new allocator with block indices `[0, max_blocks)`.
    pub fn new(max_blocks: usize) -> Self {
        let free_stack: Vec<usize> = (0..max_blocks).rev().collect();
        Self {
            free_stack: Mutex::new(free_stack),
            max_blocks,
            epoch_log: Mutex::new(Vec::new()),
        }
    }

    /// Allocates a single KV block, tagged with the given `epoch`.
    ///
    /// Returns `None` if all blocks are exhausted.
    pub fn alloc(&self, epoch: usize) -> Option<usize> {
        let mut stack = self.free_stack.lock().unwrap();
        let block_idx = stack.pop()?;

        let mut log = self.epoch_log.lock().unwrap();
        log.push((epoch, block_idx));

        Some(block_idx)
    }

    /// Frees a single block, returning it to the pool. Does **not** remove
    /// the block from the epoch log (use [`rollback`] for bulk freeing).
    pub fn free(&self, block_idx: usize) {
        let mut stack = self.free_stack.lock().unwrap();
        stack.push(block_idx);
    }

    /// Frees all blocks that were allocated under `dead_epoch` and removes
    /// their entries from the epoch log.
    ///
    /// This is called during a rollback when the consumer rejects a drafted
    /// token. All speculative KV cache entries from the dead timeline are
    /// reclaimed.
    pub fn rollback(&self, dead_epoch: usize) {
        let mut log = self.epoch_log.lock().unwrap();
        let mut stack = self.free_stack.lock().unwrap();

        log.retain(|&(epoch, block_idx)| {
            if epoch == dead_epoch {
                stack.push(block_idx);
                false // remove from log
            } else {
                true // keep
            }
        });
    }

    /// Returns the number of free blocks available for allocation.
    pub fn available(&self) -> usize {
        self.free_stack.lock().unwrap().len()
    }

    /// Returns the maximum number of blocks this allocator manages.
    pub fn max_blocks(&self) -> usize {
        self.max_blocks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_allocator_has_all_blocks_free() {
        let alloc = KVBlockAllocator::new(16);
        assert_eq!(alloc.available(), 16);
        assert_eq!(alloc.max_blocks(), 16);
    }

    #[test]
    fn alloc_and_free() {
        let alloc = KVBlockAllocator::new(4);

        let b0 = alloc.alloc(0).unwrap();
        let b1 = alloc.alloc(0).unwrap();
        assert_eq!(alloc.available(), 2);

        alloc.free(b0);
        assert_eq!(alloc.available(), 3);

        alloc.free(b1);
        assert_eq!(alloc.available(), 4);
    }

    #[test]
    fn alloc_exhaustion() {
        let alloc = KVBlockAllocator::new(2);

        assert!(alloc.alloc(0).is_some());
        assert!(alloc.alloc(0).is_some());
        assert!(alloc.alloc(0).is_none());
    }

    #[test]
    fn rollback_frees_dead_epoch_blocks() {
        let alloc = KVBlockAllocator::new(8);

        // Epoch 0: allocate 3 blocks
        alloc.alloc(0).unwrap();
        alloc.alloc(0).unwrap();
        alloc.alloc(0).unwrap();

        // Epoch 1: allocate 2 blocks
        alloc.alloc(1).unwrap();
        alloc.alloc(1).unwrap();

        assert_eq!(alloc.available(), 3); // 8 - 5 = 3

        // Rollback epoch 1 — should free 2 blocks
        alloc.rollback(1);
        assert_eq!(alloc.available(), 5); // 3 + 2 = 5

        // Epoch 0 blocks should still be allocated
        // Rollback epoch 0 — should free 3 blocks
        alloc.rollback(0);
        assert_eq!(alloc.available(), 8);
    }

    #[test]
    fn rollback_nonexistent_epoch_is_noop() {
        let alloc = KVBlockAllocator::new(4);
        alloc.alloc(0).unwrap();

        alloc.rollback(99);
        assert_eq!(alloc.available(), 3); // unchanged
    }

    #[test]
    fn unique_block_indices() {
        let alloc = KVBlockAllocator::new(8);
        let mut indices: Vec<usize> = (0..8).map(|_| alloc.alloc(0).unwrap()).collect();
        indices.sort();
        indices.dedup();
        assert_eq!(indices.len(), 8, "all block indices must be unique");
    }
}
