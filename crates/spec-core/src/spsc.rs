use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};

use crossbeam_utils::CachePadded;

/// Single token flowing through the draft queue.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct DraftToken {
    pub token_id: i64,
    pub kv_block_idx: usize,
}

/// Lock-free Single-Producer Single-Consumer ring buffer.
///
/// # Safety Contract
///
/// This queue is safe **only** under the SPSC invariant:
/// - Exactly one thread calls [`push`].
/// - Exactly one thread calls [`pop_batch`].
/// - [`flush`] must only be called when both sides are quiesced (e.g. during a
///   rollback orchestrated by the consumer after the producer has acknowledged
///   the flush flag).
pub struct DraftQueue {
    buffer: Box<[UnsafeCell<DraftToken>]>,
    capacity: usize,
    head: CachePadded<AtomicUsize>,
    tail: CachePadded<AtomicUsize>,
}

// SAFETY: The SPSC contract guarantees that `head` is only written by the
// producer and `tail` is only written by the consumer. The buffer slots are
// never concurrently written and read for the same index because the producer
// must increment `head` (with Release) *after* writing, and the consumer must
// read `head` (with Acquire) *before* reading — ensuring proper happens-before.
unsafe impl Send for DraftQueue {}
unsafe impl Sync for DraftQueue {}

impl DraftQueue {
    /// Creates a new `DraftQueue` with the given capacity.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is zero or not a power of two.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0 && capacity.is_power_of_two(), "capacity must be a power of 2");

        let buffer: Vec<UnsafeCell<DraftToken>> = (0..capacity)
            .map(|_| UnsafeCell::new(DraftToken::default()))
            .collect();

        Self {
            buffer: buffer.into_boxed_slice(),
            capacity,
            head: CachePadded::new(AtomicUsize::new(0)),
            tail: CachePadded::new(AtomicUsize::new(0)),
        }
    }

    /// Mask to convert a monotonic counter to a buffer index.
    #[inline]
    fn mask(&self) -> usize {
        self.capacity - 1
    }

    /// Pushes a token onto the queue.
    ///
    /// Returns `true` on success, `false` if the queue is full.
    /// Must be called by the **producer** thread only.
    pub fn push(&self, token: DraftToken) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        if head.wrapping_sub(tail) >= self.capacity {
            return false; // Queue is full
        }

        // SAFETY: No other thread writes to this slot — the producer owns all
        // slots from `tail` to `head`, and we've just confirmed there is room.
        unsafe {
            *self.buffer[head & self.mask()].get() = token;
        }

        self.head.store(head.wrapping_add(1), Ordering::Release);
        true
    }

    /// Pops up to `max_k` tokens from the queue.
    ///
    /// Returns a `Vec` of tokens in FIFO order (may be empty).
    /// Must be called by the **consumer** thread only.
    pub fn pop_batch(&self, max_k: usize) -> Vec<DraftToken> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);

        let available = head.wrapping_sub(tail);
        let count = available.min(max_k);

        let mut batch = Vec::with_capacity(count);
        for i in 0..count {
            // SAFETY: The producer has already written these slots and
            // incremented `head` with Release; our Acquire on `head` ensures
            // we see the writes.
            let token = unsafe { *self.buffer[(tail.wrapping_add(i)) & self.mask()].get() };
            batch.push(token);
        }

        self.tail.store(tail.wrapping_add(count), Ordering::Release);
        batch
    }

    /// Resets the queue, logically discarding all pending tokens.
    ///
    /// # Safety Contract
    ///
    /// Must only be called when both producer and consumer are quiesced
    /// (i.e. neither is concurrently calling `push` or `pop_batch`).
    pub fn flush(&self) {
        self.head.store(0, Ordering::SeqCst);
        self.tail.store(0, Ordering::SeqCst);
    }

    /// Returns the current number of tokens in the queue (snapshot).
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        head.wrapping_sub(tail)
    }

    /// Returns `true` if the queue is empty (snapshot).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_push_pop() {
        let q = DraftQueue::new(8);

        for i in 0..5 {
            assert!(q.push(DraftToken { token_id: i, kv_block_idx: i as usize }));
        }

        let batch = q.pop_batch(10);
        assert_eq!(batch.len(), 5);
        for (i, tok) in batch.iter().enumerate() {
            assert_eq!(tok.token_id, i as i64);
            assert_eq!(tok.kv_block_idx, i);
        }
    }

    #[test]
    fn capacity_enforcement() {
        let q = DraftQueue::new(4);

        for i in 0..4 {
            assert!(q.push(DraftToken { token_id: i, kv_block_idx: 0 }));
        }

        // Queue is full
        assert!(!q.push(DraftToken { token_id: 99, kv_block_idx: 0 }));
        assert_eq!(q.len(), 4);
    }

    #[test]
    fn wrap_around() {
        let q = DraftQueue::new(4);

        // Fill and drain twice to force wrap-around
        for round in 0..3 {
            for i in 0..4 {
                let id = (round * 4 + i) as i64;
                assert!(q.push(DraftToken { token_id: id, kv_block_idx: 0 }));
            }

            let batch = q.pop_batch(4);
            assert_eq!(batch.len(), 4);
            for (i, tok) in batch.iter().enumerate() {
                assert_eq!(tok.token_id, (round * 4 + i) as i64);
            }
        }
    }

    #[test]
    fn flush_clears_queue() {
        let q = DraftQueue::new(8);

        for i in 0..5 {
            q.push(DraftToken { token_id: i, kv_block_idx: 0 });
        }
        assert_eq!(q.len(), 5);

        q.flush();
        assert_eq!(q.len(), 0);
        assert!(q.is_empty());

        let batch = q.pop_batch(10);
        assert!(batch.is_empty());
    }

    #[test]
    fn concurrent_stress() {
        use std::sync::Arc;
        use std::thread;

        const COUNT: i64 = 100_000;
        let q = Arc::new(DraftQueue::new(1024));

        let q_producer = Arc::clone(&q);
        let producer = thread::spawn(move || {
            for i in 0..COUNT {
                while !q_producer.push(DraftToken { token_id: i, kv_block_idx: i as usize }) {
                    std::hint::spin_loop();
                }
            }
        });

        let q_consumer = Arc::clone(&q);
        let consumer = thread::spawn(move || {
            let mut received = Vec::with_capacity(COUNT as usize);
            while received.len() < COUNT as usize {
                let batch = q_consumer.pop_batch(64);
                received.extend(batch);
            }
            received
        });

        producer.join().unwrap();
        let received = consumer.join().unwrap();

        assert_eq!(received.len(), COUNT as usize);
        for (i, tok) in received.iter().enumerate() {
            assert_eq!(tok.token_id, i as i64, "out-of-order at index {i}");
        }
    }

    #[test]
    #[should_panic(expected = "capacity must be a power of 2")]
    fn non_power_of_two_panics() {
        DraftQueue::new(3);
    }

    #[test]
    #[should_panic(expected = "capacity must be a power of 2")]
    fn zero_capacity_panics() {
        DraftQueue::new(0);
    }
}
