"""Integration smoke tests for the spec_engine Python module."""

from spec_engine import AsyncSpecEngine


def test_basic_push_pull():
    """Push tokens, pull them back, verify order and contents."""
    engine = AsyncSpecEngine(queue_capacity=64, max_kv_blocks=128)

    for i in range(10):
        ok = engine.push_draft_token(token_id=i, kv_ptr=i)
        assert ok, f"push_draft_token failed for token {i}"

    batch = engine.pull_draft_batch(max_k=10)
    assert len(batch) == 10, f"expected 10 tokens, got {len(batch)}"

    for i, (token_id, kv_block_idx) in enumerate(batch):
        assert token_id == i, f"token {i}: expected id={i}, got {token_id}"


def test_rollback_flushes_queue_and_frees_blocks():
    """Push tokens, trigger rollback, verify queue is flushed and blocks freed."""
    engine = AsyncSpecEngine(queue_capacity=64, max_kv_blocks=128)

    initial_blocks = engine.available_kv_blocks()

    for i in range(5):
        engine.push_draft_token(token_id=i, kv_ptr=i)

    assert engine.queue_len() == 5
    assert engine.available_kv_blocks() == initial_blocks - 5

    # Rollback
    engine.trigger_rollback(corrected_token=99)

    assert engine.queue_len() == 0, "queue should be empty after rollback"
    assert engine.available_kv_blocks() == initial_blocks, "all blocks should be freed"


def test_flush_flag_blocks_producer():
    """After rollback, push_draft_token should return False until acknowledged."""
    engine = AsyncSpecEngine(queue_capacity=64, max_kv_blocks=128)

    engine.push_draft_token(token_id=1, kv_ptr=0)
    engine.trigger_rollback(corrected_token=42)

    # Producer should be blocked
    ok = engine.push_draft_token(token_id=2, kv_ptr=0)
    assert not ok, "push should fail while flush flag is set"

    # Acknowledge flush
    engine.acknowledge_flush()

    # Producer should resume
    ok = engine.push_draft_token(token_id=3, kv_ptr=0)
    assert ok, "push should succeed after acknowledge_flush"


def test_full_lifecycle():
    """End-to-end: push, pull, rollback, acknowledge, resume."""
    engine = AsyncSpecEngine(queue_capacity=16, max_kv_blocks=32)

    # Phase 1: Normal operation
    for i in range(8):
        assert engine.push_draft_token(token_id=i, kv_ptr=i)

    batch = engine.pull_draft_batch(max_k=4)
    assert len(batch) == 4
    assert engine.queue_len() == 4  # 4 remaining

    # Phase 2: Rollback
    engine.trigger_rollback(corrected_token=100)
    assert engine.queue_len() == 0

    # Phase 3: Producer sees flush
    assert not engine.push_draft_token(token_id=50, kv_ptr=0)

    # Phase 4: Acknowledge and resume
    engine.acknowledge_flush()
    for i in range(5):
        assert engine.push_draft_token(token_id=200 + i, kv_ptr=i)

    batch = engine.pull_draft_batch(max_k=10)
    assert len(batch) == 5
    assert batch[0][0] == 200


if __name__ == "__main__":
    test_basic_push_pull()
    print("✓ test_basic_push_pull")

    test_rollback_flushes_queue_and_frees_blocks()
    print("✓ test_rollback_flushes_queue_and_frees_blocks")

    test_flush_flag_blocks_producer()
    print("✓ test_flush_flag_blocks_producer")

    test_full_lifecycle()
    print("✓ test_full_lifecycle")

    print("\nAll tests passed!")
