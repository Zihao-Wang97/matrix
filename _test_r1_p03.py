"""Round 1: verify no raw data is kept and shapes match old behavior."""
import torch
import sys

def test_no_raw_data_after_demote():
    """After demote, no _archive_k_raw/_archive_v_raw should exist."""
    from hawp_laq.runtime.latent_cache import LayerKVCache
    from hawp_laq.runtime.turboquant import TurboQuantProd, TurboQuantMSE
    torch.manual_seed(42)

    kq = TurboQuantProd(dim=64, bits=4, use_rotation=False, group_size=64)
    vq = TurboQuantMSE(dim=64, bits=4, use_rotation=False, group_size=64)
    cache = LayerKVCache(n_heads=2, head_dim=64, k_quantizer=kq, v_quantizer=vq, dtype=torch.float16)

    # No raw fields should exist
    assert not hasattr(cache, '_archive_k_raw'), "_archive_k_raw still exists!"
    assert not hasattr(cache, '_archive_v_raw'), "_archive_v_raw still exists!"

    # Append and demote
    for i in range(100):
        k = torch.randn(1, 64)
        v = torch.randn(1, 64)
        cache.append_recent(k, v)

    cache.demote_to_archive()

    # Only chunk-based archive
    assert hasattr(cache, '_archive_chunks'), "_archive_chunks missing"
    assert len(cache._archive_chunks) > 0, "no chunks after demote"
    print("[PASS] No raw data fields; only _archive_chunks exists")

def test_archive_memory_no_raw():
    """Archive memory should be much smaller than old raw+quant."""
    from hawp_laq.runtime.latent_cache import LayerKVCache
    from hawp_laq.runtime.turboquant import TurboQuantProd, TurboQuantMSE
    torch.manual_seed(42)

    kq = TurboQuantProd(dim=64, bits=4, use_rotation=False, group_size=64)
    vq = TurboQuantMSE(dim=64, bits=4, use_rotation=False, group_size=64)
    cache = LayerKVCache(n_heads=2, head_dim=64, k_quantizer=kq, v_quantizer=vq, dtype=torch.float16)

    for i in range(100):
        cache.append_recent(torch.randn(1, 64), torch.randn(1, 64))

    recent_bytes_before = cache.nbytes_recent()
    cache.demote_to_archive()
    recent_bytes_after = cache.nbytes_recent()
    archive_bytes = cache.nbytes_archive()
    total_bytes = cache.nbytes_total()

    # Old behavior: raw 100*64*2*2=25600B + quantized ~8600B = ~34200B
    # New behavior: only quantized ~8600B
    old_raw_estimate = 100 * 64 * 2 * 2  # 100 tokens * 64 dim * 2 heads * 2 bytes (fp16)
    old_total_estimate = old_raw_estimate + archive_bytes  # raw + quant

    print(f"  archive quantized: {archive_bytes}B")
    print(f"  old raw would add: {old_raw_estimate}B")
    print(f"  old total: {old_total_estimate}B  new total: {total_bytes}B")
    print(f"  memory saving: {1.0 - total_bytes / old_total_estimate:.1%}")

    # The new total should NOT include raw bytes
    assert total_bytes < old_total_estimate * 0.4, f"Memory not saved: total={total_bytes}, old_total={old_total_estimate}"
    assert recent_bytes_before > 0, "recent bytes should be >0 before demote"
    assert recent_bytes_after == 0, "recent bytes should be 0 after demote"
    print("[PASS] Archive memory is only quantized data, no raw duplicate")

def test_get_all_k_shape():
    """get_all_k/v should return same shape as before: [T, D]."""
    from hawp_laq.runtime.latent_cache import LayerKVCache
    from hawp_laq.runtime.turboquant import TurboQuantProd, TurboQuantMSE
    torch.manual_seed(42)

    kq = TurboQuantProd(dim=64, bits=4, use_rotation=False, group_size=64)
    vq = TurboQuantMSE(dim=64, bits=4, use_rotation=False, group_size=64)
    cache = LayerKVCache(n_heads=2, head_dim=64, k_quantizer=kq, v_quantizer=vq, dtype=torch.float16)

    for i in range(50):
        cache.append_recent(torch.randn(1, 64), torch.randn(1, 64))
    cache.demote_to_archive()
    for i in range(10):
        cache.append_recent(torch.randn(1, 64), torch.randn(1, 64))

    k = cache.get_all_k()
    v = cache.get_all_v()
    assert k.shape == (60, 64), f"get_all_k shape: {k.shape}, expected (60, 64)"
    assert v.shape == (60, 64), f"get_all_v shape: {v.shape}, expected (60, 64)"
    print("[PASS] get_all_k/v shape correct: [T, D]")

def test_token_counts():
    """n_recent, n_archive, total_tokens should be accurate."""
    from hawp_laq.runtime.latent_cache import LayerKVCache
    from hawp_laq.runtime.turboquant import TurboQuantProd, TurboQuantMSE
    torch.manual_seed(42)

    kq = TurboQuantProd(dim=64, bits=4, use_rotation=False, group_size=64)
    vq = TurboQuantMSE(dim=64, bits=4, use_rotation=False, group_size=64)
    cache = LayerKVCache(n_heads=2, head_dim=64, k_quantizer=kq, v_quantizer=vq, dtype=torch.float16)

    assert cache.n_recent == 0
    assert cache.n_archive == 0
    assert cache.total_tokens == 0

    for i in range(50):
        cache.append_recent(torch.randn(1, 64), torch.randn(1, 64))
    assert cache.n_recent == 50
    assert cache.n_archive == 0

    cache.demote_to_archive()
    assert cache.n_recent == 0
    assert cache.n_archive == 50
    assert cache.total_tokens == 50

    for i in range(10):
        cache.append_recent(torch.randn(1, 64), torch.randn(1, 64))
    assert cache.n_recent == 10
    assert cache.n_archive == 50
    assert cache.total_tokens == 60

    print("[PASS] Token counts accurate")

def test_drop_oldest():
    """drop_oldest should work without raw data."""
    from hawp_laq.runtime.latent_cache import LayerKVCache
    from hawp_laq.runtime.turboquant import TurboQuantProd, TurboQuantMSE
    torch.manual_seed(42)

    kq = TurboQuantProd(dim=64, bits=4, use_rotation=False, group_size=64)
    vq = TurboQuantMSE(dim=64, bits=4, use_rotation=False, group_size=64)
    cache = LayerKVCache(n_heads=2, head_dim=64, k_quantizer=kq, v_quantizer=vq, dtype=torch.float16)

    for i in range(100):
        cache.append_recent(torch.randn(1, 64), torch.randn(1, 64))
    cache.demote_to_archive()

    assert cache.n_archive == 100

    # Drop whole chunk
    dropped = cache.drop_oldest(100)
    assert dropped == 100
    assert cache.n_archive == 0

    # Drop with multiple demotes (multiple chunks)
    for i in range(50):
        cache.append_recent(torch.randn(1, 64), torch.randn(1, 64))
    cache.demote_to_archive()
    for i in range(50):
        cache.append_recent(torch.randn(1, 64), torch.randn(1, 64))
    cache.demote_to_archive()
    assert cache.n_archive == 100
    assert len(cache._archive_chunks) == 2

    # Drop 30 (from first chunk of 50)
    dropped = cache.drop_oldest(30)
    assert dropped == 30
    assert cache.n_archive == 70

    # Drop another 30 (rest of first chunk + part of second)
    dropped = cache.drop_oldest(30)
    assert dropped == 30
    assert cache.n_archive == 40

    # Verify get_all_k still works
    k = cache.get_all_k()
    v = cache.get_all_v()
    assert k.shape[0] == 40, f"expected 40 tokens, got {k.shape[0]}"
    assert v.shape[0] == 40

    print("[PASS] drop_oldest works without raw data")

def test_multiple_demotes_no_raw_accumulation():
    """Multiple demotes should not accumulate raw data."""
    from hawp_laq.runtime.latent_cache import LayerKVCache
    from hawp_laq.runtime.turboquant import TurboQuantProd, TurboQuantMSE
    torch.manual_seed(42)

    kq = TurboQuantProd(dim=64, bits=4, use_rotation=False, group_size=64)
    vq = TurboQuantMSE(dim=64, bits=4, use_rotation=False, group_size=64)
    cache = LayerKVCache(n_heads=2, head_dim=64, k_quantizer=kq, v_quantizer=vq, dtype=torch.float16)

    for round in range(5):
        for i in range(50):
            cache.append_recent(torch.randn(1, 64), torch.randn(1, 64))
        cache.demote_to_archive()

    # 5 chunks, 250 total archive tokens
    assert len(cache._archive_chunks) == 5
    assert cache.n_archive == 250

    # Archive bytes should be ~5x a single chunk, NOT growing with raw data
    single_chunk_bytes = None
    for chunk in cache._archive_chunks:
        chunk_bytes = kq.estimate_num_bytes(chunk.k_qx) + vq.estimate_num_bytes(chunk.v_qx)
        if single_chunk_bytes is None:
            single_chunk_bytes = chunk_bytes
        else:
            # Each chunk should be roughly the same size
            ratio = chunk_bytes / single_chunk_bytes
            assert 0.8 < ratio < 1.2, f"chunk size varies too much: {ratio:.2f}"

    total_archive = cache.nbytes_archive()
    print(f"  5 chunks, each ~{single_chunk_bytes}B, total={total_archive}B")
    print(f"  If raw was kept: ~{250*64*2*2}B raw alone would dwarf this")
    print("[PASS] Multiple demotes don't accumulate raw data")

if __name__ == "__main__":
    print("=" * 60)
    print("Round 1: LayerKVCache no-raw verification")
    print("=" * 60)
    try:
        test_no_raw_data_after_demote()
        test_archive_memory_no_raw()
        test_get_all_k_shape()
        test_token_counts()
        test_drop_oldest()
        test_multiple_demotes_no_raw_accumulation()
        print("\n[ALL PASSED] Round 1")
    except AssertionError as e:
        print(f"\n[FAILED] {e}")
        sys.exit(1)
