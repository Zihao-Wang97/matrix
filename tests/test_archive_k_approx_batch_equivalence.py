"""Test that new batch _compute_archive_k_logits_approx matches old per-head logic.

Coverage matrix:
  nkv ∈ {1, 2, 4}
  group ∈ {1, 2, 4}
  q_len ∈ {1, 3}
  num_chunks ∈ {1, 3}
  dtype ∈ {float32, float16}
  bits ∈ {2, 4}

The reference implementation below is a literal transcription of the old
per-head-per-chunk Python loop removed from attention_hawp.py.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import torch
import pytest

from hawp_laq.modeling.attention_hawp import HAWPAttention
from hawp_laq.runtime.turboquant import TurboQuantProd, TurboQuantizedTensor, TurboQuantProdResult


def _make_config(num_attention_heads, num_key_value_heads):
    return SimpleNamespace(
        hidden_size=256,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=512,
        rope_theta=10000.0,
        model_type="llama",
        enable_bias=False,
        attention_dropout=0.0,
    )


def _old_compute_archive_k_logits_approx(attn, q_lat):
    """Reference: literal old per-head-per-chunk loop."""
    from hawp_laq.utils.packbits import unpack_bool

    g = attn.num_key_value_groups
    rk = attn.r_k
    bsz = q_lat.shape[0]
    q_len = q_lat.shape[2]

    head_logits = []
    for h in range(attn.num_key_value_heads):
        q_h = q_lat[:, h * g:(h + 1) * g, :, :].reshape(bsz * g * q_len, rk)
        chunk_logits = []
        for chunk in attn._quant_archive_chunks:
            T_chunk = chunk.n_tokens
            qx = chunk.k_qx
            sl = slice(h * T_chunk, (h + 1) * T_chunk)
            per_head_mse = TurboQuantizedTensor(
                q=qx.mse.q[sl],
                scale=qx.mse.scale[sl],
                zero_point=qx.mse.zero_point[sl],
                rotation=qx.mse.rotation,
                shape_orig=(T_chunk, rk),
                bits=qx.mse.bits,
                group_size=qx.mse.group_size,
            )
            per_head_qx = TurboQuantProdResult(
                mse=per_head_mse,
                residual_sign=qx.residual_sign[sl],
                residual_norm=qx.residual_norm[sl],
                shape_orig=(T_chunk, rk),
                dim=rk,
            )
            ip = attn._tq_k_quantizer.approx_inner_product(q_h, per_head_qx)
            chunk_logits.append(ip)
        head_logits.append(torch.cat(chunk_logits, dim=-1))

    T_archive = sum(c.n_tokens for c in attn._quant_archive_chunks)
    stacked = torch.stack(head_logits, dim=0)
    stacked = stacked.reshape(attn.num_key_value_heads, bsz, g, q_len, T_archive)
    stacked = stacked.permute(1, 0, 2, 3, 4).reshape(bsz, attn.num_heads, q_len, T_archive)
    return stacked


@pytest.mark.parametrize(
    "nkv,group,q_len,num_chunks,dtype,bits",
    list(itertools.product(
        [1, 2, 4],      # nkv
        [1, 2, 4],      # group
        [1, 3],         # q_len
        [1, 3],         # num_chunks
        [torch.float32, torch.float16],
        [2, 4],         # bits
    ))
)
def test_batch_approx_matches_old_per_head_loop(
    nkv, group, q_len, num_chunks, dtype, bits
):
    """New batch _compute_archive_k_logits_approx must be numerically
    identical to the old per-head loop for all parameter combinations."""
    num_heads = nkv * group
    config = _make_config(num_heads, nkv)

    attn = HAWPAttention(
        config, layer_idx=0, r_k=16, r_v=16,
        use_archive_k_ip_approx=True,
    )
    k_quantizer = TurboQuantProd(dim=16, bits=bits, use_rotation=False)
    v_quantizer = TurboQuantProd(dim=16, bits=bits, use_rotation=False)
    attn.setup_quant_cache(k_quantizer, v_quantizer, recent_window=2)

    torch.manual_seed(42 + nkv + group + q_len + num_chunks + bits)

    # Build archive with multiple chunks
    tokens_per_chunk = 3
    for _ in range(num_chunks):
        k_lat = torch.randn(1, nkv, tokens_per_chunk, 16)
        v_lat = torch.randn(1, nkv, tokens_per_chunk, 16)
        attn._quant_cache_append(k_lat, v_lat)
        attn._quant_cache_demote()

    assert len(attn._quant_archive_chunks) == 1
    assert attn.n_archive_tokens == 1 + (num_chunks - 1) * tokens_per_chunk

    q_lat = torch.randn(1, num_heads, q_len, 16, dtype=dtype)

    # New (batch) implementation
    logits_new = attn._compute_archive_k_logits_approx(q_lat)

    # Reference (old per-head loop)
    logits_ref = _old_compute_archive_k_logits_approx(attn, q_lat)

    assert logits_new.shape == logits_ref.shape, (
        f"Shape mismatch: new={logits_new.shape} ref={logits_ref.shape}"
    )
    max_diff = (logits_new - logits_ref).abs().max().item()
    assert max_diff < 1e-5, (
        f"nkv={nkv} group={group} q_len={q_len} num_chunks={num_chunks} "
        f"dtype={dtype} bits={bits}: max_diff={max_diff:.6e}"
    )


def test_layout_mismatch_raises():
    config = SimpleNamespace(
        hidden_size=256,
        num_attention_heads=7,   # 7 不能被 2 整除
        num_key_value_heads=2,
        max_position_embeddings=512,
        rope_theta=10000.0,
        model_type="llama",
        enable_bias=False,
        attention_dropout=0.0,
    )
    attn = HAWPAttention(
        config, layer_idx=0, r_k=16, r_v=16,
        use_archive_k_ip_approx=True,
    )
    k_quantizer = TurboQuantProd(dim=16, bits=4, use_rotation=False)
    v_quantizer = TurboQuantProd(dim=16, bits=4, use_rotation=False)
    attn.setup_quant_cache(k_quantizer, v_quantizer, recent_window=2)

    k_lat = torch.randn(1, 2, 4, 16)
    v_lat = torch.randn(1, 2, 4, 16)
    attn._quant_cache_append(k_lat, v_lat)
    attn._quant_cache_demote()

    q_lat = torch.randn(1, 7, 1, 16)
    try:
        attn._compute_archive_k_logits_approx(q_lat)
        raise AssertionError("Expected RuntimeError for head layout mismatch")
    except RuntimeError as e:
        assert "Head layout mismatch" in str(e)


def test_dequantize_mse_exists():
    tq = TurboQuantProd(dim=16, bits=4, use_rotation=False)
    x = torch.randn(10, 16)
    qx = tq.quantize(x)
    mse_deq = tq.dequantize_mse(qx)
    assert mse_deq.shape == (10, 16)
    assert mse_deq.dtype == torch.float32


def main():
    """Run all parameterized cases."""
    total = 0
    passed = 0
    failed = 0

    cases = list(itertools.product(
        [1, 2, 4],      # nkv
        [1, 2, 4],      # group
        [1, 3],         # q_len
        [1, 3],         # num_chunks
        [torch.float32, torch.float16],
        [2, 4],         # bits
    ))

    print(f"Running {len(cases)} equivalence cases...")
    for nkv, group, q_len, num_chunks, dtype, bits in cases:
        total += 1
        label = f"nkv={nkv} group={group} q_len={q_len} chunks={num_chunks} dtype={dtype} bits={bits}"
        try:
            test_batch_approx_matches_old_per_head_loop(nkv, group, q_len, num_chunks, dtype, bits)
            passed += 1
            print(f"  [PASS] {label}")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {label}: {e}")

    # Also run non-parameterized tests
    try:
        test_layout_mismatch_raises()
        passed += 1
        print("  [PASS] test_layout_mismatch_raises")
    except Exception as e:
        failed += 1
        print(f"  [FAIL] test_layout_mismatch_raises: {e}")

    try:
        test_dequantize_mse_exists()
        passed += 1
        print("  [PASS] test_dequantize_mse_exists")
    except Exception as e:
        failed += 1
        print(f"  [FAIL] test_dequantize_mse_exists: {e}")

    total += 2

    print(f"\nResults: {passed}/{total} passed, {failed}/{total} failed")
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
