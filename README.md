# HAWP-LAQ

Holistic Attention Weight Pruning with Learned Adaptive Quantization

## Quick Start

```bash
pip install -e .
```

## Architecture

### TurboQuant on Compressed KV

HAWP-LAQ compresses KV cache through a three-stage pipeline:

1. **Low-rank projection**: Project K/V into latent space via learned `P_k`, `P_v` matrices
   - `k_lat = k @ P_k[:, :r_k]`, `v_lat = v @ P_v[:, :r_v]`
   - Reduces per-token dimension from `head_dim` to `r_k` / `r_v`

2. **TurboQuant compression**: Quantize latent KV for archive tokens
   - **K** uses `TurboQuantProd` (MSE + 1-bit residual) — preserves inner-product fidelity for attention scores
   - **V** uses `TurboQuantMSE` (MSE-optimized) — preserves reconstruction quality for value aggregation
   - Recent tokens (within `recent_window`) kept as fp16, only archive tokens are quantized

3. **Three-state scheduler** (`hawp_quant_sched` mode):
   - **HIGH**: Recent tokens, fp16 latent, no compression
   - **LOW**: Archive tokens, TurboQuant compressed (K=TurboQuantProd, V=TurboQuantMSE)
   - **DROP**: Tokens exceeding budget, removed from attention entirely
   - Drop strategies: `position` (oldest first) or `norm` (smallest K-norm first)

### Running Modes

| Mode | Low-rank | TurboQuant | Scheduler | Description |
|------|----------|------------|-----------|-------------|
| `baseline` | No | No | No | Original model |
| `hawp_only` | Yes | No | No | Low-rank projection only |
| `quant_only` | No | Yes | No | TurboQuant on original KV space |
| `hawp_quant` | Yes | Yes | No | Low-rank + TurboQuant (recent/archive) |
| `hawp_quant_sched` | Yes | Yes | Yes | Full pipeline with DROP |

## Local Dev (Smoke Test)

Uses `facebook/opt-125m` on CPU, small seq_len:

```bash
# Generation
python scripts/04_run_generation_eval.py configs/dev_local.yaml --mode baseline
python scripts/04_run_generation_eval.py configs/dev_local.yaml --mode hawp_only
python scripts/04_run_generation_eval.py configs/dev_local.yaml --mode quant_only
python scripts/04_run_generation_eval.py configs/dev_local.yaml --mode hawp_quant
python scripts/04_run_generation_eval.py configs/dev_local.yaml --mode hawp_quant_sched

# Long context + perplexity + needle (small)
python scripts/05_run_long_context_eval.py configs/dev_local.yaml --mode hawp_quant

# KV memory profiling
python scripts/06_profile_kv_memory.py configs/dev_local.yaml --mode hawp_quant

# Debug quantization quality
python scripts/07_debug_latent_quant.py --seq-len 128 --r-k 64 --r-v 64
```

## Server (Full Experiment)

Uses `meta-llama/Llama-2-7b-hf` on GPU with 4-bit loading:

```bash
# Generation
python scripts/04_run_generation_eval.py configs/run_server.yaml --mode baseline
python scripts/04_run_generation_eval.py configs/run_server.yaml --mode hawp_quant
python scripts/04_run_generation_eval.py configs/run_server.yaml --mode hawp_quant_sched

# Perplexity + Needle-in-Haystack + Long context speed
python scripts/05_run_long_context_eval.py configs/run_server.yaml --mode baseline
python scripts/05_run_long_context_eval.py configs/run_server.yaml --mode hawp_quant
python scripts/05_run_long_context_eval.py configs/run_server.yaml --mode hawp_quant_sched

# KV memory profiling across seq_lens
python scripts/06_profile_kv_memory.py configs/run_server.yaml --mode hawp_quant_sched \
    --seq-lens 512 1024 2048 4096 8192
```

## Pipeline Steps

```bash
# 1. Collect calibration data
python scripts/01_collect_calib_data.py configs/dev_local.yaml

# 2. Train projectors
python scripts/02_train_projectors.py configs/dev_local.yaml

# 3. Build compressor package
python scripts/03_build_compressor.py configs/dev_local.yaml

# 4-7. Run evaluations (see above)
```

## Output Metrics

All eval scripts report:

- **Generation examples**: Input/output text pairs
- **Perplexity**: WikiText-2 test PPL (script 05)
- **Needle-in-Haystack**: Retrieval accuracy at various context lengths and depths (script 05)
- **Per-layer KV bytes**: Recent + archive bytes per transformer layer
- **Total KV bytes**: Aggregated across all layers
- **Saving ratio**: `1 - quantized_bytes / baseline_fp16_bytes`
