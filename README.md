# HAWP-LAQ

Holistic Attention Weight Pruning with Learned Adaptive Quantization.

This repository implements low-rank KV projection plus TurboQuant KV-cache
compression for long-context causal language model evaluation. The current
main experiment config is `configs/new_rank.yaml`.

## Quick Start

```bash
pip install -e .
```

For server runs, edit `configs/new_rank.yaml` so that `model.model_id`,
`data.root`, `calib.output_dir`, `projector.output_dir`, and
`eval.longbench.data_dir` point to paths on the machine.

## Core Idea

HAWP-LAQ replaces each attention layer with a HAWP attention module:

```text
K_lat = K @ P_K
V_lat = V @ P_V
attention = softmax((Q @ P_K) (K @ P_K)^T * scale)
output_lat = attention @ V_lat
output = output_lat @ P_V.T
```

Projector files contain:

```text
p_k, p_v, gamma, r_k, r_v, logit_scale_mode
```

Newer projector files may also contain:

```text
d_v
```

At runtime:

```text
if d_v exists: output = output_lat @ d_v
else:          output = output_lat @ P_V.T
```

So old projector directories without `d_v` remain valid.

## Runtime Modes

| Mode | Low-rank | TurboQuant | Scheduler | Description |
| --- | --- | --- | --- | --- |
| `baseline` | No | No | No | Original model |
| `hawp_only` | Yes | No | No | Low-rank projector only |
| `quant_only` | No | Yes | No | TurboQuant in full KV space |
| `pure_quant_only` | No | Yes | No | Hook-based original attention with quantized KV |
| `hawp_quant` | Yes | Yes | No | Low-rank plus TurboQuant cache |
| `hawp_quant_all` | Yes | Yes | No | HAWP quant with `recent_window=0` |
| `hawp_quant_sched` | Yes | Yes | Yes | HAWP quant plus token budget scheduler |

With `sched.recent_window: 0`, all historical KV tokens are archived through
TurboQuant. The current token is still handled by the active forward pass
before it is appended to the archive.

## Main Pipeline

The current server flow is:

```text
optional mixed calib text -> collect Q/K/V calib -> train P_K/P_V/gamma
-> optional distillation/refinement -> optional D_V fit -> LongBench-E eval
```

### 0. Build MultiQA + Code Calibration Text

This is optional, but useful when optimizing Code and MultiQA.

```bash
python scripts/build_multiqa_code_calib.py \
  --n-hotpot 2500 \
  --n-2wiki 2500 \
  --n-trivia 500 \
  --n-code 4000 \
  --n-wiki 500 \
  --code-languages python,javascript,java,c++,go,rust \
  --replace-wikitext2
```

This writes a mixed text file and, with `--replace-wikitext2`, replaces the
local WikiText-2 training text used by calibration.

### 1. Collect Calibration Activations

```bash
python -u scripts/01_collect_calib_data.py configs/new_rank.yaml
```

This writes:

```text
<calib.output_dir>/meta.pt
<calib.output_dir>/layer_0.pt
...
```

Each layer file stores captured `q`, `k`, and `v` tensors.

### 2. Train Projectors

Use the parallel script on multi-GPU servers:

```bash
python -u scripts/02_train_projectors_parallel.py configs/new_rank.yaml \
  --mode all \
  --workers 8 \
  --gpus 0 1 2 3 4 5 6 7
```

The script reads selected ranks from:

```text
<rank_search.output_dir>/selected_ranks.json
```

or falls back to existing `projector.output_dir/ranks.json`, then writes:

```text
<projector.output_dir>/layer_0/projector.pt
...
<projector.output_dir>/ranks.json
```

To run rank search first:

```bash
python -u scripts/02_train_projectors_parallel.py configs/new_rank.yaml \
  --mode rank_search \
  --workers 8 \
  --gpus 0 1 2 3 4 5 6 7
```

### 3. Optional Projector Refinement

Attention distillation on saved Q/K/V calibration:

```bash
python -u scripts/02b_refine_projectors_attention_distill.py configs/new_rank.yaml \
  --workers 8 \
  --gpus 0 1 2 3 4 5 6 7 \
  --clean-output-dir
```

Attention-module distillation on hidden-state chunks:

```bash
python -u scripts/01b_collect_layer_distill_data.py configs/new_rank.yaml

python -u scripts/02d_refine_projectors_attention_module_distill.py configs/new_rank.yaml \
  --workers 8 \
  --gpus 0 1 2 3 4 5 6 7 \
  --clean-output-dir
```

Layer micro distillation:

```bash
python -u scripts/02e_refine_projectors_layer_micro_distill.py configs/new_rank.yaml \
  --workers 8 \
  --gpus 0 1 2 3 4 5 6 7 \
  --clean-output-dir
```

If a refinement stage is used, feed its output directory into the next stage
or into evaluation.

### 4. Optional Output-Aware D_V Fit

`D_V` fitting is a post-processing stage. It does not retrain `P_K` or `P_V`.
It scans calibration data, accumulates closed-form ridge-regression statistics,
and writes a new projector directory containing optional `d_v` tensors.

Compressed-teacher D_V:

```bash
python -u scripts/02f_fit_output_aware_dv.py configs/new_rank.yaml \
  --workers 8 \
  --gpus 0 1 2 3 4 5 6 7 \
  --clean-output-dir
```

Default output:

```text
/data/wangzihao/work/matrix/artifacts/projectors_new_rank_dv
```

Cross-attention-teacher D_V:

```bash
python -u scripts/02g_fit_cross_attention_teacher_dv.py configs/new_rank.yaml \
  --workers 8 \
  --gpus 0 1 2 3 4 5 6 7 \
  --clean-output-dir
```

Default output:

```text
/data/wangzihao/work/matrix/artifacts/projectors_new_rank_dv_cross
```

`02f` uses:

```text
U = alpha_compressed @ V
Z = alpha_compressed @ QDQ(V @ P_V)
min ||U - Z D_V||^2 + lambda ||D_V||^2
```

`02g` uses:

```text
U = alpha_full @ V
Z = alpha_compressed @ QDQ(V @ P_V)
min ||U - Z D_V||^2 + lambda ||D_V||^2
```

`02g` also supports a mixed target:

```bash
python -u scripts/02g_fit_cross_attention_teacher_dv.py configs/new_rank.yaml \
  --teacher mixed_cross \
  --full-teacher-weight 0.2 \
  --workers 8 \
  --gpus 0 1 2 3 4 5 6 7 \
  --clean-output-dir
```

To evaluate a D_V directory, set:

```yaml
projector:
  output_dir: /path/to/projectors_new_rank_dv
```

or:

```yaml
projector:
  output_dir: /path/to/projectors_new_rank_dv_cross
```

To evaluate the original no-DV projector, point `projector.output_dir` back to
the original `projectors_new_rank` directory.

### 5. LongBench-E Evaluation

Run LongBench-E only:

```bash
python -u scripts/09b_eval_metrics_compare.py configs/new_rank.yaml \
  --modes hawp_quant \
  --only-longbench
```

Compare multiple modes:

```bash
python -u scripts/09b_eval_metrics_compare.py configs/new_rank.yaml \
  --modes baseline hawp_quant quant_only \
  --only-longbench
```

Useful options:

```bash
--longbench-tasks hotpotqa 2wikimqa lcc repobench-p
--longbench-max-samples-per-task 20
--longbench-max-input-tokens 8192
--longbench-max-new-tokens 512
--output-dir /path/to/eval_output
```

The evaluator writes:

```text
metrics_summary.csv
metrics_summary.json
longbench_predictions.jsonl
longbench_scores.json
```

and one subdirectory per evaluated mode.

## Important Config Sections

`projector` controls base projector training:

```yaml
projector:
  output_dir: /data/wangzihao/work/matrix/artifacts/projectors_new_rank
  lambda_z: 2.0
  lambda_o: 3.0
  lambda_v: 0.05
  lambda_topk: 0.05
  lambda_kl: 0.05
  lambda_logit_topm: 0.02
```

`attention_distill` controls optional attention-level refinement.

`dv_output_aware` controls `02f` and `02g`:

```yaml
dv_output_aware:
  input_dir: /data/wangzihao/work/matrix/artifacts/projectors_new_rank
  output_dir: /data/wangzihao/work/matrix/artifacts/projectors_new_rank_dv
  teacher: compressed
  full_teacher_weight: 1.0
  quant_aware: true
  current_token_fp: true
  lambda_ridge: 1e-3
```

`quant` controls runtime KV quantization:

```yaml
quant:
  enabled: true
  k_method: turbo_prod
  v_method: turbo_mse
  k_bits: 4
  v_bits: 8
  use_rotation_for_k: true
  use_rotation_for_v: false
```

## Reproducibility Notes

`do_sample: false` makes decoding greedy, but it does not by itself guarantee
identical LongBench-E scores across repeated runs.

When `quant.use_rotation_for_k: true`, each runtime K quantizer builds a random
orthogonal rotation matrix. Re-running evaluation with the same projector files
can therefore produce slightly different logits and different generated text.

For more stable comparisons:

```yaml
quant:
  use_rotation_for_k: false
```

This keeps K quantization enabled but removes the random pre-quantization
rotation. It may reduce quantization quality, so use it as a stability check or
make sure random seeds and quantizer states are controlled when reporting final
numbers.

## Script Map

| Script | Purpose |
| --- | --- |
| `01_collect_calib_data.py` | Capture Q/K/V calibration tensors |
| `01b_collect_layer_distill_data.py` | Capture hidden states for module/layer distillation |
| `02_train_projectors_parallel.py` | Multi-GPU rank search and projector training |
| `02b_refine_projectors_attention_distill.py` | Refine projectors on attention/value losses |
| `02d_refine_projectors_attention_module_distill.py` | Refine with attention-module output distillation |
| `02e_refine_projectors_layer_micro_distill.py` | Refine with layer micro distillation |
| `02f_fit_output_aware_dv.py` | Closed-form compressed-teacher D_V fitting |
| `02g_fit_cross_attention_teacher_dv.py` | Closed-form cross-teacher D_V fitting |
| `03_build_compressor.py` | Build compressor package from projectors |
| `04_run_generation_eval.py` | Small generation checks |
| `05_run_long_context_eval.py` | PPL, needle, and long-context checks |
| `06_profile_kv_memory.py` | KV memory profiling |
| `07_debug_latent_quant.py` | Latent quantization debug utility |
| `09b_eval_metrics_compare.py` | Main quality, memory, speed, LongBench-E evaluator |
| `build_multiqa_code_calib.py` | Build mixed MultiQA + code calibration text |

## Output Metrics

The main evaluator reports:

- LongBench-E category scores: SingleQA, MultiQA, Summarization, Few-shot,
  Synthetic, Code
- LongBench-E average and task average
- Perplexity and stepwise NLL when enabled
- Needle-in-haystack retrieval when enabled
- Cache runtime bytes, compressed bytes, and KV compression ratio
- Peak GPU allocation and speed profiles when enabled
