# 09b_eval_metrics_compare.py 参数与指标说明

`scripts/09b_eval_metrics_compare.py` 用于对比不同运行模式的质量、长上下文能力、KV cache 占用和速度。常用模式：

```bash
baseline quant_only hawp_quant
```

## 常用命令

只跑 LongBench-E：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/09b_eval_metrics_compare.py configs/run_server_longbench_e.yaml \
  --modes baseline \
  --only-longbench
```

跑 LongBench-E + Speed/KV：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/09b_eval_metrics_compare.py configs/run_server_longbench_e.yaml \
  --modes baseline \
  --run-longbench \
  --skip-ppl \
  --skip-needle \
  --skip-distribution
```

## 参数说明

| 参数 | 作用 |
|---|---|
| `config` | YAML 配置文件路径，例如 `configs/run_server_longbench_e.yaml` |
| `--modes` | 选择测试模式，例如 `baseline quant_only hawp_quant` |
| `--output-dir` | 指定输出目录；不写则使用配置里的 `eval.output_dir` |
| `--skip-ppl` | 跳过 PPL / NLL |
| `--skip-needle` | 跳过 Needle-In-A-Haystack |
| `--skip-speed` | 跳过速度和独立 KV profile |
| `--skip-distribution` | 跳过 KL / Argmax / Top-k overlap |
| `--run-longbench` | 开启 LongBench-E，同时保留其它未 skip 的指标 |
| `--skip-longbench` | 强制跳过 LongBench-E |
| `--only-longbench` | 只跑 LongBench-E；自动跳过 PPL、Needle、Speed、Distribution |
| `--longbench-data-dir` | LongBench 数据目录 |
| `--longbench-tasks` | 指定 LongBench-E 子任务，例如 `qasper hotpotqa` |
| `--longbench-max-samples-per-task` | 每个任务最多测试多少条样本；冒烟测试可设为 `2` |
| `--longbench-max-input-tokens` | 限制输入 prompt token 数；默认 `8192` |
| `--longbench-max-new-tokens` | 全局限制生成 token 数；不写则使用每个任务默认值 |
| `--longbench-chat-template` | 是否使用 tokenizer chat template：`auto` / `always` / `never` |

## 输出目录

脚本会写顶层汇总，并为每个 mode 写单独子目录：

```text
output_dir/
  metrics_summary.csv
  metrics_summary.json
  needle_details.jsonl
  speed_details.json
  longbench_predictions.jsonl
  longbench_scores.json
  baseline/
  quant_only/
  hawp_quant/
```

多卡并行时，建议每张卡跑一个 mode，重点看各自子目录。

## 指标说明

| 指标 | 含义 |
|---|---|
| `ppl` | Perplexity，越低越好，表示语言建模困惑度 |
| `nll` | Negative Log-Likelihood，越低越好 |
| `delta_ppl` | 相对 baseline 的 PPL 变化 |
| `kl_mean` / `kl_p95` / `kl_max` | 候选 mode 与 baseline logits 分布的 KL divergence，越低越接近 baseline |
| `argmax_agreement` | 候选 mode 与 baseline 的 top-1 token 是否一致，越高越好 |
| `top{k}_overlap` | top-k token 集合重合率，越高越好 |
| `needle_recall` | Needle-In-A-Haystack 找回率，越高越好 |
| `delta_needle_recall` | 相对 baseline 的 Needle recall 变化 |
| `speed_tokens_per_s_mean` | 多个长度下的平均生成速度 |
| `speed_profile_tokens_per_s` | 最大测试长度下的生成速度 |
| `cache_runtime_bytes` | 运行时 KV cache 占用 |
| `cache_compressed_bytes` | 压缩后归档 KV 大小 |
| `baseline_kv_bytes` | 同长度下 baseline fp16 KV 理论大小 |
| `kv_compression_ratio` | KV 压缩比，约等于 `baseline_kv_bytes / cache_runtime_bytes` |
| `bytes_per_token` | 每 token KV 平均字节数 |
| `peak_gpu_bytes` | 整个测试过程的 GPU 峰值显存，不只包含 KV cache |

## LongBench-E 指标

| 指标 | 包含任务 |
|---|---|
| `longbench_singleqa_score` | `qasper`, `multifieldqa_en` |
| `longbench_multiqa_score` | `hotpotqa`, `2wikimqa` |
| `longbench_summarization_score` | `gov_report`, `multi_news` |
| `longbench_few_shot_score` | `trec`, `triviaqa`, `samsum` |
| `longbench_synthetic_score` | `passage_count`, `passage_retrieval_en` |
| `longbench_code_score` | `lcc`, `repobench-p` |
| `longbench_average` | 上面 6 类分数的平均，是 LongBench-E 主指标 |
| `longbench_task_average` | 所有任务分数的直接平均 |

LongBench-E 还会输出 KV 汇总：

| 指标 | 含义 |
|---|---|
| `longbench_cache_runtime_bytes_mean` | LongBench 样本平均运行时 KV 占用 |
| `longbench_cache_runtime_bytes_max` | LongBench 样本最大运行时 KV 占用 |
| `longbench_cache_compressed_bytes_mean` | LongBench 样本平均压缩 KV 大小 |
| `longbench_baseline_kv_bytes_mean` | LongBench 样本平均 baseline KV 大小 |
| `longbench_kv_compression_ratio_mean` | LongBench 样本平均 KV 压缩比 |
| `longbench_peak_gpu_bytes_max` | LongBench 测试中的最大 GPU 峰值显存 |

## 注意

使用 `--only-longbench` 时，终端表格里的 `PPL`、`KL`、`needle`、`KVx`、`cache`、`peak`、`tok/s` 会显示 `NA`，这是因为这些模块被跳过了。LongBench 结果仍然有效，主要看 `LB avg`、`longbench_scores.json` 和 `longbench_predictions.jsonl`。
