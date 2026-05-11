# new_rank.yaml 实验配置说明

本文档对应 `configs/new_rank.yaml`，用于说明当前实验的基础设置、PPL、Needle、Speed、LongBench-E 和校准数据配置。

## 基础设置

| 项目 | 当前配置 |
| --- | --- |
| 运行模式 | `server` |
| 数据根目录 | `/data/wangzihao/work/matrix/data` |
| 缓存目录 | `/data/wangzihao/work/matrix/cache` |
| 模型 | `/data/wangzihao/work/matrix/Llama-3.1-8B-Instruct` |
| 模型精度 | `float16` |
| 4bit 加载 | `false` |
| 运行设备 | `cuda` |
| 默认评测模式 | `baseline`, `quant_only`, `hawp_quant` |
| 评测输出目录 | `/data/wangzihao/work/matrix/artifacts/eval_llama31_8b_instruct_longbench_e` |

默认生成配置：

```yaml
generation:
  max_new_tokens: 128
  do_sample: false
  temperature: 1.0
  top_p: 1.0
```

说明：

- `do_sample: false` 表示使用 greedy decoding。
- `generation.max_new_tokens: 128` 是通用默认值，但 PPL、Needle、Speed、LongBench-E 在 `09b_eval_metrics_compare.py` 中会使用各自的设置覆盖它。

## PPL 设置

```yaml
eval:
  ppl:
    seq_len: 1024
    nsamples: 32
```

含义：

- 每个模式评测 32 个片段。
- 每个片段长度为 1024 token。
- 评测脚本会报告 perplexity、NLL，以及 prefill/decode 分段 NLL。

## Needle 设置

```yaml
eval:
  needle:
    context_lens: [512, 1024, 2048, 4096]
    depths: [0, 25, 50, 75, 100]
    max_new_tokens: 32
```

含义：

- 上下文长度测试 4 档：512、1024、2048、4096。
- needle 插入深度测试 5 档：0%、25%、50%、75%、100%。
- 每个模式共 `4 x 5 = 20` 个 Needle case。
- `max_new_tokens: 32` 只限制答案最多生成 32 个新 token，不限制输入上下文长度。

## Speed 设置

```yaml
eval:
  speed:
    seq_lens: [512, 1024, 2048, 4096]
    max_new_tokens: 64
```

含义：

- 输入长度测试 4 档：512、1024、2048、4096。
- 每个长度下生成 64 个新 token。
- 评测脚本会记录吞吐、缓存字节数、压缩比和峰值显存。

## LongBench-E 设置

```yaml
eval:
  longbench:
    enabled: true
    data_dir: /data/wangzihao/work/matrix/data/longbench
    tasks:
      - qasper
      - multifieldqa_en
      - hotpotqa
      - 2wikimqa
      - gov_report
      - multi_news
      - trec
      - triviaqa
      - samsum
      - passage_count
      - passage_retrieval_en
      - lcc
      - repobench-p
    max_new_tokens: 512
```

任务分组：

| 类别 | 任务 |
| --- | --- |
| SingleQA | `qasper`, `multifieldqa_en` |
| MultiQA | `hotpotqa`, `2wikimqa` |
| Summarization | `gov_report`, `multi_news` |
| Few-shot | `trec`, `triviaqa`, `samsum` |
| Synthetic | `passage_count`, `passage_retrieval_en` |
| Code | `lcc`, `repobench-p` |

当前 `09b_eval_metrics_compare.py` 的实际行为：

- `eval.longbench.enabled: true` 会启用 LongBench-E。
- 默认任务来自 `eval.longbench.tasks`。
- 默认数据目录来自 `eval.longbench.data_dir`。
- `eval.longbench.max_new_tokens: 512` 当前不是 `09b` 的默认生成 cap。
- 若命令行不传 `--longbench-max-new-tokens`，实际使用任务级生成长度。

任务级生成长度：

| 任务 | max_new_tokens |
| --- | ---: |
| `qasper` | 128 |
| `multifieldqa_en` | 64 |
| `hotpotqa` | 32 |
| `2wikimqa` | 32 |
| `gov_report` | 512 |
| `multi_news` | 512 |
| `trec` | 64 |
| `triviaqa` | 32 |
| `samsum` | 128 |
| `passage_count` | 32 |
| `passage_retrieval_en` | 32 |
| `lcc` | 64 |
| `repobench-p` | 64 |

如果传入：

```bash
--longbench-max-new-tokens N
```

则实际生成长度为：

```text
min(任务级 max_new_tokens, N)
```

例如 `--longbench-max-new-tokens 128` 会把 `gov_report` 和 `multi_news` 从 512 截到 128，但不会把 32 或 64 的任务变长。

`09b` 的 LongBench-E 默认输入截断上限来自命令行默认值：

```text
--longbench-max-input-tokens 8192
```

## 推理压缩设置

```yaml
quant:
  enabled: true
  k_method: turbo_prod
  v_method: turbo_mse
  k_bits: 4
  v_bits: 8
  use_rotation_for_k: true
  use_rotation_for_v: false
  k_group_size: 128
  v_group_size: 128
  outlier_threshold: null

sched:
  total_budget: 4096
  recent_window: 0
  high_ratio: 0.25
  low_ratio: 0.60
  drop_strategy: position
```

含义：

- `quant.enabled: true` 启用 KV cache 量化。
- K 使用 `turbo_prod`，4 bit，目标是尽量保持 attention inner-product 质量。
- V 使用 `turbo_mse`，8 bit，目标是尽量保持 value latent 重建质量。
- `use_rotation_for_k: true` 表示 K 量化前会使用随机正交旋转；这可能提升量化质量，但重复评测同一 projector 时可能引入轻微波动。
- `use_rotation_for_v: false` 表示 V 量化前不做随机旋转。
- `recent_window: 0` 表示历史 token 不保留 fp recent window，都会进入量化 archive；当前正在计算的 token 仍在本次 forward 中以未归档状态参与计算，随后才写入 cache。
- `total_budget/high_ratio/low_ratio/drop_strategy` 主要用于 `hawp_quant_sched` 调度模式；当前默认评测模式里使用的是 `hawp_quant`，不会启用 DROP 调度。

## 校准数据设置

```yaml
calib:
  nsamples: 64
  seq_len: 2048
  output_dir: /data/wangzihao/work/matrix/artifacts/calib_new_rank
  dataset: wikitext2
  capture_mode: auto
```

含义：

- 收集 64 条校准样本。
- 每条样本长度为 2048 token。
- 数据集名为 `wikitext2`。如果此前用 `build_multiqa_code_calib.py --replace-wikitext2` 替换了本地 WikiText-2 文本，则这里实际会使用替换后的混合校准文本。
- `capture_mode: auto` 会根据模型结构自动选择校准捕获方式。
- 输出目录中会包含 `meta.pt` 和每层的 `layer_*.pt`，用于后续 projector 训练和 D_V 拟合。

## 常用评测命令

只跑 LongBench-E：

```bash
python -u scripts/09b_eval_metrics_compare.py configs/new_rank.yaml \
  --modes hawp_quant \
  --only-longbench
```

跑配置里的默认完整评测：

```bash
python -u scripts/09b_eval_metrics_compare.py configs/new_rank.yaml
```

默认完整评测会运行：

```text
baseline, quant_only, hawp_quant
```

并包含 PPL、Needle、Speed 和 LongBench-E；`distribution.enabled: false`，因此 distribution 指标默认不运行。
