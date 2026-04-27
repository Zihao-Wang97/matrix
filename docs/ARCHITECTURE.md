# HAWP-LAQ 技术文档

> Holistic Attention Weight Pruning with Learned Adaptive Quantization
> 版本 0.1.0

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 系统架构](#2-系统架构)
- [3. 核心模块](#3-核心模块)
  - [3.1 模型层 (modeling)](#31-模型层-modeling)
  - [3.2 量化引擎 (turboquant)](#32-量化引擎-turboquant)
  - [3.3 调度与缓存 (scheduler & cache)](#33-调度与缓存-scheduler--cache)
  - [3.4 离线训练 (offline)](#34-离线训练-offline)
  - [3.5 运行时生成 (runtime)](#35-运行时生成-runtime)
  - [3.6 评估 (eval)](#36-评估-eval)
  - [3.7 工具库 (utils)](#37-工具库-utils)
- [4. 运行模式](#4-运行模式)
- [5. 配置体系](#5-配置体系)
- [6. CLI 脚本](#6-cli-脚本)
- [7. 测试覆盖](#7-测试覆盖)
- [8. 快速上手](#8-快速上手)

---

## 1. 项目概述

HAWP-LAQ 是一个端到端的 KV Cache 压缩框架，通过 **低秩投影 + TurboQuant 量化 + 三态调度** 三级流水线，在保持生成质量的同时大幅降低长上下文推理的内存开销。

核心思路：

```
原始 K/V
  │
  ├─ 低秩投影 (P_k, P_v) ──→ 潜空间 k_lat, v_lat (维度 head_dim → r_k/r_v)
  │
  ├─ TurboQuant 压缩 ──→ archive token 量化存储
  │     ├─ K: TurboQuantProd (MSE + 1-bit 残差，保内积)
  │     └─ V: TurboQuantMSE  (MSE 优化，保重建)
  │
  └─ 三态调度 ──→ 根据 budget 分配 token 状态
        ├─ HIGH: recent fp16（近处保留全精度）
        ├─ LOW:  TurboQuant archive（远处压缩存储）
        └─ DROP: 超出预算丢弃（不参与 attention）
```

---

## 2. 系统架构

```
hawp_laq/
├── modeling/                    # 模型层
│   ├── attention_hawp.py        #   HAWPAttention 核心层
│   ├── modeling_llama_hawp.py   #   模型转换
│   └── rope_utils.py            #   RoPE 位置编码
├── runtime/                     # 运行时
│   ├── turboquant.py            #   TurboQuant 量化器
│   ├── quantizer.py             #   简化版量化器
│   ├── scheduler.py             #   三态调度器
│   ├── latent_cache.py          #   逐层 KV 缓存
│   ├── cache_manager.py         #   缓存管理 + Coordinator
│   ├── latent_quant_bridge.py   #   潜空间量化桥
│   ├── compressor.py            #   压缩包构建
│   ├── projector_bank.py        #   投影器存取
│   ├── generate.py              #   生成入口（5 种模式）
│   └── server.py                #   服务端（桩）
├── offline/                     # 离线训练
│   ├── hooks.py                 #   Q/K/V hook
│   ├── collector.py             #   校准数据收集
│   ├── dataset.py               #   校准数据集
│   ├── losses.py                #   训练损失
│   ├── projector_trainer.py     #   投影器训练器
│   └── rank_search.py           #   Rank 搜索
├── eval/                        # 评估
│   ├── metrics.py               #   KV 指标收集
│   ├── perplexity.py            #   困惑度
│   └── needle.py                #   Needle-in-Haystack
├── utils/                       # 工具库
│   ├── io.py                    #   文件 I/O
│   ├── math_utils.py            #   数学工具
│   ├── memory.py                #   内存估算
│   ├── seed.py                  #   随机种子
│   ├── logging.py               #   日志
│   └── packbits.py              #   int4 打包
└── config.py                    #   配置体系
```

---

## 3. 核心模块

### 3.1 模型层 (modeling)

#### `attention_hawp.py` — HAWPAttention

替换标准 attention 的核心层，支持三种前向路径：

| 方法 | 条件 | 行为 |
|------|------|------|
| `forward()` → full-rank | `r_k == head_dim` 且无量化缓存 | 标准 attention + P_k/P_v 投影 |
| `forward()` → low-rank | `r_k < head_dim` 或 `use_cache_manager=True` | 潜空间 attention + 量化缓存 |
| `_forward_low_rank()` | — | q/k/v 投影到潜空间，attention 后反投影 |

**量化缓存机制：**

```
token 进入
  │
  ▼
_quant_cache_append()  ──→  recent (fp16)
  │
  │  recent 超过 recent_window?
  ▼
_quant_cache_demote()  ──→  archive (TurboQuant 压缩)
                             K: TurboQuantProd
                             V: TurboQuantMSE
```

**DROP 方法：**

| 方法 | 策略 |
|------|------|
| `drop_oldest_from_archive(n)` | 丢弃 archive 中最早的 n 个 token |
| `drop_least_important_from_archive(n)` | 按 K 潜空间范数丢弃最小的 n 个 token |

**KV 读取：** `_quant_cache_get_kv()` 反量化 archive + 拼接 recent，返回完整 KV 序列。

#### `modeling_llama_hawp.py` — 模型转换

`convert_llama_to_hawp(model, r_k, r_v)` 自动识别并替换以下架构的 attention：

- LlamaDecoderLayer
- OPTDecoderLayer
- MistralDecoderLayer
- Qwen2DecoderLayer

### 3.2 量化引擎 (turboquant)

#### `TurboQuantMSE` — V 量化器

MSE 优化重建质量的量化器：

```
输入 x [T, D]
  │
  ├─ (可选) 随机正交旋转 R ──→ x' = x @ R^T
  │
  └─ 分组仿射量化 ──→ q (uint8), scale, zero_point
       group_size=128, bits=2/3/4/8
```

反量化：`(q - zero_point) * scale`，再逆旋转 `@ R`。

#### `TurboQuantProd` — K 量化器

保内积精度的两级量化器：

```
输入 x [T, D]
  │
  ├─ Stage 1: TurboQuantMSE ──→ x_hat (粗重建)
  │
  └─ Stage 2: 1-bit 残差 ──→ residual_sign (bool), residual_norm (float)
       r = x - x_hat
       sign = r >= 0
       norm = ||r||_2
```

反量化重建：`x_hat + (norm / sqrt(D)) * (2*sign - 1)`

**近似内积** `approx_inner_product(q, qx)`：MSE 部分直接矩阵乘，残差部分用 1-bit sign 做快速估计，无需全量反量化。

#### 对比

| | TurboQuantMSE | TurboQuantProd |
|---|---|---|
| 用途 | V（保重建） | K（保内积） |
| 精度 | 分组仿射 | 分组仿射 + 1-bit 残差 |
| 额外开销 | 无 | 每行 1 个 norm + D 个 sign bit |
| 内积质量 | 一般 | 更好（残差保相关性） |

### 3.3 调度与缓存 (scheduler & cache)

#### `TokenBudgetScheduler` — 三态调度器

给定 `total_budget` 和 `recent_window`，将 token 分配到三种状态：

| 状态 | 条件 | 存储 |
|------|------|------|
| HIGH | 在 `recent_window` 内 | fp16 latent，不压缩 |
| LOW | 超出 window 但在 budget 内 | TurboQuant 压缩 |
| DROP | 超出 budget | 丢弃，不参与 attention |

**关键方法：**

- `rebalance()` → `SchedulerDecision(n_high, n_low, n_drop)`
- `compute_drop_count()` → 增量计算新需 drop 的 token 数
- `on_tokens(n)` / `on_new_token()` → 推进序列长度

#### `ModelCacheCoordinator` — 调度执行器

连接 `TokenBudgetScheduler` 和 `HAWPAttention` 层：

```
on_prefill(prompt_len)  ──→  scheduler.on_tokens()  ──→  _apply_drop()
on_new_token()          ──→  scheduler.on_new_token() ──→  _apply_drop()

_apply_drop():
  drop_count = scheduler.compute_drop_count()
  for layer in HAWPAttention layers:
    position:  layer.drop_oldest_from_archive(n)
    norm:      layer.drop_least_important_from_archive(n)
```

#### `LayerKVCache` — 逐层缓存（独立实现）

与 `HAWPAttention` 内置缓存平行的独立实现，用于 `CacheManager`：

- `append_recent(k, v)` — 添加 fp16 latent
- `demote_to_archive()` — recent → archive 量化压缩
- `drop_oldest(n)` — 丢弃最老 n 个 archive token
- `get_all_k/v()` — 反量化 archive + 拼接 recent

### 3.4 离线训练 (offline)

训练流程：

```
01 收集校准数据          02 训练投影器           03 构建压缩包
   │                       │                      │
   ▼                       ▼                      ▼
hooks + collector    projector_trainer      compressor
   │                       │                      │
   ├─ Q/K/V 激活       ├─ P_k, P_v, gamma    ├─ KV 字节估算
   ├─ per-layer .pt    ├─ 正交化约束         ├─ per-layer profile
   └─ meta.pt          └─ 多目标损失         └─ compressor_meta.json
```

**损失函数** (`losses.py`)：

| 损失 | 公式 | 作用 |
|------|------|------|
| `logits_mse_loss` | MSE(logits, logits_hat) | 保持输出 logits 一致 |
| `attention_output_mse_loss` | MSE(attn_out, attn_out_hat) | 保持 attention 输出一致 |
| `value_reconstruction_loss` | MSE(V, gamma * V_hat @ P_v_up) | 保持 V 重建质量 |
| `total_projector_loss` | w1·logits + w2·attn + w3·value | 加权总损失 |

**Rank 搜索** (`rank_search.py`)：对每层独立评估候选 rank（默认 [16, 32, 64, 128, 256]），选最小满足容差的。

### 3.5 运行时生成 (runtime)

#### `generate.py` — 5 种生成模式

| 函数 | 模式 | 模型转换 | 量化缓存 | 调度器 |
|------|------|----------|----------|--------|
| `run_baseline` | baseline | 无 | 无 | 无 |
| `run_hawp_only` | hawp_only | 低秩投影 | 无 | 无 |
| `run_quant_only` | quant_only | 恒等投影(r=head_dim) | TurboQuant | 无 |
| `run_hawp_quant` | hawp_quant | 低秩投影 | TurboQuant | 无 |
| `run_hawp_quant_sched` | hawp_quant_sched | 低秩投影 | TurboQuant | 三态调度 |

**生成方式：** `generate_text()` 使用 HF `model.generate()`（baseline/hawp_only），`generate_hawp_quant()` 使用逐步解码 + 内部量化缓存管理（其余模式）。

### 3.6 评估 (eval)

| 模块 | 功能 | 输出 |
|------|------|------|
| `metrics.py` | 收集模型 KV 缓存指标 | per-layer bytes, total bytes, saving ratio |
| `perplexity.py` | WikiText-2 困惑度 | PPL, NLL, token 数 |
| `needle.py` | Needle-in-Haystack 检索 | 各 context_len × depth 的准确率 |

### 3.7 工具库 (utils)

| 模块 | 功能 |
|------|------|
| `io.py` | `.pt` / JSON 存取，自动创建目录 |
| `math_utils.py` | SVD 正交化 `orthogonalize()`、topk recall、hinge ranking loss |
| `memory.py` | `tensor_nbytes()` 字节估算、`format_nbytes()` 人类可读格式 |
| `seed.py` | `set_seed()` 设置 random/numpy/torch 全局种子 |
| `packbits.py` | `pack_int4()` / `unpack_int4()` 两个 int4 打包到一个 uint8 |

---

## 4. 运行模式

| 模式 | 低秩投影 | TurboQuant | 三态调度 | 典型场景 |
|------|----------|------------|----------|----------|
| `baseline` | ✗ | ✗ | ✗ | 对照基线 |
| `hawp_only` | ✓ | ✗ | ✗ | 仅测低秩投影效果 |
| `quant_only` | ✗ | ✓ | ✗ | 仅测量化压缩效果（无降维） |
| `hawp_quant` | ✓ | ✓ | ✗ | 低秩 + 量化，不限 token 数 |
| `hawp_quant_sched` | ✓ | ✓ | ✓ | 完整流水线，budget 约束 + DROP |

**token 生命周期（hawp_quant_sched）：**

```
新 token ──→ HIGH (recent fp16)
                │
                │  recent 超过 window
                ▼
            LOW (archive TurboQuant)
                │
                │  总数超过 budget
                ▼
            DROP (丢弃)
```

---

## 5. 配置体系

`config.py` 定义以下 dataclass，通过 YAML 加载：

| 配置类 | 关键字段 |
|--------|----------|
| `ModelConfig` | `model_id`, `torch_dtype`, `load_in_4bit` |
| `GenerationConfig` | `max_new_tokens`, `do_sample`, `prompts` |
| `ProjectorConfig` | `r_k`, `r_v`, `lr`, `n_steps`, `orthogonalize_every` |
| `QuantConfig` | `k_method`(turbo_prod), `v_method`(turbo_mse), `k_bits`(4), `v_bits`(8), `use_rotation` |
| `SchedConfig` | `total_budget`, `recent_window`, `high_ratio`, `low_ratio`, `drop_strategy`(position/norm) |
| `CalibConfig` | `nsamples`, `seq_len`, `dataset` |
| `RankSearchConfig` | `rank_candidates`, `relative_tolerance` |

**工厂函数：**

- `build_k_quantizer(cfg, r_k)` → 默认构建 `TurboQuantProd`
- `build_v_quantizer(cfg, r_v)` → 默认构建 `TurboQuantMSE`

**配置文件：**

| 文件 | 场景 | 模型 | 设备 | budget | window |
|------|------|------|------|--------|--------|
| `dev_local.yaml` | 本地 smoke test | opt-125m | CPU fp32 | 32 | 8 |
| `run_server.yaml` | 服务器实验 | Llama-2-7b | CUDA fp16+4bit | 4096 | 64 |

---

## 6. CLI 脚本

| 脚本 | 用途 | 示例 |
|------|------|------|
| `01_collect_calib_data.py` | 收集校准数据 | `python scripts/01_collect_calib_data.py configs/dev_local.yaml` |
| `02_train_projectors.py` | 训练投影器 | `python scripts/02_train_projectors.py configs/dev_local.yaml` |
| `03_build_compressor.py` | 构建压缩包 | `python scripts/03_build_compressor.py configs/dev_local.yaml` |
| `04_run_generation_eval.py` | 生成评估 | `python scripts/04_run_generation_eval.py configs/dev_local.yaml --mode hawp_quant` |
| `05_run_long_context_eval.py` | 综合评估 (PPL+Needle+速度) | `python scripts/05_run_long_context_eval.py configs/run_server.yaml --mode hawp_quant_sched` |
| `06_profile_kv_memory.py` | KV 内存 profiling | `python scripts/06_profile_kv_memory.py configs/run_server.yaml --mode hawp_quant` |
| `07_debug_latent_quant.py` | 量化质量调试 | `python scripts/07_debug_latent_quant.py --seq-len 128 --r-k 64` |

**脚本 05 选项：**

| 选项 | 说明 |
|------|------|
| `--mode` | baseline / hawp_only / quant_only / hawp_quant / hawp_quant_sched |
| `--seq-lens` | 测试序列长度列表 |
| `--skip-ppl` | 跳过困惑度测试 |
| `--skip-needle` | 跳过 Needle 测试 |
| `--max-new-tokens` | 生成的最大新 token 数 |

---

## 7. 测试覆盖

| 测试文件 | 覆盖范围 | 测试数 |
|----------|----------|--------|
| `test_scheduler_turboquant.py` | 三态调度 + TurboQuant + DROP + OPT-125m 端到端 | 21 |
| `test_scheduler.py` | TokenBudgetScheduler 基础逻辑 | 5 |
| `test_cache_manager_turboquant.py` | LayerKVCache + CacheManager + TurboQuant | 16 |
| `test_cache_manager.py` | CacheManager 旧版量化器 | 7 |
| `test_turboquant_mse.py` | TurboQuantMSE 全 bit roundtrip、旋转、序列化 | ~12 |
| `test_turboquant_prod.py` | TurboQuantProd 残差、近似内积、字节估算 | ~10 |
| `test_latent_quant_bridge.py` | KV 量化桥、压缩比 | ~10 |
| `test_quantizer.py` | int4/int8 量化器、打包/解包、正交化 | ~12 |
| `test_quantizer_builders.py` | 工厂函数 | ~5 |
| `test_quantizer_kv_split.py` | K=Prod / V=MSE 配置 | ~7 |
| `test_attention_equivalence.py` | HAWPAttention 等价性 + 生成对比 | ~8 |
| `test_calib.py` | 校准数据收集 | ~3 |
| `test_projector.py` | 投影器训练 | ~4 |
| `test_losses.py` | 损失函数 | ~6 |
| `test_generation.py` | 生成配置 | ~4 |
| `test_config.py` | 配置加载 | ~3 |
| `test_shapes.py` | 工具函数 | ~8 |

---

## 8. 快速上手

### 安装

```bash
pip install -e .
```

### 本地 Smoke Test

```bash
# 完整流水线
python scripts/01_collect_calib_data.py configs/dev_local.yaml
python scripts/02_train_projectors.py configs/dev_local.yaml
python scripts/03_build_compressor.py configs/dev_local.yaml

# 生成评估（5 种模式）
python scripts/04_run_generation_eval.py configs/dev_local.yaml --mode baseline
python scripts/04_run_generation_eval.py configs/dev_local.yaml --mode hawp_only
python scripts/04_run_generation_eval.py configs/dev_local.yaml --mode quant_only
python scripts/04_run_generation_eval.py configs/dev_local.yaml --mode hawp_quant
python scripts/04_run_generation_eval.py configs/dev_local.yaml --mode hawp_quant_sched

# 综合评估
python scripts/05_run_long_context_eval.py configs/dev_local.yaml --mode hawp_quant

# KV 内存 profiling
python scripts/06_profile_kv_memory.py configs/dev_local.yaml --mode hawp_quant

# 量化质量调试
python scripts/07_debug_latent_quant.py --seq-len 128 --r-k 64 --r-v 64
```

### 服务器实验

```bash
# 完整流水线
python scripts/01_collect_calib_data.py configs/run_server.yaml
python scripts/02_train_projectors.py configs/run_server.yaml
python scripts/03_build_compressor.py configs/run_server.yaml

# 生成评估
python scripts/04_run_generation_eval.py configs/run_server.yaml --mode baseline
python scripts/04_run_generation_eval.py configs/run_server.yaml --mode hawp_quant
python scripts/04_run_generation_eval.py configs/run_server.yaml --mode hawp_quant_sched

# 综合评估 (PPL + Needle + 长文本 + KV)
python scripts/05_run_long_context_eval.py configs/run_server.yaml --mode baseline
python scripts/05_run_long_context_eval.py configs/run_server.yaml --mode hawp_quant
python scripts/05_run_long_context_eval.py configs/run_server.yaml --mode hawp_quant_sched

# KV 内存 profiling
python scripts/06_profile_kv_memory.py configs/run_server.yaml --mode hawp_quant_sched \
    --seq-lens 512 1024 2048 4096 8192
```

### 输出指标

| 指标 | 说明 | 来源脚本 |
|------|------|----------|
| Generation examples | 输入/输出文本 | 04 |
| Perplexity | WikiText-2 PPL | 05 |
| Needle accuracy | 各长度/深度检索准确率 | 05 |
| Per-layer KV bytes | 每层 recent + archive 字节数 | 05, 06 |
| Total KV bytes | 全层汇总字节数 | 05, 06 |
| Saving ratio | `1 - quantized / baseline_fp16` | 05, 06 |
| Generation speed | tok/s + GPU peak memory | 05 |
