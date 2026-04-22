# HAWP-LAQ Attention 计算路径静态分析

> 基于代码实际执行路径，不做设计假设。分析日期：2026-04-20

---

## A. 入口与调用链

### baseline

```
scripts/04_run_generation_eval.py 或 scripts/08_compare_modes.py
  → hawp_laq/runtime/generate.py::run_baseline()
    → generate_text()
      → model.generate()  (HF 原生，use_cache=True)
        → OPTSdpaAttention.forward()   ← 不经过 HAWPAttention
```

### hawp_only

```
scripts/08_compare_modes.py
  → hawp_laq/runtime/generate.py::run_hawp_only()
    → convert_llama_to_hawp()          ← 替换为 HAWPAttention
    → generate_text()
      → model.generate()               (HF 原生，use_cache=True)
        → HAWPAttention.forward()
          → _is_low_rank or use_cache_manager 判断
          → r_k < head_dim 时走 _forward_low_rank()
          → r_k == head_dim 且 use_cache_manager=False 时走 full-rank 分支
```

### hawp_quant

```
scripts/08_compare_modes.py
  → hawp_laq/runtime/generate.py::run_hawp_quant()
    → convert_llama_to_hawp()
    → setup_quant_cache()               ← use_cache_manager=True
    → generate_hawp_quant()             (逐步解码，use_cache=False)
      → HAWPAttention.forward()
        → use_cache_manager=True → _forward_low_rank()
```

### quant_only

```
scripts/08_compare_modes.py
  → hawp_laq/runtime/generate.py::run_quant_only()
    → convert_llama_to_hawp(r_k=head_dim, r_v=head_dim)  ← full-rank，无低秩投影
    → setup_quant_cache()               ← use_cache_manager=True
    → generate_hawp_quant()             (逐步解码，use_cache=False)
      → HAWPAttention.forward()
        → use_cache_manager=True → _forward_low_rank()   ← r_k==head_dim 但仍走 low_rank 路径
```

### hawp_quant_all

```
scripts/08_compare_modes.py
  → hawp_laq/runtime/generate.py::run_hawp_quant_all()
    → convert_llama_to_hawp()
    → setup_quant_cache(recent_window=0) ← 所有 token 全部走 archive，无 recent 分层
    → generate_hawp_quant()             (逐步解码，use_cache=False)
      → HAWPAttention.forward()
        → use_cache_manager=True → _forward_low_rank()
        → recent_window==0 → _quant_cache_append_to_archive()  ← 直接入 archive
```

### hawp_quant_sched

```
scripts/08_compare_modes.py
  → hawp_laq/runtime/generate.py::run_hawp_quant_sched()
    → convert_llama_to_hawp() + setup_quant_cache()
    → ModelCacheCoordinator.from_model()
    → generate_hawp_quant(coordinator=coordinator)
      → HAWPAttention.forward() → _forward_low_rank()
      → coordinator.on_prefill() / on_new_token()
        → scheduler.compute_drop_count()
        → layer.drop_oldest_from_archive(n)
```

---

## B. baseline 的 attention 如何计算

**完全调用 HF 原生 OPTSdpaAttention**，不涉及 HAWPAttention。

```
Q = q_proj(hidden) * scaling          # scaling = 1/√64 = 0.125
K = k_proj(hidden)
V = v_proj(hidden)

reshape → [bsz, 12, seq_len, 64]

logits = Q @ K^T * scaling            # SDPA kernel 内部
softmax(logits)
output = softmax @ V

reshape → o_proj → [bsz, seq_len, 768]
```

---

## C. hawp_only 的 attention 如何计算

### C1. r_k == head_dim (如 r_k=64)

`_is_low_rank=False`，`use_cache_manager=False`，走 `forward()` full-rank 分支。

```
attention_hawp.py L338-398:

Q = q_proj(hidden) * 0.125
K = k_proj(hidden)
V = v_proj(hidden)
reshape → [bsz, 12, seq, 64]

RoPE: OPT 跳过

KV cache update (HF cache, 存 full-dim K/V)

K = _apply_pk(K)     ← L378
V = _apply_pv(V)     ← L379

_repeat_kv(K, V)

OPT: _eager_attn(Q, K, V, scaling=1.0)
  logits = Q @ K^T              # Q 已含 0.125，等价 baseline scaling
  softmax(logits)
  output = softmax @ V

o_proj(output)
```

**`_apply_pk` 逻辑 (L515-520):**

| 条件 | 行为 |
|------|------|
| `r_k >= head_dim` 且 `p_k.requires_grad=False` | `return k`（恒等，不修改） |
| `r_k >= head_dim` 且 `p_k.requires_grad=True` | `k @ P_k @ P_k^T`（full-rank 变换） |
| `r_k < head_dim` | `k @ P_k[:,:r_k] @ P_k[:r_k,:]`（投影+回投） |

**`_apply_pv` 逻辑 (L522-527):**

| 条件 | 行为 |
|------|------|
| `r_v >= head_dim` 且 `p_v.requires_grad=False` | `return v` |
| `r_v >= head_dim` 且 `p_v.requires_grad=True` | `γ * v @ P_v @ P_v^T` |
| `r_v < head_dim` | `γ * v @ P_v[:,:r_v] @ P_v[:r_v,:]` |

**关键：** 当 `load_projectors` 加载了训练好的 P_k（requires_grad 可能变为 True），即使是 r_k=64，也会执行 `K @ P_k @ P_k^T`，这是一个 full-rank 变换，**不等价 baseline**。

### C2. r_k < head_dim (如 r_k=50)

`_is_low_rank=True`，进入 `_forward_low_rank()`。

```
attention_hawp.py L426-:

pk_down = P_k[:,:50]    # [64, 50]
pv_down = P_v[:,:50]    # [64, 50]
pv_up   = P_v[:50,:]   # [50, 64]

q_lat = Q @ pk_down     # [bsz, 12, seq, 50]
k_lat = K @ pk_down     # [bsz, 12, seq, 50]
v_lat = V @ pv_down     # [bsz, 12, seq, 50]

# use_cache_manager=False → 走 HF KV cache 分支 (L470-476)
k_lat, v_lat = past_key_value.update(k_lat, v_lat, ...)

# ⚠️ BUG: HF cache 按 head_dim=64 设计，存入 50 维 k_lat
#   后续 cache 读取时维度不匹配

logits = q_lat @ k_lat^T     # scaling=1.0 in _eager_attn
softmax(logits)
attn_output_lat = softmax @ v_lat

attn_output = γ * (attn_output_lat @ pv_up)    # 回投到 64 维
o_proj(attn_output)
```

### Shape 流程 (opt-125m, r_k=r_v=64, bsz=1, seq=5)

```
hidden:         [1, 5, 768]
Q = q_proj*0.125: [1, 12, 5, 64]
K = k_proj:       [1, 12, 5, 64]
V = v_proj:       [1, 12, 5, 64]

_apply_pk(K):     [1, 12, 5, 64]   (k @ I @ I = k，若 P_k=I)
_apply_pv(V):     [1, 12, 5, 64]   (γ * v @ I @ I = γv)

logits = Q @ K^T:  [1, 12, 5, 5]   (Q 含 0.125)
softmax:            [1, 12, 5, 5]
output = soft @ V:  [1, 12, 5, 64]
o_proj:             [1, 5, 768]
```

### Shape 流程 (opt-125m, r_k=r_v=50, bsz=1, seq=5)

```
hidden:           [1, 5, 768]
Q = q_proj*0.125: [1, 12, 5, 64]
K = k_proj:       [1, 12, 5, 64]
V = v_proj:       [1, 12, 5, 64]

q_lat = Q @ pk_down: [1, 12, 5, 50]
k_lat = K @ pk_down: [1, 12, 5, 50]
v_lat = V @ pv_down: [1, 12, 5, 50]

logits = q_lat @ k_lat^T: [1, 12, 5, 5]
softmax:                    [1, 12, 5, 5]
attn_output_lat:            [1, 12, 5, 50]
attn_output = γ*(lat@pv_up): [1, 12, 5, 64]
o_proj:                      [1, 5, 768]
```

---

## D. hawp_quant 的 attention 如何计算

### D1. 投影

```python
# attention_hawp.py L440-446
pk_down = self.p_k[:, :self.r_k]     # [head_dim, r_k]
pv_down = self.p_v[:, :self.r_v]     # [head_dim, r_v]
pv_up   = self.p_v[:self.r_v, :]     # [r_v, head_dim]

q_lat = query_states @ pk_down       # [bsz, n_heads, q_len, r_k]
k_lat = key_states @ pk_down         # [bsz, n_kv_heads, k_len, r_k]
v_lat = value_states @ pv_down       # [bsz, n_kv_heads, k_len, r_v]
```

**RoPE 在投影前已应用（L350-358），投影后 q_lat/k_lat 不含位置编码。**

### D2. 缓存管理

```python
# attention_hawp.py L450-469

# 第一步（prefill）：cache_was_empty=True
_quant_cache_append(k_lat, v_lat)     # 存入 recent (fp16)
# cache_was_empty=True → 不读缓存，直接用当前 k_lat/v_lat

# 后续步骤（decode）：cache_was_empty=False
_quant_cache_append(k_lat, v_lat)     # 新 token latent 追加到 recent

if recent_k.shape[1] > recent_window:
    _quant_cache_demote()              # recent → archive (全量重新量化)

k_cached, v_cached = _quant_cache_get_kv()  # 读取完整 KV
k_lat = k_cached                      # 替换为缓存值
v_lat = v_cached
```

### D3. 反量化时机

**在 `_quant_cache_get_kv()` 中（L237-252），attention 计算之前：**

```python
# 1. 反量化 archive K/V
k_deq = self._tq_k_quantizer.dequantize(self._quant_archive_k_qx)
v_deq = self._tq_v_quantizer.dequantize(self._quant_archive_v_qx)

# 2. reshape 回 [n_kv_heads, T_archive, r_k/r_v]
k_deq = k_deq.reshape(nkv, T, rk)
v_deq = v_deq.reshape(nkv, T, rv)

# 3. 拼接 archive(反量化) + recent(fp16)
k = torch.cat([k_deq, self._quant_recent_k.float()], dim=1)
v = torch.cat([v_deq, self._quant_recent_v.float()], dim=1)
```

### D4. Quantizer 类型

| 路径 | Quantizer 类 | 特点 |
|------|-------------|------|
| K (archive) | `TurboQuantProd` | MSE + 1-bit 残差，保内积精度 |
| V (archive) | `TurboQuantMSE` | MSE 优化，保重建精度 |
| K (recent) | 无 (fp16) | 全精度 |
| V (recent) | 无 (fp16) | 全精度 |

### D5. Attention 计算

**attention 在反量化后的 latent 空间计算，不是 full K/V 空间。**

```python
# attention_hawp.py L484-500

k_lat_expanded = _repeat_kv(k_lat)    # GQA expand
v_lat_expanded = _repeat_kv(v_lat)

# OPT:
logits = q_lat @ k_lat_expanded^T     # scaling=1.0
softmax(logits)
attn_output_lat = softmax @ v_lat_expanded

# 回投：
attn_output = gamma * (attn_output_lat @ pv_up)    # latent → head_dim
attn_output = o_proj(attn_output)
```

### D6. Shape 流程 (decode 步骤, r_k=r_v=64, recent_window=8)

```
# 输入: 1 个新 token
Q:         [1, 12, 1, 64]
K_new:     [1, 12, 1, 64]
V_new:     [1, 12, 1, 64]

# 投影 (P=I 时不变)
q_lat:     [1, 12, 1, 64]
k_lat_new: [1, 12, 1, 64]
v_lat_new: [1, 12, 1, 64]

# 缓存: recent=7, archive=5 (假设已有 12 个 token)
_quant_cache_append → recent 变为 8 个 token

# recent(8) > window(8)? 否（不 demote）
# 不会触发 demote，除非 > window

# 实际读取：
k_lat = cat([dequant(archive_5tok), recent_8tok]) = [12, 13, 64] → [1, 12, 13, 64]
v_lat = cat([dequant(archive_5tok), recent_8tok]) = [12, 13, 64] → [1, 12, 13, 64]

logits = q_lat[:,:,:1,:] @ k_lat^T:  [1, 12, 1, 13]
softmax:                              [1, 12, 1, 13]
attn_output_lat:                      [1, 12, 1, 64]
attn_output = γ * (lat @ pv_up):      [1, 12, 1, 64]
o_proj:                                [1, 1, 768]
```

---

## E. quant_only 的 attention 如何计算

quant_only 是 hawp_quant 的特例：**不做低秩投影，仅做量化**。r_k = r_v = head_dim，P_k = P_v = I。

### E1. 进入路径

```python
# generate.py L206-234
model = convert_llama_to_hawp(model, r_k=head_dim, r_v=head_dim)  # r_k = head_dim
k_quantizer = build_k_quantizer(cfg, r_k=head_dim)
v_quantizer = build_v_quantizer(cfg, r_v=head_dim)
module.setup_quant_cache(k_quantizer, v_quantizer, recent_window=recent_window)
```

虽然 `_is_low_rank = False`（r_k == head_dim），但 `use_cache_manager = True`，仍进入 `_forward_low_rank()`：

```python
# attention_hawp.py L360-365
if self._is_low_rank or self.use_cache_manager:  # use_cache_manager=True → 进入
    return self._forward_low_rank(...)
```

### E2. 投影 (P = I，恒等)

```python
pk_down = P_k[:, :head_dim] = I[:, :64] = I    # [64, 64]
pv_down = P_v[:, :head_dim] = I                 # [64, 64]
pv_up   = P_v[:head_dim, :] = I                 # [64, 64]

q_lat = Q @ I = Q                               # 不变
k_lat = K @ I = K                               # 不变
v_lat = V @ I = V                               # 不变
```

### E3. 缓存管理

与 hawp_quant 完全一致：
- recent: fp16 存储
- archive: TurboQuant 量化存储
- recent_window 正常生效

### E4. Attention 计算

```python
# 与 hawp_quant 相同路径
k_lat_expanded = _repeat_kv(k_lat)
v_lat_expanded = _repeat_kv(v_lat)

# OPT:
logits = q_lat @ k_lat_expanded^T     # scaling=1.0, Q 已含 0.125
softmax(logits)
attn_output_lat = softmax @ v_lat_expanded

# 回投 (P=I):
attn_output = gamma * (attn_output_lat @ I) = gamma * attn_output_lat

o_proj(attn_output)
```

### E5. 与 baseline 的等价性分析

| 因素 | quant_only | baseline | 等价？ |
|------|-----------|----------|--------|
| Q scaling | 0.125 内含 | 0.125 内含 | 等价 |
| K/V 变换 | 无 (P=I) | 无 | 等价 |
| γ 因子 | `gamma * output` | 无 γ | **不等价**（γ ≠ 1 时） |
| 量化误差 | archive 有量化误差 | 无 | **不等价** |

**关键结论：** quant_only 在 γ=1 且不使用量化（所有 token 都在 recent 中）时等价 baseline。但：
1. γ 初始值为 1.0，如果 `load_projectors` 加载了非 1.0 的 γ，则不等价
2. 一旦有 token 进入 archive，反量化引入误差，不等价
3. scaling 用 `/ sqrt(head_dim)` 而非 `/ sqrt(r_k)`，此处 r_k == head_dim 所以等价

### E6. Shape 流程 (opt-125m, decode 步骤, recent_window=8)

```
# 输入: 1 个新 token
Q:         [1, 12, 1, 64]
K_new:     [1, 12, 1, 64]
V_new:     [1, 12, 1, 64]

# 投影 (P=I，不变)
q_lat:     [1, 12, 1, 64]
k_lat_new: [1, 12, 1, 64]
v_lat_new: [1, 12, 1, 64]

# 缓存: recent=7, archive=5 (假设已有 12 个 token)
_quant_cache_append → recent 变为 8 个 token
recent(8) > window(8)? 否

# 读取缓存:
k_lat = cat([dequant(archive_5tok), recent_8tok]) = [12, 13, 64] → [1, 12, 13, 64]
v_lat = cat([dequant(archive_5tok), recent_8tok]) = [12, 13, 64] → [1, 12, 13, 64]

logits = q_lat[:,:,:1,:] @ k_lat^T:  [1, 12, 1, 13]
softmax:                              [1, 12, 1, 13]
attn_output_lat:                      [1, 12, 1, 64]
attn_output = γ * (lat @ I):          [1, 12, 1, 64]  (γ=1 时)
o_proj:                                [1, 1, 768]
```

---

## F. hawp_quant_all 的 attention 如何计算

hawp_quant_all 与 hawp_quant 的区别仅在于 **recent_window=0**，即所有 token 全部走 archive 量化路径，不存在 fp16 recent 分层。

### F1. 进入路径

```python
# generate.py L183-203
model = convert_llama_to_hawp(model, r_k=r_k, r_v=r_v)
model = model.to(device).eval()

# 加载 projectors
load_projectors(model, projector_dir)

k_quantizer = build_k_quantizer(cfg, r_k=r_k)
v_quantizer = build_v_quantizer(cfg, r_v=r_v)

# 关键: recent_window=0
for module in model.modules():
    if isinstance(module, HAWPAttention):
        module.setup_quant_cache(k_quantizer, v_quantizer, recent_window=0)
```

### F2. 缓存管理差异

`_forward_low_rank()` 中的缓存分支（L450-469）：

```python
if self.use_cache_manager and self._tq_k_quantizer is not None:
    cache_was_empty = (self._quant_recent_k is None and self._quant_archive_k_qx is None)

    if self.recent_window == 0:
        # ← hawp_quant_all 走这个分支
        self._quant_cache_append_to_archive(k_lat, v_lat)   # 直接入 archive，不经 recent
        if not cache_was_empty:
            k_cached, v_cached = self._quant_cache_get_kv()
            k_lat = k_cached.unsqueeze(0).to(q_lat.device, q_lat.dtype)
            v_lat = v_cached.unsqueeze(0).to(q_lat.device, q_lat.dtype)
            kv_from_cache = True
    else:
        # ← hawp_quant 走这个分支 (recent_window > 0)
        self._quant_cache_append(k_lat, v_lat)              # 先入 recent
        if self._quant_recent_k.shape[1] > self.recent_window:
            self._quant_cache_demote()                       # recent 满后降级到 archive
        ...
```

### F3. `_quant_cache_append_to_archive` 逻辑 (L179-202)

与 `_quant_cache_demote` 不同，`_quant_cache_append_to_archive` **增量追加**而非全量重新量化：

```python
def _quant_cache_append_to_archive(self, k_lat, v_lat):
    k_new = k_lat[0].detach()        # [nkv, T_new, r_k]
    v_new = v_lat[0].detach()

    if self._quant_archive_k_raw is not None:
        # 拼接 raw
        self._quant_archive_k_raw = cat([existing_raw, k_new], dim=1)
        self._quant_archive_v_raw = cat([existing_raw, v_new], dim=1)

    nkv, T, rk = self._quant_archive_k_raw.shape

    if T == n_new:   # archive 之前为空，这是第一批 → 全量量化
        k_flat = self._quant_archive_k_raw.reshape(nkv * T, rk).float()
        v_flat = self._quant_archive_v_raw.reshape(nkv * T, rv).float()
        self._quant_archive_k_qx = k_quantizer.quantize(k_flat)
        self._quant_archive_v_qx = v_quantizer.quantize(v_flat)
    else:            # archive 已有内容 → 增量量化并 merge
        k_new_flat = k_new.reshape(nkv * n_new, rk).float()
        v_new_flat = v_new.reshape(nkv * n_new, rv).float()
        new_k_qx = k_quantizer.quantize(k_new_flat)
        new_v_qx = v_quantizer.quantize(v_new_flat)
        self._quant_archive_k_qx = self._merge_quantized(old_qx, new_k_qx, ...)
        self._quant_archive_v_qx = self._merge_quantized(old_qx, new_v_qx, ...)
```

**关键：** 增量量化只对新 token 做量化，旧 token 的量化数据不变。这与 `_quant_cache_demote` 的全量重新量化不同，**可能导致量化精度差异**（因为旧+新一起量化可以利用全局统计量，而增量方式只能用局部统计量）。

### F4. 投影与 Attention 计算

投影和 attention 计算与 hawp_quant 完全一致：

```python
q_lat = query_states @ pk_down
k_lat = key_states @ pk_down
v_lat = value_states @ pv_down

# 缓存读取（全部从 archive 反量化）:
k_lat = dequantize(archive_k_qx).unsqueeze(0)    # 无 recent 部分
v_lat = dequantize(archive_v_qx).unsqueeze(0)

k_lat_expanded = _repeat_kv(k_lat)
v_lat_expanded = _repeat_kv(v_lat)

# OPT:
logits = q_lat @ k_lat_expanded^T     # scaling=1.0
softmax(logits)
attn_output_lat = softmax @ v_lat_expanded

attn_output = gamma * (attn_output_lat @ pv_up)
o_proj(attn_output)
```

### F5. 与 hawp_quant 的关键区别

| 方面 | hawp_quant | hawp_quant_all |
|------|-----------|----------------|
| recent_window | > 0（如 8, 64） | **0** |
| 最新 token 存储 | fp16 (recent) | 量化 (archive) |
| 旧 token 入 archive 方式 | `_quant_cache_demote`（全量重新量化） | `_quant_cache_append_to_archive`（增量量化） |
| 最新 token 精度 | **全精度** | **量化精度** |
| KV 读取时 recent 部分 | 有（fp16） | **无** |
| 内存占用 | 较高（recent fp16） | **最低**（全部量化） |
| 最新 token attention 精度 | 最高 | 最低（量化损失） |

### F6. Shape 流程 (opt-125m, r_k=r_v=50, decode 步骤, recent_window=0)

```
# 输入: 1 个新 token (已有 12 个 token 在 archive)
Q:         [1, 12, 1, 64]
K_new:     [1, 12, 1, 64]
V_new:     [1, 12, 1, 64]

# 投影
q_lat:     [1, 12, 1, 50]
k_lat_new: [1, 12, 1, 50]
v_lat_new: [1, 12, 1, 50]

# 缓存: recent_window=0 → 直接入 archive
_quant_cache_append_to_archive(k_lat, v_lat)
# archive 增量: 12 → 13 个 token

# 读取缓存:
k_lat = dequantize(archive_13tok).unsqueeze(0) = [1, 12, 13, 50]
v_lat = dequantize(archive_13tok).unsqueeze(0) = [1, 12, 13, 50]

logits = q_lat[:,:,:1,:] @ k_lat^T:  [1, 12, 1, 13]
softmax:                              [1, 12, 1, 13]
attn_output_lat:                      [1, 12, 1, 50]
attn_output = γ * (lat @ pv_up):      [1, 12, 1, 64]
o_proj:                                [1, 1, 768]
```

---

## G. hawp_quant_sched 的 attention 如何计算

### G1. 三态处理时机

三态在 **attention 计算之前** 由 coordinator 处理：

```
generate_hawp_quant():
  coordinator.on_prefill(prompt_len)
    → scheduler.on_tokens(prompt_len)
    → _apply_drop()
      → drop_count = scheduler.compute_drop_count()
      → for layer in HAWPAttention layers:
          layer.drop_oldest_from_archive(drop_count)

  coordinator.on_new_token()
    → scheduler.on_new_token()
    → _apply_drop()
```

### G2. DROP 的实际效果

```python
# attention_hawp.py L254-275
def drop_oldest_from_archive(self, n):
    # 从 _quant_archive_k_raw 切掉前 n 个 token
    self._quant_archive_k_raw = self._quant_archive_k_raw[:, n:, :]
    self._quant_archive_v_raw = self._quant_archive_v_raw[:, n:, :]
    # 对剩余的重新量化
    self._quant_archive_k_qx = k_quantizer.quantize(remaining_k_flat)
    self._quant_archive_v_qx = v_quantizer.quantize(remaining_v_flat)
```

### G3. 各状态 token 的处理

| 状态 | 存储位置 | 格式 | 是否参与 attention |
|------|---------|------|------------------|
| HIGH | `_quant_recent_k/v` | fp16 | 是 |
| LOW | `_quant_archive_k/v` | 量化后 | 是，反量化后拼接 |
| DROP | 已从 archive 删除 | 无 | **否，完全不参与** |

### G4. 最终 K/V 张量组装

与 hawp_quant 完全相同的 `_quant_cache_get_kv()`：

```python
K = cat([dequantize(archive_k_qx), recent_k.float()], dim=1)  # [nkv, T_archive+T_recent, r_k]
V = cat([dequantize(archive_v_qx), recent_v.float()], dim=1)  # [nkv, T_archive+T_recent, r_v]
```

被 DROP 的 token 已从 `_quant_archive_k_raw` 中物理删除，对应的量化数据也已重新生成（不含被删 token），**完全不参与 attention 计算**。

### G5. 调度器决策逻辑

```
TokenBudgetScheduler.rebalance():
  n_high = min(recent_window, total_tokens)
  n_low  = min(total_tokens - n_high, total_budget - n_high)
  n_drop = total_tokens - n_high - n_low
```

示例 (total_budget=32, recent_window=8, total_tokens=37):
```
n_high = 8
n_low  = min(29, 24) = 24
n_drop = 37 - 8 - 24 = 5
```

---

## H. 最终结论

### 1. 当前代码是"直接在 latent 空间做 attention"还是"重建回 full K/V 后再做 attention"？

**直接在 latent 空间做 attention。**

证据：
- `q_lat = Q @ pk_down` (L444)，`k_lat = K @ pk_down` (L445)，维度都是 r_k
- `logits = q_lat @ k_lat^T` (L414/489)，在 r_k 维度做内积
- `attn_output_lat = softmax @ v_lat` (L423/498)，在 r_v 维度做 value 聚合
- 只在最后一步 `attn_output = γ * (attn_output_lat @ pv_up)` (L500) 回投到 head_dim

### 2. K 路径和 V 路径是否采用了不同处理方式？

**是，在量化层面不同，在 attention 计算层面相同。**

- **量化不同**：K 用 `TurboQuantProd`（MSE + 1-bit 残差保内积），V 用 `TurboQuantMSE`（MSE 保重建）
- **attention 计算**：K 和 V 在同一 latent 空间，logits = q_lat @ k_lat^T，value = softmax @ v_lat
- **回投不同**：V 路径通过 pv_up 回投并乘 γ；K 路径只在 logits 内积中使用，不需要回投

### 3. full-rank 时是否严格等价 baseline？

**不一致，存在差异。**

**(a) hawp_only，r_k == head_dim，full-rank 分支：**

```
K = _apply_pk(K)  → K @ P_k @ P_k^T
V = _apply_pv(V)  → γ * V @ P_v @ P_v^T
```

当 P_k ≠ I（load_projectors 后），K/V 被 full-rank 变换，**不等价 baseline**。
当 P_k = I 且 requires_grad=False 时，`_apply_pk` 直接 return k，**等价 baseline**。

**(b) hawp_quant，r_k == head_dim，_forward_low_rank 分支：**

```
q_lat = Q @ I = Q
k_lat = K @ I = K
v_lat = V @ I = V
logits = q_lat @ k_lat^T (scaling=1.0, Q 含 0.125)
attn_output = γ * (attn_output_lat @ I) = γ * attn_output_lat
```

scaling 等价（Q 已含 0.125），但 **γ 因子** 在 baseline 中不存在。当 γ ≠ 1.0 时输出不等价。

### 4. 为什么 hawp_only 与 baseline 不一致？

**根本原因：latent KV 存入 HF cache 时维度不匹配。**

当 `r_k < head_dim` 时，hawp_only 走 `_forward_low_rank()`，其中 `use_cache_manager=False`，走 HF KV cache 分支（L470-476）：

```python
k_lat, v_lat = past_key_value.update(k_lat, v_lat, self.layer_idx, cache_kwargs)
```

HF 的 KV cache 按 `head_dim=64` 设计，但存入的是 `r_k=50` 维的 k_lat。后续 cache 读取时，attention mask 和 cache 机制无法正确处理维度变化，**导致 attention 计算错误**。

**当 r_k == head_dim 时**，维度匹配，不存在此问题。但此时 `_apply_pk` / `_apply_pv` 的 `P @ P^T` 变换仍可能引入差异。

---

## 不一致点汇总

| # | 位置 | 描述 | 影响 |
|---|------|------|------|
| 1 | `_forward_low_rank` L470-476 | hawp_only 存 50 维 k_lat 到 HF cache（按 64 维设计） | r_k < head_dim 时输出崩溃 |
| 2 | `_apply_pk` L518-520 | full-rank 分支做 K @ P @ P^T 变换 | P≠I 时不等价 baseline |
| 3 | `_forward_low_rank` L500 | 输出乘 γ 因子 | baseline 无 γ，γ≠1 时不等价 |
| 4 | `_opt_attn_forward` L406-410 | scaling=1.0，依赖 Q 已含 0.125 | 对 OPT 等价，但对 Llama 等模型，L493 用 `/ sqrt(head_dim)` 而非 `/ sqrt(r_k)` |
| 5 | L493 | `/ math.sqrt(self.head_dim)` 用于 latent attention | latent 空间维度是 r_k 不是 head_dim，scale 应为 1/√r_k |
| 6 | `_quant_cache_append_to_archive` L179-202 | 增量量化只对新 token 独立量化，`_merge_quantized` 简单拼接 | 与 `_quant_cache_demote` 全量重新量化结果不同，hawp_quant_all 精度可能劣于 hawp_quant |
| 7 | `drop_least_important_from_archive` L277-301 | 按 K latent norm 排序决定丢弃 token，但丢弃时从 K/V archive 同步删除相同位置 | V 的 norm 分布可能与 K 不同，按 K norm 丢弃可能丢失 V 重要信息 |
| 8 | `_forward_low_rank` L488 | OPT 模式下，`kv_from_cache=True` 时 `mask_for_opt=None`，跳过 attention_mask | decode 阶段生成时未传 mask 可能导致非因果 attention（但逐步解码只有 1 个 q token，实际无影响） |

---

## I. 六模式全量对比表

| 维度 | baseline | hawp_only | quant_only | hawp_quant | hawp_quant_all | hawp_quant_sched |
|------|----------|-----------|------------|------------|----------------|------------------|
| **Attention 类** | OPTSdpaAttention | HAWPAttention | HAWPAttention | HAWPAttention | HAWPAttention | HAWPAttention |
| **低秩投影** | 无 | 有 (r_k/r_v) | **无** (r_k=head_dim) | 有 (r_k/r_v) | 有 (r_k/r_v) | 有 (r_k/r_v) |
| **forward 路径** | HF 原生 | `_is_low_rank` ? `_forward_low_rank` : `forward` | `_forward_low_rank` (use_cache_manager) | `_forward_low_rank` | `_forward_low_rank` | `_forward_low_rank` |
| **KV 缓存** | HF Cache | HF Cache | quant cache | quant cache | quant cache | quant cache |
| **Recent 窗口** | N/A | N/A | >0 | >0 | **0** | >0 |
| **Archive 量化** | 无 | 无 | TurboQuant | TurboQuant | TurboQuant | TurboQuant |
| **Token DROP** | 无 | 无 | 无 | 无 | 无 | **有** |
| **Generation 方式** | model.generate() | model.generate() | generate_hawp_quant() | generate_hawp_quant() | generate_hawp_quant() | generate_hawp_quant() |
| **use_cache** | True | True | False | False | False | False |
| **K 量化器** | 无 | 无 | TurboQuantProd | TurboQuantProd | TurboQuantProd | TurboQuantProd |
| **V 量化器** | 无 | 无 | TurboQuantMSE | TurboQuantMSE | TurboQuantMSE | TurboQuantMSE |
| **Attention 空间** | full [head_dim] | r_k < head_dim 时 latent | full [head_dim] | latent [r_k] | latent [r_k] | latent [r_k] |
| **γ 因子** | 无 | `_apply_pv` 中 | `_forward_low_rank` 中 | `_forward_low_rank` 中 | `_forward_low_rank` 中 | `_forward_low_rank` 中 |
| **Scaling** | SDPA 内部 | `_eager_attn` scaling=1.0 | `_eager_attn` scaling=1.0 | `_eager_attn` scaling=1.0 | `_eager_attn` scaling=1.0 | `_eager_attn` scaling=1.0 |
| **与 baseline 等价条件** | — | P=I, γ=1, r_k=head_dim | γ=1, archive 为空 | P=I, γ=1, archive 为空 | P=I, γ=1, 无量化误差 | 不可能（调度导致 token 丢弃） |

### Attention 计算核心路径一览

```
baseline:
  Q·K^T (full-dim) → softmax → ·V (full-dim) → o_proj

hawp_only (r_k < head_dim):
  (Q·P_k↓)·(K·P_k↓)^T (latent-dim) → softmax → ·(V·P_v↓) (latent-dim) → γ·(·P_v↑) → o_proj

hawp_only (r_k == head_dim, P_k=I, P_v=I):
  Q·K^T (full-dim) → softmax → ·V (full-dim) → o_proj   [≈ baseline, 但 γ=1 时才等价]

quant_only:
  Q·K^T (full-dim, cached+dequant) → softmax → ·V (full-dim, cached+dequant) → γ· → o_proj

hawp_quant:
  (Q·P_k↓)·(K·P_k↓)^T (latent-dim, cached+dequant) → softmax → ·(V·P_v↓) (latent-dim, cached+dequant) → γ·(·P_v↑) → o_proj

hawp_quant_all:
  同 hawp_quant，但所有 K/V 都经量化（无 recent fp16），且 archive 增量追加而非全量重新量化

hawp_quant_sched:
  同 hawp_quant，但 archive 会按调度策略丢弃最老/最不重要 token
```

### KV 缓存内存模型

| 模式 | 单层单 head 存储 | 压缩比 vs baseline |
|------|-----------------|-------------------|
| baseline | `2 × T × head_dim × 2B` (fp16) | 1× |
| hawp_only (r_k=50) | `2 × T × 50 × 2B` (fp16, 但有 BUG) | 50/64 ≈ 0.78× |
| quant_only | archive: `T × head_dim × bits/8 + overhead`; recent: `T × head_dim × 2B` | 视 bits 而定 |
| hawp_quant (r_k=50) | archive: `T × 50 × bits/8 + overhead`; recent: `W × 50 × 2B` | 最优平衡 |
| hawp_quant_all (r_k=50) | `T × 50 × bits/8 + overhead` (无 recent fp16) | 最低内存 |
| hawp_quant_sched (r_k=50) | archive ≤ budget; recent ≤ window; DROP 部分 0 开销 | 受 budget 上限控制 |
