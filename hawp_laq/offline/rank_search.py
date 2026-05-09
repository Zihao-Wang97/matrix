from __future__ import annotations

import math
import warnings
from pathlib import Path

import torch

from hawp_laq.offline.projector_trainer import ProjectorTrainer
from hawp_laq.utils.io import load_pt, save_json


# ------------------------------------------------------------------
# Candidate-pair builder
# ------------------------------------------------------------------


def build_rank_pairs(
    rank_candidates: list | None = None,
    r_k_candidates: list | None = None,
    r_v_candidates: list | None = None,
    rank_pair_candidates: list | None = None,
    head_dim: int = 64,
) -> list[tuple[int, int]]:
    """Build a validated, deduplicated, sorted list of (r_k, r_v) pairs.

    Priority (first non-empty wins):
      1. ``rank_pair_candidates`` — explicit per-pair list
      2. ``r_k_candidates`` × ``r_v_candidates`` — Cartesian product
      3. ``rank_candidates`` — legacy symmetric list → [(r, r), ...]

    Each pair is validated against ``head_dim``.  Out-of-range values raise
    ``ValueError`` (fail-fast, no silent filtering).
    """

    # ---- resolve raw pairs ----
    if rank_pair_candidates:
        raw_pairs: list[tuple[int, int]] = []
        for item in rank_pair_candidates:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                raw_pairs.append((int(item[0]), int(item[1])))
            else:
                raise ValueError(
                    f"rank_pair_candidates item must be [r_k, r_v], got {item!r}"
                )
    elif r_k_candidates and r_v_candidates:
        raw_pairs = []
        for rk in r_k_candidates:
            rk = int(rk)
            for rv in r_v_candidates:
                rv = int(rv)
                raw_pairs.append((rk, rv))
    elif rank_candidates:
        raw_pairs = [(int(r), int(r)) for r in rank_candidates]
    else:
        return []

    # ---- validate ----
    for rk, rv in raw_pairs:
        if not (1 <= rk <= head_dim):
            raise ValueError(
                f"r_k={rk} out of range [1, head_dim={head_dim}]"
            )
        if not (1 <= rv <= head_dim):
            raise ValueError(
                f"r_v={rv} out of range [1, head_dim={head_dim}]"
            )

    # ---- deduplicate (preserve stable order) ----
    seen: set[tuple[int, int]] = set()
    deduped: list[tuple[int, int]] = []
    for p in raw_pairs:
        if p not in seen:
            seen.add(p)
            deduped.append(p)

    return deduped


# ------------------------------------------------------------------
# Dimension inference & multi-head conversion helpers
# ------------------------------------------------------------------


def infer_calib_dims(
    x: torch.Tensor,
    n_heads: int,
    meta: dict | None = None,
) -> tuple[int, int]:
    """Infer (d_model, head_dim) from calibration tensor *x*.

    Supports three input layouts:
      - ``[B, T, d_model]``
      - ``[B, H, T, d_h]``
      - ``[B*H, T, d_h]``

    Priority: use ``meta`` keys (``d_model`` / ``hidden_size`` / ``model_dim``)
    when available; fall back to heuristic only when meta is absent.
    """
    if n_heads is None or n_heads <= 0:
        raise ValueError("n_heads must be a positive integer")

    meta = meta or {}
    model_d_model = (
        meta.get("d_model")
        or meta.get("hidden_size")
        or meta.get("model_dim")
    )

    if model_d_model is not None and model_d_model % n_heads != 0:
        raise ValueError(
            f"meta d_model={model_d_model} is not divisible by n_heads={n_heads}"
        )

    head_dim_model = model_d_model // n_heads if model_d_model is not None else None

    if x.ndim == 4:
        dh = x.shape[-1]
        if model_d_model is not None:
            if dh != head_dim_model:
                raise ValueError(
                    f"4-D tensor last dim {dh} != meta head_dim "
                    f"{head_dim_model} (d_model={model_d_model}, n_heads={n_heads})"
                )
            return model_d_model, head_dim_model
        return dh * n_heads, dh

    if x.ndim == 3:
        D = x.shape[-1]
        if model_d_model is not None:
            if D == model_d_model:
                return model_d_model, head_dim_model
            if D == head_dim_model:
                return model_d_model, head_dim_model
            raise ValueError(
                f"3-D tensor last dim {D} matches neither d_model={model_d_model} "
                f"nor head_dim={head_dim_model}"
            )
        if D % n_heads == 0:
            warnings.warn(
                f"infer_calib_dims: no meta d_model; fallback assuming "
                f"[B,T,d_model] with d_model={D}, head_dim={D // n_heads}",
                UserWarning,
                stacklevel=2,
            )
            return D, D // n_heads
        raise ValueError(
            f"3-D tensor last dim {D} is not divisible by n_heads={n_heads} "
            f"and no meta d_model provided"
        )

    raise ValueError(
        f"Unsupported calibration tensor ndim={x.ndim}, expected 3 or 4"
    )


def to_mh_for_signal(
    x: torch.Tensor,
    n_heads: int,
    d_model: int,
    head_dim: int,
) -> torch.Tensor:
    """Convert Q/K/V calibration tensor to ``[B_eff, H_eff, T, head_dim]``.

    Supported input layouts:
      - ``[B, T, d_model]`` → reshape to ``[B, n_heads, T, head_dim]``
      - ``[B, H, T, d_h]``  → pass-through (must have d_h == head_dim)
      - ``[B*H, T, d_h]``   → unsqueeze to ``[B*H, 1, T, head_dim]``
    """
    if x.ndim == 4:
        if x.shape[-1] != head_dim:
            raise ValueError(
                f"4-D tensor last dim {x.shape[-1]} != head_dim {head_dim}"
            )
        return x

    if x.ndim == 3:
        D = x.shape[-1]
        if D == d_model:
            B, T, _ = x.shape
            return x.reshape(B, T, n_heads, head_dim).transpose(1, 2)
        if D == head_dim:
            return x.unsqueeze(1)
        raise ValueError(
            f"3-D tensor last dim {D} matches neither d_model={d_model} "
            f"nor head_dim={head_dim}"
        )

    raise ValueError(
        f"Unsupported tensor ndim={x.ndim}, expected 3 or 4"
    )


# ------------------------------------------------------------------
# Single candidate evaluation
# ------------------------------------------------------------------


def _evaluate_rank(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r_k: int,
    r_v: int,
    d_model: int,
    n_heads: int,
    *,
    n_steps: int = 200,
    lr: float = 1e-3,
    orthogonalize_every: int = 10,
    w_logits: float = 1.0,
    w_attn: float = 1.0,
    w_value: float = 0.5,
    device: str = "cpu",
    warmup_steps: int = 30,
    row_batch_size: int | None = None,
    lr_pk: float = 5e-3,
    lr_pv: float = 5e-3,
    lr_xi: float = 1e-2,
    beta1: float = 0.9,
    beta2: float = 0.99,
    grad_clip: float = 1.0,
    lambda_z: float = 1.0,
    lambda_o: float = 2.0,
    lambda_v: float = 0.05,
    lambda_topk: float = 0.0,
    lambda_kl: float = 0.0,
    lambda_logit_topm: float = 0.0,
    topk_k: int = 8,
    hard_neg_m: int = 32,
    kl_top_m: int = 64,
    topk_margin: float = 0.05,
    topk_loss_start_after_warmup: bool = True,
    topk_metric_ks: list[int] | tuple[int, ...] = (5, 10),
    eval_every: int = 50,
    early_stopping: bool = True,
    patience: int = 5,
    min_delta: float = 1e-4,
    min_delta_mode: str = "relative",
    gamma_min: float = 1e-4,
    logit_scale_mode: str = "rk",
    eps_loss: float = 1e-8,
    adam_eps: float = 1e-8,
    optimizer: str = "riemannian_adam",
) -> dict:
    trainer = ProjectorTrainer(
        d_model=d_model,
        rank_k=r_k,
        rank_v=r_v,
        n_heads=n_heads,
        lr=lr,
        orthogonalize_every=orthogonalize_every,
        w_logits=w_logits,
        w_attn=w_attn,
        w_value=w_value,
        device=device,
    )
    result = trainer.train_one_group(
        q, k, v,
        n_steps=n_steps,
        warmup_steps=warmup_steps,
        row_batch_size=row_batch_size,
        lr_pk=lr_pk,
        lr_pv=lr_pv,
        lr_xi=lr_xi,
        beta1=beta1,
        beta2=beta2,
        grad_clip=grad_clip,
        lambda_z=lambda_z,
        lambda_o=lambda_o,
        lambda_v=lambda_v,
        lambda_topk=lambda_topk,
        lambda_kl=lambda_kl,
        lambda_logit_topm=lambda_logit_topm,
        topk_k=topk_k,
        hard_neg_m=hard_neg_m,
        kl_top_m=kl_top_m,
        topk_margin=topk_margin,
        topk_loss_start_after_warmup=topk_loss_start_after_warmup,
        topk_metric_ks=topk_metric_ks,
        eval_every=eval_every,
        early_stopping=early_stopping,
        patience=patience,
        min_delta=min_delta,
        min_delta_mode=min_delta_mode,
        gamma_min=gamma_min,
        logit_scale_mode=logit_scale_mode,
        eps_loss=eps_loss,
        adam_eps=adam_eps,
        optimizer=optimizer,
    )

    # Prefer Riemannian-Adam best_calib_* fields; fall back to metrics[-1]
    # for legacy trainer (Adam+orthogonalize).
    if "best_calib_total" in result:
        total = result["best_calib_total"]
        logits = result["best_calib_logits"]
        attn = result["best_calib_attn"]
        value = result["best_calib_value"]
    else:
        m = result["metrics"]
        total = m["total"][-1]
        logits = m["logits"][-1]
        attn = m["attn"][-1]
        value = m["value"][-1]

    return {
        "r_k": r_k,
        "r_v": r_v,
        "logit_scale_mode": logit_scale_mode,
        "best_calib_total": total,
        "best_calib_logits": logits,
        "best_calib_attn": attn,
        "best_calib_value": value,
        "best_calib_topk": result.get("best_calib_topk", 0.0),
        "best_calib_kl_topm": result.get("best_calib_kl_topm", 0.0),
        "best_calib_logit_topm": result.get("best_calib_logit_topm", 0.0),
        "best_calib_top_recalls": result.get("best_calib_top_recalls", {}),
        "best_step": result.get("best_step", n_steps),
        "actual_steps": result.get("actual_steps", n_steps),
        "stopped_early": result.get("stopped_early", False),
        "rank_cost": r_k + r_v,
        "p_k_shape": tuple(result["p_k"].shape),
        "p_v_shape": tuple(result["p_v"].shape),
    }


def _component_pass(
    result_loss: float,
    baseline_loss: float,
    relative_tolerance: float,
    abs_tolerance: float,
    eps: float = 1e-8,
) -> bool:
    if baseline_loss >= eps:
        return result_loss <= baseline_loss * (1.0 + relative_tolerance)
    return result_loss <= abs_tolerance


# ------------------------------------------------------------------
# Signal-normalized selection helpers
# ------------------------------------------------------------------


def get_layer_tolerance_scale(
    layer_idx: int, layer_tolerance_scale: list | None
) -> float:
    """Return the tolerance multiplier for a given layer."""
    if layer_tolerance_scale is None:
        return 1.0
    for rule in layer_tolerance_scale:
        if layer_idx in rule["layers"]:
            return float(rule["scale"])
    return 1.0


def get_layer_rank_floor(
    layer_idx: int, layer_rank_floor: list | None
) -> tuple[int, int]:
    """Return (min_r_k, min_r_v) for a given layer."""
    if layer_rank_floor is None:
        return (1, 1)
    for rule in layer_rank_floor:
        if layer_idx in rule["layers"]:
            return (
                int(rule.get("min_r_k", 1)),
                int(rule.get("min_r_v", 1)),
            )
    return (1, 1)


def compute_signal_scales(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_heads: int,
    d_model: int | None = None,
    head_dim: int | None = None,
    row_batch_size: int | None = None,
    batch_chunk_size: int = 4,
) -> dict[str, float]:
    """Compute per-component signal strengths from full-precision Q/K/V.

    Supports three calibration layouts: ``[B,T,d_model]``,
    ``[B,H,T,d_h]``, and ``[B*H,T,d_h]``.

    Applies a causal mask to match the optimizer's objective, so signal
    magnitudes are consistent with the training loss space.

    Returns mean squared magnitude of: logits, attention weights, value output.
    These are used as signal references for normalized error thresholds.
    """
    if d_model is None or head_dim is None:
        _dm, _hd = infer_calib_dims(q, n_heads)
        d_model = d_model or _dm
        head_dim = head_dim or _hd

    q_mh = to_mh_for_signal(q, n_heads, d_model, head_dim)
    k_mh = to_mh_for_signal(k, n_heads, d_model, head_dim)
    v_mh = to_mh_for_signal(v, n_heads, d_model, head_dim)

    if not (q_mh.shape == k_mh.shape == v_mh.shape):
        raise ValueError(
            f"Shape mismatch after to_mh_for_signal: q={q_mh.shape}, "
            f"k={k_mh.shape}, v={v_mh.shape}"
        )

    T = q_mh.shape[-2]
    if row_batch_size is not None and row_batch_size < T:
        row_idx = torch.linspace(0, T - 1, steps=row_batch_size, device=q_mh.device).long()
        q_rows = row_batch_size
    else:
        row_idx = None
        q_rows = T

    row_positions = row_idx if row_idx is not None else torch.arange(T, device=q_mh.device)
    causal = torch.arange(T, device=q_mh.device).unsqueeze(0) <= row_positions.unsqueeze(1)
    batch_chunk_size = max(1, int(batch_chunk_size))

    logits_sum = torch.zeros((), device=q_mh.device, dtype=torch.float32)
    attn_sum = torch.zeros((), device=q_mh.device, dtype=torch.float32)
    value_sum = torch.zeros((), device=q_mh.device, dtype=torch.float32)
    valid_count = 0
    value_count = 0

    for start in range(0, q_mh.shape[0], batch_chunk_size):
        end = min(start + batch_chunk_size, q_mh.shape[0])
        qc = q_mh[start:end]
        kc = k_mh[start:end]
        vc = v_mh[start:end]
        if row_idx is not None:
            qc = qc[:, :, row_idx, :]

        logits_fp = torch.matmul(qc, kc.transpose(-2, -1)) / math.sqrt(head_dim)
        causal_exp = causal.unsqueeze(0).unsqueeze(0).expand_as(logits_fp)
        logits_masked = logits_fp.masked_fill(~causal_exp, float("-inf"))
        attn_fp = torch.softmax(logits_masked, dim=-1)
        value_fp = torch.matmul(attn_fp, vc)

        logits_sum = logits_sum + logits_fp[causal_exp].float().pow(2).sum()
        attn_sum = attn_sum + attn_fp[causal_exp].float().pow(2).sum()
        value_sum = value_sum + value_fp.float().pow(2).sum()
        valid_count += int(causal.sum()) * logits_fp.shape[0] * logits_fp.shape[1]
        value_count += value_fp.numel()

    signal_logits = logits_sum.item() / max(valid_count, 1)
    signal_attn = attn_sum.item() / max(valid_count, 1)
    signal_value = value_sum.item() / max(value_count, 1)

    return {
        "signal_logits": signal_logits,
        "signal_attn": signal_attn,
        "signal_value": signal_value,
    }


def _signal_normalized_pass(
    result: dict,
    signal_scales: dict[str, float],
    logits_signal_tolerance: float,
    attn_signal_tolerance: float,
    value_signal_tolerance: float,
    layer_scale: float = 1.0,
    eps: float = 1e-8,
) -> bool:
    """Judge a candidate pair via signal-normalized errors.

    Writes ``normalized_*_error`` and ``*_pass`` keys into *result*.
    Returns ``all_pass``.
    """
    sig_logits = max(signal_scales["signal_logits"], eps)
    sig_attn = max(signal_scales["signal_attn"], eps)
    sig_value = max(signal_scales["signal_value"], eps)

    n_logits = float(result["best_calib_logits"]) / sig_logits
    n_attn = float(result["best_calib_attn"]) / sig_attn
    n_value = float(result["best_calib_value"]) / sig_value

    tol_logits = logits_signal_tolerance * layer_scale
    tol_attn = attn_signal_tolerance * layer_scale
    tol_value = value_signal_tolerance * layer_scale

    result["normalized_logits_error"] = n_logits
    result["normalized_attn_error"] = n_attn
    result["normalized_value_error"] = n_value
    result["logits_pass"] = n_logits <= tol_logits
    result["attn_pass"] = n_attn <= tol_attn
    result["value_pass"] = n_value <= tol_value
    result["all_pass"] = (
        result["logits_pass"]
        and result["attn_pass"]
        and result["value_pass"]
    )
    return result["all_pass"]


# ------------------------------------------------------------------
# Per-layer rank search
# ------------------------------------------------------------------


def _print_header(c1: str, c2: str, c3: str) -> None:
    print(f"  {'r_k':>4} {'r_v':>4}  {c1:>10}  {c2:>10}  {c3:>10}  {'cost':>6}  {'result':>8}")


def _print_row(r: dict, c1, c2, c3) -> None:
    tag = "PASS" if r.get("all_pass", False) else "REJECT"
    print(
        f"  {r['r_k']:>4d} {r['r_v']:>4d}"
        f"  {str(c1):>10s}"
        f"  {str(c2):>10s}"
        f"  {str(c3):>10s}"
        f"  {r['rank_cost']:>6d}"
        f"  {tag:>8}"
    )


def search_rank_per_layer(
    calib_dir: str | Path,
    rank_pairs: list[tuple[int, int]] | None = None,
    # -- backward-compat: old callers passed a list[int] --
    rank_candidates: list[int] | None = None,
    n_steps: int = 1500,
    lr: float = 1e-3,
    orthogonalize_every: int = 10,
    w_logits: float = 1.0,
    w_attn: float = 1.0,
    w_value: float = 0.5,
    device: str = "cpu",
    output_dir: str | Path | None = None,
    relative_tolerance: float = 0.10,
    logits_abs_tolerance: float = 1e-6,
    attn_abs_tolerance: float = 1e-5,
    value_abs_tolerance: float = 1e-4,
    # -- signal-normalized selection --
    selection_mode: str = "constraint",
    logits_signal_tolerance: float = 0.01,
    attn_signal_tolerance: float = 0.01,
    value_signal_tolerance: float = 0.02,
    layer_tolerance_scale: list | None = None,
    layer_rank_floor: list | None = None,
    # -- Riemannian-Adam optimizer params (forwarded to train_one_group) --
    warmup_steps: int = 30,
    row_batch_size: int | None = None,
    lr_pk: float = 5e-3,
    lr_pv: float = 5e-3,
    lr_xi: float = 1e-2,
    beta1: float = 0.9,
    beta2: float = 0.99,
    grad_clip: float = 1.0,
    lambda_z: float = 1.0,
    lambda_o: float = 2.0,
    lambda_v: float = 0.05,
    lambda_topk: float = 0.0,
    lambda_kl: float = 0.0,
    lambda_logit_topm: float = 0.0,
    topk_k: int = 8,
    hard_neg_m: int = 32,
    kl_top_m: int = 64,
    topk_margin: float = 0.05,
    topk_loss_start_after_warmup: bool = True,
    topk_metric_ks: list[int] | tuple[int, ...] = (5, 10),
    eval_every: int = 50,
    early_stopping: bool = True,
    patience: int = 5,
    min_delta: float = 1e-4,
    min_delta_mode: str = "relative",
    gamma_min: float = 1e-4,
    logit_scale_mode: str = "rk",
    eps_loss: float = 1e-8,
    adam_eps: float = 1e-8,
    optimizer: str = "riemannian_adam",
) -> dict[int, tuple[int, int]]:

    # ---- resolve rank_pairs (backward-compat) ----
    _is_legacy = False
    if rank_pairs is None:
        if rank_candidates is not None:
            rank_pairs = [(int(r), int(r)) for r in rank_candidates]
            _is_legacy = True
        else:
            raise ValueError(
                "Either rank_pairs or rank_candidates must be provided"
            )

    _valid_modes = {"constraint", "signal_normalized", "total", "logits", "attn_value_abs"}
    if selection_mode not in _valid_modes:
        raise ValueError(
            f"Unknown selection_mode={selection_mode!r}. "
            f"Supported: {sorted(_valid_modes)}"
        )

    calib_dir = Path(calib_dir)
    meta = load_pt(calib_dir / "meta.pt")
    n_layers = meta.get("n_layers", 0)
    n_heads = meta.get("n_heads")
    if n_heads is None:
        model_id = meta.get("model_id")
        if model_id is None:
            raise ValueError(
                "Cannot infer n_heads: meta.pt must contain either "
                "'n_heads' or 'model_id'."
            )
        from transformers import AutoConfig
        cfg_auto = AutoConfig.from_pretrained(model_id)
        n_heads = cfg_auto.num_attention_heads
    if n_heads is None or n_heads <= 0:
        raise ValueError(
            "n_heads must be a positive integer; "
            "set meta['n_heads'] or provide meta['model_id']"
        )

    if n_layers == 0:
        for p in sorted(calib_dir.glob("layer_*.pt")):
            idx = int(p.stem.split("_")[1])
            n_layers = max(n_layers, idx + 1)

    if not rank_pairs:
        raise ValueError("rank_pairs is empty — nothing to search")

    # Keep a pristine copy — layer-specific filtering must not mutate the
    # original list and leak into subsequent layers.
    _base_rank_pairs = list(rank_pairs)

    selected_ranks: dict[int, tuple[int, int]] = {}

    for layer_idx in range(n_layers):
        layer_path = calib_dir / f"layer_{layer_idx}.pt"
        if not layer_path.exists():
            print(f"[rank_search] layer {layer_idx}: calib data not found, skipping")
            continue

        data = load_pt(layer_path)
        q = data["q"].float()
        k = data["k"].float()
        v = data["v"].float()
        d_model, head_dim = infer_calib_dims(q, n_heads, meta)

        # Per-layer copy — floor filtering and head_dim validation must
        # only affect this layer, never leak into subsequent ones.
        layer_rank_pairs = list(_base_rank_pairs)

        # ---- layer-aware parameters ----
        layer_scale = get_layer_tolerance_scale(layer_idx, layer_tolerance_scale)
        min_r_k, min_r_v = get_layer_rank_floor(layer_idx, layer_rank_floor)
        if min_r_k > 1 or min_r_v > 1:
            _before = len(layer_rank_pairs)
            layer_rank_pairs = [
                (rk, rv) for rk, rv in layer_rank_pairs
                if rk >= min_r_k and rv >= min_r_v
            ]
            if not layer_rank_pairs:
                raise ValueError(
                    f"[rank_search] layer {layer_idx}: all candidates "
                    f"filtered out by rank floor (min_r_k={min_r_k}, "
                    f"min_r_v={min_r_v})"
                )
            if len(layer_rank_pairs) < _before:
                print(f"[rank_search] layer {layer_idx}: rank floor "
                      f"(min_r_k={min_r_k}, min_r_v={min_r_v}) "
                      f"filtered {_before} → {len(layer_rank_pairs)} candidates")

        print(f"\n[rank_search] layer {layer_idx}: d_model={d_model} "
              f"n_heads={n_heads} head_dim={head_dim}")

        # Validate / filter candidates against head_dim
        if _is_legacy:
            # Legacy mode: silently filter out-of-range pairs with a warning
            _filtered: list[tuple[int, int]] = []
            _removed: list[int] = []
            for rk, rv in layer_rank_pairs:
                if 1 <= rk <= head_dim and 1 <= rv <= head_dim:
                    _filtered.append((rk, rv))
                else:
                    _removed.append(rk)
            if _removed:
                _valid_ranks = sorted({rk for rk, _ in _filtered})
                warnings.warn(
                    f"[rank_search] layer {layer_idx}: head_dim={head_dim}, "
                    f"filtering out rank candidates exceeding head_dim: "
                    f"{_removed}. Valid candidates: {_valid_ranks}",
                    UserWarning,
                    stacklevel=2,
                )
            if not _filtered:
                raise ValueError(
                    f"[rank_search] layer {layer_idx}: head_dim={head_dim}, "
                    f"no valid rank candidates remain after filtering. "
                    f"Original candidates: {layer_rank_pairs}. "
                    f"All candidates exceed head_dim."
                )
            layer_rank_pairs = _filtered
        else:
            for rk, rv in layer_rank_pairs:
                if not (1 <= rk <= head_dim):
                    raise ValueError(
                        f"[rank_search] layer {layer_idx}: r_k={rk} out of range "
                        f"[1, head_dim={head_dim}]"
                    )
                if not (1 <= rv <= head_dim):
                    raise ValueError(
                        f"[rank_search] layer {layer_idx}: r_v={rv} out of range "
                        f"[1, head_dim={head_dim}]"
                    )

        mode_label = f"selection={selection_mode}"
        if selection_mode == "signal_normalized":
            mode_label += f" layer_scale={layer_scale}"
            if min_r_k > 1 or min_r_v > 1:
                mode_label += f" floor=({min_r_k},{min_r_v})"

        print(f"[rank_search] candidates={layer_rank_pairs}"
              f"  n_steps={n_steps}  {mode_label}"
              f"  rel_tol={relative_tolerance}"
              f"  logits_abs_tol={logits_abs_tolerance}"
              f"  attn_abs_tol={attn_abs_tolerance}"
              f"  value_abs_tol={value_abs_tolerance}")

        # ---- evaluate every candidate pair ----
        signal_scales = None
        if selection_mode == "signal_normalized":
            signal_scales = compute_signal_scales(
                q, k, v,
                n_heads=n_heads,
                d_model=d_model,
                head_dim=head_dim,
                row_batch_size=row_batch_size,
            )
            print(
                f"[rank_search] signal logits={signal_scales['signal_logits']:.6f}"
                f" attn={signal_scales['signal_attn']:.6f}"
                f" value={signal_scales['signal_value']:.6f}"
            )

        results = []
        for rk, rv in layer_rank_pairs:
            r = _evaluate_rank(
                q, k, v, rk, rv, d_model, n_heads,
                n_steps=n_steps, lr=lr,
                orthogonalize_every=orthogonalize_every,
                w_logits=w_logits, w_attn=w_attn, w_value=w_value,
                device=device,
                warmup_steps=warmup_steps,
                row_batch_size=row_batch_size,
                lr_pk=lr_pk, lr_pv=lr_pv, lr_xi=lr_xi,
                beta1=beta1, beta2=beta2,
                grad_clip=grad_clip,
                lambda_z=lambda_z, lambda_o=lambda_o, lambda_v=lambda_v,
                lambda_topk=lambda_topk, lambda_kl=lambda_kl,
                lambda_logit_topm=lambda_logit_topm,
                topk_k=topk_k, hard_neg_m=hard_neg_m, kl_top_m=kl_top_m,
                topk_margin=topk_margin,
                topk_loss_start_after_warmup=topk_loss_start_after_warmup,
                topk_metric_ks=topk_metric_ks,
                eval_every=eval_every,
                early_stopping=early_stopping,
                patience=patience,
                min_delta=min_delta, min_delta_mode=min_delta_mode,
                gamma_min=gamma_min,
                logit_scale_mode=logit_scale_mode,
                eps_loss=eps_loss, adam_eps=adam_eps,
                optimizer=optimizer,
            )
            results.append(r)
            cost = r["rank_cost"]
            print(
                f"  r_k={r['r_k']:>3d} r_v={r['r_v']:>3d}"
                f"  calib_total={r['best_calib_total']:.6f}"
                f"  calib_logits={r['best_calib_logits']:.6f}"
                f"  calib_attn={r['best_calib_attn']:.6f}"
                f"  calib_value={r['best_calib_value']:.6f}"
                f"  topk={r.get('best_calib_topk', 0.0):.6f}"
                f"  top10={r.get('best_calib_top_recalls', {}).get('top10_recall', 0.0):.4f}"
                f"  cost={cost}  step={r['best_step']} stopped={r['stopped_early']}"
            )

        if not results:
            continue

        # ---- judge each candidate ----
        if selection_mode == "signal_normalized":
            _nsig = signal_scales
            for r in results:
                _signal_normalized_pass(
                    r, _nsig,
                    logits_signal_tolerance,
                    attn_signal_tolerance,
                    value_signal_tolerance,
                    layer_scale,
                )
        elif selection_mode == "constraint":
            baseline_result = min(results, key=lambda x: x["best_calib_total"])
            for r in results:
                r["logits_pass"] = _component_pass(
                    r["best_calib_logits"], baseline_result["best_calib_logits"],
                    relative_tolerance, logits_abs_tolerance,
                )
                r["attn_pass"] = _component_pass(
                    r["best_calib_attn"], baseline_result["best_calib_attn"],
                    relative_tolerance, attn_abs_tolerance,
                )
                r["value_pass"] = _component_pass(
                    r["best_calib_value"], baseline_result["best_calib_value"],
                    relative_tolerance, value_abs_tolerance,
                )
                r["all_pass"] = r["logits_pass"] and r["attn_pass"] and r["value_pass"]
        elif selection_mode == "total":
            best_total = min(r["best_calib_total"] for r in results)
            for r in results:
                r["total_pass"] = _component_pass(
                    r["best_calib_total"], best_total,
                    relative_tolerance, best_total * relative_tolerance + 1e-8,
                )
                r["all_pass"] = r["total_pass"]
        elif selection_mode == "logits":
            best_logits = min(r["best_calib_logits"] for r in results)
            for r in results:
                r["logits_pass"] = _component_pass(
                    r["best_calib_logits"], best_logits,
                    relative_tolerance, logits_abs_tolerance,
                )
                r["all_pass"] = r["logits_pass"]
        elif selection_mode == "attn_value_abs":
            for r in results:
                r["attn_pass"] = r["best_calib_attn"] <= attn_abs_tolerance
                r["value_pass"] = r["best_calib_value"] <= value_abs_tolerance
                r["all_pass"] = r["attn_pass"] and r["value_pass"]

        # ---- print per-candidate assessment ----
        if selection_mode == "signal_normalized":
            print(f"\n  --- signal-normalized check "
                  f"(logits_tol={logits_signal_tolerance:.4f}*{layer_scale}={logits_signal_tolerance*layer_scale:.4f}, "
                  f"attn_tol={attn_signal_tolerance:.4f}*{layer_scale}={attn_signal_tolerance*layer_scale:.4f}, "
                  f"value_tol={value_signal_tolerance:.4f}*{layer_scale}={value_signal_tolerance*layer_scale:.4f}) ---")
            _print_header("nlogits", "nattn", "nvalue")
            for r in results:
                _print_row(r, r.get("normalized_logits_error", 0), r.get("normalized_attn_error", 0), r.get("normalized_value_error", 0))
        elif selection_mode == "attn_value_abs":
            print(f"\n  --- attn_value_abs check "
                  f"(attn_tol={attn_abs_tolerance:.1e}, value_tol={value_abs_tolerance:.1e}) ---")
            _print_header("attn", "value", "cost")
            for r in results:
                _print_row(r, f"{'PASS' if r['attn_pass'] else 'FAIL':>10}", f"{'PASS' if r['value_pass'] else 'FAIL':>10}", r['rank_cost'])
        else:
            label = selection_mode
            print(f"\n  --- {label} check (rel_tol={relative_tolerance}) ---")
            _print_header("logits", "attn", "value")
            for r in results:
                tag = "PASS" if r["all_pass"] else "REJECT"
                _print_row(r, f"{tag:>10}", "", f"{r['rank_cost']:>6d}")

        # ---- selection: prefer smallest rank_cost among passing pairs ----
        passing = [r for r in results if r["all_pass"]]
        if passing:
            passing.sort(key=lambda x: (x["rank_cost"], x["best_calib_total"]))
            best = passing[0]
        else:
            results.sort(key=lambda x: x["best_calib_total"])
            best = results[0]

        chosen_rk, chosen_rv = best["r_k"], best["r_v"]
        selected_ranks[layer_idx] = (chosen_rk, chosen_rv)

        print(
            f"[rank_search] layer {layer_idx}: selected r_k={chosen_rk} r_v={chosen_rv}"
            f"  cost={best['rank_cost']}  calib_total={best['best_calib_total']:.6f}"
            f"  best_step={best.get('best_step', '?')} stopped_early={best.get('stopped_early', '?')}"
        )

        # ---- per-layer output (optional) ----
        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            _norm_results = []
            for _r in results:
                _nr = dict(_r)
                _nr.setdefault("r_k", _r.get("rank_k", _r.get("r_k", 0)))
                _nr.setdefault("r_v", _r.get("rank_v", _r.get("r_v", 0)))
                _nr.setdefault("rank_cost", _r.get("rank_cost", _nr["r_k"] + _nr["r_v"]))
                _norm_results.append(_nr)
            _json = {
                "layer_idx": layer_idx,
                "selected_r_k": chosen_rk,
                "selected_r_v": chosen_rv,
                "selection_method": selection_mode,
                "best_calib_total": best.get("best_calib_total", 0.0),
                "best_calib_logits": best.get("best_calib_logits", 0.0),
                "best_calib_attn": best.get("best_calib_attn", 0.0),
                "best_calib_value": best.get("best_calib_value", 0.0),
                "best_calib_topk": best.get("best_calib_topk", 0.0),
                "best_calib_kl_topm": best.get("best_calib_kl_topm", 0.0),
                "best_calib_logit_topm": best.get("best_calib_logit_topm", 0.0),
                "best_calib_top_recalls": best.get("best_calib_top_recalls", {}),
                "best_step": best.get("best_step", 0),
                "actual_steps": best.get("actual_steps", 0),
                "stopped_early": best.get("stopped_early", False),
                "relative_tolerance": relative_tolerance,
                "logits_abs_tolerance": logits_abs_tolerance,
                "attn_abs_tolerance": attn_abs_tolerance,
                "value_abs_tolerance": value_abs_tolerance,
                "candidates": layer_rank_pairs,
                "results": _norm_results,
            }
            if selection_mode == "signal_normalized":
                _json["layer_tolerance_scale"] = layer_scale
                _json["rank_floor"] = {"min_r_k": min_r_k, "min_r_v": min_r_v}
                _json["signal_scales"] = signal_scales
                _json["logits_signal_tolerance"] = logits_signal_tolerance
                _json["attn_signal_tolerance"] = attn_signal_tolerance
                _json["value_signal_tolerance"] = value_signal_tolerance
            save_json(_json, out / f"layer_{layer_idx}_rank_search.json")

    return selected_ranks


def run_rank_search_from_config(config) -> dict[int, tuple[int, int]]:
    from hawp_laq.config import HAWPLAQConfig

    # Resolve head_dim to validate candidates early
    calib_dir = Path(config.calib.output_dir)
    meta = load_pt(calib_dir / "meta.pt")
    n_heads = meta.get("n_heads")
    if n_heads is None:
        model_id = meta.get("model_id")
        if model_id is None:
            raise ValueError(
                "Cannot infer n_heads: meta.pt must contain either "
                "'n_heads' or 'model_id'."
            )
        from transformers import AutoConfig
        model_cfg = AutoConfig.from_pretrained(model_id)
        n_heads = model_cfg.num_attention_heads
    if n_heads is None or n_heads <= 0:
        raise ValueError(
            "n_heads must be a positive integer; "
            "set meta['n_heads'] or provide meta['model_id']"
        )

    # Find sample data to determine d_model / head_dim
    for p in sorted(calib_dir.glob("layer_*.pt")):
        sample = load_pt(p)
        d_model, head_dim = infer_calib_dims(sample["q"], n_heads, meta)
        break
    else:
        raise FileNotFoundError(f"No layer_*.pt files found in {calib_dir}")

    rank_pairs = build_rank_pairs(
        rank_candidates=getattr(config.rank_search, "rank_candidates", None),
        r_k_candidates=getattr(config.rank_search, "r_k_candidates", None),
        r_v_candidates=getattr(config.rank_search, "r_v_candidates", None),
        rank_pair_candidates=getattr(config.rank_search, "rank_pair_candidates", None),
        head_dim=head_dim,
    )

    output_dir = config.rank_search.output_dir
    n_steps = getattr(config.rank_search, "n_steps", None)
    if n_steps is None:
        n_steps = config.projector.n_steps

    return search_rank_per_layer(
        calib_dir=config.calib.output_dir,
        rank_pairs=rank_pairs,
        n_steps=n_steps,
        lr=config.projector.lr,
        orthogonalize_every=config.projector.orthogonalize_every,
        w_logits=config.projector.w_logits,
        w_attn=config.projector.w_attn,
        w_value=config.projector.w_value,
        device=config.train.device,
        output_dir=output_dir,
        relative_tolerance=config.rank_search.relative_tolerance,
        logits_abs_tolerance=config.rank_search.logits_abs_tolerance,
        attn_abs_tolerance=config.rank_search.attn_abs_tolerance,
        value_abs_tolerance=config.rank_search.value_abs_tolerance,
        selection_mode=getattr(config.rank_search, "selection_mode", "constraint"),
        logits_signal_tolerance=getattr(config.rank_search, "logits_signal_tolerance", 0.01),
        attn_signal_tolerance=getattr(config.rank_search, "attn_signal_tolerance", 0.01),
        value_signal_tolerance=getattr(config.rank_search, "value_signal_tolerance", 0.02),
        layer_tolerance_scale=getattr(config.rank_search, "layer_tolerance_scale", None),
        layer_rank_floor=getattr(config.rank_search, "layer_rank_floor", None),
        # -- Riemannian-Adam params from config.projector --
        warmup_steps=getattr(config.projector, "warmup_steps", 30),
        row_batch_size=getattr(config.projector, "row_batch_size", None),
        lr_pk=getattr(config.projector, "lr_pk", 5e-3),
        lr_pv=getattr(config.projector, "lr_pv", 5e-3),
        lr_xi=getattr(config.projector, "lr_xi", 1e-2),
        beta1=getattr(config.projector, "beta1", 0.9),
        beta2=getattr(config.projector, "beta2", 0.99),
        grad_clip=getattr(config.projector, "grad_clip", 1.0),
        lambda_z=getattr(config.projector, "lambda_z", 1.0),
        lambda_o=getattr(config.projector, "lambda_o", 2.0),
        lambda_v=getattr(config.projector, "lambda_v", 0.05),
        lambda_topk=getattr(config.projector, "lambda_topk", 0.0),
        lambda_kl=getattr(config.projector, "lambda_kl", 0.0),
        lambda_logit_topm=getattr(config.projector, "lambda_logit_topm", 0.0),
        topk_k=getattr(config.projector, "topk_k", 8),
        hard_neg_m=getattr(config.projector, "hard_neg_m", 32),
        kl_top_m=getattr(config.projector, "kl_top_m", 64),
        topk_margin=getattr(config.projector, "topk_margin", 0.05),
        topk_loss_start_after_warmup=getattr(config.projector, "topk_loss_start_after_warmup", True),
        topk_metric_ks=getattr(config.projector, "topk_metric_ks", (5, 10)),
        eval_every=getattr(config.projector, "eval_every", 50),
        early_stopping=getattr(config.projector, "early_stopping", True),
        patience=getattr(config.projector, "patience", 5),
        min_delta=getattr(config.projector, "min_delta", 1e-4),
        min_delta_mode=getattr(config.projector, "min_delta_mode", "relative"),
        gamma_min=getattr(config.projector, "gamma_min", 1e-4),
        logit_scale_mode=getattr(config.hawp, "logit_scale_mode", "rk"),
        eps_loss=getattr(config.projector, "eps_loss", 1e-8),
        adam_eps=getattr(config.projector, "adam_eps", 1e-8),
        optimizer=getattr(config.projector, "optimizer", "riemannian_adam"),
    )
