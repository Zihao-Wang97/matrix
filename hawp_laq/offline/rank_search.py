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
    n_steps: int,
    lr: float,
    orthogonalize_every: int,
    w_logits: float,
    w_attn: float,
    w_value: float,
    device: str,
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
    result = trainer.train_one_group(q, k, v, n_steps=n_steps)
    metrics = result["metrics"]
    return {
        "r_k": r_k,
        "r_v": r_v,
        "final_loss": metrics["total"][-1],
        "final_logits_loss": metrics["logits"][-1],
        "final_attn_loss": metrics["attn"][-1],
        "final_value_loss": metrics["value"][-1],
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
) -> dict[str, float]:
    """Compute per-component signal strengths from full-precision Q/K/V.

    Returns mean squared magnitude of: logits, attention weights, value output.
    These are used as signal references for normalized error thresholds.
    """
    d_model = q.shape[-1]
    head_dim = d_model // n_heads

    # Reshape to multi-head: [B, T, D] → [B, n_heads, T, head_dim]
    b, s, _ = q.shape
    q_mh = q.reshape(b, s, n_heads, head_dim).transpose(1, 2)
    k_mh = k.reshape(b, s, n_heads, head_dim).transpose(1, 2)
    v_mh = v.reshape(b, s, n_heads, head_dim).transpose(1, 2)

    logits_fp = torch.matmul(q_mh, k_mh.transpose(-2, -1)) / math.sqrt(head_dim)
    attn_fp = torch.softmax(logits_fp, dim=-1)
    value_fp = torch.matmul(attn_fp, v_mh)

    return {
        "signal_logits": logits_fp.pow(2).mean().item(),
        "signal_attn": attn_fp.pow(2).mean().item(),
        "signal_value": value_fp.pow(2).mean().item(),
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

    n_logits = result["final_logits_loss"] / sig_logits
    n_attn = result["final_attn_loss"] / sig_attn
    n_value = result["final_value_loss"] / sig_value

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

    _valid_modes = {"constraint", "signal_normalized"}
    if selection_mode not in _valid_modes:
        raise ValueError(
            f"Unknown selection_mode={selection_mode!r}. "
            f"Supported: {sorted(_valid_modes)}"
        )

    calib_dir = Path(calib_dir)
    meta = load_pt(calib_dir / "meta.pt")
    n_layers = meta.get("n_layers", 0)
    n_heads = meta.get("n_heads")

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
        d_model = q.shape[-1]

        if n_heads is None:
            from transformers import AutoConfig
            cfg_auto = AutoConfig.from_pretrained(
                meta.get("model_id", "facebook/opt-125m")
            )
            n_heads = cfg_auto.num_attention_heads

        head_dim = d_model // n_heads

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
            signal_scales = compute_signal_scales(q, k, v, n_heads)
            print(
                f"[rank_search] signal logits={signal_scales['signal_logits']:.6f}"
                f" attn={signal_scales['signal_attn']:.6f}"
                f" value={signal_scales['signal_value']:.6f}"
            )

        results = []
        for rk, rv in layer_rank_pairs:
            r = _evaluate_rank(
                q, k, v, rk, rv, d_model, n_heads,
                n_steps, lr, orthogonalize_every,
                w_logits, w_attn, w_value, device,
            )
            results.append(r)
            cost = r.get("rank_cost", r.get("r_k", 0) + r.get("r_v", 0))
            print(
                f"  r_k={r.get('r_k', r.get('rank_k', 0)):>3d}"
                f" r_v={r.get('r_v', r.get('rank_v', 0)):>3d}"
                f"  total={r['final_loss']:.6f}"
                f"  logits={r['final_logits_loss']:.6f}"
                f"  attn={r['final_attn_loss']:.6f}"
                f"  val={r['final_value_loss']:.6f}"
                f"  cost={cost}"
            )

        if not results:
            continue

        # ---- judge each candidate ----
        if selection_mode == "signal_normalized":
            _nsig = signal_scales  # already computed
            for r in results:
                _signal_normalized_pass(
                    r, _nsig,
                    logits_signal_tolerance,
                    attn_signal_tolerance,
                    value_signal_tolerance,
                    layer_scale,
                )
        else:
            # ---- legacy constraint mode ----
            baseline_result = min(results, key=lambda x: x["final_loss"])
            for r in results:
                r["logits_pass"] = _component_pass(
                    r["final_logits_loss"],
                    baseline_result["final_logits_loss"],
                    relative_tolerance,
                    logits_abs_tolerance,
                )
                r["attn_pass"] = _component_pass(
                    r["final_attn_loss"],
                    baseline_result["final_attn_loss"],
                    relative_tolerance,
                    attn_abs_tolerance,
                )
                r["value_pass"] = _component_pass(
                    r["final_value_loss"],
                    baseline_result["final_value_loss"],
                    relative_tolerance,
                    value_abs_tolerance,
                )
                r["all_pass"] = (
                    r["logits_pass"] and r["attn_pass"] and r["value_pass"]
                )

        # ---- print per-candidate assessment ----
        if selection_mode == "signal_normalized":
            print(f"\n  --- signal-normalized check "
                  f"(logits_tol={logits_signal_tolerance:.4f}*{layer_scale}={logits_signal_tolerance*layer_scale:.4f}, "
                  f"attn_tol={attn_signal_tolerance:.4f}*{layer_scale}={attn_signal_tolerance*layer_scale:.4f}, "
                  f"value_tol={value_signal_tolerance:.4f}*{layer_scale}={value_signal_tolerance*layer_scale:.4f}) ---")
            print(
                f"  {'r_k':>4} {'r_v':>4}"
                f"  {'nlogits':>10}"
                f"  {'nattn':>10}"
                f"  {'nvalue':>10}"
                f"  {'cost':>6}"
                f"  {'result':>8}"
            )
            for r in results:
                tag = "PASS" if r["all_pass"] else "REJECT"
                rk = r.get("r_k", r.get("rank_k", 0))
                rv = r.get("r_v", r.get("rank_v", 0))
                cost = r.get("rank_cost", rk + rv)
                print(
                    f"  {rk:>4d} {rv:>4d}"
                    f"  {r.get('normalized_logits_error', 0):>10.4f}"
                    f"  {r.get('normalized_attn_error', 0):>10.4f}"
                    f"  {r.get('normalized_value_error', 0):>10.4f}"
                    f"  {cost:>6d}"
                    f"  {tag:>8}"
                )
        else:
            print(f"\n  --- constraint check (rel_tol={relative_tolerance}) ---")
            print(
                f"  {'r_k':>4} {'r_v':>4}"
                f"  {'logits':>10}"
                f"  {'attn':>10}"
                f"  {'value':>10}"
                f"  {'cost':>6}"
                f"  {'result':>8}"
            )
            for r in results:
                tag = "PASS" if r["all_pass"] else "REJECT"
                rk = r.get("r_k", r.get("rank_k", 0))
                rv = r.get("r_v", r.get("rank_v", 0))
                cost = r.get("rank_cost", rk + rv)
                print(
                    f"  {rk:>4d} {rv:>4d}"
                    f"  {'PASS' if r['logits_pass'] else 'FAIL':>10}"
                    f"  {'PASS' if r['attn_pass'] else 'FAIL':>10}"
                    f"  {'PASS' if r['value_pass'] else 'FAIL':>10}"
                    f"  {cost:>6d}"
                    f"  {tag:>8}"
                )

        # ---- selection: prefer smallest rank_cost among passing pairs ----
        passing = [r for r in results if r["all_pass"]]
        if passing:
            def _cost(r):
                return r.get("rank_cost", r.get("r_k", r.get("rank_k", 0)) + r.get("r_v", r.get("rank_v", 0)))
            passing.sort(key=lambda x: (_cost(x), x["final_loss"]))
            best = passing[0]
        else:
            results.sort(key=lambda x: x["final_loss"])
            best = results[0]

        chosen_rk = best.get("r_k", best.get("rank_k", 0))
        chosen_rv = best.get("r_v", best.get("rank_v", 0))
        cost = best.get("rank_cost", chosen_rk + chosen_rv)
        selected_ranks[layer_idx] = (chosen_rk, chosen_rv)

        print(
            f"[rank_search] layer {layer_idx}: selected r_k={chosen_rk} r_v={chosen_rv}"
            f"  cost={cost}  total_loss={best['final_loss']:.6f}"
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
        from transformers import AutoConfig
        model_cfg = AutoConfig.from_pretrained(
            meta.get("model_id", "facebook/opt-125m")
        )
        n_heads = model_cfg.num_attention_heads

    # Find sample data to determine d_model / head_dim
    for p in sorted(calib_dir.glob("layer_*.pt")):
        sample = load_pt(p)
        d_model = sample["q"].shape[-1]
        head_dim = d_model // n_heads
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
    )
