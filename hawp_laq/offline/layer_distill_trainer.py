from __future__ import annotations

import inspect
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from hawp_laq.modeling.attention_hawp import HAWPAttention
from hawp_laq.offline.low_rank_attention_optimizer_torch import RiemannianAdam, clip_by_global_norm
from hawp_laq.runtime.projector_bank import normalize_projector_data
from hawp_laq.utils.io import load_pt, save_pt


@dataclass
class LayerDistillResult:
    metrics: dict
    best_step: int
    actual_steps: int
    stopped_early: bool
    best_eval_loss: float
    best_eval_mse: float
    best_eval_normalized: float


def discover_layer_chunk_paths(data_dir: str | Path, layer_idx: int) -> list[Path]:
    layer_dir = Path(data_dir) / f"layer_{layer_idx}"
    return sorted(layer_dir.glob("chunk_*.pt"))


def _hawp_modules(layer: torch.nn.Module) -> list[HAWPAttention]:
    return [m for m in layer.modules() if isinstance(m, HAWPAttention)]


def _param_dtype(layer: torch.nn.Module) -> torch.dtype:
    for module in _hawp_modules(layer):
        q_proj = getattr(module, "q_proj", None)
        weight = getattr(q_proj, "weight", None)
        if weight is not None and weight.is_floating_point():
            return weight.dtype
    for name, p in layer.named_parameters():
        if name.endswith(("p_k", "p_v", "gamma")):
            continue
        if p.is_floating_point():
            return p.dtype
    for p in layer.parameters():
        if p.is_floating_point():
            return p.dtype
    return torch.float32


def _make_causal_mask(batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    valid = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    mask = torch.where(
        valid,
        torch.zeros(seq_len, seq_len, device=device, dtype=dtype),
        torch.full((seq_len, seq_len), -1e4, device=device, dtype=dtype),
    )
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)


def _find_attention_module(layer: torch.nn.Module) -> torch.nn.Module | None:
    for attr in ("self_attn", "attention"):
        if hasattr(layer, attr):
            return getattr(layer, attr)
    return None


def _get_or_create_rotary_embedding(layer: torch.nn.Module, device: torch.device) -> torch.nn.Module | None:
    attn = _find_attention_module(layer)
    for owner in (attn, layer):
        rotary = getattr(owner, "rotary_emb", None) if owner is not None else None
        if rotary is not None:
            return rotary

    cached = getattr(layer, "_hawp_layer_rotary_emb", None)
    if cached is not None:
        return cached

    config = getattr(attn, "config", None) if attn is not None else None
    config = config if config is not None else getattr(layer, "config", None)
    model_type = str(getattr(config, "model_type", "")).lower()
    if config is None or "llama" not in model_type:
        return None

    try:
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    except Exception:
        return None

    try:
        rotary = LlamaRotaryEmbedding(config=config, device=device)
    except TypeError:
        rotary = LlamaRotaryEmbedding(config, device=device)
    rotary.to(device)
    setattr(layer, "_hawp_layer_rotary_emb", rotary)
    return rotary


def _call_rotary_embedding(
    rotary: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = int(position_ids.max().item()) + 1 if position_ids.numel() else hidden_states.shape[1]
    try:
        sig = inspect.signature(rotary.forward)
        params = sig.parameters
    except (TypeError, ValueError):
        params = {}

    if "position_ids" in params:
        return rotary(hidden_states, position_ids)
    if "seq_len" in params:
        return rotary(hidden_states, seq_len=seq_len)

    try:
        return rotary(hidden_states, position_ids)
    except Exception:
        return rotary(hidden_states, seq_len=seq_len)


def _make_position_embeddings(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    rotary = _get_or_create_rotary_embedding(layer, hidden_states.device)
    if rotary is None:
        return None
    return _call_rotary_embedding(rotary, hidden_states, position_ids)


def _call_decoder_layer(layer: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    bsz, seq_len, _ = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype

    kwargs = {}
    try:
        sig = inspect.signature(layer.forward)
        params = sig.parameters
    except (TypeError, ValueError):
        params = {}

    if "attention_mask" in params:
        kwargs["attention_mask"] = _make_causal_mask(bsz, seq_len, device, dtype)
    position_ids = None
    if "position_ids" in params:
        position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        kwargs["position_ids"] = position_ids
    if "cache_position" in params:
        kwargs["cache_position"] = torch.arange(seq_len, device=device, dtype=torch.long)
    if "position_embeddings" in params:
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        position_embeddings = _make_position_embeddings(layer, hidden_states, position_ids)
        if position_embeddings is not None:
            kwargs["position_embeddings"] = position_embeddings
    if "past_key_value" in params:
        kwargs["past_key_value"] = None
    if "output_attentions" in params:
        kwargs["output_attentions"] = False
    if "use_cache" in params:
        kwargs["use_cache"] = False

    out = layer(hidden_states, **kwargs)
    return out[0] if isinstance(out, tuple) else out


def _loss_stats(student: torch.Tensor, target: torch.Tensor, loss_mode: str, eps_loss: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    diff_sq = (student - target).pow(2).sum()
    teacher_sq = target.pow(2).sum()
    mse = diff_sq / max(target.numel(), 1)
    normalized = diff_sq / (teacher_sq + eps_loss)
    if loss_mode == "absolute":
        loss = mse
    elif loss_mode == "normalized":
        loss = normalized
    else:
        raise ValueError(f"layer_distill.loss_mode must be 'absolute' or 'normalized', got {loss_mode!r}")
    return loss, mse, normalized


def _load_chunk(path: Path, device: torch.device, dtype: torch.dtype, sample_batch_size: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    data = load_pt(path)
    hidden_in = data["hidden_in"]
    hidden_out = data["hidden_out"]
    if sample_batch_size is not None and 0 < sample_batch_size < hidden_in.shape[0]:
        idx = torch.randperm(hidden_in.shape[0])[:sample_batch_size]
        hidden_in = hidden_in[idx]
        hidden_out = hidden_out[idx]
    return hidden_in.to(device=device, dtype=dtype), hidden_out.to(device=device, dtype=dtype)


def _orth_err(P: torch.Tensor, rank: int) -> float:
    basis = P[:, :rank].float()
    eye = torch.eye(rank, device=basis.device, dtype=basis.dtype)
    return float(torch.linalg.norm(basis.transpose(0, 1) @ basis - eye).detach().cpu())


def _qr_orthogonalize_basis(basis: torch.Tensor) -> torch.Tensor:
    device = basis.device
    dtype = basis.dtype
    work = basis.detach().float()
    if not torch.isfinite(work).all():
        raise RuntimeError("Cannot orthogonalize projector basis: contains NaN or Inf")
    try:
        q, r = torch.linalg.qr(work, mode="reduced")
    except RuntimeError:
        q, r = torch.linalg.qr(work.cpu(), mode="reduced")
        q = q.to(device)
        r = r.to(device)
    diag = torch.diagonal(r, dim1=-2, dim2=-1)
    signs = torch.where(diag >= 0, torch.ones_like(diag), -torch.ones_like(diag))
    q = q * signs.unsqueeze(0)
    return q.to(device=device, dtype=dtype)


def _complete_to_full_basis_qr(basis: torch.Tensor, dim: int) -> torch.Tensor:
    basis = _qr_orthogonalize_basis(basis.float()).float()
    rank = basis.shape[1]
    if rank >= dim:
        return basis
    rand = torch.randn(dim, dim - rank, device=basis.device, dtype=basis.dtype)
    rand = rand - basis @ (basis.transpose(0, 1) @ rand)
    q_comp, _ = torch.linalg.qr(rand, mode="reduced")
    return torch.cat([basis, q_comp], dim=1)


def _orthogonalize_module(module: HAWPAttention) -> None:
    with torch.no_grad():
        if module.r_k < module.head_dim:
            basis = _qr_orthogonalize_basis(module.p_k.data[:, :module.r_k])
            module.p_k.data[:, :module.r_k].copy_(basis)
        if module.r_v < module.head_dim:
            basis = _qr_orthogonalize_basis(module.p_v.data[:, :module.r_v])
            module.p_v.data[:, :module.r_v].copy_(basis)


def _is_finite_tensor(x: torch.Tensor | None) -> bool:
    return x is not None and torch.isfinite(x.detach()).all().item()


def _all_finite_tensors(values: list[torch.Tensor | None]) -> bool:
    return all(v is None or _is_finite_tensor(v) for v in values)


def _gamma_trainable(module: HAWPAttention, train_gamma: bool) -> bool:
    if not train_gamma:
        return False
    if module.gamma_mode not in ("learned", "fixed"):
        return False
    return module.r_k < module.head_dim


def _prepare_trainable_projectors_fp32(
    layer: torch.nn.Module,
    *,
    train_pk: bool = True,
    train_pv: bool = True,
    train_gamma: bool,
    gamma_min: float,
    gamma_max: float,
) -> list[HAWPAttention]:
    for p in layer.parameters():
        p.requires_grad_(False)

    modules = _hawp_modules(layer)
    for module in modules:
        module.p_k.data = module.p_k.data.float()
        module.p_v.data = module.p_v.data.float()
        module.gamma.data = module.gamma.data.float().clamp_(min=gamma_min, max=gamma_max)

        module.p_k.requires_grad_(train_pk and module.r_k < module.head_dim)
        module.p_v.requires_grad_(train_pv and module.r_v < module.head_dim)
        module.gamma.requires_grad_(_gamma_trainable(module, train_gamma))
    return modules


def _set_trainable_projectors(
    layer: torch.nn.Module,
    *,
    train_gamma: bool,
    gamma_min: float,
) -> list[torch.nn.Parameter]:
    for p in layer.parameters():
        p.requires_grad_(False)

    params: list[torch.nn.Parameter] = []
    for module in _hawp_modules(layer):
        module_is_low_rank = module.r_k < module.head_dim or module.r_v < module.head_dim
        if module.r_k < module.head_dim:
            module.p_k.requires_grad_(True)
            params.append(module.p_k)
        if module.r_v < module.head_dim:
            module.p_v.requires_grad_(True)
            params.append(module.p_v)
        if train_gamma and module_is_low_rank and module.gamma_mode in ("learned", "fixed"):
            module.gamma.requires_grad_(True)
            with torch.no_grad():
                module.gamma.clamp_(min=gamma_min)
            params.append(module.gamma)
    return params


def _snapshot_hawp_state(layer: torch.nn.Module) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    return [
        (m.p_k.detach().clone(), m.p_v.detach().clone(), m.gamma.detach().clone())
        for m in _hawp_modules(layer)
    ]


def _restore_hawp_state(layer: torch.nn.Module, state: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> None:
    with torch.no_grad():
        for module, (p_k, p_v, gamma) in zip(_hawp_modules(layer), state):
            module.p_k.data.copy_(p_k.to(module.p_k.device, module.p_k.dtype))
            module.p_v.data.copy_(p_v.to(module.p_v.device, module.p_v.dtype))
            module.gamma.data.copy_(gamma.to(module.gamma.device, module.gamma.dtype))


@torch.no_grad()
def _eval_layer(
    layer: torch.nn.Module,
    chunk_paths: list[Path],
    *,
    device: torch.device,
    dtype: torch.dtype,
    sample_batch_size: Optional[int],
    eval_max_batches: Optional[int],
    loss_mode: str,
    eps_loss: float,
) -> dict:
    paths = chunk_paths
    if eval_max_batches is not None and eval_max_batches > 0:
        paths = paths[:eval_max_batches]

    total_diff = torch.zeros((), device=device, dtype=torch.float32)
    total_teacher = torch.zeros((), device=device, dtype=torch.float32)
    total_count = 0
    for path in paths:
        hidden_in, target = _load_chunk(path, device, dtype, sample_batch_size)
        student = _call_decoder_layer(layer, hidden_in)
        total_diff = total_diff + (student.float() - target.float()).pow(2).sum()
        total_teacher = total_teacher + target.float().pow(2).sum()
        total_count += target.numel()

    mse = total_diff / max(total_count, 1)
    normalized = total_diff / (total_teacher + eps_loss)
    if loss_mode == "absolute":
        loss = mse
    elif loss_mode == "normalized":
        loss = normalized
    else:
        raise ValueError(f"layer_distill.loss_mode must be 'absolute' or 'normalized', got {loss_mode!r}")
    return {
        "loss": float(loss.detach().cpu()),
        "mse": float(mse.detach().cpu()),
        "normalized": float(normalized.detach().cpu()),
    }


def _snapshot_riemannian_optimizer(opt: RiemannianAdam | None) -> dict | None:
    if opt is None:
        return None
    return {
        "step_num": opt.step_num,
        "m": opt.m.detach().clone(),
        "v": opt.v.detach().clone(),
        "lr": opt.lr,
    }


def _restore_riemannian_optimizer(opt: RiemannianAdam | None, state: dict | None) -> None:
    if opt is None or state is None:
        return
    opt.step_num = int(state["step_num"])
    opt.lr = float(state["lr"])
    opt.m.copy_(state["m"].to(opt.m.device, opt.m.dtype))
    opt.v.copy_(state["v"].to(opt.v.device, opt.v.dtype))


def _snapshot_torch_optimizer(opt: torch.optim.Optimizer | None) -> dict | None:
    if opt is None:
        return None
    state = {
        "state": {},
        "param_groups": [dict(g) for g in opt.param_groups],
    }
    for param, value in opt.state.items():
        cloned = {}
        for k, v in value.items():
            cloned[k] = v.detach().clone() if torch.is_tensor(v) else v
        state["state"][param] = cloned
    return state


def _restore_torch_optimizer(opt: torch.optim.Optimizer | None, state: dict | None) -> None:
    if opt is None or state is None:
        return
    for group, saved_group in zip(opt.param_groups, state["param_groups"]):
        for key, value in saved_group.items():
            if key != "params":
                group[key] = value
    opt.state.clear()
    for param, value in state["state"].items():
        restored = {}
        for k, v in value.items():
            restored[k] = v.detach().clone().to(param.device) if torch.is_tensor(v) else v
        opt.state[param] = restored


def _backoff_lrs(
    pk_opt: RiemannianAdam | None,
    pv_opt: RiemannianAdam | None,
    gamma_opt: torch.optim.Optimizer | None,
    factor: float,
) -> None:
    factor = float(factor)
    if not (0.0 < factor <= 1.0):
        return
    if pk_opt is not None:
        pk_opt.lr *= factor
    if pv_opt is not None:
        pv_opt.lr *= factor
    if gamma_opt is not None:
        for group in gamma_opt.param_groups:
            group["lr"] *= factor


def _backoff_torch_optimizer(opt: torch.optim.Optimizer | None, factor: float) -> None:
    if opt is None or not (0.0 < float(factor) <= 1.0):
        return
    for group in opt.param_groups:
        group["lr"] *= float(factor)


def _active_update_blocks(modules: list[HAWPAttention], step: int, alternate: bool) -> tuple[bool, bool]:
    can_pk = any(m.r_k < m.head_dim for m in modules)
    can_pv = any(m.r_v < m.head_dim for m in modules)
    if not alternate or not (can_pk and can_pv):
        return can_pk, can_pv
    return (step % 2 == 1), (step % 2 == 0)


def refine_layer_output_projector(
    layer: torch.nn.Module,
    chunk_paths: list[Path],
    *,
    device: str = "cpu",
    n_steps: int = 300,
    sample_batch_size: Optional[int] = None,
    eval_every: int = 25,
    eval_max_batches: Optional[int] = 16,
    optimizer: str = "adam",
    lr: float = 1e-4,
    lr_pk: float = 1e-5,
    lr_pv: float = 1e-5,
    lr_xi: float = 1e-6,
    beta1: float = 0.9,
    beta2: float = 0.99,
    grad_clip: float = 1.0,
    train_gamma: bool = True,
    gamma_min: float = 1e-4,
    gamma_max: float = 2.0,
    eps_loss: float = 1e-8,
    adam_eps: float = 1e-8,
    orthogonalize_every: int = 1,
    alternate_pk_pv: bool = True,
    finite_guard: bool = True,
    bad_step_patience: int = 20,
    lr_backoff: float = 0.5,
    loss_mode: str = "normalized",
    early_stopping: bool = True,
    patience: int = 5,
    min_delta: float = 1e-5,
    min_delta_mode: str = "relative",
    seed: int = 0,
    verbose: bool = True,
) -> LayerDistillResult:
    if not chunk_paths:
        raise FileNotFoundError("No layer distill chunk_*.pt files found")
    if eval_every <= 0:
        raise ValueError("eval_every must be > 0")
    if orthogonalize_every <= 0:
        raise ValueError("orthogonalize_every must be > 0")
    if min_delta_mode not in ("relative", "absolute"):
        raise ValueError(f"min_delta_mode must be relative or absolute, got {min_delta_mode!r}")
    optimizer = optimizer.lower()
    if optimizer not in ("adam", "riemannian_adam"):
        raise ValueError(f"layer_distill optimizer must be 'adam' or 'riemannian_adam', got {optimizer!r}")
    if gamma_max <= gamma_min:
        raise ValueError(f"gamma_max must be > gamma_min, got gamma_min={gamma_min}, gamma_max={gamma_max}")

    torch.manual_seed(seed)
    rng = random.Random(seed)
    dev = torch.device(device)
    layer.to(dev)
    layer.eval()
    dtype = _param_dtype(layer)

    modules = _prepare_trainable_projectors_fp32(
        layer,
        train_gamma=train_gamma,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
    )
    if not modules:
        raise RuntimeError("Layer has no HAWPAttention modules")
    trainable_params = [
        p for module in modules for p in (module.p_k, module.p_v, module.gamma)
        if p.requires_grad
    ]
    if not trainable_params:
        raise RuntimeError("Layer has no trainable low-rank projector parameters")

    adam_optimizer = (
        torch.optim.Adam(trainable_params, lr=lr, betas=(beta1, beta2), eps=adam_eps)
        if optimizer == "adam"
        else None
    )
    pk_opts = {
        module.layer_idx: RiemannianAdam(
            (module.head_dim, module.r_k), dev, torch.float32,
            lr_pk, beta1, beta2, adam_eps,
        )
        for module in modules
        if module.r_k < module.head_dim
    }
    pv_opts = {
        module.layer_idx: RiemannianAdam(
            (module.head_dim, module.r_v), dev, torch.float32,
            lr_pv, beta1, beta2, adam_eps,
        )
        for module in modules
        if module.r_v < module.head_dim
    }
    gamma_params = [
        module.gamma for module in modules
        if _gamma_trainable(module, train_gamma)
    ]
    gamma_optimizer = (
        torch.optim.Adam(gamma_params, lr=lr_xi, betas=(beta1, beta2), eps=adam_eps)
        if optimizer == "riemannian_adam" and gamma_params
        else None
    )

    best_eval_loss = float("inf")
    best_eval_mse = 0.0
    best_eval_normalized = 0.0
    best_state = _snapshot_hawp_state(layer)
    best_step = 0
    stale_checks = 0
    bad_steps = 0
    stopped_early = False
    history: list[dict] = []

    for step in range(1, n_steps + 1):
        path = chunk_paths[rng.randrange(len(chunk_paths))]
        hidden_in, target = _load_chunk(path, dev, dtype, sample_batch_size)

        state_before = _snapshot_hawp_state(layer)
        pk_state = {k: _snapshot_riemannian_optimizer(v) for k, v in pk_opts.items()}
        pv_state = {k: _snapshot_riemannian_optimizer(v) for k, v in pv_opts.items()}
        gamma_state = _snapshot_torch_optimizer(gamma_optimizer)
        adam_state = _snapshot_torch_optimizer(adam_optimizer)

        student = _call_decoder_layer(layer, hidden_in)
        if finite_guard and not _all_finite_tensors([student]):
            _restore_hawp_state(layer, state_before)
            bad_steps += 1
            for opt_obj in pk_opts.values():
                opt_obj.lr *= lr_backoff
            for opt_obj in pv_opts.values():
                opt_obj.lr *= lr_backoff
            _backoff_torch_optimizer(adam_optimizer, lr_backoff)
            _backoff_lrs(None, None, gamma_optimizer, lr_backoff)
            if verbose:
                print(f"[warn] step={step:04d} skipped: non-finite student output", flush=True)
            if bad_steps >= bad_step_patience:
                stopped_early = True
                break
            continue

        loss, _mse, _normalized = _loss_stats(student.float(), target.float(), loss_mode, eps_loss)
        if not loss.requires_grad:
            raise RuntimeError(
                "Layer distill loss has no gradient path. This usually means the layer is full-rank "
                "or has no trainable HAWP projector parameters."
            )
        if finite_guard and not _is_finite_tensor(loss):
            _restore_hawp_state(layer, state_before)
            bad_steps += 1
            for opt_obj in pk_opts.values():
                opt_obj.lr *= lr_backoff
            for opt_obj in pv_opts.values():
                opt_obj.lr *= lr_backoff
            _backoff_torch_optimizer(adam_optimizer, lr_backoff)
            _backoff_lrs(None, None, gamma_optimizer, lr_backoff)
            if verbose:
                print(f"[warn] step={step:04d} skipped: non-finite loss", flush=True)
            if bad_steps >= bad_step_patience:
                stopped_early = True
                break
            continue

        if optimizer == "adam":
            adam_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grads = [p.grad for p in trainable_params]
            if finite_guard and not _all_finite_tensors(grads):
                _restore_hawp_state(layer, state_before)
                _restore_torch_optimizer(adam_optimizer, adam_state)
                bad_steps += 1
                _backoff_torch_optimizer(adam_optimizer, lr_backoff)
                if verbose:
                    print(f"[warn] step={step:04d} skipped: non-finite Adam gradient", flush=True)
                if bad_steps >= bad_step_patience:
                    stopped_early = True
                    break
                continue
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            adam_optimizer.step()
        else:
            update_pk, update_pv = _active_update_blocks(modules, step, alternate_pk_pv)
            grad_params: list[torch.nn.Parameter] = []
            grad_specs: list[tuple[str, HAWPAttention]] = []
            for module in modules:
                if update_pk and module.r_k < module.head_dim:
                    grad_params.append(module.p_k)
                    grad_specs.append(("pk", module))
                if update_pv and module.r_v < module.head_dim:
                    grad_params.append(module.p_v)
                    grad_specs.append(("pv", module))
            if gamma_params and update_pk:
                for module in modules:
                    if _gamma_trainable(module, train_gamma):
                        grad_params.append(module.gamma)
                        grad_specs.append(("gamma", module))

            grads = torch.autograd.grad(loss, grad_params, allow_unused=True)
            grads = list(grads)
            sliced_grads: list[torch.Tensor | None] = []
            for spec, grad in zip(grad_specs, grads):
                kind, module = spec
                if grad is None:
                    sliced_grads.append(None)
                elif kind == "pk":
                    sliced_grads.append(grad[:, :module.r_k])
                elif kind == "pv":
                    sliced_grads.append(grad[:, :module.r_v])
                else:
                    sliced_grads.append(grad)

            if finite_guard and not _all_finite_tensors(sliced_grads):
                _restore_hawp_state(layer, state_before)
                for k, v in pk_opts.items():
                    _restore_riemannian_optimizer(v, pk_state.get(k))
                for k, v in pv_opts.items():
                    _restore_riemannian_optimizer(v, pv_state.get(k))
                _restore_torch_optimizer(gamma_optimizer, gamma_state)
                bad_steps += 1
                for opt_obj in pk_opts.values():
                    opt_obj.lr *= lr_backoff
                for opt_obj in pv_opts.values():
                    opt_obj.lr *= lr_backoff
                _backoff_lrs(None, None, gamma_optimizer, lr_backoff)
                if verbose:
                    print(f"[warn] step={step:04d} skipped: non-finite Riemannian gradient", flush=True)
                if bad_steps >= bad_step_patience:
                    stopped_early = True
                    break
                continue

            clipped_grads = clip_by_global_norm(sliced_grads, grad_clip)
            if gamma_optimizer is not None:
                gamma_optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                for (kind, module), grad in zip(grad_specs, clipped_grads):
                    if grad is None:
                        continue
                    if kind == "pk":
                        pk_opts[module.layer_idx].step_(module.p_k[:, :module.r_k], grad.float())
                    elif kind == "pv":
                        pv_opts[module.layer_idx].step_(module.p_v[:, :module.r_v], grad.float())
                    else:
                        module.gamma.grad = grad.float()

            if gamma_optimizer is not None and any(kind == "gamma" for kind, _ in grad_specs):
                gamma_optimizer.step()

        with torch.no_grad():
            for module in modules:
                module.gamma.clamp_(min=gamma_min, max=gamma_max)
                if optimizer == "adam" and step % orthogonalize_every == 0:
                    _orthogonalize_module(module)

        if finite_guard:
            values = []
            for module in modules:
                values.extend([
                    module.p_k[:, :module.r_k],
                    module.p_v[:, :module.r_v],
                    module.gamma,
                ])
            if not _all_finite_tensors(values):
                _restore_hawp_state(layer, state_before)
                if optimizer == "adam":
                    _restore_torch_optimizer(adam_optimizer, adam_state)
                else:
                    for k, v in pk_opts.items():
                        _restore_riemannian_optimizer(v, pk_state.get(k))
                    for k, v in pv_opts.items():
                        _restore_riemannian_optimizer(v, pv_state.get(k))
                    _restore_torch_optimizer(gamma_optimizer, gamma_state)
                bad_steps += 1
                for opt_obj in pk_opts.values():
                    opt_obj.lr *= lr_backoff
                for opt_obj in pv_opts.values():
                    opt_obj.lr *= lr_backoff
                _backoff_lrs(None, None, gamma_optimizer, lr_backoff)
                if verbose:
                    print(f"[warn] step={step:04d} skipped: non-finite parameters after update", flush=True)
                if bad_steps >= bad_step_patience:
                    stopped_early = True
                    break
                continue

        bad_steps = 0
        do_eval = step == 1 or step % eval_every == 0 or step == n_steps
        if do_eval:
            eval_stats = _eval_layer(
                layer,
                chunk_paths,
                device=dev,
                dtype=dtype,
                sample_batch_size=sample_batch_size,
                eval_max_batches=eval_max_batches,
                loss_mode=loss_mode,
                eps_loss=eps_loss,
            )

            gamma_val = float(modules[0].gamma.detach().float().cpu()) if modules else 1.0
            orth_k = max((_orth_err(m.p_k, m.r_k) for m in modules), default=0.0)
            orth_v = max((_orth_err(m.p_v, m.r_v) for m in modules), default=0.0)
            eval_loss = eval_stats["loss"]
            eval_mse = eval_stats["mse"]
            eval_norm = eval_stats["normalized"]

            history.append({
                "step": step,
                "eval_loss": eval_loss,
                "eval_mse": eval_mse,
                "eval_normalized": eval_norm,
                "gamma": gamma_val,
                "orthK": orth_k,
                "orthV": orth_v,
                "optimizer": optimizer,
                "bad_steps": bad_steps,
            })
            if verbose:
                print(
                    f"step={step:04d} layer_loss={eval_loss:.6e} "
                    f"mse={eval_mse:.6e} normalized={eval_norm:.6e} "
                    f"gamma={gamma_val:.5f} orthK={orth_k:.2e} orthV={orth_v:.2e}",
                    flush=True,
                )

            if eval_loss < best_eval_loss:
                prev_best = best_eval_loss
                best_eval_loss = eval_loss
                best_eval_mse = eval_mse
                best_eval_normalized = eval_norm
                best_state = _snapshot_hawp_state(layer)
                best_step = step
                if prev_best == float("inf"):
                    significant = True
                elif min_delta_mode == "relative":
                    significant = (prev_best - eval_loss) / (abs(prev_best) + eps_loss) > min_delta
                else:
                    significant = (prev_best - eval_loss) > min_delta
                stale_checks = 0 if significant else stale_checks + 1
            else:
                stale_checks += 1

        if early_stopping and stale_checks >= patience:
            if verbose:
                print(f"Early stopping at step {step} (no improvement for {stale_checks} eval checks).", flush=True)
            stopped_early = True
            break

    _restore_hawp_state(layer, best_state)
    layer.eval()
    return LayerDistillResult(
        metrics={"layer_distill": history},
        best_step=best_step,
        actual_steps=step,
        stopped_early=stopped_early,
        best_eval_loss=best_eval_loss,
        best_eval_mse=best_eval_mse,
        best_eval_normalized=best_eval_normalized,
    )


def _format_projector(p: torch.Tensor, *, head_dim: int, rank: int, original: torch.Tensor, save_format: str) -> torch.Tensor:
    save_format = save_format.lower()
    if save_format == "auto":
        save_format = "full" if original.ndim == 2 and original.shape[1] == head_dim else "low_rank"
    basis = _qr_orthogonalize_basis(p.detach().cpu()[:, :rank]).contiguous()
    if save_format == "low_rank":
        return basis
    if save_format == "full":
        return _complete_to_full_basis_qr(basis.float(), head_dim).to(dtype=p.detach().cpu().dtype)
    raise ValueError(f"layer_distill.save_format must be auto, low_rank, or full; got {save_format!r}")


def save_refined_layer_projector(
    layer: torch.nn.Module,
    original_projector: dict,
    result: LayerDistillResult,
    out_path: str | Path,
    *,
    save_format: str,
    source_path: str | Path,
) -> None:
    modules = _hawp_modules(layer)
    if len(modules) != 1:
        raise RuntimeError(f"Expected exactly one HAWPAttention in decoder layer, found {len(modules)}")
    module = modules[0]
    original_projector = normalize_projector_data(dict(original_projector), module.layer_idx)
    r_k = int(original_projector["r_k"])
    r_v = int(original_projector["r_v"])
    out = dict(original_projector)
    out["p_k"] = _format_projector(
        module.p_k, head_dim=module.head_dim, rank=r_k,
        original=original_projector["p_k"], save_format=save_format,
    )
    out["p_v"] = _format_projector(
        module.p_v, head_dim=module.head_dim, rank=r_v,
        original=original_projector["p_v"], save_format=save_format,
    )
    out["gamma"] = module.gamma.detach().cpu()
    out["r_k"] = r_k
    out["r_v"] = r_v
    out["logit_scale_mode"] = module.logit_scale_mode
    out["layer_distill"] = {
        "source_projector": str(source_path),
        "logit_scale_mode": module.logit_scale_mode,
        "best_step": result.best_step,
        "actual_steps": result.actual_steps,
        "stopped_early": result.stopped_early,
        "best_eval_loss": result.best_eval_loss,
        "best_eval_mse": result.best_eval_mse,
        "best_eval_normalized": result.best_eval_normalized,
        "metrics": result.metrics,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_pt(out, out_path)
