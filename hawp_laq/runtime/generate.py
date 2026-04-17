from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hawp_laq.config import HAWPLAQConfig, load_config
from hawp_laq.modeling.modeling_llama_hawp import convert_llama_to_hawp

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def print_device_info(device: str) -> None:
    cuda_ok = torch.cuda.is_available()
    print(f"[device] cuda.is_available() = {cuda_ok}")
    print(f"[device] target device = {device}")
    if cuda_ok:
        idx = torch.cuda.current_device()
        print(f"[device] GPU = {torch.cuda.get_device_name(idx)}")
        allocated = torch.cuda.memory_allocated(idx)
        reserved = torch.cuda.memory_reserved(idx)
        print(f"[device] mem allocated = {_fmt_bytes(allocated)}")
        print(f"[device] mem reserved   = {_fmt_bytes(reserved)}")


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def load_baseline_model(cfg: HAWPLAQConfig):
    model_id = cfg.model.model_id
    dtype = _DTYPE_MAP.get(cfg.model.torch_dtype, torch.float32)
    device = _resolve_device(cfg.train.device)

    print(f"[load] model_id = {model_id}")
    print(f"[load] dtype = {cfg.model.torch_dtype}")
    print(f"[load] load_in_4bit = {cfg.model.load_in_4bit}")
    print(f"[load] device = {device}")

    tok_kwargs: dict[str, Any] = {}
    model_kwargs: dict[str, Any] = {"torch_dtype": dtype, "device_map": device}
    if cfg.model.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model_kwargs.pop("torch_dtype", None)
        model_kwargs.pop("device_map", None)

    tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if not cfg.model.load_in_4bit:
        model = model.to(device)
    model.eval()
    return model, tokenizer, device


def _resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda requested but not available, falling back to cpu")
        return "cpu"
    return device


@torch.inference_mode()
def generate_text(model, tokenizer, prompts: list[str], cfg: HAWPLAQConfig) -> list[str]:
    gen_kw = {
        "max_new_tokens": cfg.generation.max_new_tokens,
        "do_sample": cfg.generation.do_sample,
        "temperature": cfg.generation.temperature,
        "top_p": cfg.generation.top_p,
    }
    results = []
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        out_ids = model.generate(**inputs, **gen_kw)
        text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        results.append(text)
    return results


def _print_results(prompts: list[str], outputs: list[str]) -> None:
    for i, (p, o) in enumerate(zip(prompts, outputs)):
        print(f"\n--- prompt {i} ---")
        print(f"IN:  {p}")
        print(f"OUT: {o}")


def run_baseline(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    print("=" * 60)
    print(f"[mode] baseline")
    print_device_info(cfg.train.device)
    print("=" * 60)

    model, tokenizer, device = load_baseline_model(cfg)
    prompts = cfg.generation.prompts
    print(f"[baseline] running {len(prompts)} prompt(s) ...")

    outputs = generate_text(model, tokenizer, prompts, cfg)
    _print_results(prompts, outputs)


@torch.inference_mode()
def run_hawp_only(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    print("=" * 60)
    print("[mode] hawp_only")
    print_device_info(cfg.train.device)
    print("=" * 60)

    model, tokenizer, device = load_baseline_model(cfg)

    r_k = cfg.projector.r_k
    r_v = cfg.projector.r_v
    model = convert_llama_to_hawp(model, r_k=r_k, r_v=r_v)
    model = model.to(device)
    model.eval()

    from hawp_laq.runtime.projector_bank import load_projectors
    projector_dir = cfg.projector.output_dir
    if Path(projector_dir).exists():
        load_projectors(model, projector_dir)
        print(f"[hawp_only] loaded projectors from {projector_dir}")
    else:
        print(f"[hawp_only] no trained projectors at {projector_dir}, using identity")

    print(f"[hawp_only] r_k={r_k}, r_v={r_v}")
    prompts = cfg.generation.prompts
    print(f"[hawp_only] running {len(prompts)} prompt(s) ...")

    outputs = generate_text(model, tokenizer, prompts, cfg)
    _print_results(prompts, outputs)


@torch.inference_mode()
def run_hawp_quant(config_path: str | Path) -> None:
    from hawp_laq.runtime.cache_manager import CacheManager
    from hawp_laq.runtime.scheduler import TokenBudgetScheduler

    cfg = load_config(config_path)
    print("=" * 60)
    print("[mode] hawp_quant")
    print_device_info(cfg.train.device)
    print("=" * 60)

    model, tokenizer, device = load_baseline_model(cfg)

    r_k = cfg.projector.r_k
    r_v = cfg.projector.r_v
    model = convert_llama_to_hawp(model, r_k=r_k, r_v=r_v)
    model = model.to(device)
    model.eval()

    from hawp_laq.runtime.projector_bank import load_projectors
    projector_dir = cfg.projector.output_dir
    if Path(projector_dir).exists():
        load_projectors(model, projector_dir)
        print(f"[hawp_quant] loaded projectors from {projector_dir}")

    n_layers = sum(1 for _, m in model.named_modules() if type(m).__name__ == "HAWPAttention")
    head_dim = r_k if r_k is not None else 64

    for name, module in model.named_modules():
        if type(module).__name__ == "HAWPAttention":
            head_dim = module.head_dim
            break

    scheduler = TokenBudgetScheduler(
        total_budget=cfg.sched.total_budget,
        recent_window=cfg.sched.recent_window,
        high_ratio=cfg.sched.high_ratio,
        low_ratio=cfg.sched.low_ratio,
    )
    cache_mgr = CacheManager(
        n_layers=n_layers,
        n_heads=cfg.model.num_attention_heads if hasattr(cfg.model, 'num_attention_heads') else 12,
        head_dim=head_dim,
        scheduler=scheduler,
        k_group_size=cfg.quant.k_group_size,
        v_group_size=cfg.quant.v_group_size,
        use_rotation=cfg.quant.use_rotation,
        outlier_threshold=cfg.quant.outlier_threshold,
    )
    print(f"[hawp_quant] cache_manager created  n_layers={n_layers}  k_group={cfg.quant.k_group_size}  v_group={cfg.quant.v_group_size}")

    print(f"[hawp_quant] r_k={r_k}, r_v={r_v}")
    prompts = cfg.generation.prompts
    print(f"[hawp_quant] running {len(prompts)} prompt(s) ...")

    outputs = generate_text(model, tokenizer, prompts, cfg)
    _print_results(prompts, outputs)
    print(f"\n[hawp_quant] cache summary: {cache_mgr.summary()}")


@torch.inference_mode()
def run_hawp_quant_sched(config_path: str | Path) -> None:
    from hawp_laq.runtime.cache_manager import CacheManager
    from hawp_laq.runtime.scheduler import TokenBudgetScheduler

    cfg = load_config(config_path)
    print("=" * 60)
    print("[mode] hawp_quant_sched")
    print_device_info(cfg.train.device)
    print("=" * 60)

    model, tokenizer, device = load_baseline_model(cfg)

    r_k = cfg.projector.r_k
    r_v = cfg.projector.r_v
    model = convert_llama_to_hawp(model, r_k=r_k, r_v=r_v)
    model = model.to(device)
    model.eval()

    from hawp_laq.runtime.projector_bank import load_projectors
    projector_dir = cfg.projector.output_dir
    if Path(projector_dir).exists():
        load_projectors(model, projector_dir)
        print(f"[hawp_quant_sched] loaded projectors from {projector_dir}")

    n_layers = sum(1 for _, m in model.named_modules() if type(m).__name__ == "HAWPAttention")
    head_dim = 64
    n_heads = 12
    for name, module in model.named_modules():
        if type(module).__name__ == "HAWPAttention":
            head_dim = module.head_dim
            n_heads = module.num_key_value_heads
            break

    scheduler = TokenBudgetScheduler(
        total_budget=cfg.sched.total_budget,
        recent_window=cfg.sched.recent_window,
        high_ratio=cfg.sched.high_ratio,
        low_ratio=cfg.sched.low_ratio,
    )
    cache_mgr = CacheManager(
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        scheduler=scheduler,
        k_group_size=cfg.quant.k_group_size,
        v_group_size=cfg.quant.v_group_size,
        use_rotation=cfg.quant.use_rotation,
        outlier_threshold=cfg.quant.outlier_threshold,
    )
    print(f"[hawp_quant_sched] cache_manager + scheduler ready  budget={cfg.sched.total_budget}")

    print(f"[hawp_quant_sched] r_k={r_k}, r_v={r_v}")
    prompts = cfg.generation.prompts
    print(f"[hawp_quant_sched] running {len(prompts)} prompt(s) ...")

    outputs = generate_text(model, tokenizer, prompts, cfg)
    _print_results(prompts, outputs)

    drops = cache_mgr.apply_scheduler()
    print(f"[hawp_quant_sched] scheduler dropped {len(drops)} tokens")
    print(f"[hawp_quant_sched] cache summary: {cache_mgr.summary()}")
