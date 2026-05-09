from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union, get_type_hints, get_args
import warnings

import yaml


@dataclass
class DataConfig:
    root: Path = Path("./data")
    cache: Path = Path("./cache")


@dataclass
class ModelConfig:
    model_id: str = "facebook/opt-125m"
    torch_dtype: str = "float32"
    load_in_4bit: bool = False


@dataclass
class GenerationConfig:
    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    prompts: list = field(default_factory=lambda: ["Hello, world!"])


@dataclass
class CalibConfig:
    nsamples: int = 8
    seq_len: int = 128
    output_dir: Path = Path("artifacts/calib")
    dataset: str = "wikitext2"
    capture_mode: str = "auto"


@dataclass
class ProjectorConfig:
    rank: Optional[int] = None
    r_k: Optional[int] = None
    r_v: Optional[int] = None
    # -- legacy training fields (retained for backward compat) --
    lr: float = 1e-3
    # Default max training steps for projector training (single_group).
    # May be overridden by ``rank_search.n_steps`` during rank search.
    n_steps: int = 200
    orthogonalize_every: int = 10
    target_layer: int = 0
    w_logits: float = 1.0
    w_attn: float = 1.0
    w_value: float = 0.5
    output_dir: Path = Path("artifacts/projectors")
    # -- Riemannian-Adam optimizer fields --
    optimizer: str = "riemannian_adam"
    warmup_steps: int = 30
    row_batch_size: Optional[int] = None
    lr_pk: float = 5e-3
    lr_pv: float = 5e-3
    lr_xi: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0
    lambda_z: float = 1.0
    lambda_o: float = 2.0
    lambda_v: float = 0.05
    lambda_topk: float = 0.0
    lambda_kl: float = 0.0
    lambda_logit_topm: float = 0.0
    topk_k: int = 8
    hard_neg_m: int = 32
    kl_top_m: int = 64
    topk_margin: float = 0.05
    topk_loss_start_after_warmup: bool = True
    topk_metric_ks: list[int] = field(default_factory=lambda: [5, 10])
    eval_every: int = 50
    early_stopping: bool = True
    patience: int = 5
    min_delta: float = 1e-4
    min_delta_mode: str = "relative"
    gamma_min: float = 1e-4
    eps_loss: float = 1e-8
    adam_eps: float = 1e-8


@dataclass
class AttentionDistillConfig:
    input_dir: Path = Path("artifacts/projectors")
    output_dir: Path = Path("artifacts/projectors_attn_distill")
    n_steps: int = 300
    sample_batch_size: Optional[int] = 128
    row_batch_size: Optional[int] = 128
    eval_every: int = 25
    eval_batch_size: int = 32
    eval_max_batches: Optional[int] = 16
    lr_pk: float = 1e-3
    lr_pv: float = 1e-3
    lr_xi: float = 1e-3
    optimizer: str = "riemannian_adam"
    lr: float = 1e-3
    orthogonalize_every: int = 1
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0
    gamma_min: float = 1e-4
    eps_loss: float = 1e-8
    adam_eps: float = 1e-8
    train_pk: bool = True
    train_gamma: bool = True
    loss_mode: str = "absolute"
    lambda_topk: float = 0.0
    lambda_kl: float = 0.0
    lambda_logit_topm: float = 0.0
    topk_k: int = 8
    hard_neg_m: int = 32
    kl_top_m: int = 64
    topk_margin: float = 0.05
    topk_metric_ks: list[int] = field(default_factory=lambda: [5, 10])
    early_stopping: bool = True
    patience: int = 5
    min_delta: float = 1e-5
    min_delta_mode: str = "relative"
    seed: int = 0
    save_format: str = "auto"


@dataclass
class LayerDistillConfig:
    data_dir: Path = Path("artifacts/layer_distill")
    input_dir: Path = Path("artifacts/projectors_attn_distill")
    output_dir: Path = Path("artifacts/projectors_layer_distill")
    nsamples: Optional[int] = None
    seq_len: Optional[int] = None
    batch_size: int = 1
    storage_dtype: str = "float16"
    n_steps: int = 300
    sample_batch_size: Optional[int] = None
    eval_every: int = 25
    eval_max_batches: Optional[int] = 16
    optimizer: str = "adam"
    lr: float = 1e-4
    lr_pk: float = 1e-5
    lr_pv: float = 1e-5
    lr_xi: float = 1e-6
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0
    train_gamma: bool = True
    gamma_min: float = 1e-4
    gamma_max: float = 2.0
    eps_loss: float = 1e-8
    adam_eps: float = 1e-8
    orthogonalize_every: int = 1
    alternate_pk_pv: bool = True
    finite_guard: bool = True
    bad_step_patience: int = 20
    lr_backoff: float = 0.5
    loss_mode: str = "normalized"
    early_stopping: bool = True
    patience: int = 5
    min_delta: float = 1e-5
    min_delta_mode: str = "relative"
    seed: int = 0
    save_format: str = "auto"


@dataclass
class QuantConfig:
    enabled: bool = False
    k_method: str = "turbo_prod"
    v_method: str = "turbo_mse"
    k_bits: int = 4
    v_bits: int = 8
    use_rotation_for_k: bool = True
    use_rotation_for_v: bool = True
    k_group_size: int = 128
    v_group_size: int = 128
    outlier_threshold: Optional[float] = None


@dataclass
class SchedConfig:
    total_budget: int = 4096
    recent_window: int = 64
    high_ratio: float = 0.25
    low_ratio: float = 0.60
    drop_strategy: str = "position"


@dataclass
class RankSearchConfig:
    rank_candidates: list = field(default_factory=lambda: [16, 32, 48, 64])
    r_k_candidates: list | None = None
    r_v_candidates: list | None = None
    rank_pair_candidates: list | None = None
    output_dir: Path = Path("artifacts/ranks")
    # Max training steps per candidate during rank search.
    # Overrides ``projector.n_steps`` when present. If missing / None,
    # the rank search runner falls back to ``projector.n_steps``.
    n_steps: int = 1500
    relative_tolerance: float = 0.10
    logits_abs_tolerance: float = 1e-6
    attn_abs_tolerance: float = 1e-5
    value_abs_tolerance: float = 1e-4
    # --- signal-normalized selection mode ---
    selection_mode: str = "constraint"
    logits_signal_tolerance: float = 0.01
    attn_signal_tolerance: float = 0.01
    value_signal_tolerance: float = 0.02
    layer_tolerance_scale: list | None = None
    layer_rank_floor: list | None = None


@dataclass
class TrainConfig:
    batch_size: int = 4
    epochs: int = 100
    lr: float = 1e-4
    device: str = "cpu"


@dataclass
class LogConfig:
    level: str = "INFO"
    dir: Path = Path("./logs")


@dataclass
class ServingConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 4


@dataclass
class HAWPConfig:
    logit_scale_mode: str = "rk"
    gamma_mode: str = "learned"
    gamma_value: float | None = None
    use_archive_k_ip_approx: bool = True


@dataclass
class EvalPPLConfig:
    seq_len: int = 1024
    nsamples: Optional[int] = 32


@dataclass
class EvalNeedleConfig:
    context_lens: list[int] = field(default_factory=lambda: [512, 1024, 2048, 4096])
    depths: list[int] = field(default_factory=lambda: [0, 25, 50, 75, 100])
    max_new_tokens: int = 32


@dataclass
class EvalSpeedConfig:
    seq_lens: list[int] = field(default_factory=lambda: [512, 1024, 2048, 4096])
    max_new_tokens: int = 64


@dataclass
class EvalDistributionConfig:
    enabled: bool = True
    seq_len: int = 512
    nsamples: Optional[int] = 8
    top_k: list[int] = field(default_factory=lambda: [1, 5, 10])
    seed: int = 0


@dataclass
class EvalLongBenchConfig:
    enabled: bool = False
    data_dir: Path = Path("data/longbench")
    tasks: list[str] = field(default_factory=list)
    max_new_tokens: int = 128


@dataclass
class EvalConfig:
    modes: list[str] = field(default_factory=lambda: ["baseline", "quant_only", "hawp_quant"])
    output_dir: Path = Path("artifacts/eval")
    ppl: EvalPPLConfig = field(default_factory=EvalPPLConfig)
    needle: EvalNeedleConfig = field(default_factory=EvalNeedleConfig)
    speed: EvalSpeedConfig = field(default_factory=EvalSpeedConfig)
    distribution: EvalDistributionConfig = field(default_factory=EvalDistributionConfig)
    longbench: EvalLongBenchConfig = field(default_factory=EvalLongBenchConfig)


@dataclass
class HAWPLAQConfig:
    mode: str = "local"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    calib: CalibConfig = field(default_factory=CalibConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    attention_distill: AttentionDistillConfig = field(default_factory=AttentionDistillConfig)
    layer_distill: LayerDistillConfig = field(default_factory=LayerDistillConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    sched: SchedConfig = field(default_factory=SchedConfig)
    rank_search: RankSearchConfig = field(default_factory=RankSearchConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    log: LogConfig = field(default_factory=LogConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    hawp: HAWPConfig = field(default_factory=HAWPConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _is_optional_like(tp) -> bool:
    """Return True if *tp* is Optional[T] / T|None (typing.Union or types.UnionType)."""
    if tp is None:
        return False
    args = get_args(tp)
    return len(args) >= 2 and type(None) in args


def _unwrap_optional(tp) -> type | None:
    """Return the concrete type inside Optional[T] / T|None, or None."""
    if tp is None:
        return None
    args = get_args(tp)
    if not args:
        return None
    non_none = [a for a in args if a is not type(None)]
    if len(non_none) == 1 and len(args) == 2:
        return non_none[0]
    return None


def _coerce_scalar(val, field_type):
    """Coerce a YAML scalar *val* to match *field_type*.

    Handles: float, int, Optional[float], Optional[int], float|None, int|None.
    bool is never coerced to int.  None passes through.  Path is handled
    separately by the caller.
    """
    if val is None:
        return None
    if isinstance(val, bool):
        return val

    concrete = field_type
    if _is_optional_like(field_type):
        concrete = _unwrap_optional(field_type)
        if concrete is None:
            return val

    if concrete is float and isinstance(val, str):
        return float(val)
    if concrete is int and isinstance(val, str):
        return int(val)
    return val


def _to_dataclass(cls: type, raw: dict[str, Any]) -> Any:
    init_kwargs: dict[str, Any] = {}
    hints = get_type_hints(cls)
    for f in cls.__dataclass_fields__:
        if f not in raw:
            continue
        val = raw[f]
        field_type = hints.get(f)
        if isinstance(val, dict) and hasattr(field_type, "__dataclass_fields__"):
            init_kwargs[f] = _to_dataclass(field_type, val)
        elif field_type is Path:
            init_kwargs[f] = Path(str(val))
        else:
            init_kwargs[f] = _coerce_scalar(val, field_type)
    unknown = set(raw.keys()) - set(cls.__dataclass_fields__.keys())
    if unknown:
        warnings.warn(
            f"Unknown fields in {cls.__name__}: {sorted(unknown)}. "
            f"These fields are not recognized and will be ignored.",
            UserWarning,
            stacklevel=3,
        )
    return cls(**init_kwargs)


def load_config(path: str | Path) -> HAWPLAQConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _to_dataclass(HAWPLAQConfig, raw)


_SUPPORTED_METHODS = ("turbo_mse", "turbo_prod")


def _check_method(method: str) -> None:
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported quant method '{method}'. "
            f"Supported: {_SUPPORTED_METHODS}"
        )


def build_k_quantizer(
    cfg: HAWPLAQConfig,
    r_k: int,
    device: str | None = None,
):
    """Build the K quantizer from config.

    Default method is ``turbo_prod`` (TurboQuantProd: MSE + 1-bit residual)
    which preserves inner-product fidelity for attention score computation.

    Args:
        cfg: HAWP-LAQ configuration.
        r_k: Latent key dimension.
        device: Optional device override.

    Returns:
        A ``TurboQuantProd`` or ``TurboQuantMSE`` instance.

    Raises:
        ValueError: If ``cfg.quant.k_method`` is not supported.
    """
    _check_method(cfg.quant.k_method)
    if cfg.quant.k_method == "turbo_prod":
        from hawp_laq.runtime.turboquant import TurboQuantProd
        return TurboQuantProd(
            dim=r_k,
            bits=cfg.quant.k_bits,
            use_rotation=cfg.quant.use_rotation_for_k,
            group_size=cfg.quant.k_group_size,
            device=device,
        )
    from hawp_laq.runtime.turboquant import TurboQuantMSE
    return TurboQuantMSE(
        dim=r_k,
        bits=cfg.quant.k_bits,
        use_rotation=cfg.quant.use_rotation_for_k,
        group_size=cfg.quant.k_group_size,
        device=device,
    )


def build_v_quantizer(
    cfg: HAWPLAQConfig,
    r_v: int,
    device: str | None = None,
):
    """Build the V quantizer from config.

    Default method is ``turbo_mse`` (TurboQuantMSE) which minimizes
    reconstruction MSE for value aggregation.

    Args:
        cfg: HAWP-LAQ configuration.
        r_v: Latent value dimension.
        device: Optional device override.

    Returns:
        A ``TurboQuantMSE`` or ``TurboQuantProd`` instance.

    Raises:
        ValueError: If ``cfg.quant.v_method`` is not supported.
    """
    _check_method(cfg.quant.v_method)
    if cfg.quant.v_method == "turbo_prod":
        from hawp_laq.runtime.turboquant import TurboQuantProd
        return TurboQuantProd(
            dim=r_v,
            bits=cfg.quant.v_bits,
            use_rotation=cfg.quant.use_rotation_for_v,
            group_size=cfg.quant.v_group_size,
            device=device,
        )
    from hawp_laq.runtime.turboquant import TurboQuantMSE
    return TurboQuantMSE(
        dim=r_v,
        bits=cfg.quant.v_bits,
        use_rotation=cfg.quant.use_rotation_for_v,
        group_size=cfg.quant.v_group_size,
        device=device,
    )


def resolve_projector_ranks(
    projector_cfg: ProjectorConfig,
    head_dim: int,
    mode: str = "hawp_quant",
) -> tuple[int, int]:
    r_k = projector_cfg.r_k
    r_v = projector_cfg.r_v
    rank = projector_cfg.rank

    has_partial = (r_k is not None) != (r_v is not None)
    if has_partial:
        raise ValueError(
            f"Must provide both r_k and r_v together (or neither). "
            f"Got: r_k={r_k}, r_v={r_v}"
        )

    if r_k is not None and r_v is not None:
        pass
    elif r_k is None and r_v is None and rank is not None:
        r_k = rank
        r_v = rank
    elif mode == "quant_only":
        r_k = head_dim
        r_v = head_dim
    else:
        raise ValueError(
            f"Cannot resolve projector ranks for mode '{mode}'. "
            f"Set projector.r_k and projector.r_v (or projector.rank as alias), "
            f"or use mode='quant_only' for explicit full-rank. "
            f"Got: r_k={r_k}, r_v={r_v}, rank={rank}"
        )

    if not (1 <= r_k <= head_dim):
        raise ValueError(
            f"r_k must satisfy 1 <= r_k <= head_dim, got r_k={r_k}, head_dim={head_dim}"
        )
    if not (1 <= r_v <= head_dim):
        raise ValueError(
            f"r_v must satisfy 1 <= r_v <= head_dim, got r_v={r_v}, head_dim={head_dim}"
        )

    return r_k, r_v


def load_projector_ranks_from_dir(projector_dir: str | Path) -> dict[int, tuple[int, int]]:
    projector_dir = Path(projector_dir)
    ranks_path = projector_dir / "ranks.json"
    if not ranks_path.exists():
        return {}
    from hawp_laq.utils.io import load_json
    raw = load_json(ranks_path)
    return {int(k): (v["r_k"], v["r_v"]) for k, v in raw.items()}
