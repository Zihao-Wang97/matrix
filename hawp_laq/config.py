from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, get_type_hints, Union

import yaml


@dataclass
class DataConfig:
    root: Path = Path("./data")
    cache: Path = Path("./cache")


@dataclass
class ModelConfig:
    name: str = "hawp_laq_base"
    model_id: str = "facebook/opt-125m"
    backbone: str = "resnet50"
    pretrained: Optional[Path] = None
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


@dataclass
class ProjectorConfig:
    rank: int = 64
    r_k: Optional[int] = None
    r_v: Optional[int] = None
    lr: float = 1e-3
    n_steps: int = 200
    orthogonalize_every: int = 10
    target_layer: int = 0
    w_logits: float = 1.0
    w_attn: float = 1.0
    w_value: float = 0.5
    output_dir: Path = Path("artifacts/projectors")


@dataclass
class QuantConfig:
    k_group_size: int = 128
    v_group_size: int = 128
    use_rotation: bool = False
    outlier_threshold: Optional[float] = None


@dataclass
class SchedConfig:
    total_budget: int = 4096
    recent_window: int = 64
    high_ratio: float = 0.25
    low_ratio: float = 0.60


@dataclass
class RankSearchConfig:
    rank_candidates: list = field(default_factory=lambda: [16, 32, 64, 128, 256])
    tolerance: float = 0.02
    output_dir: Path = Path("artifacts/ranks")


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
class HAWPLAQConfig:
    mode: str = "local"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    calib: CalibConfig = field(default_factory=CalibConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    sched: SchedConfig = field(default_factory=SchedConfig)
    rank_search: RankSearchConfig = field(default_factory=RankSearchConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    log: LogConfig = field(default_factory=LogConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)


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
            init_kwargs[f] = val
    return cls(**init_kwargs)


def load_config(path: str | Path) -> HAWPLAQConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _to_dataclass(HAWPLAQConfig, raw)
