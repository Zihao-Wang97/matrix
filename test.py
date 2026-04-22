from hawp_laq.config import load_config
from hawp_laq.runtime.generate import load_baseline_model, _convert_and_load_projectors
from hawp_laq.modeling.attention_hawp import HAWPAttention

cfg = load_config("configs/dev_local.yaml")
model, tok, dev = load_baseline_model(cfg)
model, r_k, r_v = _convert_and_load_projectors(model, cfg, dev, "hawp_only")

for m in model.modules():
    if isinstance(m, HAWPAttention):
        print("layer", m.layer_idx, "r_k", m.r_k, "r_v", m.r_v, "low_rank", m._is_low_rank)