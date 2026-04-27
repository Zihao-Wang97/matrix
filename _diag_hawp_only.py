import torch
from hawp_laq.config import load_config
from hawp_laq.runtime.generate import _convert_and_load_projectors, load_baseline_model
from hawp_laq.modeling.attention_hawp import HAWPAttention

cfg = load_config("configs/dev_local.yaml")
model, tokenizer, device = load_baseline_model(cfg)
result = _convert_and_load_projectors(model, cfg, device, "hawp_only")
model = result[0] if isinstance(result, tuple) else result
model.eval()

attn = None
for m in model.modules():
    if isinstance(m, HAWPAttention):
        attn = m
        break

pk = attn.p_k.float()
print(f"p_k is identity? {torch.allclose(pk, torch.eye(64), atol=1e-4)}")
print(f"p_k max deviation from I: {(pk - torch.eye(64)).abs().max().item():.6f}")
print(f"r_k={attn.r_k}  head_dim={attn.head_dim}  p_k.requires_grad={attn.p_k.requires_grad}")
print(f"_apply_pk is NO-OP? {attn.r_k >= attn.head_dim and not attn.p_k.requires_grad}")
print(f"_is_low_rank={attn._is_low_rank}")
print(f"logit_scale_mode={attn.logit_scale_mode}")

prompt = "Hello, my name is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

with torch.no_grad():
    out_hawp = model(input_ids, use_cache=False)

# Fresh baseline
model2, tokenizer2, _ = load_baseline_model(cfg)
model2.eval()
with torch.no_grad():
    out_base = model2(input_ids, use_cache=False)

diff = (out_hawp.logits - out_base.logits).abs().max().item()
print(f"\nMax logit diff hawp_only vs baseline: {diff:.6f}")
print(f"Identical? {diff < 1e-4}")

# Check logit scale
import math
if attn.is_opt:
    rk_scale = math.sqrt(attn.head_dim) / math.sqrt(attn.r_k)
    dh_scale = math.sqrt(attn.head_dim) / math.sqrt(attn.head_dim)
    print(f"\nrk mode scale = sqrt(dh)/sqrt(rk) = sqrt({attn.head_dim})/sqrt({attn.r_k}) = {rk_scale:.4f}")
    print(f"dh mode scale = sqrt(dh)/sqrt(dh) = {dh_scale:.4f}")
    print(f"When r_k=head_dim, rk mode equals dh mode: {abs(rk_scale - dh_scale) < 1e-6}")
