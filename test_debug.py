import torch
from hawp_laq.offline.projector_trainer import ProjectorTrainer

def orth_err(P):
    return torch.linalg.norm(P.T @ P - torch.eye(P.shape[1], dtype=P.dtype)).item()

B, T, H, dh = 2, 12, 4, 16
d_model = H * dh
rank_k, rank_v = 8, 6

trainer = ProjectorTrainer(
    d_model=d_model,
    rank_k=rank_k,
    rank_v=rank_v,
    n_heads=H,
    device="cpu",
)

common_kwargs = dict(
    n_steps=8,
    warmup_steps=2,
    row_batch_size=6,
    eval_every=2,
    patience=10,
    early_stopping=True,
    optimizer="riemannian_adam",
    seed=123,
)

print("=== Case A: [B,T,d_model] ===")
q = torch.randn(B, T, d_model)
k = torch.randn(B, T, d_model)
v = torch.randn(B, T, d_model)
r = trainer.train_one_group(q, k, v, **common_kwargs)
print("p_k", tuple(r["p_k"].shape), "p_v", tuple(r["p_v"].shape))
print("r", r["r_k"], r["r_v"], "best_step", r["best_step"], "best", r["best_calib_total"])
print("orth", orth_err(r["p_k"]), orth_err(r["p_v"]))

print("\n=== Case B: [B,H,T,d_h] ===")
q = torch.randn(B, H, T, dh)
k = torch.randn(B, H, T, dh)
v = torch.randn(B, H, T, dh)
r = trainer.train_one_group(q, k, v, **common_kwargs)
print("p_k", tuple(r["p_k"].shape), "p_v", tuple(r["p_v"].shape))
print("r", r["r_k"], r["r_v"], "best_step", r["best_step"], "best", r["best_calib_total"])
print("orth", orth_err(r["p_k"]), orth_err(r["p_v"]))

print("\n=== Case C: [B*H,T,d_h] ===")
q = torch.randn(B * H, T, dh)
k = torch.randn(B * H, T, dh)
v = torch.randn(B * H, T, dh)
r = trainer.train_one_group(q, k, v, **common_kwargs)
print("p_k", tuple(r["p_k"].shape), "p_v", tuple(r["p_v"].shape))
print("r", r["r_k"], r["r_v"], "best_step", r["best_step"], "best", r["best_calib_total"])
print("orth", orth_err(r["p_k"]), orth_err(r["p_v"]))

print("\n=== Negative Case: invalid last dim ===")
try:
    q = torch.randn(B, T, 10)
    k = torch.randn(B, T, 10)
    v = torch.randn(B, T, 10)
    trainer.train_one_group(q, k, v, **common_kwargs)
    raise AssertionError("Expected ValueError, but train_one_group succeeded")
except ValueError as e:
    print("ValueError OK:", str(e).splitlines()[0])

print("\nALL LOCAL SHAPE TESTS PASSED")
