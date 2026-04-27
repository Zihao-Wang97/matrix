import torch
for i in range(12):
    d = torch.load(f"artifacts/projectors/layer_{i}/projector.pt", map_location="cpu", weights_only=False)
    rk = d["r_k"]
    rv = d["r_v"]
    g = d["gamma"].item()
    print(f"L{i:2d}  r_k={rk}  r_v={rv}  gamma={g:.4f}")
