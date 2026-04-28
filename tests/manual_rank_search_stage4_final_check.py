from pathlib import Path
from tempfile import TemporaryDirectory
import torch

from hawp_laq.offline.rank_search import (
    infer_calib_dims,
    compute_signal_scales,
    search_rank_per_layer,
)
from hawp_laq.utils.io import load_json


def test_signal_shapes():
    print("=== signal shape tests ===")

    B, T, H, dh = 2, 12, 4, 16
    d_model = H * dh
    meta = {"hidden_size": d_model}

    cases = {
        "A_[B,T,d_model]": (
            torch.randn(B, T, d_model),
            torch.randn(B, T, d_model),
            torch.randn(B, T, d_model),
        ),
        "B_[B,H,T,dh]": (
            torch.randn(B, H, T, dh),
            torch.randn(B, H, T, dh),
            torch.randn(B, H, T, dh),
        ),
        "C_[B*H,T,dh]": (
            torch.randn(B * H, T, dh),
            torch.randn(B * H, T, dh),
            torch.randn(B * H, T, dh),
        ),
    }

    for name, (q, k, v) in cases.items():
        dm, hd = infer_calib_dims(q, H, meta)
        sig = compute_signal_scales(q, k, v, H, d_model=dm, head_dim=hd)

        print(name)
        print("  d_model:", dm, "head_dim:", hd)
        print("  signal:", sig)

        assert dm == d_model
        assert hd == dh
        assert sig["signal_logits"] > 0
        assert sig["signal_attn"] > 0
        assert sig["signal_value"] > 0

    try:
        bad = torch.randn(B, T, 10)
        infer_calib_dims(bad, H, meta)
        raise AssertionError("Expected ValueError for invalid shape")
    except ValueError as e:
        print("negative shape OK:", str(e).splitlines()[0])

    try:
        infer_calib_dims(torch.randn(B, T, d_model), None, meta)
        raise AssertionError("Expected ValueError for invalid n_heads")
    except ValueError as e:
        print("negative n_heads OK:", str(e).splitlines()[0])

    print("signal shape tests PASSED\n")


def test_rank_search_smoke():
    print("=== rank_search smoke test ===")

    B, T, H, dh = 2, 10, 4, 16
    d_model = H * dh

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        calib_dir = root / "calib"
        out_dir = root / "rank_out"
        calib_dir.mkdir(parents=True)

        torch.save(
            {
                "n_layers": 1,
                "n_heads": H,
                "hidden_size": d_model,
                "model_id": "dummy/model",
            },
            calib_dir / "meta.pt",
        )

        q = torch.randn(B, T, d_model)
        k = torch.randn(B, T, d_model)
        v = torch.randn(B, T, d_model)

        torch.save({"q": q, "k": k, "v": v}, calib_dir / "layer_0.pt")

        selected = search_rank_per_layer(
            calib_dir=calib_dir,
            rank_pairs=[(4, 4), (8, 6)],
            n_steps=6,
            warmup_steps=2,
            row_batch_size=5,
            eval_every=2,
            patience=10,
            early_stopping=True,
            min_delta=1e-4,
            min_delta_mode="relative",
            lr_pk=5e-3,
            lr_pv=5e-3,
            lr_xi=1e-2,
            beta1=0.9,
            beta2=0.99,
            grad_clip=1.0,
            lambda_z=1.0,
            lambda_o=2.0,
            lambda_v=0.05,
            gamma_min=1e-4,
            eps_loss=1e-8,
            adam_eps=1e-8,
            optimizer="riemannian_adam",
            selection_mode="constraint",
            relative_tolerance=0.10,
            output_dir=out_dir,
            device="cpu",
        )

        print("selected:", selected)
        assert 0 in selected
        assert selected[0] in [(4, 4), (8, 6)]

        json_path = out_dir / "layer_0_rank_search.json"
        assert json_path.exists(), f"missing {json_path}"

        data = load_json(json_path)
        print("json selected:", data["selected_r_k"], data["selected_r_v"])
        print("json best_calib_total:", data["best_calib_total"])

        assert "best_calib_total" in data
        assert "best_calib_logits" in data
        assert "best_calib_attn" in data
        assert "best_calib_value" in data
        assert "results" in data
        assert all("best_calib_total" in r for r in data["results"])

    print("rank_search smoke PASSED\n")


def test_missing_model_id_raises():
    print("=== missing model_id / n_heads test ===")

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        calib_dir = root / "calib"
        calib_dir.mkdir(parents=True)

        torch.save({"n_layers": 1, "hidden_size": 64}, calib_dir / "meta.pt")
        torch.save(
            {
                "q": torch.randn(2, 8, 64),
                "k": torch.randn(2, 8, 64),
                "v": torch.randn(2, 8, 64),
            },
            calib_dir / "layer_0.pt",
        )

        try:
            search_rank_per_layer(
                calib_dir=calib_dir,
                rank_pairs=[(4, 4)],
                n_steps=2,
                output_dir=root / "rank_out",
                device="cpu",
            )
            raise AssertionError("Expected ValueError when both n_heads and model_id are missing")
        except ValueError as e:
            msg = str(e)
            print("missing metadata OK:", msg.splitlines()[0])
            assert "n_heads" in msg and "model_id" in msg

    print("missing model_id / n_heads test PASSED\n")


if __name__ == "__main__":
    test_signal_shapes()
    test_rank_search_smoke()
    test_missing_model_id_raises()
    print("ALL MANUAL STAGE-4 FINAL TESTS PASSED")