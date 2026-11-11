"""Fine tune the 930D estimator with 135N stepping replay, retaining standing accuracy.

The previous test splits are never read here. Validation averages performance
over both force protocols, with a constraint against losing high-force accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from force_action_branch_v2 import load_action_force_estimator


def load_many(paths):
    parts = []
    for path in paths:
        with np.load(path) as z:
            parts.append({k: z[k].copy() for k in ("x", "force_body_n", "phase", "run_id")})
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


@torch.inference_mode()
def evaluate(model, data, device):
    x = data["x"]
    y = data["force_body_n"]
    pred = np.concatenate([model(torch.from_numpy(x[s:s+8192]).to(device)).cpu().numpy()
                           for s in range(0, len(x), 8192)])
    err = np.linalg.norm(pred-y, axis=1)
    phase = data["phase"]
    return {"active_mae_n": float(err[phase == 1].mean()),
            "pre_force_mae_n": float(err[phase == 0].mean()),
            "release_mae_n": float(err[phase == 2].mean()),
            "active_frames": int((phase == 1).sum())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/personal_fast_storage/Route_B_SuperMimic"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--lr", type=float, default=4e-5)
    args = p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(args.out)
    root = args.root
    low_dir = root / "data/force_aug_full930_test_v1"
    high_dir = root / "data/critical_spring_force_full930_v1"
    constant_dir = root / "data/force_constant_high_full930_v1"
    low_train = load_many([low_dir / f"train_{i}.npz" for i in range(3)])
    high_train = load_many([high_dir / f"train_{i}.npz" for i in range(3)])
    constant_train = load_many([constant_dir / f"train_{i}.npz" for i in range(3)])
    x90_train = load_many([constant_dir / f"x90_train_{i}.npz" for i in range(3)])
    adaptive_train = load_many([root / "data/adaptive_step_force_full930_v1/train.npz"])
    low_val = load_many([low_dir / "val_0.npz"])
    high_val = load_many([high_dir / "val_0.npz"])
    constant_val = load_many([constant_dir / "val_0.npz"])
    x90_val = load_many([constant_dir / "x90_val_0.npz"])
    adaptive_val = load_many([root / "data/adaptive_step_force_full930_v1/val.npz"])
    for a, b in ((low_train, low_val), (high_train, high_val),
                 (constant_train, constant_val), (x90_train, x90_val),
                 (adaptive_train, adaptive_val),
                 (low_train, high_train), (low_train, constant_train),
                 (high_train, constant_train), (x90_train, low_train),
                 (x90_train, high_train), (x90_train, constant_train),
                 (adaptive_train, low_train), (adaptive_train, high_train),
                 (adaptive_train, constant_train), (adaptive_train, x90_train)):
        if set(a["run_id"]) & set(b["run_id"]):
            raise ValueError("Run IDs overlap")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    source = root / "models/force_action_mixed_v3_release/seed0/best.pt"
    model = load_action_force_estimator(source, device=device)
    model.requires_grad_(True)
    # Decoder is not used by force inference, and remains frozen.
    for parameter in model.vae.decoder.parameters():
        parameter.requires_grad_(False)
    tensors = {}
    index = {}
    for name, data in (("low", low_train), ("high", high_train),
                       ("constant", constant_train), ("x90", x90_train),
                       ("adaptive", adaptive_train)):
        tensors[name] = (torch.from_numpy(data["x"]).to(device),
                         torch.from_numpy(data["force_body_n"]).to(device))
        index[name] = {"active": np.flatnonzero(data["phase"] == 1),
                       "release": np.flatnonzero(data["phase"] == 2),
                       "pre": np.flatnonzero(data["phase"] == 0)}
    optimizer = torch.optim.AdamW((v for v in model.parameters() if v.requires_grad),
                                  lr=args.lr, weight_decay=1e-4)
    args.out.mkdir(parents=True)
    config = {"source": str(source), "seed": args.seed, "epochs": args.epochs,
              "steps_per_epoch": args.steps_per_epoch, "lr": args.lr,
              "train_runs": {"low": len(set(low_train["run_id"])),
                             "high": len(set(high_train["run_id"])),
                             "constant": len(set(constant_train["run_id"])),
                             "x90": len(set(x90_train["run_id"])),
                             "adaptive": len(set(adaptive_train["run_id"]))},
              "val_runs": {"low": len(set(low_val["run_id"])),
                           "high": len(set(high_val["run_id"])),
                           "constant": len(set(constant_val["run_id"])),
                           "x90": len(set(x90_val["run_id"])),
                           "adaptive": len(set(adaptive_val["run_id"]))},
              "batch": "per step/domain: 384 active + 256 release + 128 pre-force",
              "loss": "SmoothL1(pred/30, label/30); active weight 2, release 1, x90 release 4, adaptive release 2, pre 0.25",
              "selection": "minimum adaptive active + 0.25*adaptive release val MAE, with low/spring/grid/x90 active <= source + 1N"}
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    baseline_low = evaluate(model, low_val, device)
    baseline_high = evaluate(model, high_val, device)
    baseline_constant = evaluate(model, constant_val, device)
    baseline_x90 = evaluate(model, x90_val, device)
    baseline_adaptive = evaluate(model, adaptive_val, device)
    initial_low = baseline_low["active_mae_n"]
    initial_high = baseline_high["active_mae_n"]
    initial_constant = baseline_constant["active_mae_n"]
    initial_x90 = baseline_x90["active_mae_n"]
    history = [{"epoch": 0, "low": baseline_low, "high": baseline_high,
                "constant": baseline_constant, "x90": baseline_x90,
                "adaptive": baseline_adaptive}]
    best = float("inf")
    for epoch in range(1, args.epochs+1):
        model.train()
        for _ in range(args.steps_per_epoch):
            losses = []
            for name in ("low", "high", "constant", "x90", "adaptive"):
                x, y = tensors[name]
                ia = torch.from_numpy(rng.choice(index[name]["active"], 384)).to(device)
                ir = torch.from_numpy(rng.choice(index[name]["release"], 256)).to(device)
                ip = torch.from_numpy(rng.choice(index[name]["pre"], 128)).to(device)
                ids = torch.cat((ia, ir, ip))
                pred = model(x[ids]) / 30
                loss = F.smooth_l1_loss(pred, y[ids]/30, reduction="none").mean(-1)
                losses.append(2.0*loss[:384].mean() +
                              (4.0 if name == "x90" else 2.0 if name == "adaptive" else 1.0)*loss[384:640].mean() +
                              0.25*loss[640:].mean())
            total = sum(losses)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        lm = evaluate(model, low_val, device)
        hm = evaluate(model, high_val, device)
        cm = evaluate(model, constant_val, device)
        xm = evaluate(model, x90_val, device)
        am = evaluate(model, adaptive_val, device)
        history.append({"epoch": epoch, "low": lm, "high": hm, "constant": cm,
                        "x90": xm, "adaptive": am})
        (args.out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"epoch={epoch} low={lm['active_mae_n']:.3f} spring={hm['active_mae_n']:.3f}"
              f" grid={cm['active_mae_n']:.3f} x90={xm['active_mae_n']:.3f}"
              f" x90_release={xm['release_mae_n']:.3f}"
              f" adaptive={am['active_mae_n']:.3f} adaptive_release={am['release_mae_n']:.3f}", flush=True)
        score=am["active_mae_n"]+0.25*am["release_mae_n"]
        if (lm["active_mae_n"] <= initial_low + 1 and
                hm["active_mae_n"] <= initial_high + 1 and
                cm["active_mae_n"] <= initial_constant + 1 and
                xm["active_mae_n"] <= initial_x90 + 1 and score < best):
            best = score
            torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "config": config, "epoch": epoch, "val": history[-1]}, args.out / "best.pt")
    print("best_adaptive_validation_score_n", best, "saved", args.out / "best.pt", flush=True)


if __name__ == "__main__":
    main()
