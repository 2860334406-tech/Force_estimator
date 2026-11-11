"""Run V4 inference on a saved 930-dimensional observation array."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from force_action_branch_v2 import load_action_force_estimator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--obs", type=Path, required=True,
                        help="NumPy file with shape [930] or [N, 930]")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    observation = np.load(args.obs, allow_pickle=False)
    if observation.ndim == 1:
        observation = observation[None, :]
    if observation.ndim != 2 or observation.shape[1] != 930:
        raise ValueError(
            f"Expected observation shape [930] or [N, 930], got "
            f"{observation.shape}"
        )

    estimator = load_action_force_estimator(
        args.checkpoint,
        device=args.device,
    )
    observation_t = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=args.device,
    )

    with torch.no_grad():
        prediction = estimator(observation_t).cpu().numpy()

    print("input_shape:", observation.shape)
    print("output_shape:", prediction.shape)
    print("prediction_n:", prediction)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.out, prediction)
        print("saved:", args.out)


if __name__ == "__main__":
    main()
