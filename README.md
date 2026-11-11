# G1 Route B V4 Force Estimator

This repository contains the trained V4 external-force estimator for the
Unitree G1 humanoid, together with its inference code, VAE definition,
training metadata and evaluation results.

## Model summary

- Checkpoint: `models/v4_force_estimator.pt`
- SHA-256: `b1f4e81db9d5c31b5fd96224f6a8516256b0accf5cf8cccd867a3626e8482286`
- Input: SuperMimic `obs["policy"]`, shape `[B, 930]`, `float32`
- Output: `[B, 3]`, ordered as `[Fx, Fy, Fz]`
- Unit: N
- Frame: `torso_link` local frame
- Control period: 20 ms

The model estimates a simulated 3D external force applied to the robot torso.
It does not output contact points, external moments, contact-force
decomposition, or future forces.

## Input contract

The 930-dimensional observation is a 10-frame history:

```text
joint_pos        10 x 29 = 290   [0:290]
joint_vel        10 x 29 = 290   [290:580]
base_ang_vel     10 x 3  = 30    [580:610]
projected_gravity 10 x 3 = 30    [610:640]
previous actions 10 x 29 = 290   [640:930]
```

Use the post-step observation returned by `env.step(action)`. Do not crop the
input to 640 dimensions and do not normalize it outside the model.

## Network

- VAE encoder: `930 -> 512 -> 256 -> 128`
- Latent distribution: 7-dimensional mean and log-variance
- Action-history encoder: `290 -> 64 -> 16`
- Force head: `(7 + 16) -> 64 -> 64 -> 3`
- Output linear layer scale: `30`

## Environment

```bash
export MUJOCO_GL=disable
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
```

## Python usage

```python
import torch
from force_action_branch_v2 import load_action_force_estimator

estimator = load_action_force_estimator(
    "models/v4_force_estimator.pt",
    device="cuda:0",
)

# next_obs is the post-step observation returned by env.step(action).
with torch.no_grad():
    force_torso_n = estimator(next_obs["policy"].float())

print(force_torso_n.shape)  # [batch_size, 3]
```

## Command-line example

Upload or create a NumPy observation with shape `[930]` or `[N, 930]`:

```bash
python examples/infer.py \
  --checkpoint models/v4_force_estimator.pt \
  --obs observation.npy \
  --out prediction.npy
```

## Repository layout

```text
src/          V4 inference and training-reference code
models/       V4 checkpoint, config, history and SHA-256
docs/         Inference handoff and evaluation notes
evaluations/  Final comparison data and figure
examples/     Minimal command-line inference example
```

## Scope and limitations

- Results are from simulation.
- The checkpoint estimates a force applied to the torso.
- The reported accuracy is not a hardware-sensor accuracy claim.
- The model does not establish closed-loop unloading performance.
- A different force frame, contact model or robot-control distribution requires
  independent validation.
