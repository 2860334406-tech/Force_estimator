# V4 Force Estimator Model Card

## Purpose

Estimate the current three-dimensional simulated external force applied to the
Unitree G1 torso from a 930-dimensional SuperMimic observation history.

## Deployment contract

- Frame: `torso_link`
- Unit: N
- Output order: `[Fx, Fy, Fz]`
- Sampling period: 20 ms
- Input must be the post-step policy observation
- Model internally normalizes its input

## Reported independent-test vector MAE

| Condition | MAE |
| --- | ---: |
| 0-30 N constant force | 2.08 N |
| 0-150 N spring pusher | 10.47 N |
| Multi-direction constant force | 12.13 N |
| +X 90 N with random release | 3.72 N |
| 135 N stepping replay | 6.71 N |

These values are simulation results and are not directly comparable across
different force frames or label definitions.

## Out of scope

- Hardware force accuracy
- Contact point estimation
- External moment estimation
- Future-force prediction
- Guaranteed closed-loop unloading improvement
