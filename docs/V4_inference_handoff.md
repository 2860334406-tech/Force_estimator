# V4 力估计器：调用交接

检查日期：2026-11-09。已核对本地权重、加载代码、观测配置和既有评估文件。本文件只说明如何使用已训练的网络，不包含训练流程。

## 文件位置

| 内容 | 本地路径 | VPS 路径 |
| --- | --- | --- |
| V4 权重 | `/media/lzx/Lexar/datasets/force_estimator/Route_B_SuperMimic/models/force_action_mixed_v4_adaptive/seed0/best.pt` | `/personal_fast_storage/Route_B_SuperMimic/models/force_action_mixed_v4_adaptive/seed0/best.pt` |
| 加载接口 | `/media/lzx/Lexar/datasets/force_estimator/Route_B_SuperMimic/scripts/force_action_branch_v2.py` | `/personal_fast_storage/Route_B_SuperMimic/scripts/force_action_branch_v2.py` |
| VAE 结构依赖 | `/media/lzx/Lexar/datasets/force_estimator/Route_B_SuperMimic/scripts/route_b_vae_full930.py` | `/personal_fast_storage/Route_B_SuperMimic/scripts/route_b_vae_full930.py` |
| SuperMimic 项目 | `/home/lzx/桌面/work/论文复现/SuperMimic/SuperMimic-Reproduction` | `/personal_fast_storage/SuperMimic-Reproduction` |
| 已验证的 VPS Python | — | `/personal_fast_storage/SuperMimic-Reproduction/.venv/bin/python` |

本地 V4 权重 SHA-256：`b1f4e81db9d5c31b5fd96224f6a8516256b0accf5cf8cccd867a3626e8482286`。使用 VPS 文件前可重新运行 `sha256sum` 核对。

## 输入与输出

- 输入是 SuperMimic 环境 `obs["policy"]`，形状 `[环境数, 930]`，`float32`。它是 **10 帧历史**：关节位置 `0:290`、关节速度 `290:580`、基座角速度 `580:610`、投影重力 `610:640`、上一动作历史 `640:930`；仿真控制周期 **0.02 s**。直接传原始策略观测，加载的模型内部使用训练时保存的均值和标准差归一化，不要在外面重复归一化。
- 输出形状 `[环境数, 3]`，顺序 `[Fx, Fy, Fz]`，单位 **N**，坐标系为 **`torso_link` 局部坐标系**。它估计的是仿真中加到躯干的三维外力；**不是**世界系或 `pelvis` 基座系，也不输出作用点、力矩、接触力或关节动作。
- 采集和监督采用 `env.step(action)` **之后**的 `next_obs["policy"]`，标签是该步的世界系施加力按**步后**躯干姿态旋转所得。在线调用应在一步仿真结束后估计这一步的力，供下一步控制使用。不要把新观测与下一步将施加的力对齐。
- `obs["goal"]` 属于基础策略输入，**不传给力估计器**。推理不能读取真实外力、接触力、施力点或标签。

上述合同见 [加载接口](/media/lzx/Lexar/datasets/force_estimator/Route_B_SuperMimic/scripts/force_action_branch_v2.py:31) 与 [采集代码](/media/lzx/Lexar/datasets/force_estimator/Route_B_SuperMimic/scripts/collect_force_aug_full930_test_v1.py:103)。

## VPS 最小调用

每次新开 SSH 会话，先设置环境变量：

```bash
export MUJOCO_GL=disable
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/personal_fast_storage/Route_B_SuperMimic/scripts:${PYTHONPATH:-}"
```

在使用 SuperMimic 虚拟环境的 Python 代码中：

```python
import torch
from force_action_branch_v2 import load_action_force_estimator

checkpoint = (
    "/personal_fast_storage/Route_B_SuperMimic/models/"
    "force_action_mixed_v4_adaptive/seed0/best.pt"
)
estimator = load_action_force_estimator(checkpoint, device="cuda:0")

# next_obs 是 env.step(action) 返回的步后观测；不裁成前 640 维。
with torch.no_grad():
    force_torso_n = estimator(next_obs["policy"].float())  # [N, 3]
```

`load_action_force_estimator` 已将模型置于 `eval()` 并关闭参数梯度。网络内部将 930 维观测编码为 VAE 均值，并单独编码最后 290 维动作历史，再由力头输出三维力；调用时只需要提供完整 930 维。若用于控制，必须另行设计从估计力到下一步动作的映射，加载网络本身不会改变 SuperMimic 动作。

## 已有验证与使用边界

评估指标为受力有效帧的 `mean(||预测三维力－真实三维力||₂)`，即三维向量 MAE。独立测试结果：0–30 N 恒力 **2.08 N**；名义 0–150 N 弹簧推手 **10.47 N**；30/60/90/120 N 多方向恒力 **12.13 N**；持续 +X 90 N、随机撤力 **3.72 N**；名义 135 N 迈步轨迹离线重放 **6.71 N**。详见 [评估汇总](/media/lzx/Lexar/datasets/force_estimator/Route_B_SuperMimic/evaluations/full930_prior_conditions_v1/final_v4_comparison.json) 和 [复核日志](/media/lzx/Lexar/datasets/force_estimator/Route_B_SuperMimic/实验日志/2026-11-09_930维力估计器全工况复核与修正.md)。

这些数字来自**仿真躯干外加力**，不代表闭环卸力效果或实机精度。135 N 数据是既有控制器生成的轨迹，V4 仅离线推理；不能据此宣称 V4 接入控制后会改善卸力。球形物理接触的标签不同，V4 接触帧误差为 **12.44 N**；20 秒持续 +X 90 N 在越过原站立阈值后的误差为 **90.51 N**。若新任务改成 `pelvis` 基座外力或真实接触力，需要重新定义坐标系、标签和验证集，不能直接把此权重当成对应任务的已验证模型。
