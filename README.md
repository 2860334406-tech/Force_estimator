# G1 Route B V4 力估计器

一个根据 Unitree G1 人形机器人本体状态历史，估计机器人躯干当前受到多少
三维外力的神经网络。

> 输入：机器人最近 10 帧本体状态与动作历史。
> 输出：作用在 `torso_link` 上的三维外力 `[Fx, Fy, Fz]`，单位为 N。

## 这是什么

机器人在被手、球体、弹簧推杆或其他物体推动时，通常希望知道：

- 力有多大。
- 力的方向是什么。
- 是否需要后撤、迈步或调整动作以保持平衡。

这个仓库提供的是一个已经训练好的 **V4 力估计器**。它读取机器人控制器本来
就能获得的本体观测，估计当前施加在机器人躯干上的外力。

模型结合了：

- VAE 本体状态编码器。
- 动作历史编码器。
- 三维外力输出头。

该模型主要面向 **仿真中的 SuperMimic + Unitree G1** 环境。
SuperMimic 是一个根据人体动作参考控制人形机器人运动的策略。

## 它能做什么

- 从 930 维本体观测中估计当前三维外力。
- 为力反馈控制、卸力策略和抗扰动策略提供输入。
- 分析机器人在碰撞或持续推力下的受力变化。
- 在相同数据契约下批量运行离线推理。

## 它不能做什么

当前 V4 模型不输出：

- 接触点位置。
- 多接触点集合。
- 外力矩。
- 未来时刻的力。
- 接触力的法向/切向分解。
- 新的关节动作。

这个模型也不是经过实机标定的力传感器。当前报告精度来自仿真数据，不能直接
视为真实机器人上的测量精度。

## 输入与输出

### 输入

输入必须来自 SuperMimic 的步后观测：

```python
next_obs["policy"]
```

形状：

```text
[batch_size, 930]
```

数据类型：

```text
float32
```

不要在外部再次做归一化，模型内部会使用训练时保存的均值和标准差。

930 维输入的组成：

| 输入内容 | 维度 | 索引 |
| --- | ---: | --- |
| 关节位置 | 10 × 29 = 290 | `[0:290]` |
| 关节速度 | 10 × 29 = 290 | `[290:580]` |
| 基座角速度 | 10 × 3 = 30 | `[580:610]` |
| 投影重力 | 10 × 3 = 30 | `[610:640]` |
| 上一动作历史 | 10 × 29 = 290 | `[640:930]` |

这些数据对应最近 10 个控制帧。控制周期为 **20 ms**。

输入中不包含：

- `obs["goal"]`。
- 真实外力标签。
- 推手位置。
- 接触点。
- 未来状态。

### 输出

输出形状：

```text
[batch_size, 3]
```

三个分量依次为：

```text
[Fx, Fy, Fz]
```

- 单位：N。
- 坐标系：`torso_link` 局部坐标系。
- 每个样本表示该控制帧估计的躯干外力。

例如：

```python
tensor([[12.5, -3.1, 48.2]])
```

表示当前估计力为：

```text
Fx = 12.5 N
Fy = -3.1 N
Fz = 48.2 N
```

## 最快运行方式

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 设置 Python 路径

在仓库根目录执行：

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
```

### 3. 加载并测试模型

```python
import torch

from force_action_branch_v2 import load_action_force_estimator

estimator = load_action_force_estimator(
    "models/v4_force_estimator.pt",
    device="cuda:0",  # 没有 GPU 时改成 "cpu"
)

policy_obs = torch.zeros((1, 930), dtype=torch.float32, device="cuda:0")

with torch.no_grad():
    force = estimator(policy_obs)

print(force.shape)  # torch.Size([1, 3])
```

如果这段代码能够输出 `[1, 3]`，说明权重、代码和运行环境已经正确加载。

## 完整调用顺序

模型必须使用 `env.step(action)` **之后**的观测，因为训练标签与该步执行后的
机器人状态对齐。

```python
from force_action_branch_v2 import load_action_force_estimator

estimator = load_action_force_estimator(
    "models/v4_force_estimator.pt",
    device="cuda:0",
)

obs, info = env.reset()

for step in range(max_steps):
    # 1. 根据当前观测生成动作。
    action = policy(obs["policy"], obs["goal"])

    # 2. 执行动作。
    next_obs, reward, terminated, truncated, info = env.step(action)

    # 3. 使用步后观测估计当前外力。
    with torch.no_grad():
        force_torso_n = estimator(next_obs["policy"].float())

    # 4. 下游控制、记录或可视化可以使用 force_torso_n。
    # 注意：加载估计器本身不会修改 policy 输出的动作。

    obs = next_obs

    if (terminated | truncated).any():
        break
```

常见错误是使用执行动作之前的 `obs`。正确对齐方式是：

```text
action_t -> env.step(action_t) -> next_obs_t -> 标签属于第 t 步外力
```

## 命令行推理

准备一个 NumPy 文件，形状为 `[930]` 或 `[N, 930]`：

```bash
python examples/infer.py \
  --checkpoint models/v4_force_estimator.pt \
  --obs observation.npy \
  --out prediction.npy
```

输入输出示例：

```text
observation.npy: [100, 930]
prediction.npy:  [100, 3]
```

## 模型结构

### 本体状态 VAE

```text
930 -> 512 -> 256 -> 128
```

输出：

```text
mu:     7 维
logvar: 7 维
```

推理时力头使用均值 `mu`。

### 动作历史编码器

```text
290 -> 64 -> 16
```

### 三维力输出头

```text
(7 + 16) -> 64 -> 64 -> 3
```

最后一层输出乘以 `30`，转换为以 N 为单位的力。

## 权重文件

```text
models/v4_force_estimator.pt
```

SHA-256：

```text
b1f4e81db9d5c31b5fd96224f6a8516256b0accf5cf8cccd867a3626e8482286
```

校验：

```bash
sha256sum -c models/SHA256SUMS
```

## 已报告的仿真测试结果

表中的误差为真实受力有效帧上的三维力向量 MAE：

```text
mean(||预测三维力 - 真实三维力||₂)
```

| 测试条件 | 三维力 MAE |
| --- | ---: |
| 0–30 N 恒力 | 2.08 N |
| 0–150 N 弹簧推手 | 10.47 N |
| 30/60/90/120 N 多方向恒力 | 12.13 N |
| 持续 +X 90 N、随机撤力 | 3.72 N |
| 名义 135 N 迈步轨迹离线重放 | 6.71 N |

结果文件：

- `evaluations/final_v4_comparison.json`
- `evaluations/final_v4_comparison.png`
- `docs/V4_inference_handoff.md`
- `docs/V4_evaluation_notes.md`

这些结果的前提是：

- 数据来自仿真。
- 力标签为仿真中施加到躯干的外力。
- 不同力坐标系、控制策略或接触模型不能直接套用这些数值。

## 使用边界

### 必须使用完整 930 维输入

不要截取前 640 维，也不要删除动作历史。

### 不要重复归一化

模型内部已经执行归一化。外部再次归一化会改变输入尺度，导致输出失真。

### 输出是躯干坐标系的力

如果下游程序使用世界坐标系、骨盆坐标系或其他安装坐标系，需要执行正确的
旋转和坐标变换。

### 仿真精度不等于实机精度

实机部署前需要重新定义：

- 可用传感器。
- 力坐标系。
- 低通滤波。
- 推理时延。
- 真值测量方式。

### 力估计准确不等于控制一定改善

模型只提供力估计。控制器如何使用该估计值、是否需要阻抗或导纳控制，需要
单独设计和闭环验证。

## 仓库结构

```text
src/
  force_action_branch_v2.py          V4 推理接口
  route_b_vae_full930.py             VAE 网络结构
  train_force_action_mixed_v4_adaptive.py
                                      V4 训练参考代码

models/
  v4_force_estimator.pt              V4 权重
  v4_config.json                     训练配置
  v4_history.json                    训练历史
  SHA256SUMS                         权重校验文件

docs/
  V4_inference_handoff.md            原调用交接文档
  V4_evaluation_notes.md             全工况复核记录
  MODEL_CARD.md                      模型卡

evaluations/
  final_v4_comparison.json           最终评估数据
  final_v4_comparison.png            最终评估图

examples/
  infer.py                           最小命令行推理示例
```

## 训练代码说明

`src/train_force_action_mixed_v4_adaptive.py` 是 V4 的训练参考代码。

本仓库没有包含：

- 原始训练数据。
- 上一阶段 V3 源权重。
- 完整的 SuperMimic 训练环境。

因此不能只凭本仓库直接从头复训 V4。训练文件主要用于查看数据接口、损失函数
和微调流程。

## 常见问题

### `ModuleNotFoundError: force_action_branch_v2`

在仓库根目录执行：

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
```

### 输入维度错误

确认输入为：

```text
[batch_size, 930]
```

### 模型输出异常或全部接近零

检查：

- 是否使用步后观测。
- 是否传入完整的 930 维。
- 是否错误地只传了前 640 维。
- 是否在外部重复归一化。
- 动作历史是否与当前观测正确对齐。

### GPU 不可用

将设备改为 CPU：

```python
estimator = load_action_force_estimator(
    "models/v4_force_estimator.pt",
    device="cpu",
)
```

### V4 是否支持接触点或未来力

不支持。V4 只输出当前三维躯干外力。多接触点、位置和未来力属于后续模型。
