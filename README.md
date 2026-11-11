# G1 Route B V4 力估计器

这是一个面向 Unitree G1 人形机器人的三维外力估计器。模型读取
SuperMimic 的 930 维本体状态历史，输出作用在机器人躯干上的三维外力。

仓库中包括：

- 已训练的 V4 权重。
- V4 推理代码。
- VAE 与动作历史编码器结构。
- 训练参考代码。
- 训练配置、评估结果和使用说明。

## 快速开始

### 1. 环境要求

- Python 3.10 或更高版本。
- PyTorch。
- NumPy。
- 如果有 NVIDIA GPU，建议使用 CUDA。

安装依赖：

```bash
pip install -r requirements.txt
```

### 2. 设置代码路径

在仓库根目录执行：

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
```

在 VPS 或没有图形界面的服务器上，还需要：

```bash
export MUJOCO_GL=disable
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}"
```

### 3. 加载模型

```python
import torch

from force_action_branch_v2 import load_action_force_estimator

checkpoint = "models/v4_force_estimator.pt"
estimator = load_action_force_estimator(
    checkpoint,
    device="cuda:0",  # 没有 GPU 时改为 "cpu"
)

# next_obs 必须是 env.step(action) 执行后的观测。
with torch.no_grad():
    force_torso_n = estimator(next_obs["policy"].float())

print(force_torso_n.shape)  # [batch_size, 3]
print(force_torso_n)        # [Fx, Fy, Fz]
```

## 模型输入

输入是 SuperMimic 环境中的：

```python
next_obs["policy"]
```

形状：

```text
[batch_size, 930]
```

数据类型应为 `float32`。

930 维输入由最近 10 帧本体状态和动作历史组成：

| 内容 | 维度 | 索引范围 |
| --- | ---: | --- |
| 关节位置 | 10 × 29 = 290 | `[0:290]` |
| 关节速度 | 10 × 29 = 290 | `[290:580]` |
| 基座角速度 | 10 × 3 = 30 | `[580:610]` |
| 投影重力 | 10 × 3 = 30 | `[610:640]` |
| 上一动作历史 | 10 × 29 = 290 | `[640:930]` |

控制周期为 **20 ms**，即 50 Hz。

## 模型输出

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
- 物理含义：仿真中施加到机器人躯干上的三维外力估计。

该模型**不输出**：

- 接触点。
- 外力矩。
- 多接触点集合。
- 未来时刻的力。
- 关节动作。

## 网络结构

### VAE 编码器

```text
930 -> 512 -> 256 -> 128
```

输出：

- 7 维隐变量均值 `mu`。
- 7 维隐变量对数方差 `logvar`。

### 动作历史编码器

```text
290 -> 64 -> 16
```

### 力输出头

```text
(7 + 16) -> 64 -> 64 -> 3
```

最后一层输出乘以 `30`，得到以 N 为单位的三维力。

## 权重信息

权重文件：

```text
models/v4_force_estimator.pt
```

SHA-256：

```text
b1f4e81db9d5c31b5fd96224f6a8516256b0accf5cf8cccd867a3626e8482286
```

校验权重：

```bash
sha256sum -c models/SHA256SUMS
```

## 命令行推理

准备一个 NumPy 观测文件，形状为 `[930]` 或 `[N, 930]`。

```bash
python examples/infer.py \
  --checkpoint models/v4_force_estimator.pt \
  --obs observation.npy \
  --out prediction.npy
```

输出文件中的形状为：

```text
[N, 3]
```

## 训练代码说明

`src/train_force_action_mixed_v4_adaptive.py` 是 V4 的训练参考代码。

仓库没有上传原始训练数据和上一阶段的 V3 源权重，因此不能仅凭本仓库直接
从头复现 V4 训练。该文件主要用于查看损失函数、数据接口和微调流程。

## 已报告的独立测试结果

以下指标是受力有效帧上的三维力向量 MAE：

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

详细结果见：

- `evaluations/final_v4_comparison.json`
- `evaluations/final_v4_comparison.png`
- `docs/V4_inference_handoff.md`
- `docs/V4_evaluation_notes.md`

## 使用注意事项

### 使用完整 930 维输入

不要只截取前 640 维，也不要自行删除动作历史。模型需要完整 930 维输入。

### 输入必须对齐到步后观测

训练时使用的是：

```python
next_obs, reward, terminated, truncated, info = env.step(action)
```

因此推理时应使用 `next_obs["policy"]`，不要错误地对齐到执行动作之前的观测。

### 不要重复归一化

模型内部已经保存并使用训练时的均值与标准差。外部再次归一化会破坏输入尺度。

### 当前输出使用躯干坐标系

输出位于 `torso_link` 局部坐标系。如果下游控制器使用世界坐标系、骨盆坐标系
或安装坐标系，必须自行完成旋转和平移转换。

## 适用边界

- 当前精度来自仿真实验，不代表实机传感器精度。
- 模型估计的是施加到躯干的三维外力，不等同于通用接触力传感器。
- 力估计准确不代表闭环卸力效果一定提升。
- 更换控制策略、力坐标系、接触模型或机器人柔顺参数后，需要重新验证。
- 球体接触、多接触点、持续大力和跌倒后的状态属于不同数据分布，应单独测试。

## 常见问题

### 出现 `ModuleNotFoundError: force_action_branch_v2`

确认在仓库根目录执行过：

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
```

### 出现维度错误

检查输入是否为：

```text
[batch_size, 930]
```

### 输出全为零或异常

检查输入是否已经归一化、是否使用了正确的观测历史，以及是否错误地截取了前
640 维。

### GPU 不可用

将加载模型时的设备改为：

```python
estimator = load_action_force_estimator(
    "models/v4_force_estimator.pt",
    device="cpu",
)
```

## 仓库目录

```text
src/          V4 推理代码、VAE 结构和训练参考代码
models/       V4 权重、训练配置、训练历史和校验文件
docs/         调用交接、评估说明和模型卡
evaluations/  最终评估数据和图片
examples/     最小命令行推理示例
```
