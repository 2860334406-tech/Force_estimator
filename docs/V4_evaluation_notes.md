# 930 维力估计器：既有工况复核与修正

检查日期：2026-11-09。证据范围：本地移动硬盘上的采集/训练/评估脚本、NPZ、模型配置与权重，及当日 VPS 实际运行产物。VPS：`/personal_fast_storage/Route_B_SuperMimic`；本地备份：`/media/lzx/Lexar/datasets/force_estimator/Route_B_SuperMimic`。原模型、原数据和原策略未覆盖。

## 1. 本次要回答的问题与指标

目标是核对已有 **930 维、力监督 VAE** 在过去使用的施力工况中估计三维外力的精度，并修复暴露出的可复现问题。部分历史视频/控制评估未保存完整 930 维观测，因此这里按同类施力协议重新采集独立轨迹；不声称逐一重放了过去每个视频的原始帧。训练/评估标签为同一步、`torso_link` 坐标系中的**仿真外加力**，单位 N；不是 pelvis 力、接触求解器力、机器人是否成功卸力，也不代表实机精度。

以下主要指标均为选定有效帧上的 `mean(||预测三维力－标签三维力||₂)`，即**三维向量 MAE**。受力期、施力前、撤力后分别计算；不同工况之间的力范围和帧数不同，只能在同一行比较两个模型。所有配对比较使用同一条轨迹、同一标签和同一有效帧。原始对照权重为 `models/force_action_branch_v2_long/full_seed0/best.pt`，修正权重为 `models/force_action_mixed_v4_adaptive/seed0/best.pt`。另外的 640 维低力/高力模型和原 930 维 seed1/2 也由评估脚本计算，详见各工况 `metrics.json`。

输入合同见 [推理模块](/media/lzx/Lexar/datasets/force_estimator/Route_B_SuperMimic/scripts/force_action_branch_v2.py:22)：930 维是 SuperMimic 的 10 帧本体观测历史 640 维和动作历史 290 维，模型以 VAE 均值和动作编码为力头输入，输出 `[N,3]`。采集脚本在 `env.step(action)` **之后**取 `next_obs["policy"]` 和施力后的躯干姿态，把世界系外力旋转为同一步躯干系标签；无效/终止帧通过相邻有效帧筛选排除。代码位置：[采集脚本](/media/lzx/Lexar/datasets/force_estimator/Route_B_SuperMimic/scripts/collect_force_aug_full930_test_v1.py:103)。力施加在 `torso_link`，偏心位置由四个局部偏移给出，世界力矩为 `rotate(torso_quat, offset) × force_world`，通过 `write_external_wrench_to_sim` 写入；采样周期 0.02 s。

## 2. 发现、修正与判定依据

**发现 1：原 930 维模型的训练工况与一部分旧测试工况不匹配。** 原权重 [配置](/media/lzx/Lexar/datasets/force_estimator/Route_B_SuperMimic/models/force_action_branch_v2_long/full_seed0/config.json) 只记录高力弹簧数据训练：768 条训练轨迹、256 条验证轨迹，力监督 VAE 的 930 维输入。它在新的 0–30 N 恒力、30–120 N 恒力、+X 90 N 持续力和 135 N 推手迈步轨迹上均有明显误差；特别是 +X 90 N 的撤力段预测残留 35.04 N。由训练覆盖与定向补充训练后的改善，可以认定**工况覆盖不足是重要原因**；不能由此断言是唯一原因。新增动作 290 维本身并未保证提高精度：低力测试中原 930 维 4.78 N，旧低力专用 640 维 2.92 N，二者训练集不同，不能单独归因于动作输入。

**修正方法：** 保留原 930 维 VAE+动作分支结构，从原模型开始分阶段微调，最终 v4 从 v3 初始化。v4 使用五组互不重叠的训练/验证轨迹：低力恒力 768/256、高力弹簧 768/256、高力恒力 768/256、+X 90 N 384/128、135 N 推手轨迹 192/64（最后一组按物理方向划分，包含四个控制分支）。具体训练采样、受力/撤力/无力损失权重及静态验证约束见 [训练脚本](/media/lzx/Lexar/datasets/force_estimator/Route_B_SuperMimic/scripts/train_force_action_mixed_v4_adaptive.py:53) 与 [v4 配置](/media/lzx/Lexar/datasets/force_estimator/Route_B_SuperMimic/models/force_action_mixed_v4_adaptive/seed0/config.json)。VAE 解码器冻结，估计器仍为端到端力监督的 VAE 编码器、动作编码器和力头；没有加入真实外力或接触力作推理输入。修正的重点是训练覆盖和撤力监督，并未更换基础控制策略。

**发现 2：动态控制分布也是重要误差来源。** 在旧 135 N 推手与迈步策略轨迹上，原 930 维为 23.50 N、v3 为 19.69 N；把这些轨迹按方向分成训练与验证后，v4 的**新方向新轨迹**误差为 6.71 N，原 930 维在同一批轨迹为 21.72 N。新轨迹由 [v10 清单](/media/lzx/Lexar/datasets/force_estimator/Route_B_SuperMimic/data/critical_pusher_forward_holdout16_v10_manifest.csv) 指定，共 16 个物理推手工况、四个既有控制分支、64 条分支轨迹。v10 不进入 v4 训练/选模，方向是新抽样但未保证与训练方向存在最小夹角。该轨迹仍由旧估计器/旧控制器在线运行；新模型只**离线重放估计**，不能据此宣称闭环卸力改善。时间对齐的内部检查：重算旧在线估计的最大差异 0.645 N，尚未发现一个足以解释 21.72 N 误差的整步错位；GPU 批大小数值差异仍存在。

## 3. 配对结果

| 工况与独立评估文件夹 | 受力有效帧/有力轨迹 | 实际受力均值 | 原 930D MAE | 修正 v4 MAE | 修正减原模型的轨迹重采样 95% 区间 |
|---|---:|---:|---:|---:|---:|
| 0–30 N 恒力 `final_low_v4` | 9750 / 206 | 20.53 N | 4.78 N | **2.08 N** | −2.95 至 −2.47 N |
| 名义 0–150 N 弹簧推手 `final_spring_base_v4` | 15016 / 221 | 60.14 N | 13.89 N | **10.47 N** | −3.94 至 −2.93 N |
| 30/60/90/120 N 多方向恒力 `final_grid_v4` | 18273 / 256 | 71.47 N | 23.84 N | **12.13 N** | −13.20 至 −10.32 N |
| +X 90 N 持续、随机撤力 `final_x90_v4` | 11096 / 128 | 90.00 N | 48.89 N | **3.72 N** | −45.79 至 −44.55 N |
| 名义 135 N 推手、四控制分支重放 `adaptive_step_135_fresh_v10` | 3618 / 16 个物理工况、64 条分支轨迹 | 随距离和运动变化 | 21.72 N | **6.71 N** | −18.13 至 −12.27 N |

区间是按**轨迹**配对重采样 5000 次所得，135 N 工况按 16 个物理推手工况重采样，四控制分支不被当作独立物理推手。逐帧与逐轨迹数据、全部分段指标见 [汇总 JSON](/media/lzx/Lexar/datasets/force_estimator/Route_B_SuperMimic/evaluations/full930_prior_conditions_v1/final_v4_comparison.json)、[对照图](/media/lzx/Lexar/datasets/force_estimator/Route_B_SuperMimic/evaluations/full930_prior_conditions_v1/final_v4_comparison.png) 与各行文件夹内的 `metrics.json`、`per_run.csv`、`predictions.npz`。评估公式和筛选见 [静态评估脚本](/media/lzx/Lexar/datasets/force_estimator/Route_B_SuperMimic/scripts/eval_full930_prior_conditions_v1.py:73)、[动态评估脚本](/media/lzx/Lexar/datasets/force_estimator/Route_B_SuperMimic/scripts/eval_adaptive_step_obs_force_v1.py)。

分段结果：低力施力前 0.553→0.441 N、撤力后 1.822→1.108 N；弹簧施力前 0.545→0.421 N、撤力后 3.959→2.429 N；高恒力施力前 0.565→0.503 N、撤力后 6.751→4.323 N；+X 90 N 撤力后 35.042→1.340 N；新 135 N 动态轨迹施力前 2.827→0.641 N、撤力后 5.954→1.344 N。低力受力期 v4 与 v3 接近（2.079 对 2.062 N）；高恒力 v4 比 v3 略差（12.127 对 11.931 N），不能声称所有旧模型上均逐项提升。高恒力实际 75–150 N 帧的 v4 误差仍有 **19.81 N**，是未解决的精度问题。

此前飞书表格中的 B 2.318 N / A 9.003 N 来自**另一批低力站立盲测轨迹及另一组旧权重**，与本表的测试集、模型和工况不同，不能把 2.318 N 直接当成本次 930 维的对照值。

## 4. 其他旧工况的边界测试

**球形推手物理接触：** 新采集 256 条、7414 个接触有效帧，目标是仿真接触求解器输出的接触力，并非本模型训练的 `write_external_wrench_to_sim` 外加力。原 930 维 13.135 N，v4 12.443 N；旧接触专用 640 维模型 4.934 N。详见 [接触迁移结果](/media/lzx/Lexar/datasets/force_estimator/Route_B_SuperMimic/evaluations/full930_prior_conditions_v1/sphere_contact_transfer_v4/metrics.json)。v4 在接触任务上仍明显不及专用模型，**不能**以外加力精度替代接触精度。无接触帧 v4 预测模长均值 1.000 N，旧接触专用门控模型 0.073 N。

**20 秒 +X 90 N 持续力：** 16 条轨迹都在施力后越过旧站立阈值，首次越线时间为实验开始后 3.30–4.68 s。越线前 1336 个受力帧原 930 维 48.415 N、v4 13.187 N；越线后 13032 个受力帧原模型 112.044 N、v4 **90.510 N**，甚至略差于总预测 0 N 的 90 N 误差。数据与逐帧预测见 [20 秒采集](/media/lzx/Lexar/datasets/force_estimator/Route_B_SuperMimic/data/force_long_full930_v1/x90_20s_seed20261131.npz) 和 [评估文件夹](/media/lzx/Lexar/datasets/force_estimator/Route_B_SuperMimic/evaluations/full930_prior_conditions_v1/long_x90_20s_v4/metrics.json)。因此 v4 **只验证了原站立阈值内的短时施力/推手轨迹，不支持摔倒后或任意长时大外力估计**。该工况没有因为越线而停止仿真，统计时必须明确越线前后。

## 5. 排查结论与仍待验证

| 结论 | 证据 | 对后续使用的影响 |
|---|---|---|
| 原模型训练覆盖不足是已证实的重要问题 | 原配置只有高力弹簧数据；定向补充低力、恒力、撤力、迈步轨迹后，同条件独立评估大幅降低误差 | 使用 v4 时仍应保持“外加躯干力、既有力范围/时长”的任务边界 |
| 930 维动作历史并非独立充分条件 | 原 930D 在低力新轨迹比旧低力 640D 误差更高，训练集不同；v4 保留 930D 结构而改善 | 不应把精度改善单独归因为增加动作历史 |
| 暂无明显整步错位证据 | 同步采集合同、重算旧在线估计最大差 0.645 N | 仍需在新控制器在线集成时检查动作、观测和力标签时序 |
| 接触求解器力是不同目标 | 球形接触测试 12.443 N，专用接触模型 4.934 N | 若任务改为真实接触力，须另定标签和训练集 |
| 倒地后长时施力不可靠 | 20 s 试验越线后 v4 误差 90.510 N | 不能据此做全程 20 s 或摔倒后力反馈控制 |
| 修正尚未验证闭环卸力收益 | 135 N 测试中控制器未换，只离线重放模型 | 下一步接入控制器后，要重新跑同条件闭环指标 |

仍未分离验证的可能因素：高恒力 75–150 N 段的方向/作用点分布与倒地姿态共同影响；力矩、估计不可观测性和姿态变化的各自贡献；未见过的控制器策略与实机噪声。现有结果只证明本仿真协议内的配对估计精度提高，不证明普适因果机制或真实接触力精度。

## 6. 可复现文件与校验

- 修正权重：[best.pt](/media/lzx/Lexar/datasets/force_estimator/Route_B_SuperMimic/models/force_action_mixed_v4_adaptive/seed0/best.pt)，SHA-256 `b1f4e81db9d5c31b5fd96224f6a8516256b0accf5cf8cccd867a3626e8482286`；本地与 VPS 已核对一致。VPS 路径：`/personal_fast_storage/Route_B_SuperMimic/models/force_action_mixed_v4_adaptive/seed0/best.pt`。
- 训练代码：[train_force_action_mixed_v4_adaptive.py](/media/lzx/Lexar/datasets/force_estimator/Route_B_SuperMimic/scripts/train_force_action_mixed_v4_adaptive.py)，训练记录：[history.json](/media/lzx/Lexar/datasets/force_estimator/Route_B_SuperMimic/models/force_action_mixed_v4_adaptive/seed0/history.json)。加载接口仍是 `force_action_branch_v2.load_action_force_estimator(checkpoint, device="cuda:0")`，输入 `[B,930]`，输出 `[B,3]`，单位 N、躯干系。
- 配对评估代码：[eval_full930_prior_conditions_v1.py](/media/lzx/Lexar/datasets/force_estimator/Route_B_SuperMimic/scripts/eval_full930_prior_conditions_v1.py)、[eval_adaptive_step_obs_force_v1.py](/media/lzx/Lexar/datasets/force_estimator/Route_B_SuperMimic/scripts/eval_adaptive_step_obs_force_v1.py)、[summarize_full930_v4_final.py](/media/lzx/Lexar/datasets/force_estimator/Route_B_SuperMimic/scripts/summarize_full930_v4_final.py)。动态新轨迹原始 NPZ/JSON：[v10 文件夹](/media/lzx/Lexar/datasets/force_estimator/Route_B_SuperMimic/evaluations/adaptive_step_obs_holdout_v10_raw)。
- 原始低力/高弹簧/高恒力/+X 90 N 采集数据在本地 `data/force_aug_full930_test_v1/`、`data/critical_spring_force_full930_v1/`、`data/force_constant_high_full930_v1/`；训练/验证标识在文件名和 JSON，测试文件只供评估。每个静态评估 `metrics.json` 记录输入 NPZ 的 SHA-256 以及所有 checkpoint SHA-256，可直接核对协议。

**仍未解决：** 物理接触力目标、倒地后持续 90 N、较大多方向恒力的残差误差、新控制器在线闭环效果、仿真外迁移。后续若要做闭环，先冻结此权重并设计新的独立控制评估，不要用本次离线 MAE 代替卸力指标。
