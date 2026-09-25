# E-RAFT + RGB-D 6DoF 研究基线

在现有 `ev6d` 上增量补齐：**E-RAFT 事件光流 → 完整二维观测的六维空间速度 KF → 位置/四元数 UKF + 低频位姿与延迟重放**。提供独立 DSEC-Flow 训练/微调、恢复、光流缓存、因果跟踪回放、评测及投影图；原 triplet 前端和命令保留。

这是三篇论文方法的工程组合，不代表论文精度或实时性复现。本次 **256 项测试通过**，已跑通短训练/恢复、权重加载及直接/缓存跟踪。已运行与未运行项目见 [本次验证](docs/completion_validation.md)，方法来源见 [审计报告](docs/method_audit.md)，附件逐项对应见 [实现索引](docs/requirements_mapping.md)，状态/公式见 [稠密几何](docs/dense_geometry.md)。[旧 README](docs/legacy_triplet_readme.md) 和旧实验记录保留作历史参考。

## 保护与来源

只修改当前 `merge`，未改写相邻 `6DOF` 的源码、数据和结果。源码原件已备份至 `backup_completion_20260923_194704/`。当前目录没有 Git 元数据；未创建提交或 PR。新训练、缓存和稠密跟踪拒绝覆盖非空输出目录，恢复训练使用新目录。命令在项目根目录执行，配置中的相对路径相对执行目录。

E-RAFT 固定提交 `c58ce0524ea0ebfa9849991caafb547f44fe9bfd`，ROFT 固定提交 `cee79752d9bc3759cc2ed02c1e57a278ae342d6d`。保留官方 E-RAFT 的 MIT 许可证与[兼容修改说明](ev6d/vendor/eraft/UPSTREAM.md)。原始 PDF 和完整需求文件的实际位置、逐项核对见方法审计。

## 安装

推荐 Python 3.10、PyTorch 2.5.1+cu124。选择适合驱动的 PyTorch，再安装项目：

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python -m pip install -e ".[flow,test,visualization]"
.venv/Scripts/python -m ev6d --help
```

Linux 改用 `.venv/bin/python`。CPU 使用 CPU PyTorch 和 `device: "cpu"`。旧 triplet 仅需 `.[test]`，YCB 渲染另需 `.[synthetic]`。本次已建立当前目录 `.venv`；实际硬件为 **RTX 2060，非 RTX 4090**。未启动正式长训练。

## 数据检查和已有基线

已核对相邻 `6DOF/data/ycb_suite_labelled/`：640×480 事件、60Hz RGB-D、事件平面掩码、模拟低频位姿和已知初始位姿。模拟位姿标记 `simulated_noisy_pose_NOT_DOPE`，不能当真实 DOPE 推理。输入沿用 [schema-1](docs/data_format.md)，新接口见 [训练及稠密数据](docs/flow_training.md)。

```powershell
python -m ev6d check-data --dataset ../6DOF/data/ycb_suite_labelled/005_tomato_soup_can/regular
python -m ev6d generate --output output/demo_data --scenario mixed --duration 0.4 --width 80 --height 60
python -m ev6d track --dataset output/demo_data --output output/triplet_result
python -m ev6d evaluate --dataset output/demo_data --result output/triplet_result
```

## E-RAFT 推理、缓存和跟踪

从 [官方 README](https://github.com/uzh-rpg/E-RAFT/tree/c58ce0524ea0ebfa9849991caafb547f44fe9bfd) 获取 [DSEC checkpoint](https://download.ifi.uzh.ch/rpg/ERAFT/checkpoints/dsec.tar)，保存为 `checkpoints/dsec.tar`，无需解压。本机遗留的 `.partial` 及不完整 tar 不能使用。严格核对键与形状；权重缺失不退回随机推理。

```powershell
python -m ev6d dense-config --output output/dense_defaults.json
python -m ev6d precompute-flow --dataset output/demo_data --checkpoint checkpoints/dsec.tar --config configs/dense_eraft.json --output output/demo_flow --device cuda
python -m ev6d track-dense --dataset output/demo_data --flow-cache output/demo_flow --config configs/dense_eraft.json --output output/dense_result --allow-oracle-pose
python -m ev6d evaluate --dataset output/demo_data --result output/dense_result
python -m ev6d reproject --dataset output/demo_data --result output/dense_result
python -m ev6d visualize --dataset output/demo_data --result output/dense_result
```

直接运行网络：

```powershell
python -m ev6d track-dense --dataset path/to/sequence --checkpoint checkpoints/dsec.tar --config configs/dense_eraft.json --output output/online_result --device cuda
```

已知合成/GT 初始化或 oracle 位姿须显式 `--allow-oracle-pose`，并记录来源；真实外部初始化不需要。`--velocity-only` 禁用低频观测并标记速度积分模式，`--max-intervals 2` 用于短验证。“在线”指按采集/到达时间因果回放，无传感器驱动或硬实时承诺。`visualize` 是需要 GT 的三维对照；无 GT 可用 `evaluate` 定性曲线和 `reproject`。

默认 100ms 固定窗口：`[t0-dt,t0)`、`[t0,t1)` 输出 `[1,2,H,W]` 前向位移 pixel/interval。网络使用最后全分辨率预测；后端只做一次 dt 转换。第二窗口结束及可用时间后更新速度，用于后续预测，不回写已发布轨迹。源深度/掩码在 t0 冻结，不能配终点几何。无 warm-start；标准 E-RAFT 没有原生已标定置信度。

深度使用有限年龄的历史采样，没有运动补偿。长窗口下的恒速、一阶光流关系是近似；不能缩放输出就声称 5/10ms 与训练条件等价。延迟位姿使用有界历史回溯重放；KF/UKF 间以及跨时段速度相关性仍作近似。

## 独立训练、微调和恢复

按 [DSEC 实际格式](docs/flow_training.md) 准备不同训练/验证序列，编辑 [训练配置](configs/train_dsec.json) 的目录、序列和权重。配置中序列是可改示例，不宣称官方划分。`initialization.checkpoint` 指定官方权重以微调，`null` 为从头训练。

```powershell
python -m ev6d train-flow --config configs/train_dsec.json
python -m ev6d train-flow --config configs/train_dsec.json --resume output/dsec_finetune/last.pt --output output/dsec_resume
python -m ev6d evaluate-flow --config configs/train_dsec.json --checkpoint output/dsec_finetune/last.pt --output output/dsec_eval.json
```

训练独立于 KF/UKF，包含迭代有效像素监督、验证、优化器/调度器、裁剪、种子和恢复，保存模型/配置/优化器/调度器/进度/随机状态/体素元数据和序列划分。无跨样本 warm-start 时序训练，与论文设置有区别。

只验证可运行性的小样例：

```powershell
python -m ev6d flow-fixture --output output/completion_20260923/dsec_fixture
python -m ev6d train-flow --config configs/train_flow_smoke.json --max-steps 2
python -m ev6d train-flow --config configs/train_flow_smoke.json --resume output/completion_20260923/flow_smoke/last.pt --output output/completion_20260923/flow_resume --max-steps 4
python -m ev6d evaluate-flow --config configs/train_flow_smoke.json --checkpoint output/completion_20260923/flow_resume/last.pt --output output/completion_20260923/flow_eval.json
```

若路径已存在请选新目录并修改配置。`--max-steps` 为累计目标优化步数。128×128 单像素平移 fixture 只检验训练/保存/恢复管线，权重可由跟踪器加载，但无真实精度意义。DSEC 驾驶光流数据不能替代目标物体 RGB-D/位姿下游数据。

## 输出和评估

| 文件 | 内容 |
| --- | --- |
| `trajectory.npz/csv` | 时间、位置、xyzw 四元数、`v_O`、角速度、参考点线速度 `v_reference`、状态与方差 |
| `runtime.json`、`diagnostics.json` | 来源、拒绝原因、深度年龄、观测数、重放、各模块及端到端耗时 |
| 缓存 `manifest.json`、`flow_*.npz` | 起止/可用时间、源坐标系、掩码、预处理、序列校验 |
| `metrics.json`、`errors.npz`、`tracking.png` | 有 GT 时的位姿、空间速度、参考点线速度、角速度误差 |
| `qualitative_report.json` | 无 GT 时的定性报告，不生成伪 RMSE |
| `reprojection/` | RGB 相机上的坐标轴及已知尺寸包围盒；非网格渲染 |

`T_A_B` 把 B 变到 A，秒/米/rad/s，`X_dot=v_O+omega×X`，`v_reference=v_O+omega×p`。参考点是物体坐标原点，未必几何中心；相对相机运动不宣称世界绝对速度。ADD/ADD-S 仅在显式提供物体点、米单位及指标约定时计算。

公平对照须固定划分、深度、掩码、RGB 位姿和到达时序，分别缓存冻结网络和微调网络，与同一序列的 triplet 比较。换权重不等于精度提升，网络 forward 时间不等于系统延迟。

资源准备好后可执行对照脚本；它记录权重哈希和共同输入，另在相同输出时刻插值评测，并明确记录旧 triplet 与新后端的延迟处理差异：

```powershell
python scripts/compare_frontends.py --dataset path/to/sequence --frozen-checkpoint checkpoints/dsec.tar --finetuned-checkpoint output/dsec_finetune/last.pt --dense-config configs/dense_eraft.json --output output/comparison
```

目标区域光流 GT 的 `evaluate-flow-cache` 用法见 [数据与训练说明](docs/flow_training.md)。

```powershell
python -m pytest -q
```

本次实际结果和资源缺口见 [交付验证](docs/completion_validation.md)。真实 DSEC、完整官方权重、带标定实测目标数据/外部位姿/真值、4090 全分辨率性能仍须验证。未新增位姿网络训练、实例分割、传感器驱动或 ROFT 联合速度/位姿 UKF。
