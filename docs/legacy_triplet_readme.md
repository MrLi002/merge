# Event + Frame 6-DoF Tracking：独立方法复现

根据 Li 等人的 [6-DoF Object Tracking with Event-based Optical Flow and Frames](https://arxiv.org/abs/2508.14776) 补齐的 Python 实现。原目录已有数据适配、仿真和主流程，但缺失核心算法及入口。现在提供事件三元组光流、6D 速度 KF、四元数姿态 UKF、深度配准、DOPE 输出转换、离线评估和测试。

**这是依据论文建立的可运行方法复现，不是作者原始源码，也不代表复现了论文中的准确率或实时性能。** 未披露的参数、公式歧义、候选搜索限制及参考资料见 [复现说明](docs/reproduction_notes.md)。内置长方体与新生成的 YCB 数据均为独立合成序列，不是作者的 Unreal 原始数据。DOPE 提供输出适配器，需要用户另行运行官方模型；内置低频位姿是明确标注的带噪声模拟观测。

## 快速运行

需要 Python 3.10 或以上。在本目录运行：

```powershell
python -m pip install -e ".[test]"
python -m ev6d generate --output output/demo_data --scenario mixed --duration 1.2
python -m ev6d track --dataset output/demo_data --output output/demo_result
python -m ev6d evaluate --dataset output/demo_data --result output/demo_result
python -m pytest -q
```

也可以只安装 `requirements.txt` 后使用 `python -m ev6d`；安装项目后另有 `ev6d` 命令。无需下载模型或使用 GPU 即可运行合成验证。

查看或导出所有默认参数：

```powershell
python -m ev6d config --output output/defaults.json
python -m ev6d track --dataset output/demo_data --output output/custom_result --config configs/default.json
```

配置文件允许局部覆盖；未知参数会报错。默认值是可审查的工程初值，并非从论文恢复出的作者超参数。

已通过 146 项测试（包含 GPU 网格渲染），并运行七种场景及混合运动消融；实际误差、运行耗时和仍存在的问题见 [验证记录](docs/validation_results.md)。

## 生成论文方法所需的 YCB 合成数据

四个官方 YCB 物体可下载并生成 500 Hz 事件、60 Hz 运动模糊 RGB-D、5 Hz 模拟位姿及六自由度真值。详见 [YCB 数据集制作说明](docs/ycb_dataset.md)。快速命令：

```powershell
python -m pip install -e ".[synthetic,test]"
python -m ev6d fetch-ycb --assets assets/ycb
python -m ev6d generate-ycb --assets assets/ycb --output data/my_ycb_suite --duration 1 --width 640 --height 480
```

这是基于论文公开流程的新合成数据；所需原始 Unreal 场景和原作者数据未公开。YCB 网格渲染使用 OpenGL 3.3，适合有兼容显卡驱动的机器。

## 输出与实验

| 文件 | 内容 |
| --- | --- |
| `trajectory.npz` / `trajectory.csv` | 时间、位置、xyzw 四元数、空间速度；NPZ 另含姿态切空间方差 |
| `flows.npy` | 每行 `t_emit,x,y,flow_x,flow_y,depth,depth_valid,t_source,t_oldest_support` |
| `diagnostics.json` | 每批速度更新的权重、可观测性等诊断 |
| `runtime.json` | 输入/目标事件数、有效光流数、位姿拒绝计数、离线耗时和事件年龄 |
| `config.json` | 本次运行的完整参数 |
| `metrics.json` / `errors.npz` | 离线位置、旋转及速度误差；需要 ground truth |
| `tracking.png` | 轨迹和误差曲线；由 `evaluate` 生成 |
| `trajectory_3d.html` | 可离线打开的估计/真值三维轨迹和逐时刻姿态；由 `visualize` 生成 |

`track` 不读取 ground truth，也不导入仿真器或评估模块。只有 `evaluate` 使用 `ground_truth.npz`。评估在真值时间范围内插值位置并以 SLERP 插值四元数，不对真值范围之外外推。旋转误差使用相对旋转的 SO(3) 对数。

在服务器上生成交互式三维对比图，然后将 HTML 复制到本地浏览器打开：

```bash
python -m pip install -e ".[visualization]"
python -m ev6d visualize --dataset data/ycb_suite_labelled/005_tomato_soup_can/regular --result output/ycb_result
```

默认输出为 `output/ycb_result/trajectory_3d.html`。图中蓝线是真值轨迹，橙线是估计轨迹；拖动时间滑块可比较同一时刻的位置和物体坐标轴方向。位置真值按估计时间线性插值，姿态真值使用 SLERP，与 `evaluate` 一致。图中的坐标仍为事件相机坐标（X 向右、Y 向下、Z 向前），并未显示 YCB 网格或事件点云。

支持 `full`、`no_normal`、`no_weight`、`pose_only`、`velocity_only` 五种模式，例如：

```powershell
python -m ev6d track --dataset output/demo_data --output output/pose_only --variant pose_only
python scripts/validate_scenarios.py --output output/validation --duration 0.8
```

验证脚本运行静止、平移、旋转、混合运动、事件缺失、位姿缺失和异常事件/位姿共七种场景，并在混合场景运行全部五种模式。异常位姿在 t=0.6 s 注入，所以验证该项应使用大于 0.6 s 的时长。各场景和模式的指标汇总到 `summary.json`，不预设完整方法一定胜过所有消融。

## 输入自己的数据

先看 [数据格式](docs/data_format.md)。一个目录至少包含：

```text
dataset.json        标定、时间区间、初始位姿、模型/掩码和帧索引
events.npy          N x 4，按时间排序的 [t, x, y, polarity]
poses.csv           可选，外部低频位姿观测
frames/depth_*.npy  原始深度相机的 Z 深度，单位米
```

运行 `generate` 可获得完整、合法的格式示例。全流程统一使用秒、米、弧度、pixel/s。变换 `T_event_object` 把物体坐标转换为事件相机坐标；四元数顺序为 `qx,qy,qz,qw`。

DOPE 官方离线 JSON 可转换为观测文件：

```powershell
python -m ev6d convert-dope --input path/to/dope_json --timestamps path/to/timestamps.csv --dataset path/to/dataset --output path/to/dataset/poses.csv --object cracker --length-unit cm --pose-frame rgb
```

`timestamps.csv` 的列为 `file,t,available_at`（最后一列可省略，此时假定零延迟）。必须明确长度单位和输出相机坐标系。RGB 坐标下的位姿通过 `T_event_rgb` 转换。DOPE 类别名称须与 JSON 完全一致；同类多个目标会报错，需先选择目标实例。

## 当前边界

- 默认目标筛选使用已知长方体的投影轮廓加深度门限；任意形状目标须提供事件相机校正坐标下的逐帧掩码。YCB 合成数据生成器提供理想掩码，但尚未集成实例分割网络。
- 深度先反投影，再经外参投影到事件相机；用 Z-buffer 处理遮挡，空洞保持无效。过期或尚未到达的深度不参与速度更新。
- 位姿在其真实到达时间调度；**延迟到达的历史位姿会明确拒绝并计数**，当前没有固定滞后回放。请勿将时间戳改成到达时间来伪装零延迟。真实 DOPE 推理有延迟，接入实机前需扩展此部分。
- 速度使用上一批估计向前传播姿态；当前批光流更新影响后续预测，不倒填已经输出的历史姿态。
- 默认先积累 50 ms 事件历史，再用光流更新速度，以降低空历史启动时的错误匹配；可通过 `flow_warmup_s` 修改。位姿异常值门限默认关闭，配置 `pose.gate_threshold` 为正数可启用；过严的门限会拒绝正常高速运动观测。
- `v_o` 是相机原点处的空间速度分量，物体中心速度是 `v_o + omega × position`。二者不能混用。
- 没有事件不一定意味着静止，但本实现遵循论文的衰减先验；传感器掉线和真实停止之间无法仅凭空事件流区分。
- Python 参考实现以可读性、公式核验和离线验证为目标。耗时和事件年龄有记录，但未验证 640×480 实机实时运行；当前也没有传感器驱动。

## 代码位置

| 模块 | 职责 |
| --- | --- |
| `ev6d/flow.py` | 因果三元组匹配和 ROI 候选集合共识 |
| `ev6d/geometry.py` | 投影、空间速度雅可比、四元数、深度配准 |
| `ev6d/filters.py` | 速度 KF、法向约束、鲁棒权重、姿态 UKF |
| `ev6d/pipeline.py` | 事件、深度、位姿的因果调度与输出 |
| `ev6d/data.py` / `target.py` | 数据检查、DOPE 适配、目标事件筛选 |
| `ev6d/synthetic.py` | 独立 RGB-D 渲染、对数亮度阈值事件仿真 |
| `ev6d/ycb_synthetic.py` / `assets.py` / `mesh_renderer.py` | 官方 YCB 模型获取、纹理网格渲染和八条序列生成 |
| `ev6d/evaluation.py` / `cli.py` | 评估和命令行入口 |
| `tests/` | 数值、单位、因果性及端到端回归测试 |

主要公开参考：[论文](https://arxiv.org/abs/2508.14776)、[Triplet Matching](https://arxiv.org/abs/2212.12218)、[ROFT 官方实现](https://github.com/hsp-iit/roft)、[DOPE 官方实现](https://github.com/NVlabs/Deep_Object_Pose)。本项目独立编写，未复制这些仓库的代码或下载其训练权重。
