# Event / RGB-D 6-DoF Object Tracker

**唯一主线：E-RAFT 事件光流 → 六维速度 KF → RGB 图像的 NVIDIA DOPE 位姿 → 四元数 UKF。**

基于 Li et al., *6-DoF Object Tracking with Event-based Optical Flow and Frames* 的两阶段滤波思路，并使用 E-RAFT 的稠密事件光流。项目按照 [Hybrid Object Tracking with Events and Frames](https://github.com/event-driven-robotics/Hybrid-object-tracking-with-events-and-frames) 的公开 UE/YCB 数据目录组织输入。

**已移除** Triplet、事件合成和 YCB 数据生成、DSEC 训练/微调、备选跟踪路线及与主流程无关的实验脚本。保留 E-RAFT 的 MIT 上游许可。DOPE 源代码、模型权重及公开数据集不复制进本仓库。

## 公开数据集

参考项目公开数据入口：[Zenodo 22897787](https://zenodo.org/records/22897787)。
请先下载并解压，程序不下载也不生成数据。目录：

```text
sequence/
├── filter_param.json
├── velocity.csv              # 真值或参考速度，仅供单独评估，不输入跟踪器
├── result.csv                # 本程序保存的六维速度（按光流更新时刻排序）
├── dope_poses_ycb.txt        # DOPE 独立生成的预测位姿，不是数据集真值
└── photorealistic1/
    ├── _camera_settings.json
    ├── _object_settings.json
    ├── binaryevents.log
    ├── xx.png
    ├── xx.depth.mm.16.png
    ├── xx.cs.png
    ├── xx.is.png
    ├── xx._ec.png
    └── xx.json               # 数据集标注真值，跟踪器只读取 timestamp
```

- 通过 `bimvee` 的 IIT-YARP importer 读取 `binaryevents.log`，单位秒。按数据集帧 JSON 时间戳构造事件对。若两时钟零点不同，使用 `event_time_offset_s` 显式修正。
- 从 `_camera_settings.json` 读取 fx/fy/cx/cy。16-bit depth PNG 由毫米变为米。
- `xx.is.png` / `xx.cs.png` 属于**标注掩码**（oracle segmentation）；需要显式 `--allow-annotation-mask`，所有实验标明 oracle mask，不声称已经实现目标检测/分割。
- 本适配器只支持已校准为**同一像素坐标平面**的 RGB/深度/事件，需在配置中显式设置 `assume_coregistered: true`。如果数据实际上未配准，必须先实现传感器重投影，不能直接混合像素或相机内参。
- `filter_param.json` 被读取并保留用于原数据记录；新六维 KF/UKF 的有效参数由 `configs/tracking.json` 设置，旧 1D 测量参数不直接套到 2D E-RAFT。
- `velocity.csv` 与每帧 `xx.json` 的 GT 位姿不用于滤波器初始化或校正。

## 安装

推荐 Python 3.10。先根据显卡安装 PyTorch，再运行：

```bash
python -m pip install -e .
python -m ev6d --help
```

依赖包括 NumPy、SciPy、OpenCV、PyTorch、bimvee。E-RAFT 为保留上游网络实现，默认 15 bins、100ms、12 次迭代；**必须提供完整兼容的预训练 checkpoint**，不允许随机网络替代。

## 用 RGB 跑 DOPE（需要官方仓库、目标权重和标定）

```bash
python -m ev6d prepare-rgb --sequence /data/sequence --output work/rgb
python -m ev6d run-dope \
  --dope-repo /path/to/Deep_Object_Pose \
  --rgb-dir work/rgb --output work/dope \
  --weights /path/to/ycb_weights.pth \
  --config /path/to/config_pose.yaml \
  --camera /path/to/camera_info.yaml \
  --object cracker
```

运行使用独立的 [NVIDIA DOPE](https://github.com/NVlabs/Deep_Object_Pose) 及其 Python 环境，输出位于 `work/dope/<weights_stem>/xx.json`。注意 `--camera` 应是 **RGB 相机的**标定。也可以直接使用公开数据自带的 `dope_poses_ycb.txt`，省去再次运行 DOPE 的步骤。

DOPE 输入解析约定：
- `dope_poses_ycb.txt` 每行 `x y z axis_x axis_y axis_z angle_rad`，没有时间戳时**必须与 RGB 帧数相同**；稀疏观测使用 `t x y z axis_x axis_y axis_z angle_rad` 八列。
- 或 `--dope-json-dir` 直接读取 NVIDIA DOPE 输出的 `xx.json`，字段 `objects[0].location`、`quaternion_xyzw`，缺失检测就跳过，绝不退回数据集 GT。
- 位置单位 `dope_translation_unit` 必须按实际 DOPE 数据选择 `m/cm/mm`；旋转统一为 xyzw；如 RGB 与事件相机不重合，提供 `T_event_rgb` 并完成图像配准。

## 单命令跟踪

```bash
python -m ev6d track \
  --sequence /data/sequence \
  --checkpoint /path/to/dsec.tar \
  --config configs/tracking.json \
  --output output/sequence_run1 \
  --device cuda \
  --allow-annotation-mask
```

若使用官方 DOPE JSON，追加 `--dope-json-dir work/dope/<weights_stem>`。 `--max-intervals 2` 可小规模验证。

输出：
- `sequence/result.csv` 六列 `l_v_x,l_v_y,l_v_z,a_v_x,a_v_y,a_v_z`，对应一个光流时间窗一次速度后验；保持参考仓库列名。
- `output/.../trajectory.csv` 逐时刻 `t,p_xyz,q_xyzw,v_O,omega,v_reference`。
- `output/.../diagnostics.json` 记录是否启用标注 mask、观测数量、光流来源、耗时、状态拒绝原因。

重要：`v_O` 是相机坐标系下的空间 twist 平移分量，物体参考点线速度为 `v_reference=v_O+omega×p`，两者不能混为一谈。光流是 **pixel/interval**，KF 显式乘一次 dt。DOPE 提供低频估计而非真实位姿真值。初始 DOPE 必须在其采集时间已可用；若存在初始延迟，需要另供可用初值。后续延迟观测使用有限历史重放。默认 DOPE 更新限制为 5 Hz（\`dope_max_hz\`），可修改配置。

为保护已有实验，如果 sequence/result.csv 已存在，跟踪结束时会拒绝覆盖；请先重命名旧文件。

## 验证边界

这是仅保留单一路线的工程重构；未声称已达到原论文的准确率、10ms 高频速度估计或实机实时性能。窗口约 100ms 带来更新延迟，DSEC 训练的 E-RAFT 在 UE/YCB 上存在域迁移风险。原始合成数据结果不是此公开数据集上的有效基准。必须使用真实官方权重、DOPE 目标权重、完整公开数据和独立精度评估后才能汇报最终结果。
