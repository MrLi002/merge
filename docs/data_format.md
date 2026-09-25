# 数据接口（schema_version = 1）

> 本文件原有的 pixel/s、延迟观测拒绝等描述对应 `track` triplet 路径。
> `track-dense` 的 FlowResult 使用 pixel/interval，并以 PoseHistory 实现有界延迟重放；
> 详细差异见 [稠密/训练格式](flow_training.md) 和 [几何后端](dense_geometry.md)。

建议先运行 `python -m ev6d generate --output output/example --duration 0.1`，以生成的 `dataset.json` 为模板替换数据路径。所有文件路径相对于数据集根目录，NumPy 数组禁止 pickle。

## 坐标、单位和时间

元数据须包含下面的精确声明：

```json
{
  "schema_version": 1,
  "units": {"time": "s", "length": "m", "angle": "rad", "flow": "pixel/s"},
  "pose_convention": "T_event_object; quaternion_xyzw; omega_event"
}
```

相机坐标为 x 向右、y 向下、z 向前。`T_A_B` 满足 `X_A = R_A_B X_B + t_A_B`，不接受相反方向的外参。角速度在事件相机坐标系表达，`dX/dt = v_o + omega × X`。所有设备时间须在导入前同步到同一个时间基准。

`start_time` 和 `end_time` 必须有限，且结束晚于开始。事件必须处于该闭区间内。`initial_pose` 包含 `position`（长度 3）和 `quaternion`（长度 4 的非零 xyzw 四元数）。初始位姿应由外部初始化器/首帧估计给出；合成器明确记录使用已知初始姿态。

## 标定 calibration

`event`、`depth`、`rgb` 各包含整数 `width,height`、3×3 内参 `K` 和可选 `distortion`。支持标准无 skew 的针孔内参和 OpenCV pinhole 畸变参数（空或 4/5/8/12/14 个），不直接支持鱼眼模型。

`T_event_depth` 和 `T_event_rgb` 是 4×4 刚体变换，旋转须属于 SO(3)。深度反投影时去除深度相机畸变，转换后投影到无畸变事件相机平面；原始事件同样去畸变，并四舍五入到整数像素以用于三元组匹配。

## 事件 events.npy

浮点数组形状为 `N×4`，列顺序 `t,x,y,p`。t 为秒，x/y 是原始传感器范围内的整数像素，p 可为 -1/+1 或 0/1。时间非递减，可以相同；算法不从零时间差计算速度。文件可含零事件，此时形状仍须为 `(0,4)`。原始 AEDAT/RAW/ROS bag 应先用相应传感器工具解码并转换单位，本项目不猜测原始格式和时间单位。

## frames

每项例子：

```json
{
  "depth_t": 0.1,
  "depth_available_at": 0.102,
  "depth": "frames/depth_0006.npy",
  "rgb_t": 0.1,
  "rgb": "frames/rgb_0006.png",
  "target_mask_event": "frames/mask_0006.npy"
}
```

`depth` 是深度相机原始分辨率的浮点数组，每个像素保存光轴 Z 深度（米），不是射线距离。NaN/非正深度无效。`depth_available_at` 可省略，默认等于采集时间；这是一项明确的零传输延迟假设。到达时间不得早于采集时间，当前不支持深度乱序。只在事件批次开始前已到达的深度可供该批使用，并检查其采集时间是否过期。

`rgb`/`rgb_t` 供外部位姿估计器或可视化使用，跟踪器不读取 RGB 来运行神经网络。`target_mask_event` 可选，为与校正事件图像同尺寸的布尔/0-1 NumPy 数组。一般目标需要此掩码；若元数据包含 `model: {"type":"cuboid","size":[0.1,0.2,0.3]}`，跟踪器使用预测姿态投影长方体生成掩码，模型长度单位为米。

## poses.csv

```csv
t,tx,ty,tz,qx,qy,qz,qw,source,available_at
0.1,0.0,0.0,1.0,0.0,0.0,0.0,1.0,dope_offline,0.1
```

`t` 是观测对应的采集时间，`available_at` 是结果真实到达时间。未提供后者时显式采用零延迟假设。算法不会把未来观测提前使用，也不把旧观测静默当成当前位姿；当前版本拒绝延迟历史观测，在 `runtime.json` 的 `late_pose_rejected` 中计数。缺少位姿文件表示仅靠速度与初始姿态传播。

## ground_truth.npz（仅评估）

包含 `t` (N)、`position` (N×3)、`quaternion` (N×4)，可选 `velocity` (N×6)。至少两条严格递增时间，所有数据有限，单位/空间速度约定与估计相同。跟踪器不需要此文件，评估器没有从生成函数重新获取真值。
