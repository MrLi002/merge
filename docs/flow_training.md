# 稠密光流、训练与评测接口

当前实现是固定窗口、无历史初始化的独立监督训练基线，保留官方 E-RAFT 网络。训练不导入速度/位姿滤波器，不把官方测试脚本改名当训练。首轮建议先跑合成格式样例，再下载真实 DSEC 和官方权重。按用户 2026-09-24 的补充要求，不再自动下载这些资源。

## 1. 实际 DSEC-Flow 监督目录

`data.root` 指向包含序列子目录的目录，不自动猜测 train/test 结构。将官方下载的对应事件、校正映射、前向光流与时间戳合并为：

```text
data/dsec/train/
  zurich_city_01_a/
    events/left/events.h5
    events/left/rectify_map.h5
    flow/forward/000000.png
    flow/forward/000001.png
    flow/forward_timestamps.txt
  zurich_city_02_a/
    ...
```

依据 [DSEC 官方格式](https://dsec.ifi.uzh.ch/data-format/)、E-RAFT loader 和同作者团队 [BFlow 监督目录定义](https://github.com/uzh-rpg/bflow/blob/master/data/dsec/subsequence/base.py) 核对目录；体素仍采用 E-RAFT 原约定，不使用 BFlow 的扩展未来窗口：

- HDF5 包含 `/events/{t,x,y,p}`、`/ms_to_idx`、`/t_offset`；原始 t 为微秒，先加 offset 再转秒。极性为 0/1，x/y 为原始整数像素。`hdf5plugin` 提供官方 Blosc/Zstd 解码。
- `rectify_map[y,x]` 返回校正后的浮点 x/y，再三线性累积。映射文件路径使用官方 loader 的单数 `rectify_map.h5`。
- 前向 PNG 为 uint16 RGB：R 水平位移，G 垂直位移，B 为 0/1 有效掩码。位移 `(channel-32768)/128`。OpenCV 解码先 BGR→RGB，不能把第一通道当 mask。
- `forward_timestamps.txt` 每行两个整数微秒时间，行序对应排序后的前向 PNG；光流并不覆盖所有事件区间，禁止用固定文件序号推测时间。默认 100ms 位移。
- 单个样本需要 `[t0-dt,t0)` 和 `[t0,t1)` 的完整事件覆盖。越界记录有计数；空/退化体素、无有效监督按明确规则不做优化，不当零光流 GT。

裁剪同步作用于两个体素、光流和 mask；水平翻转同时取反 u 分量。当前训练不做图像缩放，避免遗漏位移缩放。跟踪不裁剪/缩放，padding 反填充保证像素与相机内参一致。

## 2. 训练配置和恢复

`configs/train_dsec.json` 的 `model` 指定 bins、迭代数、窗口、归一化、设备；`data` 指定 root、训练/验证序列、裁剪及增强；`training` 指定输出、epochs/累计 max_steps、batch、workers、seed、lr、weight_decay、epsilon、clip_grad、损失 gamma、max_flow、保存/验证频率及冻结 BN；`initialization.checkpoint` 为微调起点，null 代表从头训练。

训练/验证序列必须非空且无交集，不能把相邻窗口随机拆入两者。示例序列名单是待用户按下载内容修改的实验划分。验证 EPE 汇总有效像素，不能当目标区域精度。多次迭代的 L1 以 gamma 衰减加权，仅计有效有限且幅值阈值内的 GT。无监督样本跳过并计数。

模型和预处理元数据、优化器、调度器、进度、配置、划分与随机状态随 checkpoint 保存；恢复会检查兼容性。恢复要求原数据内容不变：当前核对路径、配置和划分，不计算整个大型训练集的内容哈希；CUDA 非确定性模式不保证逐位一致。`--max-steps` 表示累计目标步数，不是追加数量，也不改变配置中的余弦调度总长度。恢复写入新输出目录，保护旧实验。

```powershell
python -m ev6d train-flow --config configs/train_dsec.json
python -m ev6d train-flow --config configs/train_dsec.json --resume output/dsec_finetune/last.pt --output output/dsec_resume
python -m ev6d evaluate-flow --config configs/train_dsec.json --checkpoint output/dsec_finetune/last.pt --output output/dsec_eval.json
```

本实现未复现 E-RAFT 论文跨样本可微 warm-start。小样例使用完整网络和真实梯度，仍只是管线测试。不能据其 loss/EPE 得出真实数据精度或微调收益结论。

## 3. FlowResult、缓存与跟踪输入

`FlowResult(flow,t_start,t_end,source_frame,valid_mask,available_at,confidence,preprocessing)`：

| 字段 | 约定 |
| --- | --- |
| flow | float `[B,2,H,W]`，水平/垂直前向位移，pixel/interval |
| t_start / t_end | 秒，后者严格较大；事件分段用半开区间 |
| source_frame | `event_rectified` 等明确的源相机平面 |
| valid_mask | bool `[B,H,W]`；无效像素保存有限占位值 |
| available_at | 不早于 t_end；传感器时间轴的可用时间 |
| confidence | 标准 E-RAFT 为 null；若外部提供必须注明 confidence_source |
| preprocessing | bins/窗口/归一化/体素方法、checkpoint、padding、事件诊断、计算耗时 |

NPZ 中的 metadata 是 JSON 字符串，加载不允许 pickle。预计算生成 manifest 及 `flow_*.npz`，绑定事件内容和标定/元数据，复用时校验序列、时间与图像几何。它保证同一前端结果可用于多个后端配置。

`DenseSequence` 适配已经存在的 schema-1 文件，不改变旧 triplet 的 pixel/s 声明；稠密 FlowResult 有自己的单位。深度先经明确的 `T_event_depth` 变换、Z-buffer 配准，再选源时刻已到达且不过期的历史深度；不把终点深度配给起点光流，不用单一物距替代逐像素 Z。供给 mask 必须在校正事件平面，与深度采集时间一致；RGB/depth 原生掩码须先在导入阶段完成标定重投影，代码会拒绝静默共面假设。

如果模型为 cuboid，可用当时已知的源位姿投影配合深度筛选；其他目标需提供源平面 mask。已有 YCB 的理想 mask、带噪模拟位姿和已知初始位姿分别标记；已知合成/GT 初始化需 `--allow-oracle-pose`，不等于真实外部初始估计。

外部 DOPE 文件接口沿用 `convert-dope`，明确长度单位、pose-frame 和 `t/available_at`。新稠密后端有界重放延迟位姿；旧 triplet 仍拒绝历史位姿，不能混用其延迟验证结论。

## 4. 目标区域 EPE

真实目标序列若提供起点平面光流 GT，可用以下独立评估，不从跟踪估计制造真值：

```powershell
python -m ev6d evaluate-flow-cache --prediction output/demo_flow --ground-truth data/my_sequence/flow_gt.json --output output/target_flow_metrics.json
```

GT 同样用 `FlowResult.save` 保存，以 manifest 显式声明来源。起止时间、坐标平面和 shape 必须与预测一致；每区间只能匹配一次：

```json
{
  "kind": "ground_truth_flow",
  "flow_unit": "pixel/interval",
  "source": "name of measured or synthetic ground-truth generator",
  "source_frame": "event_rectified",
  "records": [
    {"file": "gt/flow_000000.npz", "target_mask": "gt/source_mask_000000.npy"}
  ]
}
```

`target_mask` 是 **GT 起点时间、同一平面** 的 H×W bool/0-1 数组。指标分别汇总所有有效像素与目标像素 EPE，同时报告缺失预测数；没有目标 mask 或有效目标像素时 `target_epe_px=null`，不报告假零误差。

## 5. 下游指标、可视化和实验对照

`evaluate` 使用保存的 GT，报告位置 RMSE、旋转角、空间线速度和角速度 RMSE，并将 `v_reference=v_O+omega×p` 与对应 GT 比较。GT 缺失时仅写定性报告/位置曲线。`visualize` 的 3D 对照要求 GT，`reproject` 不要求 GT；后者使用 `inverse(T_event_rgb)` 及 RGB 内参/畸变投影坐标轴和尺寸包围盒，记录离线插值，不是滤波观测或网格可见性渲染。

需要 ADD/ADD-S 时，在 dataset.json 显式提供：

```json
{"evaluation_model": {"points": "model_points.npy", "units": "m", "metric": "ADD"}}
```

points 为 N×3 的物体坐标点，与位姿原点一致；`ADD-S` 明确使用最近邻对称评估。没有此声明就不猜测物体对称性或网格坐标。

冻结官方 E-RAFT、微调 E-RAFT 与 triplet 的比较必须固定下游序列、几何、观测来源及时间。测试集合与训练序列分开；GT 初始化、理想 mask、合成位姿须分别披露。分别记录体素、同步后的 GPU 网络、观测整理、KF/UKF 和端到端耗时；窗口等待、配置的可用延迟和实际计算耗时不是同一个量。当前没有实机驱动与端到端硬实时验证。
