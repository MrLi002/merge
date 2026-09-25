# 方法、原始材料与官方代码核对（2026-09-23）

完整阅读了工作区上级的 `Codex_E-RAFT_6DoF_Implementation_Prompt.md`，并阅读三篇论文全文。采用 PDF 技能核对原始文件，重新运行 Poppler `pdftotext -layout` 后，三份结果与 `research/{event6d,eraft,roft}.txt` 的文本逐字一致。另查看了方法页的完整渲染图，确认公式符号和双栏顺序，未只根据文件名判定材料。

## 原始 PDF 与版本记录

| 材料 | 本机原始 PDF | 页数 / 字节数 | 版本 |
| --- | --- | --- | --- |
| 6-DoF Object Tracking with Event-based Optical Flow and Frames | `D:/postGraduate/论文/姿态估计/6-DoF Object Tracking with Event-based Optical Flow and Frames.pdf` | 8 / 1,314,651 | arXiv:2508.14776v1, 2025-08-20 |
| E-RAFT: Dense Optical Flow from Event Cameras | `D:/postGraduate/论文/光流/E-RAFT：Dense Optical Flow from Event Cameras.pdf` | 10 / 14,575,487 | arXiv:2108.10552v3, 2021-10-21 |
| ROFT: Real-Time Optical Flow-Aided 6D Object Pose and Velocity Tracking | `D:/postGraduate/论文/姿态估计/ROFT：Real-Time Optical Flow-Aided 6D Object Pose and Velocity Tracking.pdf` | 8 / 4,840,949 | 作者接收稿，DOI:10.1109/LRA.2021.3119379 |

SHA-256（顺序对应上表）：

```text
event6d cc0a5dbdbf0fc9ab3b2f0146601aa13027927e00d6e06f4c4d136820e489aaaf
eraft   c2640fcbfc528654c4ce976e6e80c7a38da559c1f1ddedf7cdebd392f932763d
roft    719e5671cf64fe09545b78dce3037361d17e591bd8bf2a3756e797847725e0c1
```

通过本次用户指定的 GitHub 插件读取提交记录，核实本地代码采用以下固定版本，并未将其描述为最新版本：

- [E-RAFT c58ce0524ea0ebfa9849991caafb547f44fe9bfd](https://github.com/uzh-rpg/E-RAFT/commit/c58ce0524ea0ebfa9849991caafb547f44fe9bfd)，2022-10-08，修复 DSEC 16-bit PNG 读取。原始归档位于 `research/eraft_archive/E-RAFT-c58ce0524ea0ebfa9849991caafb547f44fe9bfd`。
- [ROFT cee79752d9bc3759cc2ed02c1e57a278ae342d6d](https://github.com/hsp-iit/roft/commit/cee79752d9bc3759cc2ed02c1e57a278ae342d6d)，2023-05-22，CI 调度变更。本地 `git -C research/roft rev-parse HEAD` 返回同一提交。

## 论文原方法、官方行为与项目选择

### E-RAFT 前端

核对 [model/eraft.py](https://github.com/uzh-rpg/E-RAFT/blob/c58ce0524ea0ebfa9849991caafb547f44fe9bfd/model/eraft.py)、`model/{extractor,update,corr,utils}.py`、`loader/loader_dsec.py`、`utils/dsec_utils.py`、`utils/image_utils.py`、`main.py`、`test.py` 和 `config/dsec_standard.json`。

论文 3.2 的两个连续窗口为 `E(t0-dt,t0)`、`E(t0,t1)`，预测 `F(t0→t1)`。官方代码保留这一约定：特征编码器处理两块体素，共享参数；上下文编码器输入第二块体素。`forward` 返回低分辨率位移和全分辨率迭代预测列表。跟踪取列表最后一项，输出 `[B,2,H,W]`、pixel/interval，通道为水平/垂直位移。

源码差异比较确认 `extractor.py` 与官方一致，`update.py` 仅有末尾空白差异，`corr.py` 仅调整包导入及显式 `meshgrid(indexing='ij')`。特征/上下文编码器、相关体次序、GRU、学习上采样、参数名及形状保留。项目修正了每次 forward 的 padding 状态重置，并将旧 autocast 调用更新为 `torch.amp.autocast`；混合精度仍按官方默认关闭。

官方图像使用左侧/顶部零填充至 32 的倍数。项目对较小样例额外填充到每轴至少 128，使四层相关体最小尺寸至少为 2，避免 `align_corners=True` 在单像素层除零。480×640 DSEC 输入不受此最小值影响；这不是缩放或新增训练策略，padding 会反向移除。

官方 DSEC 为 15 bins、100 ms、480×640、归一化打开；这些由读取的 loader/config 核实，不适用于任意权重。体素时间按窗口内首末事件归一化，极性映射为正负票，校正后的浮点坐标作三线性分配，仅非零元素用无偏标准差归一化。项目另外支持 `-1/+1` 极性，显式剔除并计数越界坐标，拒绝时间乱序，以 `[start,end)` 避免重复。官方边界像素的插值足迹可能保留部分越界事件；本项目选择整条剔除越界坐标，因此只声明有效图像范围内的数值一致。空事件、零时间跨度和无非零体素均标记为无有效观测，不生成可信零运动。

论文 3.3 的可微 warm-start 会向前传播旧位移，训练时作时序反传。当前基线没有跨样本 warm-start，也不宣称复现其时序训练。论文 3.4 对每次迭代的有效像素使用加权 L1（γ=0.8）；本项目训练入口独立实现监督优化和验证，而官方所检查入口主要用于推理。官方 DSEC 测试 loader 要求 `test_forward_flow_timestamps.csv`，不能原样当成有 GT 的训练 loader。

### 速度和位姿后端

核对 ROFT `ImageOpticalFlowMeasurement.hpp`、`SpatialVelocityModel.cpp`、`SKFCorrection.cpp`、`CartesianQuaternionModel.cpp`、`QuaternionModel.cpp`、`CartesianQuaternionMeasurement.cpp`、`UKFCorrection.cpp` 和论文 III-B/III-C。ROFT 用前一帧的深度和掩码；二维观测矩阵乘采样时间；速度 KF 为常速度，校正按小块处理。它的 UKF 联合状态包含位置、参考点线速度、四元数、角速度，速度测量关系为 `v_O=v_reference+p×ω`。

本项目按用户要求默认采用事件 6-DoF 论文 III-C 的 `[p,q]` 状态，速度作为外部输入，且保留 `p_dot=v_O+ω×p`。它与 ROFT 联合状态定义不同。现有 triplet 法向观测路径保留，新增 E-RAFT 路径使用完整二维位移残差和独立 `DenseVelocityKF`，不会调用旧法向残差默认值。默认过程先验为常速度，`Q=Qc·dt`；可选连续时间衰减，不直接复制事件论文每步 `α=0.5`。

四元数采用 `xyzw`、物体到相机、相机轴角速度、左乘增量；UKF 在 `R³×SO(3)` 的切空间计算均值/差值和扰动。常值空间 twist 以精确指数积分替代论文一阶 Euler 位置式。速度协方差通过增广 sigma 点传播，忽略两阶段之间、跨区间输入之间的相关性；这是明示近似，不等价于联合滤波。延迟位姿使用有界历史恢复、校正和重放。更详细公式、观测门控、噪声和重放限制见 `docs/dense_geometry.md`。

### 需要纠正的论文公式问题

事件论文第 3 页式 (8) 中第二行的 `ω_x` 系数，原 PDF 明确印为正号；ROFT 第 3 页式 (9) 与官方 C++ 则为负号。从声明的 `X_dot=v_O+ω×X` 和针孔投影求导得到正确项 `-(fy+(v-cy)^2/fy)`。因此本项目采用负号，并使用独立三维刚体变换与投影的中央有限差分验证，未照抄这个印刷符号。

事件论文式 (2) 的 triplet 量纲为 pixel/time，而式 (7) 又出现 `ΔT`。本项目明确规定 E-RAFT 输出为区间像素位移，`F≈Jξdt` 只做一次时间换算。它近似区间运动，不能直接声称是区间终点瞬时速度；100 ms 的几何线性化误差须在目标数据上评估。

## 权重兼容修复及实际可用性

原适配器仅调用 `torch.load(weights_only=True)`。实际官方下载文件的 `archive/data.pkl`（使用 `pickletools` 静态检查，不执行 pickle）包含 NumPy 标量指标：`numpy.core.multiarray.scalar` 与 `numpy.dtype`。直接安全加载会在现行 NumPy/PyTorch 下失败。现已在单次加载的作用域内允许有限的数字 dtype 和受限标量还原函数，同时兼容 NumPy 1.x/2.x 名称，绝不回退到 `weights_only=False`。任意对象、对象 dtype 和不匹配参数仍拒绝。

加载支持官方 `model`、`state_dict`、`model_state_dict` 或裸状态字典；一致的 `module.` 前缀可剥离。缺失、多余、shape 不匹配、非有限参数、空/非字典 metadata 均有诊断。已保存的项目权重检查 bins、窗口、归一化、体素方式和上下文约定；官方权重没有本项目 metadata 时，`checkpoint_metadata_verified=false`，不能把用户配置当成已从文件核实的信息。

2026-09-23 实际检查：

| 文件 | 字节数 | 结论 |
| --- | ---: | --- |
| `../6DOF/checkpoints/dsec.tar` | 8,847,897 | PyTorch ZIP 缺少中央目录，截断 |
| `../6DOF/checkpoints/dsec_complete.tar` | 239 | HTML 错误页，不是权重 |
| `checkpoints/dsec_official_download.partial` | 12,935,897 | PyTorch ZIP 缺少中央目录，截断 |
| `checkpoints/dsec_official_attempt_20260923.partial` | 1,719,897 | 本次独立下载 180 秒超时，保留为 partial |

所有原有文件保留。通过官方 README 的 [DSEC 权重地址](https://download.ifi.uzh.ch/rpg/ERAFT/checkpoints/dsec.tar) 实际 HEAD 请求返回 `Content-Length: 64171800`、`ETag: "3d32f18-5cf4463a9dab8"`、支持 byte range。长度只能发现截断，不能单独证明内容正确。本次下载网络约 9.5 KB/s，未完成；因此没有执行真正预训练权重推理或基于其精度的验证。没有将随机网络视作预训练模型。

## 本文件对应的实测验证

在当前 `merge` 工作区执行：

```powershell
./.venv/Scripts/python.exe -m pytest tests/test_eraft_adapter.py -q
```

结果 `8 passed in 4.20s`，Python 3.10、PyTorch 2.5.1+cu124、NumPy 2.2.6；此组测试明确使用 CPU，避免与短训练共享 GPU。覆盖安全加载的 NumPy 新旧命名、HTML/缺失权重/不支持对象、参数前缀/shape/非有限值/metadata 诊断、分数坐标/极性/无偏标准差、与归档官方体素实现数值一致、空事件/乱序/边界，以及完整 E-RAFT 的实际 forward、上下文输入、最终分辨率、切换尺寸、冻结推理和无观测处理。完整网络测试使用随机 fixture 权重，只验证实现连接和数值有限性。

原始 PDF 复核、官方源码差异和上述测试已经执行。真实 DSEC 全序列训练/精度、官方权重推理、真实物体跨模态配准后的跟踪精度、RTX 4090 性能均不能由这些测试替代。训练短跑和全项目回归以总验证报告为准。

## 引用与第三方代码

- Li, Glover, Bartolozzi, Natale. *6-DoF Object Tracking with Event-based Optical Flow and Frames*. arXiv:2508.14776v1, 2025.
- Gehrig, Millhäusler, Gehrig, Scaramuzza. *E-RAFT: Dense Optical Flow from Event Cameras*. 3DV, 2021.
- Piga, Onyshchuk, Pasquale, Pattacini, Natale. *ROFT: Real-Time Optical Flow-Aided 6D Object Pose and Velocity Tracking*. IEEE RA-L 7(1), 159–166, 2022; DOI:10.1109/LRA.2021.3119379.

E-RAFT vendored 文件保留 `ev6d/vendor/eraft/LICENSE` 的 MIT 声明及 `UPSTREAM.md` 来源和改动记录。ROFT 仓库保留 `research/roft/LICENSE.GPL-2`；主要源文件声明 GPL-2+，`UKFCorrection.cpp` 的头部另注明源自 BSD 3-Clause。Python 新后端按公开数学关系实现，未直接粘贴 ROFT C++；不会把仓库所有源文件统一标为 MIT。论文 PDF 为用户本机参考材料，未复制到交付源码目录。
