# E-RAFT 稠密光流的几何与滤波后端

本增量新增 `ev6d/dense_filter.py` 与 `ev6d/pose_replay.py`，复用既有 `geometry.py` 的刚体几何和 `filters.py` 的位置/四元数 UKF。原 `VelocityKF`、triplet/法向光流路径保持原状。本文件说明新后端；E-RAFT、数据和训练入口由项目主文档说明。

## 方法依据与实际代码核对

已逐页读取提供的事件 6-DoF 论文与 ROFT 论文文本，并核对官方 ROFT 提交 `cee79752d9bc3759cc2ed02c1e57a278ae342d6d` 的以下实现：

| 来源 | 原方法或官方行为 | 本项目选择 |
| --- | --- | --- |
| 事件 6-DoF 论文 III-B，式 (4)–(10) | 空间速度 `[v_O,ω]`；离散衰减模型；triplet 法向约束；Laplacian 权重 | E-RAFT 使用完整二维位移残差；默认常速度；噪声按秒定义；二维创新门控 |
| 事件 6-DoF 论文 III-C，式 (11)–(14) | 位置和单位四元数状态；速度作为输入；低频绝对位姿校正 | 维持这一状态定义，用已有流形 UKF；空间速度作精确常值积分 |
| ROFT 论文 III-B，式 (6)–(9) | 源时刻 mask/depth，二维位移，Jacobian 包含 `ΔT` | 同一几何观测关系；时间缩放只做一次 |
| ROFT `ImageOpticalFlowMeasurement.hpp` | `previous_depth_`、`previous_segmentation_`；`measurement_matrix *= sample_time_`；`check_observability` 仅判断点数至少 3 | 由数据层保证源时刻几何，KF 额外检查实际秩和条件数 |
| ROFT `SpatialVelocityModel.cpp` | `F=I`，每步固定 `Q` | 默认常速度，但 `Q(dt)=Qc·dt`，避免改变更新频率便改变先验 |
| ROFT `SKFCorrection.cpp` | 逐个二维测量更新；可选 Laplacian 权重 | 以 Cholesky 白化得到等价的小矩阵批量更新；无大型 `2N×2N` 求逆 |
| ROFT `CartesianQuaternionModel.cpp`、`CartesianQuaternionMeasurement.cpp`、`UKFCorrection.cpp` | 联合位置、参考点线速度、四元数、角速度状态；测量模型将参考点速度转换为空间速度 | 未混入默认状态，也没有宣称复现其联合后端 |
| ROFT `ROFTFilter.cpp` 和位姿 measurement buffer | 恢复过去 belief，再用保存的速度更新重放 | 有界时间戳历史，支持区间内部测量、乱序测量时刻、到达顺序验证、后续校正重放 |

官方代码链接：[ROFT 固定提交](https://github.com/hsp-iit/roft/tree/cee79752d9bc3759cc2ed02c1e57a278ae342d6d)。以上是方法核对，新 Python 文件为基于公开数学关系编写的实现，未粘贴官方 C++ 代码。官方仓库各文件许可证以其头部和仓库许可证为准。

事件 6-DoF 论文存在需要解释的公式差异：式 (2) 的 triplet 流为 pixel/s，式 (7) 又出现 `ΔT`；本项目接口明确选择 pixel/interval，不照抄含混的单位。式 (8) 第二行 `ω_x` 系数印为正号，而从本项目声明的 `X_dot=v_O+ω×X` 推导应为负号。ROFT 官方代码也是负号，独立三维投影有限差分测试验证此符号。

## 坐标、单位与速度定义

采用已校正事件图像平面与相匹配的零 skew 内参；`T_A_B` 将 B 坐标变换到 A。目标位姿为 `T_event_object`。位置单位米，时间单位秒；像素 x 向右、y 向下、相机 Z 向前。四元数顺序为 `[x,y,z,w]`，将物体坐标旋转到事件相机坐标；空间角速度在相机轴上表达，增量旋转左乘。

状态 `xi=[v_O,ω]` 满足 `X_dot=v_O+ω×X`。物体参考点 `p` 的线速度是 `v_reference=v_O+ω×p`；`v_O` 本身不是物体中心速度。只有物体坐标原点确实位于中心时，才能将 `v_reference` 称为中心速度。本后端输出相对相机的速度，不宣称世界坐标绝对运动。

令 `a=u-cx`、`b=v-cy`，则对源像素 `(u,v)` 及源深度 Z：

```text
J = [ fx/Z   0    -a/Z    -a*b/fy       fx+a*a/fx     -b*fx/fy ]
    [  0    fy/Z  -b/Z   -(fy+b*b/fy)    a*b/fx        a*fy/fx ]

F(t0 -> t1) ~= (t1-t0) * J * xi + epsilon
```

所有输入深度都是源相机 Z 深度，不能用终点深度同位置的值替代。上式是局部近似，有限区间内旋转、加速、遮挡会产生模型误差；窗口平均位移不能自动解释为终点瞬时速度。测试中采用小时间步验证速度恢复，不能据此声称 100 ms 窗口下模型误差可忽略。

位姿传播复用精确常值空间速度积分：`p'=R_delta p + J_left(ω dt) v_O dt`，`q'=q_delta ⊗ q`。这保留 `ω×p` 造成的参考点平移，比直接使用 `p+=v_O dt` 更符合约定。

## DenseVelocityKF

```python
from ev6d.dense_filter import DenseVelocityConfig, DenseVelocityKF

kf = DenseVelocityKF(DenseVelocityConfig())
kf.predict_to(t0)
kf.predict_to(t1)                  # 无观测时也预测并传播 Q
info = kf.update(uv, source_z, displacement_xy, K, dt=t1-t0)
xi, covariance = kf.x, kf.P
```

`uv` 为 `[N,2]`，`source_z` 为 `[N]`，`displacement_xy` 为 `[N,2]`。`update` 不改变 KF 时间戳，调度由调用者负责。只有在 `t1` 第二事件窗口收集完成后才可估计 `[t0,t1]` 的速度。若用这个区间速度积分对应位姿，也只能在 `t1` 可用时完成，不能把离线可用视作提前得到结果。

默认配置：最多 512 个分层点、最少 6 个点、二维 Mahalanobis 平方阈值 25、秩阈值相对最大奇异值 `1e-8`、条件数上限 `1e6`。条件数在 SI 速度单位对应的未做列归一化矩阵上计算，因此其阈值具有单位依赖性。较高点数并不能替代秩检查。

分层采样按候选像素包围盒划分网格，优先每格中心附近的一个点，再轮转填充，始终不超过数量上限。它减少空间重复信息，但不能消除神经网络的相关误差。默认 `noise_inflation=4` 是工程噪声膨胀系数，既不是 E-RAFT 原生置信度，也不是已标定不确定性。

默认 `flow_noise_std=1` 单位为 pixel/interval。若显式设置 `flow_noise_units="velocity"`，噪声标准差的单位是 pixel/s，在构造位移噪声时乘一次 `dt`。更新采用 `R_i=I·noise_inflation·sigma_displacement²`。零位移是有效的二维观测；缺失观测、空事件或没有深度则不能当作零位移。

连续过程模型默认不衰减。`process_linear_std=.5` 和 `process_angular_std=1` 分别是 `(m/s)/sqrt(s)` 和 `(rad/s)/sqrt(s)` 的扩散标准差。可设置 `decay_rate_per_s=λ>0` 使用连续 OU 先验，此时 `A=e^(-λdt)I`、`Q=Qc(1-e^(-2λdt))/(2λ)`，不是直接沿用论文每步的 `α=.5`。

门控根据预测先验的完整二维创新 `r_i` 与 `S_i=H_i P H_i^T+R_i` 计算 `r_i^T S_i^-1 r_i`；随后对保留点重新检查秩和条件数。批量校正令 `P=L L^T`、`A=H L/sigma`，仅求解 `I+A^T A` 的 `6×6` Cholesky。后验协方差直接从平方根因子构造，避免大矩阵求逆和减法造成的半正定性损失。

诊断结果包括输入/无效/采样/门控/保留数量、约束数、秩、条件数、残差 RMS、实际像素噪声、拒绝原因与协方差最小特征值。形状、时间、内参、状态或协方差异常直接报错；无效样本逐个剔除并计数。所有拒绝更新均保持预测状态和协方差，不把状态置零。

## PoseHistory 与延迟重放

```python
from ev6d.filters import PoseUKF
from ev6d.pose_replay import PoseHistory

pose = PoseHistory(PoseUKF(p0, q0_xyzw, t0), max_history_s=2., max_intervals=1000)
pose.predict_to(t1, xi_interval, covariance_interval)
result = pose.update(p_measured, q_measured_xyzw,
                     measurement_time=t_measurement,
                     arrival_time=t_arrival,
                     measurement_id="rgb-pose-42")
snapshot = pose.snapshot()         # 返回独立的数组副本
statistics = pose.diagnostics()
```

维护时间范围和数量上限；默认最多 1000 个 twist 区间及 1000 个 pose 测量。必须先预测到到达时间，不能提前融合未来测量。`measurement_time` 可乱序，`arrival_time` 必须按交付顺序非递减。过期测量、尚未到达的测量、容量超限和重复测量均有独立原因和计数；未来测量不被消耗，可待预测后再次提交。

每个预测记录起止时间、速度均值及协方差，每次延迟校正恢复有效历史起点，将过去测量按测量时间排序，在必要的区间内部时刻切分传播，再重放后续速度和绝对位姿校正。原滤波器外部引用仍然有效。重复调用相同终点的 `predict_to` 不重复积分；提供 `measurement_id` 可避免重复融合，同一缓存内相同时间/位置/等价四元数的无 ID 测量也会被去重。

位姿 UKF 在 `R³×SO(3)` 局部切空间生成 sigma 点，旋转均值、差值、扰动和协方差重置均遵守同一左扰动约定，处理 `q` 与 `-q` 等价性。速度协方差通过既有 UKF 的增广 sigma 点传播。这里仍是两阶段近似：忽略位姿与速度的互相关，以及不同时段速度输入的互相关；若绝对位姿位于某个速度区间内部，切分两段时重新增广同一速度协方差，没有维护跨两段共享的隐变量。测试验证与采用相同切分和近似的顺序滤波一致，不宣称为完整联合后验。被门控拒绝的内部位姿不会人为改变原预测的切分。

`accepted_on_arrival` 统计首次处理时接受的观测数量；历史重放会重新计算后续门控结果，因此 `accepted_in_history` 则报告当前缓存中实际接受的数量。重放改变当前结果，不回写已经输出的历史轨迹行；如需离线平滑轨迹，应独立标记为平滑结果。

## 实际验证

以下为增量开发前继承的历史记录；当前 `merge` 项目的本次验证见 [completion_validation.md](completion_validation.md)。

在本机 Windows、Python 3.13.2、NumPy 2.4.6、SciPy 1.17.1、OpenCV 4.13.0 环境执行：

```powershell
cd D:\postGraduate\code\ztgj\6DOF
python -m pytest tests/test_dense_geometry.py tests/test_pose_replay.py -q
```

结果：`31 passed`。覆盖独立矩阵指数推进三维点再投影的纯平移/纯旋转/混合运动恢复、六列 Jacobian 中央有限差分、单位与时间尺度、缩放/裁剪、等价批量 KF、无观测预测、无效深度、秩/条件拒绝、二维门控、有界采样、连续过程模型的更新频率一致性、四元数 antipode、延迟与顺序滤波等价、保留后续位姿校正、窗口边界、时间乱序、缓存超期/超限、重复调用和速度不确定性。

这些是 CPU 几何与滤波验证，不使用训练网络、真实 checkpoint 或真实传感器数据，不代表真实 E-RAFT 推理效果、真实数据精度或实时性。该后端未启动正式训练；真实网络/训练验证结果请看总交付报告。
