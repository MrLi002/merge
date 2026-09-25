# 附件要求与实现对应

附件为上级目录 `Codex_E-RAFT_6DoF_Implementation_Prompt.md`，以下按其 12 节逐项映射。实际执行范围以 [验证记录](completion_validation.md) 为准；有代码不等于完成实测精度验证。

| 附件要求 | 对应实现/说明 | 实验限制 |
| --- | --- | --- |
| 1 论文与官方版本、首版范围 | [方法审计](method_audit.md)，三份原 PDF 校验；官方固定提交；E-RAFT+二维 KF+[p,q] UKF | 未实现 ROFT 联合 UKF；不冒充作者代码/论文数字 |
| 2 模块分离、FlowResult | `dense_data.py`、`event_voxel.py`、`eraft.py`、`dense_filter.py`、`pose_replay.py`、`dense_pipeline.py`、`flow_training.py` | RTX 2060 实测，不是 4090 |
| 3 官方结构与权重 | vendored 官方网络、最后全分辨率预测、当前体素上下文、strict 键/shape、NumPy checkpoint 元数据兼容、冻结推理 | 完整官方权重由用户后续下载；无原生 confidence/warm-start |
| 4 窗口/时间/单位 | 两段等长半开区间、秒、前向位移、t_start/t_end/available_at、一次 dt 转换 | 固定 100ms 基线；短窗口及高速线性化需实测 |
| 5 实际数据与几何 | 检查已有 schema-1 YCB；标定重投影深度、源时刻快照、掩码检查、age/reason 诊断 | 其他相机的 mask 需在导入阶段显式重投影；无任意原始传感器解码 |
| 6 速度 KF | 完整二维残差、空间采样、二维门控、SVD 秩/条件、小矩阵 Cholesky、连续时间 Q | 启发式相关误差膨胀未校准；区间速度近似 |
| 7 四元数 UKF/延迟 | 复用流形 UKF，增广速度方差，`PoseHistory` 回溯重放、超期/重复计数；DOPE 外部文件适配 | 两阶段及跨区间相关性近似；没有真实 DOPE 推理结果 |
| 8 独立训练/微调 | DSEC HDF5/PNG、序列隔离、多迭代有效损失、优化器/调度/裁剪/种子、验证、save/resume | 不做跨样本 warm-start 时序训练；只运行短合成验证 |
| 9 配置/命令/输出 | `cli.py` 与 `configs/`；缓存、公平复用；trajectory、速度、状态、diagnostics、runtime；RGB 投影图 | 因果文件回放，不是实时传感器服务 |
| 10 下游评估 | EPE及 ROI 覆盖率，位姿/两种线速度/角速度 RMSE，显式模型 ADD/ADD-S，缺 GT 定性；对照脚本 | 真正冻结/微调/原前端准确率对照尚需资源；不声称提升 |
| 11 验证 | 独立三维几何、单位、时序、无效输入、重放、网络 forward、短训练/恢复、缓存串联测试 | 几何/合成模型验证分别报告，不能替代实测数据 |
| 12 交付 | 增量源码、依赖、配置、中文 README、数据格式、方法来源、验证记录、许可证和备份 | 用户后续下载数据/权重；见验证文档资源清单 |

`scripts/compare_frontends.py` 使用相同输入文件比较两个用户提供 checkpoint 和原 triplet。原 triplet 拒绝延迟位姿，新后端重放延迟位姿；有延迟时必须明确这是前端与后端共同变化，不能把结果全部归因于 E-RAFT。理想初始化、掩码、模拟位姿、合成事件需在实验报告中各自披露。
