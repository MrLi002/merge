# 本次补全与实际验证

本次在 `D:/postGraduate/code/ztgj/merge` 增量开发，2026-09-23 开始，2026-09-24 继续。修改前备份 `backup_completion_20260923_194704/`；相邻 6DOF 仅只读。按用户补充要求，checkpoint 和真实数据由用户后续下载，不再发起下载。

## 环境和已核对输入

- 当前目录 `.venv`：Python 3.10.20、PyTorch 2.5.1+cu124、RTX 2060、NumPy 2.2.6、SciPy 1.15.3、h5py 3.16.0、pytest 9.1.1；editable 安装项目 0.2.0 成功。安装依赖后 OpenCV 为 5.0.0.93。此环境借用现有 conda 的只读系统包，新增依赖安装在当前 `.venv`，没有修改相邻项目环境。
- 三份原始 PDF、官方固定提交、公式及权重兼容核对见 [方法审计](method_audit.md)。没有把历史文档结果当作本次结果。
- 实际完整运行 `check-data` 读取 `../6DOF/data/ycb_suite_labelled/005_tomato_soup_can/regular`：444,271 事件、480×640、61 帧、6 个 `simulated_noisy_pose_NOT_DOPE` 位姿，已知合成初始化及理想掩码。全部深度/掩码文件通过格式校验。

## 已执行旧基线及报告验证

以下命令均使用 `./.venv/Scripts/python.exe`：

```powershell
python -m pip install -e ".[flow,test]"
python -m ev6d generate --output output/completion_20260923/tracking_data --scenario mixed --duration 0.4 --width 80 --height 60 --render-hz 500
python -m ev6d track --dataset output/completion_20260923/tracking_data --output output/completion_20260923/triplet_result
python -m ev6d evaluate --dataset output/completion_20260923/tracking_data --result output/completion_20260923/triplet_result
python -m ev6d reproject --dataset output/completion_20260923/tracking_data --result output/completion_20260923/triplet_result --max-frames 4
python -m ev6d visualize --dataset output/completion_20260923/tracking_data --result output/completion_20260923/triplet_result
python -m ev6d flow-fixture --output output/completion_20260923/dsec_fixture --size 128 --samples 2
```

旧 triplet 实测：34,901 原始事件、21,343 有效光流、3 次位姿校正；处理 6.366 秒/0.4 秒序列。41 个输出姿态，位置 RMSE 58.922mm、旋转 RMSE 9.409°、空间线速度 RMSE 0.7627m/s、角速度 RMSE 1.2391rad/s、参考点线速度 RMSE 0.6828m/s。结果来自独立合成数据，不是论文 benchmark，不证明实时或泛化精度。4 张重投影图和离线交互 HTML 已生成。

单独报告测试验证了参考点线速度与空间线速度的区别、缺 GT 不产生 RMSE、ADD 必须显式声明模型、RGB 投影使用正确的外参逆变换、Unicode 路径及非刚体标定拒绝。缓存 EPE 测试验证目标区域、背景和缺预测的区分。

## 独立训练与恢复：已实际运行

```powershell
python -m ev6d train-flow --config configs/train_flow_smoke.json --max-steps 2
python -m ev6d train-flow --config configs/train_flow_smoke.json --resume output/completion_20260923/flow_smoke/last.pt --output output/completion_20260923/flow_resume --max-steps 4
python -m ev6d evaluate-flow --config configs/train_flow_smoke.json --checkpoint output/completion_20260923/flow_resume/last.pt --output output/completion_20260923/flow_eval.json
```

RTX 2060、完整 E-RAFT、15 bins、128×128、2 次迭代：第一次完成 step 1–2（2.278s 训练流程计时），从完整 checkpoint 恢复并完成 step 3–4（1.312s）；32,256 个合成验证像素，最终 EPE 0.738912px。独立评测命令获得相同 EPE。数据和权重均标记 `synthetic_fixture=true`、`initialization=random`，这只验证梯度、优化器、调度器、保存、恢复和评测路径，不是 DSEC benchmark，也不是官方预训练结果。

`tests/test_flow_training.py` 的 CPU 完整网络测试另做首次训练、恢复、连续训练和从 checkpoint 初始化微调，共 5 次优化。恢复和连续运行的所有模型张量逐位一致，Adam 的 step 及余弦调度状态一致；训练权重能被冻结的 `ERAFTFrontend` 严格重载。全无效监督场景 6 批均跳过，global_step=0，明确生成 error.json，未输出可被误认为已训练的 checkpoint。

## 网络与 KF/UKF 完整串联：已实际运行

```powershell
python -m ev6d precompute-flow --dataset output/completion_20260923/tracking_data --checkpoint output/completion_20260923/flow_resume/last.pt --config configs/dense_smoke.json --output output/completion_20260923/dense_cache
python -m ev6d track-dense --dataset output/completion_20260923/tracking_data --flow-cache output/completion_20260923/dense_cache --config configs/dense_smoke.json --output output/completion_20260923/dense_cached
python -m ev6d track-dense --dataset output/completion_20260923/tracking_data --checkpoint output/completion_20260923/flow_resume/last.pt --config configs/dense_smoke.json --output output/completion_20260923/dense_direct
python -m ev6d evaluate --dataset output/completion_20260923/tracking_data --result output/completion_20260923/dense_cached
python -m ev6d reproject --dataset output/completion_20260923/tracking_data --result output/completion_20260923/dense_cached --max-frames 4
```

3 个100ms光流区间全部执行网络；3 次速度校正、192 个采样观测、3 次绝对位姿校正。输出模式 `oracle_pose_fusion`，因为初始化使用已知合成位姿；后续观测是带噪模拟位姿而非 GT 文件，跟踪没有加载 ground_truth.npz。缓存与直接推理的时刻相同，位置最大差 2.47e-8m，速度最大差 6.09e-7；以 atol=1e-6、rtol=1e-5 数值等价校验通过。CUDA 本次非确定性模式，未声称逐位相同，差异原始记录保存在 `direct_cache_equivalence.json`。

缓存轨迹5个时刻：位置RMSE 49.220mm、旋转RMSE 6.656°、空间线速度RMSE 1.0885m/s、角速度RMSE 1.5117rad/s、参考点线速度RMSE 0.5931m/s。这是从随机初始化短训练4步得到的玩具权重，**不能与上面的41时刻 triplet 指标直接做优劣结论**，也不能当作正式前端比较。共同采样时刻、权重和数据需要正式实验统一。

缓存生成总计3.678s（含模型加载）；体素0.169s、同步网络0.267s、写缓存0.015s。缓存读取跟踪0.098s，其时间不能单独当作完整系统实时性能。网络推理与跟踪实际耗时保存在各 runtime/manifest 内；此短跑不是4090全分辨率基准。

## 最终回归与交付检查

`./.venv/Scripts/python.exe -m pytest -q`：最终 **256 passed in 19.28s**，日志 `final_pytest.txt`。包括原有生成/事件/几何/YCB渲染回归和新加载器、网络适配、训练、评测、缓存及稠密跟踪测试。另执行 `compileall`、模块 CLI、安装后的 `ev6d.exe dense-config`。系统 Python 3.13 / PyTorch 2.12 CPU 的适配器测试最初暴露 safe_globals 使用 qualname 的兼容差异，修复后同组 **8 passed in 4.78s**；未降低安全加载约束。

最后审查新增缓存生产者权重哈希/训练来源校验、序列尾部延迟位姿排空及统计。在 `dense_cache_final`、`dense_cached_final`、`dense_direct_final` 新目录重新运行完整网络→缓存/直接跟踪，保留此前输出。最终缓存/直接轨迹位置最大差1.34e-9m、速度最大差4.62e-7，数值容差校验通过；来源中记录真实 checkpoint SHA-256 和 `synthetic_fixture=true`。新缓存清单必须有来源字段，早期 `dense_cache` 仅保留历史记录，最终代码请读取 `dense_cache_final`。

对照脚本也以本地 step2 与 step4 的**合成训练权重**实际跑完三条路径，生成 `comparison_smoke/comparison.json`。它除原始采样指标外，还对三个后端在完全相同的5个时刻 `[0,.1,.2,.3,.4]` 离线插值评测，保留原始轨迹。命令如下；参数名是实验角色，本次并没有官方预训练/真实数据对照：

```powershell
python scripts/compare_frontends.py --dataset output/completion_20260923/tracking_data --frozen-checkpoint output/completion_20260923/flow_smoke/last.pt --finetuned-checkpoint output/completion_20260923/flow_resume/last.pt --dense-config configs/dense_smoke.json --output output/completion_20260923/comparison_smoke
```

`final_manifest.json` 保存最终环境及62份源码/测试/配置/依赖文件的SHA-256。对照脚本未运行正式冻结/微调精度实验，因为用户尚未提供资源。

完整结果位于 `output/completion_20260923/`，源码与附件逐项对应见 [requirements_mapping.md](requirements_mapping.md)。当前README中的默认路径指向本次结果，重复执行应选新目录以免触发保护。

## 验证边界

几何/滤波测试使用独立三维刚体指数变换再投影，覆盖纯平移/纯旋转/混合速度恢复、六列 Jacobian 中央差分、时间及噪声单位、缩放/裁剪主点、秩/条件、空观测、协方差、四元数符号及有界延迟重放；这些不代表学习光流的真实精度。

模型适配测试实际执行完整 E-RAFT forward，使用临时随机参数及明确的测试 checkpoint 验证结构/接口，不能冒充官方预训练网络。已核验官方权重的下载链接及 checkpoint 元数据，但本机现有文件截断/HTML 错误；一次此前的有限下载失败已记录并停止。没有完整官方权重推理结果。

后续需用户准备真实 DSEC-Flow、完整官方权重、实测物体事件/RGB-D/标定/外部位姿及 GT。还需在 RTX 4090 验证目标分辨率显存、正式训练、下游冻结/微调对照及实机延迟。代码不声称仅换权重即可提升准确率。
