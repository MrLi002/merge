# 制作论文方法可用的 YCB 合成数据集

本项目已实际下载并校验 [YCB 官方 Google 16k 模型](https://ycb-benchmarks.s3.amazonaws.com/index.html) 的四个物体：`003_cracker_box`、`005_tomato_soup_can`、`006_mustard_bottle`、`010_potted_meat_can`。模型为米制 OBJ、MTL 与纹理；下载包 SHA-256、文件清单、来源和许可保存在 `assets/ycb/sources.json`。YCB 数据按 CC BY 4.0 提供；若分发模型或基于它们生成的数据，请保留来源、作者署名、许可链接和变更说明。更多模型坐标约定见 [来源记录](ycb_sources.md)。

## 一条命令生成八条序列

在项目目录中安装可选的 GPU 渲染依赖，然后下载模型并生成。Windows 需要可创建 OpenGL 3.3 standalone context 的显卡和驱动。本机已用 NVIDIA RTX 2060 测通。

```powershell
python -m pip install -e ".[synthetic,test]"
python -m ev6d fetch-ycb --assets assets/ycb
python -m ev6d generate-ycb --assets assets/ycb --output data/my_ycb_suite --duration 1 --width 640 --height 480
```

本机也可使用已配置好的 `.venv-ycb\Scripts\python.exe` 代替 `python`。默认生成四个物体各一条 `regular` 和 `fast`，共八条；每条为独立 `Dataset`。快速试运行：

```powershell
python -m ev6d generate-ycb --assets assets/ycb --output data/ycb_small --objects 003_cracker_box --speeds regular --duration 0.2 --width 160 --height 120
python -m ev6d track --dataset data/ycb_small/003_cracker_box/regular --output data/ycb_small_result
python -m ev6d evaluate --dataset data/ycb_small/003_cracker_box/regular --result data/ycb_small_result
```

已有输出目录中的同名序列不会被覆盖，以免误删结果。更换输出目录或仅指定尚未生成的物体/速度组合。`--render-hz`、`--frame-hz`、`--pose-hz`、`--contrast-threshold`、`--threshold-sigma`、`--refractory-s`、`--supersample`、`--seed` 和 `--duration` 可调整，所有参数写入各序列的 `dataset.json`。生成过程需要显存、内存和磁盘；默认 640×480 的八条一秒序列约需 GB 级磁盘空间。使用 `python scripts/validate_ycb_suite.py --dataset data/my_ycb_suite` 检查全部序列并生成 `validation.json` 和 `preview.png`。

## 数据构造

[目标论文第 IV-A 节](https://arxiv.org/html/2508.14776v1) 的公开描述包含四个 YCB 物体、Unreal 随机六自由度运动、500 FPS 清晰图生成事件、固定曝光平均成 60 FPS RGB、常速与快速运动，以及仿真位姿/速度真值。下面是本项目的独立实现：

1. 保留官方 OBJ 顶点的米制尺度，记录原始包围盒中心；渲染采用中心化后的物体坐标。GPU 透视投影同时输出线性 RGB、米制 Z 深度和实例掩码。
2. 为了使官方模型的标签正面进入视野，先设置物体特定的初始朝向（cracker box 绕 Y 轴 +90°，其余绕 X 轴 −90°）；随后由种子控制的双谐波六自由度轨迹给出 `regular`，`fast` 使用相同路径和三倍时间尺度。初始朝向、随机参数与解析空间速度均保存，物体位置在事件相机前方约 0.75 m。
3. 500 Hz 清晰 RGB 按线性亮度与逐像素对数对比度阈值生成 `events.npy`。阈值默认 0.2；时间在相邻清晰样本间线性内插。此过程不读取真值光流。
4. RGB 先转线性光，再对连续的高帧率采样作分段线性积分，按精确 `1/60 s` 全曝光输出。500/60 并非整数，因此未采用固定八张平均。写 PNG 时才编码成 sRGB。
5. 深度取曝光结束瞬间的 RGB-D 相机 Z；事件坐标下的实例掩码在同一时刻生成。相机基线为已记录的 `[0.018,-0.004,0.002]` m 工程设定，非论文标定。第零帧为 `t=0` 的清晰启动帧，不属于曝光序列。
6. 5 Hz 位姿观测由真值加入位置和旋转向量高斯噪声生成，明确标记 `simulated_noisy_pose_NOT_DOPE`。真值只存于 `ground_truth.npz` 用于离线评估，跟踪器不会读它。初始位姿为已知仿真真值。

每个序列目录包含 `dataset.json`、`events.npy`、`poses.csv`、`ground_truth.npz`、`frames/rgb_*.png`、`frames/depth_*.npy`、`frames/mask_event_*.npy`。顶层 `suite.json` 汇总序列与事件/帧数。事件行是 `[t,x,y,p]`，单位秒、整数像素和 ±1 极性；深度为米。所有帧的时间、曝光起止、标定和坐标变换均可从 `dataset.json` 读取。

命令行的 `--assets`、`--output`、`--dataset` 和 `--result` 都可以写相对路径，但它们相对于**运行命令时的当前工作目录**，不是脚本文件的位置。先进入项目目录再运行即可。每条序列的 `dataset.json` 用序列目录相对路径引用事件、图像、深度与掩码；`suite.json` 也用套件根目录相对路径索引序列，因此数据目录搬到服务器后仍能加载。元数据中的原始 OBJ 绝对路径仅作生成机器的来源记录，跟踪/评估不会读取它。

**适用范围：** 这是可用于本项目训练/验证数据管线与离线算法测试的合成数据集，并非作者未公开的 Unreal 原始序列，也不能复现论文数值。渲染没有真实噪声、复杂遮挡和相机运动；事件模型比真实传感器简单；5 Hz 观测不是 DOPE 推理。跟踪时使用的逐帧物体掩码是理想合成分割，属于 oracle 辅助输入。若评估端到端识别性能，应替换为独立的分割器输出，并单独报告该输入。当前数据集不能替代真实相机数据。

## 本机生成和验证结果

已在 `data/ycb_suite_labelled/` 生成四物体 × 常速/快速八条一秒的 640×480 序列，合计 **10,788,141** 个事件，每条 61 帧（含 t=0 启动帧），总数据约 **1.15 GB**。`validation.json` 记录逐条检查结果，`preview.png` 提供八条序列的曝光 RGB 预览。测试套件为 **146 passed**。

以 `005_tomato_soup_can/regular` 为全分辨率端到端例子，跟踪处理 444,271 个事件，得到 373,786 个有深度的光流测量；相对于自身合成真值，位置 RMSE 为 **0.0128 m**，旋转 RMSE 为 **5.13°**。处理耗时约 90 秒/1 秒序列，只说明可运行，不能作为实时性或原论文结果的证据。结果在 `data/ycb_labelled_tracking_tomato/`。
