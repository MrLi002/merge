# YCB 模型来源与数据生成约定

核查日期：2026-09-21。下面的公开模型可用于生成目标论文四类物体的新合成序列；它们不是目标论文作者公开的实验序列。

## 官方纹理模型

[YCB 官方数据索引](https://ycb-benchmarks.s3.amazonaws.com/index.html) 提供 Google 扫描器的 16k、64k 和 512k 模型及纹理。使用 16k 包可以减少本地渲染负担。下列 URL 均来自该索引，HTTP HEAD 检查均返回 200；体积是该次响应的 Content-Length，下载后仍须检查完整性。

| 对象 | Google 16k 原始包 | 压缩包字节数 |
| --- | --- | ---: |
| `003_cracker_box` | [下载](https://ycb-benchmarks.s3.amazonaws.com/data/google/003_cracker_box_google_16k.tgz) | 11,434,893 |
| `005_tomato_soup_can` | [下载](https://ycb-benchmarks.s3.amazonaws.com/data/google/005_tomato_soup_can_google_16k.tgz) | 10,606,298 |
| `006_mustard_bottle` | [下载](https://ycb-benchmarks.s3.amazonaws.com/data/google/006_mustard_bottle_google_16k.tgz) | 10,703,139 |
| `010_potted_meat_can` | [下载](https://ycb-benchmarks.s3.amazonaws.com/data/google/010_potted_meat_can_google_16k.tgz) | 9,656,325 |

总计 42,400,655 字节，约 40.4 MiB。带纹理模型由 OBJ、MTL 和 PNG 组成；实际使用时应核验 OBJ 的材质和 UV 引用，避免只加载几何而丢失包装纹理。YCB 官方索引说明数据采用 **CC BY 4.0**，附带代码采用 MIT；这两个许可不能混为一谈。分发模型或派生数据时保留 YCB 作者署名、来源、[许可链接](https://creativecommons.org/licenses/by/4.0/)及变更说明。

建议来源记录至少保存：完整下载 URL、下载日期、归档 SHA-256、实际模型/材质/纹理文件名、模型单位缩放、原始包围盒以及任何坐标变换。S3 的分片 ETag 不应直接当作归档 SHA-256。

## 尺度和坐标原点

Google 16k 官方索引未给出一个可直接套用到所有模型的“原点位于重心”说明，因此不要假定原点等于质心或包围盒中心。原始顶点坐标应保留；若为了轨迹设计将模型中心化，需显式存储原坐标到新坐标的刚体变换，并同步变换真值位姿。渲染、位姿观测、初始状态、深度和评估必须使用同一个模型坐标系。

不能直接把 BOP 的 YCB-V 模型替换为 Google 16k 模型并沿用位姿。根据 [BOP 官方 YCB-V 说明](https://bop.felk.cvut.cz/datasets/)，BOP 将原始 YCB-V 的模型从米改为毫米，将三维包围盒中心移到原点，并相应修改真值姿态。因此使用 BOP 模型需要明确毫米到米的转换和坐标对齐；同名对象不保证模型原点相同。

以下名义实体尺寸可用来检查导入比例是否明显错误，不能用来强制缩放扫描网格，也不表示三个数一定分别对应网格的 x/y/z 轴。尺寸来自 YCB 共同作者 Arjun Singh 的 [Berkeley 博士论文，第 5 章表格](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-142.pdf)：

| 对象 | 名义尺寸（mm） |
| --- | --- |
| Cracker box | 60 × 158 × 210 |
| Tomato soup can | 66 × 101（圆柱直径与高度） |
| Mustard bottle | 58 × 95 × 190 |
| Potted meat can | 50 × 97 × 82 |

[YCB 官网实体清单](https://www.ycbbenchmarks.com/wp-content/uploads/2015/09/object-list-Sheet1.pdf) 对 mustard 列出的尺寸是 50 × 85 × 175 mm，与上述论文不同；本次未核实这种差异的原因。该清单的行序号也不同于模型目录编号。导入时应记录实际网格的 AABB/OBB，在米尺度下核对数量级，而不是用名义表格覆盖网格几何。

## 与目标论文数据生成方式的对应

依据用户提供的目标论文第 IV-A 节及其 [arXiv 版本](https://arxiv.org/html/2508.14776v1)，作者使用 Unreal 和上述四类 YCB 物体，生成常速和快速的六自由度随机运动；先获得 500 FPS RGB，以此生成事件，再在固定时间窗内平均出 60 FPS RGB，从而产生运动模糊；位姿与速度真值来自仿真器。论文中的真实双相机数据为 640 × 480，且没有位姿真值。

新生成器应保留以下区别：

- 采用其他渲染器、背景、光照、轨迹或噪声模型时，应将它们记录为本项目选择；不能称为作者的 Unreal 序列。
- 事件来自未模糊的高帧率强度及亮度变化模型。不能用 60 FPS 模糊图推算事件后宣称等同于论文的 500 FPS 输入。
- 500/60 不是整数。RGB 曝光应使用准确的 1/60 秒时间窗，例如按高帧率采样区间与曝光窗的重叠时长积分；固定每 8 帧输出实际是 62.5 FPS。
- 深度为相机光轴 Z 值、单位米；RGB 曝光时间戳、深度采集时间、事件时间及外部位姿到达时间应明确记录。
- 保存常速/快速各自的轨迹参数、随机种子及像素速度统计，不能仅以文件夹名“fast”证明运动难度。
- 低频位姿若由真值加噪声产生，须标明为模拟观测；5 Hz 模拟观测不是 DOPE 网络推理。真值保存在评估文件中，运行时跟踪不可读取它。
- 保存 `T_event_object` 位姿及同约定的空间速度；如果通过位置差分得到物体原点速度 `t_dot`，应转换 `v_o = t_dot − omega × t` 后作为六维空间速度真值。

这些约束可用于构建可重复的新测试数据，但作者未公开的轨迹、事件阈值、曝光设置、标定及原始场景不能由模型包恢复。
