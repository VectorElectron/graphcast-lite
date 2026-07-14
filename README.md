# GraphCast-Lite: High-Performance GPU-Accelerated Weather Forecasting Inference Engine

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Accelerated-green.svg)](https://developer.nvidia.com/cuda-zone)
[![TensorRT](https://img.shields.io/badge/TensorRT-Optimized-orangered.svg)](https://developer.nvidia.com/tensorrt)

> **⚡ 40秒闪电跑完未来10天全球0.25°高分辨率气象预报！**
> 
> **⚙️ 显存奇迹：仅需 8G 显存，即可在消费级显卡（如 RTX 2080 Ti、RTX 3070、RTX 4060 等）上强行流畅跑起顶配 0.25° 全球高分辨率 Float32 物理模型，毫无保留地释放最高精度，榨出 Google 官方 TPU v4 级别的巅峰推理吞吐量！**
> 
> **GraphCast-Lite** 带着极致的“反过度封装”理念诞生。我们彻底打破了气象大模型的算力与显存壁垒，通过精妙的 **Turbo** 高性能自回归算子重构、双端分块消减和硬编码静态显存锁定技术，彻底告别了传统大模型动辄需要 A100/H100 等工业级显卡高昂开销的时代，让平民级设备爆发出了毁灭性的性能！

---

传统气象大模型框架往往伴随着令人窒息的过度封装。从官方基于 JAX/Haiku 的黑盒实现，到各类基于 PyG（PyTorch Geometric）或 DGL 的臃肿图神经网络库，不仅让代码可读性极差，还带来了巨大的额外性能损耗与显存碎片。

本项目将复杂的球形多尺度网格数据准备工作剥离至仅依赖 `NumPy` 和 `SciPy` 的基础生态，核心网络采用纯粹、精炼的 `PyTorch` 构建。在此之上，**Turbo** 子模块通过手动解耦计算图、定制 TensorRT 子图引擎以及精细化静态显存池管理，将大图神经网络的硬件开销压榨到了绝对极限。

---

## ✨ 核心工程设计与项目特色

### 🍃 GraphCast-Lite 核心：去过度封装与多端对齐
* **零臃肿依赖**：网格剖分、几何空间双向映射及稀疏图构建完全基于原生 `NumPy` 和 `SciPy.spatial.cKDTree` 实现。无需安装庞大的地理空间数据库或第三方图学习框架（如 PyG）。
* **纯粹的 PyTorch 实现**：网络主体采用标准 PyTorch 编写，抛弃了复杂的图对象封装和黑盒算子，核心逻辑一目了然，极易进行二次开发、微调与算子级调试。
* **极简数据管线 (`ncutil`)**：彻底摒弃了气象领域常用的、极其沉重的 `xarray` 和 `dask` 依赖。我们基于原生 `netCDF4` 编写了 `ncutil` 数据模块，实现**纯粹、见底、无任何黑盒包装的数据读取**。直接面向底层多维数组切片进行标准化（Normalize）与动态反归一化，大幅简化了数据流，将数据加载延迟减小到可以忽略不计的程度。
* **高精度太阳辐射场计算 (`forcefield`)**：在计算顶层大气层（TOA）太阳辐射等关键时间强迫场特征时，我们没有沿用 DeepMind 官方在时间步上采用离散近似采样的传统实现。本项目采用**纯数学解析积分方案**。在自回归多步预测中，不仅确保了物理守恒性、提供了更高的积分精度，其计算性能更是暴涨 **100 倍以上**，彻底消除了强迫场计算的 CPU 瓶颈。
* **多端一致性对齐**：提供了完全对齐的多端推理基线（原生 PyTorch、纯 NumPy/CuPy、ONNX Runtime），确保各端在自回归滚动（Rollout）中的预测数值严密一致。

### 🚀 Turbo 子模块：硬件协同与极致显存优化
`turbo` 目录是本项目的高性能生产级推理加速引擎，专注于手动内存管理与计算图的极致压榨：
* **静态虚拟显存池管理 (Zero-Allocation VRAM Pool)**：独创 `PoolManager` 分配器。在初始化时向 GPU 申请整块连续物理显存，并借助编译期精确画像出的 `contextlen` 偏置（Static Offset）实现局部临时计算特征的**原地覆盖写**。整个自回归预报期间，**CUDA 显存申请次数为零**，彻底避免显存碎片导致的 OOM。
* **多子图 TensorRT 解耦与编译 (Sub-Graph Serialization)**：将庞大的 GraphCast 骨干网手动解耦为 5 个强收敛的计算子图：`MeshEncoder`、`G2MAggregate`、`M2MProcessor`、`M2GInteraction` 与 `ForceField`。子图级算子深度融合，极大削减了 GPU Kernel Launch 的延迟。
* **双端空间分块消减显存峰值 (Dynamic Dual-Chunking Engine)**：
  * *Grid-to-Mesh 边分块 (Edge Chunking, `echunk`)*：在稀疏图聚合阶段，通过时序滑窗分块读取并配合 CuPy 原地稀疏累加。
  * *Mesh-to-Grid 还原分块 (Grid Chunking, `gchunk`)*：在空间反向映射阶段，分块还原球形网格至二维地理格点。
  * **显存奇迹突破**：通过物理层面的流式分块运算，成功将 0.25° 顶配物理模型在 **Float32 原生单精度精度**下的显存占用峰值精准控制在 **8G** 以内，让平民笔记本显卡运行无损高精度模型成为现实！
* **物理指针直连与零拷贝 (Direct Pointer Binding & Zero-Copy)**：利用 CuPy 数组共享 GPU 物理显存，无需通过 CPU 中转，直接将显存物理指针（`.data.ptr`）通过 `set_tensor_address` 绑定给 TensorRT 的 Execution Context。自回归滚动时仅通过交换指针完成状态交替，实现 **0 字节设备端内存拷贝**。

---

## 📥 模型权重与测试数据资源列表

> ⚠️ **重要提示：本项目中的 Lite 模式与 Turbo 极致加速模式使用两套完全不同的模型文件。**
> * **Lite 模式** 使用纯 PyTorch 导出的权重包（`.pth` / `.pkl` 等格式），适用于快速验证与轻量微调。
> * **Turbo 模式** 使用为静态执行、多子图切分编译准备的 ONNX 模型以及对应的运行图定义。
> 
> 请根据你要运行的模式，从下表中下载对应的压缩包，并严格按路径对齐解压至项目根目录下的 `weights/` 目录中：

| 资源类别 | 规格分辨率 | 适用场景与说明 | 下载链接 |
| :--- | :--- | :--- | :--- |
| **通用测试数据** | 基础测试数据与归一化特征包 | 包含自回归测试输入、均值、标准差等归一化参数（通用必下） | [📥 点击下载测试数据与归一化包](https://github.com/VectorElectron/graphcast-lite/releases/download/weights/testdata_and_normvector.zip) |
| **Lite 模式权重** | 1.0° 基础模型 (PyTorch 原生端) | 原生 PyTorch / NumPy / CuPy 快速验证与轻量微调 | [📥 点击下载 Lite-1.0° 权重包](https://github.com/VectorElectron/graphcast-lite/releases/download/weights/lite_5mesh_13level_1deg.zip) |
| **Lite 模式权重** | 0.25° 高清模型 (PyTorch 原生端) | 原生 PyTorch 高精度研究、自回归滚动与模型微调 | [📥 点击下载 Lite-0.25° 权重包](https://github.com/VectorElectron/graphcast-lite/releases/download/weights/lite_6mesh_37level_0.25deg.zip) |
| **Turbo 模式权重** | 1.0° 编译源包 (TensorRT 加速端) | 1.0度工业级自回归推理编译 (包含特定 ONNX 与 configs) | [📥 点击下载 Turbo-1.0° 编译源包](https://github.com/VectorElectron/graphcast-lite/releases/download/weights/turbo_5mesh_13level_1deg.zip) |
| **Turbo 模式权重** | 0.25° 编译源包 (TensorRT 加速端) | 0.25度全球极速生产级自回归推理编译 (包含特定 ONNX 与 configs) | [📥 点击下载 Turbo-0.25° 编译源包](https://github.com/VectorElectron/graphcast-lite/releases/download/weights/turbo_weights_0.25deg.tar.gz) |

> 📌 **路径对齐要求**：测试数据请放置在 `weights/testdata_and_normvector/` 目录下；模型权重请放置在如 `weights/para_5mesh_13level_1deg/...` 或 `weights/para_6mesh_37level_0.25deg/...` 的对应目录下。

---

## 🛠️ 环境依赖

请确保本地已配置好 CUDA 环境（建议 CUDA 11.8+ 或 12.x），并安装与你 CUDA 版本相对应的 `CuPy`：
```bash
# 请根据本地 nvcc --version 调整对应 cupy 后缀 (例如 cupy-cuda12x)
pip install cupy-cuda12x
pip install tensorrt netCDF4 scipy torch onnxruntime
```

---

## 🍃 快速上手：GraphCast-Lite (轻量基线端)

Lite 模式旨在用最干净的代码 and 标准的 PyTorch 框架跑通推理与训练，方便进行算法改写与多端数值校验。**请确保您已下载上方对应的 Lite 模型文件。**

#### 1.1 运行 PyTorch 原生推理
```bash
cd infer/
python infer_torch.py
```

#### 1.2 运行纯 NumPy/CuPy 算法验证
可以直接无缝在 CPU (NumPy) 与 GPU (CuPy) 算子级推理后端间快速切换，用于严密的数值核对：
```bash
python infer_numpy.py
```

#### 1.3 启动去过度封装的极简训练
```bash
cd graphcast/
python train.py
```

---

## 🚀 快速上手：Turbo (极致性能端)

Turbo 子模块用于生产环境部署，通过手动内存管理和 TensorRT 编译，压榨出极致的预报吞吐量。**请确保您已下载上方对应的 Turbo 专属模型文件。**

#### 2.1 自动化子图编译与显存画像
使用 `model_compile.py` 将下载并解压的 5 个 GraphCast ONNX 子模型转换成高性能 TensorRT 引擎。
编译完成后，它会深度画像各子图运行所需的显存，并自动反写更新 `GraphCastInfer.json` 配置文件中的显存长度 `contextlen`：
```bash
# 编译并开启 FP16（以 1deg 5mesh 13level 精度配置为例）
python turbo/model_compile.py --onnx-dir ../weights/para_5mesh_13level_1deg --fp16
```
*(注：由于我们的 `forcefield` 采用了完全重写的物理高精度太阳辐射解析算子，它在编译时默认保持 FP32 精度，以防止由于时间步变化导致的太阳辐射数值溢出或物理偏差。)*

#### 2.2 执行极速静态内存池自回归推理
运行生产级推理主控器，体验基于静态物理显存池、双端分块与零拷贝技术的极速全球气象预报：
```bash
python turbo/infer_tensorrt.py
```

---

## 📊 Turbo 模块在不同硬件设备上的实测数据

以下为 **Turbo** 极致加速引擎在不同主流 GPU 硬件以及计算精度下，跑完全球中期气象自回归预报（40 步滚动预测，即未来 10 天天气形势）的实测性能指标：

| 测试硬件 (GPU Hardware) | 运行模式/计算精度 (Precision) | 推理总耗时 (Total Latency / 40 frames) | 平均每步耗时 (Per-frame Latency) | 峰值显存占用 (Peak VRAM) | 显存控制特征 (VRAM Profile) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NVIDIA RTX 2080 Ti** | Turbo TRT (Float32 / 精度无损) | 115 s | 2.875 s / frame | **~ 7.8 GB (突破8G)** | **绝对直线（硬编码锁定）** |
| **NVIDIA RTX 2080 Ti** | Turbo TRT (FP16) | **40 s** | **1.000 s / frame** | **~ 6.0 GB** | **绝对直线（硬编码锁定）** |
| **NVIDIA RTX 4090** | Turbo TRT (TF32) | 29 s | 0.725 s / frame | **~ 7.8 GB (突破8G)** | **绝对直线（硬编码锁定）** |
| **NVIDIA RTX 4090** | Turbo TRT (BF16) | **15 s** | **0.375 s / frame** | **~ 6.0 GB** | **绝对直线（硬编码锁定）** |

> 💡 **性能分析与设计亮点：**
> * **平民显卡跑 FP32 顶配模型**：在传统的推演架构中，运行 Float32 的全球 0.25° 模型因内存膨胀动辄需要 20GB 以上的显存空间。而在本项目的 **Turbo 引擎下，通过双端空间分块与就地覆盖写分配机制，成功在 8G 显存内保真无损地运行了 FP32 全精度自回归推理**，为硬件条件受限的科研人员带来了福音。
> * **硬编码显存锁定**：得益于我们的静态虚拟显存池管理（`PoolManager`），不论自回归预测推演到何种深度，显存监控曲线始终呈现一条**绝对平直的硬锁死直线**。完全消除了高并发或长时间步预测下的显存抖动，从根本上杜绝了 OOM 风险。
> * **算力极致榨取**：配合 RTX 4090 的 Tensor Core 硬件算力，在开启 **BF16 模式**后，单步运行全球气象预报仅需 **0.375 秒**！这意味着仅需 15 秒即可算出长达 10 天的全球高分辨率精细气象趋势，为实时预报、高频滚动同化和集成气象研究提供了超强的生产级效率。

---

## 🔮 前景展望：迈向 0.1° 极高分辨率的新纪元

从 1.0° 的粗放预报到 0.25° 的精细推演，GraphCast-Lite 已经用硬核的工程重构证明了“算法瘦身”的巨大威力。然而，这并不是我们压榨硬件潜能的终点。

随着气象界对百米级、公里级（Convection-Permitting）高分辨率临近预报需求的日益迫切，**全球 0.1° 极高分辨率模型**代表了 AI 气象推演的下一个终极战场。根据我们的工程数学模型与显存足迹精细评估：

* **理论显存推算**：在 0.1° 分辨率下，全球地表格点数和垂直层特征维度将呈指数级爆发。但通过进一步升级的 **物理多级级联分块（Multi-level Cascade Chunking）** 与 **自适应动态张量重构（Dynamic Tensor Rematerialization）**，我们能够将海量中间特征的峰值显存消耗，牢牢锁定在 **24G 物理显存** 的红线以内。
* **平民生产化革命**：这意味着，在未来，科研人员和中小型气象机构甚至**不需要租用昂贵的企业级 A100/H100 显卡**。仅需一张日常消费级旗舰显卡（如 RTX 3090 / RTX 4090 或下一代 24G 显卡），就能独立本地流畅运行 0.1° 全球顶级精细度气象自回归预报，让极致精度真正走向大众、赋能科研。

---

## 🤝 商业合作与学术探讨（微调权重加速及非标准模型定制）

本项目虽然完全开源，但需要说明的是，**Turbo 极致加速引擎中的 ONNX 子图分解与重新编排，深度借助了我们内部自研的高效工具链，并结合了极其复杂的手工计算图物理重组与算子融合技术**。这导致该加速方案难以针对任意结构的模型进行全自动化的通用生成。

如果您正在面临以下场景，欢迎与我们联系，开展深入的**学术探讨**或**商业合作**：

* **自有微调权重（Fine-tuned Weights）**：您已基于 GraphCast 官方架构微调出了特定区域（如中国区域高分辨率）或特定要素的私有权重，希望能无缝嵌入到 Turbo 引擎中，在低配硬件上压榨出极致的推理吞吐量。
* **非标准架构与其他稀疏图模型（如 GenCast）**：除了 GraphCast 之外，如果您有 **GenCast** 等其他基于稀疏图神经网络（GNN）的气象/物理模型同样面临显存瓶颈、多卡开销昂贵等工程痛点，我们非常欢迎共同探讨，利用我们的分块消减和零拷贝静态显存技术为其定制专属的高性能工程优化方案。

> 📩 **联络通道**：欢迎提交 Issue 或通过仓库主页的联系方式与我们取得联系。让顶尖的工程技术，助您的科研与商业应用突破算力极限！

---

## 🤝 参与贡献与开源许可

我们致力于为气象科研界与工业界提供一套**清爽、见底、不失威力**的轻量化气象大模型工具链。如果你有更好的物理算子实现方案或能进一步降低图交互阶段的显存边界，欢迎提交 Issue 或 Pull Request！

本项目基于 **BSD 3-Clause License** 许可协议开源。您可以自由修改、分发及将其应用于商业、科研项目中，但请保留原作者的版权声明及许可条款。
```