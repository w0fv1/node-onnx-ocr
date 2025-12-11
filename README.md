这是一个为 **AI Agent（智能体）、RPA（流程自动化）及屏幕解析场景** 深度定制的高性能 Node.js OCR 项目。

本项目基于 **ONNX Runtime** 推理引擎，集成了 **DBNet**（文本检测）与 **CRNN**（文本识别）轻量级模型，旨在提供**毫秒级、高吞吐**的端侧 OCR 能力。

> **⚠️ 设计哲学说明 (Design Philosophy)**
>
> 本项目并非为生成“人类可读文档”而设计，而是专为 **机器视觉（Computer Vision for AI）** 打造。
> *   **目标对象**：LLM（大语言模型）、UI 自动化脚本、屏幕阅读器。
> *   **核心策略**：优先保证**文本块（Block）的语义完整性**与**结构化数据的清洗**，而非仅仅还原视觉排版。它内置了激进的噪点过滤、非语义字符清洗及布局分析算法，以确保输入给 AI 的数据是高信噪比的。
> *   如果你需要生成 PDF 或人类阅读的文档，请自行调整 `config` 中的过滤与排序参数。

---

## 1. 技术架构与原理 (Technical Architecture)

全流程采用流水线（Pipeline）设计，数据流转如下：

### 1.1 图像预处理 (Preprocessing)
*   **引擎**：基于 `sharp` 进行高性能图像变换。
*   **逻辑**：执行灰度化、动态缩放（基于 `targetPPI` 或 `detLimitSide`）及 Tensor 归一化（Normalization）。支持 `NHWC` / `NCHW` 布局转换以适配不同模型精度（Float32 / UInt8）。

### 1.2 文本检测 (Text Detection)
*   **模型**：采用 **DBNet (Differentiable Binarization)** 架构的轻量级 ONNX 模型。
*   **后处理**：
    *   **二值化图生成**：计算 Segmentation Map。
    *   **连通域分析**：使用并查集或递归搜索提取文本框坐标。
    *   **Box Unclip**：通过多边形扩充算法修正检测框边界。

### 1.3 结构化清洗 (Structured Cleaning & Filtering)
这是本项目针对“屏幕解析”场景的核心优化环节：
*   **形状过滤 (Shape Filter)**：基于中位数统计学，剔除极端长宽比、过小或过细的异常框（通常是 UI 装饰线或噪点）。
*   **密集簇去除 (Dense Cluster Removal)**：引入 Heatmap 机制检测高重叠区域，自动剔除文本过度密集的区域（如复杂的背景纹理误检）。
*   **拓扑排序 (Topological Sort)**：支持 `column` 模式，通过投影分析将文本块按“视觉列”进行聚类排序，而非简单的从上到下，确保表单类数据的逻辑连贯性。

### 1.4 文本识别 (Text Recognition)
*   **模型**：**CRNN (Convolutional Recurrent Neural Network)** 架构。
*   **推理优化**：支持 **Batch Inference**（批量推理）。系统自动根据 Crop 图片宽度进行分组，通过 Padding 对齐 Tensor 维度，并行执行推理以最大化 CPU 利用率。
*   **解码策略**：Greedy Decode 配合 **Smart Spacing** 算法，根据字符类型（CJK 与 Latin）动态调整空格插入阈值，解决中英混排粘连问题。

---

## 2. 快速开始 (Quick Start)

### 2.1 环境依赖
需安装 Node.js (建议 v16+)。

```bash
# 安装项目依赖
npm install
# 核心依赖包括：onnxruntime-node, sharp
```

### 2.2 模型准备
请确保根目录下存在 `model` 文件夹，并包含以下文件（需自行获取或转换对应的 ONNX 模型）：
*   `det.mobile.onnx`: 文本检测模型
*   `rec.mobile.nhwc_uint8.onnx`: 文本识别模型 (支持量化模型以提升速度)
*   `dict.txt`: 字符字典文件

### 2.3 运行推理
将待识别图片命名为 `sample.png` 放置于根目录。

```bash
node index.js
```

程序将在控制台输出识别结果，并在 `debug/` 目录下生成可视化中间件结果。

---

## 3. 可视化调试 (Visual Debugging)

本项目内置了详细的 Debug 系统，用于分析模型表现及参数调优。运行后 `debug/` 目录将生成以下文件：

1.  **`*_01_det_raw.png` (原始检测)**
    *   展示 DBNet 输出的原始候选框。
    *   用于判断**漏检**情况。如果文字未被框选，需调整 `detDbThresh` 或 `unclipRatio`。

2.  **`*_02_final_result.png` (最终结果)**
    *   展示经过清洗、排序、识别后的最终状态。
    *   **绿色框 (Valid)**：最终输出给 AI 的有效文本块，标注了 `[排序索引] 文本内容`。
    *   **红色框 (Rejected)**：被算法过滤掉的块，标注了拒绝原因（如 `too_small`, `aspect_ratio`, `dense_cluster`）。
    *   **蓝色框 (Pending)**：处理中间态（较少见）。

---

## 4. 关键配置详解 (Configuration)

通过调整 `DEFAULT_CONFIG` 可适配不同场景：

### 硬件与并发
| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| `executionProviders` | `string[]` | 默认 `['cpu']`，如有 GPU 环境可配置 CUDA。 |
| `detThreads` / `recThreads` | `number` | Intra-op 线程数，建议分别设置为 CPU 核心数的 1/2 和 1/4。 |
| `batchSize` | `number` | 识别阶段的 Batch Size，默认 `16`。 |

### 检测与清洗 (Detection & Filtering)
| 参数 | 默认值 | 专业说明 |
| :--- | :--- | :--- |
| `detDbThresh` | `0.3` | DBNet 二值化阈值。调低可提高召回率（Recall），但增加噪点。 |
| `filterStrength` | `'low'` | 形状过滤强度 (`none`\|`low`\|`medium`\|`high`)。越高越倾向于只保留标准文本行。 |
| `removeDenseClusters` | `true` | 是否启用基于重叠密度的噪点剔除（针对复杂背景图）。 |
| `sortMode` | `'column'` | 排序模式。`column` 适合表单/多列布局；`top-down` 适合单栏文档。 |

### 识别与后处理 (Recognition & Post-process)
| 参数 | 默认值 | 专业说明 |
| :--- | :--- | :--- |
| `textMinLength` | `2` | 过滤短文本。长度小于此值的 Block 会被标记为 Invalid。 |
| `smartSpacing` | `true` | 是否开启智能空格插入（根据字符类型判断是否加空格）。 |
| `removeEmoji` | `true` | 移除 Emoji 及非标准图形字符。 |
| `trashSparseThresh` | `2` | 稀疏噪点阈值。用于检测并过滤由离散单字符组成的无意义噪点串。 |

---

## 5. 性能指标 (Performance)

在主流桌面级 CPU (如 Intel i7 / Apple M1) 环境下：
*   **Initialization**: < 200ms
*   **Inference (720p Screen)**: ~300ms - 800ms (取决于文本密度与 Batch Size 设置)
*   **Memory Footprint**: ~300MB RSS

---

## 6. License

MIT License. 仅供技术研究与交流使用。
