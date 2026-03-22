# ArXiv 调研报告 - 2026-03-22
**统计周期：2026年3月15日 – 2026年3月22日**

---

## 📌 方向一：大模型训练效率优化

### 1. POET-X: Memory-efficient LLM Training by Scaling Orthogonal Transformation
- **arXiv**: [2603.05500](https://arxiv.org/abs/2603.05500)
- **摘要**: POET-X 是 POET 的可扩展内存高效变体，通过正交等价变换实现谱保持优化。在单块 NVIDIA H100 GPU 上完成十亿参数级别 LLM 预训练，而同等设置下标准 AdamW 优化器会 out-of-memory。是首个实现单卡训练亿级参数 LLM 且保证训练稳定性的工作。

### 2. Understanding Quantization of Optimizer States in LLM Pre-training: Dynamics of State Staleness and Effectiveness of State Resets
- **arXiv**: [2603.08185](https://arxiv.org/abs/2603.08185)
- **投稿时间**: 2026年3月17日
- **摘要**: 研究低精度 EMA 优化器状态，发现量化会导致更新被四舍五入回同一个存储值，使状态变得陈旧，减缓自适应学习率效果。同时探索状态重置策略在恢复训练动态方面的有效性。为理解优化器状态量化提供了重要的理论分析。

### 3. SERQ: Saliency-Aware Low-Rank Error Reconstruction for LLM Quantization
- **arXiv**: [2603.08185](https://arxiv.org/abs/2603.08185)
- **摘要**: 后训练量化方法，提出显著性感知误差重建，通过单一低秩补偿矩阵联合缓解激活和权重显著性带来的量化误差。在 W4A4 和 W4A8 设定下均超越以往方法，精度甚至高于当前最先进的基于旋转的 W4A4 方法。

### 4. EDA: Efficiently Aligning Draft Models via Parameter- and Data-Efficient Adaptation
- **arXiv**: [2603.09527](https://arxiv.org/abs/2603.09527)
- **摘要**: 解决投机解码中目标模型微调后 draft model 分布不匹配问题。只需更新轻量私有组件即可实现高效适配，大幅降低训练成本，同时实现更高的平均接受长度，恢复甚至超越原始投机解码性能。

---

## 📌 方向二：Fault-tolerant Training（容错训练）

### 1. SPARe: Stacked Parallelism with Adaptive Reordering for Fault-Tolerant LLM Pretraining Systems with 100k+ GPUs
- **arXiv**: [2602.00277](https://arxiv.org/abs/2602.00277)
- **摘要**: 在 100k+ GPU 规模下，硬件故障从偶发事件变成常态。SPARe 提出堆叠并行性与自适应重排序的系统方案，显著减少故障恢复时间和训练效率损失。这是面向超大规模分布式 LLM 训练的实用系统级工作。

### 2. SHIFT: Exploring the Boundary of RDMA Network Fault Tolerance
- **arXiv**: [2602.22760](https://arxiv.org/abs/2602.22760)
- **摘要**: 在 gang scheduling 机制下，RDMA 网络故障是导致训练效率下降的重要原因。该工作深入探索 RDMA 网络故障边界条件，揭示大规模分布式训练在通信层的脆弱性，并提出相应容错增强方案。

---

## 📌 方向三：视频生成（Video Generation & World Models）

### 1. Helios: Real Real-Time Long Video Generation Model
- **arXiv**: [2603.04379](https://arxiv.org/abs/2603.04379)
- **摘要**: ⭐ 本周最大突破。首个 14B 视频生成模型，在单块 NVIDIA H100 GPU 上达到 19.5 FPS，支持分钟级视频生成，无需 KV-cache、稀疏注意力等标准加速技术。通过在训练中显式模拟漂移并在源头消除重复运动实现长视频抗漂移。是视频生成在效率和规模上的重要里程碑。

### 2. V-Dreamer: Automating Robotic Simulation and Trajectory Synthesis via Video Generation Priors
- **arXiv**: [2603.18811](https://arxiv.org/abs/2603.18811)
- **摘要**: 从自然语言指令直接生成开放词汇、模拟就绪的操控环境和可执行专家轨迹。用 LLM 和 3D 生成模型构建物理可行的 3D 场景，利用视频生成模型作为运动先验，将视频预测映射为可执行机器人轨迹。在真实机械臂上验证了轨迹可迁移性。是视频生成模型直接服务机器人操控数据自动化生成的里程碑工作。

### 3. EVA: Aligning Video World Models with Executable Robot Actions via Inverse Dynamics Rewards
- **arXiv**: [2603.17808](https://arxiv.org/abs/2603.17808)
- **摘要**: 解决视频 world model 缺乏显式可执行性约束的核心矛盾。视频 world model 生成的视觉连贯 rollout 经逆动力学模型解码后可能产生不可行控制命令。EVA 用 RL 框架弥合可执行性差距，将逆动力学差距作为训练信号，用奖励模型评估生成视频的动作序列质量。显著减少生成 rollout 中的本体特定伪影。

### 4. OmniForcing: Unleashing Real-time Joint Audio-Visual Generation
- **arXiv**: [2603.11647](https://arxiv.org/abs/2603.11647)
- **摘要**: 首个实时音视频联合生成框架。核心挑战是双流架构的因果蒸馏会因模态间时间不对称和 token 稀疏引发训练不稳定。通过非对称块因果对齐、Audio Sink Token 机制和联合自强制蒸馏，在单 GPU 上实现 ~25 FPS 实时音视频同步生成。

---

## 📌 方向四：Diffusion 模型蒸馏（Distillation & Step Compression）

### 1. DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression
- **arXiv**: [2603.13162](https://arxiv.org/abs/2603.13162)
- **摘要**: 用 Diffusion Transformer 完全替代 U-Net，将扩散过程搬到 32x 降采样潜空间，通过方差引导重建流、自蒸馏对齐和潜条件引导机制将多步 DiT 适配为单步重建模型。解码速度提升 30 倍，16GB 笔记本 GPU 即可重建 2048×2048 高分辨率图像。

### 2. Streaming Autoregressive Video Generation via Diagonal Distillation
- **arXiv**: [2603.09488](https://arxiv.org/abs/2603.09488)（ICLR 2026）
- **摘要**: 针对视频时间维度的蒸馏方法。提出不对称的 chunk 生成策略（早期 chunk 用更多步、后期 chunk 用更少步）和对齐隐式噪声预测来缓解误差传播和长程序列过饱和问题。5 秒视频 2.61 秒生成，最高 31 FPS，相比原始模型加速 277.3 倍。

---

## 🔥 本周亮点

| 亮点 | 内容 |
|------|------|
| **⭐ 最大突破** | **Helios** — 首个 14B 参数在单 H100 上 19.5 FPS 实时生成长视频，是视频生成领域的里程碑 |
| **💡 新兴方向** | 视频 world model 服务机器人操控（V-Dreamer、EVA）成为新热点，从数据生成和 reward 建模两个路径解决物理可执行性问题 |
| **⚡ 训练效率** | **POET-X** 实现单卡训练十亿参数 LLM；**DiT-IC** 将扩散图像解码速度提升 30 倍 |
| **🏗️ 系统层面** | **SPARe** 面向 100k+ GPU 规模的容错预训练系统，直接回应超大规模训练的工业现实挑战 |
| **📉 量化进展** | **SERQ** 在 W4A4 精度上超越所有已有方法，为极致低比特 LLM 部署提供新路径 |

**整体趋势**：本周论文呈现三个交叉热点——(1) 视频生成正加速迈向实时和长时；(2) 视频生成模型作为 world model 服务机器人成为新兴范式；(3) 扩散模型的高效化和端侧部署在蒸馏和架构创新双轨驱动下快速推进。
