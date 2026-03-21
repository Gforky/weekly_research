# ArXiv 调研报告

## 周期：每周六上午 8 点发送

## 主题方向
1. 大模型训练效率优化
2. Fault-tolerance training
3. 视频生成
4. Diffusion 模型蒸馏

---

## 2026-03-14 报告

### 一、大模型训练效率优化

#### 1. Capacity-Aware Mixture Law Enables Efficient LLM Data Optimization
**链接**: https://arxiv.org/html/2603.08022v1
**摘要**: 研究 LLM 中期训练的数据混合优化方法，提出容量感知混合法则，可降低 50% 优化成本，下游基准性能提升最高 3%。

#### 2. OOMB: Highly Memory Efficient Training System for LLMs
**链接**: https://arxiv.org/html/2602.02108v4
**摘要**: 提出百万级上下文训练系统，集成页式 KV 缓存管理、稀疏注意力和异步 CPU 卸载，实现长上下文训练时常数级内存占用。

#### 3. Lost in Backpropagation: The LM Head is a Gradient Bottleneck
**链接**: https://arxiv.org/html/2603.10145
**摘要**: 发现 LM head 是梯度瓶颈，低秩 LM head 会阻碍优化动态，降低训练效率。

#### 4. Efficient Preference Optimization for Large Language Models
**链接**: https://arxiv.org/pdf/2602.24082
**摘要**: 提出 Preference Packing 方法，增强 DPO 等偏好优化技术的资源效率，训练吞吐量提升 3.22 倍。

#### 5. A-3PO: Accelerating Asynchronous LLM Training
**链接**: https://arxiv.org/html/2512.06547v3
**摘要**: 异步 LLM 训练加速方法，引入陈旧感知近端概率插值，训练时间加速 1.8 倍。

---

### 二、容错训练 (Fault-Tolerance Training)

#### 1. Training LLMs with Fault Tolerant HSDP on 100,000 GPUs
**链接**: https://arxiv.org/html/2602.00277v1
**摘要**: 探索 10 万 GPU 规模 LLM 训练挑战，提出 FT-HSDP 异步容错训练方法。故障恢复只需 3 分钟，12 副本训练保持 450 TFlops/GPU/s 吞吐量。

#### 2. SPARe: Stacked Parallelism with Adaptive Reordering
**链接**: https://arxiv.org/html/2603.00357v1
**摘要**: 提出用于 10 万+ GPU 容错 LLM 预训练的堆叠并行方法，实现自适应重排序的故障容忍系统。

#### 3. KevlarFlow: Fault Tolerance in LLM Serving
**链接**: https://arxiv.org/html/2601.22438v1
**摘要**: 首个支持 LLM 推理阶段节点故障容忍的框架，MTTR 降低 20 倍，故障条件下延迟提升 3.1 倍。

#### 4. Distributed LLM Pretraining During Renewable Curtailment Windows
**链接**: https://arxiv.org/html/2602.22760v1
**摘要**: 研究地理分布 GPU 集群在可再生能源 curtailment 窗口期间的 LLM 训练，提出弹性训练方案。

---

### 三、视频生成

#### 1. Physical Simulator In-the-Loop Video Generation
**链接**: https://arxiv.org/abs/2603.06408
**摘要**: CVPR 2026 接收论文，将物理模拟器嵌入视频生成循环，提升物理一致性。

#### 2. Real-Time Video Generation with Interactive Motion Controls
**链接**: https://arxiv.org/html/2511.01266v5
**摘要**: 实现交互式运动控制的实时视频生成，提出因果蒸馏管道支持长视频生成。

#### 3. Rethinking Video Generation Model for the Embodied World
**链接**: https://arxiv.org/abs/2601.15282
**摘要**: 重新思考视频生成模型在具身智能中的应用，用于生成多样化机器人数据。

#### 4. A Mechanistic View on Video Generation as World Models
**链接**: https://arxiv.org/abs/2601.17067
**摘要**: 从机制视角研究视频生成模型作为世界模型的物理一致性。

#### 5. DreamWorld: Unified World Modeling in Video Generation
**链接**: https://arxiv.org/abs/2603.00466
**摘要**: 视频生成中的统一世界建模方法。

#### 6. VideoAR: Autoregressive Video Generation
**链接**: https://arxiv.org/abs/2601.05966
**摘要**: 首个大规模视觉自回归 (VAR) 视频生成框架，结合多尺度预测。

#### 7. Real-Time Physical Action-Conditioned Video Generation
**链接**: https://arxiv.org/abs/2603.05449
**摘要**: 实时物理动作条件视频生成。

#### 8. Video Generation Models in Robotics
**链接**: https://arxiv.org/abs/2601.07823
**摘要**: 视频生成模型在机器人领域的应用综述，探讨研究挑战和未来方向。

---

### 四、Diffusion 模型蒸馏 (本周较少)

#### 1. TempoSyncDiff: Distilled Temporally-Consistent Diffusion
**链接**: https://arxiv.org/html/2603.06057
**日期**: 2026年3月6日
**摘要**: 提出用于低延迟音频驱动 Talking Head 的时间一致性蒸馏扩散模型。

#### 2. FiDeSR: High-Fidelity and Detail-Preserving One-Step Diffusion
**链接**: https://arxiv.org/html/2603.02692
**日期**: 2026年3月
**摘要**: 通过蒸馏将迭代扩散过程压缩为单步前向传递，实现高保真细节保留的单步生成。

#### 3. From Flow to One Step: Real-Time Multi-Modal Trajectory Policies
**链接**: https://arxiv.org/html/2603.09415
**日期**: 2026年3月
**摘要**: 提出基于 IMLE 分布蒸馏的框架，将多步 CFM 专家压缩为单步学生策略，实现 125Hz 实时推理。

#### 4. Streaming Autoregressive Video Generation via Diagonal Distillation
**链接**: https://arxiv.org/html/2603.09488
**日期**: 2026年3月
**摘要**: 提出针对视频生成的 flow-aware 对角蒸馏框架，利用时间上下文保持一致性同时减少采样步骤。

#### 5. TDM-R1: Reinforcing Few-Step Diffusion Models
**链接**: https://arxiv.org/html/2603.07700
**日期**: 2026年3月
**摘要**: 提出新型强化学习范式，使少步扩散模型能有效利用非可微奖励，性能超越多步模型。

---

## 本周亮点

- **容错训练**: 10 万 GPU 级别的 FT-HSDP 和 SPARe 论文很值得关注
- **视频生成**: 多篇 CVPR 2026 接收论文，物理一致性是热点
- **训练效率**: 内存优化和偏好训练加速是重点方向
- **Diffusion 蒸馏**: 本周论文较少，可能因为近期会议截稿
