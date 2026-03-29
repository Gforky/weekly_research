# ArXiv 调研报告 - 2026-03-29

本周 ArXiv 共发表 LLM 训练效率、视频生成、Diffusion 加速相关论文若干。本报告覆盖三个方向（Fault Tolerance 方向本周无新论文，予以跳过）。

---

## 一、大模型训练效率优化（LLM Training Efficiency）

本周在 LLM 训练与推理效率方向有多个值得关注的工作，涵盖 RL 后训练、LoRA 参数高效微调、推理加速以及联邦学习场景下的对齐优化。

**Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation**
链接：https://arxiv.org/abs/2603.19220

NVIDIA 研究团队推出了 Nemotron-Cascade 2 系列，一个拥有 30B 参数（激活 3B）的 MoE 模型，在数学和代码推理上接近前沿模型，并成为继 DeepSeekV3.2-Speciale 之后第二个在 IMO、IOI 和 ICPC World Finals 上获得金牌的开源模型。核心技术是在 SFT 之后大幅扩展了 Cascade RL 的覆盖范围，使其涵盖更广泛的推理和 agent 领域，并通过多域在线策略蒸馏实现了极强的智能密度——参数量减少 20 倍。

这项工作的意义在于展示了 RL 后训练阶段 Scaling 的重要性，尤其是在推理和 agent 任务上的突破 [HIGH]。

---

**Adaptive Layerwise Perturbation: Unifying Off-Policy Corrections for LLM RL**
链接：https://arxiv.org/abs/2603.19470

这篇论文聚焦 LLM RL 训练中的核心瓶颈：离策略（off-policy）带来的策略过期和训练-推理分布不匹配。当策略在局部尖锐（locally sharp）时，重要性比率呈重尾分布，进一步放大尖锐梯度并可能导致更新超出信任域。

作者提出 ALP（Adaptive Layerwise Perturbation），在每层输入隐藏状态上注入小的可学习扰动，将其作为重要性比率的分子来对抗推理策略的未变化部分。通过对中间表征添加受控噪声，ALP 防止更新后策略与推理策略偏差过大，同时扩大策略家族以覆盖含噪声的推理策略家族，从而使分布变平、收紧不确定性估计 [HIGH]。

---

**ParallelVLM: Lossless Video-LLM Acceleration with Visual Alignment Aware Parallel Speculative Decoding**
链接：https://arxiv.org/abs/2603.19610

视频理解模型的自主码解码效率受到大量视频 token 的严重制约。现有视觉 token 剪枝方法存在信息丢失和加速比有限的问题。

ParallelVLM 是一种无需训练的"先草稿后验证"推测解码框架，解决了长视频设置中草稿模型和目标模型之间的"相互等待"以及加速比受限的双重问题。其核心创新在于两个并行化阶段最大化硬件利用率，以及无偏验证器引导剪枝策略（UAGP），通过消除注意力引导剪枝中的位置偏差来更好对齐草稿和目标模型。实验表明，ParallelVLM 在 LLaVA-Onevision-72B 上实现了 3.36 倍加速，Qwen2.5-VL-32B 上实现 2.42 倍加速 [HIGH]。

---

**BEAVER: Training-Free Hierarchical Prompt Compression via Structure-Aware Page Selection**
链接：https://arxiv.org/abs/2603.19635

随着 LLM 上下文窗口指数级扩展，长文档理解能力得到解锁，但同时也带来了严重的推理延迟和信息利用瓶颈。现有的压缩方法要么训练成本高，要么因激进 token 剪枝导致语义碎片化。

BEAVER 是一个无需训练的框架，将压缩从线性 token 删除转变为结构感知的分层选择。具体来说，它通过双路径池化将变长上下文映射到密集的 page 级张量以最大化硬件并行性，并通过语义和词法双分支选择加句子平滑的混合 planner 来保持话语完整性。在 128k 上下文上延迟降低 26.4 倍，性能可与 LongLLMLingua 等 SOTA 方法比肩 [MEDIUM]。

---

**FedPDPO: Federated Personalized Direct Preference Optimization**
链接：https://arxiv.org/abs/2603.19741

在联邦学习场景下对齐 LLM 与人类偏好面临三大挑战：去中心化、隐私敏感性和高度非 IID 的偏好数据。直接将 DPO 应用于联邦学习会在非 IID 数据下出现严重性能下降。

FedPDPO 提出参数高效微调架构，每个客户端维护一个冻结的预训练 LLM 主干加上 LoRA 适配器，实现通信高效的聚合。为解决非 IID 异构性，方法包括：（1）全局共享 LoRA 适配器配合个性化客户端特定 LLM 头；（2）带个性化客户端显式奖励头的个性化 DPO 训练策略，弥补隐式奖励的不足 [MEDIUM]。

---

**F2LLM-v2: Inclusive, Performant, and Efficient Embeddings for a Multilingual World**
链接：https://arxiv.org/abs/2603.19223

这篇论文提出 F2LLM-v2，一个支持 200+ 语言的通用多语言 embedding 模型家族（80M 到 14B 共 8 个规模）。通过两阶段 LLM-based embedding 训练管道，结合 matryoshka learning、模型剪枝和知识蒸馏技术，在 MTEB 基准上 14B 模型排名第一，同时远小于此前 LLM-based embedding 模型的规模 [MEDIUM]。

---

**Reinforcement Distillation of LLMs via Explanatory Inversion**
链接：https://arxiv.org/abs/2603.19266

将大语言模型的推理能力蒸馏到小模型中仍具挑战——蒸馏模型常出现浅层模式记忆和泛化不足的问题。

本文提出 Explanatory Inversion（EI）框架，生成针对性的"解释性探针"迫使学生模型阐述答案背后的逻辑而非简单记忆；同时提出 Explanatory GRPO（EXGRPO），通过新颖的对话结构效用奖励显式奖励学生维持连贯的推理过程。在 12 个数据集上的评估显示显著改进——使用 Gemma-7b 作为学生模型优于基线方法 [MEDIUM]。

---

## 二、视频生成（Video Generation）

视频生成是本周最活跃的方向，涵盖 DiT 架构效率优化、4K 图像到视频生成、3D 一致性、混合模态以及视频压缩等多个子方向。

**SVOO: Training-Free Sparse Attention for Fast Video Generation via Offline Layer-Wise Sparsity Profiling and Online Bidirectional Co-Clustering**
链接：https://arxiv.org/abs/2603.18636

DiT 在视频生成上效果出色但 3D 注意力带来高推理成本。现有免训练稀疏注意力方法有两个未解决局限：忽视注意力剪枝中的层间异质性，以及忽视块划分中的 query-key 耦合。

SVOO 的核心发现是每层的注意力稀疏度是其内在属性，对不同输入影响很小。方法分两阶段：（i）离线逐层敏感度分析以得出每层剪枝级别；（ii）在线稀疏注意力通过新型双向共聚类算法实现。在 7 个视频生成模型上验证了质量-加速权衡的改善 [HIGH]。

---

**NVFP4/INT8 Mixed-Precision Quantization for Video Diffusion Models**
链接：https://arxiv.org/abs/2603.18742

视频扩散模型的实用部署受到高内存占用和计算成本的严重制约。现有 PTQ 方法通常采用静态位宽分配，忽视了不同扩散时间步上激活量化的难度差异。

本文发现块的输入输出差异与内部线性层量化敏感度之间存在强线性相关。基于此，设计轻量预测器动态分配 NVFP4 给时间稳定层以最大化内存压缩，同时选择性地保留 INT8 给不稳定层以确保鲁棒性。此外观察到残差连接在保护关键信息中的作用。实现了无需训练的自适应精度策略 [HIGH]。

---

**FrescoDiffusion: 4K Image-to-Video with Prior-Regularized Tiled Diffusion**
链接：https://arxiv.org/abs/2603.17555

基于扩散的图生视频（I2V）模型在超高分输入（如 4K）上表现困难：按模型原生分辨率生成会丢失细粒度结构，而高分瓦片去噪能保留局部细节但破坏全局布局一致性——在壁画动画场景中尤为突出（多角色、多物体、不同语义子场景需空间连贯）。

FrescoDiffusion 的核心思路是用预计算的潜在先验增强瓦片去噪：先生成低分辨率视频并上采样其潜在轨迹作为全局参考，然后对 4K 生成计算每块噪声预测并在边界处与参考融合。实现无需训练的 coherent 大幅面 I2V 生成 [MEDIUM]。

---

**TAPESTRY: From Geometry to Appearance via Consistent Turntable Videos**
链接：https://arxiv.org/abs/2603.17735

从无纹理 3D 模型自动生成逼真且自一致的 appearance 是数字内容创建的关键挑战。现有通用视频扩散模型难以在全部视角范围内保持严格几何一致性和 appearance 稳定性。

TAPESTRY 提出将 3D appearance 生成重新表述为几何条件视频扩散问题：给定 3D 网格，先渲染并编码多模态几何特征，以像素级精度约束视频生成过程。同时提出 hallucination-as-supervision 管道，使用微调扩散模型为此前未观察到的身体区域生成密集监督信号 [MEDIUM]。

---

**ChopGrad: Pixel-Wise Losses for Latent Video Diffusion via Truncated Backpropagation**
链接：https://arxiv.org/abs/2603.17812

视频扩散模型的循环帧处理机制使得在像素域训练时激活跨整个视频序列累积，导致 prohibitive 的内存成本，也使得用像素损失微调长/高分辨率视频变得计算上不可行。

ChopGrad 提出截断反向传播方案，将梯度计算限制在局部帧窗口同时保持全局一致性。将训练内存从与帧数的线性关系降低到常数内存，且能在视频超分辨率、视频修复、视频编辑等任务上与 SOTA 视频扩散模型比肩 [MEDIUM]。

---

**Motion-Adaptive Temporal Attention for Lightweight Video Generation with Stable Diffusion**
链接：https://arxiv.org/abs/2603.17398

这篇论文为基于冻结 Stable Diffusion 的参数高效视频生成提出运动自适应时间注意力机制：动态根据估计的运动内容调整时间注意力感受野——高运动序列局部关注帧间变化，低运动序列全局关注以强化场景一致性。

通过级联策略将轻量时间注意力模块注入所有 UNet transformer 块（仅 25.8M 可训练参数，占 UNet 的 2.9%），在 WebVid 验证集上展现出竞争力 [MEDIUM]。

---

**Hybrid Spatial Memory for Controllable Video World Models**
链接：https://arxiv.org/abs/2603.17117

视频扩散模型正从短片段走向世界模拟器，需要在摄像机运动、回顾和干预下保持一致性。空间记忆仍是关键瓶颈——显式 3D 结构可以改善基于重投影的一致性但存在局限。

本文提出混合空间记忆机制，在视频世界模型中实现更可控的长期一致性 [MEDIUM]。

---

## 三、Diffusion 模型蒸馏（Diffusion Distillation）

Diffusion 蒸馏方向本周的核心进展集中在采样加速和训练效率优化。

**Few-Step Diffusion Sampling Through Instance-Aware Discretizations**
链接：https://arxiv.org/abs/2603.17671

扩散和流匹配模型通过模拟由 ODE/SDE 定义的路径来生成数据。概率流 ODE 公式使得可以使用高级数值求解器加速采样。正交于求解器设计的是离散化策略——现有方法大多对所有样本强制统一的时间步调度，无法适应实例级复杂度差异。

本文在受控合成数据实验中发现全局调度在实例特定动力学下的次优性，进而提出实例感知离散化框架：学习基于输入相关先验自适应时间分配，将基于梯度的离散化搜索扩展到条件生成设置。在多种设置（合成数据、图像、文本到图像）上验证了有效性 [HIGH]。

---

**Diff-SIT: Efficient Video Diffusion with Sparse Information Transmission**
链接：https://arxiv.org/abs/2603.18501

视频压缩的目标是以最小比特率最大化重建质量。超低比特率下，传统端到端压缩模型产生感知质量差的模糊图像；现有生成压缩方法通常独立处理帧，时间一致性和效率存在局限。

Diff-SIT 提出稀疏时间编码模块（STEM）将原始帧序列稀疏编码为信息丰富的中间序列，实现显著比特率节省；随后通过单步视频扩散和帧类型嵌入器（ODFTE）处理中间序列。帧类型嵌入器（FTE）引导扩散模型根据不同帧类型执行自适应重建以优化整体质量 [MEDIUM]。

---

**GeCO: Time Unconditional Flow Matching for Adaptive Robotic Control**
链接：https://arxiv.org/abs/2603.17834

扩散模型和流匹配已成为机器人模仿学习的基石，但存在结构性低效：推理通常绑定于固定的整合调度表，与状态复杂度无关，导致简单动作和复杂任务消耗相同计算预算。

GeCO（Generative Control as Optimization）将动作合成从轨迹整合转变为迭代优化。学习动作序列空间中的平稳速度场，专家行为形成稳定吸引子，测试时推理成为基于收敛的自适应过程——简单状态提前退出，复杂状态精细化。此外，平稳几何提供一个内在的、免训练的安全信号 [MEDIUM]。

---

**CUCo: An Agentic Framework for Compute and Communication Co-design**
链接：https://arxiv.org/abs/2603.02376

自定义 CUDA 内核开发对大规模分布式 LLM 训练和推理中的 GPU 利用率至关重要，但联合优化计算和通信的手动内核编写仍然劳动密集且容易出错。

CUCo 是一个免训练 agent 驱动的工作流，自动生成高性能 CUDA 内核联合编排计算和通信。超越仅关注计算的现有方法，将通信内核纳入联合优化，端到端延迟降低最高 1.57 倍 [MEDIUM]。

---

## 本周亮点

本周有三个值得重点关注的方向：

**1. LLM 后训练阶段正在成为新的 Scaling 方向。** Nemotron-Cascade 2 通过扩展 Cascade RL 的覆盖范围和 Multi-Domain On-Policy Distillation 在 30B MoE 规模上实现了接近前沿模型的推理能力，而 ALP 则在 RL 训练稳定性上带来突破。这表明后训练阶段（而非预训练）的效率优化正在成为新的研究热点。

**2. DiT 视频生成效率优化进入爆发期。** 本周多篇论文从不同角度攻破 DiT 效率瓶颈：SVOO 通过稀疏注意力、NVFP4/INT8 混合精度量化、ChopGrad 的截断反向传播等方法共同指向一个趋势——DiT 视频生成正在从"能生成"向"能实时"快速演进。

**3. Diffusion 采样加速从统一调度走向实例感知。** 2603.17671 的工作打破了传统上对所有样本使用相同时间步调度的范式，是扩散模型采样理论的重要进展，标志着个性化调度研究的兴起。

**Fault Tolerance 方向本周（2026-03-22 至 2026-03-29）无相关新论文，该方向跳过。**

---

*本报告由自动化 ArXiv 监控系统生成，覆盖 cs.LG、cs.CL、cs.AI、cs.CV 等分类下相关论文。置信度标注：[HIGH] 表示多来源一致且有充分实验验证，[MEDIUM] 表示单一来源或验证有限。*
