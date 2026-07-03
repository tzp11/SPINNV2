# 文献调研报告

> 调研时间：2026-06-20
> 覆盖方向：星载推理框架、NN软件容错、保护策略优化、故障注入与脆弱性分析
> 总计文献：~70篇（去重后）

---

## 一、星载/边缘神经网络推理框架与部署（25篇）

### 1.1 星载专用推理框架与里程碑任务

**[S1] Giuffrida G. et al. "The Phi-Sat-1 Mission: The First On-Board Deep Neural Network Demonstrator for Satellite Earth Observation"**
- IEEE Transactions on Geoscience and Remote Sensing, Vol. 60, 2022
- DOI: 10.1109/TGRS.2021.3125567
- 核心贡献：ESA Phi-Sat-1任务，首次在卫星上运行深度CNN（CloudScout）进行星上云检测。使用Intel Movidius Myriad 2 VPU，推理325ms，功耗1.8W，模型仅2.1MB。里程碑意义的工作。

**[S2] Mateo-Garcia G. et al. "In-orbit demonstration of a re-trainable machine learning payload for processing optical imagery"**
- Nature Scientific Reports, 13, 10391, 2023
- 核心贡献：在D-Orbit ION卫星上演示可重训练ML负载WorldFloods，星上洪水地图生成。使用Intel Myriad X VPU（1 TFLOPS / ~1W），证明在轨模型更新可行性。

**[S3] "Implementing a Neural Network Execution Framework in Realistic Space Hardware and Software as a Pseudo On-Orbit Demonstration"**
- Small Satellite Conference, 2024
- 核心贡献：提出NNEF（Neural Network Execution Framework），运行于NASA core Flight System (cFS)之上。最接近"星载通用推理框架"概念的工作。

**[S4] Denby B. & Lucia B. "Orbital Edge Computing: Nanosatellite Constellations as a New Class of Computer System"**
- ASPLOS 2020 (Best Paper Award)
- DOI: 10.1145/3373376.3378473
- 核心贡献：提出轨道边缘计算（OEC）范式。使用Jetson TX2星上推理，OEC比传统弯管架构减少24倍地面基础设施，延迟降低617倍。

**[S5] Rapuano E. et al. "An FPGA-Based Hardware Accelerator for CNNs Inference on Board Satellites"**
- Remote Sensing (MDPI), 13(8), 1518, 2021
- 核心贡献：Zynq UltraScale+上实现CloudScout的FPGA加速，推理141.68ms（比Myriad 2快2.4倍），功耗3.4W。VPU vs FPGA星载trade-off对比。

### 1.2 星载推理综述

**[S6] Furano G. et al. "Review on Hardware Devices and Software Techniques Enabling Neural Network Inference Onboard Satellites"**
- Remote Sensing (MDPI), 16(21), 3957, 2024
- 核心贡献：系统综述星载NN推理硬件（Myriad 2/X, Jetson, FPGA, 辐射加固处理器）和软件（ONNX, OpenVINO, TensorRT, 量化/剪枝）技术。

**[S7] Yang Y. et al. "FPGA-Based Neural Network Accelerators for Space Applications: A Survey"**
- arXiv: 2504.16173, April 2025
- 核心贡献：面向空间应用的FPGA NN加速器综述，分析辐射容忍、重构性、功耗效率。

**[S8] Al-Hababi M. et al. "Advancing Earth Observation: A Survey on AI-Powered Image Processing in Satellites"**
- arXiv: 2501.12030, January 2025 / International Journal of Remote Sensing, 2025
- 核心贡献：综述AI驱动的卫星图像处理，覆盖10,500+颗活跃卫星的数据处理挑战。

**[S9] "Satellite Edge AI with Large Models: Architectures and Technologies"**
- Science China Information Sciences, 2025
- 核心贡献：大型AI模型在卫星边缘的部署架构。

**[S10] "A Comprehensive Survey of Orbital Edge Computing: Systems, Applications, and Algorithms"**
- Chinese Journal of Aeronautics, 2024/2025
- 核心贡献：轨道边缘计算全面综述，GEO（空间云）+ LEO（边缘）分层架构。

### 1.3 星载推理与辐射容错

**[S11] Li J. et al. "When Single Event Upset Meets Deep Neural Networks"**
- arXiv: 1909.04697, 2019
- 核心贡献：首次系统研究SEU对DNN推理的影响，提出选择性加固策略。

**[S12] "Research on Spaceborne Neural Network Accelerator and Its Fault Tolerance Design"**
- Remote Sensing (MDPI), 17(1), 69, 2025
- 核心贡献：FPGA星载CNN加速器，融合静态冗余（权重纠错）和动态冗余（推理TMR），推理264.72ms，功耗1.6W。

**[S13] "Mitigating Multiple Single-Event Upsets During DNN Inference Using Fault-Aware Training"**
- arXiv: 2502.09374, February 2025
- 核心贡献：容错感知训练（FAT），无需修改硬件提升DNN对SEU鲁棒性最高3倍。

**[S14] Pitonak M. et al. "CloudSatNet-1: FPGA-Based Hardware-Accelerated Quantized CNN for Satellite On-Board Cloud Coverage Classification"**
- Remote Sensing (MDPI), 14(13), 3180, 2022
- 核心贡献：Zynq Z7020上极低位宽（2-4bit）量化CNN，BNN功耗2.4W，推理2.8ms，比GPU快7.9倍。

**[S15] "A Case for Application-Aware Space Radiation Tolerance in Orbital Computing" (RedNet)**
- arXiv: 2407.11853, July 2024
- 核心贡献：RedNet利用DNN各层对SEU的不均匀敏感性，改造激活函数+多出口策略。已在超瞳-1 SAR卫星上部署验证。唯一真正上星的容错NN工作。

**[S16] "Radiation Tolerant Deep Learning Processor Unit (DPU) based platform using Xilinx 20nm Kintex UltraScale FPGA"**
- IEEE Conference, 2022
- 核心贡献：辐射容忍XQRKU060 FPGA上深度学习处理单元，TMR MicroBlaze + SEM-IP + FAT三重防护。5.7 TOPS INT8。

### 1.4 通用嵌入式推理框架

**[S17] "Optimizing Deep Learning Models for On-Orbit Deployment Through Neural Architecture Search"**
- Nature Scientific Reports, 2025
- 核心贡献：CubeSat的NAS框架，在Jetson AGX Orin和Myriad X上联合优化。模型<1MB。

**[S18] Lin J. et al. "MCUNet: Tiny Deep Learning on IoT Devices"**
- NeurIPS 2020 (Spotlight)
- 核心贡献：TinyNAS + TinyEngine联合设计，MCU上>70% ImageNet top-1。TinyEngine比TFLite Micro减少3.4倍峰值SRAM。

**[S19] Liu C. et al. "Deploying Machine Learning Models to Ahead-of-Time Runtime on Edge Using MicroTVM"**
- CODAI 2023 / arXiv: 2304.04842
- 核心贡献：端到端代码生成器，预训练模型→C源码库，ARM Cortex-M4F裸机AOT推理。

**[S20] "Efficient FPGA-accelerated CNNs for Cloud Detection on CubeSats"**
- arXiv: 2504.03891, April 2025
- 核心贡献：Vitis AI DPU上通道剪枝+INT8量化，Scene-Net 57.14 FPS / 2.5W。

**[S21] "Opportunities and Challenges of On-Board AI-Based Image Recognition for Small Satellite Earth Observation Missions"**
- Advances in Space Research, 2024
- 核心贡献：小卫星星上AI的机遇与挑战系统分析。

**[S22] "Efficient Acceleration of Deep Learning Inference on Resource-Constrained Edge Devices: A Review"**
- IEEE Proceedings, 2022
- 核心贡献：边缘设备推理加速方法综述。

**[S23] "Edge Intelligence: A Review of Deep Neural Network Inference in Resource-Limited Environments"**
- Electronics (MDPI), 14(12), 2495, 2025
- 核心贡献：边缘智能推理最新进展综述。

**[S24] Lockheed Martin. "AI/ML for Mission Processing Onboard Satellites"**
- AIAA 2022-1472
- 核心贡献：工业界星载AI部署，SmartSat平台。

**[S25] "Fast Model Inference and Training On-Board of Satellites (RaVAEn)"**
- arXiv: 2307.08700, 2023
- 核心贡献：D-Orbit ION卫星上部署RaVAEn，推理0.110s/tile并支持星上训练。

---

## 二、神经网络软件容错保护（25篇）

### 2.1 综述

**[F1] Bolchini C., Cassano L., Miele A. "Resilience of Deep Learning Applications: A Systematic Literature Review of Analysis and Hardening Techniques"**
- Computer Science Review, Vol. 54, 2024 (also arXiv:2309.16733)
- 核心贡献：最全面的近期综述，覆盖85篇论文(2019-2024)。分类DNN韧性分析和加固技术。附GitHub数据集。

**[F2] Liu C. et al. "Fault-Tolerant Deep Learning: A Hierarchical Perspective"**
- IEEE VTS 2022 (arXiv:2204.01942)
- 核心贡献：从模型层、架构层、电路层、跨层四个视角综述容错DNN设计。

**[F3] Rech P. et al. "Artificial Neural Networks for Space and Safety-Critical Applications: Reliability Issues and Potential Solutions"**
- IEEE Transactions on Nuclear Science, Vol. 71, pp. 377-404, 2024
- 核心贡献：桥接辐射效应和DNN可靠性两个社区。覆盖辐射对NN硬件的影响、故障传播、加固方案。

### 2.2 选择性加固 / 脆弱性驱动保护

**[F4] Mahmoud A. et al. "HarDNN: Feature Map Vulnerability Evaluation in CNNs"**
- arXiv:2002.09786, 2020 / NVIDIA Research
- 核心贡献：首个feature map级脆弱性分析。保护30%最脆弱feature map即可提升SqueezeNet韧性10倍。

**[F5] Mahmoud A. et al. "Optimizing Selective Protection for CNN Resilience" (FILR)**
- IEEE ISSRE 2021, pp. 127-138 (IEEE Micro Top Picks)
- 核心贡献：FLR（静态feature map保护）+ ILR（运行时异常推理重跑）。FILR实现99.78%错误覆盖，ResNet50仅20.8%额外MACs，Jetson Xavier上4.6%运行时开销。选择性保护领域最重要的baseline。

**[F6] Ruospo A. et al. "Selective Hardening of Critical Neurons in Deep Neural Networks"**
- DDECS 2022, pp. 136-141
- 核心贡献：软件TMR方法，识别关键神经元并选择性加固。ResNet和DenseNet上开销远低于全TMR。

**[F7] "Cost-Effective Fault Tolerance for CNNs Using Parameter Vulnerability Based Hardening and Pruning"**
- arXiv:2405.10658, 2024
- 核心贡献：选择性filter复制+校正层。15%加固比率即可在高故障率下可靠运行，开销12%。脆弱性引导剪枝后性能提升24%。

### 2.3 范围检查与激活值限制

**[F8] Chen Z., Li G., Pattabiraman K. "Ranger: A Low-Cost Fault Corrector for DNNs through Range Restriction"**
- IEEE/IFIP DSN 2021 (Best Paper Runner-up)
- 核心贡献：自动推导值域范围并插入限制操作。8个DNN上韧性提升3x-50x，开销可忽略。必引文献。

**[F9] Hoang L.-H. et al. "FT-ClipAct: Resilience Analysis of DNNs and Improving their Fault Tolerance using Clipped Activation"**
- DATE 2020, pp. 1241-1246
- 核心贡献：首个系统性激活值裁剪方法。VGG-16在1e-5故障率下精度提升68.92%。Ranger和FitAct的前驱工作。

**[F10] Ghavami B. et al. "FitAct: Error Resilient DNNs via Fine-Grained Post-Trainable Activation Functions"**
- DATE 2022, pp. 1239-1244
- 核心贡献：神经元级bounded激活函数(FitReLU)。训练后轻量级微调。优于Ranger和ClipAct，<12%运行时开销。

**[F11] "Dr. DNA: Combating Silent Data Corruptions in Deep Learning using Distribution of Neuron Activations"**
- ACM ASPLOS 2024
- 核心贡献：从激活值分布中提取SDC特征。95%平均SDC检测率，<1%内存，<2.5%延迟开销。

### 2.4 ABFT（算法级容错）

**[F12] Zhao K. et al. "FT-CNN: Algorithm-Based Fault Tolerance for Convolutional Neural Networks"**
- IEEE TPDS, Vol. 32, No. 7, 2021
- 核心贡献：CNN系统性ABFT方案，基于checksum技术，支持所有卷积实现方式。4-8%运行时开销。

**[F13] Kosaian J., Rashmi K.V. "Arithmetic-Intensity-Guided Fault Tolerance for NN Inference on GPUs"**
- SC'21, ACM
- 核心贡献：根据层的计算密度自适应选择ABFT方案。开销降低1.09-5.3x。

**[F14] Xue X. et al. "ApproxABFT: Approximate Algorithm-Based Fault Tolerance for Neural Network Processing"**
- arXiv:2302.10469, 2023
- 核心贡献：近似ABFT，利用DNN固有容错性设自适应阈值。减少43.39%计算开销，有效BER范围扩展一个数量级。使用贝叶斯优化。

### 2.5 DMR/TMR

**[F15] Li Y. et al. "D2NN: A Fine-Grained Dual Modular Redundancy Framework for Deep Neural Networks"**
- ACSAC 2019, pp. 138-147
- 核心贡献：首个细粒度DMR框架。选择性复制高敏感神经元。

**[F16] Xue X. et al. "Exploring Winograd Convolution for Cost-Effective Neural Network Fault Tolerance"**
- IEEE TVLSI, Vol. 31, No. 11, 2023
- 核心贡献：Winograd卷积的固有容错性可减少55.77%的TMR开销。

**[F17] Caro M., Brando A., Abella J. "Semantic Diverse DMR and TMR for High-Integrity AI-Based Functions"**
- ACM Trans. Cyber-Physical Systems, Vol. 9, No. 2, 2025
- 核心贡献：语义多样化的DMR/TMR，施加语义中性输入变换检测故障和模型错误。YOLOv4验证。

### 2.6 容错感知训练

**[F18] Zahid U. et al. "FAT: Training Neural Networks for Reliable Inference Under Hardware Faults"**
- IEEE ITC 2020
- 核心贡献：训练阶段注入故障模型提升QNN容错性。

**[F19] Cavagnero N. et al. "Transient-Fault-Aware Design and Training to Enhance DNNs Reliability with Zero-Overhead"**
- IEEE IOLTS 2022
- 核心贡献：零开销方案：激活函数选择+层排序+容错感知训练。误预测率降低一个数量级。

### 2.7 综合保护系统

**[F20] "Adaptive Soft Error Protection for Neural Network Processing" (GNN-based)**
- arXiv:2407.19664, 2024
- 核心贡献：揭示NN脆弱性是输入依赖的。GNN模型运行时预测脆弱性，自适应保护。比静态保护减少42.12%开销。

**[F21] Zheng W. et al. "SAVE: Software-Implemented Fault Tolerance for Model Inference against GPU Memory Bit Flips"**
- USENIX ATC'25
- 核心贡献：四阶段软件容错系统（Selection, Allocation, Verification, Edit）。<9%开销下容忍4K同时bit flip。

### 2.8 星载/辐射环境特定

**[F22] RedNet (同S15)**
- arXiv:2407.11853, 2024
- 核心贡献：激活函数改造+多出口策略。已在超瞳-1卫星上验证。

**[F23] dos Santos F.F. et al. "Characterizing a Neutron-Induced Fault Model for Deep Neural Networks"**
- IEEE TNS, Vol. 70, 2022
- 核心贡献：单bit-flip模型比中子束实验低估8.66倍错误率。ECC对DNN关键错误无效（61%关键错误率）。

**[F24] dos Santos F.F., Rech P. et al. "Analyzing and Increasing the Reliability of CNNs on GPUs"**
- IEEE Trans. Reliability, Vol. 68, No. 2, 2019
- 核心贡献：YOLO/Faster R-CNN/ResNet在三种NVIDIA GPU上的中子束测试。单GPU故障传播至多线程。

**[F25] "Demystifying GPU Reliability: Comparing and Combining Beam Experiments, Fault Simulation, and Profiling"**
- IEEE/IFIP DSN 2021 (NVIDIA Research)
- 核心贡献：直接对比中子束实验（1300万年辐射暴露）vs 架构级故障仿真 vs profiling。

---

## 三、保护策略优化方法（19篇）

### 3.1 ILP/数学优化（研究空白区）

**[O1] Bertoa et al. "Fault-Tolerant Neural Network Accelerators with Selective TMR"**
- IEEE TCAD, 2022
- 核心贡献：自动化工具分析NN加速器敏感计算并选择性TMR。本质是背包问题变体，但用启发式而非ILP求解。最接近ILP保护分配的工作。

**[O2] "ApproxABFT" (同F14)**
- arXiv:2302.10469, 2023
- 核心贡献：贝叶斯优化多参数协同优化。ILP的替代方法。

**[O3] "A Methodology for Fault-tolerant Pareto-optimal Approximate Designs of FPGA-based Accelerators"**
- ACM TECS, 2022
- 核心贡献：Pareto搜索确定哪些单元精确/近似+哪些TMR。最接近"预算约束下Pareto最优保护"的工作。

### 3.2 脆弱性驱动的选择性加固

**[O4] HarDNN (同F4)**
**[O5] FILR (同F5)**

**[O6] "Cost-Effective Fault Tolerance" (同F7)**

**[O7] Libano F. et al. "Selective Hardening for Neural Networks in FPGAs"**
- IEEE TNS, Vol. 66, No. 1, 2019
- 核心贡献：开创性工作，FPGA上NN的层级选择性TMR。关键错误降低14%，仅8%额外开销。

### 3.3 预算约束下的保护

**[O8] "NAPER: Fault Protection for Real-Time Resource-Constrained Deep Neural Networks"**
- arXiv:2504.06591, 2025
- 核心贡献：可靠性 vs 准确度 vs 时效性三方困境。异构模型冗余的集成学习方案。

**[O9] FT-CNN (同F12)**

### 3.4 风险画像/脆弱性排序方法论

**[O10] Yvinec E. et al. "SAfER: Layer-Level Sensitivity Assessment for Efficient and Robust NN Inference"**
- arXiv:2308.04753, 2023
- 核心贡献：层级敏感度评估，同时服务于压缩和容错。与加权复合评分直接可比的方法论。

**[O11] "Adaptive Soft Error Protection" (同F20, GNN-based)**

**[O12] "Analyzing the Impact of Soft Errors in VGG Networks on GPUs" (KVF/LVF)**
- Microelectronics Reliability, 2020
- 核心贡献：核函数脆弱性因子(KVF) + 层脆弱性因子(LVF)。仅加固Im2col即降低85.67% SDC率。

**[O13] "Efficient TMR for Reliability" (XAI-guided Selective TMR)**
- arXiv:2507.08829, 2025
- 核心贡献：LRP层级相关性传播识别关键权重，选择性TMR。84.37%面积缩减。

### 3.5 其他优化方法

**[O14] "Applying Reinforcement Learning to Protect DNNs from Soft Errors"**
- Sensors 2025, 25(13), 4196
- 核心贡献：RL-based agent动态选择保护脆弱位。ILP的重要竞争对比方法。

**[O15] Ranger (同F8)**
**[O16] FitAct (同F10)**
**[O17] Ares (同下G3)**

**[O18] "Fault-Aware Design and Training to Enhance DNNs Reliability with Zero-Overhead"**
- arXiv:2205.14420, 2022
- 核心贡献：归一化层统计量学习问题导致FAT效果有限，改变归一化和激活层顺序可显著改善。

**[O19] 综述 (同F1)**

---

## 四、故障注入框架与脆弱性分析（20篇）

### 4.1 故障注入工具

**[G1] Chen Z. et al. "TensorFI: A Flexible Fault Injection Framework for TensorFlow Applications"**
- IEEE ISSRE 2020, pp. 426-435 (arXiv:2004.01743)
- 核心贡献：首个TensorFlow计算图故障注入框架，operator级注入。

**[G2] Mahmoud A. et al. "PyTorchFI: A Runtime Perturbation Tool for DNNs"**
- IEEE/IFIP DSN-W 2020, pp. 25-31
- 核心贡献：PyTorch hook机制运行时故障注入，支持权重和激活值扰动。

**[G3] Reagen B. et al. "Ares: A Framework for Quantifying the Resilience of Deep Neural Networks"**
- DAC 2018
- 核心贡献：首次大规模DNN弹性实证研究。容错性差异达数量级。经硬件验证，误差12%。奠基性工作。

**[G4] Tsai T. et al. "NVBitFI: Dynamic Fault Injection for GPUs"**
- IEEE/IFIP DSN 2021
- 核心贡献：GPU架构级故障注入工具，无需源代码。Kepler到Ampere全系列支持。

**[G5] Zheng H. et al. "MRFI: An Open Source Multi-Resolution Fault Injection Framework"**
- IEEE TCAD 2024 (arXiv:2306.11758)
- 核心贡献：多分辨率故障注入工具。YAML配置驱动，不修改模型代码。

**[G6] Chen Z., Li G., Pattabiraman K. "BinFI: An Efficient Fault Injector for Safety-Critical ML Systems"**
- IEEE/IFIP DSN 2019
- 核心贡献：二分搜索高效定位99.56%安全关键位，远低于随机注入成本。

### 4.2 故障模型

**[G7] Zhao H. et al. "A Survey of Bit-Flip Attacks on Deep Neural Network and Corresponding Defense Methods"**
- MDPI Electronics, Vol. 12, No. 4, 2023
- 核心贡献：bit-flip攻击方法（BFA/T-BFA等）和防御手段系统综述。

**[G8] Hanif M.A. et al. "FAQ: Mitigating the Impact of Faults through Fault-Aware Quantization"**
- IJCNN 2023 (arXiv:2305.12590)
- 核心贡献：故障感知量化，stuck-at故障率0.04时精度提升76.38%。

### 4.3 脆弱性量化

**[G9] Li G. et al. "Understanding Error Propagation in Deep Learning Neural Network (DNN) Accelerators and Applications"**
- SC'17
- 核心贡献：DNN推理中软错误传播的奠基性工作。定义SDC-1/SDC-5分类标准。高位指数位0-to-1 flip最易导致SDC。

**[G10] "Resilience of Deep Learning Applications" (同F1)**

**[G11] Ahmadilivani M. et al. "DeepVigor: Vulnerability Value Ranges and Factors for DNNs' Reliability Assessment"**
- IEEE, 2023
- 核心贡献：参数级脆弱性值范围和因子的定量分析框架。

### 4.4 硬件 vs 软件故障注入对比

**[G12] dos Santos F.F. et al. "Demystifying GPU Reliability" (同F25)**
**[G13] "FPGA-Based Emulation and Fault Injection for CNN Inference Accelerators"**
- arXiv:2501.12818, 2025
- 核心贡献：Zynq UltraScale+ FPGA仿真平台，比软件仿真快一个数量级。

**[G14] "Evaluating Different Fault Injection Abstractions on the Assessment of DNN SW Hardening Strategies"**
- arXiv:2412.08466, 2024
- 核心贡献：APP级和ISA级故障注入结果存在显著差异，抽象层选择影响加固策略评估。

### 4.5 任务类型差异

**[G15] "Single-Event Upset Analysis of a Systolic Array based DNN Accelerator"**
- arXiv:2405.15381, 2024
- 核心贡献：脉动阵列DNN加速器SEU分析。故障率0.0003%时精度从97.4%跌至7.75%。

**[G16] "Soft Errors in DNN Accelerators: A Comprehensive Review"**
- Microelectronics Reliability, 2020
- 核心贡献：LeNet-5 vs YOLO不同任务类型对比。定点数据在内存/弹性间最佳权衡。

**[G17] "Evaluating and Enhancing YOLOv8's Soft Error Resilience" (YOLO-FI)**
- MDPI Electronics, Vol. 15, No. 7, 2026
- 核心贡献：YOLO-FI故障注入框架。选择性DMR仅对最后模块加固+激活值归零。

---

## 五、关键发现与研究空白

### 5.1 已被充分研究的方向
- 故障注入工具和方法论（TensorFI, PyTorchFI, Ares, MRFI等）
- Range-based保护（Ranger系列已成熟）
- DMR/TMR在NN中的应用（D2NN, 选择性TMR等）
- ABFT for CNN（FT-CNN等）

### 5.2 明确的研究空白
- **ILP/MILP形式化NN保护资源分配**：目前无直接工作
- **异构保护机制的统一优化框架**：多数工作研究单一机制
- **面向星载推理的端到端容错部署流水线**：现有工作要么只做推理框架，要么只做容错分析

### 5.3 最需要对比的baseline
1. FILR (Mahmoud et al., ISSRE 2021) — 选择性保护最强baseline
2. Ranger (Chen et al., DSN 2021) — 范围检查baseline
3. RedNet (2024) — 唯一星载验证的容错NN工作
4. FT-CNN (Zhao et al., TPDS 2021) — ABFT baseline

### 5.4 需要在limitations中讨论的问题
- 软件bit-flip注入 vs 真实中子束辐射差距8.66倍 [G12/F23]
- APP级 vs ISA级故障注入抽象差异 [G14]
- 静态脆弱性 vs 输入依赖的动态脆弱性 [F20]
