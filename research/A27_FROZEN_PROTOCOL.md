# A27 冻结机制诊断协议

来源是 A26 `2603f3bd16c48e48587fedc4363975779def17f5`。A25/A26 原代码、结果、gate、checkpoint 和标签保持冻结。A27 不训练、不生成标签、不调 rank、不打开 locked ID/OOD。仅 known-background 二维 TM、固定复相位标签和已暴露 development 的证据。

## 一次 post-decoder 修正

全部 16 validation parents、每 parent 八条件（128），N3 主比较、N1 辅助；保留 NONE/L/F/TASK_R32。在线函数只接受预测 mass、测量、已知 sigma/background，不接受 target。各臂从同一 CNN 输出开始。

共有目标是 A25 归一化数据目标 `0.5||F(chi)-d||²/||yb||² + lambda TV(chi-chi_b)`，lambda 为原 A25 噪声规则乘0.1。共有 box 不变。各臂在自身模型求一次数据梯度及方向曲率，依固定规则确定初始步长；最多八次 halving，认证 A25 accelerated PDHG prox。更新质量/是否接受只能依据 full F 目标、full data residual、固定 mass 信赖半径0.15及冻结先验。要求 full data loss不增，Armijo依真实候选步的norm计算；拒绝返回原图。每臂均支付 full F 接受审核，R不能据此宣称独立廉价部署。此次是一次 gradient/prox修正，不是GN、完整新重建或最优物理headroom上界。

从相同预测重算起点 full F（无共用隐藏免费状态）并记录 standalone部署费用；离线分析可共享已付状态，另列actual成本。原A25用多次更新和各自模型目标，本试验改变为共同full-F审核，因此不能将单次结果与A25 K12直接解释为唯一因果。

## 算子与信息机制

全 cohort在 common BP、一次L物理初值、N1/N3预测及各臂post输出比较L/F/R；独立随机方向/负载seed固定，不依质量挑选。target仅在离线误差alignment使用。R32不安全失败留账，无full fallback。背景与预测state局部谱诊断在元数据固定16代表条件×N1/N3（32预测states）；所有方向/谱只作OFFLINE可观测性诊断，不是在线特征。soft confidence不是后验不确定性。

## 决策与计费

五个研究问题是机制判别，没有publication级blind PASS。运行前冻结开发筛选：Case A要求F对NONE复杂家族平均parent收益≥5%、对L≥2%、配对interval下限>0且至少两个复杂家族改善。Case C为F/L材料收益差点估计绝对≤2%、interval含0；其余数据拟合主收益归入Case B或明确inconclusive，不能强行选正。使用2000次parent paired bootstrap，先平均八条件，缺配对/失败单独计入完整性。

CPU/GPU新增各≤3600s，所有失败/准备/审核/发布记账，GPUhostCPU另计。已有数据/checkpoints是历史成本，当前读取/重新物理动作收费。无SHA256检查。无loss/network/标签扩展。

## Task 4 工程对照

保持A26冻结结果；新增no-beta-only、同state-LU-reuse-only、combined三个独立lossless版本。删除仅用于beta诊断的L前向须显式标记beta不可用；features/material不得改变。材料域、core、forward/adjoint backward residual及owner/cache保护保持。native全程外层同步，与分阶段CUDA-event诊断分开。
