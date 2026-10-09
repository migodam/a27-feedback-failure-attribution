# A27：反馈失效归因与架构余量

**开发诊断完成；未训练、未新增标签、locked ID/OOD未打开。** 完整反馈在CNN之后有小幅材料收益，主要改善数据拟合；冻结的明显复杂结构收益门槛未通过。结论带范围限定，不能改写A25/A26历史结果。

- N3复杂三家族F材料收益：1.137% [0.923, 1.351]；F对L增量：0.160% [0.073, 0.240]。
- 固定谱代表状态约96.2%的误差在弱空间，约99.97%的数据梯度在强空间。
- 删beta-only诊断约13.37%节省，仍慢于普通BP+CNN；安全LU reuse减少分解却变慢。
- 训练时current/field监督标签可行，效果为SPEC_ONLY。

[完整中文机制报告](A27_MECHANISM_DIAGNOSIS.md) → [CNN后修正](A27_POST_DECODER_CORRECTION.md) → [弱空间/先验](A27_OBSERVABILITY_AND_PRIOR.md) → [实际延迟](A27_LATENCY_ATTRIBUTION.md) → [监督规格](A27_INTERMEDIATE_SUPERVISION_SPEC.md) → [GPT Pro设计约束](A27_RECOMMENDATION_FOR_GPT_PRO.md)。

[冻结gate](A27_GATE_DECISION.json) · [复现](REPRODUCING_A27.md) · [成本/失败](A27_COST_AND_RUNTIME.md)。原A26的根目录`GATE_DECISION.json`与历史报告保持冻结，不是A27 gate。

网页入口：[仓库](https://github.com/migodam/a27-feedback-failure-attribution) · [中文报告](https://github.com/migodam/a27-feedback-failure-attribution/blob/main/A27_MECHANISM_DIAGNOSIS.md) · [给GPT的匿名raw](https://raw.githubusercontent.com/migodam/a27-feedback-failure-attribution/main/A27_RECOMMENDATION_FOR_GPT_PRO.md) · [Release](https://github.com/migodam/a27-feedback-failure-attribution/releases/tag/a27-v0.1.0)。发布是否完成由实际HTTP/commit receipt判断，科学筛选与发布分开。
