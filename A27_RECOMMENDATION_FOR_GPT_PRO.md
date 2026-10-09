# 给 GPT Pro 的定量架构设计约束

请从“材料误差位于哪些方向、能被哪些在线信息纠正、代价多少”出发。A27不是另一个训练项目，没有新网络或盲测；以下是下一实验的条件性约束，不是预设架构成功。

## 必须解释的五项证据

1. 冻结BP处反馈/噪声median0.169，0/64超过噪声；N3 CNN后median1.425，36/64超过噪声。相同网络的物理特征应避免只在反馈弱的位置计算；但“稍后加反馈”本身不够。
2. N3复杂形状full-F后处理收益 **1.137% [0.923, 1.351]**，F相对L **0.160% [0.073, 0.240]**；full data收益 **25.270% [23.657, 26.743]**。约96.2%代表材料误差在局部弱空间，99.97%数据梯度在强空间。不能把data loss当唯一优化目标或用更准的surrogate掩盖弱结构信息不足。
3. 初值R32梯度准确；CNN/post状态并不统一通过0.99cos。实际one-step R图像近F是共同full-F接受下的证据，不是廉价独立反馈的统一认证。禁止再次把rank32仅forward误差当成设计目标。
   实际复杂家族R/F位移相对差约10.03%，L/F约64.04%，但材料终点增量很小；原修正分母与零步状态都已保存。
4. 冻结N2本会话约100ms，删beta-only后约86ms，仍约2.13×普通BP+CNN；安全LU复用约114ms反而慢。exact后处理约260ms且质量只小改善。必须以完整raw-to-image latency衡量结构价值。
5. 现有TRAIN可提供电流/总场/相对源标签，不需要新labels。训练时current/field分支可与共同材料约束配合，部署可尝试移除；尚无训练效果，不能预写为sample-efficiency或shell优势。

## 哪些计算应成为显式结构

保留共同材料、passive box、相同噪声/packing、物理残差验收、state/owner失效保护。若需要物理更新，应由当前decoder材料假设驱动，并把新图encoder、adjoint、closure和接受审计计入部署。完整F可以作有限开发teacher/校验，不默认每block调用；当前数据不能支持为了小收益每图强制付费exact correction。

## 哪些应先通过训练时监督检验

优先一项小干预：在同材料decoder、数据、参数权限与更新数下增加源共享current辅助head，再用 `e=inc+Gc` 和 `c_s≈χ e_s` 约束共同材料。对照材料-only、current/receiver-data、再加内部field/material-update-aware weighting。辅助输出不得成为训练时teacher输入，test不得用truth计算M。若部署主输出不读取辅助head，才有移除可能；必须验证移除一致性和实际latency。

物理加权材料loss是更便宜的备选，只能称为weighting，不能声称学会Maxwell电流。弱空间confidence只可作软指标，posterior uncertainty须另外校准，不能硬宣告每一像素可恢复。

## 哪些直接删除或暂停

- 删除不作为CNN输入的beta-only L forward，诊断单独运行；不能伪造beta值。
- 暂停冻结弱初值R32编码器的扩展、rank sweep和以分解数下降为卖点的当前LU reuse。
- 不引入更大CNN、VAE/diffusion或用新标签补救此次结果。
- 不用微小residual、少参数或低rank代替复杂材料和总延迟指标。

## 下一实验如何可证伪

先冻结同一个BP/OM material decoder和训练权限，只检验一个training-only监督因素；原N3必须保留。family-wise材料、edge/IoU、full data、完整时间共同判定。若feedback或中间监督仅改善data而不改善复杂家族，淘汰其部署结构；若普通prior一样有效，删除OM特殊编码。若有可靠development结构收益，才单独冻结新协议并申请独立验证，不能重用本暴露集当holdout。

本项目未提供样本效率、层数减少或新的SOM-Net优越性证据。2D、known-background、固定复相位和4-parent谱范围必须出现在任何后续claim里。

完整入口：[机制报告](A27_MECHANISM_DIAGNOSIS.md)、[逐parent/CI](results/a27/artifacts_post_fgp/PAIRED_STATISTICS.json)、[延迟](A27_LATENCY_ATTRIBUTION.md)、[监督规格](A27_INTERMEDIATE_SUPERVISION_SPEC.md)。
