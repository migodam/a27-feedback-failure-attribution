# 训练时物理中间量监督：最小可实施方案

**SPEC_ONLY；未实现新网络、未训练，效果 NOT_RUN。** 现有材料误差监督没有直接要求 CNN 解释每个照明的电流或总场。训练时增加这类约束可以检验先验是否学到了与 Maxwell 一致的材料结构；它不会增加测量信息，也不能保证弱空间里的材料唯一恢复。

## 已有标签与单位

A25 TRAIN 已保存 `current[512,576,8]` complex128，属于64 parents、256个独立 clean states。clean/20dB 两行共享同一材料、电流和无噪声数据；不是512个独立对象。总场和相对对比源没有单独保存，但可从已有标签恢复：

\[
e_s=e_s^{inc}+Gc_s,\quad c_s=\chi e_s,\quad
w_s=(\chi-\chi_b)e_s=L_b(c_s-c_{b,s}),\quad
\Delta y_s=D_bw_s.
\]

这里是二维 TM、质量归一化的代码单位，`V=1/576`；不能把它当成三维矢量电流或未经换算的 SI 单位。全256状态核验的最大相对误差分别为：本构 `1.83245e-15`、两种 w 定义 `7.13161e-14`、相对数据恒等式 `1.22567e-15`。没有执行新的非线性求解或生成标签。详细审计与标量原始数据见 [标签审计](research/delegated/a27_supervision/SUPERVISION_AUDIT.md)。单独 A26 Release 的简化输入不带 current，需要已有 A25 TRAIN 电流缓存；缺失时必须报告 DATA_NOT_AVAILABLE。

## 只增加一个训练辅助分支

保留材料 decoder 的部署图。从共同主干接一个跨源共享权重的 current head，在线输入仅为测量编码、源位置和已知背景场，输出 \(\hat c_s\)。令 \(\hat e_s=e_s^{inc}+G\hat c_s\)，不增加可独立违背场关系的自由 field head。材料主输出不读取 auxiliary head；训练后可以删除 head、G 作用与辅助 loss。删除前后材料输出一致性仍需数值验证，不能预写为已通过。

各对照须有相同标签权限、辅助参数、更新数、采样和归一化。建议依次比较原材料 loss、current/field loss、再加 material-update-aware loss；每次只增加一种干预。所有新训练均需下一实验另行授权和冻结，不构成本报告自动启动的训练矩阵。

使用 TRAIN-only 尺度：\(s_h=.169475875, s_c=.00280370125, s_e=.00368412791, s_w=.000580129418\)。基础辅助项为 \(\|\hat c-c^*\|^2/s_c^2\) 和 \(\|\hat e-e^*\|^2/s_e^2\)。后者是已有电流标签的确定变换，改变误差权重而不增加独立标签信息。

面向材料更新的项应约束八源共享同一材料：

\[
\tilde\chi_i={\sum_s\overline{\hat e_{is}}\hat c_{is}\over\sum_s|\hat e_{is}|^2},\quad
L_{ratio}=\operatorname{mean}|\tilde\chi-\chi^*|^2/s_h^2,
\]
\[
L_{common}=\operatorname{mean}_{i,s}|\hat c_{is}-(\chi_{b,i}+\hat h_i)\hat e_{is}|^2/s_c^2.
\]

比值保留原 `1e-10` 相对分母安全检查；失败留账，不能加入隐藏 ridge、使用 teacher 分母或丢掉失败 parent。有限 loss 是软训练约束，不是物理证书。完整电流含有较大背景分量，必须同时报告相对源和材料误差，避免只学会背景而损失很小。

更小的备选是训练时的场加权材料误差 \(\operatorname{mean}|(\hat h-h^*)e^*|^2/s_w^2\)。它不需要辅助 head 或每更新 G action，但只能称为物理加权的材料监督，不能声称网络独立学会了电流。

## 成本与 SOM-Net 对照

256个 teacher 总场的一次准备需要2048个既有 G source-column actions，不需要新 full-wave labels。存储 e 或 w 各18 MiB，按512观测展开各36 MiB。batch4 的 current-head 方案每次 G 前向处理32源 RHS、约10,616,832个复数乘加，反向至少相同量级；head、标签搬运、ratio 与显存还需另测。这里给的是运算量，**未实测训练 latency**。若辅助分支被材料推理读取，则不能移除，也不能声称部署成本为零。

[SOM-Net 原论文](https://arxiv.org/abs/2209.03567) 的损失包含 induced current、receiver scattered-field 与材料 SSIM/MSE。其 scattered-field loss 不是内部 total-field loss；内部 e、相对 w 和 material-update-aware weighting 是需要独立验证的干预，不能包装成官方复现或首次物理监督。薄 shell/asymmetric 可能从跨源材料一致性获益，但目前没有实验支持这个效果；也不能据此预设 N2 会优于 BP+CNN。
