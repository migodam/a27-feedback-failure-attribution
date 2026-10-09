# A27｜现有监督量审计

现有 A25 TRAIN 数据足够提供完整诱导电流监督，并可恢复按源排列的总场与相对对比源。A26 的实际学习并没有使用这些电流标签。增加电流/场监督可以设计为独立的训练干预；目前没有实验说明这种干预会改善 shell 或产生反馈增益。

本审计使用冻结 A26 source `2603f3bd16c48e48587fedc4363975779def17f5`，只读取 TRAIN、已知背景、TRAIN 特征缓存、权重与文字记录。`TRAIN_ARRAY_AUDIT.json` 为逐项标量结果，`TRAIN_STATE_SUMMARY.json` 为256个既有训练状态的标量摘要，`EVIDENCE_TABLE.md` 给出路径与行号。没有读取 validation/locked 数组，没有重算 full-forward labels，没有线性求解或训练。[E01–E18]

## 实际保存内容

| 量 | 实际保存 | 实际布局 / 数量 | A26 训练实际采用 |
|---|---|---|---|
| 完整材料 chi | 是 | complex128 [512,576] | 是，使用 h=chi-chi_b |
| 完整诱导电流 c | 是，在 A25 TRAIN | complex128 [512,576,8]；36 MiB未压缩 | 否 |
| clean 接收散射数据 | 是，在 A25 TRAIN | complex128 [512,32,8] | 否；A26只用观测形成输入 |
| 总场 e | 否 | 可从 c 和已知 G/incident 恢复 [512,576,8] | 否；特征仅含估计场的源均值功率 |
| 相对对比源 w | 否 | 可从 c、chi与已知背景恢复 [512,576,8] | 否；特征仅含估计 w 的源均值功率 |

512条观测属于64个独立parent：每个parent有2背景×2材料幅度×2噪声条件。独立clean状态为256个，每个状态保留8源完整列；共有1,179,648个独立状态中的复电流坐标。两种噪声条件的 chi/current/clean 标签逐元素完全相同。256个原始TRAIN label分片的文件名与这256个状态相符。[E03,E04,E09]

A27 checkout没有 `data/`；其默认数据路径指向 A25 sibling。A26 sibling保留了三组TRAIN缓存与九个checkpoint，而它的 `results/a26/repro_inputs/data/train.npz` 只带材料/观测/元数据，不带电流、场或w。故单独下载A26复现实验输入并不足以复现新的current/field supervision。缺少A25 current数组时，应明确标为 `DATA_NOT_AVAILABLE`；不能由chi另跑forward造标签来偷偷补齐。[E12–E14]

## 单位与精确恢复

这里 n=576、s=8、V=1/576、sqrt(V)=1/24，最后一维始终是source。A24把vendor场乘sqrt(V)，把receiver矩阵除sqrt(V)：

\[
 e^{N}=\sqrt V E^{vendor},\qquad c^{N}=\chi e^{N},\qquad
 S^{N}=S^{vendor}/\sqrt V.
\]

下文省略N上标。现有材料mass是 `sqrt(V)*[Re chi, Im chi]`。这些是代码的标量TM电流单位；原SOM-Net用含 `-i k0/eta0` 的物理对比系数，不能直接把本数组当作其SI电流。[E05,E16]

从已保存标签与已知背景可计算

\[
 e_s=e^{inc}_s+Gc_s,\quad h=\chi-\chi_b,\quad
 c_{b,s}=\chi_b e_{b,s},\quad L_b=I-\operatorname{diag}(\chi_b)G,
\]
\[
 w_s=h e_s=L_b(c_s-c_{b,s}),\qquad y_s=y_{b,s}+D_bw_s=Sc_s.
\]

w是材料增量驱动的相对对比源；它不是完整电流c，也不是电流差c-c_b。`R_b=L_b^{-1}` 的理论恢复为 `c=c_b+R_b w`、`e=e_b+G R_b w`。现有缓存保存L_b而非对象的full inverse；本审计只用矩阵作用核对恒等式，未求R_b或执行inverse。[E06,E07]

| 全部256个TRAIN状态的检查 | 最大相对误差 |
|---|---:|
| c与chi e | 1.83245e-15 |
| h e与L_b(c-c_b) | 7.13161e-14 |
| S c与已保存clean | 0 |
| y_b+D_b w与已保存clean | 1.22567e-15 |
| 所有8源共同ratio与chi | 9.37161e-16 |

这些数字验证了现有TRAIN标签的内容、尺度和恢复定义；它们不是网络成像或科学泛化结果。weak/ring分别128个clean状态；w恒等式较大的ring相对误差仍为浮点舍入量级，其分母是w范数。[E07,E08]

以所有TRAIN状态/体素/source的 `sqrt(mean(abs(.)**2))` 固定尺度：

| 标签量 | TRAIN RMS |
|---|---:|
| h | 0.16947587474138345 |
| c | 0.00280370125084841 |
| e | 0.003684127912995651 |
| w | 0.0005801294178652024 |

两种噪声重复完全一致，按512条观测计算同一尺度也相同。h尺度与三个已保存A26 checkpoint的material_scale一致；三个checkpoint各有31,810个参数，最佳步为N1 1300/N2 1100/N3 1400，targets_in_checkpoint=false。[E09,E14]

shell的64个clean状态属于16个parent。其 `min_i sum_s|e_is|² / max_i sum_s|e_is|²` 最小0.215913，均值0.346549。TRAIN真值总场没有触发现有1e-10相对floor。shell的平均非零h支撑占7.6823%（以|h|>1e-12计），这是已有TRAIN标签的描述；不能据此宣称场损失会修复薄壁，也不能据此保证网络预测的ratio安全。[E08,E09]

## 与现有训练和原论文的关系

A25代码的matched_loss允许material/data/current/SSIM/TV，并对material/current使用TRAIN RMS；但完整科学训练被预算门槛阻止，模型/训练更新为0，不能把health loss当作辅助监督有效性证据。A26三模型各2000次材料训练更新，loss为 `MSE(h)/scale_h² + .05*(1-SSIM(h))`，physical_backward_calls=0。[E10–E14]

原SOM-Net arXiv v2文本已在附近本地取得：输出完整电流，再解析更新接收散射场和共同材料；Eq7是current+receiver-scattered-field+SSIM+material-MSE，lambda1=lambda2=2。原论文的场loss是接收散射场，不是内部总场e。拟议的内部场监督是A27的新训练干预，不能称逐项恢复了原loss或已知有效。[E16]

源：[原SOM-Net](https://arxiv.org/abs/2209.03567)，[冻结A26报告](https://github.com/migodam/a26-om-feedback-prior-completion/blob/2603f3bd16c48e48587fedc4363975779def17f5/A26_RESEARCH_REPORT_ZH.md)。这些链接指向既有来源；本A27审计文件仍为本地草案，未发布。本worker未执行新的匿名网页可达性核验。

最终物理/数学解释与A27是否允许训练由parent Codex负责；本文件提供的数据可用性及内容核验已完成。loss设计见 `TRAINING_ONLY_LOSS_SPEC.md`，其效果统一为 `NOT_RUN`。
