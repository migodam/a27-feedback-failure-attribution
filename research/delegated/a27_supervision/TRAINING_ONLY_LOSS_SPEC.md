# A27｜仅用于训练的current/field与材料更新loss草案

状态：`SPEC_ONLY / EFFECT_NOT_RUN`。这是给parent Codex审阅的独立A27训练干预，不修改冻结A26训练、rank、solver、encoder或部署图。本worker未实现网络、训练或打开locked。数据可用性已由 `SUPERVISION_AUDIT.md` 核验，符号与units完全沿用那里。

## 要回答的问题

现有BP/L/R材料CNN只被要求输出正确的材料图。新增监督可检验：让训练表示同时解释8源电流与内部总场，并明确面向共同材料更新，是否在相同推理流程下改善材料结果。shell改善、反馈壳层表示更好、训练样本效率更高都只是待否证假设；已有A26结果不支持这些结论。[E12–E18]

## 一个可移除的训练分支

保留A26冻结encoder→同一31,810参数材料CNN→同一box的部署主分支。训练时从同一个主干表示z另接source-shared辅助current head，得到 `c_hat[B,n,s]`；其额外输入只能是在线已有的源标识/坐标与已知背景的incident/eb等，不允许chi_true/current_true/e_true/w_true作为网络输入。源共享权重、固定声明的source顺序；若源数组被置换，辅助输出与相关元数据必须相同置换。单纯由source-mean特征猜源标签的固定槽位head只能用于冻结源几何，不能宣称源泛化。

辅助总场由固定物理作用定义：

\[
 \widehat e=e^{inc}+G\widehat c.
\]

因此无需另造可自由违背场更新的field head，也没有新的非线性forward solve。它仍产生可训练的current+field监督。整个auxiliary current head、G作用、ratio及以下辅助loss仅在训练图中存在；材料输出没有读取这些head输出，inference时可以全部删除。删掉后须验证同一权重/输入的材料输出逐元素一致，并报告部署参数与调用数；未运行该验证前，removability只属架构设计性质。[E14,E17]

如果后来把aux current/field/ratio输出送回材料更新或encoder，就成为新的推理架构，不能继续称免费或可移除。SOM-Net原N0的current是材料推理所需中间输出，不适用这一移除结论。[E16]

## 固定单位、标签与loss

TRAIN教师为c*、e*=incident+Gc*、h*=chi*-chi_b，必要时w*=h*e*。只准从metadata-first验证的64个TRAIN parent读取；所有归一化只来自TRAIN且所有arm共享。建议直接冻结本审计得到的尺度 `s_h=.16947587474138345`、`s_c=.00280370125084841`、`s_e=.003684127912995651`、`s_w=.0005801294178652024`，以完整mean而非只对非零支撑取mean。引用256 unique states或其等权512 noise rows均可，但声明清楚，不重复计独立训练样本。[E01,E09]

保持既有主loss

\[
 L_0=\operatorname{mean}_i|\widehat h_i-h_i^*|^2/s_h^2
 +.05(1-\operatorname{SSIM}_{\Delta}(\widehat h,h^*)).
\]

对所有体素与8源定义

\[
 L_c=\operatorname{mean}_{i,s}|\widehat c_{is}-c^*_{is}|^2/s_c^2,\quad
 L_e=\operatorname{mean}_{i,s}|\widehat e_{is}-e^*_{is}|^2/s_e^2.
\]

e*是c*经过固定G的确定变换，不是新的独立标签信息；添加L_e改变误差的权重和梯度，不能把它计成另一批独立full-forward labels。full-current RMS中包含已知背景，宜另报背景/材料增量误差，不把小的总电流相对误差自动解释成材料正确。[E07,E09]

## 面向共同材料更新

同一体素全部source必须对应同一个复scalar材料。采用现有代码的等源权重与无ridge LS比值：

\[
 D_i=\sum_{s=1}^{8}|\widehat e_{is}|^2,\quad
 \widetilde\chi_i=\frac{\sum_s\overline{\widehat e_{is}}\widehat c_{is}}{D_i}.
\]

对每个predicted sample执行与既有ratio相同的安全条件 `D_i > 1e-10*max_j D_j` 且finite；失败留在failure ledger并停止该声明的训练尝试，不能丢掉parent、偷偷clamp分母、用teacher分母替代或调整floor直到成功。若父线程选择ridge则须先注册eta与bias解释，所有arms一致；那是另一项loss设计。[E08,E17]

材料更新loss和主分支的跨源共同材料loss为

\[
 L_{ratio}=\operatorname{mean}_i|\widetilde\chi_i-\chi_i^*|^2/s_h^2,
\]
\[
 L_{common}=\operatorname{mean}_{i,s}|\widehat c_{is}
 -(\chi_{b,i}+\widehat h_i)\widehat e_{is}|^2/s_c^2.
\]

L_ratio直接惩罚解析material update的结果，而非仅惩罚全局current/field MSE。L_common把辅助状态连接到部署主分支的唯一材料，禁止每个source各拟合一个不同chi。单个ratio可计算不代表多源电流都满足同一constitutive law；必须同时报告跨源residual，不能称有限loss是hard physics certificate。parent保留对这一训练干预的数学审阅权。[E08,E17]

可另注册相对材料监督：

\[
 \widehat w=L_b(\widehat c-c_b),\qquad
 L_w=\operatorname{mean}_{i,s}|\widehat w_{is}-w^*_{is}|^2/s_w^2.
\]

此项去除已知背景并强调相对contrast source，但会增加L_b作用及其反向。它不是换basis、rank或solver，也不自动等同thin-shell edge loss。若选独立w-head后再lift到current，则增加每批所有source的background solve，必须明确计时；本草案主候选使用current-head以避免该solve。[E06,E07]

## 最小等权限对照

| 组别 | 训练目标 | 必要比较 |
|---|---|---|
| S0 | 原L0 | 复现原材料信息基准；辅助效果未成立 |
| S1 | L0+.1L_c+.1L_e | N1/N2/N3均加相同head/信息权限/权重，隔离current/field监督 |
| S2 | S1+.1L_ratio+.1L_common | 同三个arms，隔离material-update-aware约束 |
| S3，只有新预算明确注册才允许 | S2+.1L_w | 同三个arms，隔离relative-source weighting |

`.1`是待冻结的设计候选，沿用A25 current权重的量级；field/ratio/common/w权重没有经验最优性证据。主候选应由parent在训练结果前选择，不能偷偷通过已暴露shell结果调权，也不能把S0/S1/S2/S3称为已经获准的训练矩阵。若完整matched矩阵不符合新预算，只能缩小并重注册问题，不能先训练N2再决定哪些controls值得训练。

与generic CNN/BP control比较时，各组必须拥有相同TRAIN current、field、relative-source和材料信息，及同一个source-shared辅助head、主干、width/depth、optimizer、actual updates、采样序列、normalizer、box与exposed-validation checkpoint规则。N1/N2/N3各组内训练参数数目一致；和S0相比新增的训练参数必须单列。单独给N2多监督、增加训练updates或额外auxiliary容量，不允许解释成feedback机制收益。新监督需要新的A27 trainer-only读取接口；不要修改原A26 load_split，让它事后看似曾使用current。[E10,E12,E14,E17]

保留原材料validation作为主checkpoint选择；current/field/ratio诊断若被用于选模型则是额外选择权限，必须预先注册并同给所有arm。locked仍闭合，任何TRAIN-only结果不构成独立holdout或任意复相位材料证明。[E01,E18]

## 无head的较小训练干预备选

只使用teacher场做材料增量transport weighting：

\[
 L_{transport}=\operatorname{mean}_{i,s}|(\widehat h_i-h_i^*)e^*_{is}|^2/s_w^2.
\]

它无需aux head或每step G action，部署完全相同；但只是有物理权重的材料loss，不能说网络学会独立电流或总场。它依赖的TRAIN标签权限仍须给予所有controls，原L0保留以避免低场区域被忽略。若实验预算不足以支持current-head matched matrix，可以把这个问题单独注册；不能与S2结果混为同一干预。

## 必须计费的新增训练工作

一次构造教师e*需对256个唯一状态作用G，共2048 source columns；w*由点乘he或L_b action恢复。可在trainer专属缓存中保存，但其数据来历、row alignment、bytes与成本要登记。本审计没有保存这些教师数组；将来保存只是确定的既有标签转换，不能计成新独立样本或零成本。

current-head方案每批B=4：forward G处理32个source columns，dense乘加数为 `B*n*n*s=10,616,832` 个complex multiply-accumulate，反向至少同量G-adjoint。S3再加相同规模的L_b forward/adjoint。若加receiver数据loss，S forward为589,824个complex multiply-accumulate并有相应adjoint；它不在最小S1/S2中。全forward非线性solve、rank/basis rebuild与部署动作都为0；这一计数只是数学工作量，不是实测latency。

每个[B=4,n=576,s=8] complex128 current/field/w张量是294,912字节（288 KiB），反向及head activations另算。保存256-state e或w各需18 MiB；按512观测展开各36 MiB。head本身的forward/backward、target I/O/transfer、guards、ratio/common reductions、optimizer与训练峰值显存均需inclusive实测，训练成本不得从部署latency中消失。[E03,E06,E14]

最终需报告材料h误差、full nonlinear receiver residual、current/field/w误差、跨源共同材料residual、ratio failures、各family paired结果、inclusive训练成本和删头后的完整部署latency。shell收益、N2对controls增益、速度/质量优势、样本效率一律等待完整相应实验；当前所有新loss效果为 `NOT_RUN`。
