# 原生延迟与无损工程归因

删除仅用于beta诊断的L前向可减少工作；同state复用LU在这套安全实现里减少一次分解，却增加总时间。算子数与实际单图延迟必须分别裁决。

## 冻结基准与本次匹配测量

A26历史warm为N1 40.8274ms、N2 126.084ms、N3 38.5077ms，原结果和NO_GO不变。本次另测同机resident warm：每臂元数据固定16条件×3次，轮换顺序、清空每图材料/递归state，不复用最终图像。不同会话的GPU/框架状态和计时波动不能据此改写历史126ms。以下优化只能与本次未改动的original匹配比较。

| 实际原生计时臂 | median ms | 样本数 |
| --- | --- | --- |
| ORIGINAL_N1 | 45.445 | 48 |
| ORIGINAL_N2 | 99.836 | 48 |
| ORIGINAL_N3 | 40.642 | 48 |
| N2_NO_BETA_ONLY | 86.491 | 48 |
| N2_LU_REUSE_ONLY | 114.493 | 48 |
| N2_COMBINED | 97.693 | 48 |
| POST_NONE | 38.928 | 48 |
| POST_L | 391.038 | 48 |
| POST_F | 259.714 | 48 |
| POST_TASK_R32 | 251.029 | 48 |

beta-only相对本次original N2降低13.37%，仍约2.13×N3。LU-only慢14.68%；combined只快2.15%，且慢于beta-only。没有建立20%同质量端到端提速。N3→F修正约260ms、NONE约39ms，R修正约251ms也包含付费完整F接受审核；不能称为纯R部署成本。

## 数值与动作配对

96个repeat0等价receipt通过；所有features/initializer完全一致，LU变体base最大绝对差3.47e-18、decoder最大相对差4.10e-17。288个native variant的等价flag由这些96个receipt传播，不是288次独立比较。beta被删除时明确返回不可用，不伪装成相同beta。

| N2变体 | L+R forward源RHS | Db forward actions | core LU | core solve RHS | 新增core backward audits |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 16 | 16 | 2 | 16 | 0 |
| beta-only | 8 | 8 | 2 | 16 | 0 |
| LU-only | 16 | 16 | 1 | 16 | 2 |
| combined | 8 | 8 | 1 | 16 | 2 |

同state forward/adjoint可用同LU；owner、材料、core和背景失效检查仍在，不能跨图借LU。每次core guard和原forward残差检查保留；reuse还保存值/版本快照并核验core solve backward residual，不加jitter/pinv/fallback。

## 分阶段profile及同步限制

另有96个CUDA事件/dispatch观察调用，记录background adjoint、L/R forward、core assembly、solve/独立LU、feedback adjoint、源汇集、material ratio、tensor搬运、CNN、guard和logging。native432行没有stage观察器；raw-to-corrected192行包含修正内部两次计时同步和外层fences。分阶段profile不能替代真实native计时。

instrumented N2 core assembly约2ms；baseline合并LU/solve约7.81ms host区间，单独LU时长不可用；source aggregation约1.6–1.7ms、material ratio含guard约6–6.5ms。普通N2有41个可见host结果/拷贝机会，beta-only32、LU-only55、combined46。实际wait秒数未测，CUDA事件也包含host引发的stream空闲，不是纯kernel时间。

安全检查、微小矩阵launch与host决策是可信的成本嫌疑，但不能给出“launch占总成本多少”的证据。LU-only的新增安全/快照开销与变慢相伴，不能将其全部归因为同步等待。stage区间嵌套重叠；N2 profiler额外耗时不等于book-meter的8.81ms，不能相减得到虚构kernel时间。完整明细见[实际profile审阅](research/delegated/a27_latency/ACTUAL_OUTPUT_REVIEW.md)。

**删除建议：**从主推理去掉不作为特征的beta-only L诊断，保留独立开发诊断。**保留建议：**数值安全、跨源共同材料与状态失效约束。**暂不采用：**当前LU复用实现作为加速卖点；它只有分解数减少证据。没有关闭安全检查或继续优化rank。

[原生CSV](results/a27/latency/NATIVE_INFERENCE.csv) · [修正全流程CSV](results/a27/latency/POST_RAW_TO_IMAGE.csv) · [质量—时间图](results/a27/artifacts_post_fgp/figures/latency_quality.png)。背景setup、checkpoint载入、首次observable SVD和paid warmups计入实验成本，未进入resident每图窗口；没有新增disk-cold/进程冷启动性能证据。
