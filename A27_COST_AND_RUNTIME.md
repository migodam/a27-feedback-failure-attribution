# A27成本与失败边界

当前发布准备cutoff `science-and-preparation-complete-before-publication`：CPU **2571.745s**，GPU占用 **2539.746s**，各硬上限3600s；terminal receipts 66，未知未结算作业 []，此cutoff已知活动作业 `a27-publication-001`。发布过程另有terminal receipt，不能把当前snapshot称为发布结束的完整总数。最终网页sidecar/本地账本保存后续cutoff。

CPU作业按wall/process较大者；CUDA作业host process CPU另计，GPU按整个占用wall，包含CPU prox、setup、审核、失败和重跑。部分transport CPU等待与remote GPU占用重叠，属于保守资源记账，不能把二者相加叫物理算法CPU时间。人工/模型推理与源码编辑时间不伪装为本机科学计算；可测的离线标签审计、表图fixtures、报告/打包/发布单独入账。失败wrapper的1秒是显式上界收费，不是性能样本。

| 主要执行 | 状态 | device | wall s | CPU charge s | GPU occupied s |
| --- | --- | --- | --- | --- | --- |
| a27-backend-audit-001 | COMPLETE | cuda | 15.372 | 7.203 | 15.372 |
| a27-latency-001 | COMPLETE | cuda | 103.879 | 79.469 | 103.879 |
| a27-local-report-001 | COMPLETE | cpu | 7.283 | 7.283 | 0.000 |
| a27-local-report-002 | COMPLETE | cpu | 7.258 | 7.258 | 0.000 |
| a27-observability-001 | COMPLETE | cuda | 24.777 | 6.109 | 24.777 |
| a27-post-decoder-002 | COMPLETE | cuda | 1449.014 | 1267.938 | 1449.014 |
| a27-post-fgp-001 | COMPLETE | cuda | 278.179 | 212.250 | 278.179 |
| a27-prox-health-001 | COMPLETE | cuda | 10.721 | 8.500 | 10.721 |
| a27-state-diagnostics-001 | COMPLETE | cuda | 219.948 | 154.641 | 219.948 |

最终修正阶段 `post_fgp` 1024/1024有效。旧accelerated-PDHG的387个case失败、早期计费开销中断、测试import/平台路径问题和一次Windows进程启动退出均保留；进程退出原因未确定，不能当成物理失败或成功。新FGP、完整backend和安全core没有静默fallback。数据下降/材料变差是有效的科学负结果，保留在主表，另列于UPDATE_AND_FAILURE_COUNTS。

逐job动作在receipt和raw ledger中。`linear_solve_rhs`包含具体role RHS，`LU_factorizations`包含full/background/core角色；它们相互重叠，不能全部相加成“总Maxwell向量数”。receiver/G/Db动作、full RHS/core RHS和LU分别报告；offline full-J/SVD、train-label audit和online correction也分列。已存cache是历史资产，当前读取/setup/刷新/warmup均付费。

resident warm原生计时覆盖raw observation→encoder→CNN→box→host image。post计时包含prox、line search和所有full-F接受审核，以及修正内部两次计时fence。offline truth指标、谱审计另列，计入campaign资源但不是默认部署操作。没有新增进程冷启动/disk-cold/吞吐性能claim。

[逐job费用](A27_COST_LEDGER.jsonl) · [失败](A27_FAILURE_LEDGER.jsonl) · [修正actions](results/a27/artifacts_post_fgp/tables/CORRECTION_ACTION_COUNTS.csv) · [native latency](A27_LATENCY_ATTRIBUTION.md)。所有new labels=0、training=NOT_RUN、locked=UNOPENED。
