# A27 计费检查对 CPU prox 的干扰与修复

第一次 `a27-post-decoder-001` 原样保留为 INTERRUPTED，GPU428.0907372秒、hostCPU372.8125秒；终止前完整进度48个N3观测、192行、144个修正失败。失败动作原始ledger保留，gap约2e-10到4e-10、可行性0，但KKT约2e-5到3e-5，约800次迭代触及原2秒上限。这些失败不构成feedback无效的实验裁决。

原因：CPU proximal 内每次 `_check_book` 都执行 shared lock 文件检查；Windows I/O约占迭代成本，使同一个凸算法不能在冻结wall cap内达到原精度。第二次尝试只新增 CPUProxMeter：outer span/solver entry-exit立即检查预算与owner，CPU迭代每50ms取一次真实昂贵检查；原算法仍每次迭代执行自身数值操作、周期KKT/gap认证与wall clock检查。所有动作计数直接写到父job，完整job/CPU/GPUwall全部收费。

没有改变目标、TV、步长规则、material box、KKT1e-6、可行性1e-10、20,000iterations或两秒上限。GPU物理core/backward guards不变。这是计费工程修复，不是根据truth改变科学protocol。

另保留 local-tests-002 的普通fixture import失败和在local-tests运行期间push被accounting lock拒绝的操作失败；无物理启动、无lock删除或代填成功结果。
