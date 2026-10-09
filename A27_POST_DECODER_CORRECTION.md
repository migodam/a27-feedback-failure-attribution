# CNN 后的一次物理修正

完整 cohort 的完整 Maxwell F 能改善材料，但幅度较小。主网络 N3 的复杂家族 mean-parent 收益为 **1.137% [0.923, 1.351]**；F 相对同条件 L 的增量为 **0.160% [0.073, 0.240]**。数据残差收益为 **25.270% [23.657, 26.743]**。这些是先平均每 parent 八条件后的配对收益，不是两个总体平均数之比；所有16 parents均保留。

## 相同起点与在线规则

128个条件：16个已暴露 parents，四家族各四个；两已知背景、两幅度、clean/20dB。N3 BP+CNN是主比较，N1是辅助。冻结 A26 best checkpoint，T0 重现全部预测的相对误差为0。NONE/L/F/TASK_R32均从同一 CNN 材料出发。在线函数只接受预测材料、测量、背景和噪声，不接受 truth、真值J或oracle步。

实质量坐标为 `x=sqrt(V)[Reχ,Imχ]`，`V=1/576`。完整在线目标为

\[
\Phi(x)=\tfrac{1}{2}\|F(x)-d\|^2/\|y_b\|^2+\lambda\,TV(\chi-\chi_b),
\quad\lambda=.001(1+\sigma/\mathrm{RMS}(y_b)).
\]

各臂以自身J和数据梯度确定方向曲率及初始步，按冻结规则作TV/box prox，mass信赖半径0.15、最多8次halving。完整F目标Armijo和完整数据不增共同决定接受；真值只在之后评分。此处是一次 gradient/prox更新，不是GN、重新非线性重建或最优反馈收益上界。L/R也支付完整F接受审核，不能据此声称廉价独立部署。

## 数值修复与失败

原 accelerated-PDHG 的完整尝试1024行中有387行数值失败，另一个早期因逐迭代文件计费开销被中断的尝试也保留。小gap不替代KKT。数值修订先在固定首个ring parent的12个无truth问题核验FGP，12/12认证后才统一使用同一FGP重跑。目标、TV、box、接受规则、KKT/gap1e-6、feasibility1e-10和20,000次/2秒上限均未改变。最终1024/1024行有效、零prox/core失败、材料全部可行被动。原失败不被覆盖，不作success-only统计。

一个同输入PDHG控制在2秒后仍有primal stationarity4.91e-6，dual fixed-point1.19e-9，gap1.65e-11；此处是primal收敛瓶颈，不能笼统称为A24历史的dual残差问题。FGP健康问题270–410次、约0.26–0.41秒。健康JSON的通用`sigma`字段被内部dual步长覆盖；它不是观测噪声sigma，实际目标使用原数据sigma，主CSV噪声字段正确。

## 主网络结果

| 起点/修正 | Δχ NRMSE | full χ NRMSE | full data residual | IoU | edge F1 | SSIM | 接受/128 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NONE | 0.603592 | 0.223795 | 0.157983 | 0.628517 | 0.600204 | 0.803954 | 0 |
| L | 0.592221 | 0.222379 | 0.131242 | 0.626688 | 0.593156 | 0.816776 | 125 |
| F | 0.589772 | 0.221414 | 0.115357 | 0.627681 | 0.593019 | 0.833434 | 128 |
| TASK_R32 | 0.590001 | 0.221532 | 0.116835 | 0.628441 | 0.593733 | 0.833617 | 128 |

| 家族 | F vs NONE 材料收益/95%区间 | F vs L 材料收益/95%区间 |
| --- | --- | --- |
| shell | 0.995% [0.888, 1.103] | 0.064% [-0.125, 0.253] |
| asymmetric_piecewise | 1.378% [1.060, 1.633] | 0.199% [0.140, 0.252] |
| smooth | 8.090% [5.266, 10.913] | 1.838% [0.943, 2.663] |
| mixed_scale | 1.039% [0.479, 1.531] | 0.217% [0.139, 0.361] |
| 复杂三家族 | 1.137% [0.923, 1.351] | 0.160% [0.073, 0.240] |
| 全部四家族 | 2.875% [1.425, 4.735] | 0.580% [0.211, 1.028] |

完整数据下降但材料变差的F条件有 **6/128**，没有被删除。N3的平均IoU/edge未因F取得结构性跃升；SSIM改善不能单独等同于薄边界恢复。完整逐条件、parent和全部指标的bootstrap见[原始表](results/a27/artifacts_post_fgp/tables/PER_PARENT.csv)、[配对统计](results/a27/artifacts_post_fgp/PAIRED_STATISTICS.json)。

## N1辅助对照

| N1修正 | Δχ NRMSE | full data residual | 接受/128 |
| --- | --- | --- | --- |
| NONE | 0.600837 | 0.158355 | 0 |
| L | 0.589222 | 0.133425 | 124 |
| F | 0.585935 | 0.111448 | 128 |
| TASK_R32 | 0.586155 | 0.111512 | 128 |

N1支持相同的“数据收益大于材料收益”方向，不能用辅助网络替代预注册N3结论。[实际图像](results/a27/artifacts_post_fgp/figures/n3_fixed_delta_real.png)、[虚部图](results/a27/artifacts_post_fgp/figures/n3_fixed_delta_imag.png)是元数据固定的首parent/高幅度/20dB，不按图像质量挑例子。

actual修正向量、步范数分母、L/F与R/F差异、trust/backtracking、证书和actions详见[修正审计](results/a27/artifacts_post_fgp/CORRECTION_AUDIT.json)。已有的box-proxy诊断使用固定背景步长，不等于实际自适应TV更新。预prox向量未保存，只能报告其标量梯度/曲率/步长，不能伪造方向比较。

实际N3复杂家族的R/F修正向量相对差为10.028%，cos平均0.990630；L/F相对差为64.039%。它们是从相同NONE到实际返回材料的mass位移比较，不能当成GN-step error。L拒绝步产生零向量，使完整L/F cosine汇总为HOLD，而不是删掉零步算“成功cos”。记录的1076次试探中没有TRUST_RADIUS拒绝；最终F接受128/128，所以小材料收益不是由数值失败或信赖半径筛掉有利F样本造成。
