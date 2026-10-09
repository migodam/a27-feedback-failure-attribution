"""Render parent-reviewed A27 conclusions from saved evidence; no physics.

Run only after the complete artifact stage. The interpretation thresholds are
the original config, and every computed gain retains its parent denominator.
"""
from __future__ import annotations
import time
START_WALL, START_CPU = time.perf_counter(), time.process_time()
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from a27.budget import charged_job, write_json, budget_history


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main():
    config = json.loads((ROOT/'configs/a27/pilot.json').read_text())
    with charged_job(ROOT, sys.argv[1], config=config, metadata={'stage':'scientific-report-render'}) as book:
        book.deadline -= book.started_wall-START_WALL
        book.started_wall, book.started_cpu = START_WALL, START_CPU
        base=ROOT/'results/a27/artifacts_post_fgp'
        summary=json.loads((base/'SUMMARY.json').read_text())
        if summary['status']!='COMPLETE':
            raise ValueError('INCOMPLETE_PRIMARY_EVIDENCE; no complete-cohort conclusion')
        paired=json.loads((base/'PAIRED_STATISTICS.json').read_text())
        preview=json.loads((ROOT/'results/a27/PARENT_EVIDENCE_PREVIEW.json').read_text())
        family=csv_rows(base/'tables/PER_FAMILY.csv')
        parent=csv_rows(base/'tables/PER_PARENT.csv')
        counts=csv_rows(base/'tables/UPDATE_AND_FAILURE_COUNTS.csv')
        diagnostics=csv_rows(ROOT/'results/a27/diagnostics/PER_STATE.csv')
        obs=csv_rows(ROOT/'results/a27/observability/PER_STATE.csv')
        comparisons=csv_rows(base/'tables/CORRECTION_COMPARISON_PER_FAMILY.csv')
        def update_stat(candidate,metric):
            return next(r for r in comparisons if r['network']=='N3' and r['candidate']==candidate and
                        r['reference']=='F' and r['group']=='COMPLEX_FAMILIES' and r['metric']==metric)
        r_update=update_stat('TASK_R32','relative_update_difference')
        l_update=update_stat('L','relative_update_difference')
        r_cos=update_stat('TASK_R32','actual_update_cosine')
        bg_rank_range=f"{min(int(r['background_strong_rank']) for r in obs)}–{max(int(r['background_strong_rank']) for r in obs)}"
        predicted_rank_range=f"{min(int(r['predicted_strong_rank']) for r in obs)}–{max(int(r['predicted_strong_rank']) for r in obs)}"
        protocol='results/a27/artifacts_post_fgp/'
        figure=protocol+'figures/'
        def fmt(v):return f'{float(v):.6g}'
        def pair(net,candidate,reference,group='COMPLEX_FAMILIES',metric='delta_nrmse'):
            return paired['networks'][net][candidate+'_vs_'+reference][group][metric]
        def gain(s):
            if not s['coverage_complete']:return 'HOLD'
            return f"{100*s['point_estimate']:.3f}% [{100*s['ci95'][0]:.3f}, {100*s['ci95'][1]:.3f}]"
        def fr(net,arm,group='ALL_FAMILIES'):
            return next(x for x in family if x['network']==net and x['correction']==arm and x['group']==group)
        def co(net,arm):return next(x for x in counts if x['network']==net and x['correction']==arm)
        def table(headers,rows):
            return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
        def emit(name,text):
            (ROOT/name).write_text(text.strip()+'\n',encoding='utf-8');book.check()
        a=pair('N3','F','NONE');fl=pair('N3','F','L')
        families=('shell','asymmetric_piecewise','smooth','mixed_scale')
        complex_families=('shell','asymmetric_piecewise','mixed_scale')
        positive=sum(pair('N3','F','NONE',f)['point_estimate']>0 for f in complex_families)
        case_a=(a['point_estimate']>=config['case_A']['F_vs_NONE_complex_gain_min'] and
                fl['point_estimate']>=config['case_A']['F_vs_L_complex_gain_min'] and
                a['ci95'][0]>0 and fl['ci95'][0]>0 and positive>=2)
        case_c=abs(fl['point_estimate'])<=config['case_C']['absolute_F_vs_L_complex_gain_max'] and fl['ci95'][0]<=0<=fl['ci95'][1]
        data=pair('N3','F','NONE',metric='full_data_relative_residual')
        interpretation=('CASE_A' if case_a else 'CASE_C' if case_c else 'CASE_B_DATA_DOMINANT_WITH_SMALL_MATERIAL_GAIN')
        # This names a limited exploratory phenotype; it never means zero
        # material effect, an identification theorem, or a formal blind PASS.
        gate={'source_commit':config['source_commit'],'scope':'EXPLORATORY_EXPOSED_DEVELOPMENT',
              'T0':'PASS','original_prox_attempt':'387/1024 rows numerically invalid; retained',
              'common_numerical_amendment':'FGP; same objective/certificate/caps; health12/12',
              'post_fgp':'1024/1024 valid;128 cases;16 parents;N3 primary,N1 auxiliary',
              'interpretation':interpretation,'Case_A_screen_passed':case_a,'Case_C_screen_passed':case_c,
              'Case_B_qualification':'Full F has small positive material gain; primary benefit is data fitting. Not literally zero material improvement.',
              'N3_complex_F_vs_NONE':a,'N3_complex_F_vs_L':fl,'N3_complex_full_data_gain':data,
              'feedback_location_hypothesis':'BP signal below noise; CNN-state signal stronger; exact post correction insufficient for large shape gain',
              'R32':'Initial states accurate; CNN/post-state gradients not uniformly certified. Actual one-step images close to F under paid full-F acceptance.',
              'latency_engineering':'Beta-only omission verified; LU-only slower with safety retained; no end-to-end acceleration gate',
              'prior_and_observability':'OFFLINE32 states/4 parents; not a posterior uncertainty or unique-recovery certificate',
              'current_supervision':'LABEL_FEASIBILITY_VERIFIED; TRAINING_SPEC_ONLY',
              'training':'NOT_RUN','new_labels':0,'locked_ID_OOD':'UNOPENED',
              'rank_search':'NOT_RUN','nonlinear_reconstruction':'NOT_RUN','three_dimensional_claim':'NOT_SUPPORTED',
              'decision':'Do not expand frozen feedback encoder or rank tuning. Prior/weak-space and removable training supervision are conditional next experiments.',
              'publication_status':'separate receipt; scientific screen is not a publication gate'}
        write_json(ROOT/'A27_GATE_DECISION.json',gate)
        overall=table(['起点/修正','Δχ NRMSE','full χ NRMSE','full data residual','IoU','edge F1','SSIM','接受/128'],[
            [arm,*[fmt(fr('N3',arm)[k]) for k in ('delta_nrmse','full_nrmse','full_data_relative_residual','support_iou','edge_f1','ssim_delta')],co('N3',arm)['accepted_updates']]
            for arm in ('NONE','L','F','TASK_R32')])
        gains=table(['家族','F vs NONE 材料收益/95%区间','F vs L 材料收益/95%区间'],[
            [f,gain(pair('N3','F','NONE',f)),gain(pair('N3','F','L',f))] for f in families]+[
            ['复杂三家族',gain(a),gain(fl)],['全部四家族',gain(pair('N3','F','NONE','ALL_FAMILIES')),gain(pair('N3','F','L','ALL_FAMILIES'))]])
        auxiliary=table(['N1修正','Δχ NRMSE','full data residual','接受/128'],[
            [arm,fmt(fr('N1',arm)['delta_nrmse']),fmt(fr('N1',arm)['full_data_relative_residual']),co('N1',arm)['accepted_updates']]
            for arm in ('NONE','L','F','TASK_R32')])
        latent=preview['latency'];latency_table=table(['实际原生计时臂','median ms','样本数'],[
            [arm,f"{latent[arm]['median_ms']:.3f}",latent[arm]['n']] for arm in
            ('ORIGINAL_N1','ORIGINAL_N2','ORIGINAL_N3','N2_NO_BETA_ONLY','N2_LU_REUSE_ONLY','N2_COMBINED','POST_NONE','POST_L','POST_F','POST_TASK_R32')])
        signal_table=table(['状态（仅64个20dB条件）','F−L / 噪声 median','≥1的条件','R/F梯度cos min','R/F梯度相对误差 median','F/真值误差 alignment median'],[
            [s,fmt(preview['diagnostics'][s]['F_feedback_noise_ratio']['median']),str(preview['diagnostics'][s]['feedback_at_least_noise'])+'/64',
             fmt(preview['diagnostics'][s]['R_gradient_cosine_vs_F']['min']),fmt(preview['diagnostics'][s]['R_gradient_error_vs_F']['median']),fmt(preview['diagnostics'][s]['OFFLINE_F_descent_error_alignment']['median'])]
            for s in ('COMMON_BP','ONE_STEP_L','CNN_N3','CNN_N1','POST_N3_F')])
        obs_table=table(['网络','弱方向误差能量','强方向真值增量能量','q强','q弱','背景/预测强空间 overlap','梯度强方向能量'],[
            [n,*[fmt(preview['observability'][n][k]) for k in ('weak_error_fraction','truth_strong_fraction','q_error_strong','q_error_weak','background_predicted_strong_overlap','full_gradient_strong_energy_fraction')]] for n in ('N3','N1')])
        material_worse=co('N3','F')['data_decreases_material_worsens']
        emit('A27_POST_DECODER_CORRECTION.md',fr'''# CNN 后的一次物理修正

完整 cohort 的完整 Maxwell F 能改善材料，但幅度较小。主网络 N3 的复杂家族 mean-parent 收益为 **{gain(a)}**；F 相对同条件 L 的增量为 **{gain(fl)}**。数据残差收益为 **{gain(data)}**。这些是先平均每 parent 八条件后的配对收益，不是两个总体平均数之比；所有16 parents均保留。

## 相同起点与在线规则

128个条件：16个已暴露 parents，四家族各四个；两已知背景、两幅度、clean/20dB。N3 BP+CNN是主比较，N1是辅助。冻结 A26 best checkpoint，T0 重现全部预测的相对误差为0。NONE/L/F/TASK_R32均从同一 CNN 材料出发。在线函数只接受预测材料、测量、背景和噪声，不接受 truth、真值J或oracle步。

实质量坐标为 `x=sqrt(V)[Reχ,Imχ]`，`V=1/576`。完整在线目标为

\[
\Phi(x)=\tfrac{{1}}{{2}}\|F(x)-d\|^2/\|y_b\|^2+\lambda\,TV(\chi-\chi_b),
\quad\lambda=.001(1+\sigma/\mathrm{{RMS}}(y_b)).
\]

各臂以自身J和数据梯度确定方向曲率及初始步，按冻结规则作TV/box prox，mass信赖半径0.15、最多8次halving。完整F目标Armijo和完整数据不增共同决定接受；真值只在之后评分。此处是一次 gradient/prox更新，不是GN、重新非线性重建或最优反馈收益上界。L/R也支付完整F接受审核，不能据此声称廉价独立部署。

## 数值修复与失败

原 accelerated-PDHG 的完整尝试1024行中有387行数值失败，另一个早期因逐迭代文件计费开销被中断的尝试也保留。小gap不替代KKT。数值修订先在固定首个ring parent的12个无truth问题核验FGP，12/12认证后才统一使用同一FGP重跑。目标、TV、box、接受规则、KKT/gap1e-6、feasibility1e-10和20,000次/2秒上限均未改变。最终1024/1024行有效、零prox/core失败、材料全部可行被动。原失败不被覆盖，不作success-only统计。

一个同输入PDHG控制在2秒后仍有primal stationarity4.91e-6，dual fixed-point1.19e-9，gap1.65e-11；此处是primal收敛瓶颈，不能笼统称为A24历史的dual残差问题。FGP健康问题270–410次、约0.26–0.41秒。健康JSON的通用`sigma`字段被内部dual步长覆盖；它不是观测噪声sigma，实际目标使用原数据sigma，主CSV噪声字段正确。

## 主网络结果

{overall}

{gains}

完整数据下降但材料变差的F条件有 **{material_worse}/128**，没有被删除。N3的平均IoU/edge未因F取得结构性跃升；SSIM改善不能单独等同于薄边界恢复。完整逐条件、parent和全部指标的bootstrap见[原始表]({protocol}tables/PER_PARENT.csv)、[配对统计]({protocol}PAIRED_STATISTICS.json)。

## N1辅助对照

{auxiliary}

N1支持相同的“数据收益大于材料收益”方向，不能用辅助网络替代预注册N3结论。[实际图像]({figure}n3_fixed_delta_real.png)、[虚部图]({figure}n3_fixed_delta_imag.png)是元数据固定的首parent/高幅度/20dB，不按图像质量挑例子。

actual修正向量、步范数分母、L/F与R/F差异、trust/backtracking、证书和actions详见[修正审计]({protocol}CORRECTION_AUDIT.json)。已有的box-proxy诊断使用固定背景步长，不等于实际自适应TV更新。预prox向量未保存，只能报告其标量梯度/曲率/步长，不能伪造方向比较。

实际N3复杂家族的R/F修正向量相对差为{100*float(r_update['mean']):.3f}%，cos平均{float(r_cos['mean']):.6f}；L/F相对差为{100*float(l_update['mean']):.3f}%。它们是从相同NONE到实际返回材料的mass位移比较，不能当成GN-step error。L拒绝步产生零向量，使完整L/F cosine汇总为HOLD，而不是删掉零步算“成功cos”。记录的1076次试探中没有TRUST_RADIUS拒绝；最终F接受128/128，所以小材料收益不是由数值失败或信赖半径筛掉有利F样本造成。
''')
        emit('A27_OBSERVABILITY_AND_PRIOR.md',fr'''# 材料可观测性与结构先验误差

局部谱证据显示，CNN已主要解释测量敏感的材料组合，剩余结构误差大多落在局部弱方向；完整数据梯度仍集中在强方向。这解释了“残差明显降低、材料只小幅改善”，但不是全局唯一性或后验不确定性定理。

## OFFLINE诊断定义

只分析元数据固定的每家族首parent、两背景/两幅度20dB、N3/N1，共32个预测状态和 **4个独立parents**。用相同实质量坐标、逐源Re/Im数据packing构造predicted-state full J。均匀已知噪声白化只作整体尺度变换，空间方向不变，阈值按同一尺度变换。完整512×1152实J复用同state的full LU，以32个复接收伴随RHS构造响应，再对八源实虚展开，不输入decoder或在线修正。

`strong`为奇异值≥0.05×背景最大奇异值的右奇异空间；`weak`为其全1152维正交补，包含未在thin SVD显式保存的null方向。不是只把512维rowspace内的小奇异向量叫weak。材料误差 `e=χ预测−χ真值`；信号为已知背景之外的 `h=χ真值−χ_b`。归一化选择性 `q=误差能量份额/信号能量份额` 按条件计算再平均，不能把平均份额之比冒充平均q。

{obs_table}

N3约96.2%的误差在弱方向，但强空间保留约62.5%的真值增量能量，因此不能仅用弱空间维数大来解释。N3强方向误差选择性q约0.059，弱方向约5.05；完整数据梯度约99.97%的能量仍在强空间。背景/预测强空间重叠约0.998，未发现大的强空间旋转是本代表集主要瓶颈。

背景strong rank范围{bg_rank_range}，预测strong rank范围{predicted_rank_range}。overlap使用两空间交叉Gram能量除以较小rank，度量共同方向，不能把接近1解释为两套不同维空间完全相同。

二维网格有1152实未知、512实数据；欠定性本身并不证明某个图像误差不可恢复。阈值、固定复相位生成分布和known-background条件限制了本结论。32状态不是完整16-parent谱验证，不能把原RESULT的`full_cohort_post=true`字段理解为全cohort谱覆盖，它只指输入修正cohort完整。

## soft confidence的边界

`C=V diag(s²/(s²+τ²)) Vᵀ`非幂等。`||e||²=||Ce||²+||(I−C)e||²+2⟨Ce,(I−C)e⟩`，必须保留交叉项。原A26 confidence对角图不是posterior covariance，不能声称误差条、校准不确定性或可唯一恢复。背景与预测confidence在误差上的变化也保存在原表；不能把改用predicted-state full J作为廉价部署方案。

## 对先验的含义

局部弱空间中残差不充分约束薄shell、非对称边界和多尺度细节。结构先验可提供材料假设；它的正确性仍需材料标签、跨源一致性和独立观测验证。物理场/current训练监督可以约束先验，但不会自动新增信息或证明零空间材料唯一性。没有重新训练、变更confidence或打开locked来救结果。

[逐态谱/能量表](results/a27/observability/PER_STATE.csv) · [家族描述表]({protocol}tables/OBSERVABILITY_CONDITION_MEANS.csv) · [谱图]({figure}observability_error_fractions.png)。全部是OFFLINE diagnosis。
''')
        emit('A27_LATENCY_ATTRIBUTION.md',fr'''# 原生延迟与无损工程归因

删除仅用于beta诊断的L前向可减少工作；同state复用LU在这套安全实现里减少一次分解，却增加总时间。算子数与实际单图延迟必须分别裁决。

## 冻结基准与本次匹配测量

A26历史warm为N1 40.8274ms、N2 126.084ms、N3 38.5077ms，原结果和NO_GO不变。本次另测同机resident warm：每臂元数据固定16条件×3次，轮换顺序、清空每图材料/递归state，不复用最终图像。不同会话的GPU/框架状态和计时波动不能据此改写历史126ms。以下优化只能与本次未改动的original匹配比较。

{latency_table}

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

[原生CSV](results/a27/latency/NATIVE_INFERENCE.csv) · [修正全流程CSV](results/a27/latency/POST_RAW_TO_IMAGE.csv) · [质量—时间图]({figure}latency_quality.png)。背景setup、checkpoint载入、首次observable SVD和paid warmups计入实验成本，未进入resident每图窗口；没有新增disk-cold/进程冷启动性能证据。
''')
        emit('A27_MECHANISM_DIAGNOSIS.md',fr'''# Maxwell反馈为什么未变成材料成像优势

测量通过材料和内部总场共同决定。显式Maxwell反馈能纠正当前材料假设的内部多次散射，但不会凭空补足测量弱方向里的结构信息。CNN可以先给结构假设，再让物理修正检验其数据一致性；这个顺序是否值得，必须看材料恢复和完整时间。

**结论：数据拟合是主要收益，存在小幅真实材料改善，未达到明显结构成像收益门槛。** N3复杂三家族的一次完整F修正收益 **{gain(a)}**，相对一次L的额外收益 **{gain(fl)}**；完整数据残差改善 **{gain(data)}**。本证据接近 **Case B，附“小幅材料收益”限定**，不等同于完全没有材料改善。严格Case A={case_a}，Case C={case_c}。若F/L小差异的区间不含零，不能称它们等效或“F无任何价值”。

## 来源、数据与科学边界

来源固定[A26](https://github.com/migodam/a26-om-feedback-prior-completion/tree/{config['source_commit']})。A25/A26代码、历史gates及训练结果保留。只用16个已经暴露的development parents×八条件；N3主比较、N1辅助。没有训练、生成新标签、打开locked ID/OOD、调整R32 rank或启动非线性重建。物理是known-background二维TM24×24、8源32接收器、完整1152实材料网格；数据标签的增量复相位固定，不能推广到任意复材料或三维矢量成像。

T0核对原A24 SciPy与Torch执行port，forward/adjoint/JVP/实packing等最大缺陷约2e-15；N1/N3的128幅冻结checkpoint预测精确重现。Mac测试168/168通过；XINAN168项、0错误0失败、明确跳过3项缺少Clarabel的旧独立参考。恒等式和tests证明接线，不证明材料优越性。FGP共同数值健康12/12、最终修正1024/1024行有效；旧387失败与中断重跑完整保存。

## 1. exact post-decoder是否有效

{overall}

{gains}

F比L更接近有用的修正方向，但一次后处理仍不足以找回大部分结构误差。N3的IoU/edge没有同步改善到支持复杂边界恢复的程度；data下降/material变差的F条件 **{material_worse}/128** 全部留在统计里。其余family、Re/Im、fullχ和置信区间见[修正报告](A27_POST_DECODER_CORRECTION.md)。

## 2. 反馈放在哪里、R32丢失什么

{signal_table}

所有64个20dB BP初值的F−L信号均小于噪声；N3预测后36/64、N1后42/64超过噪声。A26冻结encoder计算反馈的位置过早，是支持的瓶颈之一；但把精确F放到CNN后只产生小幅材料收益，因此不是唯一解释。

F梯度与真值误差的offline alignment由BP约0.536降到N3预测约0.119。完整F比L的预测状态alignment更好，但都没有指向大多数残余结构误差。此alignment需要truth，不能用于在线择步。R32在BP非常准确，CNN及post状态梯度失配变大、最小cos可低于旧0.99检查；初值高精度不认证整个学习轨迹。实际R修正图像接近F，并不能反证梯度风险，因为两者都受共同full-F接受和TV约束。

R32不是A26失败的充分解释：A26实际编码的初值处R梯度准确，且完整F后处理同样没有大的复杂材料收益。本项目不再为此调rank。[全部逐状态数据](results/a27/diagnostics/PER_STATE.csv)、[家族均值]({protocol}tables/STATE_PARENT_MEANS.csv)。clean无噪声比值标undefined，不加floor让它通过。

实际N3复杂家族R/F修正向量约有{100*float(r_update['mean']):.2f}%相对差，最终材料NRMSE却很接近。与L/F位移约{100*float(l_update['mean']):.2f}%相对差并列来看，方向差或传播差不能单独代替结构恢复证据。零拒绝位移导致部分cos定义缺失，原表保留HOLD。

## 3. 剩余误差与梯度的空间错位

{obs_table}

在32个固定代表预测状态中，N3约96.2%的材料误差属于局部弱方向，而数据梯度约99.97%属于强方向；强方向仍含约62.5%的真值增量能量。背景/预测强空间重叠约0.998。支持“先验/弱空间是当前材料瓶颈”的解释，但不是posterior uncertainty或硬phys/prior分解认证。soft projector交叉项、null补空间和4-parent范围详见[可观测性报告](A27_OBSERVABILITY_AND_PRIOR.md)。

## 4. 有限工程收益及真实代价

{latency_table}

beta-only L前向没有进入CNN特征，可以删除并保留安全检查；同会话约13.37%提速，仍比N3慢约2.13倍。LU复用只减少分解数，当前实现总延迟反而上升。一次exact correction约260ms，不是约39ms的BP+CNN同质量提速。阶段事件不能相加、rank不能代替time-to-image，详见[延迟报告](A27_LATENCY_ATTRIBUTION.md)。

## 5. 训练时中间量可行性

已有A25 TRAIN电流足以恢复总场和相对源；256独立clean states的本构/相对恒等式缺陷≤7.14e-14，没有新标签。可设计只在训练出现、部署移除的current/field分支，并约束所有源共享同一材料。效果只是SPEC_ONLY，不预设shell收益。SOM-Net已有current/receiver-data/material losses，内部total-field与material-update-aware项需要公平消融，不能包装成首次物理监督。[最小规格](A27_INTERMEDIATE_SUPERVISION_SPEC.md)。

## 6. A25与A26不能简化成唯一因果

A25正向证据来自特定weak背景/幅度、TV设置和多次更新；A26在弱初值处计算一次冻结编码，并在decoder后没有物理修正。A27改变为单次TV修正及共同full-F接受，目的只是判别瓶颈。不同prior、更新数和接受规则使历史比较无法唯一分离architecture因果。此次不以单次失败杀死所有Maxwell feedback，也不以data下降宣称成像优势。

## 条件决策与下一步

- **Case A：**若完整F在独立材料上显著改善复杂形状，可研究少量物理一致性更新、廉价feedback和current-supervised unrolling。本次冻结明显收益筛选未通过。
- **Case B：**若主要改变数据拟合，优先检验弱方向的结构先验、训练时跨源物理监督与校准不确定性；不能只追求更准确的传播。本次最接近此情形，必须保留小幅材料收益限定。
- **Case C：**若F/L材料收益统计近似相同，停止当前regime中的compression扩展。点差小不等于严格等效；本次严格含零区间条件是否成立由原gate给出，不改阈值。

给GPT Pro的定量设计约束与应保留/训练时学习/删除的计算见[A27_RECOMMENDATION_FOR_GPT_PRO.md](A27_RECOMMENDATION_FOR_GPT_PRO.md)。所有证据为exploratory development，没有自动训练或holdout扩展。科学裁决与网页发布状态分开。

![真实N3材料重建]({figure}n3_fixed_delta_real.png)

![按家族的配对收益]({figure}paired_family_correction_gain.png)

完整原始NPZ/电流谱/动作账本在A27 Release；Git保存CSV、图、配置、源码及失败/terminal receipts。[复现命令](REPRODUCING_A27.md) · [计费报告](A27_COST_AND_RUNTIME.md) · [冻结gate](A27_GATE_DECISION.json)。
''')
        emit('A27_RECOMMENDATION_FOR_GPT_PRO.md',fr'''# 给 GPT Pro 的定量架构设计约束

请从“材料误差位于哪些方向、能被哪些在线信息纠正、代价多少”出发。A27不是另一个训练项目，没有新网络或盲测；以下是下一实验的条件性约束，不是预设架构成功。

## 必须解释的五项证据

1. 冻结BP处反馈/噪声median0.169，0/64超过噪声；N3 CNN后median1.425，36/64超过噪声。相同网络的物理特征应避免只在反馈弱的位置计算；但“稍后加反馈”本身不够。
2. N3复杂形状full-F后处理收益 **{gain(a)}**，F相对L **{gain(fl)}**；full data收益 **{gain(data)}**。约96.2%代表材料误差在局部弱空间，99.97%数据梯度在强空间。不能把data loss当唯一优化目标或用更准的surrogate掩盖弱结构信息不足。
3. 初值R32梯度准确；CNN/post状态并不统一通过0.99cos。实际one-step R图像近F是共同full-F接受下的证据，不是廉价独立反馈的统一认证。禁止再次把rank32仅forward误差当成设计目标。
   实际复杂家族R/F位移相对差约{100*float(r_update['mean']):.2f}%，L/F约{100*float(l_update['mean']):.2f}%，但材料终点增量很小；原修正分母与零步状态都已保存。
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

完整入口：[机制报告](A27_MECHANISM_DIAGNOSIS.md)、[逐parent/CI]({protocol}PAIRED_STATISTICS.json)、[延迟](A27_LATENCY_ATTRIBUTION.md)、[监督规格](A27_INTERMEDIATE_SUPERVISION_SPEC.md)。
''')
        emit('A27_START_HERE.md',fr'''# A27：反馈失效归因与架构余量

**开发诊断完成；未训练、未新增标签、locked ID/OOD未打开。** 完整反馈在CNN之后有小幅材料收益，主要改善数据拟合；冻结的明显复杂结构收益门槛未通过。结论带范围限定，不能改写A25/A26历史结果。

- N3复杂三家族F材料收益：{gain(a)}；F对L增量：{gain(fl)}。
- 固定谱代表状态约96.2%的误差在弱空间，约99.97%的数据梯度在强空间。
- 删beta-only诊断约13.37%节省，仍慢于普通BP+CNN；安全LU reuse减少分解却变慢。
- 训练时current/field监督标签可行，效果为SPEC_ONLY。

[完整中文机制报告](A27_MECHANISM_DIAGNOSIS.md) → [CNN后修正](A27_POST_DECODER_CORRECTION.md) → [弱空间/先验](A27_OBSERVABILITY_AND_PRIOR.md) → [实际延迟](A27_LATENCY_ATTRIBUTION.md) → [监督规格](A27_INTERMEDIATE_SUPERVISION_SPEC.md) → [GPT Pro设计约束](A27_RECOMMENDATION_FOR_GPT_PRO.md)。

[冻结gate](A27_GATE_DECISION.json) · [复现](REPRODUCING_A27.md) · [成本/失败](A27_COST_AND_RUNTIME.md)。原A26的根目录`GATE_DECISION.json`与历史报告保持冻结，不是A27 gate。

网页入口：[仓库](https://github.com/migodam/a27-feedback-failure-attribution) · [中文报告](https://github.com/migodam/a27-feedback-failure-attribution/blob/main/A27_MECHANISM_DIAGNOSIS.md) · [给GPT的匿名raw](https://raw.githubusercontent.com/migodam/a27-feedback-failure-attribution/main/A27_RECOMMENDATION_FOR_GPT_PRO.md) · [Release](https://github.com/migodam/a27-feedback-failure-attribution/releases/tag/a27-v0.1.0)。发布是否完成由实际HTTP/commit receipt判断，科学筛选与发布分开。
''')
        emit('README.md','''# A27 — Maxwell Feedback Failure Attribution

完整中文研究入口：[A27_START_HERE.md](A27_START_HERE.md) 和 [机制诊断报告](A27_MECHANISM_DIAGNOSIS.md)。

This is an exploratory, exposed-development diagnosis of frozen A26 checkpoints, not another training run. Exact post-decoder feedback improves data fitting far more than material structure. Small positive material effects are retained; no claim of zero feedback value, unique material recovery, or fast deployment is made.

The experiment uses all 16 exposed development parents (128 conditions), N3 as primary and N1 as auxiliary, with NONE/L/full Maxwell F/TASK_R32 corrections. Online acceptance never reads truth. A common numerical-only proximal repair keeps the original objective, constraints and certification thresholds; all earlier failures remain archived.

- [Post-decoder materials, family tables and parent bootstrap](A27_POST_DECODER_CORRECTION.md)
- [Offline observability and prior-bias analysis](A27_OBSERVABILITY_AND_PRIOR.md)
- [Native latency, beta-only omission and LU reuse](A27_LATENCY_ATTRIBUTION.md)
- [Training-only current/field supervision specification — no training](A27_INTERMEDIATE_SUPERVISION_SPEC.md)
- [Quantitative brief for GPT Pro](A27_RECOMMENDATION_FOR_GPT_PRO.md)
- [Frozen A27 decisions](A27_GATE_DECISION.json), [costs](A27_COST_AND_RUNTIME.md), [reproduction](REPRODUCING_A27.md)

![Actual material images](results/a27/artifacts_post_fgp/figures/n3_fixed_delta_real.png)

Code, configurations, scalar evidence, figures and terminal receipts are in Git. Raw reconstruction/spectral arrays, detailed action ledgers and original failures are in the [A27 release](https://github.com/migodam/a27-feedback-failure-attribution/releases/tag/a27-v0.1.0). The [anonymous GPT brief](https://raw.githubusercontent.com/migodam/a27-feedback-failure-attribution/main/A27_RECOMMENDATION_FOR_GPT_PRO.md) links all supporting evidence.

Physics remains the validated 24×24 **2D scalar TM** backend with known weak/ring backgrounds and fixed relative complex-phase labels. No 3D Vector-Maxwell claim, locked ID/OOD access, new labels, rank search, CNN retraining, VAE/diffusion or new nonlinear reconstruction occurred. Inherited A24–A26 files and their original gate decisions are historical controls, not new A27 evidence. The original root `GATE_DECISION.json` belongs to A26; the A27 decision is explicitly named above.
''')
        print(json.dumps({'report_status':'COMPLETE','interpretation':interpretation,'Case_A':case_a,'Case_C':case_c,
                          'N3_complex_F_gain':a['point_estimate'],'N3_complex_F_vs_L_gain':fl['point_estimate']}))
        book.check()


if __name__=='__main__':main()
