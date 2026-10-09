"""Save A27 costs, failures, input/source provenance and raw evidence bundle.

No solver, labels, training, credentials, hashes or locked data are read here.
Reporting cutoffs are explicit; publication has a separate terminal receipt.
"""
from __future__ import annotations
import time
START_WALL, START_CPU = time.perf_counter(), time.process_time()
import csv
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from a27.budget import charged_job,write_json,budget_history


def register_external_audits():
    records=[('a27-external-supervision-audit',ROOT/'research/delegated/a27_supervision/COMMAND_RECEIPT.json'),
             ('a27-external-artifact-fixture',ROOT/'research/delegated/a27_artifacts/VERIFICATION.json')]
    for job,path in records:
        dst=ROOT/'results/a27/jobs'/job
        if (dst/'job_receipt.json').exists():continue
        data=json.loads(path.read_text())
        if 'supervision' in job:
            wall=data['total_successful_audit_wall_seconds']
            process=data['total_successful_audit_cpu_seconds']
            counts={'existing_TRAIN_label_audit_states':256,'new_full_forward_labels':0}
        else:
            wall,process=data['wall_seconds'],data['process_cpu_seconds']
            counts={'synthetic_artifact_fixture_audits':1,'new_full_forward_labels':0}
        write_json(dst/'job_manifest.json',{'job':job,'status':'EXTERNAL_MEASURED_IMPORT','source':str(path.relative_to(ROOT))})
        write_json(dst/'job_receipt.json',{'job':job,'status':'COMPLETE','device':'cpu','wall_seconds':wall,
            'process_cpu_seconds':process,'gpu_occupation_seconds':0.,'charged_cpu_seconds':max(wall,process),
            'counts':counts,'durable_reservations':{},'external_measured_receipt':True,
            'source':str(path.relative_to(ROOT)),'no_labels_generated':True,'no_training':True})
    # A parent wrapper failed before a job context could be constructed.
    # This is a declared upper-bound charge, not a fabricated timing sample.
    dst=ROOT/'results/a27/jobs/a27-preview-wrapper-failure'
    if not (dst/'job_receipt.json').exists():
        write_json(dst/'job_manifest.json',{'job':dst.name,'status':'FAILED_BEFORE_CONTEXT','measurement':'conservative one-second upper bound'})
        write_json(dst/'job_receipt.json',{'job':dst.name,'status':'FAILED','device':'cpu','wall_seconds':1.,
            'process_cpu_seconds':0.,'gpu_occupation_seconds':0.,'charged_cpu_seconds':1.,'counts':{},'durable_reservations':{},
            'accounting_kind':'CONSERVATIVE_UPPER_BOUND_NOT_NATIVE_TIMING',
            'error':'TypeError: charged_job does not accept started_wall; fixed before saved-data preview retry',
            'physics_started':False})


def render_accounting(cutoff, active_job=None):
    h=budget_history(ROOT)
    terminal=h['receipts']
    cost=[{'event':'TERMINAL_JOB','accounting_cutoff':cutoff,**row} for row in terminal]
    with (ROOT/'A27_COST_LEDGER.jsonl').open('w',encoding='utf-8') as f:
        for row in cost:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    failures=[]
    for row in terminal:
        if row['status']!='COMPLETE':
            failures.append({'kind':'JOB_FAILURE','job':row['job'],'status':row['status'],'error':row.get('error'),
                             'charged_cpu_seconds':row.get('charged_cpu_seconds'),
                             'gpu_seconds':row['gpu_occupation_seconds'],'source':'results/a27/jobs/'+row['job']+'/job_receipt.json'})
    for folder in ('post','post_fgp','diagnostics','latency'):
        p=ROOT/'results/a27'/folder/'FAILURES.json'
        if not p.exists():continue
        data=json.loads(p.read_text())
        for row in data:
            failures.append({'kind':'RETAINED_CASE_FAILURE','attempt_folder':folder,**row})
    with (ROOT/'A27_FAILURE_LEDGER.jsonl').open('w',encoding='utf-8') as f:
        for row in failures:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    unresolved=[j for j in h['unresolved_jobs'] if j!=active_job]
    summary={'cutoff':cutoff,'cpu_charged_seconds':h['cpu_seconds'],'gpu_occupied_seconds':h['gpu_seconds'],
             'cpu_cap_seconds':3600,'gpu_cap_seconds':3600,'accepted_receipts':len(terminal),
             'unresolved_jobs':unresolved,'known_active_job_at_cutoff':active_job,'failure_records':len(failures),'new_labels':0,'training':False,'locked_ID_OOD_opened':False,
             'cpu_rule':'max(wall,process CPU) for CPU job; process CPU for CUDA; conservative one-second failed-wrapper charge explicit',
             'gpu_rule':'entire CUDA job occupied wall, including host prox/diagnostics/setup/failures',
             'conservative_transport':'some CPU control/upload/download wall includes waiting already charged to remote GPU; no claim of machine-compute efficiency from this sum',
             'historical_A24_A26_costs':'provenance only; not reset or added as new A27 execution',
             'publication_receipt':'final publication accounting is a separate terminal job; read local or Release sidecar for later cutoff'}
    write_json(ROOT/'results/a27/COST_SUMMARY.json',summary)
    scientific=[]
    for row in terminal:
        if row.get('stage') in ('audit','post','postfgp','proxhealth','diagnostics','observability','latency','report'):
            scientific.append([row['job'],row['status'],row['device'],f"{row['wall_seconds']:.3f}",
                               f"{row.get('charged_cpu_seconds',row['process_cpu_seconds']):.3f}",f"{row['gpu_occupation_seconds']:.3f}"])
    body='\n'.join('| '+' | '.join(r)+' |' for r in scientific)
    (ROOT/'A27_COST_AND_RUNTIME.md').write_text(f'''# A27成本与失败边界

当前发布准备cutoff `{cutoff}`：CPU **{h['cpu_seconds']:.3f}s**，GPU占用 **{h['gpu_seconds']:.3f}s**，各硬上限3600s；terminal receipts {len(terminal)}，未知未结算作业 {unresolved}，此cutoff已知活动作业 `{active_job}`。发布过程另有terminal receipt，不能把当前snapshot称为发布结束的完整总数。最终网页sidecar/本地账本保存后续cutoff。

CPU作业按wall/process较大者；CUDA作业host process CPU另计，GPU按整个占用wall，包含CPU prox、setup、审核、失败和重跑。部分transport CPU等待与remote GPU占用重叠，属于保守资源记账，不能把二者相加叫物理算法CPU时间。人工/模型推理与源码编辑时间不伪装为本机科学计算；可测的离线标签审计、表图fixtures、报告/打包/发布单独入账。失败wrapper的1秒是显式上界收费，不是性能样本。

| 主要执行 | 状态 | device | wall s | CPU charge s | GPU occupied s |
| --- | --- | --- | --- | --- | --- |
{body}

最终修正阶段 `post_fgp` 1024/1024有效。旧accelerated-PDHG的387个case失败、早期计费开销中断、测试import/平台路径问题和一次Windows进程启动退出均保留；进程退出原因未确定，不能当成物理失败或成功。新FGP、完整backend和安全core没有静默fallback。数据下降/材料变差是有效的科学负结果，保留在主表，另列于UPDATE_AND_FAILURE_COUNTS。

逐job动作在receipt和raw ledger中。`linear_solve_rhs`包含具体role RHS，`LU_factorizations`包含full/background/core角色；它们相互重叠，不能全部相加成“总Maxwell向量数”。receiver/G/Db动作、full RHS/core RHS和LU分别报告；offline full-J/SVD、train-label audit和online correction也分列。已存cache是历史资产，当前读取/setup/刷新/warmup均付费。

resident warm原生计时覆盖raw observation→encoder→CNN→box→host image。post计时包含prox、line search和所有full-F接受审核，以及修正内部两次计时fence。offline truth指标、谱审计另列，计入campaign资源但不是默认部署操作。没有新增进程冷启动/disk-cold/吞吐性能claim。

[逐job费用](A27_COST_LEDGER.jsonl) · [失败](A27_FAILURE_LEDGER.jsonl) · [修正actions](results/a27/artifacts_post_fgp/tables/CORRECTION_ACTION_COUNTS.csv) · [native latency](A27_LATENCY_ATTRIBUTION.md)。所有new labels=0、training=NOT_RUN、locked=UNOPENED。
''',encoding='utf-8')
    return summary


def main():
    register_external_audits()
    config=json.loads((ROOT/'configs/a27/pilot.json').read_text())
    job=sys.argv[1]
    with charged_job(ROOT,job,config=config,metadata={'stage':'delivery-preparation'}) as book:
        book.deadline-=book.started_wall-START_WALL
        book.started_wall,book.started_cpu=START_WALL,START_CPU
        frozen=config['source_commit']
        check=subprocess.run(['git','diff','--quiet',frozen,'--','src/a24','src/a25','src/a26','vendor'],cwd=ROOT)
        if check.returncode:raise ValueError('FROZEN_BACKEND_CHANGED')
        render_accounting('scientific-complete-before-final-publication',active_job=job)
        reports=['README.md','A27_START_HERE.md','A27_MECHANISM_DIAGNOSIS.md','A27_POST_DECODER_CORRECTION.md',
                 'A27_OBSERVABILITY_AND_PRIOR.md','A27_LATENCY_ATTRIBUTION.md','A27_INTERMEDIATE_SUPERVISION_SPEC.md',
                 'A27_RECOMMENDATION_FOR_GPT_PRO.md','REPRODUCING_A27.md']
        missing=[]
        for name in reports:
            text=(ROOT/name).read_text()
            for dest in re.findall(r'\]\(([^\s)]+)\)',text):
                if dest.startswith(('https://','http://','#')):continue
                if not (ROOT/dest.split('#')[0]).exists():missing.append({'file':name,'link':dest})
        if missing:raise ValueError('BROKEN_LOCAL_REPORT_LINKS:'+json.dumps(missing))
        raw=ROOT/'publication/A27_RAW_EVIDENCE.zip';raw.parent.mkdir(exist_ok=True)
        candidates=sorted(p for p in (ROOT/'results/a27').rglob('*') if p.is_file() and
                          'transport_variants' not in p.parts and job not in p.parts and not p.is_symlink() and '.tmp-' not in p.name)
        with book.span('a27_raw_evidence_archive',archives=1),zipfile.ZipFile(raw,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for i,p in enumerate(candidates):
                if i%100==0:book.check()
                if 'private' in p.relative_to(ROOT).parts:raise ValueError('PRIVATE_ARCHIVE_MEMBER')
                z.write(p,p.relative_to(ROOT).as_posix())
        write_json(ROOT/'A27_SOURCE_MANIFEST.json',{'frozen_A26_commit':frozen,'inherited_backend_sources_unchanged':True,
            'new_source':'src/a27','physics':'A24/A25 2D TM existing backend; verified Torch execution port',
            'raw_evidence_asset':'A27_RAW_EVIDENCE.zip','raw_bytes':raw.stat().st_size,'archive_members':len(candidates),
            'SHA256_checks':'NOT_RUN_BY_USER_POLICY','new_labels':0,'new_training':False,'locked_inputs':False,
            'published_input_dependency':'A26 release a26-v0.1.0; supervision spec additionally uses existing A25 TRAIN current',
            'local_report_links_checked':reports,'source_commit_for_replication':'release target git commit'})
        # Current job is deliberately not included as a completed receipt yet.
        render_accounting('scientific-complete-before-final-publication',active_job=job)
        print(json.dumps({'status':'PREPARED','archive_bytes':raw.stat().st_size,'archive_members':len(candidates),'report_links':'PASS'}))
        book.check()
    render_accounting('prepared-before-publication')


if __name__=='__main__':main()
