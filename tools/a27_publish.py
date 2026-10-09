"""Authorized A27 public archive; scientific status never depends on publish.

No credential command output, SSH configuration, locked data or SHA256 checks.
"""
from __future__ import annotations
import time
START_WALL, START_CPU=time.perf_counter(),time.process_time()
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tools'))
from a27.budget import charged_job,write_json,budget_history
from a27_prepare_delivery import render_accounting,register_external_audits

REPO='migodam/a27-feedback-failure-attribution'
TAG='a27-v0.1.0'


def command(args,book,timeout=180,allow_failure=False):
    book.check()
    r=subprocess.run(args,cwd=ROOT,text=True,capture_output=True,timeout=timeout)
    book.check()
    if r.returncode and not allow_failure:
        raise RuntimeError('PUBLICATION_COMMAND_FAILED:'+args[0]+'; credential/private diagnostics suppressed')
    return r


def anonymous(url,book,needle=None):
    book.check()
    request=urllib.request.Request(url,headers={'User-Agent':'A27-public-artifact-verification'})
    with urllib.request.urlopen(request,timeout=25) as response:
        value=response.read(300000) if needle else response.read(64)
        code=response.status
    if code!=200 or (needle is not None and needle.encode() not in value):
        raise ValueError('ANONYMOUS_ARTIFACT_VERIFICATION_FAILED')
    book.check()
    return {'url':url,'status':code,'authentication_sent':False,'expected_content_checked':needle is not None}


def main():
    register_external_audits()
    config=json.loads((ROOT/'configs/a27/pilot.json').read_text())
    job=sys.argv[1]
    with charged_job(ROOT,job,config=config,metadata={'stage':'public-delivery'}) as book:
        book.deadline-=book.started_wall-START_WALL
        book.started_wall,book.started_cpu=START_WALL,START_CPU
        summary=json.loads((ROOT/'results/a27/artifacts_post_fgp/SUMMARY.json').read_text())
        if summary['status']!='COMPLETE':raise ValueError('SCIENTIFIC_DELIVERY_INCOMPLETE')
        for p in [*(ROOT/'src/a27').glob('*.py'),*(ROOT/'tools').glob('a27*.py'),*(ROOT/'tests').glob('test_a27*.py')]:
            ast.parse(p.read_text(),filename=str(p))
        render_accounting('science-and-preparation-complete-before-publication',active_job=job)
        raw=ROOT/'publication/A27_RAW_EVIDENCE.zip'
        members=sorted(p for p in (ROOT/'results/a27').rglob('*') if p.is_file() and not p.is_symlink()
                       and 'transport_variants' not in p.parts and job not in p.parts and '.tmp-' not in p.name)
        with book.span('a27_final_raw_bundle',archives=1),zipfile.ZipFile(raw,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for i,p in enumerate(members):
                if i%100==0:book.check()
                z.write(p,p.relative_to(ROOT).as_posix())
        manifest=json.loads((ROOT/'A27_SOURCE_MANIFEST.json').read_text())
        manifest.update(raw_bytes=raw.stat().st_size,archive_members=len(members),
                        publication_job_excluded_from_prepublication_raw=True)
        write_json(ROOT/'A27_SOURCE_MANIFEST.json',manifest)
        add=['.gitignore','README.md','REPRODUCING_A27.md','src/a27','configs/a27','results/a27',
             'research/A27_FROZEN_PROTOCOL.md','research/A27_METERING_REPAIR.md','research/A27_NUMERICAL_AMENDMENT_01.md',
             'research/delegated/a27_artifacts','research/delegated/a27_infrastructure','research/delegated/a27_latency',
             'research/delegated/a27_supervision']
        add += [p.name for p in ROOT.glob('A27_*') if p.is_file()]
        add += [str(p.relative_to(ROOT)) for p in (ROOT/'tools').glob('a27*.py')]
        add += [str(p.relative_to(ROOT)) for p in (ROOT/'tests').glob('test_a27*.py')]
        command(['git','add','--',*add],book)
        files=command(['git','diff','--cached','--name-only','-z'],book).stdout.split('\0')
        # Log filenames only if rejected; never print matching bytes.
        patterns=[re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----'),
                  re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}'),re.compile(rb'AKIA[A-Z0-9]{16}'),
                  re.compile(rb'(?<![A-Za-z0-9])100\.(?:[0-9]{1,3}\.){2}[0-9]{1,3}(?![0-9])')]
        rejected=[]
        for name in files:
            if not name:continue
            p=ROOT/name
            if p.is_symlink() or any(part in ('private','locked','locked_id','locked_ood') for part in p.relative_to(ROOT).parts):
                rejected.append(name);continue
            data=p.read_bytes()
            if any(pattern.search(data) for pattern in patterns):rejected.append(name)
            if p.suffix=='.npz' or p.suffix=='.pt':rejected.append(name)
        if rejected:raise ValueError('PUBLICATION_AUDIT_REJECTED_FILENAMES:'+json.dumps(sorted(set(rejected))))
        write_json(ROOT/'publication/A27_PUBLIC_ARTIFACT_AUDIT.json',{
            'status':'PASS','files':len([p for p in files if p]),'credential_scan':'matched bytes never logged',
            'locked_paths':'EXCLUDED','private_configuration':'EXCLUDED','new_training':'NOT_RUN','SHA256_checks':'NOT_RUN',
            'theory_pdf_redistribution':'NONE','raw_npz':'Release, exposed prediction/diagnostic outputs only'})
        command(['git','add','publication/A27_PUBLIC_ARTIFACT_AUDIT.json'],book)
        sourcefiles=[str(p.relative_to(ROOT)) for p in [*(ROOT/'src/a27').glob('*.py'),*(ROOT/'tools').glob('a27*.py'),*(ROOT/'tests').glob('test_a27*.py')]]
        command(['git','diff','--cached','--check','--',*sourcefiles],book)
        command(['git','commit','-m','A27: diagnose feedback headroom with full-cohort corrections and native latency'],book)
        sha=command(['git','rev-parse','HEAD'],book).stdout.strip()
        probe=command(['gh','repo','view',REPO,'--json','url,isEmpty,isPrivate'],book,allow_failure=True)
        if probe.returncode:
            command(['gh','repo','create',REPO,'--public','--description','A27 exposed-development Maxwell feedback mechanism diagnosis; no new training or labels'],book)
        else:
            existing=json.loads(probe.stdout)
            if not existing['isEmpty']:raise ValueError('EXISTING_NONEMPTY_REPOSITORY; do not overwrite unrelated work')
        remotes=command(['git','remote'],book).stdout.splitlines()
        if 'a27' not in remotes:command(['git','remote','add','a27','https://github.com/'+REPO+'.git'],book)
        if command(['git','remote','get-url','a27'],book).stdout.strip()!='https://github.com/'+REPO+'.git':
            raise ValueError('UNEXPECTED_A27_PUSH_DESTINATION')
        print(json.dumps({'stage':'PUSH','commit':sha}),flush=True)
        command(['git','push','a27','HEAD:main'],book,timeout=240)
        notes=ROOT/'publication/A27_RELEASE_NOTES.md'
        notes.write_text('''# A27 exposed-development diagnosis

Full-cohort NONE/L/exact F/TASK_R32 corrections of frozen N3/N1. No new labels, training, locked evaluation or rank tuning. Small material gains are retained; the primary benefit is data consistency. The original numerical failures, common certified solver repair, full action ledgers, raw arrays, spectra and native latency are preserved.

Read A27_START_HERE.md and A27_MECHANISM_DIAGNOSIS.md in the repository. Publication does not imply a scientific GO. A27_RAW_EVIDENCE.zip is the prepublication scientific evidence bundle; final publication verification/accounting is a separate sidecar asset.
''',encoding='utf-8')
        command(['gh','release','create',TAG,str(raw),'--repo',REPO,'--target',sha,'--title','A27 mechanism diagnosis v0.1.0','--notes-file',str(notes)],book,timeout=240)
        remote=command(['git','ls-remote','a27','refs/heads/main'],book).stdout.split()[0]
        if remote!=sha:raise ValueError('REMOTE_COMMIT_MISMATCH')
        release=json.loads(command(['gh','release','view',TAG,'--repo',REPO,'--json','url,assets,targetCommitish'],book).stdout)
        urls=['https://github.com/'+REPO,
              'https://github.com/'+REPO+'/blob/'+sha+'/A27_MECHANISM_DIAGNOSIS.md',
              'https://raw.githubusercontent.com/'+REPO+'/'+sha+'/A27_MECHANISM_DIAGNOSIS.md',
              'https://raw.githubusercontent.com/'+REPO+'/'+sha+'/A27_RECOMMENDATION_FOR_GPT_PRO.md',release['url']]
        checks=[anonymous(url,book,'Maxwell反馈为什么' if 'raw.' in url and url.endswith('A27_MECHANISM_DIAGNOSIS.md') else
                          '给 GPT Pro' if 'raw.' in url else None) for url in urls]
        for asset in release['assets']:
            url=asset.get('url') or asset.get('browser_download_url')
            if url:checks.append(anonymous(url,book))
        verification={'publication':'VERIFIED','repository':'https://github.com/'+REPO,'commit':sha,'remote_main_commit':remote,
                      'release':release['url'],'targetCommitish':release['targetCommitish'],'anonymous_checks':checks,
                      'scientific_decision':'CASE_B_DATA_DOMINANT_WITH_SMALL_MATERIAL_GAIN; no clear material headroom GO',
                      'training':'NOT_RUN','new_labels':0,'locked':'UNOPENED','SHA256_checks':'NOT_RUN'}
        write_json(ROOT/'publication/A27_PUBLICATION_VERIFICATION.json',verification)
        print(json.dumps(verification,ensure_ascii=False),flush=True)
        book.check()
    # Final native publication receipt exists now. Local accounting includes it;
    # a later sidecar transport has its own explicit accounting cutoff.
    render_accounting('publication-and-anonymous-main-asset-verification-complete')


if __name__=='__main__':main()
