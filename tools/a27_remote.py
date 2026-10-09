"""A27 bounded foreground transport, using the existing private SSH helper.

No credential value is printed, copied or archived. Every invocation is a
charged local CPU job; GPU children produce separate cumulative receipts.
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import textwrap
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "D:/AI/A27_FEEDBACK_FAILURE_ATTRIBUTION"
READ_ONLY_A26 = "D:/AI/A26_OM_FEEDBACK_PRIOR_COMPLETION"
READ_ONLY_A25_DATA = "D:/AI/A25_ADAPTIVE_OM_FEEDBACK/data"
SHARED = "D:/AI/A20_OPM_IMAGING"
SHARED_LOCK = SHARED + "/runs/gpu.lock"
PYTHON = "D:/python/python.exe"
MAX_ZIP_BYTES = 1024**3
MAX_RESULT_BYTES = 5 * 1024**3  # Bounded diagnostic/action evidence, no campaign expansion.
STAGES = ("audit", "tests", "post", "proxhealth", "postfgp", "diagnostics", "observability", "latency", "report")
# These are the exposed development inputs. Availability checks use is_file
# only: no checkpoint, cache or measurement payload is read by preflight.
A26_INPUTS = tuple(
    ["results/a26/learning/261009/" + method + "/" + name + ".pt"
     for method in ("N1", "N2", "N3") for name in ("best", "middle", "final")]
    + ["results/a26/cache/" + role + "_" + method + ".npz"
       for role in ("train", "validation") for method in ("N1", "N2", "N3")]
    + ["results/a26/cache/validation_ONLINE_MEASUREMENTS.npz",
       "results/a26/evaluation/261009/PREDICTIONS.npz"]
)
A25_INPUTS = ("manifest.json", "geometry.json", "train.npz", "validation.npz",
              "background/weak.npz", "background/ring.npz")


class TransportError(RuntimeError):
    pass


def safe_member(name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or "//" in name:
        return False
    if name.startswith("/") or any(ord(c) < 32 for c in name):
        return False
    return all(p not in {"", ".", "..", "private", ".git"} for p in name.split("/"))


def load_helper(helper_path, private_config, root=ROOT):
    path = Path(helper_path).expanduser().resolve()
    config = Path(private_config).expanduser().resolve()
    if not path.is_file() or path.suffix != ".py" or not config.is_file():
        raise TransportError("EXISTING_PRIVATE_TRANSPORT_UNAVAILABLE")
    root = Path(root).resolve()
    if config.is_relative_to(root) and "private" not in config.relative_to(root).parts:
        raise TransportError("PRIVATE_CONFIG_MUST_BE_EXTERNAL_OR_IGNORED_PRIVATE")
    try:
        spec = importlib.util.spec_from_file_location("a27_existing_transport", path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        if any(not callable(getattr(helper, key, None)) for key in ("connection", "_python", "_windows_cpu_source")):
            raise ValueError()
        helper.ROOT, helper.REMOTE, helper.SHARED = root, REMOTE, SHARED
        helper.SHARED_LOCK, helper.PYTHON = SHARED_LOCK, PYTHON
        connection = helper.connection(config, root=root)
        return helper, connection
    except BaseException:
        raise TransportError("PRIVATE_TRANSPORT_IMPORT_OR_CONNECTION_FAILED") from None


def deployment_members(root, scope, *, freeze_record=None):
    root = Path(root)
    if scope not in {"source", "train", "evaluation", "accounting"}:
        raise TransportError("EXPLICIT_DEPLOYMENT_SCOPE_REQUIRED")
    selected = set()
    if scope in {"source", "train"}:
        for folder in ("src", "tools", "vendor", "tests"):
            selected.update(p for p in (root / folder).rglob("*.py") if "__pycache__" not in p.parts)
        selected.update((root / "configs/a27").glob("*.json"))
        selected.update((root / "configs/a26").glob("*.json"))
        selected.update((root / "configs/a25").glob("*.json"))
        selected.update((root / "configs").glob("*.json"))
        for name in ("requirements.txt", "SOURCE_MANIFEST.json"):
            if (root / name).exists():
                selected.add(root / name)
    if scope == "train":
        raise TransportError("A27_REUSES_REMOTE_A26_AND_A25_READONLY_INPUTS; no data transport")
    if scope == "evaluation":
        raise TransportError("A27_DEVELOPMENT_ONLY; no locked data transport")
    # Only accepted jobs are transferred. An active local transport manifest
    # must not look like an unknown interrupted remote experiment.
    for receipt in (root / "results/a27/jobs").glob("*/job_receipt.json"):
        selected.add(receipt)
        manifest = receipt.parent / "job_manifest.json"
        if manifest.exists():
            selected.add(manifest)
    for p in selected:
        if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) or not safe_member(p.relative_to(root).as_posix()):
            raise TransportError("UNSAFE_DEPLOYMENT_MEMBER")
    return sorted(selected)


def make_archive(root, path, scope, *, freeze_record=None):
    root = Path(root)
    members = deployment_members(root, scope, freeze_record=freeze_record)
    if sum(p.stat().st_size for p in members) > MAX_ZIP_BYTES:
        raise TransportError("DEPLOYMENT_ARCHIVE_TOO_LARGE")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in members:
            archive.write(p, p.relative_to(root).as_posix())
    return [p.relative_to(root).as_posix() for p in members]


def _call(helper, connection, source, timeout=40):
    try:
        return helper._python(connection, source, timeout=timeout, force_file=False)
    except BaseException:
        raise TransportError("REMOTE_ACTION_FAILED; private diagnostics suppressed") from None


def preflight(helper, connection):
    # Short inline query, no remote file, job, lock or result-directory writes.
    code = """import json,os,pathlib,subprocess,sys
import numpy,scipy,torch
q=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True)
g=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
a26=pathlib.Path(%r);a25=pathlib.Path(%r)
a26_inputs={name:(a26/name).is_file() for name in %r}
a25_inputs={name:(a25/name).is_file() for name in %r}
locked=os.path.exists(%r)
print(json.dumps(dict(status='COMPLETE',python=sys.version.split()[0],numpy=numpy.__version__,scipy=scipy.__version__,torch=torch.__version__,cuda_available=torch.cuda.is_available(),queue_query_ok=q.returncode==0,queue_busy=bool(q.stdout.strip()),queue_clear=q.returncode==0 and not q.stdout.strip() and not locked,gpu=g.stdout.strip(),shared_lock_exists=locked,a26_read_only_root=str(a26),a25_read_only_data_root=str(a25),a26_input_availability=a26_inputs,a25_input_availability=a25_inputs,read_only_inputs_available=all(a26_inputs.values()) and all(a25_inputs.values()),input_check='path availability only; no payload opened')))
""" % (READ_ONLY_A26, READ_ONLY_A25_DATA, A26_INPUTS, A25_INPUTS, SHARED_LOCK)
    return _call(helper, connection, code)


def push(helper, connection, root, scope, freeze_record=None):
    with tempfile.TemporaryDirectory(prefix="a27-push-") as directory:
        name = "a27-input-" + uuid.uuid4().hex + ".zip"
        path = Path(directory) / name
        members = make_archive(root, path, scope, freeze_record=freeze_record)
        try:
            connection.shell("$ErrorActionPreference='Stop'; New-Item -ItemType Directory -Force '" + REMOTE + "' | Out-Null", timeout=20)
            connection.copy_to(path, name)
        except BaseException:
            raise TransportError("REMOTE_UPLOAD_FAILED; private diagnostics suppressed") from None
        source = """import json,pathlib,zipfile,os
r=pathlib.Path(%r); z=r/%r
with zipfile.ZipFile(z) as a:
 for m in a.infolist():
  p=pathlib.PurePosixPath(m.filename)
  if p.is_absolute() or any(x in ('..','private','.git') for x in p.parts) or '\\\\' in m.filename or ':' in m.filename: raise RuntimeError('UNSAFE_ARCHIVE')
  dst=r.joinpath(*p.parts)
  if dst.is_symlink() or not dst.resolve().is_relative_to(r.resolve()): raise RuntimeError('UNSAFE_DESTINATION')
  data=a.read(m)
  immutable=(m.filename.startswith(('data/','configs/','results/a27/jobs/')))
  if dst.exists() and immutable:
   if dst.read_bytes()!=data:
    if not (m.filename.endswith('/job_receipt.json') and json.loads(dst.read_text())==json.loads(data.decode('utf-8'))): raise RuntimeError('IMMUTABLE_REMOTE_INPUT_MISMATCH')
  else:
   dst.parent.mkdir(parents=True,exist_ok=True); dst.write_bytes(data)
z.unlink()
print(json.dumps({'status':'COMPLETE','scope':%r,'members':%d}))
""" % (REMOTE, name, scope, len(members))
        return _call(helper, connection, source, timeout=90)


def _run_source(helper, job, stage, device, timeout, config):
    # Child CPU is measured even after forced termination. A separate receipt
    # charges controller/import/stop tails without replacing native receipts.
    return "import os,pathlib,json,time,subprocess,sys\n" + helper._windows_cpu_source() + textwrap.dedent("""
        r=pathlib.Path(%r); sys.path.insert(0,str(r/'src'))
        from a27.budget import budget_history,write_json,BudgetExceeded
        job,stage,device=%r,%r,%r
        history=budget_history(r)
        if history['unresolved_jobs']: raise RuntimeError('UNACCOUNTED_REMOTE_INTERRUPTION')
        if (r/'results/a27/jobs'/job).exists(): raise RuntimeError('JOB_ID_ALREADY_EXISTS')
        queue=lambda: subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True)
        q=queue()
        if device=='cuda' and (q.returncode or q.stdout.strip() or pathlib.Path(%r).exists()): raise RuntimeError('SHARED_GPU_BUSY_OR_UNQUERYABLE')
        cpu_remaining=3600-history['cpu_seconds'];gpu_remaining=3600-history['gpu_seconds']
        if cpu_remaining<=30 or (device=='cuda' and gpu_remaining<=30): raise RuntimeError('CUMULATIVE_RESOURCE_CAP')
        limit=min(float(%r),cpu_remaining-15 if device=='cpu' else gpu_remaining-30)
        logs=r/'runs';logs.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,PYTHONPATH=str(r/'src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        started=time.perf_counter();controller_cpu=time.process_time();child_cpu=0.;owned_lock=None;stopped=False
        with (logs/(job+'.stdout')).open('wb') as out,(logs/(job+'.stderr')).open('wb') as err:
         entry=['tools/a27_driver.py','--stage',stage]
         child=subprocess.Popen([%r,'-B','-u']+entry+['--job',job,'--device',device,'--config',%r],cwd=r,env=env,stdout=out,stderr=err)
         while child.poll() is None:
          child_cpu=measured_cpu(child)
          lock=pathlib.Path(%r)
          if owned_lock is None and lock.exists():
           raw=lock.read_bytes()
           try:
            owner=json.loads(raw)
            if owner.get('pid')==child.pid and owner.get('job')==job: owned_lock=raw
           except ValueError: pass
          elapsed=time.perf_counter()-started
          if not child_cpu_measurement_available or elapsed>=limit or history['cpu_seconds']+child_cpu+time.process_time()-controller_cpu>=3585:
           stopped=True;child.terminate()
           try: child.wait(timeout=5)
           except subprocess.TimeoutExpired: child.kill();child.wait(timeout=5)
           break
          time.sleep(.2)
         child_cpu=measured_cpu(child)
        elapsed=time.perf_counter()-started;controller_process=time.process_time()-controller_cpu
        if not child_cpu_measurement_available: raise RuntimeError('CHILD_CPU_MEASUREMENT_UNAVAILABLE; manual accounting required')
        native_path=r/'results/a27/jobs'/job/'job_receipt.json'
        native=json.loads(native_path.read_text()) if native_path.exists() else None
        if native is None:
         counts={}; ledger=native_path.parent/'COST_LEDGER.jsonl'
         if ledger.exists():
          for line in ledger.read_text().splitlines():
           try: row=json.loads(line)
           except ValueError: raise RuntimeError('TRUNCATED_CHILD_RESERVATION_LEDGER')
           if row.get('event')=='RESERVED': counts[row['category']]=max(counts.get(row['category'],0),row['job_used'])
         native={'job':job,'status':'INTERRUPTED','device':device,'wall_seconds':elapsed,'process_cpu_seconds':child_cpu,'gpu_occupation_seconds':elapsed if device=='cuda' else 0.,'charged_cpu_seconds':max(elapsed,child_cpu) if device=='cpu' else child_cpu,'durable_reservations':counts,'measurement_role':'controller-measured interrupted child'}
         native_path.parent.mkdir(parents=True,exist_ok=True);write_json(native_path,native)
        tail_job=job+'--controller'
        tail_cpu=max(0.,child_cpu-native['process_cpu_seconds'])+controller_process
        tail_wall=max(0.,elapsed-native['wall_seconds'])
        tail_gpu=max(0.,elapsed-native.get('gpu_occupation_seconds',0.)) if device=='cuda' else 0.
        tail={'job':tail_job,'status':'COMPLETE','device':device,'wall_seconds':tail_wall,'process_cpu_seconds':tail_cpu,'gpu_occupation_seconds':tail_gpu,'charged_cpu_seconds':max(tail_wall,tail_cpu) if device=='cpu' else tail_cpu,'durable_reservations':{},'measurement_role':'exclusive controller/import/stop tail','counts':{}}
        tail_path=r/'results/a27/jobs'/tail_job/'job_receipt.json'
        if tail_path.exists(): raise RuntimeError('CONTROLLER_RECEIPT_ALREADY_EXISTS')
        write_json(tail_path,tail)
        lock=pathlib.Path(%r);q=queue()
        if lock.exists() and owned_lock is not None and lock.read_bytes()==owned_lock and child.poll() is not None and q.returncode==0 and not q.stdout.strip(): lock.unlink()
        accounting=r/'runs/a27-accounting.lock'
        if accounting.exists() and child.poll() is not None:
         raw=accounting.read_bytes()
         try:
          owner=json.loads(raw)
          if owner.get('pid')==child.pid and owner.get('job')==job and accounting.read_bytes()==raw: accounting.unlink()
         except ValueError: pass
        print(json.dumps({'status':'BUDGET_STOP' if stopped else 'COMPLETE' if child.returncode==0 else 'FAILED','job':job,'returncode':child.returncode,'native_receipt':native,'controller_receipt':tail,'queue_clear':q.returncode==0 and not q.stdout.strip(),'lock_exists':lock.exists()}))
        """ % (REMOTE, job, stage, device, SHARED_LOCK, timeout, PYTHON, config, SHARED_LOCK, SHARED_LOCK))


def run(helper, connection, root, job, stage, device="cuda", timeout=3600, config="configs/a27/pilot.json"):
    if not re.fullmatch(r"a27-[A-Za-z0-9_-]{1,100}", job) or stage not in STAGES or device not in {"cpu", "cuda"}:
        raise TransportError("INVALID_REGISTERED_REMOTE_JOB")
    try:
        valid_timeout = not isinstance(timeout, bool) and math.isfinite(float(timeout)) and float(timeout) > 0
    except (TypeError, ValueError):
        valid_timeout = False
    if not valid_timeout:
        raise TransportError("POSITIVE_FINITE_REMOTE_TIMEOUT_REQUIRED")
    if not safe_member(config) or not config.startswith("configs/a27/") or not config.endswith(".json"):
        raise TransportError("FROZEN_REMOTE_CONFIG_REQUIRED")
    push(helper, connection, root, "accounting")
    result = _call(helper, connection, _run_source(helper, job, stage, device, timeout, config), timeout=float(timeout)+40)
    sys.path.insert(0, str(Path(root) / "src"))
    from a27.budget import write_json
    for key in ("native_receipt", "controller_receipt"):
        receipt = result[key]
        path = Path(root) / "results/a27/jobs" / receipt["job"] / "job_receipt.json"
        if path.exists():
            if json.loads(path.read_text()) != receipt:
                raise TransportError("ACCEPTED_RECEIPT_MISMATCH_ON_PULL")
        else:
            write_json(path, receipt)
    return {k: v for k, v in result.items() if k not in {"native_receipt", "controller_receipt"}}


def pull(helper, connection, root):
    name = "a27-results-" + uuid.uuid4().hex + ".zip"
    source = """import pathlib,zipfile,json
r=pathlib.Path(%r);z=r/%r
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as a:
 for base in (r/'results/a27',r/'runs'):
  for p in base.rglob('*'):
   if p.is_file() and not p.is_symlink() and p.name not in ('gpu.lock','a27-accounting.lock') and 'private' not in p.parts: a.write(p,p.relative_to(r).as_posix())
print(json.dumps({'status':'COMPLETE','archive':%r}))
""" % (REMOTE, name, name)
    _call(helper, connection, source, timeout=90)
    with tempfile.TemporaryDirectory(prefix="a27-pull-") as directory:
        path = Path(directory) / name
        try:
            connection.copy_from(name, path)
        except BaseException:
            raise TransportError("REMOTE_DOWNLOAD_FAILED; private diagnostics suppressed") from None
        with zipfile.ZipFile(path) as archive:
            if sum(m.file_size for m in archive.infolist()) > MAX_RESULT_BYTES:
                raise TransportError("PULL_ARCHIVE_TOO_LARGE")
            for member in archive.infolist():
                if not safe_member(member.filename) or not member.filename.startswith(("results/a27/", "runs/")):
                    raise TransportError("UNSAFE_RESULT_MEMBER")
                destination = Path(root).joinpath(*PurePosixPath(member.filename).parts)
                if destination.is_symlink() or not destination.resolve().is_relative_to(Path(root).resolve()):
                    raise TransportError("UNSAFE_RESULT_DESTINATION")
                data = archive.read(member)
                if destination.exists() and destination.read_bytes() != data:
                    # Preserve both failure/action traces. Never overwrite a
                    # terminal receipt, nor silently combine divergent ledgers.
                    if destination.name == "job_receipt.json":
                        # Windows CRLF and macOS LF are different byte encodings
                        # of the same immutable JSON resource receipt. Verify
                        # every field; retain the already accepted local file.
                        if json.loads(destination.read_text()) != json.loads(data.decode('utf-8')):
                            raise TransportError("ACCEPTED_RECEIPT_MISMATCH_ON_PULL:" + member.filename)
                        continue
                    variant = Path(root) / "results/a27/transport_variants" / uuid.uuid4().hex / member.filename
                    variant.parent.mkdir(parents=True, exist_ok=True)
                    variant.write_bytes(data)
                elif not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(data)
    return {"status": "COMPLETE", "conflict_policy": "accepted receipts immutable; differing traces retained as variants"}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "push", "run", "pull"))
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--transport-helper", default=os.environ.get("A27_TRANSPORT_HELPER"), required=False)
    parser.add_argument("--private-config", required=True)
    parser.add_argument("--scope", choices=("source", "train", "evaluation", "accounting"), default="source")
    parser.add_argument("--freeze-record")
    parser.add_argument("--stage", choices=STAGES, default="audit")
    parser.add_argument("--job")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--timeout", type=float, default=3600.)
    parser.add_argument("--config", default="configs/a27/pilot.json")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root / "src"))
    from a27.budget import charged_job, write_json
    local_job = "a27-transport-" + args.action + "-" + uuid.uuid4().hex[:12]
    monitor = args.action == "run" and args.device == "cuda"
    with charged_job(root, local_job, device="cuda" if monitor else "cpu", gpu_host_only=monitor,
                     metadata={"stage": "transport", "action": args.action}) as book:
        helper, connection = load_helper(args.transport_helper or "", args.private_config, root)
        with book.span("remote_transport_" + args.action):
            if args.action == "preflight":
                result = preflight(helper, connection)
            elif args.action == "push":
                result = push(helper, connection, root, args.scope, args.freeze_record)
            elif args.action == "run":
                result = run(helper, connection, root, args.job or "", args.stage, args.device, args.timeout, args.config)
            else:
                result = pull(helper, connection, root)
        write_json(root / "results/a27/transport" / (local_job + ".json"), result)
        print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
