"""Fixed-cohort A27 experiment stages; labels confined to offline scoring."""
from pathlib import Path
import csv,json,math,time
import numpy as np
import torch
from a24.contracts import NumericalFailure
from a25.torch_physics import mass_pack,mass_unpack,pack_data
from a26.data import load_split,online_from_split
from a26.encoder import load_encoders
from a26.evaluation import clean,write_csv,representative_indices,CachedFullPhysics,CountOnlyOnlineBook
from a26.training import load_trained_model,load_feature_cache
from a26.metrics import image_metrics
from a27.budget import write_json,BudgetExceeded
from .exact import ExactMaxwell
from .correction import correct_prediction


def _inputs(root,data_root,a26_root,config,book,device):
    old=json.loads((Path(root)/'configs/a26/pilot.json').read_text())
    labels=load_split(data_root,'validation')
    online=online_from_split(labels)
    enc=load_encoders(data_root,old,book,device=device)
    path=Path(a26_root).resolve()/'results/a26/evaluation/261009/PREDICTIONS.npz'
    if 'locked' in {p.lower() for p in path.parts}:raise ValueError('LOCKED_INPUT_FORBIDDEN')
    with np.load(path,allow_pickle=False) as archive:
        if not np.array_equal(archive['case_id'],online['case_id']):raise ValueError('A26_CASE_ORDER_MISMATCH')
        predictions={m:archive[m].copy() for m in ('N1','N3','BASE_N1','BASE_N3','COMMON_BP_INITIAL')}
    if not np.array_equal(predictions['BASE_N1'],predictions['BASE_N3']):raise ValueError('A26_BASELINE_EQUIVALENCE_CONFLICT')
    models={b:ExactMaxwell(e.reference) for b,e in enc.items()}
    return old,labels,online,enc,predictions,models


def _meta(a,i):
    return {'case_id':str(a['case_id'][i]),'parent_id':int(a['parent_id'][i]),
            'family':str(a['family'][i]),'background_id':str(a['background_id'][i]),
            'amplitude':float(a['amplitude'][i]),'sigma':float(a['sigma'][i]),
            'noise':'clean' if a['sigma'][i]==0 else '20dB'}


def _sync(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize()


def post_source(root):
    p=Path(root)/'results/a27/POST_SOURCE.json'
    return json.loads(p.read_text())['folder'] if p.exists() else 'post'


def _numerical_config(config):
    # Numerical repair only: SAME strictly convex prox, certificate, cap,
    # prior, gradient, step and online acceptance. All arms use one method.
    return {**config,'prox_algorithm':'fgp','fgp_authorized':True,
            'numerical_amendment':'A27-NUMERICAL-01'}


def stage_proxhealth(root,data_root,a26_root,config,book,device):
    from a25.prox import ProxProblem,solve_prox,ProxFailure
    from .correction import CPUProxMeter
    _,labels,a,enc,pred,exact=_inputs(root,data_root,a26_root,config,book,device)
    rows=[];controls=[];local=_numerical_config(config)
    # Fixed metadata-only first family parent, ring, both amplitudes/noise.
    parent=min(a['parent_id']);ii=[i for i in range(128) if a['parent_id'][i]==parent and str(a['background_id'][i])=='ring']
    for i in ii:
        e=enc['ring'];ref=e.reference;x=mass_pack(torch.as_tensor(pred['N3'][i],dtype=torch.complex128,device=device),ref.volume)[None]
        obs=torch.as_tensor(a['data'][i][None],dtype=torch.complex128,device=device)
        for method in ('L','F','TASK_R32'):
            m=exact['ring'] if method=='F' else e.models['R' if method=='TASK_R32' else 'L']
            st=m(x) if method=='F' else m(x,recover_current=False);g=m.data_gradient(x,obs,st);jg=m.jvp(x,g,st)
            step=min(.8*float(g.square().sum())/float(jg.square().sum()/ref.yb_norm2),4*float(ref.step))
            rms=math.sqrt(float(ref.yb_norm2)/(ref.q*ref.s));lam=.01*(1+a['sigma'][i]/rms)*config['tv_multiplier']
            problem=ProxProblem((x-step*g).cpu().numpy()[0],mass_pack(ref.chi_b,ref.volume).cpu().numpy(),step*lam,
                                (24,24),ref.volume,config=local,owner_id='health:'+str(i)+':'+method,algorithm='fgp')
            row={**_meta(a,i),'method':method,'target_read':False}
            if i==ii[0] and method=='L':
                from dataclasses import replace
                control_problem=replace(problem,config={**config,'prox_raise_on_failure':False},algorithm='accelerated_pdhg')
                control=solve_prox(control_problem,budget=CPUProxMeter(book))
                controls.append({**row,**clean(control.metadata),'valid':control.converged,'scope':'same-objective numerical control; no truth score'})
            try:
                value=solve_prox(problem,budget=CPUProxMeter(book));row.update(value.metadata);row['valid']=value.converged
            except ProxFailure as exc:row.update(exc.meta);row['valid']=False
            rows.append(row)
    valid=len(rows)==12 and all(r['valid'] for r in rows)
    result={'status':'PASS' if valid else 'FAIL','problems':len(rows),'valid':sum(r['valid'] for r in rows),'rows':clean(rows),'PDHG_controls':controls,
            'target_read':False,'same_objective_and_tolerances':True,'locked_opened':False}
    write_json(Path(root)/'results/a27/PROX_HEALTH_FGP.json',result)
    return {k:v for k,v in result.items() if k!='rows'}


def stage_postfgp(root,data_root,a26_root,config,book,device):
    health=json.loads((Path(root)/'results/a27/PROX_HEALTH_FGP.json').read_text())
    if health['status']!='PASS':raise RuntimeError('NUMERICAL_ONLY_HEALTH_GATE_NOT_PASSED')
    result=_post(root,data_root,a26_root,_numerical_config(config),book,device,'post_fgp')
    write_json(Path(root)/'results/a27/POST_SOURCE.json',{'folder':'post_fgp','selection':'pre-registered common numerical method; no material metric selection',
               'health_gate':'PROX_HEALTH_FGP','old_attempts_retained':True,'amendment':'A27-NUMERICAL-01'})
    return result


def stage_audit(root,data_root,a26_root,config,book,device):
    old,labels,a,enc,pred,f=_inputs(root,data_root,a26_root,config,book,device)
    records=[]
    with torch.no_grad():
        for bg,e in enc.items():
            ref=e.reference
            i=next(i for i in range(len(a['data'])) if str(a['background_id'][i])==bg)
            x=torch.as_tensor(mass_pack(torch.tensor(pred['N3'][i],dtype=torch.complex128,device=device),ref.volume)[None],device=device)
            state=f[bg](x)
            cpu=CachedFullPhysics(ref,old,book)
            full=cpu.full_state(pred['N3'][i])
            relative=lambda u,v:float(np.linalg.norm(u-v)/np.linalg.norm(v))
            err=relative(state.data[0].cpu().numpy(),full.data)
            rng=np.random.default_rng(np.random.SeedSequence([config['primary_seed'],1,int(i)]))
            d=torch.as_tensor(rng.standard_normal(2*ref.n)[None],dtype=torch.float64,device=device);d/=torch.linalg.vector_norm(d)
            l=torch.as_tensor(rng.standard_normal(2*ref.q*ref.s)[None],dtype=torch.float64,device=device);l/=torch.linalg.vector_norm(l)
            j=f[bg].jvp(x,d,state);v=f[bg].adjoint(x,l,state)
            dot=float(torch.abs((j*l).sum()-(v*d).sum())/torch.maximum(torch.abs((j*l).sum()),torch.abs((v*d).sum())).clamp_min(1e-30))
            jac=f[bg].material_jacobian(x,state)
            jm=float(torch.linalg.vector_norm(j-(jac@d[0])[None])/torch.linalg.vector_norm(j))
            cg=cpu.adjoint_full(full,cpu.unpack_real(l[0].cpu().numpy()))
            ag=relative(v[0].cpu().numpy(),cg)
            rec={'background':bg,'Torch_vs_original_A24_data_error':err,'adjoint_dot_relative_defect':dot,
                 'dense_J_vs_JVP_error':jm,'Torch_vs_original_A24_adjoint_error':ag,'full_backward_residual':float(state.residual.max())}
            if max(err,dot,jm,ag)>1e-9:raise NumericalFailure('A27_T0_BACKEND_IDENTITY_FAILED:'+str(rec))
            records.append(rec)
    out=Path(root)/'results/a27';out.mkdir(parents=True,exist_ok=True)
    checkpoint_checks=[]
    for net in ('N3','N1'):
        cp=load_trained_model(Path(a26_root)/'results/a26/learning/261009'/net/'best.pt',device=device,config=old)
        cache=load_feature_cache(Path(a26_root)/'results/a26/cache'/('validation_'+net+'.npz'),role='validation',method=net,config=old)
        if not np.array_equal(cache['case_id'],a['case_id']):raise ValueError('CHECKPOINT_CACHE_ORDER_CONFLICT')
        values=[]
        with torch.no_grad(),book.span('a27_frozen_checkpoint_reproduction',frozen_checkpoint_predictions=128):
            for start in range(0,128,4):
                features=torch.as_tensor(cache['features'][start:start+4],dtype=torch.float32,device=device)
                base=torch.as_tensor(cache['base_mass'][start:start+4],dtype=torch.float64,device=device)
                values.append(mass_unpack(cp.predict_mass(features,base),1/576).cpu().numpy())
        value=np.concatenate(values)
        error=float(np.linalg.norm(value-pred[net])/np.linalg.norm(pred[net]))
        if error>1e-9:raise ValueError('FROZEN_CHECKPOINT_PREDICTION_CONFLICT:'+net)
        checkpoint_checks.append({'network':net,'prediction_relative_error':error,'rows':128,'trained_in_A27':False})
    result={'T0':'PASS','checks':records,'checkpoint_checks':checkpoint_checks,'rows':128,'parents':16,'new_labels':0,'locked_opened':False,
            'A26_source_commit':config['source_commit'],'physics':'unchanged A24 2D TM; Torch LU execution port verified against original SciPy'}
    write_json(out/'INPUT_BACKEND_AUDIT.json',result)
    return result


def stage_post(root,data_root,a26_root,config,book,device):
    return _post(root,data_root,a26_root,config,book,device,'post')


def _post(root,data_root,a26_root,config,book,device,folder):
    if not (Path(root)/'results/a27/INPUT_BACKEND_AUDIT.json').exists():raise RuntimeError('A27_T0_REQUIRED')
    old,labels,a,enc,pred,exact=_inputs(root,data_root,a26_root,config,book,device)
    out=Path(root)/'results/a27'/folder;out.mkdir(parents=True,exist_ok=True)
    if (out/'PER_CASE.csv').exists():raise RuntimeError('IMMUTABLE_A27_POST_EXISTS')
    rows=[];traces=[];failures=[];outputs={}
    for net in ('N3','N1'):
        for method in config['corrections']:outputs[net+'_'+method]=np.full((128,576),np.nan+1j*np.nan,np.complex128)
        for i in range(len(a['data'])):
            book.check();bg=str(a['background_id'][i]);e=enc[bg];ref=e.reference
            mass=torch.as_tensor(mass_pack(torch.as_tensor(pred[net][i],dtype=torch.complex128,device=device),ref.volume)[None],dtype=torch.float64,device=device)
            obs=torch.as_tensor(a['data'][i][None],dtype=torch.complex128,device=device)
            # Route owns no target. All truth calculations happen AFTER return.
            for method in config['corrections']:
                row={**_meta(a,i),'network':net,'correction':method,'valid':True}
                try:
                    if method=='NONE':
                        v=mass;state=exact[bg](v);meta={'accepted':False,'correction_wall_seconds':0.,'trials':[],'update_mass_norm':0.}
                    else:
                        model=exact[bg] if method=='F' else e.models['R' if method=='TASK_R32' else 'L']
                        result=correct_prediction(mass,obs,float(a['sigma'][i]),model,exact[bg],config,owner=net+':'+str(a['case_id'][i])+':'+method)
                        v,state,meta=result['mass'],result['state'],result['metadata']
                    chi=mass_unpack(v,ref.volume)[0].cpu().numpy()
                    outputs[net+'_'+method][i]=chi
                    row.update(image_metrics(chi,labels['chi'][i],ref.chi_b.cpu().numpy()))
                    residual=float(torch.linalg.vector_norm(state.data-obs));dn=float(torch.linalg.vector_norm(obs-ref.yb))
                    row.update(full_data_absolute_residual=residual,full_data_relative_residual=residual/dn if dn else None,
                               full_backward_residual=float(state.residual.max()),accepted=meta['accepted'],
                               correction_wall_seconds=meta['correction_wall_seconds'],update_mass_norm=meta['update_mass_norm'],
                               line_trials=len(meta['trials']))
                    traces.append({**_meta(a,i),'network':net,'correction':method,**meta})
                except BudgetExceeded:raise
                except Exception as exc:
                    row.update(valid=False,error=type(exc).__name__+':'+str(exc));failures.append(row.copy())
                rows.append(row)
            if (i+1)%16==0:
                write_json(out/'progress.json',{'network':net,'cases_completed':i+1,'rows':len(rows),'failures':len(failures)})
                write_csv(out/'PER_CASE_PARTIAL.csv',rows);write_json(out/'FAILURES_PARTIAL.json',failures)
                np.savez_compressed(out/'PARTIAL_PREDICTIONS.npz',**outputs,case_id=a['case_id'])
                print(json.dumps({'event':'POST_PROGRESS','network':net,'cases':i+1,'failures':len(failures)}),flush=True)
    write_csv(out/'PER_CASE.csv',rows);write_json(out/'TRIALS.json',clean(traces));write_json(out/'FAILURES.json',failures)
    np.savez_compressed(out/'CORRECTED_PREDICTIONS.npz',**outputs,case_id=a['case_id'])
    return {'status':'COMPLETE','rows':len(rows),'cases':128,'parents':16,'numerical_failures':len(failures),'correction_count':1,'truth_used_for_online_acceptance':False}


def _cos(a,b):
    den=float(torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b))
    return float((a*b).sum()/den) if den>0 else None


def _rel(a,b):
    den=float(torch.linalg.vector_norm(b));return float(torch.linalg.vector_norm(a-b)/den) if den>0 else None


@torch.no_grad()
def _diagnose(x,obs,sigma,truth,e,exact,config,key):
    ref=e.reference;f=exact(x);l=e.models['L'](x,recover_current=False);r=e.models['R'](x,recover_current=False)
    e._forward_residual(r)
    gf=exact.data_gradient(x,obs,f);gl=e.models['L'].data_gradient(x,obs,l);gr=e.models['R'].data_gradient(x,obs,r)
    rng=np.random.default_rng(np.random.SeedSequence([config['primary_seed'],2,int(key)]))
    d=torch.as_tensor(rng.standard_normal(x.shape),dtype=torch.float64,device=x.device);d/=torch.linalg.vector_norm(d)
    load=torch.as_tensor(rng.standard_normal((1,2*ref.q*ref.s)),dtype=torch.float64,device=x.device);load/=torch.linalg.vector_norm(load)
    jf=exact.jvp(x,d,f);af=exact.adjoint(x,load,f)
    jr=e.models['R'].jvp(x,d,r);ar=e.models['R'].adjoint(x,load,r)
    xl=ref.project(x-ref.step*gl)-x;xf=ref.project(x-ref.step*gf)-x;xr=ref.project(x-ref.step*gr)-x
    error=x-mass_pack(torch.as_tensor(truth,dtype=torch.complex128,device=x.device),ref.volume)[None]
    noise_rms=float(sigma)*math.sqrt(ref.q*ref.s)
    gapf=float(torch.linalg.vector_norm(f.data-l.data));gapr=float(torch.linalg.vector_norm(r.data-l.data))
    return {'F_minus_L_data_norm':gapf,'R_minus_L_data_norm':gapr,'noise_RMS_l2':noise_rms,
            'F_feedback_noise_ratio':gapf/noise_rms if noise_rms else None,'R_feedback_noise_ratio':gapr/noise_rms if noise_rms else None,
            'R_forward_error_vs_F':_rel(r.data,f.data),'R_differential_data_error_vs_F':_rel(r.data-ref.yb,f.data-ref.yb),
            'L_forward_error_vs_F':_rel(l.data,f.data),'R_JVP_error_vs_F':_rel(jr,jf),'R_adjoint_error_vs_F':_rel(ar,af),
            'R_gradient_cosine_vs_F':_cos(gr,gf),'L_gradient_cosine_vs_F':_cos(gl,gf),'R_gradient_error_vs_F':_rel(gr,gf),
            'R_box_update_error_vs_F':_rel(xr,xf),'L_box_update_error_vs_F':_rel(xl,xf),
            'OFFLINE_F_descent_error_alignment':_cos(gf,error),'OFFLINE_L_descent_error_alignment':_cos(gl,error),
            'OFFLINE_R_descent_error_alignment':_cos(gr,error),
            'OFFLINE_F_projected_step_alignment':_cos(xf,-error),'OFFLINE_L_projected_step_alignment':_cos(xl,-error),
            'OFFLINE_R_projected_step_alignment':_cos(xr,-error),
            'R_core_condition':float(r.margins['K']['condition'].max()),'R_core_relative_sigma_min':float(r.margins['K']['relative_sigma_min'].min()),
            'full_backward_residual':float(f.residual.max()),'scope':'EXPOSED_DEVELOPMENT; truth alignment OFFLINE ONLY'}


def stage_diagnostics(root,data_root,a26_root,config,book,device):
    old,labels,a,enc,pred,exact=_inputs(root,data_root,a26_root,config,book,device)
    out=Path(root)/'results/a27/diagnostics';out.mkdir(parents=True,exist_ok=True)
    if (out/'PER_STATE.csv').exists():raise RuntimeError('IMMUTABLE_A27_DIAGNOSTICS_EXISTS')
    with np.load(Path(root)/'results/a27'/post_source(root)/'CORRECTED_PREDICTIONS.npz',allow_pickle=False) as z:
        if not np.array_equal(z['case_id'],a['case_id']):raise ValueError('A27_CASE_ORDER_MISMATCH')
        post={k:z[k].copy() for k in z.files if k!='case_id'}
    states={'COMMON_BP':pred['COMMON_BP_INITIAL'],'ONE_STEP_L':pred['BASE_N3'],'CNN_N1':pred['N1'],'CNN_N3':pred['N3']}
    for k,v in post.items():
        if not k.endswith('_NONE'):states['POST_'+k]=v
    rows=[];failed=[]
    for name,images in states.items():
        for i,chi in enumerate(images):
            book.check();bg=str(a['background_id'][i]);ref=enc[bg].reference
            row={**_meta(a,i),'state':name,'valid':True}
            try:
                x=mass_pack(torch.as_tensor(chi,dtype=torch.complex128,device=device),ref.volume)[None]
                obs=torch.as_tensor(a['data'][i][None],dtype=torch.complex128,device=device)
                row.update(_diagnose(x,obs,a['sigma'][i],labels['chi'][i],enc[bg],exact[bg],config,i))
            except BudgetExceeded:raise
            except Exception as exc:
                row.update(valid=False,error=type(exc).__name__+':'+str(exc));failed.append(row.copy())
            rows.append(row)
        print(json.dumps({'event':'STATE_DIAGNOSED','state':name,'cases':128,'failures':len(failed)}),flush=True)
        write_json(out/'progress.json',{'state':name,'rows':len(rows),'failures':len(failed)})
    write_csv(out/'PER_STATE.csv',rows);write_json(out/'FAILURES.json',failed)
    return {'rows':len(rows),'states':len(states),'complete':not failed,'new_labels':0,'offline_truth_alignment':True}


def stage_observability(root,data_root,a26_root,config,book,device):
    old,labels,a,enc,pred,exact=_inputs(root,data_root,a26_root,config,book,device)
    out=Path(root)/'results/a27/observability';out.mkdir(parents=True,exist_ok=True)
    if (out/'PER_STATE.csv').exists():raise RuntimeError('IMMUTABLE_A27_SPECTRA_EXISTS')
    rows=[];arrays={};rep=representative_indices(a)
    for i in rep:
        bg=str(a['background_id'][i]);ref=enc[bg].reference;e=enc[bg]
        for net in ('N3','N1'):
            book.check();x=mass_pack(torch.as_tensor(pred[net][i],dtype=torch.complex128,device=device),ref.volume)[None]
            state=exact[bg](x);j=exact[bg].material_jacobian(x,state)
            with book.span('a27_OFFLINE_local_J_SVD',offline_local_J_SVDs=1):
                _,s,vh=torch.linalg.svd(j,full_matrices=False)
            threshold=float(config['spectral_relative_cut'])*e.background_spectrum[0]
            strong=s>=threshold;background_strong=e.background_spectrum>=threshold
            v=vh[strong];vb=e.background_right[background_strong]
            error=x[0]-mass_pack(torch.as_tensor(labels['chi'][i],dtype=torch.complex128,device=device),ref.volume)
            energy=float(error.square().sum());hard=float((v@error).square().sum())
            hardb=float((vb@error).square().sum())
            signal=mass_pack(torch.as_tensor(labels['chi'][i],dtype=torch.complex128,device=device)-ref.chi_b,ref.volume)
            signal_energy=float(signal.square().sum())
            signal_strong=float((v@signal).square().sum())/signal_energy if signal_energy else None
            cweights=s.square()/(s.square()+threshold.square())
            ce=vh.T@(cweights*(vh@error));we=error-ce
            cb=e.confidence@error
            transfer=(v@vb.T);overlap=float(transfer.square().sum()/max(1,min(len(v),len(vb))))
            response=j@error
            chi=mass_unpack(x,ref.volume)[0]
            grad=exact[bg].data_gradient(x,torch.as_tensor(a['data'][i][None],dtype=torch.complex128,device=device),state)
            row={**_meta(a,i),'network':net,'scope':'OFFLINE_LOCAL_SPECTRUM; NOT_POSTERIOR_UNCERTAINTY',
                 'background_strong_rank':int(background_strong.sum()),'predicted_strong_rank':int(strong.sum()),
                 'J_smax':float(s[0]),'J_smin_saved_rowspace':float(s[-1]),'absolute_cut':float(threshold),
                 'error_mass_energy':energy,'strong_error_fraction':hard/energy,'weak_error_fraction':1-hard/energy,
                 'in_grid_truth_signal_energy':signal_energy,'truth_strong_fraction':signal_strong,
                 'truth_weak_fraction':1-signal_strong if signal_strong is not None else None,
                 'q_error_strong':(hard/energy)/signal_strong if signal_strong else None,
                 'q_error_weak':(1-hard/energy)/(1-signal_strong) if signal_strong is not None and signal_strong<1 else None,
                 'background_strong_error_fraction':hardb/energy,'soft_error_fraction':float(ce.square().sum())/energy,
                 'soft_complement_error_fraction':float(we.square().sum())/energy,
                 'soft_cross_energy_fraction':float(2*(ce*we).sum())/energy,
                 'background_predicted_strong_overlap':overlap,'confidence_change_on_error':float(torch.linalg.vector_norm(ce-cb)/torch.linalg.vector_norm(error)),
                 'local_error_data_norm':float(torch.linalg.vector_norm(response)),'full_residual_norm':float(torch.linalg.vector_norm(state.data-torch.as_tensor(a['data'][i][None],device=device))),
                 'full_gradient_mass_norm':float(torch.linalg.vector_norm(grad)),
                 'full_gradient_strong_energy_fraction':float((v@grad[0]).square().sum()/grad.square().sum()) if float(grad.square().sum()) else None,
                 'OFFLINE_gradient_error_alignment':_cos(grad,error[None])}
            rows.append(row);arrays[str(i)+'_'+net+'_spectrum']=s.cpu().numpy()
            arrays[str(i)+'_'+net+'_error_strong']=((v.T@(v@error))).cpu().numpy()
            arrays[str(i)+'_'+net+'_error_weak']=(error-v.T@(v@error)).cpu().numpy()
        print(json.dumps({'event':'SPECTRUM_COMPLETE','case':str(a['case_id'][i])}),flush=True)
    write_csv(out/'PER_STATE.csv',rows);np.savez_compressed(out/'LOCAL_SPECTRA_AND_ERRORS.npz',**arrays)
    return {'status':'COMPLETE','states':len(rows),'representatives':len(rep),'full_cohort_post':True,'offline_only':True}


def stage_latency(root,data_root,a26_root,config,book,device):
    from collections import Counter
    from contextlib import contextmanager
    from a27.latency import build_profile_arms,profile_inference,compare_encoder_outputs,OnlineProfileBook
    old,labels,a,enc,pred,exact=_inputs(root,data_root,a26_root,config,book,device)
    out=Path(root)/'results/a27/latency';out.mkdir(parents=True,exist_ok=True)
    if (out/'NATIVE_INFERENCE.csv').exists():raise RuntimeError('IMMUTABLE_A27_LATENCY_EXISTS')
    cps={m:load_trained_model(Path(a26_root)/'results/a26/learning/261009'/m/'best.pt',device=device,config=old) for m in ('N1','N2','N3')}
    arms={bg:build_profile_arms(e) for bg,e in enc.items()};reps=representative_indices(a)
    native=[];profiles=[];equivalence=[];post=[];failures=[]
    def decode(net):return lambda v:cps[net].predict_mass(v['features'],v['base_mass'])
    # Every warmup is paid; no final-image, material-state or LU carryover.
    for bg,e in enc.items():
        i=next(i for i in reps if str(a['background_id'][i])==bg)
        for method in ('N1','N2','N3'):
            profile_inference(e,decode(method),a['data'][i],a['sigma'][i],method,book=book,timing_mode='native',metadata={'arm':'PAID_WARMUP','case_id':str(a['case_id'][i])})
        for name,(arm,view) in arms[bg].items():
            profile_inference(view,decode(arm.method),a['data'][i],a['sigma'][i],arm.method,book=book,timing_mode='native',metadata={'arm':'PAID_VARIANT_WARMUP_'+name,'case_id':str(a['case_id'][i])})
    for i in reps:
        bg=str(a['background_id'][i]);e=enc[bg];outputs={}
        # Original native encoders are measured explicitly, apart from the
        # additional named-span instrumented views of identical equations.
        plan=[('ORIGINAL_'+m,m,e) for m in ('N1','N2','N3')]
        plan += [(name,arm.method,view) for name,(arm,view) in arms[bg].items()]
        for repeat in range(int(config['latency_repeats'])):
            order=plan[repeat%len(plan):]+plan[:repeat%len(plan)]
            for name,method,view in order:
                book.check()
                try:
                    result=profile_inference(view,decode(method),a['data'][i],a['sigma'][i],method,book=book,timing_mode='native',metadata={**_meta(a,i),'arm':name,'network':method,'repeat':repeat})
                    native.append(result['timing'])
                    if repeat==0:outputs[name]=result
                except BudgetExceeded:raise
                except Exception as exc:failures.append({**_meta(a,i),'arm':name,'repeat':repeat,'error':str(exc)})
        for name,armview in arms[bg].items():
            arm,view=armview
            if name not in outputs:continue
            if 'ORIGINAL_'+arm.method not in outputs:
                equivalence.append({**_meta(a,i),'arm':name,'status':'UNAVAILABLE_ORIGINAL_FAILED'});continue
            original=outputs['ORIGINAL_'+arm.method]
            receipt=compare_encoder_outputs(original['encoded'],outputs[name]['encoded'])
            norm=float(torch.linalg.vector_norm(original['mass']));diff=float(torch.linalg.vector_norm(original['mass']-outputs[name]['mass']))
            mdiff=diff/norm if norm else (0. if diff==0 else float('inf'))
            if not math.isfinite(mdiff) or mdiff>1e-9:raise NumericalFailure('LOSSLESS_ENGINEERING_CHANGED_FROZEN_IMAGE:'+name)
            for row in native:
                if row.get('case_id')==str(a['case_id'][i]) and row.get('arm')==name:row['lossless_equivalence_passed']=True
            equivalence.append({**_meta(a,i),'arm':name,'decoder_mass_relative_error':mdiff,'receipt':receipt})
            try:
                prof=profile_inference(view,decode(arm.method),a['data'][i],a['sigma'][i],arm.method,book=book,timing_mode='stages',metadata={**_meta(a,i),'arm':name,'network':arm.method,'repeat':0})
                profiles.append(prof['timing'])
            except BudgetExceeded:raise
            except Exception as exc:failures.append({**_meta(a,i),'arm':name,'timing_mode':'stages','error':str(exc)})
        print(json.dumps({'event':'LATENCY_CASE','case':str(a['case_id'][i]),'native_rows':len(native),'failures':len(failures)}),flush=True)
    class CorrectionMeter(OnlineProfileBook):
        # Same outer granularity as frozen A26 native inference. CPU prox
        # keeps its own wall cap; all numerical guard decisions remain.
        def snapshot(self):return {'counts':dict(self.counts),'exclusive_walls':{}}
        def delta(self,before):return {'counts':dict(self.counts-Counter(before['counts'])),'exclusive_walls':{}}
        @contextmanager
        def span(self,label,**counts):
            self.counts.update(counts);yield {}
    # Measured raw input -> original N3 CNN -> correction -> host image.
    # This includes every paid full-F acceptance audit; not decoder-cache time.
    correction_config=_numerical_config(config) if post_source(root)=='post_fgp' else config
    for i in reps:
        bg=str(a['background_id'][i]);e=enc[bg];ref=e.reference
        for repeat in range(int(config['latency_repeats'])):
            methods=list(config['corrections']);methods=methods[repeat%4:]+methods[:repeat%4]
            for method in methods:
                book.check();oldbook=e.book;meter=CorrectionMeter(book,device,stages=False);e.bind_book(meter)
                _sync(device);start=time.perf_counter();status='OK'
                try:
                    with torch.no_grad():
                        v=e.encode(a['data'][i],float(a['sigma'][i]),'N3')
                        x=cps['N3'].predict_mass(v['features'],v['base_mass'])
                        obs=torch.as_tensor(a['data'][i][None],dtype=torch.complex128,device=device)
                        if method!='NONE':
                            m=exact[bg] if method=='F' else e.models['R' if method=='TASK_R32' else 'L']
                            c=correct_prediction(x,obs,a['sigma'][i],m,exact[bg],correction_config,owner='native:'+str(a['case_id'][i])+':'+method)
                            x=c['mass']
                        mass_unpack(x,ref.volume).cpu().numpy()
                except BudgetExceeded:raise
                except Exception as exc:
                    status='FAILED';failures.append({**_meta(a,i),'arm':'POST_'+method,'repeat':repeat,'error':str(exc)})
                finally:
                    _sync(device);elapsed=time.perf_counter()-start;e.bind_book(oldbook);book.counts.update(meter.counts);book.check()
                post.append({**_meta(a,i),'network':'N3','correction':method,'arm':'POST_'+method,'repeat':repeat,'status':status,'timing_mode':'native',
                             'native_full_inference_seconds':elapsed,'action_counts':dict(meter.counts),'full_F_acceptance_audit_included':method!='NONE',
                             'post_source_folder':post_source(root)})
    def scalar_rows(rows):
        return [{k:v for k,v in row.items() if not isinstance(v,(dict,list))} for row in rows]
    write_csv(out/'NATIVE_INFERENCE.csv',scalar_rows(native));write_json(out/'NATIVE_INFERENCE.json',clean(native))
    write_csv(out/'POST_RAW_TO_IMAGE.csv',scalar_rows(post));write_json(out/'POST_RAW_TO_IMAGE.json',clean(post))
    write_json(out/'STAGE_PROFILES.json',clean(profiles));write_json(out/'EQUIVALENCE.json',clean(equivalence));write_json(out/'FAILURES.json',failures)
    return {'native_rows':len(native),'stage_rows':len(profiles),'raw_to_corrected_rows':len(post),'numerical_failures':len(failures),
            'guard_retained':True,'new_training':False,'equivalence_cases':len(equivalence),'native_and_instrumented_runs_separate':True}


def stage_report(root,data_root,a26_root,config,book,device):
    from a27.artifacts import build_artifacts
    return build_artifacts(root,data_root,a26_root,config,book,device)
