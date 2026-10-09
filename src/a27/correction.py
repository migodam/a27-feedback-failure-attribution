"""One measurement-only certified proximal correction at a frozen CNN image."""
import math
import time
import numpy as np
import torch
from a24.material_updates import tv_value
from a24.contracts import NumericalFailure
from a25.prox import ProxProblem, solve_prox
from a25.torch_physics import mass_pack, pack_data


class CPUProxMeter:
    """CPU prox operations inside an inclusive paid CUDA job.

    The certified solver retains its per-iteration wall cap and all numerical
    tests. Expensive job file/monitor checks are sampled every 50 ms, with
    mandatory checks at entry/exit. No CUDA work is done by the CPU prox loop.
    This removes repeated lock I/O rather than changing the proximal method.
    """
    def __init__(self, parent):
        self.parent = parent
        self.counts, self.metadata = parent.counts, parent.metadata
        self.owner_id = getattr(parent, 'owner_id', None)
        self.last = -float('inf')

    def check(self):
        now = time.perf_counter()
        if now - self.last >= .05:
            self.parent.check()
            self.last = now

    def check_owner(self, owner):
        if self.owner_id is not None and str(owner) != str(self.owner_id):
            raise ValueError('A27_FOREIGN_PROX_BUDGET_OWNER')
        self.check()

    def span(self, *args, **kwargs):
        return self.parent.span(*args, **kwargs)

    def snapshot(self):
        return self.parent.snapshot()

    def delta(self, before):
        return self.parent.delta(before)


def _value(state, observation, mass, background, lam, ref):
    data = float((state.data - observation).abs().square().sum() / (2 * ref.yb_norm2))
    tv = tv_value((mass - background).detach().cpu().numpy()[0], (ref.grid, ref.grid), ref.volume)
    return data + lam * tv, data, tv


@torch.no_grad()
def correct_prediction(mass, observation, sigma, model, exact, config, *, owner):
    """No truth, case-family, labels, teachers or offline spectral quantities.

    Full F auditing is an online paid part of EVERY arm's acceptance policy.
    A rejected/zero update returns the original mass; failed numerical states
    raise and must be retained as failures by the experiment driver.
    """
    ref = exact.reference
    if not isinstance(mass, torch.Tensor) or not isinstance(observation, torch.Tensor):
        raise ValueError('A27_RAW_ONLINE_ARRAYS_REQUIRED')
    if mass.ndim == 1:
        mass = mass[None]
    if observation.ndim == 2:
        observation = observation[None]
    if len(mass) != 1 or mass.dtype != torch.float64 or observation.dtype != torch.complex128:
        raise ValueError('A27_SINGLE_MATCHED_DOUBLE_PREDICTION_REQUIRED')
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError('A27_NONNEGATIVE_NOISE_REQUIRED')
    if mass.device.type == 'cuda':
        torch.cuda.synchronize(mass.device)
    start = time.perf_counter()
    counts = dict(ref.book.counts)
    bg = mass_pack(ref.chi_b, ref.volume)[None]
    rms = math.sqrt(float(ref.yb_norm2) / (ref.q * ref.s))
    lam = .01 * (1 + sigma / rms) * float(config['tv_multiplier'])
    full = exact(mass)
    old, old_data, old_tv = _value(full, observation, mass, bg, lam, ref)
    state = full if model is exact else model(mass, recover_current=False)
    gradient = model.data_gradient(mass, observation, state)
    g2 = float(gradient.square().sum())
    trial_records = []
    accepted, result, final_full = False, mass.clone(), full
    if g2 > 0:
        response = model.jvp(mass, gradient, state)
        curvature = float(response.square().sum() / ref.yb_norm2)
        if not math.isfinite(curvature) or curvature <= 0:
            raise NumericalFailure('A27_NONZERO_GRADIENT_ZERO_DIRECTIONAL_CURVATURE')
        initial_step = min(.8 * g2 / curvature,
                           float(config['step_cap_background_multiple']) * float(ref.step))
        for trial in range(int(config['line_trials'])):
            ref.book.check()
            step = initial_step / (2 ** trial)
            problem = ProxProblem((mass - step * gradient).cpu().numpy()[0], bg.cpu().numpy()[0],
                                  step * lam, (ref.grid, ref.grid), ref.volume,
                                  config=config, owner_id=str(owner), algorithm=config.get('prox_algorithm','accelerated_pdhg'))
            prox = solve_prox(problem, budget=CPUProxMeter(ref.book))
            ref.book.check()
            if not prox.converged:
                raise NumericalFailure('A27_UNCERTIFIED_PROX')
            candidate = torch.as_tensor(prox.x[None], dtype=torch.float64, device=mass.device)
            displacement = candidate - mass
            norm = float(torch.linalg.vector_norm(displacement))
            record = {'trial': trial, 'step': step, 'update_mass_norm': norm,
                      'trust_radius_mass': config['trust_radius_mass'],
                      'prox_iterations': prox.metadata['iterations'],
                      'prox_kkt': prox.metadata['kkt_residual'],
                      'prox_normalized_gap': prox.metadata['normalized_gap']}
            if norm > float(config['trust_radius_mass']):
                record.update(accepted=False, reason='TRUST_RADIUS')
                trial_records.append(record)
                continue
            audit = exact(candidate)
            new, new_data, new_tv = _value(audit, observation, candidate, bg, lam, ref)
            armijo = old - float(config['armijo']) * norm * norm / (2 * step)
            # The allowance is roundoff, not a physics/data fitting tolerance.
            allowance = 1e-12 * max(1., abs(old))
            ok = bool(new <= armijo + allowance and new_data <= old_data + allowance)
            record.update(objective=new, data_loss=new_data, tv=new_tv,
                          accepted=ok, reason='ACCEPTED' if ok else 'FULL_OBJECTIVE_OR_DATA_REJECT')
            trial_records.append(record)
            if ok:
                accepted, result, final_full = True, candidate, audit
                break
    else:
        initial_step, curvature = 0., 0.
    if mass.device.type == 'cuda':
        torch.cuda.synchronize(mass.device)
    elapsed = time.perf_counter() - start
    final_objective, final_data, final_tv = _value(final_full, observation, result, bg, lam, ref)
    return {'mass': result, 'state': final_full, 'gradient': gradient,
            'metadata': {'accepted': accepted, 'old_objective': old, 'objective': final_objective,
                         'old_data_loss': old_data, 'data_loss': final_data,
                         'old_tv': old_tv, 'tv': final_tv, 'lambda': lam,
                         'initial_step': initial_step, 'direction_curvature': curvature,
                         'update_mass_norm': float(torch.linalg.vector_norm(result - mass)),
                         'trials': trial_records, 'correction_wall_seconds': elapsed,
                         'gradient_mass_norm': math.sqrt(g2),
                         'online_full_acceptance_audit': True,
                         'actions': {k: int(v - counts.get(k, 0)) for k, v in ref.book.counts.items()
                                     if v != counts.get(k, 0)}}}
