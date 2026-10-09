"""A27 exact TASK_R32 LU reuse and explicit diagnostic-beta omission.

These are engineering variants of the frozen A26 equations.  No cache is
shared between material states.  Domain, core, finite-solve and A26 forward
residual checks remain active; cached LU solves also check backward error.
There is no jitter, pseudoinverse, rank change, L fallback, or label input.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np
import torch

from a24.contracts import NumericalFailure
from a25.torch_physics import (TorchRelativeModel, TorchState, _batched,
                               mass_pack, mass_unpack)
from a26.encoder import FEATURE_NAMES, FrozenEncoder, _diagnostic_values


_CACHE_KEY = "_a27_validated_forward_lu"
_GUARD_KEY = "_a27_core_guard_stamp"
_REFERENCE_BUFFERS = ("chi_b", "eb", "yb", "Db", "G", "S", "incident", "Ug",
                      "Vg", "D_indices", "D_blocks", "step", "yb_norm2", "cb",
                      "Lb", "Lb_lu", "Lb_pivots")
_FROZEN_GUARDS = ("real_lower", "real_upper", "imag_lower", "imag_upper",
                  "feasibility_tolerance", "pole_relative_floor",
                  "identity_tolerance", "backward_residual_tolerance",
                  "core_relative_sigma_min", "core_condition_cap")


def _tensor_stamp(value):
    # inference_mode tensors deliberately have no version counter.  They can
    # still be used safely because cache hits compare detached snapshots too.
    try:
        version = value._version
    except RuntimeError:
        version = None
    return (id(value), version, tuple(value.shape), value.dtype, value.device)


def _reference_stamp(reference):
    return (id(reference), reference.volume, reference.grid, reference.n,
            reference.s, reference.q, reference.rank,
            tuple(_tensor_stamp(getattr(reference, name))
                                for name in _REFERENCE_BUFFERS),
            tuple((key, reference.config.get(key)) for key in _FROZEN_GUARDS))


def _owner_stamp(book):
    return (id(book), getattr(book, "owner_id", None),
            getattr(book, "metadata", {}).get("owner_id"))


@dataclass
class _ValidatedLU:
    state_id: int
    reference_stamp: tuple
    owner_stamp: tuple
    core_stamp: tuple
    core_snapshot: torch.Tensor
    mass_snapshot: torch.Tensor
    h_snapshot: torch.Tensor
    lu: torch.Tensor
    pivots: torch.Tensor


class InstrumentedTorchRelativeModel(TorchRelativeModel):
    """Frozen equations with extra named spans, without changed actions.

    This class supplies common stage boundaries for baseline and reuse arms.
    The solve in a baseline arm is still A25's torch.linalg.solve, whose LU
    factorization and triangular solve cannot be timed separately by spans.
    """
    cache_forward_lu = False

    def _core_solve(self, rhs, state, *, adjoint=False):
        return self._solve(state.extra["K"], rhs, "feedback_K", adjoint=adjoint)

    def solve_A(self, rhs, state, *, adjoint=False):
        rhs, single = _batched(rhs, 3)
        if state.extra.get("model_id") != id(self):
            raise ValueError("TORCH_STATE_MODEL_MISMATCH")
        if self.kind == "L":
            return rhs[0] if single else rhs
        ref = self.reference
        t = self._solve_P(rhs, state, adjoint=adjoint)
        if self.rank:
            if adjoint:
                gamma = self._core_solve(
                    ref.Ug.mH @ (state.extra["h_batch"].conj()[..., None] * t),
                    state, adjoint=True)
                out = t + self._solve_P(ref.Vg @ gamma, state, adjoint=True)
            else:
                gamma = self._core_solve(ref.Vg.mH @ t, state)
                out = t + state.extra["CU"] @ gamma
        else:
            out = t
        return out[0] if single else out

    def forward(self, mass, *, recover_current=True):
        # Same operations, order, failure types and checks as the frozen A25
        # implementation.  Added spans only provide identifiable boundaries.
        ref = self.reference
        mass, single = _batched(mass, 2)
        if mass.shape[-1] != 2 * ref.n or mass.dtype != torch.float64:
            raise ValueError("FLOAT64_MATERIAL_MASS_REQUIRED")
        chi = mass_unpack(mass, ref.volume)
        if not torch.isfinite(chi).all():
            raise NumericalFailure("NONFINITE_TORCH_MATERIAL")
        tolerance = float(ref.config.get("feasibility_tolerance", 1e-10))
        if (bool((chi.real < float(ref.config.get("real_lower", -.5)) - tolerance).any())
                or bool((chi.real > float(ref.config.get("real_upper", 4.)) + tolerance).any())
                or bool((chi.imag < float(ref.config.get("imag_lower", 0.)) - tolerance).any())
                or bool((chi.imag > float(ref.config.get("imag_upper", .5)) + tolerance).any())):
            raise NumericalFailure("TORCH_MATERIAL_DOMAIN_VIOLATION")
        h = chi - ref.chi_b
        state = TorchState(mass, chi, h, None, None, None,
                           extra={"model_id": id(self), "h_batch": h, "single": single})
        with self.book.span("torch_relative_" + self.kind + "_forward",
                            relative_forward_calls=len(mass),
                            relative_forward_source_rhs=len(mass) * ref.s):
            if self.kind == "R":
                if ref.D_indices.numel():
                    with self.book.span("a27_local_core_assembly"):
                        p = torch.eye(ref.D_indices.shape[-1], dtype=torch.complex128,
                                      device=mass.device) - h[:, ref.D_indices, None] * ref.D_blocks
                        state.extra["P"] = p
                    state.margins["P"] = self._guard(p, "P")
                if self.rank:
                    with self.book.span("a27_core_assembly"):
                        cu = self._solve_P(h[..., None] * ref.Ug, state)
                        k = torch.eye(self.rank, dtype=torch.complex128,
                                      device=mass.device) - ref.Vg.mH @ cu
                        state.extra.update(CU=cu, K=k)
                    state.margins["K"] = self._guard(k, "K")
                    if self.cache_forward_lu:
                        state.extra[_GUARD_KEY] = (_tensor_stamp(k), _reference_stamp(ref),
                                                  _owner_stamp(self.book))
            w = self.solve_A(h[..., None] * ref.eb, state)
            field = ref.eb + self.feedback_apply(w)
            data = ref.yb + ref.apply_db(w)
            state.w, state.field, state.data = w, field, data
            if recover_current:
                ref.canonical_current(state)
            if single:
                for name in ("mass", "chi", "h", "w", "field", "data", "current"):
                    value = getattr(state, name)
                    if value is not None:
                        setattr(state, name, value[0])
        return state

    def vjp_complex(self, mass, load, state=None):
        state = self._match(mass, state)
        field, single = _batched(state.field, 3)
        load, _ = _batched(load, 3)
        if load.shape != (len(field), self.reference.q, self.reference.s):
            raise ValueError("MATCHED_COMPLEX_DATA_LOAD_REQUIRED")
        with self.book.span("torch_explicit_adjoint", data_adjoint_directions=len(field),
                            data_adjoint_source_rhs=len(field) * self.reference.s):
            with self.book.span("a27_feedback_adjoint"):
                z = self.solve_A(self.reference.apply_db(load, adjoint=True), state,
                                 adjoint=True)
            with self.book.span("a27_source_aggregation"):
                pullback = (field.conj() * z).sum(-1) / self.reference.sqrt_volume
                result = mass_pack(pullback, 1.)
        return result[0] if single else result


class ReusingTorchRelativeModel(InstrumentedTorchRelativeModel):
    """Use one validated TASK_R32 forward LU for matching state solves.

    The inherited jvp/adjoint/vjp_complex/data_gradient interfaces are kept.
    Book-owner changes revalidate and refactor; material/reference/core changes
    reject stale states.  Each new forward state owns its own LU, even when
    its material happens to equal another state's material.
    """
    cache_forward_lu = True

    def __init__(self, reference):
        super().__init__(reference, "R")
        if (reference.rank != 32 or reference.D_indices.numel()
                or not reference.closure_provenance
                or reference.closure_provenance.get("builder") != "TASK_R32"):
            raise ValueError("A27_FROZEN_NONLOCAL_TASK_R32_REQUIRED")
        if any(_tensor_stamp(getattr(reference, name))[1] is None
               for name in _REFERENCE_BUFFERS):
            raise ValueError("A27_VERSIONED_FROZEN_REFERENCE_REQUIRED_FOR_SAFE_LU_REUSE")

    def _invalidate(self, state):
        if state is not None and state.extra.pop(_CACHE_KEY, None) is not None:
            self.book.counts["a27_forward_lu_cache_invalidations"] += 1

    def _match(self, mass, state):
        try:
            return super()._match(mass, state)
        except ValueError:
            self._invalidate(state)
            raise

    def solve_A(self, rhs, state, *, adjoint=False):
        if state.extra.get("model_id") != id(self):
            self._invalidate(state)
            raise ValueError("TORCH_STATE_MODEL_MISMATCH")
        return super().solve_A(rhs, state, adjoint=adjoint)

    def forward(self, mass, *, recover_current=True):
        state = super().forward(mass, recover_current=recover_current)
        # The frozen forward removes the batch axis on single-state returns.
        # Keep the cache's material snapshot in the public state's coordinates.
        record = state.extra.get(_CACHE_KEY)
        if record is not None:
            record.mass_snapshot = state.mass.detach().clone()
        return state

    def _backward_error(self, matrix, rhs, result, *, adjoint):
        with self.book.span("a27_core_solve_backward_residual_guard",
                            a27_core_solve_residual_audits=int(np.prod(matrix.shape[:-2]))):
            op = matrix.detach().mH if adjoint else matrix.detach()
            x, b = result.detach(), rhs.detach()
            residual = torch.linalg.vector_norm(op @ x - b, dim=(-2, -1))
            scale = (torch.linalg.vector_norm(op, dim=(-2, -1))
                     * torch.linalg.vector_norm(x, dim=(-2, -1))
                     + torch.linalg.vector_norm(b, dim=(-2, -1)))
            positive = scale > 0
            if not torch.isfinite(scale).all():
                raise NumericalFailure("A27_NONFINITE_CORE_BACKWARD_RESIDUAL_SCALE")
            ratio = torch.zeros_like(residual)
            ratio[positive] = residual[positive] / scale[positive]
            if bool(((~positive) & (residual != 0)).any()):
                raise NumericalFailure("A27_NONZERO_CORE_RESIDUAL_AT_ZERO_SCALE")
            tolerance = float(self.reference.config.get("backward_residual_tolerance", 1e-10))
            if (not torch.isfinite(ratio).all()
                    or bool((ratio > tolerance).any())):
                raise NumericalFailure("A27_CORE_SOLVE_BACKWARD_RESIDUAL_EXCEEDS_FROZEN_TOLERANCE")
        return ratio

    def _core_solve(self, rhs, state, *, adjoint=False):
        matrix = state.extra["K"]
        stamp, owner = _reference_stamp(self.reference), _owner_stamp(self.book)
        record = state.extra.get(_CACHE_KEY)
        if record is not None:
            # Equality snapshots also detect an in-place material/core change
            # when a caller presents the same mutated tensor to _match.
            reason = None
            if record.state_id != id(state):
                reason = "STATE"
            elif record.reference_stamp != stamp:
                reason = "REFERENCE"
            elif (record.core_stamp != _tensor_stamp(matrix)
                  or not torch.equal(record.core_snapshot, matrix.detach())):
                reason = "CORE"
            elif (not torch.equal(record.mass_snapshot, state.mass.detach())
                  or not torch.equal(record.h_snapshot, state.extra["h_batch"].detach())):
                reason = "MATERIAL"
            if reason is not None:
                self._invalidate(state)
                raise ValueError("A27_TORCH_STATE_" + reason + "_MISMATCH")
            if record.owner_stamp != owner:
                self._invalidate(state)
                record = None
                state.extra.pop(_GUARD_KEY, None)
        systems, directions = int(np.prod(matrix.shape[:-2])), int(rhs.shape[-1])
        if record is None:
            if state.extra.get(_GUARD_KEY) != (_tensor_stamp(matrix), stamp, owner):
                state.margins["K"] = self._guard(matrix, "K")
            with self.book.span("a27_feedback_K_LU", LU_factorizations=systems,
                                feedback_K_LU_factorizations=systems):
                try:
                    lu, pivots = torch.linalg.lu_factor(matrix)
                except RuntimeError as exc:
                    raise NumericalFailure("TORCH_SOLVE_FAILED:feedback_K") from exc
                if not torch.isfinite(lu).all():
                    raise NumericalFailure("NONFINITE_TORCH_SOLVE:feedback_K")
            record = _ValidatedLU(id(state), stamp, owner, _tensor_stamp(matrix),
                                  matrix.detach().clone(), state.mass.detach().clone(),
                                  state.extra["h_batch"].detach().clone(), lu, pivots)
        else:
            self.book.counts["a27_feedback_K_LU_reuse_calls"] += 1
            self.book.counts["a27_feedback_K_LU_reuse_systems"] += systems
        try:
            label = "torch_feedback_K_adjoint_solve" if adjoint else "torch_feedback_K_solve"
            with self.book.span(label, linear_solve_rhs=systems * directions):
                try:
                    result = torch.linalg.lu_solve(record.lu, record.pivots, rhs,
                                                   adjoint=adjoint)
                except RuntimeError as exc:
                    raise NumericalFailure("TORCH_SOLVE_FAILED:feedback_K") from exc
                if not torch.isfinite(result).all():
                    raise NumericalFailure("NONFINITE_TORCH_SOLVE:feedback_K")
            ratio = self._backward_error(matrix, rhs, result, adjoint=adjoint)
        except BaseException:
            self._invalidate(state)
            raise
        state.extra[_CACHE_KEY] = record
        state.margins["a27_core_" + ("adjoint" if adjoint else "forward") + "_backward_error"] = ratio
        return self.reference._reverse_counter(result, "feedback_K", systems * directions)


class OptimizedFrozenEncoder(FrozenEncoder):
    """Optional, independently selectable A27 encoder engineering changes.

    ``omit_beta=True`` returns beta=None with an explicit omission status.
    Features/base/initializer retain the frozen equations; beta equivalence is
    never claimed for this arm.  All methods also expose named profile spans.
    """
    def __init__(self, background_path, config, book, device="cpu", *,
                 omit_beta=False, reuse_forward_lu=False):
        super().__init__(background_path, config, book, device)
        self._configure(omit_beta, reuse_forward_lu)

    @classmethod
    def from_encoder(cls, encoder, *, omit_beta=False, reuse_forward_lu=False):
        """Resident-cache view; skips no validation done by FrozenEncoder.

        Views share immutable validated reference/operators and must be used
        sequentially because bind_book belongs to that reference.  Every view
        receives fresh model owners and never shares state-dependent LU data.
        """
        if not isinstance(encoder, FrozenEncoder):
            raise TypeError("A27_VALIDATED_FROZEN_ENCODER_REQUIRED")
        value = cls.__new__(cls)
        value.__dict__ = dict(encoder.__dict__)
        value.config = deepcopy(encoder.config)
        value.operator_metrics = deepcopy(encoder.operator_metrics)
        value._configure(omit_beta, reuse_forward_lu)
        return value

    def _configure(self, omit_beta, reuse_forward_lu):
        if type(omit_beta) is not bool or type(reuse_forward_lu) is not bool:
            raise ValueError("A27_EXPLICIT_BOOLEAN_OPTIMIZATION_FLAGS_REQUIRED")
        self.omit_beta, self.reuse_forward_lu = omit_beta, reuse_forward_lu
        self.models = {"L": InstrumentedTorchRelativeModel(self.reference, "L"),
                       "R": (ReusingTorchRelativeModel(self.reference) if reuse_forward_lu
                             else InstrumentedTorchRelativeModel(self.reference, "R"))}

    @torch.no_grad()
    def encode(self, observation, sigma, method):
        method = str(method).upper()
        if method not in {"N1", "N2", "N3"}:
            raise ValueError("A26_ENCODER_METHOD_MUST_BE_N1_N2_N3")
        observation, sigma = self._online_inputs(observation, sigma)
        ref, batch = self.reference, len(observation)
        old_counts = dict(self.book.counts)
        omitted = method == "N2" and self.omit_beta
        with self.book.span("a26_frozen_encoder_" + method, a26_encoded_images=batch):
            with self.book.span("a26_common_boxed_background_adjoint",
                                a26_common_initial_images=batch):
                load = (ref.yb - observation) / ref.yb_norm2
                adjoint = ref.apply_db(load, adjoint=True)
                background_gradient = mass_pack((ref.eb.conj() * adjoint).sum(-1), 1.) / ref.sqrt_volume
                initial = ref.project(mass_pack(ref.chi_b, ref.volume)[None] - ref.step * background_gradient)
            kind = "R" if method == "N2" else "L"
            required = (("R",) if omitted else ("L", "R")) if method == "N2" else ("L",)
            states = {key: self.models[key](initial, recover_current=False) for key in required}
            residuals = {key: self._forward_residual(state) for key, state in states.items()}
            state = states[kind]
            gradient = self.models[kind].data_gradient(initial, observation, state)
            with self.book.span("a27_fixed_material_update"):
                base = ref.project(initial - ref.step * gradient)
            if method == "N3":
                image = mass_unpack(-ref.step * background_gradient, ref.volume)
            else:
                self.prepare_observable()
                with self.book.span("a26_rank16_observation_feature",
                                    a26_observable_source_rhs=batch * ref.s):
                    w_observable = self.observable_right @ (
                        (self.observable_left.mH @ (observation - ref.yb))
                        / self.observable_singular[:, None])
                    with self.book.span("a27_observation_material_ratio_and_guard"):
                        image = mass_unpack(ref.material_ratio(
                            w_observable, ref.eb[None].expand(batch, -1, -1)), ref.volume)
            with self.book.span("a27_feature_assembly"):
                delta = mass_unpack(base, ref.volume) - ref.chi_b
                background = ref.chi_b[None].expand(batch, -1)
                confidence = self.confidence_diagonal[None].expand(batch, -1)
                n = ref.n
                features = torch.stack((delta.real, delta.imag, image.real, image.imag,
                                        gradient[:, :n], gradient[:, n:],
                                        state.field.abs().square().mean(-1),
                                        state.w.abs().square().mean(-1),
                                        background.real, background.imag,
                                        confidence[:, :n], confidence[:, n:]), dim=1)
                features = features.reshape(batch, len(FEATURE_NAMES), ref.grid, ref.grid).to(torch.float32)
            with self.book.span("a27_encoder_output_guard"):
                if not torch.isfinite(features).all() or not torch.isfinite(base).all():
                    raise NumericalFailure("A26_NONFINITE_FROZEN_ENCODER_OUTPUT")
            with self.book.span("a27_beta_diagnostic"):
                beta = (None if omitted else torch.full((batch,), float("nan"),
                        dtype=torch.float64, device=self.device))
                noisy = sigma > 0
                if method == "N2" and not omitted:
                    gap = torch.linalg.vector_norm(states["R"].data - states["L"].data, dim=(-2, -1))
                    beta[noisy] = gap[noisy] / (sigma[noisy] * math.sqrt(ref.q * ref.s))
            with self.book.span("a27_diagnostics_logging"):
                status = ("OMITTED_NO_DIAGNOSTIC_L_FORWARD" if omitted else
                          "COMPUTED_FOR_N2" if method == "N2" else "NO_FEEDBACK_NOT_COMPUTED")
                group = "OMITTED" if omitted else "NOISY" if method == "N2" else "NO_FEEDBACK"
                diagnostics = {"method": method, "physical_model": kind,
                    "feature_names": list(FEATURE_NAMES),
                    "beta_group": [group if value else "NOISE_FREE" for value in noisy.cpu().tolist()],
                    "beta_status": status, "beta_noise_free": (~noisy).cpu().tolist(),
                    "relative_forward_residual": _diagnostic_values(residuals),
                    "core_margins": _diagnostic_values(states["R"].margins) if method == "N2" else {},
                    "canonical_current_recovered": False, "background_id": self.background_id,
                    "confidence_interpretation": self.operator_metrics["confidence_interpretation"],
                    "a27_optimization": {"reuse_forward_lu": self.reuse_forward_lu and method == "N2",
                        "diagnostic_beta_omitted": omitted, "beta_comparison_eligible": not omitted},
                    "omitted_diagnostics": ["beta", "matched_L_forward", "relative_forward_residual.L"]
                        if omitted else []}
                diagnostics["physical_counts"] = {
                    key: int(value - old_counts.get(key, 0)) for key, value in self.book.counts.items()
                    if value != old_counts.get(key, 0)}
        return {"features": features.detach(), "base_mass": base.detach(),
                "initial_mass": initial.detach(), "beta": None if beta is None else beta.detach(),
                "diagnostics": diagnostics}
