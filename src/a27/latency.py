"""Outer-synchronized native timing and separate named-stage A27 profiles.

No data selection, checkpoints, jobs, labels, or full audits are opened here.
The parent supplies raw observations, the existing book, and a decode callable
that owns checkpoint normalization.  Native timing and instrumented profiles
must be distinct runs: stage-event overhead is never called native latency.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import time

import numpy as np
import torch

from a25.torch_physics import mass_unpack
from .fast_physics import OptimizedFrozenEncoder


@dataclass(frozen=True)
class ProfileArm:
    name: str
    method: str
    omit_beta: bool = False
    reuse_forward_lu: bool = False


PROFILE_ARMS = (
    ProfileArm("N1_BASELINE", "N1"),
    ProfileArm("N2_BASELINE", "N2"),
    ProfileArm("N3_BASELINE", "N3"),
    ProfileArm("N2_NO_BETA_ONLY", "N2", omit_beta=True),
    ProfileArm("N2_LU_REUSE_ONLY", "N2", reuse_forward_lu=True),
    ProfileArm("N2_COMBINED", "N2", omit_beta=True, reuse_forward_lu=True),
)


def build_profile_arms(encoder):
    """Sequential resident views of the same already validated A26 cache."""
    return {arm.name: (arm, OptimizedFrozenEncoder.from_encoder(
        encoder, omit_beta=arm.omit_beta, reuse_forward_lu=arm.reuse_forward_lu))
        for arm in PROFILE_ARMS}


def _synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class OnlineProfileBook:
    """In-memory counts/stages backed by the parent's checks and ownership.

    No inherited per-span synchronize, disk append or budget/lock read is
    called, matching frozen A26 CountOnlyOnlineBook.  Numerical guard decisions
    in the physics/encoder are untouched.  Existing budget/owner checks run
    before and after the outer image; the real book receives all action counts,
    including failed actions, at that boundary.
    """
    def __init__(self, book, device, *, stages=False):
        self.parent_book, self.device = book, torch.device(device)
        self.counts = Counter()
        self.stages = bool(stages)
        self.records, self.stack = [], []
        self.meter_host_seconds = 0.
        self.owner_id = getattr(book, "owner_id", None)
        self.metadata = getattr(book, "metadata", {})

    def check(self):
        # Same granularity as A26 CountOnlyOnlineBook: profile_inference owns
        # the paid outer checks.  This does not bypass any tensor guard.
        pass

    def _event(self):
        if not self.stages or self.device.type != "cuda":
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream(self.device))
        return event

    @contextmanager
    def span(self, label, **counts):
        self.check()
        self.counts.update(counts)
        if not self.stages:
            yield None
            return
        preparation = time.perf_counter()
        index = len(self.records)
        record = {"label": label, "counters": dict(counts),
                  "parent": self.stack[-1] if self.stack else None,
                  "children": [], "status": "FAILED",
                  "start_event": self._event()}
        if self.stack:
            self.records[self.stack[-1]]["children"].append(index)
        self.records.append(record)
        self.stack.append(index)
        self.meter_host_seconds += time.perf_counter() - preparation
        start = time.perf_counter()
        try:
            yield record
            record["status"] = "OK"
        except BaseException as exc:
            record["error"] = type(exc).__name__ + ": " + str(exc)
            raise
        finally:
            end = time.perf_counter()
            finish = time.perf_counter()
            record["host_inclusive_seconds"] = end - start
            record["end_event"] = self._event()
            self.stack.pop()
            self.meter_host_seconds += time.perf_counter() - finish

    def resolved_stages(self):
        """Call only after the outer synchronize; does not synchronize itself."""
        rows = []
        for record in self.records:
            gpu = (record["start_event"].elapsed_time(record["end_event"]) / 1000.
                   if record["start_event"] is not None else None)
            rows.append({key: value for key, value in record.items()
                         if key not in {"start_event", "end_event"}})
            rows[-1]["gpu_event_inclusive_seconds"] = gpu
        for row in rows:
            row["host_exclusive_seconds"] = max(0., row["host_inclusive_seconds"] - sum(
                rows[i]["host_inclusive_seconds"] for i in row["children"]))
            row["gpu_event_exclusive_seconds"] = (None if row["gpu_event_inclusive_seconds"] is None
                else max(0., row["gpu_event_inclusive_seconds"] - sum(
                    rows[i]["gpu_event_inclusive_seconds"] for i in row["children"])))
        return rows


def _host_result_observer():
    """Count blocking-result opportunities only in a separate stages run.

    Dispatch forwards the exact operator with unchanged arguments/results.
    The observer never synchronizes and does not classify a stream interval
    as kernel compute.  CUDA result/copy calls are opportunities for host wait,
    not evidence that each one incurred nonzero waiting time.
    """
    from torch.utils._python_dispatch import TorchDispatchMode

    class HostResultObserver(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.calls = Counter()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            name = str(func)
            if name.startswith("aten._local_scalar_dense."):
                prefix = "cuda" if args and isinstance(args[0], torch.Tensor) and args[0].device.type == "cuda" else "cpu"
                self.calls[prefix + "_scalar_host_result_calls"] += 1
            elif name.startswith("aten.equal."):
                prefix = "cuda" if args and isinstance(args[0], torch.Tensor) and args[0].device.type == "cuda" else "cpu"
                self.calls[prefix + "_tensor_equal_host_result_calls"] += 1
            elif name.startswith("aten._to_copy."):
                target = kwargs.get("device")
                if (target is not None and torch.device(target).type == "cpu"
                        and args and isinstance(args[0], torch.Tensor)
                        and args[0].device.type == "cuda"):
                    self.calls["cuda_to_cpu_copy_calls"] += 1
            elif name.startswith("aten.copy_."):
                if (len(args) > 1 and isinstance(args[0], torch.Tensor)
                        and isinstance(args[1], torch.Tensor)
                        and args[0].device.type == "cpu" and args[1].device.type == "cuda"):
                    self.calls["cuda_to_cpu_copy_calls"] += 1
            return func(*args, **kwargs)

    return HostResultObserver()


_NAMED_STAGES = {
    "input_transfer": {"a26_online_observation_transfer"},
    "background_adjoint_initializer": {"a26_common_boxed_background_adjoint"},
    "core_assembly": {"a27_core_assembly"},
    "LU_factorization": {"a27_feedback_K_LU"},
    "core_solve": {"torch_feedback_K_solve", "torch_feedback_K_adjoint_solve"},
    "feedback_adjoint": {"a27_feedback_adjoint"},
    "source_aggregation": {"a27_source_aggregation"},
    "material_ratio_and_denominator_guard": {"a27_observation_material_ratio_and_guard"},
    "physical_base_update": {"a27_fixed_material_update"},
    "features": {"a27_feature_assembly"},
    "diagnostic_beta": {"a27_beta_diagnostic"},
    "guards": {"torch_core_guard_K", "torch_core_guard_P",
               "a26_relative_forward_residual_guard",
               "a27_core_solve_backward_residual_guard", "a27_encoder_output_guard"},
    "diagnostics_logging": {"a27_diagnostics_logging"},
    "decoder": {"a27_decoder"},
    "image_transfer": {"a27_image_transfer"},
}


def _named_stage_summary(rows, *, method, reuse, stages, cuda):
    result = {}
    for name, labels in _NAMED_STAGES.items():
        matching = [row for row in rows if row["label"] in labels]
        status, reason = "MEASURED", None
        if not stages:
            status, reason = "NOT_RUN", "NATIVE_RUN_HAS_NO_STAGE_TIMING"
        elif not matching:
            status = "NOT_APPLICABLE" if (method != "N2" and name in {
                "core_assembly", "LU_factorization", "core_solve"}) else "NOT_SEPARATELY_INSTRUMENTED"
            reason = ("NO_R_CORE_FOR_L_METHOD" if status == "NOT_APPLICABLE" else
                      "BASELINE_LU_AND_SOLVE_COMBINED" if name == "LU_factorization" and not reuse else
                      "N3_USES_BACKGROUND_BP_WITHOUT_OBSERVATION_MATERIAL_RATIO" if
                      method == "N3" and name == "material_ratio_and_denominator_guard" else
                      "FROZEN_A26_CALLER_HAS_NO_NAMED_SPAN")
            if method == "N3" and name == "material_ratio_and_denominator_guard":
                status = "NOT_APPLICABLE"
        result[name] = {"status": status, "reason": reason,
            "calls": len(matching),
            "host_inclusive_seconds": sum(row["host_inclusive_seconds"] for row in matching)
                if matching else None,
            "gpu_event_inclusive_seconds": sum(row["gpu_event_inclusive_seconds"] for row in matching)
                if matching and cuda else None,
            "gpu_event_status": "MEASURED_CURRENT_STREAM_INTERVAL" if matching and cuda else
                "NOT_APPLICABLE_CPU" if not cuda else "NOT_MEASURED"}
    if method == "N2" and not reuse:
        result["core_solve"]["scope"] = "TORCH_LINALG_SOLVE_INCLUDES_LU_AND_SOLVE"
    elif method == "N2":
        result["core_solve"]["scope"] = "TORCH_LINALG_LU_SOLVE_USES_SEPARATELY_RECORDED_LU"
    return result


def profile_inference(encoder, decode, observation, sigma, method, *, book=None,
                      timing_mode="native", transfer=None, metadata=None):
    """Measure one raw-measurement -> encoder -> decode -> host image call.

    ``decode(encoded)`` returns the decoder's material mass tensor and should
    include checkpoint normalization.  Optional ``transfer(mass, encoder)``
    returns the host image; default output is a complex NumPy [batch, cells]
    array.  No intermediate synchronization is inserted.  CUDA events in a
    separate stages run measure current-stream intervals, including possible
    host-induced idle; they are neither pure kernel time nor native full wall.
    """
    if timing_mode not in {"native", "stages"}:
        raise ValueError("A27_TIMING_MODE_MUST_BE_NATIVE_OR_STAGES")
    if not callable(decode):
        raise TypeError("A27_CHECKPOINT_NORMALIZATION_AND_DECODE_CALLABLE_REQUIRED")
    method = str(method).upper()
    if method not in {"N1", "N2", "N3"}:
        raise ValueError("A26_ENCODER_METHOD_MUST_BE_N1_N2_N3")
    old_book = encoder.book
    book = old_book if book is None else book
    outer_check_start = time.perf_counter()
    book.check()
    outer_check_seconds = time.perf_counter() - outer_check_start
    meter = OnlineProfileBook(book, encoder.device, stages=timing_mode == "stages")
    cuda = torch.device(encoder.device).type == "cuda"
    encoded = mass = image = None
    status = "FAILED"
    observer = _host_result_observer() if meter.stages else None
    encoder.bind_book(meter)
    start = None
    try:
        _synchronize(encoder.device)
        start = time.perf_counter()
        with torch.no_grad(), observer if observer is not None else nullcontext():
            with meter.span("a27_total_encoder"):
                encoded = encoder.encode(observation, sigma, method)
            with meter.span("a27_decoder"):
                mass = decode(encoded)
            with meter.span("a27_image_transfer"):
                image = (mass_unpack(mass, encoder.reference.volume).detach().cpu().numpy()
                         if transfer is None else transfer(mass, encoder))
        status = "OK"
    finally:
        # Includes end synchronization in the native elapsed interval.  CPU
        # transfers inside the encoder/output are required operations, not
        # profiling barriers; all numerical guard scalar decisions remain.
        try:
            _synchronize(encoder.device)
        finally:
            end = time.perf_counter()
            encoder.bind_book(old_book)
            book.counts.update(meter.counts)
        elapsed = None if start is None else end - start
        outer_check_start = time.perf_counter()
        book.check()
        outer_check_seconds += time.perf_counter() - outer_check_start
        rows = meter.resolved_stages() if meter.stages else []
        timing = {"status": status, "timing_mode": timing_mode, "method": method,
            "native_full_inference_seconds": elapsed if not meter.stages else None,
            "instrumented_full_inference_seconds": elapsed if meter.stages else None,
            "stage_rows": rows,
            "named_stages": _named_stage_summary(rows, method=method,
                reuse=bool(getattr(encoder, "reuse_forward_lu", False)),
                stages=meter.stages, cuda=cuda),
            "action_counts": dict(meter.counts),
            "stage_meter_host_seconds": meter.meter_host_seconds if meter.stages else None,
            "outer_budget_check_host_seconds": outer_check_seconds,
            "budget_check_granularity": "OUTER_BEFORE_AND_AFTER_IMAGE_AS_FROZEN_A26",
            "per_span_budget_or_file_lock_checks": False,
            "host_synchronization_summary": {
                "status": "COUNTED_TORCH_DISPATCH_OPPORTUNITIES" if observer is not None else
                    "NOT_RUN_NATIVE_TIMING_HAS_NO_DISPATCH_OBSERVER",
                "operation_counts": dict(observer.calls) if observer is not None else None,
                "CUDA_host_wait_opportunities": sum(value for key, value in observer.calls.items()
                    if key.startswith("cuda_")) if cuda and observer is not None else
                    0 if not cuda else None,
                "actual_host_wait_seconds": None,
                "wait_time_status": "NOT_MEASURED; OPPORTUNITY_COUNT_IS_NOT_A_WAIT_DURATION",
                "outer_device_synchronizations": 2 if cuda else 0,
                "observer_only_in_separate_stage_run": True},
            "synchronization": "OUTER_BOUNDARIES_ONLY; EXISTING_NUMERICAL_GUARD_AND_HOST_TRANSFER_DECISIONS_RETAINED",
            "GPU_event_scope": "CURRENT_STREAM_INTERVALS_CAN_INCLUDE_HOST_INDUCED_IDLE" if cuda else
                "NOT_APPLICABLE_CPU",
            "stage_intervals_overlap": True,
            "stage_sum_is_not_native_latency": True,
            "new_labels": 0, "offline_full_audit_included": False,
            **dict(metadata or {})}
        append = getattr(book, "append", None)
        if append is not None:
            append({"event": "A27_ONLINE_INFERENCE_PROFILE", "status": status,
                    "timing_mode": timing_mode, "method": method,
                    "wall_seconds": elapsed, "counters": dict(meter.counts),
                    **dict(metadata or {})})
    return {"encoded": encoded, "mass": mass, "image": image, "timing": timing}


def compare_encoder_outputs(reference, candidate, *, base_rtol=1e-11,
                            base_atol=1e-12, feature_rtol=2e-6, feature_atol=1e-7):
    """Numerical equivalence receipt; beta omission stays a distinct result."""
    result = {}
    for key in ("features", "base_mass", "initial_mass"):
        left, right = reference[key], candidate[key]
        tolerance = (feature_rtol, feature_atol) if key == "features" else (base_rtol, base_atol)
        torch.testing.assert_close(left, right, rtol=tolerance[0], atol=tolerance[1])
        result[key] = {"max_abs_difference": float((left - right).abs().max().detach().cpu()),
                       "rtol": tolerance[0], "atol": tolerance[1], "passed": True}
    if candidate["beta"] is None:
        result["beta"] = {"status": "EXPLICITLY_OMITTED_NOT_COMPARED", "equivalence_claimed": False}
    else:
        torch.testing.assert_close(reference["beta"], candidate["beta"], rtol=base_rtol,
                                   atol=base_atol, equal_nan=True)
        result["beta"] = {"status": "COMPARED_WITH_NAN_GROUPS", "passed": True}
    return result
