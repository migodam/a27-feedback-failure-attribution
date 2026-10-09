"""Tiny known-background algebra/profile tests in the parent's CPU ledger.

No split, labels, locked data, GPU device, checkpoint or paid job is opened.
The portable finite-array fixture is the unchanged A26 test fixture.
"""
from __future__ import annotations

from copy import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from a24.contracts import NumericalFailure, get_ledger
from a25.torch_physics import TorchRelativeModel, mass_pack, pack_data
from a26.encoder import FrozenEncoder
from a27.fast_physics import OptimizedFrozenEncoder, ReusingTorchRelativeModel
from a27.latency import (OnlineProfileBook, PROFILE_ARMS, build_profile_arms,
                         compare_encoder_outputs, profile_inference)
from test_a26_encoder_decoder import _tiny_cache


class A27FastPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.book = get_ledger()
        if cls.book.device != "cpu":
            raise RuntimeError("A27_TINY_TESTS_REQUIRE_ACTIVE_CPU_LEDGER")
        torch.set_num_threads(1)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        with self.book.span("a27_tiny_portable_cache_fixture", a27_tiny_cache_fixtures=1):
            self.path, _, self.source_config = _tiny_cache(self.root)
        self.config = {"protocol_id": "A27-TINY-FIXTURE-v1",
                       "source_commit": "frozen-a26-fixture",
                       "observable_rank": 16, "confidence_relative_cut": .05,
                       "closure": "TASK_R32"}
        self.encoder = FrozenEncoder(self.path, self.config, self.book, "cpu")
        self.reference = self.encoder.reference

    def tearDown(self):
        self.reference.bind_book(self.book)
        self.directory.cleanup()

    def _mass(self, *, batch=2):
        mass = mass_pack(self.reference.chi_b + .2 + .015j, self.reference.volume)
        return mass[None].expand(batch, -1).clone()

    def _observation(self):
        return torch.stack((self.reference.yb * (1.12 + .025j),
                            self.reference.yb * (.93 - .01j)))

    def _counted(self, callable_):
        old = dict(self.book.counts)
        result = callable_()
        return result, {key: int(value - old.get(key, 0))
                        for key, value in self.book.counts.items()
                        if value != old.get(key, 0)}

    def test_forward_adjoint_outputs_and_actual_factor_rhs_counts(self):
        mass, obs = self._mass(), self._observation()
        baseline = TorchRelativeModel(self.reference, "R")
        fast = ReusingTorchRelativeModel(self.reference)
        expected, baseline_forward = self._counted(lambda: baseline(mass, recover_current=False))
        actual, fast_forward = self._counted(lambda: fast(mass, recover_current=False))
        for key in ("data", "field", "w"):
            torch.testing.assert_close(getattr(expected, key), getattr(actual, key),
                                       rtol=1e-12, atol=1e-13)
        expected_gradient, baseline_adjoint = self._counted(
            lambda: baseline.data_gradient(mass, obs, expected))
        actual_gradient, fast_adjoint = self._counted(
            lambda: fast.data_gradient(mass, obs, actual))
        torch.testing.assert_close(expected_gradient, actual_gradient, rtol=1e-12, atol=1e-13)
        self.assertEqual(baseline_forward["feedback_K_LU_factorizations"], 2)
        self.assertEqual(baseline_adjoint["feedback_K_LU_factorizations"], 2)
        self.assertEqual(fast_forward["feedback_K_LU_factorizations"], 2)
        self.assertEqual(fast_adjoint.get("feedback_K_LU_factorizations", 0), 0)
        self.assertEqual(fast_adjoint["a27_feedback_K_LU_reuse_systems"], 2)
        self.assertEqual(baseline_forward["linear_solve_rhs"], fast_forward["linear_solve_rhs"])
        self.assertEqual(baseline_adjoint["linear_solve_rhs"], fast_adjoint["linear_solve_rhs"])
        self.assertTrue(torch.isfinite(actual.margins["a27_core_adjoint_backward_error"]).all())

    def test_real_mass_jvp_vjp_and_higher_derivative_remain_matched(self):
        mass = self._mass().requires_grad_()
        obs = self._observation()
        fast = ReusingTorchRelativeModel(self.reference)
        state = fast(mass, recover_current=False)
        objective = .5 * (state.data - obs).abs().square().sum() / self.reference.yb_norm2
        automatic = torch.autograd.grad(objective, mass, create_graph=True)[0]
        explicit = fast.data_gradient(mass, obs, state)
        torch.testing.assert_close(explicit, automatic, rtol=1e-11, atol=1e-12)
        generator = torch.Generator().manual_seed(27001)
        direction = torch.randn(mass.shape, dtype=torch.float64, generator=generator)
        load = obs * (.2 + .07j)
        tangent = fast.jvp(mass, direction, state)
        pullback = fast.adjoint(mass, pack_data(load), state)
        torch.testing.assert_close((tangent * pack_data(load)).sum(),
                                   (direction * pullback).sum(), rtol=1e-11, atol=1e-12)
        hessian = torch.autograd.grad((explicit * direction).sum(), mass)[0]
        baseline = TorchRelativeModel(self.reference, "R")
        second_mass = mass.detach().clone().requires_grad_()
        baseline_gradient = baseline.data_gradient(second_mass, obs)
        baseline_hessian = torch.autograd.grad((baseline_gradient * direction).sum(), second_mass)[0]
        torch.testing.assert_close(hessian, baseline_hessian, rtol=1e-10, atol=1e-11)

    def test_single_state_shape_and_new_state_never_share_lu(self):
        fast, mass = ReusingTorchRelativeModel(self.reference), self._mass(batch=1)[0]
        first, counts1 = self._counted(lambda: fast(mass, recover_current=False))
        second, counts2 = self._counted(lambda: fast(mass, recover_current=False))
        self.assertEqual(counts1["feedback_K_LU_factorizations"], 1)
        self.assertEqual(counts2["feedback_K_LU_factorizations"], 1)
        first_gradient = fast.data_gradient(mass, self._observation()[0], first)
        second_gradient = fast.data_gradient(mass, self._observation()[0], second)
        torch.testing.assert_close(first_gradient, second_gradient, rtol=0., atol=0.)
        clone = copy(second)
        clone.extra = dict(second.extra)
        with self.assertRaisesRegex(ValueError, "STATE_STATE_MISMATCH"):
            fast.data_gradient(mass, self._observation()[0], clone)

    def test_material_core_reference_and_model_changes_invalidate(self):
        fast, mass = ReusingTorchRelativeModel(self.reference), self._mass()
        material_state = fast(mass, recover_current=False)
        material_state.mass[0, 0] += .001
        with self.assertRaisesRegex(ValueError, "STATE_MATERIAL_MISMATCH"):
            fast.data_gradient(material_state.mass, self._observation(), material_state)
        self.assertNotIn("_a27_validated_forward_lu", material_state.extra)
        mass = self._mass()
        core_state = fast(mass, recover_current=False)
        core_state.extra["K"][0, 0, 0] += .001
        with self.assertRaisesRegex(ValueError, "STATE_CORE_MISMATCH"):
            fast.data_gradient(mass, self._observation(), core_state)
        self.assertNotIn("_a27_validated_forward_lu", core_state.extra)
        reference_state = fast(mass, recover_current=False)
        self.reference.Ug[0, 0] += .001
        with self.assertRaisesRegex(ValueError, "STATE_REFERENCE_MISMATCH"):
            fast.data_gradient(mass, self._observation(), reference_state)
        self.assertNotIn("_a27_validated_forward_lu", reference_state.extra)
        owner_state = fast(mass, recover_current=False)
        other = ReusingTorchRelativeModel(self.reference)
        with self.assertRaisesRegex(ValueError, "TORCH_STATE_MISMATCH"):
            other.data_gradient(mass, self._observation(), owner_state)
        self.assertNotIn("_a27_validated_forward_lu", owner_state.extra)

    def test_cost_book_owner_change_revalidates_and_refactors(self):
        fast, mass = ReusingTorchRelativeModel(self.reference), self._mass()
        state = fast(mass, recover_current=False)
        expected = fast.data_gradient(mass, self._observation(), state)
        replacement = OnlineProfileBook(self.book, "cpu")
        self.reference.bind_book(replacement)
        try:
            actual = fast.data_gradient(mass, self._observation(), state)
            self.assertEqual(replacement.counts["feedback_K_LU_factorizations"], 2)
            self.assertEqual(replacement.counts["core_guards"], 2)
            self.assertEqual(replacement.counts["a27_feedback_K_LU_reuse_systems"], 0)
            self.assertEqual(replacement.counts["a27_forward_lu_cache_invalidations"], 1)
        finally:
            self.book.counts.update(replacement.counts)
            self.reference.bind_book(self.book)
        torch.testing.assert_close(expected, actual, rtol=0., atol=0.)

    def test_domain_singular_core_and_nonfinite_solve_fail_without_fallback(self):
        fast = ReusingTorchRelativeModel(self.reference)
        mass = self._mass()
        nonfinite = mass.clone()
        nonfinite[0, 0] = float("nan")
        with self.assertRaisesRegex(NumericalFailure, "NONFINITE_TORCH_MATERIAL"):
            fast(nonfinite, recover_current=False)
        outside = mass_pack(self.reference.chi_b + 5., self.reference.volume)[None]
        with self.assertRaisesRegex(NumericalFailure, "MATERIAL_DOMAIN_VIOLATION"):
            fast(outside, recover_current=False)
        with patch("torch.linalg.lu_solve", side_effect=lambda lu, piv, rhs, **kw:
                   torch.full_like(rhs, complex(float("nan"), 0.))):
            with self.assertRaisesRegex(NumericalFailure, "NONFINITE_TORCH_SOLVE"):
                fast(mass, recover_current=False)
        self.reference.Ug.zero_()
        self.reference.Vg.zero_()
        self.reference.Ug[0, 0] = 1.
        self.reference.Vg[0, 0] = 1.
        singular_mass = mass_pack(self.reference.chi_b + 1., self.reference.volume)[None]
        with patch("torch.linalg.lu_factor", side_effect=AssertionError("must reject before LU")), \
             patch("torch.linalg.pinv", side_effect=AssertionError("no pseudoinverse")):
            with self.assertRaisesRegex(NumericalFailure, "UNSAFE_TORCH_CORE"):
                fast(singular_mass, recover_current=False)

    def test_adjoint_backward_residual_guard_rejects_finite_corrupt_solve(self):
        fast, mass = ReusingTorchRelativeModel(self.reference), self._mass()
        state = fast(mass, recover_current=False)
        with patch("torch.linalg.lu_solve", side_effect=lambda lu, piv, rhs, **kw:
                   torch.zeros_like(rhs)):
            with self.assertRaisesRegex(NumericalFailure, "CORE_SOLVE_BACKWARD_RESIDUAL"):
                fast.data_gradient(mass, self._observation(), state)
        self.assertNotIn("_a27_validated_forward_lu", state.extra)

    def test_three_engineering_arms_equal_features_base_and_explicit_beta_contract(self):
        obs = self._observation()
        baseline = self.encoder.encode(obs, [.02, 0.], "N2")
        for omitted, reuse in ((True, False), (False, True), (True, True)):
            with self.subTest(omit_beta=omitted, reuse_forward_lu=reuse):
                candidate = OptimizedFrozenEncoder.from_encoder(
                    self.encoder, omit_beta=omitted, reuse_forward_lu=reuse)
                actual = candidate.encode(obs, [.02, 0.], "N2")
                receipt = compare_encoder_outputs(baseline, actual)
                torch.testing.assert_close(baseline["features"], actual["features"], rtol=0., atol=0.)
                torch.testing.assert_close(baseline["initial_mass"], actual["initial_mass"], rtol=0., atol=0.)
                self.assertTrue(receipt["base_mass"]["passed"])
                self.assertEqual(actual["diagnostics"]["canonical_current_recovered"], False)
                counts = actual["diagnostics"]["physical_counts"]
                self.assertEqual(counts["relative_forward_calls"], 2 if omitted else 4)
                self.assertEqual(counts["a26_forward_residual_audits"], 2 if omitted else 4)
                self.assertEqual(counts["feedback_K_LU_factorizations"], 2 if reuse else 4)
                if omitted:
                    self.assertIsNone(actual["beta"])
                    self.assertEqual(actual["diagnostics"]["beta_status"],
                                     "OMITTED_NO_DIAGNOSTIC_L_FORWARD")
                    self.assertEqual(receipt["beta"]["equivalence_claimed"], False)
                    self.assertEqual(set(actual["diagnostics"]["relative_forward_residual"]), {"R"})
                else:
                    torch.testing.assert_close(actual["beta"], baseline["beta"],
                                               rtol=1e-12, atol=1e-13, equal_nan=True)

    def test_common_baseline_spans_and_no_beta_skip_only_unused_L(self):
        obs = self._observation()
        view = OptimizedFrozenEncoder.from_encoder(self.encoder, omit_beta=True)
        with patch.object(view.models["L"], "forward", side_effect=AssertionError("unused L")), \
             patch.object(view.reference, "canonical_current", side_effect=AssertionError("no current")):
            result = view.encode(obs, .02, "N2")
        self.assertIsNone(result["beta"])
        for method in ("N1", "N2", "N3"):
            baseline = self.encoder.encode(obs, .02, method)
            instrumented = OptimizedFrozenEncoder.from_encoder(self.encoder).encode(obs, .02, method)
            compare_encoder_outputs(baseline, instrumented)
        # The unchanged residual and dark-material-ratio guards remain live.
        state = view.models["R"](result["initial_mass"], recover_current=False)
        state.w[0, 0, 0] += 1.
        with self.assertRaisesRegex(NumericalFailure, "FORWARD_RESIDUAL"):
            view._forward_residual(state)
        dark = self.reference.eb[None].clone()
        dark[:, -1] = 0.
        with self.assertRaisesRegex(NumericalFailure, "UNSAFE_SHARED_MATERIAL_RATIO"):
            self.reference.material_ratio(torch.ones_like(dark), dark)

    def test_native_and_stages_are_separate_and_have_no_per_span_sync(self):
        arms = build_profile_arms(self.encoder)
        self.assertEqual(set(arms), {arm.name for arm in PROFILE_ARMS})
        arm, view = arms["N2_COMBINED"]
        decode = lambda encoded: encoded["base_mass"]
        with patch.object(self.book, "synchronize", side_effect=AssertionError("per-span sync forbidden")), \
             patch("torch.cuda.Event", side_effect=AssertionError("CPU must not create CUDA events")), \
             patch("torch.cuda.synchronize", side_effect=AssertionError("CPU must not launch CUDA")):
            native = profile_inference(view, decode, self._observation(), .02, arm.method,
                                       book=self.book, timing_mode="native")
            stages = profile_inference(view, decode, self._observation(), .02, arm.method,
                                       book=self.book, timing_mode="stages")
        np.testing.assert_array_equal(native["image"], stages["image"])
        self.assertIs(self.reference.book, self.book)
        self.assertGreater(native["timing"]["native_full_inference_seconds"], 0.)
        self.assertIsNone(native["timing"]["instrumented_full_inference_seconds"])
        self.assertIsNone(stages["timing"]["native_full_inference_seconds"])
        self.assertGreater(stages["timing"]["instrumented_full_inference_seconds"], 0.)
        for name in ("core_assembly", "LU_factorization", "feedback_adjoint", "source_aggregation",
                     "material_ratio_and_denominator_guard", "guards", "diagnostics_logging"):
            self.assertEqual(stages["timing"]["named_stages"][name]["status"], "MEASURED")
            self.assertIsNone(stages["timing"]["named_stages"][name]["gpu_event_inclusive_seconds"])
        self.assertEqual(native["timing"]["action_counts"]["feedback_K_LU_factorizations"], 2)
        self.assertEqual(native["timing"]["action_counts"]["linear_solve_rhs"], 12)
        _, baseline_view = arms["N2_BASELINE"]
        baseline_profile = profile_inference(baseline_view, decode, self._observation(), .02, "N2",
                                             book=self.book, timing_mode="stages")
        self.assertEqual(baseline_profile["timing"]["named_stages"]["LU_factorization"]["reason"],
                         "BASELINE_LU_AND_SOLVE_COMBINED")

    def test_failed_decode_restores_book_and_charges_completed_physics(self):
        view = OptimizedFrozenEncoder.from_encoder(self.encoder, reuse_forward_lu=True)
        old = dict(self.book.counts)
        def failed_decode(encoded):
            raise RuntimeError("A27_TEST_DECODE_FAILURE")
        with patch.object(self.book, "append", wraps=self.book.append) as append:
            with self.assertRaisesRegex(RuntimeError, "A27_TEST_DECODE_FAILURE"):
                profile_inference(view, failed_decode, self._observation(), .02, "N2", book=self.book)
            last = append.call_args_list[-1].args[0]
        self.assertIs(self.reference.book, self.book)
        self.assertEqual(self.book.counts["feedback_K_LU_factorizations"]
                         - old.get("feedback_K_LU_factorizations", 0), 2)
        self.assertEqual(last["status"], "FAILED")
        self.assertEqual(last["event"], "A27_ONLINE_INFERENCE_PROFILE")
