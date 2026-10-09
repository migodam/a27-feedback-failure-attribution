"""A27 accounting guards, using the frozen A26 synthetic receipt fixtures."""
from pathlib import Path
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from a27.budget import (BudgetExceeded, BudgetLedger, CPU_LIMIT_SECONDS,
                        GPU_LIMIT_SECONDS, budget_history, charged_job,
                        owned_file_lock, write_json, SHARED_GPU_LOCK)

class BudgetSafetyTests(unittest.TestCase):
    def test_a27_history_is_independent_and_failed_costs_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            for job, device, wall, process, gpu, status in (("failed", "cpu", 8., 3., 0., "FAILED"),
                                                           ("cuda", "cuda", 12., 2., 12., "COMPLETE")):
                write_json(Path(directory)/"results/a27/jobs"/job/"job_receipt.json",
                           dict(job=job, device=device, wall_seconds=wall, process_cpu_seconds=process,
                                gpu_occupation_seconds=gpu, status=status, durable_reservations={}))
            old = Path(directory)/"results/a25/jobs/old/job_receipt.json"
            write_json(old, {"process_cpu_seconds": 99999., "durable_reservations": {"new_full_forward_labels": 456}})
            saved = old.read_bytes()
            a26_old = Path(directory)/"results/a26/jobs/old/job_receipt.json"
            write_json(a26_old, {"process_cpu_seconds": 99999.,
                                "durable_reservations": {"new_full_forward_labels": 456}})
            a26_saved = a26_old.read_bytes()
            history = budget_history(directory)
            self.assertEqual(history["cpu_seconds"], 10.)
            self.assertEqual(history["gpu_seconds"], 12.)
            self.assertEqual(old.read_bytes(), saved)
            self.assertEqual(a26_old.read_bytes(), a26_saved)
            self.assertEqual(len(history["receipts"]), 2)

    def test_invalid_cuda_occupation_receipt_cannot_erase_gpu_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            write_json(Path(directory)/"results/a27/jobs/invalid/job_receipt.json",
                       dict(job="invalid", status="COMPLETE", device="cuda", wall_seconds=10.,
                            process_cpu_seconds=1., gpu_occupation_seconds=0., durable_reservations={}))
            with self.assertRaises(BudgetExceeded):
                budget_history(directory)

    def test_historical_scan_happens_once_at_job_start(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("a27.budget.budget_history", wraps=budget_history) as scan:
                with charged_job(directory, "once") as book:
                    for _ in range(4):
                        book.check()
                        with book.span("synthetic_operator"):
                            pass
                self.assertEqual(scan.call_count, 1)

    def test_zero_label_cap_is_hard_for_all_aliases_positive_limits_and_history(self):
        book = BudgetLedger(enforce=False)
        for category in ("new_labels", "new_clean_labels", "new_full_forward_labels"):
            with self.subTest(category=category), self.assertRaises(BudgetExceeded):
                book.reserve(category, 999999)
        self.assertFalse(book.reservations)
        with self.assertRaises(BudgetExceeded):
            BudgetLedger(prior_reservations={"new_labels": 1})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                with charged_job(directory, "illegal", config={"new_full_forward_labels_max": 1}):
                    pass
            write_json(Path(directory)/"results/a27/jobs/illegal/job_receipt.json",
                       dict(job="illegal", status="FAILED", device="cpu", wall_seconds=1., process_cpu_seconds=1.,
                            durable_reservations={"new_full_forward_labels": 1}))
            with self.assertRaises(BudgetExceeded):
                budget_history(directory)

    def test_terminal_receipts_immutable_and_unknown_interruption_blocks_next_job(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                with charged_job(directory, "failed"):
                    raise RuntimeError("synthetic failure")
            receipt = Path(directory)/"results/a27/jobs/failed/job_receipt.json"
            original = receipt.read_bytes()
            self.assertEqual(json.loads(original)["status"], "FAILED")
            with self.assertRaises(BudgetExceeded):
                with charged_job(directory, "failed"):
                    pass
            self.assertEqual(receipt.read_bytes(), original)
            write_json(Path(directory)/"results/a27/jobs/unknown/job_manifest.json", {"status": "STARTED"})
            with self.assertRaises(BudgetExceeded):
                with charged_job(directory, "next"):
                    pass
            self.assertFalse((Path(directory)/"results/a27/jobs/next").exists())

    def test_interrupt_receipt_and_foreign_locks_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(KeyboardInterrupt):
                with charged_job(directory, "interrupt"):
                    raise KeyboardInterrupt()
            receipt = Path(directory)/"results/a27/jobs/interrupt/job_receipt.json"
            self.assertEqual(json.loads(receipt.read_text())["status"], "INTERRUPTED")
            self.assertFalse(budget_history(directory)["unresolved_jobs"])
            path = Path(directory)/"foreign.lock"
            with owned_file_lock(path, owner_id="mine", job="unit") as own:
                ledger = BudgetLedger(enforce=False, metadata={"owner_id": "mine"})
                ledger._owned_locks = [own]
                path.write_bytes(b'{"owner_id":"foreign"}')
                with self.assertRaises(BudgetExceeded):
                    ledger.check()
            self.assertEqual(path.read_bytes(), b'{"owner_id":"foreign"}')
            self.assertEqual(SHARED_GPU_LOCK.as_posix(), "D:/AI/A20_OPM_IMAGING/runs/gpu.lock")

    def test_cpu_wall_and_monitor_accounting_and_caps(self):
        book = BudgetLedger(enforce=False)
        book.started_wall -= 5.
        self.assertGreaterEqual(book.receipt()["charged_cpu_seconds"], 5.)
        monitor = BudgetLedger(device="cuda", gpu_host_only=True, enforce=False)
        monitor.started_wall -= 100.
        receipt = monitor.receipt()
        self.assertEqual(receipt["gpu_occupation_seconds"], 0.)
        self.assertLess(receipt["charged_cpu_seconds"], 5.)
        for kwargs in ({"cpu_limit": 3601}, {"gpu_limit": 3601}, {"cpu_reserve": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BudgetLedger(**kwargs)

    def test_frozen_one_hour_limits_and_cumulative_stop_boundaries(self):
        self.assertEqual(CPU_LIMIT_SECONDS, 3600.)
        self.assertEqual(GPU_LIMIT_SECONDS, 3600.)
        cpu = BudgetLedger(prior_cpu=3580., started_wall=0., started_cpu=0.)
        with patch("a27.budget.time.perf_counter", return_value=5.), \
             patch("a27.budget.time.process_time", return_value=1.):
            with self.assertRaisesRegex(BudgetExceeded, "A27_CUMULATIVE_CPU_LIMIT"):
                cpu.check()
        gpu = BudgetLedger(device="cuda", prior_gpu=3569., started_wall=0., started_cpu=0.)
        with patch("a27.budget.time.perf_counter", return_value=1.), \
             patch("a27.budget.time.process_time", return_value=0.):
            with self.assertRaisesRegex(BudgetExceeded, "A27_CUMULATIVE_GPU_OCCUPATION_LIMIT"):
                gpu.check()
        for config in ({"cpu_budget_seconds": 3601}, {"gpu_budget_seconds": 3601},
                       {"cpu_limit_seconds": float("inf")}, {"gpu_limit_seconds": float("nan")}):
            with self.subTest(config=config), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    with charged_job(directory, "expanded", config=config):
                        pass
                self.assertFalse((Path(directory)/"results/a27/jobs").exists())

    def test_durable_reservations_restore_once_and_reject_truncated_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)/"results/a27/jobs/accepted"
            write_json(job/"job_receipt.json", dict(job="accepted", status="COMPLETE", device="cpu",
                       wall_seconds=3., process_cpu_seconds=1., durable_reservations={"checks": 1}))
            ledger = job/"COST_LEDGER.jsonl"
            ledger.write_text(json.dumps(dict(event="RESERVED", category="checks", job_used=2)) + "\n")
            self.assertEqual(budget_history(directory)["reservations"], {"checks": 2})
            ledger.write_text('{"event":')
            with self.assertRaisesRegex(BudgetExceeded, "TRUNCATED_DURABLE_LEDGER"):
                budget_history(directory)


def remote_module():
    path = Path(__file__).resolve().parents[1]/"tools/a27_remote.py"
    spec = importlib.util.spec_from_file_location("a27_remote_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def driver_module():
    path = Path(__file__).resolve().parents[1]/"tools/a27_driver.py"
    spec = importlib.util.spec_from_file_location("a27_driver_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DriverInfrastructureTests(unittest.TestCase):
    def test_default_roots_and_all_test_harness_preserves_active_ledger_and_skips(self):
        driver = driver_module()
        self.assertEqual(driver.STAGES, remote_module().STAGES)
        self.assertEqual(driver.default_a26_root().name, "A26_OM_FEEDBACK_PRIOR_COMPLETION" if os.name == "nt" else "a26-om-feedback-prior-completion")
        self.assertEqual(driver.default_data_root().parts[-2:], ("A25_ADAPTIVE_OM_FEEDBACK" if os.name == "nt" else "a25-adaptive-om-feedback", "data"))
        from a24.contracts import get_ledger
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with charged_job(root, "driver-tests") as book:
                class FixtureCase(unittest.TestCase):
                    def test_active_book(self):
                        self.assertIs(get_ledger(), book)

                    @unittest.skip("explicit synthetic skip")
                    def test_optional(self):
                        raise AssertionError("skip did not apply")

                suite = unittest.defaultTestLoader.loadTestsFromTestCase(FixtureCase)
                with patch("unittest.defaultTestLoader.discover", return_value=suite) as discover:
                    result = driver.tests_stage(root, book)
                discover.assert_called_once_with(str(root/"tests"), pattern="test_*.py")
                self.assertEqual(result["tests"], 2)
                self.assertEqual(result["skipped"], 1)
                self.assertTrue(result["passed"])
                path = root/"results/a27/jobs/driver-tests"
                self.assertEqual(json.loads((path/"TESTS_SUMMARY.json").read_text()), result)
                self.assertIn("explicit synthetic skip", (path/"TESTS.log").read_text())
                self.assertEqual(result["skipped_tests"][0]["reason"], "explicit synthetic skip")

    def test_six_argument_dispatch_and_import_tail_are_charged(self):
        driver = driver_module()
        import a27
        calls = []
        pipeline = types.ModuleType("a27.pipeline")

        def audit(*args):
            calls.append(args)
            return {"status": "COMPLETE", "synthetic": True}

        pipeline.stage_audit = audit
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root/"configs/a27/pilot.json", {"cpu_limit_seconds": 3600,
                       "gpu_limit_seconds": 3600, "new_labels": False})
            driver.START_WALL -= 3.
            with patch.object(a27, "pipeline", pipeline, create=True), \
                 patch.dict(sys.modules, {"a27.pipeline": pipeline}), \
                 patch.object(sys, "path", list(sys.path)), redirect_stdout(io.StringIO()):
                driver.main(["--root", str(root), "--data-root", "/unopened/a25/data",
                             "--a26-root", "/unopened/a26", "--job", "driver-audit",
                             "--stage", "audit", "--device", "cpu"])
            self.assertEqual(len(calls), 1)
            args = calls[0]
            self.assertEqual(len(args), 6)
            self.assertEqual(args[:3], (root.resolve(), Path("/unopened/a25/data"), Path("/unopened/a26")))
            self.assertEqual(args[5], "cpu")
            path = root/"results/a27/jobs/driver-audit"
            receipt = json.loads((path/"job_receipt.json").read_text())
            self.assertEqual(receipt["status"], "COMPLETE")
            self.assertGreaterEqual(receipt["charged_cpu_seconds"], 3.)
            self.assertEqual(json.loads((path/"RESULT.json").read_text()), {"status": "COMPLETE", "synthetic": True})

    def test_stage_exception_keeps_failed_native_receipt_and_error(self):
        driver = driver_module()
        import a27
        pipeline = types.ModuleType("a27.pipeline")

        def post(*_):
            raise RuntimeError("synthetic stage failure")

        pipeline.stage_post = post
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root/"configs/a27/pilot.json", {})
            with patch.object(a27, "pipeline", pipeline, create=True), \
                 patch.dict(sys.modules, {"a27.pipeline": pipeline}), \
                 patch.object(sys, "path", list(sys.path)), \
                 self.assertRaisesRegex(RuntimeError, "synthetic stage failure"):
                driver.main(["--root", str(root), "--job", "driver-failed", "--stage", "post"])
            path = root/"results/a27/jobs/driver-failed"
            self.assertEqual(json.loads((path/"job_receipt.json").read_text())["status"], "FAILED")
            self.assertIn("synthetic stage failure", (path/"ERROR.log").read_text())
            self.assertFalse((path/"RESULT.json").exists())


class RemoteInfrastructureTests(unittest.TestCase):
    def test_invalid_timeout_and_config_are_rejected_before_transport(self):
        remote = remote_module()
        with patch.object(remote, "push") as push, patch.object(remote, "_call") as call:
            for timeout in (float("nan"), float("inf"), 0., -1., True, None, "invalid"):
                with self.subTest(timeout=timeout), self.assertRaises(remote.TransportError):
                    remote.run(object(), object(), Path("/unused"), "a27-check", "audit", timeout=timeout)
            for config in ("configs/a26/pilot.json", "configs/a27/../pilot.json", "private/config.json"):
                with self.subTest(config=config), self.assertRaises(remote.TransportError):
                    remote.run(object(), object(), Path("/unused"), "a27-check", "audit", config=config)
            push.assert_not_called()
            call.assert_not_called()

    def test_stage_registry_roots_and_generated_controller_compile(self):
        remote = remote_module()
        self.assertEqual(remote.STAGES, ("audit", "tests", "post", "proxhealth", "postfgp", "diagnostics",
                                        "observability", "latency", "report"))
        self.assertEqual(remote.REMOTE, "D:/AI/A27_FEEDBACK_FAILURE_ATTRIBUTION")
        self.assertEqual(remote.READ_ONLY_A26, "D:/AI/A26_OM_FEEDBACK_PRIOR_COMPLETION")
        self.assertEqual(remote.READ_ONLY_A25_DATA, "D:/AI/A25_ADAPTIVE_OM_FEEDBACK/data")
        self.assertEqual(remote.SHARED_LOCK, SHARED_GPU_LOCK.as_posix())

        class Helper:
            @staticmethod
            def _windows_cpu_source():
                return "child_cpu_measurement_available = True\ndef measured_cpu(child): return 1.\n"

        code = remote._run_source(Helper(), "a27-test", "audit", "cpu", 3600., "configs/a27/pilot.json")
        compile(code, "<a27-controller-fixture>", "exec")
        self.assertIn("from a27.budget", code)
        self.assertIn("tools/a27_driver.py", code)
        self.assertIn("cpu_remaining=3600", code)
        self.assertIn("gpu_remaining=3600", code)
        self.assertIn(">=3585", code)
        self.assertNotIn("results/a26/jobs", code)

    def test_source_deployment_has_config_roots_and_only_a27_accepted_receipts(self):
        remote = remote_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            required = ("src/a27/budget.py", "tools/a27_driver.py", "tests/test_a27_budget.py",
                        "configs/a27/pilot.json", "configs/a26/pilot.json", "configs/a25/campaign.json",
                        "configs/a24_frozen.json", "results/a27/jobs/done/job_receipt.json",
                        "results/a27/jobs/done/job_manifest.json")
            excluded = ("data/train.npz", "results/a26/jobs/old/job_receipt.json",
                        "results/a27/jobs/active/job_manifest.json", "private/credential.json")
            for name in required + excluded:
                target = root/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("{}")
            members = {p.relative_to(root).as_posix() for p in remote.deployment_members(root, "source")}
            self.assertEqual(members, set(required))
            accounting = {p.relative_to(root).as_posix() for p in remote.deployment_members(root, "accounting")}
            self.assertEqual(accounting, set(required[-2:]))
            for scope in ("train", "evaluation"):
                with self.subTest(scope=scope), self.assertRaises(remote.TransportError):
                    remote.deployment_members(root, scope)

    def test_preflight_is_read_only_availability_query_and_preserves_private_error_boundary(self):
        remote = remote_module()

        class Helper:
            source = None

            def _python(self, connection, source, **kwargs):
                self.source = source
                return {"status": "COMPLETE"}

        helper = Helper()
        self.assertEqual(remote.preflight(helper, object()), {"status": "COMPLETE"})
        compile(helper.source, "<a27-preflight-fixture>", "exec")
        self.assertIn("a26_input_availability", helper.source)
        self.assertIn("a25_input_availability", helper.source)
        self.assertIn("read_only_inputs_available", helper.source)
        self.assertIn("queue_clear", helper.source)
        self.assertIn("PREDICTIONS.npz", helper.source)
        self.assertIn("best.pt", helper.source)
        self.assertIn("validation_N3.npz", helper.source)
        self.assertNotIn(".read_", helper.source)
        self.assertNotIn(".write_", helper.source)
        self.assertNotIn(".mkdir", helper.source)
        self.assertNotIn("sha256", helper.source)
        for member in ("../a26/anything", "private/config.json", "D:/AI/x", "/absolute", "a\\b"):
            with self.subTest(member=member):
                self.assertFalse(remote.safe_member(member))
        with patch.object(helper, "_python", side_effect=RuntimeError("synthetic-private-value")):
            with self.assertRaises(remote.TransportError) as error:
                remote.preflight(helper, object())
        self.assertNotIn("synthetic-private-value", str(error.exception))


if __name__ == "__main__":
    unittest.main()
