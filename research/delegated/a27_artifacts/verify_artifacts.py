"""Synthetic saved-output and render verification; no research experiment."""
from contextlib import contextmanager
from pathlib import Path
import csv
import json
import math
import sys
import tempfile
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from a27 import artifacts as a


class FixtureBook:
    def __init__(self):
        self.spans = []
        self.checks = 0

    def check(self):
        self.checks += 1

    @contextmanager
    def span(self, label, **counts):
        self.spans.append({"label": label, "counts": counts})
        yield


def csvfile(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader(); writer.writerows(rows)


def fixture(out):
    expected = a._expected_metadata()
    cases = list(expected)
    post, diagnostics, observable = [], [], []
    arrays = {key: np.asarray([meta[key] for meta in expected.values()])
              for key in ("parent_id", "family", "background_id", "amplitude", "noise")}
    arrays["case_id"] = np.asarray(cases)
    arrays["sigma"] = np.asarray([.001 if meta["noise"] == "20dB" else 0. for meta in expected.values()])
    yy, xx = np.mgrid[-1:1:24j, -1:1:24j]
    shape = np.exp(-6*(xx*xx+yy*yy)).reshape(-1)
    backgrounds = {"weak": np.full(576, .1+.01j), "ring": np.full(576, .2+.02j)}
    truth = np.asarray([backgrounds[meta["background_id"]] + meta["amplitude"]*(1+.1j)*shape
                        for meta in expected.values()])
    arrays["chi"] = truth
    predictions = {"case_id": np.asarray(cases)}
    for network in a.NETWORKS:
        for correction in a.CORRECTIONS:
            factor = {"NONE": .5, "L": .4, "F": .3, "TASK_R32": .6}[correction]
            predictions[network + "_" + correction] = truth + factor*(.1+.01j)*shape
    first = {min(p for p in a.VALIDATION_PARENTS if a._family(p) == family) for family in a.FAMILIES}
    for ci, (case, meta) in enumerate(expected.items()):
        common = dict(case_id=case, **meta, sigma=.001 if meta["noise"] == "20dB" else 0.)
        for network in a.NETWORKS:
            for correction in a.CORRECTIONS:
                # Condition-varying controls expose row-level pseudoreplication.
                control = .1 if ci % 8 == 0 else 1.
                factor = {"NONE": 1., "L": .9, "F": .8, "TASK_R32": 1.1}[correction]
                error = control if ci % 8 == 0 else control*factor
                row = dict(common, network=network, correction=correction, valid=True,
                           accepted=correction != "NONE", correction_wall_seconds=.01,
                           update_mass_norm=.001, line_trials=1)
                row.update({metric: error for metric in a.METRICS})
                row.update(ssim_delta=.6+(1-factor)*.1, edge_f1=.6, support_iou=.7,
                           passive=True, feasible=True, full_backward_residual=1e-15,
                           full_data_relative_residual=.2*factor, full_data_absolute_residual=.01*factor)
                post.append(row)
        for state in a.STATE_ORDER:
            row = dict(common, state=state, valid=True, F_feedback_noise_ratio=2. if common["sigma"] else None,
                       R_feedback_noise_ratio=1.5 if common["sigma"] else None,
                       F_minus_L_data_norm=.01, R_minus_L_data_norm=.008,
                       R_forward_error_vs_F=.01, R_JVP_error_vs_F=.02, R_adjoint_error_vs_F=.03,
                       R_gradient_cosine_vs_F=.9, L_gradient_cosine_vs_F=.8,
                       R_box_update_error_vs_F=.1, L_box_update_error_vs_F=.2)
            diagnostics.append(row)
        if meta["parent_id"] in first and meta["noise"] == "20dB":
            for network in a.NETWORKS:
                observable.append(dict(common, network=network, strong_error_fraction=.2, weak_error_fraction=.8,
                    background_strong_error_fraction=.15, soft_error_fraction=.12, soft_complement_error_fraction=.64,
                    soft_cross_energy_fraction=.24, predicted_strong_rank=16, background_strong_rank=14,
                    background_predicted_strong_overlap=.6))
    csvfile(out/"results/a27/post/PER_CASE.csv", post)
    csvfile(out/"results/a27/diagnostics/PER_STATE.csv", diagnostics)
    csvfile(out/"results/a27/observability/PER_STATE.csv", observable)
    np.savez_compressed(out/"results/a27/post/CORRECTED_PREDICTIONS.npz", **predictions)
    data = out/"synthetic_backgrounds"
    for name, value in backgrounds.items():
        (data/"background").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(data/"background"/(name+".npz"), chi_b=value)
    reps = a._expected_cases(sorted(first), noisy_only=True)
    csvfile(out/"results/a27/latency/NATIVE.csv", [dict(arm=net+"_BASELINE", method=net, correction="NONE",
            timing_mode="native", status="OK", native_full_inference_seconds=.03 if net == "N3" else .02,
            case_id=case, repeat=i) for net in a.NETWORKS for case in reps for i in range(3)])
    return data, arrays, post


def main():
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    base = Path(__file__).resolve().parent
    out = base/"synthetic_preview"
    out.mkdir(exist_ok=True)
    data, arrays, post = fixture(out)
    config = {"protocol_id": "SYNTHETIC_QA_ONLY", "primary_network": "N3",
              "corrections": list(a.CORRECTIONS), "primary_seed": 271010, "bootstrap_samples": 2000}
    book = FixtureBook()
    parents, grouped, unexpected = a._parent_tables(post)
    stats, _ = a._paired(parents, config, book)
    measured = stats["networks"]["N3"]["F_vs_NONE"]["COMPLEX_FAMILIES"]["delta_nrmse"]
    expected_gain = ((.1+7*1.)/8 - (.1+7*.8)/8) / ((.1+7*1.)/8)
    assert math.isclose(measured["point_estimate"], expected_gain, rel_tol=1e-12)
    assert not math.isclose(measured["point_estimate"], 7*.2/8, rel_tol=1e-4)
    assert measured["paired_parents"] == 12 and measured["samples"] == 2000
    damaged = [row for row in post if not (row["network"] == "N3" and row["correction"] == "NONE"
                                         and row["case_id"] == a._expected_cases()[0])]
    badparents, _, _ = a._parent_tables(damaged)
    badstats, _ = a._paired(badparents, config, book)
    incomplete = badstats["networks"]["N3"]["F_vs_NONE"]["COMPLEX_FAMILIES"]["delta_nrmse"]
    assert incomplete["point_estimate"] is None and incomplete["ci95"] is None
    assert incomplete["paired_parents"] == 11
    assert incomplete["status"] == "HOLD"
    assert badparents[0]["delta_nrmse"] is None
    zero = [dict(row) for row in parents]
    for row in zero:
        if row["network"] == "N3" and row["correction"] == "NONE":
            row["delta_nrmse"] = 0.
    zerostats, _ = a._paired(zero, config, book)
    assert zerostats["networks"]["N3"]["F_vs_NONE"]["COMPLEX_FAMILIES"]["delta_nrmse"]["point_estimate"] is None
    diagnostic = a._diagnostic_case_table([dict(case_id=a._expected_cases()[0], state="COMMON_BP",
        **a._expected_metadata()[a._expected_cases()[0]], sigma=0., valid=True)])
    assert diagnostic[0]["F_feedback_noise_ratio_status"] == "UNDEFINED_ZERO_NOISE_DENOMINATOR"
    changed = [dict(row) for row in post]
    for row in changed:
        if row["correction"] == "TASK_R32":
            row["full_data_relative_residual"] = .1
    _, changed_groups, _ = a._parent_tables(changed)
    count = next(row for row in a._case_counts(changed_groups)
                 if row["network"] == "N3" and row["correction"] == "TASK_R32")
    assert count["data_decreases_material_worsens"] == 112
    assert count["data_residual_decreases"] == 128 and count["accepted_updates"] == 128
    missing_count = next(row for row in a._case_counts(a._parent_tables(damaged)[1])
                         if row["network"] == "N3" and row["correction"] == "NONE")
    assert missing_count["accepted_status_unknown"] == 1
    ring_failed = [dict(row, valid=False) if row["background_id"] == "ring" and row["correction"] != "NONE"
                   else dict(row) for row in post]
    ring_full, _, _ = a._parent_tables(ring_failed)
    ring_stats, _ = a._paired(ring_full, config, book)
    assert ring_stats["networks"]["N3"]["F_vs_NONE"]["COMPLEX_FAMILIES"]["delta_nrmse"]["status"] == "HOLD"
    weakparents, _, _ = a._parent_tables(ring_failed, background="weak")
    weakstats, _ = a._paired(weakparents, config, book, descriptive_weak=True)
    assert weakstats["descriptive_only"] and not weakstats["substitute_for_full_cohort"]
    assert weakstats["networks"]["N3"]["F_vs_NONE"]["COMPLEX_FAMILIES"]["delta_nrmse"]["coverage_complete"]
    assert all(row["expected_conditions"] == 4 for row in weakparents)
    with tempfile.TemporaryDirectory() as directory:
        isolated = Path(directory)
        source = isolated/"results/a27/POST_SOURCE.json"
        source.parent.mkdir(parents=True)
        assert a._post_source(isolated)[0] == "post"
        source.write_text(json.dumps({"folder": "post_fgp", "selection_reason": "fixed numerical certification"}))
        assert a._post_source(isolated)[0] == "post_fgp"
        source.write_text(json.dumps({"folder": "../private"}))
        try:
            a._post_source(isolated)
            raise AssertionError("non-whitelisted folder was accepted")
        except ValueError:
            pass
        native, _ = a._read_csv(out/"results/a27/latency/NATIVE.csv")
        native[0]["status"] = "FAILED"
        csvfile(isolated/"results/a27/latency/NATIVE.csv", native)
        families = a._family_tables(parents)
        latency, _ = a._latency_table(isolated, isolated/"missing_a26", families, config, book, "post")
        failed_native = next(row for row in latency if row["arm"] == "N3_BASELINE")
        assert failed_native["latency_status"] == "HOLD" and failed_native["native_full_inference_seconds"] is None
        native.pop(0)
        csvfile(isolated/"results/a27/latency/NATIVE.csv", native)
        latency, _ = a._latency_table(isolated, isolated/"missing_a26", families, config, book, "post")
        missing_native = next(row for row in latency if row["arm"] == "N3_BASELINE")
        assert missing_native["missing_samples"] == 1 and missing_native["native_full_inference_seconds"] is None
        reps = a._expected_cases([min(p for p in a.VALIDATION_PARENTS if a._family(p) == family)
                                  for family in a.FAMILIES], noisy_only=True)
        csvfile(isolated/"results/a27/latency/POST_RAW_TO_IMAGE.csv", [dict(arm="POST_F", network="N3",
            correction="F", timing_mode="native", status="OK", native_full_inference_seconds=.07,
            post_source_folder="post", case_id=case, repeat=i) for case in reps for i in range(3)])
        latency, _ = a._latency_table(isolated, isolated/"missing_a26", families, config, book, "post_fgp")
        mismatched = next(row for row in latency if row["arm"] == "POST_F")
        assert mismatched["quality_status"] == "LATENCY_AND_QUALITY_POST_SOURCE_MISMATCH"
        assert mismatched["delta_nrmse"] is None
    save = a._save_figure

    def synthetic_save(fig, *args, **kwargs):
        fig.text(.5, -.012, "SYNTHETIC QA ONLY — no experimental evidence", ha="center", fontsize=9, color="#9b3232")
        return save(fig, *args, **kwargs)

    with patch("a26.data.load_split", return_value=arrays) as loader, patch.object(a, "_save_figure", synthetic_save):
        result = a.build_artifacts(out, data, out/"missing_frozen_a26", config, book, "cpu")
        loader.assert_called_once_with(data, "validation")
    summary = json.loads((out/result["summary_path"]).read_text())
    assert result["status"] == "COMPLETE"
    assert summary["fixed_images"]["truth_cases_used"] == 8
    assert summary["latency_figure_status"] == "CREATED"
    assert not summary["scientific_verdict_performed"] and summary["new_labels"] == 0
    assert sum(record["label"] == "a27_build_saved_evidence_artifacts" for record in book.spans) == 1
    for record in summary["figures"]:
        assert (out/"results/a27/artifacts"/record["png"]).is_file()
        assert (out/"results/a27/artifacts"/record["svg"]).is_file()
    verification = {"status": "PASS", "checks": ["eight-condition average precedes gains", "12 complex parents",
        "2000 paired-parent samples", "missing condition invalidates parent and CI", "zero control error remains undefined",
        "clean noise ratio undefined", "accepted updates and data-fit/material-worsening counts", "failed acceptance count remains unknown",
        "validation loader called once only for fixed images", "fixed eight image cases",
        "whole function paid-span hook", "saved PNG/SVG/plot CSV output", "native latency/quality pairing",
        "explicit whitelisted POST_SOURCE selection", "weak-only description never replaces full HOLD",
        "fixed native cohort rejects failed/missing timing", "solver/input provenance mismatch suppresses quality pair"],
        "wall_seconds": time.perf_counter()-started_wall, "process_cpu_seconds": time.process_time()-started_cpu,
        "scope": "synthetic QA only; no remote work or research experiment", "build_interface": result,
        "figures": len(summary["figures"]), "book_checks": book.checks}
    (base/"VERIFICATION.json").write_text(json.dumps(verification, indent=2)+"\n")
    print(json.dumps(verification))


if __name__ == "__main__":
    main()
