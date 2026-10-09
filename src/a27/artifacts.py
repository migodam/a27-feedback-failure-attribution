"""Saved A27 evidence tables and figures; no physics or scientific verdict.

Every correction starts with the fixed 128-case exposed validation cohort.
Eight conditions are averaged within each independent parent before paired
gains and 2000-resample parent bootstrap intervals are computed. A missing,
failed or undefined condition makes that parent incomplete; no success-only
estimate replaces it. Validation truth is read only for metadata-fixed images.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np

from a26.metrics import COMPLEX_FAMILIES, FAMILIES, VALIDATION_PARENTS
from a27.budget import write_json

NETWORKS = ("N3", "N1")
CORRECTIONS = ("NONE", "L", "F", "TASK_R32")
PAIRS = (("NONE", "L"), ("NONE", "F"), ("NONE", "TASK_R32"),
         ("L", "F"), ("F", "TASK_R32"))
METRICS = ("delta_nrmse", "full_nrmse", "real_nrmse", "imag_nrmse",
           "ssim_delta", "edge_f1", "support_iou", "full_data_absolute_residual",
           "full_data_relative_residual", "full_backward_residual", "passive", "feasible")
HIGHER_IS_BETTER = frozenset({"ssim_delta", "edge_f1", "support_iou", "passive", "feasible"})
STATE_ORDER = ("COMMON_BP", "ONE_STEP_L", "CNN_N1", "CNN_N3",
               "POST_N3_L", "POST_N3_F", "POST_N3_TASK_R32",
               "POST_N1_L", "POST_N1_F", "POST_N1_TASK_R32")
BACKGROUND_IDS = ("weak", "ring")
AMPLITUDES = (.3, 1.2)
LATENCY_ENCODER_ARMS = (("ORIGINAL_N1", "N1"), ("ORIGINAL_N2", "N2"), ("ORIGINAL_N3", "N3"),
                        ("N1_BASELINE", "N1"), ("N2_BASELINE", "N2"), ("N3_BASELINE", "N3"),
                        ("N2_NO_BETA_ONLY", "N2"), ("N2_LU_REUSE_ONLY", "N2"), ("N2_COMBINED", "N2"))
MASS_VOLUME = 1 / 576
OFFLINE_ALIGNMENT_METRICS = tuple("OFFLINE_" + arm + "_" + kind
                                  for kind in ("descent_error_alignment", "projected_step_alignment")
                                  for arm in ("F", "L", "R"))
OBSERVABILITY_GRADIENT_METRICS = ("full_gradient_mass_norm", "full_gradient_strong_energy_fraction",
                                 "OFFLINE_gradient_error_alignment")
ACTUAL_UPDATE_SCOPE = "saved returned adaptive TV-prox update from same-network NONE start"
BOX_PROXY_SCOPE = "fixed background-step box-projection proxy; not actual adaptive TV-prox update"


def _number(value):
    if value in (True, "True", "true"):
        return 1.
    if value in (False, "False", "false"):
        return 0.
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _flag(value):
    if value in (True, 1, "True", "true", "1"):
        return True
    if value in (False, 0, "False", "false", "0"):
        return False
    return None


def _ok(row):
    return (_flag(row.get("valid", True)) is True and
            _flag(row.get("complete", True)) is True and
            row.get("status", "OK") in {"OK", "COMPLETE", "MEASURED"})


def _family(parent):
    return FAMILIES[(parent - VALIDATION_PARENTS[0]) // 4]


def _case(parent, background, amplitude_index, noisy):
    return f"p{parent}_{background}_a{amplitude_index}_" + ("snr20" if noisy else "clean")


def _expected_cases(parents=VALIDATION_PARENTS, *, noisy_only=False, high_only=False):
    return [_case(parent, bg, ai, noisy) for parent in parents for bg in BACKGROUND_IDS
            for ai in ((1,) if high_only else (0, 1))
            for noisy in ((True,) if noisy_only else (False, True))]


def _expected_metadata():
    return {_case(parent, bg, ai, noisy): dict(parent_id=parent, family=_family(parent),
                background_id=bg, amplitude=AMPLITUDES[ai], noise="20dB" if noisy else "clean")
            for parent in VALIDATION_PARENTS for bg in BACKGROUND_IDS
            for ai in (0, 1) for noisy in (False, True)}


def _metadata_matches(row, expected):
    return (_number(row.get("parent_id")) == expected["parent_id"] and
            str(row.get("family")) == expected["family"] and
            str(row.get("background_id")) == expected["background_id"] and
            _number(row.get("amplitude")) == expected["amplitude"] and
            str(row.get("noise")) == expected["noise"] and
            (_number(row.get("sigma")) == 0 if expected["noise"] == "clean" else
             (_number(row.get("sigma")) is not None and _number(row.get("sigma")) > 0)))


def _read_csv(path):
    path = Path(path)
    if not path.is_file():
        return [], "MISSING"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle)), "AVAILABLE"


def _post_source(root):
    path = Path(root)/"results/a27/POST_SOURCE.json"
    if path.is_symlink():
        raise ValueError("A27_EXPLICIT_POST_SOURCE_PATH_FORBIDDEN")
    if not path.exists():
        return "post", {"selection": "default_original", "folder": "post", "manifest": None}
    if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("A27_EXPLICIT_POST_SOURCE_PATH_FORBIDDEN")
    record = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(record, dict) or not isinstance(record.get("folder"), str) or
            record.get("folder") not in {"post", "post_fgp"}):
        raise ValueError("A27_EXPLICIT_POST_SOURCE_FOLDER_MUST_BE_WHITELISTED")
    return record["folder"], {"selection": "explicit_manifest", "folder": record["folder"],
                             "manifest_path": "results/a27/POST_SOURCE.json", "manifest": record}


def _write_csv(path, rows, book):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            if index % 4096 == 0:
                book.check()
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple))
                             else value for key, value in row.items()})


def _parent_tables(rows, *, background=None):
    if background not in {None, "weak"}:
        raise ValueError("A27_ONLY_EXPLICIT_WEAK_DESCRIPTIVE_SLICE_ALLOWED")
    expected = _expected_metadata()
    grouped, unexpected = defaultdict(list), []
    for row in rows:
        key = (str(row.get("network")), str(row.get("correction")), str(row.get("case_id")))
        if key[0] not in NETWORKS or key[1] not in CORRECTIONS or key[2] not in expected:
            unexpected.append(row)
        else:
            grouped[key].append(row)
    output = []
    for network in NETWORKS:
        for correction in CORRECTIONS:
            for parent in VALIDATION_PARENTS:
                records, errors = [], []
                conditions = [case for case in _expected_cases((parent,))
                              if background is None or expected[case]["background_id"] == background]
                for case in conditions:
                    candidates = grouped.get((network, correction, case), [])
                    if len(candidates) != 1:
                        errors.append({"case_id": case, "status": "MISSING" if not candidates else "DUPLICATED"})
                        continue
                    row = candidates[0]
                    if not _metadata_matches(row, expected[case]):
                        errors.append({"case_id": case, "status": "METADATA_MISMATCH"})
                    elif not _ok(row):
                        errors.append({"case_id": case, "status": "FAILED", "error": row.get("error")})
                    elif any(_number(row.get(metric)) is None for metric in METRICS):
                        errors.append({"case_id": case, "status": "UNDEFINED_METRIC",
                                       "metrics": [m for m in METRICS if _number(row.get(m)) is None]})
                    else:
                        records.append(row)
                complete = len(records) == len(conditions) and not errors and not unexpected
                accepted = [_flag(row.get("accepted")) for row in records]
                result = dict(network=network, correction=correction, parent_id=parent, family=_family(parent),
                              expected_conditions=len(conditions), valid_conditions=len(records), complete=complete,
                              status="COMPLETE" if complete else "INCOMPLETE", failures=errors,
                              accepted_updates=sum(flag is True for flag in accepted),
                              accepted_status_unknown=len(conditions)-len(records)+sum(flag is None for flag in accepted))
                if background:
                    result.update(background_slice=background, descriptive_only=True,
                                  scope="POST_HOC_WEAK_BACKGROUND_SLICE; never substitutes for full cohort")
                result.update({metric: float(np.mean([_number(row[metric]) for row in records]))
                               if complete else None for metric in METRICS})
                output.append(result)
    return output, grouped, unexpected


def _group_table(parents, ids, **metadata):
    indexed = {int(row["parent_id"]): row for row in parents}
    selected = [indexed.get(parent) for parent in ids]
    complete = all(row is not None and row["complete"] for row in selected)
    row = {**metadata, "expected_parents": len(ids),
           "complete_parents": sum(item is not None and item["complete"] for item in selected),
           "expected_conditions": (parents[0]["expected_conditions"] if parents else 8) * len(ids), "complete": complete,
           "status": "COMPLETE" if complete else "INCOMPLETE"}
    row.update({metric: float(np.mean([item[metric] for item in selected]))
                if complete else None for metric in METRICS})
    return row


def _family_tables(parentrows):
    output = []
    for network in NETWORKS:
        for correction in CORRECTIONS:
            rows = [row for row in parentrows if row["network"] == network and row["correction"] == correction]
            groups = [(family, [p for p in VALIDATION_PARENTS if _family(p) == family]) for family in FAMILIES]
            groups += [("COMPLEX_FAMILIES", [p for p in VALIDATION_PARENTS if _family(p) in COMPLEX_FAMILIES]),
                       ("ALL_FAMILIES", list(VALIDATION_PARENTS))]
            output.extend(_group_table(rows, ids, network=network, correction=correction, group=group)
                          for group, ids in groups)
    return output


def _paired(parentrows, config, book, *, descriptive_weak=False):
    if int(config.get("bootstrap_samples", 2000)) != 2000:
        raise ValueError("A27_FROZEN_2000_PAIRED_PARENT_BOOTSTRAP_REQUIRED")
    indexed = {(row["network"], row["correction"], row["parent_id"]): row for row in parentrows}
    seed = int(config["primary_seed"])
    result = {"samples": 2000, "seed": seed, "unit": "independent_parent",
              "condition_rule": ("mean four weak-background conditions within parent before gains" if descriptive_weak else
                                 "mean eight conditions within parent before gains"),
              "error_gain_rule": "(control_error-candidate_error)/control_error",
              "higher_metric_rule": "candidate-control, absolute difference",
              "complex_families": list(COMPLEX_FAMILIES),
              "expected_parent_ids": list(VALIDATION_PARENTS),
              "exposure": "EXPOSED_DEVELOPMENT_VALIDATION", "holdout": False,
              "scientific_verdict_performed": False, "descriptive_only": descriptive_weak,
              "scope": "POST_HOC_WEAK_BACKGROUND_SLICE" if descriptive_weak else "FULL_FROZEN_COHORT",
              "substitute_for_full_cohort": False, "networks": {}}
    plotrows = []
    groups = [("ALL_FAMILIES", list(VALIDATION_PARENTS)),
              ("COMPLEX_FAMILIES", [p for p in VALIDATION_PARENTS if _family(p) in COMPLEX_FAMILIES])]
    groups += [(family, [p for p in VALIDATION_PARENTS if _family(p) == family]) for family in FAMILIES]
    for ni, network in enumerate(NETWORKS):
        result["networks"][network] = {}
        for pi, (reference, candidate) in enumerate(PAIRS):
            comparison = result["networks"][network][candidate + "_vs_" + reference] = {}
            for gi, (group, ids) in enumerate(groups):
                comparison[group] = {}
                for mi, metric in enumerate(METRICS):
                    book.check()
                    gains, missing = [], []
                    for parent in ids:
                        control = indexed[(network, reference, parent)]
                        proposed = indexed[(network, candidate, parent)]
                        a, b = _number(control.get(metric)), _number(proposed.get(metric))
                        if not control["complete"] or not proposed["complete"] or a is None or b is None:
                            missing.append({"parent_id": parent, "status": "FAILED_MISSING_OR_UNDEFINED_PARENT"})
                        elif metric not in HIGHER_IS_BETTER and a <= 0:
                            missing.append({"parent_id": parent, "status": "UNDEFINED_ZERO_CONTROL_ERROR"})
                        else:
                            gains.append({"parent_id": parent, "gain": b-a if metric in HIGHER_IS_BETTER else (a-b)/a})
                    complete = len(gains) == len(ids) and not missing
                    stat = {"reference": reference, "candidate": candidate, "metric": metric,
                            "group": group, "relative": metric not in HIGHER_IS_BETTER,
                            "higher_is_better": metric in HIGHER_IS_BETTER,
                            "expected_parents": len(ids), "paired_parents": len(gains),
                            "coverage_complete": complete, "status": "COMPLETE" if complete else "HOLD",
                            "failed_or_undefined_parents": missing, "parent_gains": gains,
                            "point_estimate": None, "ci95": None, "samples": 2000}
                    if complete:
                        values = np.asarray([row["gain"] for row in gains])
                        rng = np.random.default_rng(np.random.SeedSequence([seed, ni, pi, gi, mi]))
                        sampled = values[rng.integers(0, len(values), (2000, len(values)))].mean(axis=1)
                        stat.update(point_estimate=float(values.mean()),
                                    ci95=[float(x) for x in np.quantile(sampled, [.025, .975])])
                    comparison[group][metric] = stat
                    if metric == "delta_nrmse":
                        plotrows.append(dict(network=network, reference=reference, candidate=candidate,
                                             group=group, gain=stat["point_estimate"],
                                             ci95_lower=stat["ci95"][0] if stat["ci95"] else None,
                                             ci95_upper=stat["ci95"][1] if stat["ci95"] else None,
                                             expected_parents=len(ids), paired_parents=len(gains), status=stat["status"]))
    return result, plotrows


def _case_counts(grouped):
    expected = _expected_metadata()
    output = []
    for network in NETWORKS:
        for correction in CORRECTIONS:
            counts = Counter()
            for case in _expected_cases():
                candidates = grouped.get((network, correction, case), [])
                reference = grouped.get((network, "NONE", case), [])
                counts["expected_cases"] += 1
                if (len(candidates) != 1 or not _ok(candidates[0]) or
                        not _metadata_matches(candidates[0], expected[case])):
                    counts["failed_or_missing_cases"] += 1
                    counts["accepted_status_unknown"] += 1
                    counts["passivity_unknown"] += 1
                    counts["feasibility_unknown"] += 1
                    counts["unavailable_NONE_pair_cases"] += 1
                    continue
                row = candidates[0]
                accepted = _flag(row.get("accepted"))
                counts["accepted_updates"] += accepted is True
                counts["accepted_status_unknown"] += accepted is None
                counts["passive_cases"] += _flag(row.get("passive")) is True
                counts["passivity_unknown"] += _flag(row.get("passive")) is None
                counts["feasible_cases"] += _flag(row.get("feasible")) is True
                counts["feasibility_unknown"] += _flag(row.get("feasible")) is None
                counts["undefined_metric_cases"] += any(_number(row.get(metric)) is None for metric in METRICS)
                if (len(reference) != 1 or not _ok(reference[0]) or
                        not _metadata_matches(reference[0], expected[case])):
                    counts["unavailable_NONE_pair_cases"] += 1
                    continue
                values = [_number(item.get(metric)) for item in (reference[0], row)
                          for metric in ("delta_nrmse", "full_data_relative_residual")]
                if any(value is None for value in values):
                    counts["unavailable_NONE_pair_cases"] += 1
                    continue
                ce, cr, pe, pr = values
                counts["complete_NONE_pair_cases"] += 1
                counts["data_residual_decreases"] += pr < cr
                counts["data_decreases_material_worsens"] += pr < cr and pe > ce
            output.append(dict(network=network, correction=correction,
                               comparison="case-level versus same-network NONE; strict numeric inequalities",
                               **{key: int(counts[key]) for key in (
                                   "expected_cases", "failed_or_missing_cases", "undefined_metric_cases", "accepted_updates",
                                   "accepted_status_unknown", "passive_cases", "passivity_unknown", "feasible_cases", "feasibility_unknown",
                                   "unavailable_NONE_pair_cases", "complete_NONE_pair_cases",
                                   "data_residual_decreases", "data_decreases_material_worsens")}))
    return output


def _saved_input_status(matches, expected):
    if not matches:
        return "MISSING"
    if len(matches) != 1:
        return "DUPLICATED"
    if not _metadata_matches(matches[0], expected):
        return "METADATA_MISMATCH"
    return "COMPLETE" if _ok(matches[0]) else "FAILED"


def _saved_correction_arrays(post_dir, book):
    """Read saved images only, with identifiers checked before pairing rows."""
    path = post_dir / "CORRECTED_PREDICTIONS.npz"
    result = {"path": str(path), "status": "MISSING", "array_status": {},
              "alignment_rule": "exact unique frozen case-ID set; each array indexed by its saved case ID"}
    if not path.is_file():
        return {}, {}, result
    if path.is_symlink() or not path.resolve().is_relative_to(post_dir.resolve()):
        result["status"] = "FORBIDDEN_INPUT_PATH"
        return {}, {}, result
    book.check()
    try:
        with np.load(path, allow_pickle=False) as saved:
            ids = saved["case_id"]
            if ids.ndim != 1:
                result["status"] = "INVALID_CASE_ID_SHAPE"
                return {}, {}, result
            ids = [str(value) for value in ids]
            expected = set(_expected_cases())
            result.update(case_count=len(ids), expected_cases=len(expected),
                          missing_case_ids=sorted(expected - set(ids)),
                          unexpected_case_ids=sorted(set(ids) - expected),
                          duplicated_case_ids=sorted(case for case, count in Counter(ids).items() if count != 1))
            if len(ids) != 128 or set(ids) != expected or len(set(ids)) != len(ids):
                result["status"] = "INVALID_CASE_ID_COHORT"
                return {}, {}, result
            arrays = {}
            required = [network + "_" + correction for network in NETWORKS for correction in CORRECTIONS]
            result["unexpected_array_keys"] = sorted(set(saved.files) - set(required) - {"case_id"})
            for key in required:
                book.check()
                if key not in saved:
                    result["array_status"][key] = "MISSING"
                elif saved[key].shape != (128, 576):
                    result["array_status"][key] = "INVALID_SHAPE"
                elif not np.issubdtype(saved[key].dtype, np.complexfloating):
                    result["array_status"][key] = "INVALID_COMPLEX_IMAGE_DTYPE"
                else:
                    arrays[key] = saved[key].copy()
                    result["array_status"][key] = "AVAILABLE"
            result["status"] = "AVAILABLE" if len(arrays) == 8 and not result["unexpected_array_keys"] else "INCOMPLETE"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result.update(status="UNREADABLE", error=type(exc).__name__ + ":" + str(exc))
        return {}, {}, result
    book.check()
    return arrays, {case: index for index, case in enumerate(ids)}, result


def _real_mass_delta(arrays, indices, network, correction, case):
    start_key, output_key = network + "_NONE", network + "_" + correction
    if case not in indices or start_key not in arrays or output_key not in arrays:
        return None, None, "MISSING_ARRAY_OR_CASE"
    start, output = arrays[start_key][indices[case]], arrays[output_key][indices[case]]
    if not np.isfinite(start).all() or not np.isfinite(output).all():
        return None, None, "NONFINITE_SAVED_PREDICTION"
    scale = math.sqrt(MASS_VOLUME)
    difference = output - start
    delta = scale * np.concatenate((difference.real, difference.imag))
    start_mass = scale * np.concatenate((start.real, start.imag))
    return delta, float(np.linalg.norm(start_mass)), "COMPLETE"


def _ratio(numerator, denominator, *, zero_status):
    if numerator is None or denominator is None:
        return None, "MISSING_OR_NONFINITE_VALUE"
    if denominator == 0:
        return None, zero_status
    value = _number(numerator / denominator)
    return value, "DEFINED" if value is not None else "UNDEFINED_NONFINITE_RATIO"


def _correction_means(rows, keys, combinations, metrics, book):
    """No accepted-only or defined-only means: all eight cases must contribute."""
    indexed = defaultdict(list)
    for row in rows:
        indexed[tuple(row[key] for key in keys) + (row["case_id"],)].append(row)
    parentrows, familyrows = [], []
    for values in combinations:
        identity = dict(zip(keys, values))
        for metric in metrics:
            selected = []
            for parent in VALIDATION_PARENTS:
                book.check()
                numbers, problems, statuses, input_statuses = [], [], [], []
                for case in _expected_cases((parent,)):
                    matches = indexed.get(values + (case,), [])
                    row = matches[0] if len(matches) == 1 else {}
                    value = _number(row.get(metric))
                    status = row.get(metric + "_status", row.get("metric_status", {}).get(metric))
                    statuses.append(status)
                    input_statuses.append(row.get("input_status"))
                    if len(matches) != 1 or row.get("input_status") != "COMPLETE" or value is None:
                        problems.append({"case_id": case, "status": row.get("input_status", "MISSING")
                                         if row.get("input_status") != "COMPLETE" else status or "UNDEFINED_OR_MISSING_METRIC"})
                    else:
                        numbers.append(value)
                complete = len(numbers) == 8 and not problems
                not_applicable = (all(status == "NOT_APPLICABLE_NONE_CONTROL" for status in statuses) and
                                  all(status == "COMPLETE" for status in input_statuses))
                row = {**identity, "parent_id": parent, "family": _family(parent), "metric": metric,
                       "expected_conditions": 8, "defined_conditions": len(numbers), "complete": complete,
                       "status": "COMPLETE" if complete else "NOT_APPLICABLE" if not_applicable else "HOLD",
                       "mean": float(np.mean(numbers)) if complete else None,
                       "failed_missing_or_undefined_cases": problems,
                       "rule": "eight conditions per parent; rejected zero updates retained", "scope": ACTUAL_UPDATE_SCOPE}
                parentrows.append(row)
                selected.append(row)
            groups = [(family, [p for p in VALIDATION_PARENTS if _family(p) == family]) for family in FAMILIES]
            groups += [("COMPLEX_FAMILIES", [p for p in VALIDATION_PARENTS if _family(p) in COMPLEX_FAMILIES]),
                       ("ALL_FAMILIES", list(VALIDATION_PARENTS))]
            for group, ids in groups:
                items = [row for row in selected if row["parent_id"] in ids]
                complete = all(row["complete"] for row in items)
                not_applicable = all(row["status"] == "NOT_APPLICABLE" for row in items)
                familyrows.append({**identity, "group": group, "metric": metric,
                    "expected_parents": len(ids), "complete_parents": sum(row["complete"] for row in items),
                    "expected_conditions": 8 * len(ids), "complete": complete,
                    "status": "COMPLETE" if complete else "NOT_APPLICABLE" if not_applicable else "HOLD",
                    "mean": float(np.mean([row["mean"] for row in items])) if complete else None,
                    "incomplete_parent_ids": [row["parent_id"] for row in items if not row["complete"]],
                    "rule": "eight-condition parent mean, then equal weight for every specified parent",
                    "scope": ACTUAL_UPDATE_SCOPE})
    return parentrows, familyrows


def _saved_correction_audit(post_dir, post_grouped, config, post_manifest, book):
    """Audit saved steps and action deltas without evaluating any physical model."""
    expected = _expected_metadata()
    path = post_dir / "TRIALS.json"
    trace_status, trace_error, traces = "MISSING", None, []
    if path.is_file():
        if path.is_symlink() or not path.resolve().is_relative_to(post_dir.resolve()):
            trace_status = "FORBIDDEN_INPUT_PATH"
        else:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
                    traces, trace_status = payload, "AVAILABLE"
                else:
                    trace_status = "INVALID_TRACE_SCHEMA"
            except (OSError, ValueError) as exc:
                trace_status, trace_error = "UNREADABLE", type(exc).__name__ + ":" + str(exc)
    arrays, indices, array_manifest = _saved_correction_arrays(post_dir, book)
    trace_grouped, unexpected = defaultdict(list), []
    for trace in traces:
        key = (str(trace.get("network")), str(trace.get("correction")), str(trace.get("case_id")))
        if key[0] not in NETWORKS or key[1] not in CORRECTIONS or key[2] not in expected:
            unexpected.append(trace)
        else:
            trace_grouped[key].append(trace)
    action_names = sorted({str(key) for trace in traces if isinstance(trace.get("actions"), dict)
                           for key in trace["actions"]})
    update_rows, trial_rows, action_rows, vectors = [], [], [], {}
    scalar_fields = ("old_objective", "objective", "old_data_loss", "data_loss", "old_tv", "tv", "lambda",
                     "initial_step", "direction_curvature", "gradient_mass_norm", "correction_wall_seconds")
    for network in NETWORKS:
        for correction in CORRECTIONS:
            for case, meta in expected.items():
                book.check()
                key = (network, correction, case)
                post_matches, trace_matches = post_grouped.get(key, []), trace_grouped.get(key, [])
                post_status, saved_trace_status = (_saved_input_status(post_matches, meta),
                                                  _saved_input_status(trace_matches, meta))
                common_start_status = _saved_input_status(post_grouped.get((network, "NONE", case), []), meta)
                post = post_matches[0] if len(post_matches) == 1 else {}
                trace = trace_matches[0] if len(trace_matches) == 1 else {}
                delta, start_norm, vector_status = _real_mass_delta(arrays, indices, network, correction, case)
                actual_norm = float(np.linalg.norm(delta)) if delta is not None else None
                if vector_status == post_status == common_start_status == "COMPLETE":
                    vectors[key] = delta
                trials = trace.get("trials")
                trials_valid = isinstance(trials, list) and all(isinstance(trial, dict) for trial in trials)
                actions = trace.get("actions")
                actions_valid = isinstance(actions, dict) and all(
                    _number(value) is not None and _number(value) >= 0 and int(_number(value)) == _number(value)
                    for value in actions.values())
                accepted = _flag(trace.get("accepted"))
                consistency = []
                if saved_trace_status == "COMPLETE":
                    if not trials_valid:
                        consistency.append("INVALID_OR_MISSING_TRIAL_LIST")
                    elif _number(post.get("line_trials")) != len(trials):
                        consistency.append("PER_CASE_TRIAL_COUNT_MISMATCH")
                    if accepted is None or _flag(post.get("accepted")) != accepted:
                        consistency.append("ACCEPTANCE_METADATA_MISMATCH")
                    if correction != "NONE" and not actions_valid:
                        consistency.append("INVALID_OR_MISSING_ACTION_COUNTS")
                    if trials_valid and ([_number(trial.get("trial")) for trial in trials] != list(range(len(trials))) or
                                         sum(_flag(trial.get("accepted")) is True for trial in trials) != int(accepted is True)):
                        consistency.append("TRIAL_SEQUENCE_OR_ACCEPTANCE_MISMATCH")
                input_complete = (post_status == saved_trace_status == vector_status == common_start_status == "COMPLETE" and not consistency)
                row = dict(network=network, correction=correction, case_id=case, **meta,
                           sigma=_number(post.get("sigma")), input_status="COMPLETE" if input_complete else "INCOMPLETE",
                           post_row_status=post_status, trace_row_status=saved_trace_status,
                           common_start_row_status=common_start_status,
                           prediction_status=vector_status, trace_consistency_issues=consistency,
                           scope=ACTUAL_UPDATE_SCOPE, common_start_array=network + "_NONE", volume=MASS_VOLUME,
                           mass_definition="sqrt(volume)*concat(real(chi),imag(chi))", common_start_mass_norm=start_norm,
                           actual_update_mass_norm=actual_norm, logged_update_mass_norm=_number(trace.get("update_mass_norm")),
                           accepted=accepted, actions=actions if actions_valid else None,
                           actions_status="RECORDED_COUNTER_DELTAS" if actions_valid else
                                          "NOT_APPLICABLE_NONE_CONTROL" if correction == "NONE" else "NOT_SAVED",
                           online_full_acceptance_audit=_flag(trace.get("online_full_acceptance_audit")),
                           normalized_data_loss_denominator=None,
                           normalized_data_loss_denominator_status="BACKGROUND_DATA_NORM_NOT_SAVED_IN_TRACE",
                           proposal_vector_status="NOT_SAVED; scalar gradient norm and steps only",
                           trial_candidate_vector_status="NOT_SAVED; prox candidate norms only",
                           algorithm_per_trace_status="NOT_RECORDED; selected input manifest retained separately")
                row.update({field: _number(trace.get(field)) for field in scalar_fields})
                row["metric_status"] = {field: "NOT_APPLICABLE_NONE_CONTROL" for field in scalar_fields
                                        if correction == "NONE" and field not in trace}
                row["update_norm_absolute_difference_from_trace"] = (abs(actual_norm - row["logged_update_mass_norm"])
                    if actual_norm is not None and row["logged_update_mass_norm"] is not None else None)
                row["update_to_common_start_norm_ratio"], row["update_to_common_start_norm_ratio_status"] = _ratio(
                    actual_norm, start_norm, zero_status="UNDEFINED_ZERO_COMMON_START_MASS_NORM")
                gradient, step = row["gradient_mass_norm"], row["initial_step"]
                row["initial_gradient_step_mass_norm"] = step * gradient if step is not None and gradient is not None else None
                row["initial_gradient_step_mass_norm_source"] = "derived initial_step*saved_gradient_mass_norm; pre-prox norm"
                if correction == "NONE":
                    row["metric_status"]["initial_gradient_step_mass_norm"] = "NOT_APPLICABLE_NONE_CONTROL"
                trial_list = trials if trials_valid else []
                row.update(trial_count=len(trials) if trials_valid else None,
                           accepted_trial_count=sum(_flag(trial.get("accepted")) is True for trial in trial_list) if trials_valid else None,
                           rejected_trial_count=sum(_flag(trial.get("accepted")) is False for trial in trial_list) if trials_valid else None,
                           trust_rejected_trial_count=sum(trial.get("reason") == "TRUST_RADIUS" for trial in trial_list) if trials_valid else None,
                           full_audit_rejected_trial_count=sum(trial.get("reason") == "FULL_OBJECTIVE_OR_DATA_REJECT" for trial in trial_list) if trials_valid else None,
                           rejection_reasons=dict(Counter(trial.get("reason", "MISSING") for trial in trial_list
                                                          if _flag(trial.get("accepted")) is not True)))
                radii = [_number(trial.get("trust_radius_mass")) for trial in trial_list]
                trust = radii[0] if radii and all(radius is not None and radius == radii[0] for radius in radii) else None
                row.update(trust_radius_mass=trust, trust_radius_source="saved per-trial metadata",
                           trust_radius_status="RECORDED_COMMON_RADIUS" if trust is not None else
                               "NOT_APPLICABLE_NONE_CONTROL" if correction == "NONE" else "MISSING_OR_NONCOMMON_RADII")
                row["returned_update_to_trust_radius_ratio"], row["returned_update_to_trust_radius_ratio_status"] = _ratio(
                    actual_norm, trust, zero_status="UNDEFINED_ZERO_TRUST_RADIUS")
                for metric, field, reducer in (("prox_kkt_max", "prox_kkt", max),
                                               ("prox_normalized_gap_max", "prox_normalized_gap", max),
                                               ("prox_iterations_total", "prox_iterations", sum)):
                    values = [_number(trial.get(field)) for trial in trial_list]
                    row[metric] = reducer(values) if values and all(value is not None for value in values) else None
                    if correction == "NONE":
                        row["metric_status"][metric] = "NOT_APPLICABLE_NONE_CONTROL"
                if correction == "NONE":
                    row["returned_update_to_trust_radius_ratio_status"] = "NOT_APPLICABLE_NONE_CONTROL"
                row["prox_certificate_record_status"] = ("NOT_APPLICABLE_NONE_CONTROL" if correction == "NONE" else
                    "RECORDED_RESIDUALS; NO_NEW_CERTIFICATION" if trial_list and all(
                        _number(trial.get(field)) is not None for trial in trial_list
                        for field in ("prox_kkt", "prox_normalized_gap", "prox_iterations")) else "MISSING_CERTIFICATE_METADATA")
                for action in action_names:
                    metric = "action__" + action
                    row[metric] = _number(actions.get(action, 0)) if actions_valid else None
                    if correction == "NONE":
                        row["metric_status"][metric] = "NOT_APPLICABLE_NONE_CONTROL"
                update_rows.append(row)
    # Retain every source trial/action, including duplicated or unexpected trace records.
    for source_index, trace in enumerate(traces):
        key = (str(trace.get("network")), str(trace.get("correction")), str(trace.get("case_id")))
        trace_meta = expected.get(key[2])
        status = (_saved_input_status(trace_grouped.get(key, []), trace_meta) if trace_meta is not None and
                  key[0] in NETWORKS and key[1] in CORRECTIONS else "UNEXPECTED_TRACE")
        identity = {field: trace.get(field) for field in ("network", "correction", "case_id", "parent_id", "family",
                                                        "background_id", "amplitude", "sigma", "noise")}
        identity.update(source_trace_index=source_index, input_status=status, scope=ACTUAL_UPDATE_SCOPE)
        trials = trace.get("trials")
        if isinstance(trials, list):
            for position, trial in enumerate(trials):
                book.check()
                row = {**identity, "trial_position": position, **trial} if isinstance(trial, dict) else {
                    **identity, "trial_position": position, "raw_trial": trial, "input_status": "INVALID_TRIAL_SCHEMA"}
                gradient, step = _number(trace.get("gradient_mass_norm")), _number(row.get("step"))
                row["gradient_mass_norm"] = gradient
                row["pre_prox_gradient_step_mass_norm"] = gradient * step if gradient is not None and step is not None else None
                row["proposal_vector_status"] = "NOT_SAVED; scalar pre-prox norm only"
                row["trust_norm_ratio"], row["trust_norm_ratio_status"] = _ratio(
                    _number(row.get("update_mass_norm")), _number(row.get("trust_radius_mass")),
                    zero_status="UNDEFINED_ZERO_TRUST_RADIUS")
                row["prox_certificate_record_status"] = ("RECORDED_RESIDUALS; NO_NEW_CERTIFICATION" if all(
                    _number(row.get(field)) is not None for field in ("prox_iterations", "prox_kkt", "prox_normalized_gap"))
                    else "MISSING_CERTIFICATE_METADATA")
                row["full_acceptance_audit_record_status"] = ("NOT_REACHED_TRUST_REJECTION" if row.get("reason") == "TRUST_RADIUS" else
                    "RECORDED" if all(_number(row.get(field)) is not None for field in ("objective", "data_loss", "tv")) else "MISSING")
                trial_rows.append(row)
        if isinstance(trace.get("actions"), dict):
            action_rows.extend({**identity, "action": action, "count": _number(value)}
                               for action, value in trace["actions"].items())
    comparison_rows = []
    for network in NETWORKS:
        for candidate in ("TASK_R32", "L"):
            for case, meta in expected.items():
                book.check()
                f, other = vectors.get((network, "F", case)), vectors.get((network, candidate, case))
                row = dict(network=network, reference="F", candidate=candidate, case_id=case, **meta,
                           input_status="COMPLETE" if f is not None and other is not None else "INCOMPLETE",
                           scope=ACTUAL_UPDATE_SCOPE, common_start_array=network + "_NONE", volume=MASS_VOLUME,
                           relative_difference_rule="||candidate_update-F_update||/||F_update||",
                           cosine_rule="real_mass_dot(candidate_update,F_update)/(||candidate_update||*||F_update||)")
                fn, cn = (float(np.linalg.norm(f)) if f is not None else None,
                          float(np.linalg.norm(other)) if other is not None else None)
                difference = float(np.linalg.norm(other - f)) if f is not None and other is not None else None
                product = fn * cn if fn is not None and cn is not None else None
                row.update(reference_update_mass_norm=fn, candidate_update_mass_norm=cn,
                           actual_update_difference_mass_norm=difference, relative_difference_denominator=fn,
                           cosine_reference_norm_denominator=fn, cosine_candidate_norm_denominator=cn,
                           cosine_denominator_product=product)
                row["relative_update_difference"], row["relative_update_difference_status"] = _ratio(
                    difference, fn, zero_status="UNDEFINED_ZERO_F_UPDATE_NORM")
                row["actual_update_cosine"], row["actual_update_cosine_status"] = _ratio(
                    float(np.dot(other, f)) if f is not None and other is not None else None, product,
                    zero_status="UNDEFINED_ZERO_BOTH_UPDATE_NORMS" if fn == cn == 0 else
                                "UNDEFINED_ZERO_F_UPDATE_NORM" if fn == 0 else "UNDEFINED_ZERO_CANDIDATE_UPDATE_NORM")
                for correction, label in (("F", "reference"), (candidate, "candidate")):
                    matches = trace_grouped.get((network, correction, case), [])
                    row[label + "_trace_status"] = _saved_input_status(matches, meta)
                    row[label + "_accepted"] = _flag(matches[0].get("accepted")) if len(matches) == 1 else None
                comparison_rows.append(row)
    update_metrics = ("actual_update_mass_norm", "logged_update_mass_norm", "update_norm_absolute_difference_from_trace",
                      "common_start_mass_norm", "update_to_common_start_norm_ratio", "gradient_mass_norm", "initial_step",
                      "direction_curvature", "initial_gradient_step_mass_norm", "trial_count", "accepted_trial_count",
                      "rejected_trial_count", "trust_rejected_trial_count", "full_audit_rejected_trial_count", "accepted",
                      "returned_update_to_trust_radius_ratio", "prox_kkt_max", "prox_normalized_gap_max", "prox_iterations_total")
    update_metrics += scalar_fields[:7] + ("correction_wall_seconds",) + tuple("action__" + action for action in action_names)
    update_parent, update_family = _correction_means(update_rows, ("network", "correction"),
        [(network, correction) for network in NETWORKS for correction in CORRECTIONS], update_metrics, book)
    comparison_metrics = ("reference_update_mass_norm", "candidate_update_mass_norm", "actual_update_difference_mass_norm",
                          "relative_update_difference", "actual_update_cosine")
    comparison_parent, comparison_family = _correction_means(comparison_rows, ("network", "reference", "candidate"),
        [(network, "F", candidate) for network in NETWORKS for candidate in ("TASK_R32", "L")], comparison_metrics, book)
    complete = (trace_status == "AVAILABLE" and len(traces) == 1024 and not unexpected and
                array_manifest["status"] == "AVAILABLE" and all(row["input_status"] == "COMPLETE" for row in update_rows) and
                all(row["input_status"] == "COMPLETE" for row in comparison_rows))
    summary = {"status": "COMPLETE" if complete else "INCOMPLETE", "scientific_verdict_performed": False,
        "scope": "saved correction arrays and recorded scalar trials only; no physics or labels",
        "post_input_selection": post_manifest, "volume": MASS_VOLUME,
        "mass_definition": "sqrt(volume)*concat(real(chi),imag(chi)); same-network NONE is common start",
        "case_alignment": array_manifest, "trace_input": {"path": str(path), "status": trace_status, "error": trace_error,
            "rows": len(traces), "expected_rows": 1024, "unexpected_rows": unexpected},
        "update_input_status_counts": dict(Counter(row["input_status"] for row in update_rows)),
        "comparison_input_status_counts": dict(Counter(row["input_status"] for row in comparison_rows)),
        "comparison_denominator_status_counts": {metric: dict(Counter(row[metric + "_status"] for row in comparison_rows))
                                                 for metric in ("relative_update_difference", "actual_update_cosine")},
        "accepted_updates": sum(row["accepted"] is True for row in update_rows),
        "unknown_acceptance": sum(row["accepted"] is None for row in update_rows),
        "trial_rows": len(trial_rows), "trial_reason_counts": dict(Counter(row.get("reason", "MISSING") for row in trial_rows)),
        "prox_certificate_record_status_counts": dict(Counter(row["prox_certificate_record_status"] for row in update_rows)),
        "action_names": action_names, "action_semantics": "saved inclusive per-correction counter deltas; omitted dictionary keys are zero only in an available delta dictionary; NONE counters were not recorded",
        "proposal_vector_status": "NOT_SAVED; only exact scalar step*gradient_norm is derived",
        "trial_candidate_vector_status": "NOT_SAVED; saved final returned updates compared directly",
        "data_normalization_denominator_status": "BACKGROUND_DATA_NORM_NOT_SAVED_IN_TRACE; not reconstructed",
        "prox_certificate_scope": "recorded per-trial KKT/gap/iterations only; no independent solver certification",
        "algorithm_per_trace_status": "NOT_RECORDED; explicit POST_SOURCE input provenance retained without inferring from original config",
        "configured_limits": {key: config.get(key) for key in ("trust_radius_mass", "line_trials", "prox_tolerance",
            "prox_feasibility_tolerance", "prox_max_iter", "prox_wall_limit_seconds")},
        "mean_rule": "all eight conditions averaged within each parent; every specified parent weighted equally; undefined or missing metric yields HOLD, never success-only",
        "diagnostic_proxy_scope": BOX_PROXY_SCOPE,
        "table_paths": ["tables/" + name for name in ("CORRECTION_UPDATE_PER_CASE.csv", "CORRECTION_COMPARISON_PER_CASE.csv",
            "CORRECTION_TRIALS.csv", "CORRECTION_ACTION_COUNTS.csv", "CORRECTION_UPDATE_PER_PARENT.csv",
            "CORRECTION_UPDATE_PER_FAMILY.csv", "CORRECTION_COMPARISON_PER_PARENT.csv", "CORRECTION_COMPARISON_PER_FAMILY.csv")]}
    return summary, {"CORRECTION_UPDATE_PER_CASE.csv": update_rows, "CORRECTION_COMPARISON_PER_CASE.csv": comparison_rows,
        "CORRECTION_TRIALS.csv": trial_rows, "CORRECTION_ACTION_COUNTS.csv": action_rows,
        "CORRECTION_UPDATE_PER_PARENT.csv": update_parent, "CORRECTION_UPDATE_PER_FAMILY.csv": update_family,
        "CORRECTION_COMPARISON_PER_PARENT.csv": comparison_parent, "CORRECTION_COMPARISON_PER_FAMILY.csv": comparison_family}


def _diagnostic_case_table(rows, observable=()):
    indexed = defaultdict(list)
    for row in rows:
        indexed[(str(row.get("state")), str(row.get("case_id")))].append(row)
    observable_index = defaultdict(list)
    for row in observable:
        observable_index[(str(row.get("network")), str(row.get("case_id")))].append(row)
    first_parents = [min(p for p in VALIDATION_PARENTS if _family(p) == family) for family in FAMILIES]
    observable_cases = set(_expected_cases(first_parents, noisy_only=True))
    output = []
    for state in STATE_ORDER:
        for case, meta in _expected_metadata().items():
            matches = indexed.get((state, case), [])
            row = dict(matches[0]) if len(matches) == 1 else {"state": state, "case_id": case, **meta}
            status = ("MISSING" if not matches else "DUPLICATED" if len(matches) != 1 else
                      "FAILED" if not _ok(row) else "METADATA_MISMATCH" if not _metadata_matches(row, meta) else "COMPLETE")
            row["input_row_status"] = status
            row["box_update_metric_scope"] = BOX_PROXY_SCOPE
            row["projected_step_alignment_scope"] = "OFFLINE truth-error alignment of " + BOX_PROXY_SCOPE
            row["descent_error_alignment_scope"] = "OFFLINE saved gradient/truth-error alignment; not online acceptance"
            row["actual_update_comparison_table"] = "CORRECTION_COMPARISON_PER_CASE.csv"
            for metric in OFFLINE_ALIGNMENT_METRICS:
                row[metric + "_status"] = ("FAILED_OR_MISSING_STATE" if status != "COMPLETE" else
                    "DEFINED" if _number(row.get(metric)) is not None else "UNDEFINED_OR_MISSING_SAVED_ALIGNMENT")
            network = state[4:] if state in ("CNN_N3", "CNN_N1") else None
            obs_matches = observable_index.get((network, case), [])
            obs_status = ("NOT_APPLICABLE_STATE" if network is None else
                          "NOT_IN_FIXED_OBSERVABILITY_COHORT" if case not in observable_cases else
                          _saved_input_status(obs_matches, meta))
            row["observability_input_row_status"] = obs_status
            row["observability_gradient_scope"] = "OFFLINE spectrum at matching frozen CNN state; first parent per family, 20dB only"
            for metric in OBSERVABILITY_GRADIENT_METRICS:
                row[metric] = _number(obs_matches[0].get(metric)) if obs_status == "COMPLETE" else None
                row[metric + "_status"] = ("DEFINED" if row[metric] is not None else
                    "UNDEFINED_OR_MISSING_SAVED_VALUE" if obs_status == "COMPLETE" else obs_status)
            for prefix in ("F", "R"):
                metric = prefix + "_feedback_noise_ratio"
                row[metric + "_status"] = ("FAILED_OR_MISSING" if status != "COMPLETE" else
                    "UNDEFINED_ZERO_NOISE_DENOMINATOR" if meta["noise"] == "clean" else
                    "UNDEFINED_NONFINITE_RATIO" if _number(row.get(metric)) is None else "DEFINED")
            output.append(row)
    return output


def _state_tables(rows):
    expected = _expected_metadata()
    indexed = defaultdict(list)
    for row in rows:
        indexed[(str(row.get("state")), str(row.get("case_id")))].append(row)
    states = list(STATE_ORDER) + sorted({str(row.get("state")) for row in rows} - set(STATE_ORDER))
    metrics = ("F_feedback_noise_ratio", "R_feedback_noise_ratio", "F_minus_L_data_norm",
               "R_minus_L_data_norm", "R_forward_error_vs_F", "R_JVP_error_vs_F",
               "R_adjoint_error_vs_F", "R_gradient_cosine_vs_F", "L_gradient_cosine_vs_F",
               "R_box_update_error_vs_F", "L_box_update_error_vs_F") + OFFLINE_ALIGNMENT_METRICS
    output = []
    for state in states:
        for family in FAMILIES:
            parents = [p for p in VALIDATION_PARENTS if _family(p) == family]
            for metric in metrics:
                parentmeans, errors = [], []
                for parent in parents:
                    values = []
                    for case in _expected_cases((parent,), noisy_only=True):
                        matches = indexed.get((state, case), [])
                        row = matches[0] if len(matches) == 1 else {}
                        value = _number(row.get(metric))
                        if len(matches) != 1 or not _ok(row) or not _metadata_matches(row, expected[case]) or value is None:
                            errors.append(case)
                        else:
                            values.append(value)
                    if len(values) == 4:
                        parentmeans.append(float(np.mean(values)))
                complete = len(parentmeans) == 4 and not errors
                output.append(dict(state=state, family=family, metric=metric,
                                   noise="20dB", expected_parents=4, expected_conditions=16,
                                   complete_parents=len(parentmeans), complete=complete,
                                   status="COMPLETE" if complete else "INCOMPLETE",
                                   mean=float(np.mean(parentmeans)) if complete else None,
                                   missing_or_failed_cases=errors,
                                   metric_scope=(BOX_PROXY_SCOPE if "box_update" in metric or "projected_step" in metric else
                                       "OFFLINE saved gradient/truth-error alignment" if metric.startswith("OFFLINE_") else
                                       "saved fixed-state diagnostic"),
                                   rule="four noisy conditions averaged per parent, then equal parent mean"))
    return output


def _observability_tables(rows):
    first = [min(p for p in VALIDATION_PARENTS if _family(p) == family) for family in FAMILIES]
    expected = _expected_metadata()
    indexed = defaultdict(list)
    for row in rows:
        indexed[(str(row.get("network")), str(row.get("case_id")))].append(row)
    metrics = ("strong_error_fraction", "weak_error_fraction", "background_strong_error_fraction",
               "soft_error_fraction", "soft_complement_error_fraction", "soft_cross_energy_fraction",
               "predicted_strong_rank", "background_strong_rank", "background_predicted_strong_overlap") + OBSERVABILITY_GRADIENT_METRICS
    metrics += tuple(metric for metric in ("truth_strong_fraction", "truth_weak_fraction", "q_error_strong", "q_error_weak")
                     if any(metric in row for row in rows))
    output = []
    for network in NETWORKS:
        for parent in first:
            for metric in metrics:
                values, errors = [], []
                for case in _expected_cases((parent,), noisy_only=True):
                    matches = indexed.get((network, case), [])
                    row = matches[0] if len(matches) == 1 else {}
                    value = _number(row.get(metric))
                    if len(matches) != 1 or not _ok(row) or not _metadata_matches(row, expected[case]) or value is None:
                        errors.append(case)
                    else:
                        values.append(value)
                complete = len(values) == 4 and not errors
                output.append(dict(network=network, family=_family(parent), parent_id=parent, metric=metric,
                                   conditions=4, available_conditions=len(values), complete=complete,
                                   status="COMPLETE" if complete else "INCOMPLETE",
                                   mean=float(np.mean(values)) if complete else None,
                                   failed_or_undefined_cases=errors,
                                   scope="fixed first parent; offline local spectrum; not posterior uncertainty"))
    return output


def _save_figure(fig, out, stem, manifest, *, scope):
    import matplotlib.pyplot as plt
    fig.savefig(out / (stem + ".png"), dpi=140, bbox_inches="tight")
    fig.savefig(out / (stem + ".svg"), bbox_inches="tight")
    plt.close(fig)
    manifest.append({"stem": stem, "png": "figures/" + stem + ".png",
                     "svg": "figures/" + stem + ".svg", "scope": scope})


def _paired_figure(rows, out, manifest, book):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, layout="constrained")
    colors = ("#527ca6", "#b96543", "#3b8a71")
    for axis, comparisons in zip(axes, (PAIRS[:3], PAIRS[3:])):
        width = .22
        for index, (reference, candidate) in enumerate(comparisons):
            selected = {row["group"]: row for row in rows if row["network"] == "N3" and
                        row["reference"] == reference and row["candidate"] == candidate}
            for fi, family in enumerate(FAMILIES):
                row = selected[family]
                x = fi + (index - (len(comparisons)-1)/2) * width
                gain = row["gain"]
                label = candidate.replace("TASK_R32", "R32") + " vs " + reference
                if gain is None:
                    axis.text(x, .04, "HOLD", rotation=90, transform=axis.get_xaxis_transform(),
                              ha="center", va="bottom", fontsize=7, color="#888888")
                else:
                    axis.bar(x, 100*gain, width, color=colors[index], label=label if fi == 0 else None)
                    low, high = row["ci95_lower"], row["ci95_upper"]
                    axis.vlines(x, 100*low, 100*high, color="#333333", linewidth=1)
                    axis.hlines([100*low, 100*high], x-.025, x+.025, color="#333333", linewidth=1)
        axis.axhline(0, color="#777777", linewidth=.8)
        axis.set_xticks(range(4), ["Shell", "Asymmetric", "Smooth", "Mixed scale"])
        axis.set_title("Corrections from frozen N3" if axis is axes[0] else "Correction comparisons")
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=.18)
    axes[0].set_ylabel("Relative delta-NRMSE improvement (%)")
    fig.suptitle("Eight conditions per parent; 2000 paired parent resamples (95% interval)", fontsize=11)
    book.check()
    _save_figure(fig, out, "paired_family_correction_gain", manifest,
                 scope="primary N3; all four family cohorts; failed evidence remains incomplete")


def _state_figure(rows, out, manifest, book):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, layout="constrained")
    for axis, family in zip(axes.flat, FAMILIES):
        for metric, label, color in (("F_feedback_noise_ratio", "Exact F minus L", "#b96543"),
                                      ("R_feedback_noise_ratio", "R32 minus L", "#3b8a71")):
            selected = {row["state"]: row for row in rows if row["family"] == family and row["metric"] == metric}
            values = [selected[state]["mean"] for state in STATE_ORDER]
            axis.plot(range(len(STATE_ORDER)), [np.nan if value is None else value for value in values],
                      "o-", label=label, color=color, linewidth=1.2, markersize=4)
            for index, value in enumerate(values):
                if value is None:
                    axis.scatter(index, 0, marker="x", color="#888888")
        axis.set_title(family.replace("_", " "))
        axis.set_yscale("symlog", linthresh=.1)
        axis.grid(alpha=.18)
        axis.set_ylabel("Feedback / noise RMS (parent mean)")
        axis.set_xticks(range(len(STATE_ORDER)), [s.replace("POST_", "").replace("TASK_R32", "R32")
                                                for s in STATE_ORDER], rotation=55, ha="right", fontsize=8)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Fixed states, 20 dB conditions; zero-noise ratios are undefined", fontsize=11)
    book.check()
    _save_figure(fig, out, "feedback_signal_state_ratio", manifest,
                 scope="noisy conditions; equal parent means; raw clean-state ratios retained as undefined")


def _observability_figure(rows, out, manifest, book):
    import matplotlib.pyplot as plt
    lookup = {(row["network"], row["family"], row["metric"]): row["mean"] for row in rows}
    keys = [(network, family) for network in NETWORKS for family in FAMILIES]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    x = np.arange(len(keys))
    values = lambda metric: np.asarray([np.nan if lookup[key + (metric,)] is None else
                                       lookup[key + (metric,)] for key in keys])
    strong, weak = values("strong_error_fraction"), values("weak_error_fraction")
    axes[0].bar(x, strong, label="Strong subspace", color="#527ca6")
    axes[0].bar(x, weak, bottom=strong, label="Weak complement", color="#bdc8d4")
    axes[0].set_title("Hard orthogonal split")
    for index, (metric, label, color) in enumerate((
            ("soft_error_fraction", "Soft weighted", "#527ca6"),
            ("soft_complement_error_fraction", "Soft complement", "#bdc8d4"),
            ("soft_cross_energy_fraction", "Cross energy", "#b96543"))):
        axes[1].bar(x+(index-1)*.23, values(metric), .23, label=label, color=color)
    axes[1].set_title("Soft energies retain the cross term")
    for axis in axes:
        axis.set_ylabel("Fraction of material error energy")
        axis.set_xticks(x, [net+" / "+family.replace("asymmetric_piecewise", "asymmetric").replace("mixed_scale", "mixed")
                           for net, family in keys], rotation=45, ha="right", fontsize=8)
        axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=.18)
    fig.suptitle("Offline local spectra: first parent per family, four noisy conditions", fontsize=11)
    book.check()
    _save_figure(fig, out, "observability_error_fractions", manifest,
                 scope="32 fixed local states; descriptive fractions; not posterior uncertainty")


def _fixed_images(root, data_root, out, manifest, book, post_grouped, post_folder):
    import matplotlib.pyplot as plt
    from a26.data import load_split

    labels = load_split(data_root, "validation")
    expected = _expected_metadata()
    ids = list(map(str, labels["case_id"]))
    first = [min(p for p in VALIDATION_PARENTS if _family(p) == family) for family in FAMILIES]
    selected = _expected_cases(first, noisy_only=True, high_only=True)
    if len(ids) != 128 or len(set(ids)) != 128 or set(ids) != set(expected):
        raise ValueError("A27_FIXED_IMAGE_VALIDATION_COHORT_MISMATCH")
    index = {case: i for i, case in enumerate(ids)}
    prediction_path = Path(root) / "results/a27" / post_folder / "CORRECTED_PREDICTIONS.npz"
    predictions, errors = {}, []
    if prediction_path.is_file():
        with np.load(prediction_path, allow_pickle=False) as archive:
            if not np.array_equal(archive["case_id"], labels["case_id"]):
                raise ValueError("A27_FIXED_IMAGE_CASE_ORDER_MISMATCH")
            unknown = set(archive.files) - {"case_id", *(net + "_" + method for net in NETWORKS for method in CORRECTIONS)}
            if unknown:
                errors.append({"status": "UNREGISTERED_PREDICTION_ARRAY_KEYS", "keys": sorted(unknown)})
            for network in NETWORKS:
                for correction in CORRECTIONS:
                    key = network + "_" + correction
                    if key in archive.files:
                        array = archive[key]
                        if array.shape != (128, 576):
                            raise ValueError("A27_FIXED_IMAGE_PREDICTION_LAYOUT_MISMATCH:" + key)
                        # Retain only the eight fixed images, not the other truth rows.
                        predictions[key] = {case: array[index[case]].copy() for case in selected}
                    else:
                        errors.append({"key": key, "status": "MISSING_PREDICTION_ARRAY"})
    else:
        errors.append({"status": "MISSING_CORRECTED_PREDICTIONS"})
    backgrounds = {}
    for background in BACKGROUND_IDS:
        path = Path(data_root)/"background"/(background + ".npz")
        if (path.is_symlink() or not path.resolve().is_relative_to(Path(data_root).resolve()) or
                "locked" in {part.lower() for part in path.resolve().parts}):
            raise ValueError("A27_FIXED_IMAGE_BACKGROUND_PATH_FORBIDDEN")
        with np.load(path, allow_pickle=False) as archive:
            backgrounds[background] = archive["chi_b"].copy()
    image_rows = []
    for network in NETWORKS:
        panels = []
        for case in selected:
            meta = expected[case]
            bg = backgrounds[meta["background_id"]]
            truth = np.asarray(labels["chi"][index[case]]) - bg
            images = [("TRUTH", truth, "COMPLETE")]
            for correction in CORRECTIONS:
                matches = post_grouped.get((network, correction, case), [])
                row = matches[0] if len(matches) == 1 else {}
                status = ("MISSING_OR_DUPLICATED_CASE" if len(matches) != 1 else
                          "FAILED_CASE" if not _ok(row) else
                          "METADATA_MISMATCH" if not _metadata_matches(row, meta) else
                          "UNDEFINED_METRIC" if any(_number(row.get(metric)) is None for metric in METRICS) else "COMPLETE")
                image = predictions.get(network + "_" + correction, {}).get(case, np.full(576, np.nan + 1j*np.nan)) - bg
                if not np.isfinite(image).all():
                    status = "FAILED_OR_MISSING_PREDICTION"
                images.append((correction, image, status))
            panels.append((case, meta, images))
        for component in ("real", "imag"):
            finite = [np.abs(getattr(image, component)).max() for _, _, images in panels for _, image, _ in images
                      if np.isfinite(image).all()]
            limit = max(finite) if finite and max(finite) > 0 else 1.
            fig, axes = plt.subplots(8, 5, figsize=(13.8, 17), layout="constrained")
            raw = []
            for ri, (case, meta, images) in enumerate(panels):
                for ci, (name, image, status) in enumerate(images):
                    value = getattr(image, component).reshape(24, 24)
                    valid = bool(np.isfinite(value).all()) and status == "COMPLETE"
                    axis = axes[ri, ci]
                    drawn = axis.imshow(value, cmap="RdBu_r", vmin=-limit, vmax=limit, origin="lower")
                    axis.set_xticks([]); axis.set_yticks([])
                    if ri == 0:
                        axis.set_title(name.replace("TASK_R32", "R32"), fontsize=10)
                    if ci == 0:
                        axis.set_ylabel(meta["family"].replace("asymmetric_piecewise", "asymmetric").replace("mixed_scale", "mixed")
                                        + "\n" + meta["background_id"], fontsize=9)
                    if not valid:
                        axis.text(.5, .5, status.lower().replace("_", " "), transform=axis.transAxes,
                                  ha="center", va="center", fontsize=7,
                                  bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
                    for y in range(24):
                        for x in range(24):
                            raw.append(dict(network=network, **meta, case_id=case, image=name, component=component,
                                            y=y, x=x, value=float(value[y, x]) if math.isfinite(value[y, x]) else None,
                                            status=status))
                    image_rows.append(dict(network=network, case_id=case, image=name, component=component,
                                           status=status))
            fig.colorbar(drawn, ax=list(axes.flat), shrink=.45, label="Delta material (" + component + ")")
            fig.suptitle(network + ": fixed first parent / high amplitude / 20 dB / both backgrounds", fontsize=12)
            stem = network.lower() + "_fixed_delta_" + component
            _write_csv(out/(stem + "_plot_data.csv"), raw, book)
            _save_figure(fig, out, stem, manifest, scope="metadata-fixed images; validation truth only for these eight cases")
            book.check()
    return {"rule": "first parent each family; amplitude 1.2; 20dB; weak and ring",
            "selected_case_ids": selected, "truth_cases_used": len(selected),
            "predictions_missing": errors, "image_statuses": image_rows}


def _json_timing_rows(value):
    if isinstance(value, list):
        return [row for item in value for row in _json_timing_rows(item)]
    if not isinstance(value, dict):
        return []
    if "native_full_inference_seconds" in value:
        return [value]
    return [row for key, item in value.items() if key not in {"stage_rows", "named_stages", "action_counts"}
            for row in _json_timing_rows(item)]


def _historical_a26_quality(a26_root):
    """Read saved scalar metrics only, retaining complete parent coverage."""
    path = Path(a26_root)/"results/a26/evaluation/261009/PER_CASE_METRICS.csv"
    rows, status = _read_csv(path)
    expected = _expected_metadata()
    output = {}
    for network in ("N1", "N2", "N3"):
        grouped = defaultdict(list)
        for row in rows:
            if row.get("method") == network:
                grouped[str(row.get("case_id"))].append(row)
        means = []
        for parent in VALIDATION_PARENTS:
            values = []
            for case in _expected_cases((parent,)):
                matches = grouped.get(case, [])
                row = matches[0] if len(matches) == 1 else {}
                value = _number(row.get("delta_nrmse"))
                meta = expected[case]
                if (len(matches) == 1 and _ok(row) and value is not None and
                        _number(row.get("parent_id")) == parent and row.get("family") == meta["family"] and
                        row.get("background_id") == meta["background_id"]):
                    values.append(value)
            if len(values) == 8:
                means.append(float(np.mean(values)))
        complete = status == "AVAILABLE" and len(means) == 16 and set(grouped) == set(expected)
        output[network] = {"delta_nrmse": float(np.mean(means)) if complete else None,
                           "status": "COMPLETE_HISTORICAL_QUALITY" if complete else "HISTORICAL_QUALITY_INCOMPLETE",
                           "source": str(path), "scope": "frozen A26 scalar metrics; not a new A27 image-quality evaluation"}
    return output


def _latency_table(root, a26_root, familyrows, config, book, post_folder):
    base = Path(root)/"results/a27/latency"
    allrows, sources = [], []
    for path in sorted(base.glob("*.csv")):
        rows, _ = _read_csv(path)
        if rows and any("native_full_inference_seconds" in row for row in rows):
            allrows.extend(rows); sources.append(str(path.relative_to(root)))
    # Prefer raw CSV if available, preventing duplication of JSON summaries.
    if not allrows:
        for path in sorted(base.glob("*.json")):
            rows = _json_timing_rows(json.loads(path.read_text(encoding="utf-8")))
            if rows:
                allrows.extend(rows); sources.append(str(path.relative_to(root)))
    grouped = defaultdict(list)
    for row in allrows:
        if row.get("timing_mode", "native") != "native":
            continue
        method = str(row.get("network", row.get("method", "UNKNOWN")))
        arm = str(row.get("arm", row.get("profile_arm", method)))
        correction = str(row.get("correction", "NONE"))
        regime = str(row.get("regime", row.get("phase", row.get("timing_regime", "native"))))
        grouped[(arm, method, correction, regime)].append(row)
    has_encoder = any((base/name).is_file() for name in ("NATIVE_INFERENCE.csv", "NATIVE_INFERENCE.json"))
    has_post = any((base/name).is_file() for name in ("POST_RAW_TO_IMAGE.csv", "POST_RAW_TO_IMAGE.json"))
    if has_encoder:
        for arm, method in LATENCY_ENCODER_ARMS:
            grouped.setdefault((arm, method, "NONE", "native"), [])
    if has_post:
        for correction in CORRECTIONS:
            grouped.setdefault(("POST_"+correction, "N3", correction, "native"), [])
    failures_path = base/"FAILURES.json"
    arm_methods = dict(LATENCY_ENCODER_ARMS)
    if failures_path.is_file():
        for row in json.loads(failures_path.read_text(encoding="utf-8")):
            if row.get("timing_mode", "native") != "native":
                continue
            arm = str(row.get("arm", "UNKNOWN"))
            method = "N3" if arm.startswith("POST_") else arm_methods.get(arm, "UNKNOWN")
            correction = arm[5:] if arm.startswith("POST_") else "NONE"
            key = (arm, method, correction, "native")
            if key not in grouped:
                continue
            identity = (str(row.get("case_id")), _number(row.get("repeat")))
            if not any((str(old.get("case_id")), _number(old.get("repeat"))) == identity for old in grouped[key]):
                grouped[key].append({**row, "network": method, "correction": correction,
                                     "status": "FAILED", "timing_mode": "native",
                                     "native_full_inference_seconds": None,
                                     "measurement_source": "FAILURES.json; omitted failed native row"})
        sources.append(str(failures_path.relative_to(root)))
    quality = {(row["network"], row["correction"]): row for row in familyrows if row["group"] == "ALL_FAMILIES"}
    historical = _historical_a26_quality(a26_root) if grouped else {}
    output = []
    first = [min(p for p in VALIDATION_PARENTS if _family(p) == family) for family in FAMILIES]
    representative_cases = set(_expected_cases(first, noisy_only=True))
    repeats = int(config.get("latency_repeats", 3))
    if repeats != 3:
        raise ValueError("A27_FROZEN_THREE_LATENCY_REPEATS_REQUIRED")
    expected_samples = {(case, repeat) for case in representative_cases for repeat in range(repeats)}
    for (arm, method, correction, regime), rows in sorted(grouped.items()):
        book.check()
        values = [_number(row.get("native_full_inference_seconds")) for row in rows]
        invalid = [index for index, (row, value) in enumerate(zip(rows, values))
                   if not _ok(row) or value is None or value < 0]
        identities = Counter((str(row.get("case_id")), _number(row.get("repeat"))) for row in rows)
        missing = sorted(expected_samples-set(identities))
        unexpected = sorted(set(identities)-expected_samples, key=str)
        duplicate = [str(key) for key, count in identities.items() if count != 1]
        complete = not invalid and not missing and not unexpected and not duplicate and bool(rows)
        quality_row = quality.get((method, correction), {})
        scope = "all 16 parents, 8 conditions each; post PER_CASE.csv"
        quality_source = "results/a27/"+post_folder+"/PER_CASE.csv"
        if not quality_row and correction == "NONE":
            quality_row = historical.get(method, {})
            scope = quality_row.get("scope", "quality unavailable")
            quality_source = quality_row.get("source")
            optimized = any(token in arm for token in ("NO_BETA", "LU_REUSE", "COMBINED"))
            equivalent = all(_flag(row.get("lossless_equivalence_passed", row.get("equivalence_passed"))) is True for row in rows)
            if optimized and not equivalent:
                quality_row = {"delta_nrmse": None, "status": "HISTORICAL_BASELINE_ONLY_EQUIVALENCE_NOT_RECORDED"}
        latency_post_sources = sorted({str(row.get("post_source_folder", "post")) for row in rows}) if correction != "NONE" else []
        if correction != "NONE" and latency_post_sources != [post_folder]:
            quality_row = {"delta_nrmse": None, "status": "LATENCY_AND_QUALITY_POST_SOURCE_MISMATCH"}
        output.append(dict(arm=arm, network=method, correction=correction, regime=regime,
                           provided_samples=len(rows), failed_or_undefined_samples=len(invalid),
                           expected_samples=len(expected_samples), missing_samples=len(missing),
                           missing_sample_ids=[{"case_id": case, "repeat": repeat} for case, repeat in missing],
                           unexpected_sample_ids=[str(item) for item in unexpected], duplicated_sample_ids=duplicate,
                           latency_status="COMPLETE_FIXED_COHORT" if complete else "HOLD",
                           native_full_inference_seconds=float(np.median(values)) if complete else None,
                           delta_nrmse=quality_row.get("delta_nrmse"),
                           quality_status=quality_row.get("status", "QUALITY_NOT_AVAILABLE"),
                           quality_scope=scope, quality_source=quality_source,
                           latency_post_source_folders=latency_post_sources,
                           latency_scope="fixed 16 representative cases x 3 repeats; no success-only median; no stage-event sums"))
    return output, sources


def _latency_figure(rows, out, manifest, book):
    import matplotlib.pyplot as plt
    visible = [row for row in rows if row["native_full_inference_seconds"] is not None and row["delta_nrmse"] is not None]
    if not visible:
        return "NOT_RUN_OR_NO_COMPLETE_QUALITY_PAIR"
    # Separate closely spaced lossless variants from post-correction quality.
    # Every raw row stays in the CSV; duplicate instrumented baseline views
    # are omitted only from this display, not from any timing statistic.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), layout="constrained")
    panels = (("ORIGINAL_N1", "ORIGINAL_N2", "ORIGINAL_N3", "N2_NO_BETA_ONLY", "N2_LU_REUSE_ONLY", "N2_COMBINED"),
              ("POST_NONE", "POST_L", "POST_F", "POST_TASK_R32"))
    names = {"ORIGINAL_N1":"N1 original", "ORIGINAL_N2":"N2 original", "ORIGINAL_N3":"N3 original",
             "N2_NO_BETA_ONLY":"N2 no-beta", "N2_LU_REUSE_ONLY":"N2 LU reuse", "N2_COMBINED":"N2 combined",
             "POST_NONE":"N3, no correction", "POST_L":"N3 + L", "POST_F":"N3 + F", "POST_TASK_R32":"N3 + R32"}
    by_arm = {row["arm"]: row for row in visible}
    for axis, arms, title in zip(axes, panels, ("Frozen encoders and lossless variants", "Raw measurement to N3 plus correction")):
        for index, arm in enumerate(arms):
            if arm not in by_arm:
                continue
            row = by_arm[arm]
            ms = 1000*row["native_full_inference_seconds"]
            axis.scatter(ms, row["delta_nrmse"], s=60, marker=("o", "s", "^", "D", "X", "*")[index],
                         label=names[arm]+f" ({ms:.1f} ms)")
        axis.set_xlabel("Native measurement-to-image median (ms)")
        axis.set_title(title)
        axis.grid(alpha=.2)
        axis.legend(loc="upper center", bbox_to_anchor=(.5, -.17), ncol=2, fontsize=8)
    axes[0].set_ylabel("All-parent delta NRMSE")
    axes[1].set_ylabel("All-parent delta NRMSE")
    fig.suptitle("Quality: 16 parents / 128 conditions; timing: 4 fixed parents / 16 noisy conditions x 3", fontsize=10)
    book.check()
    _save_figure(fig, out, "latency_quality", manifest,
                 scope="provided native wall only; incomplete rows retained in raw table")
    return "CREATED"


def build_artifacts(root, data_root, a26_root, config, book, device):
    """Read saved outputs, write evidence artifacts, and return their paths.

    Signature matches the parent pipeline stage callback. ``a26_root`` is
    retained for that interface; no old model, training data or locked split is
    opened. All loading, aggregation and rendering is inside one charged span.
    """
    root, data_root = Path(root), Path(data_root)
    with book.span("a27_build_saved_evidence_artifacts", artifact_builds=1):
        if config.get("primary_network", "N3") != "N3" or tuple(config.get("corrections", CORRECTIONS)) != CORRECTIONS:
            raise ValueError("A27_FROZEN_ARTIFACT_ARMS_REQUIRED")
        import matplotlib
        matplotlib.use("Agg")
        post_folder, post_manifest = _post_source(root)
        post_dir = root/"results/a27"/post_folder
        if post_dir.is_symlink() or not post_dir.resolve().is_relative_to(root.resolve()):
            raise ValueError("A27_POST_INPUT_FOLDER_PATH_FORBIDDEN")
        out = root/"results/a27"/("artifacts" if post_folder == "post" else "artifacts_post_fgp")
        tables, figures = out/"tables", out/"figures"
        tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
        post, post_status = _read_csv(post_dir/"PER_CASE.csv")
        diagnostics, diagnostic_status = _read_csv(root/"results/a27/diagnostics/PER_STATE.csv")
        observable, observability_status = _read_csv(root/"results/a27/observability/PER_STATE.csv")
        parentrows, grouped, unexpected = _parent_tables(post)
        familyrows = _family_tables(parentrows)
        paired, gainrows = _paired(parentrows, config, book)
        weakparents, _, _ = _parent_tables(post, background="weak")
        weakfamilies = _family_tables(weakparents)
        for row in weakfamilies:
            row.update(descriptive_only=True, background_slice="weak",
                       scope="POST_HOC_WEAK_BACKGROUND_SLICE; never substitutes for full cohort")
        weakpaired, weakgains = _paired(weakparents, config, book, descriptive_weak=True)
        counts = _case_counts(grouped)
        correction_audit, correction_tables = _saved_correction_audit(post_dir, grouped, config, post_manifest, book)
        for name, rows in correction_tables.items():
            _write_csv(tables/name, rows, book)
        write_json(out/"CORRECTION_AUDIT.json", correction_audit)
        diagnostic_cases = _diagnostic_case_table(diagnostics, observable)
        state_rows, observable_rows = _state_tables(diagnostics), _observability_tables(observable)
        _write_csv(tables/"PER_PARENT.csv", parentrows, book)
        _write_csv(tables/"PER_FAMILY.csv", familyrows, book)
        _write_csv(tables/"WEAK_ONLY_DESCRIPTIVE_PER_PARENT.csv", weakparents, book)
        _write_csv(tables/"WEAK_ONLY_DESCRIPTIVE_PER_FAMILY.csv", weakfamilies, book)
        _write_csv(tables/"WEAK_ONLY_DESCRIPTIVE_PAIRED_DELTA.csv", weakgains, book)
        _write_csv(tables/"UPDATE_AND_FAILURE_COUNTS.csv", counts, book)
        _write_csv(tables/"STATE_PARENT_MEANS.csv", state_rows, book)
        _write_csv(tables/"DIAGNOSTICS_PER_STATE.csv", diagnostic_cases, book)
        _write_csv(tables/"OBSERVABILITY_PER_STATE.csv", observable, book)
        _write_csv(tables/"OBSERVABILITY_CONDITION_MEANS.csv", observable_rows, book)
        _write_csv(figures/"paired_family_correction_gain_plot_data.csv", gainrows, book)
        _write_csv(figures/"feedback_signal_state_ratio_plot_data.csv", state_rows, book)
        _write_csv(figures/"observability_error_fractions_plot_data.csv", observable_rows, book)
        write_json(out/"PAIRED_STATISTICS.json", paired)
        write_json(out/"WEAK_ONLY_DESCRIPTIVE.json", weakpaired)
        manifest = []
        _paired_figure(gainrows, figures, manifest, book)
        _state_figure(state_rows, figures, manifest, book)
        _observability_figure(observable_rows, figures, manifest, book)
        images = _fixed_images(root, data_root, figures, manifest, book, grouped, post_folder)
        latency_rows, latency_sources = _latency_table(root, a26_root, familyrows, config, book, post_folder)
        _write_csv(tables/"LATENCY_QUALITY.csv", latency_rows, book)
        _write_csv(figures/"latency_quality_plot_data.csv", latency_rows, book)
        latency_figure_status = _latency_figure(latency_rows, figures, manifest, book)
        complete = (post_status == "AVAILABLE" and len(post) == 1024 and all(row["complete"] for row in parentrows)
                    and not unexpected and diagnostic_status == "AVAILABLE" and observability_status == "AVAILABLE"
                    and len(diagnostics) == 1280 and len(observable) == 32
                    and all(row["input_row_status"] == "COMPLETE" for row in diagnostic_cases)
                    and all(row["complete"] for row in state_rows + observable_rows)
                    and correction_audit["status"] == "COMPLETE"
                    and not images["predictions_missing"]
                    and all(row["status"] == "COMPLETE" for row in images["image_statuses"]))
        summary = {"status": "COMPLETE" if complete else "INCOMPLETE", "protocol_id": config["protocol_id"],
                   "scope": "exposed development; saved-output aggregation and fixed images",
                   "scientific_verdict_performed": False, "new_labels": 0, "training_opened": False,
                   "locked_opened": False, "bootstrap_samples": 2000, "bootstrap_unit": "independent_parent",
                   "primary_network": "N3", "primary_complex_parent_count": 12,
                   "post_input_selection": post_manifest,
                   "post_input_path": str((post_dir/"PER_CASE.csv").relative_to(root)),
                   "full_cohort_pair_status": "COMPLETE" if all(
                       comparison["ALL_FAMILIES"]["delta_nrmse"]["coverage_complete"]
                       for network in paired["networks"].values() for comparison in network.values()) else "HOLD",
                   "weak_only_descriptive": {"scope": "POST_HOC_WEAK_BACKGROUND_SLICE",
                       "conditions_per_parent": 4, "scientific_verdict_performed": False,
                       "substitute_for_full_cohort": False, "statistics_path": "WEAK_ONLY_DESCRIPTIVE.json",
                       "complete_parent_rows": sum(row["complete"] for row in weakparents),
                       "expected_parent_rows": len(weakparents)},
                   "input_status": {"post": post_status, "diagnostics": diagnostic_status,
                                    "observability": observability_status, "latency_sources": latency_sources,
                                    "correction_audit": correction_audit["status"]},
                   "correction_audit": {"status": correction_audit["status"], "path": "CORRECTION_AUDIT.json",
                       "table_paths": correction_audit["table_paths"], "actual_update_scope": ACTUAL_UPDATE_SCOPE,
                       "diagnostic_proxy_scope": BOX_PROXY_SCOPE,
                       "comparison_denominator_status_counts": correction_audit["comparison_denominator_status_counts"]},
                   "post_rows": len(post), "expected_post_rows": 1024,
                   "parent_rows": len(parentrows), "incomplete_parent_rows": sum(not row["complete"] for row in parentrows),
                   "diagnostic_rows": len(diagnostics), "expected_diagnostic_rows": 1280,
                   "diagnostic_case_status_counts": dict(Counter(row["input_row_status"] for row in diagnostic_cases)),
                   "observability_rows": len(observable), "expected_observability_rows": 32,
                   "primary_complex_comparisons": {key: value["COMPLEX_FAMILIES"]["delta_nrmse"]
                                                     for key, value in paired["networks"]["N3"].items()},
                   "unexpected_post_rows": unexpected, "case_counts": counts, "fixed_images": images,
                   "latency_figure_status": latency_figure_status, "figures": manifest,
                   "table_paths": ["tables/" + p.name for p in sorted(tables.glob("*.csv"))],
                   "paired_statistics_path": "PAIRED_STATISTICS.json"}
        write_json(out/"FIGURE_MANIFEST.json", {"figures": manifest, "fixed_images": images,
                                               "post_input_selection": post_manifest})
        write_json(out/"ARTIFACT_MANIFEST.json", {"post_input_selection": post_manifest,
            "summary": "SUMMARY.json", "correction_audit": "CORRECTION_AUDIT.json",
            "paired_statistics": "PAIRED_STATISTICS.json", "tables": summary["table_paths"],
            "figures": manifest, "diagnostic_proxy_scope": BOX_PROXY_SCOPE,
            "scientific_verdict_performed": False})
        write_json(out/"SUMMARY.json", summary)
        return {"status": summary["status"], "summary_path": str((out/"SUMMARY.json").relative_to(root)),
                "paired_statistics_path": str((out/"PAIRED_STATISTICS.json").relative_to(root)),
                "figure_manifest_path": str((out/"FIGURE_MANIFEST.json").relative_to(root)),
                "artifact_manifest_path": str((out/"ARTIFACT_MANIFEST.json").relative_to(root)),
                "correction_audit_path": str((out/"CORRECTION_AUDIT.json").relative_to(root)),
                "table_directory": str(tables.relative_to(root)), "figure_directory": str(figures.relative_to(root)),
                "scientific_verdict_performed": False, "new_labels": 0}


stage_report = build_artifacts
