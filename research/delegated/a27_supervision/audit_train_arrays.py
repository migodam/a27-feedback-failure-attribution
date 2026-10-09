"""Read existing, metadata-admitted TRAIN arrays only; never generate labels.

This audit writes scalar diagnostics/schema to its owned directory. It never
opens validation or locked archives, calls a forward solver, launches training,
changes a rank, or serializes reconstructed field/current teacher arrays.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from a26.data import load_split


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GAUSSIAN = ROOT.parents[1]
DATA = GAUSSIAN / "A25/a25-adaptive-om-feedback/data"
A26 = GAUSSIAN / "A26/a26-om-feedback-prior-completion"


def safe_path(path, root):
    path, root = Path(path), Path(root).resolve()
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or any(
        p.lower() == "locked" or p.lower().startswith(("locked_id", "locked_ood"))
        for p in resolved.parts
    ):
        raise RuntimeError("LOCKED_OR_EXTERNAL_PATH_REJECTED")
    return resolved


def schema(value):
    value = np.asarray(value)
    return {"shape": list(value.shape), "dtype": str(value.dtype), "bytes": value.nbytes}


def rel_rows(diff, ref):
    return np.linalg.norm(diff.reshape(len(diff), -1), axis=-1) / np.maximum(
        np.linalg.norm(ref.reshape(len(ref), -1), axis=-1), np.finfo(np.float64).tiny
    )


def stats(value):
    value = np.asarray(value, dtype=np.float64)
    return {"min": float(value.min()), "mean": float(value.mean()), "max": float(value.max())}


def action(matrix, value):
    # Existing fixed matrix action only, with source axis kept last.
    columns = value.transpose(1, 0, 2).reshape(value.shape[1], -1)
    out = matrix @ columns
    return out.reshape(matrix.shape[0], len(value), value.shape[2]).transpose(1, 0, 2)


def main():
    wall, cpu = time.perf_counter(), time.process_time()
    # The inherited A26 loader validates public manifest, paths, identities and
    # every TRAIN condition before material/measurement payload access.
    admitted = load_split(DATA, "train")
    train_path = safe_path(DATA / "train.npz", DATA)
    with np.load(train_path, allow_pickle=False) as z:
        for key in ("split", "parent_id", "case_id", "background_id", "variant_id"):
            if not np.array_equal(z[key], admitted[key]):
                raise RuntimeError("TRAIN_IDENTITY_CHANGED_DURING_AUDIT")
        current = z["current"].copy()
        clean = z["clean"].copy()
        archive_schema = {key: schema(z[key]) for key in sorted(z.files)}
    if current.shape != (512, 576, 8) or current.dtype != np.complex128 or not np.isfinite(current).all():
        raise RuntimeError("FULL_TRAIN_CURRENT_SCHEMA_FAILED")
    if clean.shape != (512, 32, 8) or clean.dtype != np.complex128 or not np.isfinite(clean).all():
        raise RuntimeError("TRAIN_CLEAN_DATA_SCHEMA_FAILED")
    clean_rows = np.flatnonzero(admitted["sigma"] == 0)
    noisy_rows = np.flatnonzero(admitted["sigma"] > 0)
    row_by_case = {str(c): i for i, c in enumerate(admitted["case_id"])}
    pairs = [(i, row_by_case[str(admitted["case_id"][i]).removesuffix("_clean") + "_snr20"])
             for i in clean_rows]
    noise_duplicate_checks = {
        key: bool(all(np.array_equal(value[i], value[j]) for i, j in pairs))
        for key, value in {"chi": admitted["chi"], "current": current, "clean": clean}.items()
    }
    result = {
        "status": "READ_ONLY_TRAIN_AUDIT_COMPLETE",
        "source_commit": "2603f3bd16c48e48587fedc4363975779def17f5",
        "data_root": str(DATA), "archive_path": str(train_path),
        "parents": sorted(map(int, np.unique(admitted["parent_id"]))),
        "observation_rows": len(current), "unique_clean_states": len(clean_rows),
        "noise_free_rows": len(clean_rows), "noisy_rows": len(noisy_rows),
        "states_per_parent": 4, "source_count": 8, "spatial_cells": 576,
        "scalar_current_values_unique_states": int(len(clean_rows) * 576 * 8),
        "archive_schema": archive_schema,
        "field_saved_as_archive_key": "field" in archive_schema,
        "relative_w_saved_as_archive_key": "w" in archive_schema,
        "noise_variants_share_identical_teacher": noise_duplicate_checks,
        "backgrounds": {}, "family_metrics": {}, "feature_caches": {}, "checkpoints": {},
        "locked_files_opened": 0, "validation_files_opened": 0,
        "new_labels": 0, "full_forward_calls": 0, "linear_solves": 0,
        "training_updates": 0, "hash_checks": 0, "reconstructed_teacher_arrays_saved": False,
    }
    h_square = j_square = e_square = w_square = 0.0
    h_count = j_count = e_count = w_count = 0
    teacher_stats = []
    for bg in ("weak", "ring"):
        bg_path = safe_path(DATA / "background" / (bg + ".npz"), DATA)
        keys = ("chi_b", "eb", "yb", "Db", "G", "S", "incident", "L_b", "volume", "grid")
        with np.load(bg_path, allow_pickle=False) as z:
            known = {key: z[key].copy() for key in keys}
            known_schema = {key: schema(z[key]) for key in sorted(z.files)}
        chi_b, eb, incident, g, lb = (known[k] for k in ("chi_b", "eb", "incident", "G", "L_b"))
        cb = chi_b[:, None] * eb
        rows = clean_rows[admitted["background_id"][clean_rows] == bg]
        j, chi = current[rows], admitted["chi"][rows]
        e = incident[None] + action(g, j)
        h = chi - chi_b[None]
        w = h[..., None] * e
        w_from_current = action(lb, j - cb[None])
        data_from_current = action(known["S"], j)
        data_from_w = known["yb"][None] + action(known["Db"], w)
        denominator = np.sum(np.abs(e) ** 2, axis=-1)
        ratio = np.sum(np.conj(e) * j, axis=-1) / denominator
        source_current_norm = np.linalg.norm(j, axis=1)
        source_w_norm = np.linalg.norm(w, axis=1)
        source_field_norm = np.linalg.norm(e, axis=1)
        checks = {
            "full_constitutive_relative_current": stats(rel_rows(j - chi[..., None] * e, j)),
            "relative_w_identity": stats(rel_rows(w_from_current - w, w)),
            "clean_measurement_from_current": stats(rel_rows(data_from_current - clean[rows], clean[rows])),
            "clean_measurement_from_w": stats(rel_rows(data_from_w - clean[rows], clean[rows])),
            "common_material_ratio_error_relative_chi": stats(rel_rows(ratio - chi, chi)),
            "Lb_identity_abs_max": float(np.max(np.abs(lb - (np.eye(576) - chi_b[:, None] * g)))),
            "background_field_identity_relative": float(np.linalg.norm(eb - incident - g @ cb) / np.linalg.norm(eb)),
        }
        result["backgrounds"][bg] = {
            "path": str(bg_path), "schema": known_schema, "clean_states": len(rows),
            "checks": checks,
            "common_ratio_denominator_abs": stats(denominator),
            "common_ratio_min_to_state_max": stats(denominator.min(-1) / denominator.max(-1)),
            "source_current_norm": stats(source_current_norm),
            "source_w_norm": stats(source_w_norm), "source_field_norm": stats(source_field_norm),
        }
        for r, ei, hi, ji, wi, di in zip(rows, e, h, j, w, denominator):
            teacher_stats.append({
                "case_id": str(admitted["case_id"][r]), "parent_id": int(admitted["parent_id"][r]),
                "family": str(admitted["family"][r]), "background_id": bg,
                "amplitude": float(admitted["amplitude"][r]),
                "h_rms": float(np.sqrt(np.mean(np.abs(hi) ** 2))),
                "current_rms": float(np.sqrt(np.mean(np.abs(ji) ** 2))),
                "field_rms": float(np.sqrt(np.mean(np.abs(ei) ** 2))),
                "w_rms": float(np.sqrt(np.mean(np.abs(wi) ** 2))),
                "ratio_den_min_to_max": float(di.min() / di.max()),
                "relative_w_support_fraction": float(np.mean(np.abs(hi) > 1e-12)),
            })
        for value, name in ((h, "h"), (j, "j"), (e, "e"), (w, "w")):
            square = float(np.sum(np.abs(value) ** 2))
            if name == "h": h_square += square; h_count += value.size
            if name == "j": j_square += square; j_count += value.size
            if name == "e": e_square += square; e_count += value.size
            if name == "w": w_square += square; w_count += value.size
    result["train_only_rms_unique_states"] = {
        "h": float(np.sqrt(h_square / h_count)), "current": float(np.sqrt(j_square / j_count)),
        "total_field": float(np.sqrt(e_square / e_count)), "relative_w": float(np.sqrt(w_square / w_count)),
        "definition": "sqrt(mean(abs(value)**2)) over 256 noise-independent TRAIN states; equal noise replication gives the same 512-row scales",
    }
    for family in ("shell", "asymmetric_piecewise", "smooth", "mixed_scale"):
        selected = [row for row in teacher_stats if row["family"] == family]
        result["family_metrics"][family] = {"unique_states": len(selected), "parents": len({row["parent_id"] for row in selected})}
        for key in ("h_rms", "current_rms", "field_rms", "w_rms", "ratio_den_min_to_max", "relative_w_support_fraction"):
            result["family_metrics"][family][key] = stats([row[key] for row in selected])
    label_paths = sorted((DATA / "labels/train").glob("*.npz"))
    expected_labels = {str(admitted["case_id"][r]).removesuffix("_clean") for r in clean_rows}
    result["train_label_shards"] = {"count": len(label_paths), "filename_set_matches_clean_states": {p.stem for p in label_paths} == expected_labels}
    # One representative original shard confirms export schema and provenance.
    sample_path = safe_path(DATA / "labels/train/p250001_weak_a0.npz", DATA)
    with np.load(sample_path, allow_pickle=False) as z:
        if str(z["manifest_id"]) != str(admitted["manifest_id"]):
            raise RuntimeError("ORIGINAL_TRAIN_LABEL_MANIFEST_MISMATCH")
        result["train_label_shards"]["sample_path"] = str(sample_path)
        result["train_label_shards"]["sample_schema"] = {key: schema(z[key]) for key in sorted(z.files)}
    for method in ("N1", "N2", "N3"):
        path = safe_path(A26 / "results/a26/cache" / ("train_" + method + ".npz"), A26)
        with np.load(path, allow_pickle=False) as z:
            if set(map(str, np.atleast_1d(z["split"]))) != {"train"} or not np.array_equal(z["parent_id"], admitted["parent_id"]):
                raise RuntimeError("TRAIN_FEATURE_IDENTITY_MISMATCH")
            result["feature_caches"][method] = {
                "path": str(path), "schema": {key: schema(z[key]) for key in sorted(z.files)},
                "teacher_keys_present": sorted(set(z.files) & {"chi", "current", "field", "w", "clean"}),
            }
    repro_root = A26 / "results/a26/repro_inputs/data"
    repro_path = safe_path(repro_root / "train.npz", repro_root)
    with np.load(repro_path, allow_pickle=False) as z:
        if set(map(str, np.atleast_1d(z["split"]))) != {"train"} or not np.array_equal(z["parent_id"], admitted["parent_id"]):
            raise RuntimeError("REPRO_TRAIN_IDENTITY_MISMATCH")
        result["a26_repackaged_train"] = {
            "path": str(repro_path), "keys": sorted(z.files),
            "current_teacher_present": "current" in z.files,
            "field_teacher_present": "field" in z.files,
            "relative_w_teacher_present": "w" in z.files,
            "chi_matches_a25_train": bool(np.array_equal(z["chi"], admitted["chi"])),
        }
    import torch
    for method in ("N1", "N2", "N3"):
        folder = A26 / "results/a26/learning/261009" / method
        paths = {kind: folder / (kind + ".pt") for kind in ("best", "middle", "final")}
        payload = torch.load(safe_path(paths["best"], A26), map_location="cpu", weights_only=True)
        manifest = json.loads((folder / "training_manifest.json").read_text())
        result["checkpoints"][method] = {
            "available_paths": {kind: {"path": str(path), "exists": path.is_file()} for kind, path in paths.items()},
            "inspected_kind": "best", "payload_keys": sorted(payload), "step": int(payload["step"]),
            "targets_in_checkpoint": payload.get("targets_in_checkpoint"),
            "model_parameter_scalars": sum(value.numel() for value in payload["state_dict"].values()),
            "model_state_keys": sorted(payload["state_dict"]),
            "material_scale": float(payload["material_scale"]),
            "checkpoint_metadata_supervision": payload["metadata"].get("supervision"),
            "training_manifest_supervision": manifest.get("supervision"),
            "training_manifest_physical_backward_calls": manifest.get("physical_backward_calls"),
            "locked_truth_opened": payload.get("locked_truth_opened"),
        }
    result["duration_seconds"] = {"wall": time.perf_counter() - wall, "cpu": time.process_time() - cpu}
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "TRAIN_ARRAY_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "TRAIN_STATE_SUMMARY.json").write_text(json.dumps(teacher_stats, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status", "observation_rows", "unique_clean_states", "train_only_rms_unique_states", "duration_seconds",
        "locked_files_opened", "new_labels", "full_forward_calls", "linear_solves", "training_updates")}, indent=2))


if __name__ == "__main__":
    main()
