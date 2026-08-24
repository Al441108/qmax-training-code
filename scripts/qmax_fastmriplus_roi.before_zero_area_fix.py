#!/usr/bin/env python3
from __future__ import annotations

"""Validation-only fastMRI+ lesion ROI analysis for QMax-Full vs zero-filled.

The script has two deliberately separated modes:

* ``preflight`` maps fastMRI+ annotations to the locked validation cohort and
  renders reference-image overlays.  It never loads the QMax checkpoint.
* ``evaluate`` requires an explicit approval JSON produced after visual review
  of those overlays, then performs QMax-Full and zero-filled inference only.

No model is trained, selected, or modified by this script.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_qmax_counterfactuals import ManifestDataset  # noqa: E402
from scripts.qmax_common import IndexedDataset, make_dataset, sha256_file  # noqa: E402
from scripts.render_six_slice_qualitative import (  # noqa: E402
    load_qmax_full,
    predict_aux_model,
    predict_reference_and_zf,
)


PROTOCOL = "QMax-fastMRIplus-validation-ROI-v1"
FILE_RE = re.compile(r"(?i)(file\d+)")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Refusing to write an empty CSV: {path}")
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_token(value: Any) -> str | None:
    match = FILE_RE.search(str(value))
    return match.group(1).lower() if match else None


def load_annotations(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"file", "slice", "x", "y", "width", "height", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"fastMRI+ CSV lacks columns: {sorted(missing)}")
        for raw in reader:
            token = file_token(raw["file"])
            if token is None:
                continue

            geometry_fields = ("x", "y", "width", "height")
            geometry_text = {
                field: str(raw.get(field, "") or "").strip()
                for field in geometry_fields
            }

            # fastMRI+ includes study-level annotations without bounding-box
            # geometry. They are valid catalogue entries but cannot support
            # pixel-level ROI analysis, so exclude them explicitly.
            if not any(geometry_text.values()):
                continue
            if not all(geometry_text.values()):
                raise RuntimeError(
                    "Partially specified fastMRI+ bounding box: "
                    f"file={raw.get('file')} slice={raw.get('slice')} "
                    f"geometry={geometry_text}"
                )

            slice_text = str(raw.get("slice", "") or "").strip()
            if not slice_text:
                raise RuntimeError(
                    "Box-level fastMRI+ annotation lacks slice index: "
                    f"file={raw.get('file')}"
                )

            geometry = {
                field: float(value)
                for field, value in geometry_text.items()
            }
            if (
                not all(math.isfinite(value) for value in geometry.values())
                or geometry["width"] <= 0
                or geometry["height"] <= 0
            ):
                raise RuntimeError(
                    "Invalid fastMRI+ bounding-box geometry: "
                    f"file={raw.get('file')} slice={slice_text} "
                    f"geometry={geometry}"
                )

            row = dict(raw)
            row.update(
                {
                    "file_token": token,
                    "slice": int(float(slice_text)),
                    **geometry,
                    "label": str(raw["label"]).strip(),
                }
            )
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No usable fastMRI+ annotations in {path}")
    return rows


def locked_pairs(manifest: Mapping[str, Any]) -> set[Tuple[str, int]]:
    return {
        (str(patient["patient_id"]), int(index))
        for patient in manifest["patients"]
        for index in patient["slice_indices"]
    }


def metadata_file_map(metadata_csv: Path) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Map original fastMRI file tokens to patient IDs without guessing IDs."""
    candidates: Dict[str, set[str]] = defaultdict(set)
    column_hits: Dict[str, int] = defaultdict(int)
    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "patient_id" not in (reader.fieldnames or []):
            raise RuntimeError("metadata CSV lacks patient_id")
        for row in reader:
            patient = str(row["patient_id"])
            for column, value in row.items():
                token = file_token(value)
                if token is not None:
                    candidates[token].add(patient)
                    column_hits[column] += 1
    ambiguous = {key: sorted(value) for key, value in candidates.items() if len(value) != 1}
    mapping = {
        key: next(iter(value))
        for key, value in candidates.items()
        if len(value) == 1
    }
    audit = {
        "num_unique_file_tokens": len(mapping),
        "num_ambiguous_file_tokens": len(ambiguous),
        "ambiguous_examples": dict(list(sorted(ambiguous.items()))[:10]),
        "metadata_columns_with_file_tokens": dict(sorted(column_hits.items())),
    }
    return mapping, audit


def build_locked_dataset(metadata_csv: Path, manifest_path: Path):
    manifest = read_json(manifest_path)
    # The dataset API accepts train/val/test.  "Validation" is the cohort's
    # human-facing name; its executable split token is "val".
    source = IndexedDataset(make_dataset(str(metadata_csv), split="val"))
    dataset = ManifestDataset(source, manifest)
    lookup = {
        (str(record["patient_id"]), int(record["slice_idx"])): index
        for index, record in enumerate(dataset.records)
    }
    return manifest, dataset, lookup


def bbox_to_target(
    row: Mapping[str, Any],
    height: int,
    width: int,
    annotation_height: int,
    annotation_width: int,
    flip_up_down: bool,
) -> Tuple[int, int, int, int]:
    if (height, width) != (annotation_height, annotation_width):
        raise RuntimeError(
            "Target and annotation coordinate sizes differ: "
            f"target={(height, width)} annotation={(annotation_height, annotation_width)}. "
            "Do not silently rescale clinical boxes."
        )
    x0 = float(row["x"])
    x1 = x0 + float(row["width"])
    y0 = float(row["y"])
    y1 = y0 + float(row["height"])
    if flip_up_down:
        y0, y1 = float(height) - y1, float(height) - y0
    left = max(0, min(width, int(math.floor(x0))))
    right = max(0, min(width, int(math.ceil(x1))))
    top = max(0, min(height, int(math.floor(y0))))
    bottom = max(0, min(height, int(math.ceil(y1))))
    if right <= left or bottom <= top:
        raise RuntimeError(f"Empty transformed box for annotation {dict(row)}")
    return left, top, right, bottom


def matched_annotations(
    annotations: Sequence[Mapping[str, Any]],
    file_map: Mapping[str, str],
    wanted: set[Tuple[str, int]],
    slice_offset: int,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for source in annotations:
        patient = file_map.get(str(source["file_token"]))
        if patient is None:
            continue
        target_slice = int(source["slice"]) + int(slice_offset)
        if (patient, target_slice) not in wanted:
            continue
        row = dict(source)
        row["patient_id"] = patient
        row["target_slice_idx"] = target_slice
        output.append(row)
    return output


def select_overlay_cases(rows: Sequence[Mapping[str, Any]], limit: int) -> List[Tuple[str, int]]:
    cases = sorted({(str(row["patient_id"]), int(row["target_slice_idx"])) for row in rows})
    if not cases:
        return []
    count = min(int(limit), len(cases))
    positions = np.linspace(0, len(cases) - 1, count).round().astype(int)
    return [cases[int(position)] for position in positions]


def target_for_case(dataset: ManifestDataset, lookup: Mapping[Tuple[str, int], int], case):
    item = dataset[lookup[case]]
    target = item["pdfs_target_raw"]
    if torch.is_tensor(target):
        target = target.detach().float().cpu().numpy()
    target = np.asarray(target).squeeze()
    if target.ndim != 2:
        raise RuntimeError(f"Unexpected target shape {target.shape} for {case}")
    return target


def render_overlays(
    path: Path,
    cases: Sequence[Tuple[str, int]],
    rows_by_case: Mapping[Tuple[str, int], Sequence[Mapping[str, Any]]],
    dataset: ManifestDataset,
    lookup: Mapping[Tuple[str, int], int],
    annotation_height: int,
    annotation_width: int,
    flip_up_down: bool,
) -> None:
    columns = 3
    rows_n = int(math.ceil(len(cases) / columns))
    fig, axes = plt.subplots(rows_n, columns, figsize=(12, 4 * rows_n), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    colours = plt.get_cmap("tab10")
    for axis, case in zip(axes.flat, cases):
        target = target_for_case(dataset, lookup, case)
        positive = target[target > 0]
        low = float(np.percentile(positive, 1)) if positive.size else float(target.min())
        high = float(np.percentile(target, 99.5))
        axis.imshow(target, cmap="gray", vmin=low, vmax=max(high, low + 1e-8))
        labels = []
        for box_index, annotation in enumerate(rows_by_case[case]):
            left, top, right, bottom = bbox_to_target(
                annotation,
                target.shape[0],
                target.shape[1],
                annotation_height,
                annotation_width,
                flip_up_down,
            )
            colour = colours(box_index % 10)
            axis.add_patch(
                Rectangle(
                    (left, top), right - left, bottom - top,
                    fill=False, linewidth=1.5, edgecolor=colour,
                )
            )
            labels.append(str(annotation["label"]))
        axis.set_title(
            f"{case[0][:10]} · slice {case[1]}\n" + "; ".join(sorted(set(labels))),
            fontsize=9,
        )
        axis.axis("off")
    fig.suptitle("fastMRI+ boxes mapped back to locked validation references", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def mapping_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "file": row["file"],
            "file_token": row["file_token"],
            "patient_id": row["patient_id"],
            "csv_slice": row["slice"],
            "target_slice_idx": row["target_slice_idx"],
            "x": row["x"],
            "y": row["y"],
            "width": row["width"],
            "height": row["height"],
            "label": row["label"],
        }
        for row in rows
    ]


def run_preflight(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    if "heldout" in str(args.full_clean_manifest).lower() or "test" in Path(args.full_clean_manifest).name.lower():
        raise RuntimeError("Refusing a held-out/test manifest; this analysis is validation-only")
    manifest, dataset, lookup = build_locked_dataset(
        Path(args.metadata_csv), Path(args.full_clean_manifest)
    )
    wanted = locked_pairs(manifest)
    annotations = load_annotations(Path(args.annotations_csv))
    file_map, map_audit = metadata_file_map(Path(args.metadata_csv))
    rows = matched_annotations(annotations, file_map, wanted, args.slice_offset)
    if not rows:
        raise RuntimeError(
            "No fastMRI+ lesion boxes mapped into the locked validation cohort. "
            f"Mapping audit={map_audit}"
        )
    by_case: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[(str(row["patient_id"]), int(row["target_slice_idx"]))].append(row)
    cases = select_overlay_cases(rows, args.num_overlays)
    render_overlays(
        output / "fastmriplus_coordinate_overlay.png",
        cases,
        by_case,
        dataset,
        lookup,
        args.annotation_height,
        args.annotation_width,
        args.flip_up_down,
    )
    mapped_csv = output / "mapped_locked_validation_boxes.csv"
    write_csv(mapped_csv, mapping_rows(rows))
    patients = {str(row["patient_id"]) for row in rows}
    audit = {
        "protocol_version": PROTOCOL,
        "status": "requires_manual_coordinate_approval",
        "scope": "locked validation only; held-out test not accessed",
        "dataset_split": "val",
        "slice_offset": int(args.slice_offset),
        "flip_up_down": bool(args.flip_up_down),
        "annotation_shape": [int(args.annotation_height), int(args.annotation_width)],
        "num_source_annotations": len(annotations),
        "num_mapped_locked_boxes": len(rows),
        "num_mapped_locked_slices": len(by_case),
        "num_mapped_locked_patients": len(patients),
        "labels": sorted({str(row["label"]) for row in rows}),
        "mapping_audit": map_audit,
        "hashes": {
            "annotations_csv": sha256_file(Path(args.annotations_csv)),
            "metadata_csv": sha256_file(Path(args.metadata_csv)),
            "full_clean_manifest": sha256_file(Path(args.full_clean_manifest)),
            "mapped_boxes_csv": sha256_file(mapped_csv),
        },
        "outputs": {
            "overlay": str(output / "fastmriplus_coordinate_overlay.png"),
            "mapped_boxes": str(mapped_csv),
        },
        "approval_instruction": (
            "Inspect every displayed box. Only then run approve mode with "
            "--confirm BOXES_VISUALLY_ALIGNED. If boxes are vertically mirrored, "
            "rerun preflight with the opposite --flip_up_down value; if slices are "
            "off by one, rerun with --slice_offset 0 instead of -1."
        ),
    }
    write_json(output / "preflight_audit.json", audit)
    print(json.dumps(audit, indent=2))


def run_approve(args: argparse.Namespace) -> None:
    if args.confirm != "BOXES_VISUALLY_ALIGNED":
        raise RuntimeError("Exact confirmation token required: BOXES_VISUALLY_ALIGNED")
    preflight_path = Path(args.preflight_audit)
    audit = read_json(preflight_path)
    if audit.get("status") != "requires_manual_coordinate_approval":
        raise RuntimeError("Unexpected preflight status")
    approval = {
        "protocol_version": PROTOCOL,
        "status": "approved",
        "confirmation": args.confirm,
        "preflight_audit": str(preflight_path),
        "preflight_audit_sha256": sha256_file(preflight_path),
        "slice_offset": int(audit["slice_offset"]),
        "flip_up_down": bool(audit["flip_up_down"]),
        "annotation_shape": audit["annotation_shape"],
        "input_hashes": audit["hashes"],
    }
    write_json(Path(args.output), approval)
    print(json.dumps(approval, indent=2))


def masks_from_boxes(
    shape: Tuple[int, int],
    boxes: Sequence[Tuple[int, int, int, int]],
    target: np.ndarray,
    ring_pixels: int,
) -> Dict[str, np.ndarray]:
    height, width = shape
    lesion = np.zeros(shape, dtype=bool)
    expanded = np.zeros(shape, dtype=bool)
    for left, top, right, bottom in boxes:
        lesion[top:bottom, left:right] = True
        expanded[
            max(0, top - ring_pixels):min(height, bottom + ring_pixels),
            max(0, left - ring_pixels):min(width, right + ring_pixels),
        ] = True
    ring = expanded & ~lesion
    threshold = 0.05 * max(float(np.max(target)), 1e-8)
    foreground = np.asarray(target) > threshold
    nonlesion = foreground & ~expanded
    return {"lesion": lesion, "perilesional_ring": ring, "nonlesion_foreground": nonlesion}


def roi_l1(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return float("nan")
    scale = max(float(np.max(target)), 1e-8)
    return float(np.mean(np.abs(prediction[mask] / scale - target[mask] / scale)))


def bootstrap_delta(values: np.ndarray, seed: int, iterations: int) -> Dict[str, float]:
    if values.size < 2:
        return {"mean": float(np.mean(values)), "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(iterations, values.size))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def verify_approval(args: argparse.Namespace) -> Mapping[str, Any]:
    approval = read_json(Path(args.approval_json))
    if approval.get("status") != "approved" or approval.get("confirmation") != "BOXES_VISUALLY_ALIGNED":
        raise RuntimeError("Coordinate approval is missing or invalid")
    current = {
        "annotations_csv": sha256_file(Path(args.annotations_csv)),
        "metadata_csv": sha256_file(Path(args.metadata_csv)),
        "full_clean_manifest": sha256_file(Path(args.full_clean_manifest)),
        "mapped_boxes_csv": sha256_file(Path(args.mapped_boxes_csv)),
    }
    if current != approval.get("input_hashes"):
        raise RuntimeError(f"Approved input hashes changed: current={current} approved={approval.get('input_hashes')}")
    return approval


def run_evaluate(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    if any(token in str(args.full_clean_manifest).lower() for token in ("heldout", "held_out")):
        raise RuntimeError("Refusing held-out inputs in post-hoc ROI analysis")
    approval = verify_approval(args)
    _manifest, dataset, lookup = build_locked_dataset(
        Path(args.metadata_csv), Path(args.full_clean_manifest)
    )
    mapped = []
    with Path(args.mapped_boxes_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        mapped = list(csv.DictReader(handle))
    by_case: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for raw in mapped:
        row = dict(raw)
        for key in ("csv_slice", "target_slice_idx"):
            row[key] = int(float(row[key]))
        for key in ("x", "y", "width", "height"):
            row[key] = float(row[key])
        by_case[(str(row["patient_id"]), int(row["target_slice_idx"]))].append(row)

    device = torch.device("cuda")
    model = load_qmax_full(Path(args.qmax_checkpoint), device)
    slice_rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for case in sorted(by_case):
        if case not in lookup:
            raise RuntimeError(f"Approved mapped case disappeared from locked dataset: {case}")
        batch = next(iter(DataLoader(Subset(dataset, [lookup[case]]), batch_size=1, num_workers=0)))
        reference, zero_filled = predict_reference_and_zf(batch, device)
        qmax = predict_aux_model(model, batch, device, q_key="q_hat")
        target = np.asarray(reference["prediction"], dtype=np.float32)
        zf_prediction = np.asarray(zero_filled["prediction"], dtype=np.float32)
        qmax_prediction = np.asarray(qmax["prediction"], dtype=np.float32)
        boxes = [
            bbox_to_target(
                row,
                target.shape[0], target.shape[1],
                int(approval["annotation_shape"][0]),
                int(approval["annotation_shape"][1]),
                bool(approval["flip_up_down"]),
            )
            for row in by_case[case]
        ]
        masks = masks_from_boxes(target.shape, boxes, target, args.ring_pixels)
        if int(masks["lesion"].sum()) < args.min_lesion_pixels:
            skipped.append({"patient_id": case[0], "slice_idx": case[1], "reason": "lesion_mask_too_small"})
            continue
        labels = ";".join(sorted({str(row["label"]) for row in by_case[case]}))
        base = {
            "patient_id": case[0],
            "slice_idx": case[1],
            "num_boxes": len(boxes),
            "labels": labels,
            "q_mean": float(qmax["q"]),
        }
        for region, mask in masks.items():
            pixels = int(mask.sum())
            if pixels == 0:
                continue
            zf_l1 = roi_l1(zf_prediction, target, mask)
            qmax_l1 = roi_l1(qmax_prediction, target, mask)
            slice_rows.append(
                {
                    **base,
                    "region": region,
                    "num_pixels": pixels,
                    "zero_filled_l1": zf_l1,
                    "qmax_full_l1": qmax_l1,
                    "delta_qmax_minus_zero_filled": qmax_l1 - zf_l1,
                    "relative_change_percent": 100.0 * (qmax_l1 - zf_l1) / max(zf_l1, 1e-12),
                    "qmax_better": int(qmax_l1 < zf_l1),
                }
            )

    if not slice_rows:
        raise RuntimeError("No ROI rows were evaluated")
    write_csv(output / "slice_level_roi.csv", slice_rows)

    accum: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in slice_rows:
        accum[(str(row["patient_id"]), str(row["region"]))].append(row)
    patient_rows: List[Dict[str, Any]] = []
    for (patient, region), rows in sorted(accum.items()):
        zf_l1 = float(np.mean([float(row["zero_filled_l1"]) for row in rows]))
        qmax_l1 = float(np.mean([float(row["qmax_full_l1"]) for row in rows]))
        patient_rows.append(
            {
                "patient_id": patient,
                "region": region,
                "num_slices": len(rows),
                "zero_filled_l1": zf_l1,
                "qmax_full_l1": qmax_l1,
                "delta_qmax_minus_zero_filled": qmax_l1 - zf_l1,
                "relative_change_percent": 100.0 * (qmax_l1 - zf_l1) / max(zf_l1, 1e-12),
                "qmax_better": int(qmax_l1 < zf_l1),
            }
        )
    write_csv(output / "patient_level_roi.csv", patient_rows)

    summary: Dict[str, Any] = {}
    for region in ("lesion", "perilesional_ring", "nonlesion_foreground"):
        rows = [row for row in patient_rows if row["region"] == region]
        if not rows:
            continue
        deltas = np.asarray([float(row["delta_qmax_minus_zero_filled"]) for row in rows])
        ci = bootstrap_delta(deltas, args.seed, args.bootstrap_iterations)
        summary[region] = {
            "num_patients": len(rows),
            "zero_filled_mean_l1": float(np.mean([float(row["zero_filled_l1"]) for row in rows])),
            "qmax_full_mean_l1": float(np.mean([float(row["qmax_full_l1"]) for row in rows])),
            "delta_qmax_minus_zero_filled": ci["mean"],
            "paired_bootstrap_ci95": [ci["ci95_low"], ci["ci95_high"]],
            "patients_qmax_better": int(sum(int(row["qmax_better"]) for row in rows)),
            "patients_zero_filled_better_or_equal": int(sum(not int(row["qmax_better"]) for row in rows)),
        }
    write_json(output / "roi_summary.json", summary)
    audit = {
        "protocol_version": PROTOCOL,
        "status": "passed",
        "scope": "post-hoc exploratory; locked validation only; held-out test not accessed",
        "comparison": ["zero_filled", "qmax_full_stagea_epoch60"],
        "checkpoint": str(args.qmax_checkpoint),
        "checkpoint_sha256": sha256_file(Path(args.qmax_checkpoint)),
        "approval_json": str(args.approval_json),
        "approval_json_sha256": sha256_file(Path(args.approval_json)),
        "num_evaluated_slices": len({(row["patient_id"], row["slice_idx"]) for row in slice_rows}),
        "num_evaluated_patients": len({row["patient_id"] for row in patient_rows}),
        "num_skipped_slices": len(skipped),
        "skipped": skipped,
        "region_definitions": {
            "lesion": "union of all fastMRI+ boxes on the slice",
            "perilesional_ring": f"{args.ring_pixels}-pixel expanded boxes minus lesion union",
            "nonlesion_foreground": "target > 5% target maximum, excluding expanded lesion boxes",
        },
        "statistical_unit": "patient; slice ROIs averaged within patient before paired bootstrap",
        "bootstrap_iterations": int(args.bootstrap_iterations),
        "outputs": {
            "slice_level": str(output / "slice_level_roi.csv"),
            "patient_level": str(output / "patient_level_roi.csv"),
            "summary": str(output / "roi_summary.json"),
        },
    }
    write_json(output / "roi_audit.json", audit)
    print(json.dumps({"audit": audit, "summary": summary}, indent=2))


def add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--full_clean_manifest", required=True)
    parser.add_argument("--annotations_csv", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    preflight = sub.add_parser("preflight")
    add_shared(preflight)
    preflight.add_argument("--output_dir", required=True)
    preflight.add_argument("--slice_offset", type=int, default=-1)
    preflight.add_argument("--flip_up_down", action=argparse.BooleanOptionalAction, default=True)
    preflight.add_argument("--annotation_height", type=int, default=320)
    preflight.add_argument("--annotation_width", type=int, default=320)
    preflight.add_argument("--num_overlays", type=int, default=12)

    approve = sub.add_parser("approve")
    approve.add_argument("--preflight_audit", required=True)
    approve.add_argument("--confirm", required=True)
    approve.add_argument("--output", required=True)

    evaluate = sub.add_parser("evaluate")
    add_shared(evaluate)
    evaluate.add_argument("--mapped_boxes_csv", required=True)
    evaluate.add_argument("--approval_json", required=True)
    evaluate.add_argument("--qmax_checkpoint", required=True)
    evaluate.add_argument("--output_dir", required=True)
    evaluate.add_argument("--ring_pixels", type=int, default=8)
    evaluate.add_argument("--min_lesion_pixels", type=int, default=16)
    evaluate.add_argument("--bootstrap_iterations", type=int, default=10000)
    evaluate.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "preflight":
        run_preflight(args)
    elif args.mode == "approve":
        run_approve(args)
    elif args.mode == "evaluate":
        run_evaluate(args)
    else:
        raise AssertionError(args.mode)


if __name__ == "__main__":
    main()
