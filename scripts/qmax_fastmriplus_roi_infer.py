#!/usr/bin/env python3
from __future__ import annotations

"""Cache frozen QMax-Full inference and ROI-weighted reliability diagnostics.

This is stage 2A of the fastMRI+ post-hoc validation analysis.  It performs
model inference exactly once per annotated locked-validation slice and writes
restartable per-slice caches.  It deliberately performs no bootstrap analysis
and creates no manuscript figures; those are stage 2B responsibilities.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import fastmri
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.qmax_fastmriplus_roi as roi_base  # noqa: E402
from scripts.qmax_common import prepare_batch, sha256_file  # noqa: E402
from scripts.render_six_slice_qualitative import load_qmax_full  # noqa: E402
from src.fft_utils import center_crop  # noqa: E402
from src.m2_prnf_varnet import _pad_to_multiple  # noqa: E402


PROTOCOL = "QMax-fastMRIplus-ROI-inference-cache-v2"
FORMAL_CHECKPOINT_SHA256 = (
    "1285dd76f7900859d7ca57e68fa4f54509bed540865a7a638397e20d5012b5aa"
)
SCALE_NAMES = ("H/2", "H/4", "H/8", "H/16")
REGIONS = (
    "lesion",
    "perilesional_ring",
    "matched_nonannotated_control",
    "nonlesion_foreground",
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def load_mapped_boxes(path: Path) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
    output: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: Dict[str, Any] = dict(raw)
            for key in ("csv_slice", "target_slice_idx"):
                row[key] = int(float(row[key]))
            for key in ("x", "y", "width", "height"):
                row[key] = float(row[key])
            case = (str(row["patient_id"]), int(row["target_slice_idx"]))
            output[case].append(row)
    if not output:
        raise RuntimeError(f"No mapped boxes in {path}")
    return output


def gradient_energy(image: np.ndarray, mask: np.ndarray) -> float:
    gy, gx = np.gradient(np.asarray(image, dtype=np.float32))
    magnitude = np.sqrt(gx * gx + gy * gy)
    return float(magnitude[mask].mean()) if int(mask.sum()) else float("nan")


def select_matched_control(
    lesion: np.ndarray,
    excluded: np.ndarray,
    foreground: np.ndarray,
    target: np.ndarray,
    stride: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Translate the lesion-union shape to a deterministic matched location.

    The selected region has exactly the same binary shape and area as the
    lesion union.  It must not overlap the lesion/perilesional exclusion and
    at least 90% of its pixels must lie in the target-derived foreground.
    Mean intensity and gradient energy are matched without inspecting either
    reconstruction, preserving a metric-blind control selection.
    """
    coordinates = np.argwhere(lesion)
    empty = np.zeros_like(lesion, dtype=bool)
    if coordinates.size == 0:
        return empty, {"status": "unavailable", "reason": "empty_lesion"}
    top, left = coordinates.min(axis=0)
    bottom, right = coordinates.max(axis=0) + 1
    relative = coordinates - np.asarray([top, left])
    box_h, box_w = int(bottom - top), int(right - left)
    height, width = lesion.shape
    source_mean = float(target[lesion].mean())
    source_gradient = gradient_energy(target, lesion)
    target_std = max(float(target[foreground].std()) if foreground.any() else 0.0, 1e-8)
    gy, gx = np.gradient(np.asarray(target, dtype=np.float32))
    grad = np.sqrt(gx * gx + gy * gy)
    grad_std = max(float(grad[foreground].std()) if foreground.any() else 0.0, 1e-8)

    rows = list(range(0, max(height - box_h + 1, 1), max(1, int(stride))))
    cols = list(range(0, max(width - box_w + 1, 1), max(1, int(stride))))
    if rows and rows[-1] != height - box_h:
        rows.append(height - box_h)
    if cols and cols[-1] != width - box_w:
        cols.append(width - box_w)
    candidates: List[Tuple[float, int, int, float, float]] = []
    for candidate_top in rows:
        rr = relative[:, 0] + int(candidate_top)
        for candidate_left in cols:
            cc = relative[:, 1] + int(candidate_left)
            if bool(excluded[rr, cc].any()):
                continue
            if float(foreground[rr, cc].mean()) < 0.90:
                continue
            mean_value = float(target[rr, cc].mean())
            gradient_value = float(grad[rr, cc].mean())
            score = (
                abs(mean_value - source_mean) / target_std
                + abs(gradient_value - source_gradient) / grad_std
            )
            candidates.append(
                (score, int(candidate_top), int(candidate_left), mean_value, gradient_value)
            )
    if not candidates:
        return empty, {
            "status": "unavailable",
            "reason": "no_nonoverlapping_foreground_translation",
        }
    score, candidate_top, candidate_left, mean_value, gradient_value = min(candidates)
    rr = relative[:, 0] + candidate_top
    cc = relative[:, 1] + candidate_left
    control = np.zeros_like(lesion, dtype=bool)
    control[rr, cc] = True
    return control, {
        "status": "available",
        "selection": "target-only deterministic translated lesion shape",
        "stride": int(stride),
        "top": candidate_top,
        "left": candidate_left,
        "height": box_h,
        "width": box_w,
        "num_pixels": int(control.sum()),
        "foreground_fraction": float(foreground[control].mean()),
        "target_mean": mean_value,
        "target_gradient_energy": gradient_value,
        "lesion_target_mean": source_mean,
        "lesion_target_gradient_energy": source_gradient,
        "matching_score": float(score),
    }


def embed_crop_mask(mask: np.ndarray, full_hw: Tuple[int, int], device: torch.device) -> torch.Tensor:
    full_h, full_w = (int(full_hw[0]), int(full_hw[1]))
    crop_h, crop_w = mask.shape
    if crop_h > full_h or crop_w > full_w:
        raise RuntimeError(f"ROI {mask.shape} exceeds model image {full_hw}")
    top = (full_h - crop_h) // 2
    left = (full_w - crop_w) // 2
    tensor = torch.zeros(1, 1, full_h, full_w, device=device, dtype=torch.float32)
    tensor[:, :, top : top + crop_h, left : left + crop_w] = torch.from_numpy(
        mask.astype(np.float32)
    ).to(device)
    padded, _pads = _pad_to_multiple(tensor, 16)
    return padded


def weighted_mean_map(value: torch.Tensor, weight: torch.Tensor) -> float:
    denominator = float(weight.sum().item())
    if denominator <= 1e-8:
        return float("nan")
    return float((value.float() * weight.float()).sum().item() / denominator)


class SpatialReliabilityCapture:
    """Read-only hooks that reproduce selector outputs for ROI diagnostics."""

    def __init__(
        self,
        model: torch.nn.Module,
        padded_masks: Mapping[str, torch.Tensor],
    ):
        self.model = model
        self.padded_masks = dict(padded_masks)
        self.records: List[Dict[str, Any]] = []
        self.handles: List[Any] = []
        self.call_counts = [0 for _ in model.controllers]
        self.max_direct_rms_reproduction_error = 0.0

    def __enter__(self):
        for scale_index, controller in enumerate(self.model.controllers):
            handle = controller.register_forward_hook(
                self._make_hook(scale_index), with_kwargs=True
            )
            self.handles.append(handle)
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False

    def _make_hook(self, scale_index: int):
        def hook(module, _args, kwargs, output):
            cascade_index = self.call_counts[scale_index]
            self.call_counts[scale_index] += 1
            target = kwargs["target"]
            auxiliary_u0 = kwargs["auxiliary_u0"]
            dc_evidence = kwargs["dc_evidence"]
            availability = kwargs["availability"].reshape(target.shape[0])
            alpha = kwargs["alpha"]
            detail_neutral = bool(kwargs.get("detail_neutral", False))
            alignment_off = bool(kwargs.get("alignment_off", False))

            # Recompute only child heads in eval mode.  No controller output is
            # changed and no gradients are enabled.
            stable = module.blur(auxiliary_u0)
            detail = auxiliary_u0 - stable
            alignment_evidence = torch.cat(
                [target, stable, detail, torch.abs(target - stable), target * stable], dim=1
            )
            dc_scaled = F.interpolate(
                dc_evidence.detach(), size=target.shape[-2:], mode="area"
            )
            detail_evidence = alignment_evidence
            if module.uses_dc_evidence:
                detail_evidence = torch.cat([detail_evidence, dc_scaled], dim=1)
            detail_gate = 2.0 * torch.sigmoid(module.detail_head(detail_evidence))
            if detail_neutral:
                detail_gate = torch.ones_like(detail_gate)
            alignment = module.alignment_head(alignment_evidence)
            if alignment_off:
                alignment = torch.zeros_like(alignment)
            selected = stable + detail_gate * detail + alignment

            fused, diagnostics = output
            q_hat = diagnostics["q_hat"].reshape(target.shape[0])
            m = availability.to(target.dtype).view(target.shape[0], 1, 1, 1)
            q = q_hat[:, None, None, None]
            pre_q_direct = alpha * selected
            direct = m * q * pre_q_direct
            correction = fused - target - direct
            total = fused - target

            reproduced = (
                direct.detach().float().square().mean((1, 2, 3)).sqrt()
                / target.detach().float().square().mean((1, 2, 3)).sqrt().clamp_min(1e-8)
            )
            expected = diagnostics["direct_to_target_rms"].detach().float()
            rms_error = float((reproduced - expected).abs().max().item())
            self.max_direct_rms_reproduction_error = max(
                self.max_direct_rms_reproduction_error, rms_error
            )

            maps = {
                "pre_q_direct_energy": pre_q_direct.detach().float().square().mean(1).sqrt(),
                "gated_direct_energy": direct.detach().float().square().mean(1).sqrt(),
                "correction_energy": correction.detach().float().square().mean(1).sqrt(),
                "final_auxiliary_energy": total.detach().float().square().mean(1).sqrt(),
            }
            record: Dict[str, Any] = {
                "cascade": int(cascade_index),
                "scale": int(scale_index),
                "scale_name": SCALE_NAMES[scale_index],
                "q": float(q_hat.detach().float().mean().item()),
                "alpha": float(alpha.detach().float().item()),
            }
            for region, padded_mask in self.padded_masks.items():
                weight = F.interpolate(
                    padded_mask,
                    size=target.shape[-2:],
                    mode="area",
                )[:, 0]
                for name, energy_map in maps.items():
                    record[f"{region}_{name}"] = weighted_mean_map(energy_map, weight)
            self.records.append(record)

        return hook


def effective_reliability(
    records: Sequence[Mapping[str, Any]], region: str, available: bool = True
) -> Dict[str, Any]:
    if not available:
        return {"status": "unavailable", "reason": "auxiliary_missing", "q_eff": None}
    q_values: List[float] = []
    weights: List[float] = []
    for row in records:
        q = float(row["q"])
        weight = float(row[f"{region}_pre_q_direct_energy"])
        if math.isfinite(q) and math.isfinite(weight) and weight >= 0:
            q_values.append(q)
            weights.append(weight)
    denominator = float(np.sum(weights)) if weights else 0.0
    if denominator <= 1e-8:
        return {
            "status": "not_estimable",
            "reason": "near_zero_pre_q_auxiliary_energy",
            "q_eff": None,
            "weight_sum": denominator,
        }
    q_eff = float(np.dot(q_values, weights) / denominator)
    return {
        "status": "available",
        "q_eff": q_eff,
        "weight_sum": denominator,
        "q_min": float(min(q_values)),
        "q_max": float(max(q_values)),
        "num_paths": len(q_values),
    }


def qeff_unit_checks() -> Dict[str, bool]:
    synthetic = [
        {"q": 0.2, "lesion_pre_q_direct_energy": 1.0},
        {"q": 0.5, "lesion_pre_q_direct_energy": 1.0},
        {"q": 0.8, "lesion_pre_q_direct_energy": 1.0},
    ]
    uniform = effective_reliability(synthetic, "lesion")
    zero = [dict(row, lesion_pre_q_direct_energy=0.0) for row in synthetic]
    zero_result = effective_reliability(zero, "lesion")
    missing = effective_reliability(synthetic, "lesion", available=False)
    return {
        "uniform_weights_equal_arithmetic_mean": abs(float(uniform["q_eff"]) - 0.5) < 1e-12,
        "zero_denominator_is_not_estimable": zero_result["status"] == "not_estimable",
        "missing_is_unavailable": missing["status"] == "unavailable",
    }


def qeff_from_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    output = {region: effective_reliability(records, region) for region in REGIONS}
    for region, result in output.items():
        if result["status"] != "available":
            continue
        q_eff = float(result["q_eff"])
        if not (float(result["q_min"]) - 1e-7 <= q_eff <= float(result["q_max"]) + 1e-7):
            raise RuntimeError(f"q_eff outside active q range for {region}: {result}")
    return output


def prepare_case_masks(
    target: np.ndarray,
    boxes: Sequence[Tuple[int, int, int, int]],
    ring_pixels: int,
    control_stride: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    masks = roi_base.masks_from_boxes(target.shape, boxes, target, ring_pixels)
    threshold = 0.05 * max(float(np.max(target)), 1e-8)
    foreground = target > threshold
    excluded = masks["lesion"] | masks["perilesional_ring"]
    control, control_audit = select_matched_control(
        masks["lesion"], excluded, foreground, target, control_stride
    )
    masks["matched_nonannotated_control"] = control
    return masks, control_audit


def run_prediction(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    device: torch.device,
    masks: Mapping[str, np.ndarray],
    capture_spatial: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], List[Dict[str, Any]], float]:
    kspace, mask, pd, target, _indices = prepare_batch(batch, device)
    available = torch.ones(kspace.shape[0], device=device)
    padded_masks = {
        name: embed_crop_mask(value, (kspace.shape[-3], kspace.shape[-2]), device)
        for name, value in masks.items()
    }
    capture = SpatialReliabilityCapture(model, padded_masks)
    with torch.inference_mode():
        zero_filled = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace)), dim=1)
        if capture_spatial:
            with capture:
                prediction, auxiliary = model(
                    kspace, mask, pd, available, return_aux=True
                )
        else:
            prediction, auxiliary = model(kspace, mask, pd, available, return_aux=True)
    prediction = center_crop(prediction, target.shape[-2], target.shape[-1])
    zero_filled = center_crop(zero_filled, target.shape[-2], target.shape[-1])
    target_np = target[0].detach().float().cpu().numpy()
    zf_np = zero_filled[0].detach().float().cpu().numpy()
    prediction_np = prediction[0].detach().float().cpu().numpy()
    auxiliary_summary = {
        "q_mean": float(auxiliary["q_hat"].detach().float().mean().item()),
        "q_min": float(auxiliary["q_hat"].detach().float().min().item()),
        "q_max": float(auxiliary["q_hat"].detach().float().max().item()),
        "q_shape": list(auxiliary["q_hat"].shape),
    }
    return (
        target_np,
        zf_np,
        prediction_np,
        auxiliary_summary,
        capture.records,
        capture.max_direct_rms_reproduction_error,
    )


def case_stem(case: Tuple[str, int]) -> str:
    return f"{case[0]}_slice{case[1]:03d}"


def validate_existing_case(npz_path: Path, json_path: Path) -> Dict[str, Any] | None:
    if not npz_path.exists() or not json_path.exists():
        return None
    metadata = read_json(json_path)
    if metadata.get("protocol_version") != PROTOCOL or metadata.get("status") != "passed":
        raise RuntimeError(f"Existing case cache has incompatible metadata: {json_path}")
    if sha256_file(npz_path) != metadata.get("npz_sha256"):
        raise RuntimeError(f"Existing case cache hash mismatch: {npz_path}")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--full_clean_manifest", required=True)
    parser.add_argument("--annotations_csv", required=True)
    parser.add_argument("--mapped_boxes_csv", required=True)
    parser.add_argument("--approval_json", required=True)
    parser.add_argument("--qmax_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ring_pixels", type=int, default=8)
    parser.add_argument("--control_stride", type=int, default=4)
    parser.add_argument("--min_lesion_pixels", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    cache_dir = output / "cases"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if any(token in str(args.full_clean_manifest).lower() for token in ("heldout", "held_out", "test")):
        raise RuntimeError("Refusing held-out/test inputs; this analysis is locked-validation-only")
    approval = roi_base.verify_approval(args)
    checkpoint = Path(args.qmax_checkpoint)
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != FORMAL_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Formal checkpoint SHA-256 mismatch: {checkpoint_hash} != {FORMAL_CHECKPOINT_SHA256}"
        )
    _manifest, dataset, lookup = roi_base.build_locked_dataset(
        Path(args.metadata_csv), Path(args.full_clean_manifest)
    )
    by_case = load_mapped_boxes(Path(args.mapped_boxes_csv))
    for case in by_case:
        if case not in lookup:
            raise RuntimeError(f"Approved case absent from locked dataset: {case}")

    device = torch.device("cuda")
    model = load_qmax_full(checkpoint, device)
    unit_checks = qeff_unit_checks()
    if not all(unit_checks.values()):
        raise RuntimeError(f"q_eff unit checks failed: {unit_checks}")
    case_entries: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    output_equivalence_max_abs = 0.0
    direct_rms_error_max = 0.0
    equivalence_checked = False

    for case_index, case in enumerate(sorted(by_case)):
        stem = case_stem(case)
        npz_path = cache_dir / f"{stem}.npz"
        json_path = cache_dir / f"{stem}.json"
        existing = validate_existing_case(npz_path, json_path)
        if existing is not None:
            case_entries.append(existing)
            cached_direct_error = float(
                existing.get("direct_rms_reproduction_error_max", 0.0)
            )
            direct_rms_error_max = max(direct_rms_error_max, cached_direct_error)
            if bool(existing.get("diagnostic_output_equivalence_checked", False)):
                equivalence_checked = True
                output_equivalence_max_abs = max(
                    output_equivalence_max_abs,
                    float(existing["diagnostic_output_equivalence_max_abs"]),
                )
            continue
        batch = next(
            iter(DataLoader(Subset(dataset, [lookup[case]]), batch_size=1, num_workers=0))
        )
        target_tensor = batch["pdfs_target_raw"]
        if torch.is_tensor(target_tensor):
            target_preview = target_tensor.detach().float().cpu().numpy().squeeze()
        else:
            target_preview = np.asarray(target_tensor).squeeze()
        boxes = [
            roi_base.bbox_to_target(
                row,
                target_preview.shape[0],
                target_preview.shape[1],
                int(approval["annotation_shape"][0]),
                int(approval["annotation_shape"][1]),
                bool(approval["flip_up_down"]),
            )
            for row in by_case[case]
        ]
        masks, control_audit = prepare_case_masks(
            target_preview, boxes, args.ring_pixels, args.control_stride
        )
        if int(masks["lesion"].sum()) < int(args.min_lesion_pixels):
            skipped.append(
                {"patient_id": case[0], "slice_idx": case[1], "reason": "lesion_mask_too_small"}
            )
            continue

        baseline_prediction = None
        if not equivalence_checked:
            _, _, baseline_prediction, _, _, _ = run_prediction(
                model, batch, device, masks, capture_spatial=False
            )
        target, zero_filled, qmax, auxiliary, records, direct_error = run_prediction(
            model, batch, device, masks, capture_spatial=True
        )
        if len(records) != 48:
            raise RuntimeError(f"Expected 12x4 path records, observed {len(records)} for {case}")
        if baseline_prediction is not None:
            output_equivalence_max_abs = float(
                np.max(np.abs(baseline_prediction.astype(np.float64) - qmax.astype(np.float64)))
            )
            if output_equivalence_max_abs >= 1e-6:
                raise RuntimeError(
                    "Read-only diagnostic hooks changed model output: "
                    f"max_abs={output_equivalence_max_abs}"
                )
            equivalence_checked = True
        direct_rms_error_max = max(direct_rms_error_max, float(direct_error))
        if direct_rms_error_max >= 1e-5:
            raise RuntimeError(
                f"Reproduced direct-path RMS disagrees with model diagnostics: {direct_rms_error_max}"
            )

        qeff = qeff_from_records(records)
        labels = sorted({str(row["label"]) for row in by_case[case]})
        arrays = {
            "target": target.astype(np.float32),
            "zero_filled": zero_filled.astype(np.float32),
            "qmax_full": qmax.astype(np.float32),
            "lesion_mask": masks["lesion"].astype(np.uint8),
            "perilesional_ring_mask": masks["perilesional_ring"].astype(np.uint8),
            "matched_nonannotated_control_mask": masks[
                "matched_nonannotated_control"
            ].astype(np.uint8),
            "nonlesion_foreground_mask": masks["nonlesion_foreground"].astype(np.uint8),
            "boxes": np.asarray(boxes, dtype=np.int32),
        }
        atomic_npz(npz_path, **arrays)
        metadata = {
            "protocol_version": PROTOCOL,
            "status": "passed",
            "patient_id": case[0],
            "slice_idx": case[1],
            "labels": labels,
            "box_labels": [str(row["label"]) for row in by_case[case]],
            "num_boxes": len(boxes),
            "boxes": [list(value) for value in boxes],
            "q_summary": auxiliary,
            "q_eff": qeff,
            "path_records": records,
            "control_audit": control_audit,
            "diagnostic_output_equivalence_checked": baseline_prediction is not None,
            "diagnostic_output_equivalence_max_abs": (
                output_equivalence_max_abs if baseline_prediction is not None else None
            ),
            "direct_rms_reproduction_error_max": float(direct_error),
            "npz": str(npz_path),
            "npz_sha256": sha256_file(npz_path),
        }
        atomic_json(json_path, metadata)
        case_entries.append(metadata)
        print(
            f"[{case_index + 1}/{len(by_case)}] cached {case[0][:10]} slice {case[1]}",
            flush=True,
        )

    if not case_entries:
        raise RuntimeError("No annotated validation cases were cached")
    if not equivalence_checked:
        raise RuntimeError(
            "No cache records a hooked-versus-unhooked output-equivalence check. "
            "Remove one case cache and resume stage 2A."
        )
    qeff_unavailable = sum(
        entry["q_eff"]["lesion"]["status"] != "available" for entry in case_entries
    )
    qeff_unavailable_fraction = qeff_unavailable / len(case_entries)
    qeff_aggregate_allowed = qeff_unavailable_fraction <= 0.10
    manifest = {
        "protocol_version": PROTOCOL,
        "status": "passed",
        "scope": "post-hoc exploratory; locked validation only; held-out test not accessed",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "approval_json": str(args.approval_json),
        "approval_json_sha256": sha256_file(Path(args.approval_json)),
        "mapped_boxes_csv": str(args.mapped_boxes_csv),
        "mapped_boxes_csv_sha256": sha256_file(Path(args.mapped_boxes_csv)),
        "num_cached_slices": len(case_entries),
        "num_cached_patients": len({entry["patient_id"] for entry in case_entries}),
        "num_skipped_slices": len(skipped),
        "skipped": skipped,
        "qeff_definition": (
            "sum_cs(q_cs * ROI-mean RMS_ch(alpha_cs * U_cs)) / "
            "sum_cs(ROI-mean RMS_ch(alpha_cs * U_cs)); weights are before q"
        ),
        "qeff_interpretation": (
            "ROI-weighted effective reliability; not predicted lesion reliability "
            "and not a spatial q-map"
        ),
        "qeff_unit_checks": unit_checks,
        "diagnostic_output_equivalence_max_abs": output_equivalence_max_abs,
        "direct_rms_reproduction_error_max": direct_rms_error_max,
        "lesion_qeff_not_estimable_slices": int(qeff_unavailable),
        "lesion_qeff_not_estimable_fraction": float(qeff_unavailable_fraction),
        "qeff_aggregate_reporting_allowed": bool(qeff_aggregate_allowed),
        "case_metadata": [str(cache_dir / f"{case_stem((e['patient_id'], e['slice_idx']))}.json") for e in case_entries],
        "case_caches": [str(cache_dir / f"{case_stem((e['patient_id'], e['slice_idx']))}.npz") for e in case_entries],
    }
    atomic_json(output / "inference_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
