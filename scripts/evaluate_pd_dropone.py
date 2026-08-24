#!/usr/bin/env python3
"""Fixed-checkpoint scale/cascade drop-one audit for Global-direct.

The intervention is applied at each cascade-specific M2PRNFFeatureFusion
output.  Its fused feature is replaced by the target feature that entered the
module, while the diagnostic dictionary is preserved.  This removes exactly
that auxiliary addition without changing q, alpha, the shared controller, or
the target branch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument(
        "--interventions",
        default="cell,cascade,scale",
        help="Comma-separated subset of cell,cascade,scale.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fusion_grid(model: torch.nn.Module) -> list[list[torch.nn.Module]]:
    grid: list[list[torch.nn.Module]] = []
    for cascade_index, cascade in enumerate(model.cascades):
        regulariser = getattr(cascade, "regulariser", None)
        fusions = getattr(regulariser, "fusions", None)
        if fusions is None:
            raise RuntimeError(
                f"cascades.{cascade_index}.regulariser.fusions was not found"
            )
        row = list(fusions)
        if not row:
            raise RuntimeError(f"Cascade {cascade_index} has no fusion modules")
        if any(module.__class__.__name__ != "M2PRNFFeatureFusion" for module in row):
            classes = [module.__class__.__name__ for module in row]
            raise RuntimeError(
                f"Unexpected fusion classes at cascade {cascade_index}: {classes}"
            )
        grid.append(row)
    widths = {len(row) for row in grid}
    if len(widths) != 1:
        raise RuntimeError(f"Inconsistent number of scales across cascades: {widths}")
    if len({id(module) for row in grid for module in row}) != sum(map(len, grid)):
        raise RuntimeError("Fusion modules are unexpectedly shared across cells")
    return grid


def drop_hook(
    _module: torch.nn.Module,
    inputs: tuple[Any, ...],
    output: Any,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not inputs or not torch.is_tensor(inputs[0]):
        raise RuntimeError("Fusion hook did not receive target as its first input")
    if not isinstance(output, tuple) or len(output) != 2:
        raise RuntimeError(
            "Expected M2PRNFFeatureFusion output (fused, diagnostics)"
        )
    target = inputs[0]
    fused, diagnostics = output
    if not torch.is_tensor(fused) or fused.shape != target.shape:
        raise RuntimeError(
            f"Fusion/target shape mismatch: {getattr(fused, 'shape', None)} "
            f"versus {target.shape}"
        )
    if not isinstance(diagnostics, dict):
        raise RuntimeError("Fusion diagnostics are not a dictionary")
    return target, diagnostics


def noop_hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
    return output


@contextmanager
def installed_hooks(
    modules: list[torch.nn.Module],
    hook,
) -> Iterator[None]:
    handles = [module.register_forward_hook(hook) for module in modules]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def make_interventions(
    grid: list[list[torch.nn.Module]], requested: set[str]
) -> list[dict[str, Any]]:
    n_cascades = len(grid)
    n_scales = len(grid[0])
    result: list[dict[str, Any]] = [
        {
            "condition": "baseline_r2",
            "kind": "baseline",
            "cascade": None,
            "scale": None,
            "modules": [],
        },
        {
            "condition": "drop_all",
            "kind": "all",
            "cascade": None,
            "scale": None,
            "modules": [module for row in grid for module in row],
        },
    ]
    if "cascade" in requested:
        for cascade in range(n_cascades):
            result.append(
                {
                    "condition": f"drop_cascade_{cascade:02d}",
                    "kind": "cascade",
                    "cascade": cascade,
                    "scale": None,
                    "modules": list(grid[cascade]),
                }
            )
    if "scale" in requested:
        for scale in range(n_scales):
            result.append(
                {
                    "condition": f"drop_scale_{scale}",
                    "kind": "scale",
                    "cascade": None,
                    "scale": scale,
                    "modules": [grid[cascade][scale] for cascade in range(n_cascades)],
                }
            )
    if "cell" in requested:
        for cascade in range(n_cascades):
            for scale in range(n_scales):
                result.append(
                    {
                        "condition": f"drop_cell_c{cascade:02d}_s{scale}",
                        "kind": "cell",
                        "cascade": cascade,
                        "scale": scale,
                        "modules": [grid[cascade][scale]],
                    }
                )
    return result


def aggregate_patients(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    metadata: dict[str, dict[str, Any]] = {}
    for row in rows:
        grouped[(row["condition"], row["patient_id"])].append(float(row["l1"]))
        metadata[row["condition"]] = {
            "kind": row["kind"],
            "cascade": row["cascade"],
            "scale": row["scale"],
        }
    result = []
    for (condition, patient_id), values in sorted(grouped.items()):
        result.append(
            {
                "condition": condition,
                "patient_id": patient_id,
                "num_slices": len(values),
                "l1": float(np.mean(values)),
                **metadata[condition],
            }
        )
    return result


def bootstrap_ci(
    values: np.ndarray, replicates: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(
        values, size=(replicates, values.size), replace=True
    ).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def summarize(
    patient_rows: list[dict[str, Any]],
    definitions: list[dict[str, Any]],
    bootstrap_replicates: int,
) -> list[dict[str, Any]]:
    by_condition: dict[str, dict[str, float]] = defaultdict(dict)
    for row in patient_rows:
        by_condition[row["condition"]][row["patient_id"]] = float(row["l1"])
    baseline = by_condition["baseline_r2"]
    result = []
    for index, definition in enumerate(definitions):
        condition = definition["condition"]
        patients = sorted(set(baseline) & set(by_condition[condition]))
        delta = np.asarray(
            [
                by_condition[condition][patient] - baseline[patient]
                for patient in patients
            ],
            dtype=np.float64,
        )
        ci_low, ci_high = bootstrap_ci(
            delta, bootstrap_replicates, 20260727 + index
        )
        result.append(
            {
                "condition": condition,
                "kind": definition["kind"],
                "cascade": definition["cascade"],
                "scale": definition["scale"],
                "num_patients": len(patients),
                "baseline_patient_l1": float(
                    np.mean([baseline[patient] for patient in patients])
                ),
                "intervention_patient_l1": float(
                    np.mean(
                        [by_condition[condition][patient] for patient in patients]
                    )
                ),
                "delta_l1_drop_minus_baseline": float(delta.mean()),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "patients_where_retained_injection_helps": int((delta > 0).sum()),
                "patients_where_drop_helps": int((delta < 0).sum()),
            }
        )
    return result


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1000:
        raise ValueError("--bootstrap-replicates must be at least 1000")
    requested = {item.strip() for item in args.interventions.split(",") if item.strip()}
    unknown = requested - {"cell", "cascade", "scale"}
    if unknown:
        raise ValueError(f"Unknown interventions: {sorted(unknown)}")

    project_root = Path(args.project_root).resolve()
    metadata_csv = Path(args.metadata_csv).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    for path in (metadata_csv, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "scripts"))

    from evaluate_pd_oracle_stage2a import (
        center_crop,
        condition_pd,
        l1_per_sample,
        make_model,
        prepare_common,
    )
    from src.dataset_paired_multicoil_aux_pd_r2 import (
        PairedMulticoilAuxPDToPDFSDataset,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    if config.get("fusion_design") != "global_direct":
        raise ValueError(
            "This exact drop intervention is validated only for global_direct; "
            f"got {config.get('fusion_design')}"
        )
    if config.get("variant") != "prnf_no_need":
        raise ValueError(
            "Expected the selected Global-direct prnf_no_need checkpoint; "
            f"got {config.get('variant')}"
        )

    model = make_model(config, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    grid = fusion_grid(model)
    definitions = make_interventions(grid, requested)
    if len(grid) != int(config.get("num_cascades", 12)):
        raise RuntimeError("Cascade count disagrees with checkpoint configuration")

    patient_ids = config.get(f"{args.split}_patient_ids")
    dataset = PairedMulticoilAuxPDToPDFSDataset(
        metadata_csv=str(metadata_csv),
        split=args.split,
        pdfs_acceleration=int(config.get("acceleration", 8)),
        pd_aux_acceleration=int(config.get("pd_aux_acceleration", 2)),
        patient_ids=patient_ids,
        slices_per_patient=None,
        edge_weight=1.0,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    rows: list[dict[str, Any]] = []
    max_drop_all_no_pd_abs = 0.0
    noop_max_abs: float | None = None
    with torch.inference_mode():
        for sample_number, batch in enumerate(loader):
            if args.max_samples is not None and sample_number >= args.max_samples:
                break
            kspace, mask, target = prepare_common(batch, device)
            pd_r2, available = condition_pd(batch, "r2_zf", device)
            pd_none, unavailable = condition_pd(batch, "no_pd", device)
            patient_id = str(batch["patient_id"][0])
            slice_idx = int(batch["slice_idx"][0])

            predictions: dict[str, torch.Tensor] = {}
            for definition in definitions:
                modules = definition["modules"]
                if modules:
                    with installed_hooks(modules, drop_hook):
                        prediction = model(
                            kspace, mask, pd_r2, available, return_aux=False
                        )
                else:
                    prediction = model(
                        kspace, mask, pd_r2, available, return_aux=False
                    )
                prediction = center_crop(
                    prediction, target.shape[-2], target.shape[-1]
                )
                predictions[definition["condition"]] = prediction
                rows.append(
                    {
                        "condition": definition["condition"],
                        "kind": definition["kind"],
                        "cascade": definition["cascade"],
                        "scale": definition["scale"],
                        "patient_id": patient_id,
                        "slice_idx": slice_idx,
                        "l1": float(l1_per_sample(prediction, target)[0].item()),
                    }
                )

            no_pd_prediction = model(
                kspace, mask, pd_none, unavailable, return_aux=False
            )
            no_pd_prediction = center_crop(
                no_pd_prediction, target.shape[-2], target.shape[-1]
            )
            max_drop_all_no_pd_abs = max(
                max_drop_all_no_pd_abs,
                float(
                    (
                        predictions["drop_all"] - no_pd_prediction
                    ).abs().max().item()
                ),
            )

            if sample_number == 0:
                with installed_hooks(
                    [module for row in grid for module in row], noop_hook
                ):
                    noop_prediction = model(
                        kspace, mask, pd_r2, available, return_aux=False
                    )
                noop_prediction = center_crop(
                    noop_prediction, target.shape[-2], target.shape[-1]
                )
                noop_max_abs = float(
                    (
                        predictions["baseline_r2"] - noop_prediction
                    ).abs().max().item()
                )
                if noop_max_abs != 0.0:
                    raise RuntimeError(
                        f"No-op hook changed output (max abs {noop_max_abs})"
                    )

            if (sample_number + 1) % 10 == 0:
                print(
                    f"Evaluated {sample_number + 1}/{len(dataset)} slices "
                    f"across {len(definitions)} intervention conditions",
                    flush=True,
                )

    patient_rows = aggregate_patients(rows)
    summary_rows = summarize(
        patient_rows, definitions, args.bootstrap_replicates
    )
    summary_rows.sort(
        key=lambda row: (
            {"all": 0, "scale": 1, "cascade": 2, "cell": 3, "baseline": 4}.get(
                str(row["kind"]), 9
            ),
            -float(row["delta_l1_drop_minus_baseline"]),
        )
    )
    manifest = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "metadata_csv": str(metadata_csv),
        "metadata_sha256": sha256_file(metadata_csv),
        "split": args.split,
        "max_samples": args.max_samples,
        "device": str(device),
        "num_dataset_slices": len(dataset),
        "num_evaluated_slices": len(
            {(row["patient_id"], row["slice_idx"]) for row in rows}
        ),
        "num_cascades": len(grid),
        "num_scales": len(grid[0]),
        "num_intervention_conditions": len(definitions),
        "drop_semantics": (
            "At selected cascades.N.regulariser.fusions.S outputs, replace "
            "(target + auxiliary_term, diagnostics) with (target, diagnostics)."
        ),
        "delta_definition": (
            "L1(drop) - L1(baseline R2); positive means the removed injection "
            "was helpful, negative means dropping it improved reconstruction."
        ),
        "controls": {
            "noop_hook_max_abs_output_difference_first_slice": noop_max_abs,
            "drop_all_vs_no_pd_max_abs_output_difference": max_drop_all_no_pd_abs,
            "drop_all_equivalence_tolerance": 1e-6,
            "drop_all_equivalent_to_no_pd": max_drop_all_no_pd_abs <= 1e-6,
        },
        "nonadditivity_warning": (
            "Cell, cascade, and scale effects are separate interventions in a "
            "nonlinear model and must not be arithmetically summed."
        ),
        "ranked_summary": summary_rows,
    }
    write_csv(output_dir / "dropone_slice_metrics.csv", rows)
    write_csv(output_dir / "dropone_patient_metrics.csv", patient_rows)
    write_csv(output_dir / "dropone_summary.csv", summary_rows)
    (output_dir / "dropone_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if max_drop_all_no_pd_abs > 1e-6:
        raise RuntimeError(
            "drop_all did not reproduce no-PD within 1e-6; do not interpret "
            "drop-one results until the hook semantics are reviewed"
        )


if __name__ == "__main__":
    main()
