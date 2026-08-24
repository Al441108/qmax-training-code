#!/usr/bin/env python3
"""Stage-1 audit for the R=2 PD auxiliary image generation path.

Run this script from the fastmri_pipeline project root. It performs no writes
outside --output-dir and never modifies the dataset or a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


AUDIT_TERMS = (
    "pd_aux_image",
    "pd_aux_acceleration",
    "pd_mask",
    "mask",
    "ifft",
    "rss",
    "root_sum",
    "normalize",
    "crop",
    "complex_abs",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and visualize the R=2 PD auxiliary input."
    )
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--target-acceleration", type=int, default=8)
    parser.add_argument("--pd-acceleration", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument(
        "--indices",
        default=None,
        help="Optional comma-separated dataset indices; overrides --num-samples.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "numel") and value.numel() == 1:
        return value.detach().cpu().item()
    if hasattr(value, "shape"):
        return {
            "type": type(value).__name__,
            "shape": list(value.shape),
            "dtype": str(getattr(value, "dtype", "unknown")),
        }
    return repr(value)


def tensor_to_image(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if np.iscomplexobj(array):
        array = np.abs(array)
    elif array.ndim >= 3 and array.shape[-1] == 2:
        array = np.sqrt(np.square(array[..., 0]) + np.square(array[..., 1]))
    array = np.squeeze(array)
    while array.ndim > 2:
        array = array[array.shape[0] // 2]
    if array.ndim != 2:
        raise ValueError(f"Cannot convert shape {array.shape} to a 2-D image")
    return array.astype(np.float32, copy=False)


def robust_limits(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, [1.0, 99.5])
    if not np.isfinite(high) or high <= low:
        low, high = float(np.min(finite)), float(np.max(finite))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def spectrum(image: np.ndarray) -> np.ndarray:
    centered = image - float(np.mean(image))
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(centered))))


def source_excerpts(source_path: Path, context: int = 3) -> list[dict[str, Any]]:
    lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines()
    hit_lines = {
        index
        for index, line in enumerate(lines)
        if any(term.lower() in line.lower() for term in AUDIT_TERMS)
    }
    groups: list[tuple[int, int]] = []
    for hit in sorted(hit_lines):
        start, end = max(0, hit - context), min(len(lines), hit + context + 1)
        if groups and start <= groups[-1][1]:
            groups[-1] = (groups[-1][0], max(groups[-1][1], end))
        else:
            groups.append((start, end))
    return [
        {
            "start_line": start + 1,
            "end_line": end,
            "text": "\n".join(
                f"{line_number + 1:5d}: {lines[line_number]}"
                for line_number in range(start, end)
            ),
        }
        for start, end in groups
    ]


def choose_indices(dataset: Any, num_samples: int, requested: str | None) -> list[int]:
    if requested:
        indices = [int(item.strip()) for item in requested.split(",") if item.strip()]
        if not indices:
            raise ValueError("--indices did not contain any valid index")
        return indices

    records = getattr(dataset, "records", None)
    if records:
        patient_indices: dict[str, list[int]] = {}
        for index, record in enumerate(records):
            patient = None
            if isinstance(record, dict):
                patient = record.get("patient_id")
            else:
                patient = getattr(record, "patient_id", None)
            patient_key = str(patient) if patient is not None else f"index-{index}"
            patient_indices.setdefault(patient_key, []).append(index)
        selected = [
            indices[len(indices) // 2]
            for indices in list(patient_indices.values())[:num_samples]
        ]
        if selected:
            return selected
    count = min(len(dataset), num_samples)
    if count == 0:
        return []
    return np.linspace(0, len(dataset) - 1, count, dtype=int).tolist()


def make_contact_sheet(
    auxiliary_images: list[np.ndarray],
    reference_images: list[np.ndarray],
    masks: list[np.ndarray | None],
    labels: list[str],
    output_path: Path,
) -> None:
    if not auxiliary_images:
        return
    figure, axes = plt.subplots(
        len(auxiliary_images),
        5,
        figsize=(18, max(3.4 * len(auxiliary_images), 4.0)),
        squeeze=False,
    )
    for row, (auxiliary, reference, mask, label) in enumerate(
        zip(auxiliary_images, reference_images, masks, labels)
    ):
        ref_low, ref_high = robust_limits(reference)
        axes[row, 0].imshow(reference, cmap="gray", vmin=ref_low, vmax=ref_high)
        axes[row, 0].set_title(f"{label}: full PD reference")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(auxiliary, cmap="gray", vmin=ref_low, vmax=ref_high)
        axes[row, 1].set_title("R=2 zero-filled PD (reference window)")
        axes[row, 1].axis("off")

        difference = auxiliary - reference
        diff_limit = float(np.percentile(np.abs(difference), 99.5))
        diff_limit = max(diff_limit, np.finfo(np.float32).eps)
        axes[row, 2].imshow(
            difference,
            cmap="coolwarm",
            vmin=-diff_limit,
            vmax=diff_limit,
        )
        axes[row, 2].set_title("R2-ZF - full PD")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(spectrum(difference), cmap="magma")
        axes[row, 3].set_title("log |FFT(R2-ZF - full PD)|")
        axes[row, 3].axis("off")

        if mask is None:
            axes[row, 4].text(0.5, 0.5, "pd_aux_mask unavailable", ha="center")
        else:
            axes[row, 4].step(np.arange(mask.size), mask, where="mid")
            axes[row, 4].set_ylim(-0.05, 1.05)
            axes[row, 4].set_title(
                f"PD mask: {int(np.sum(mask))}/{mask.size} lines"
            )
            axes[row, 4].set_xlabel("phase-encoding column")
        axes[row, 4].grid(alpha=0.2)

    figure.suptitle(
        "Stage-1 PD audit: full reference versus R=2 zero-filled auxiliary",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_markdown(
    path: Path,
    args: argparse.Namespace,
    module_path: Path,
    class_signature: str,
    source_reports: list[dict[str, Any]],
    inventory: dict[str, Any],
) -> None:
    lines = [
        "# R=2 PD auxiliary provenance audit",
        "",
        "## Runtime configuration",
        "",
        f"- Project root: `{Path(args.project_root).resolve()}`",
        f"- Metadata CSV: `{Path(args.metadata_csv).resolve()}`",
        f"- Split: `{args.split}`",
        f"- Target acceleration: `{args.target_acceleration}`",
        f"- PD auxiliary acceleration: `{args.pd_acceleration}`",
        f"- Dataset module: `{module_path}`",
        f"- Dataset SHA-256: `{sha256_file(module_path)}`",
        f"- Constructor: `{class_signature}`",
        "",
        "## Dataset item inventory",
        "",
        "```json",
        json.dumps(inventory, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Source excerpts requiring manual interpretation",
        "",
        "Check the excerpts in this order: auxiliary k-space selection/mask, IFFT, "
        "coil combination, crop, magnitude conversion, normalization, return dict.",
        "",
    ]
    for report in source_reports:
        lines.extend(
            [
                f"### `{report['path']}`",
                "",
                f"SHA-256: `{report['sha256']}`",
                "",
            ]
        )
        if not report["excerpts"]:
            lines.extend(["No audit-term matches found.", ""])
        for excerpt in report["excerpts"]:
            lines.extend(
                [
                    f"#### Lines {excerpt['start_line']}-{excerpt['end_line']}",
                    "",
                    "```python",
                    excerpt["text"],
                    "```",
                    "",
                ]
            )
    lines.extend(
        [
            "## Questions to answer",
            "",
            "- Is `pd_aux_image` computed from masked R=2 k-space?",
            "- Is it a direct zero-filled IFFT/RSS image or a learned reconstruction?",
            "- What mask and ACS are used for PD?",
            "- Where are magnitude conversion, crop and normalization applied?",
            "- Are train/val/test transformations identical apart from augmentation?",
            "- Does the contact sheet show sampling-pattern-consistent structure?",
            "",
            "Do not infer causality from the image alone. The fixed-checkpoint "
            "R2-ZF/R2-Recon/No-PD intervention is Stage 2.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.pd_acceleration != 2:
        raise ValueError("Stage 1 is pre-specified for pd-acceleration=2")
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")

    project_root = Path(args.project_root).resolve()
    metadata_csv = Path(args.metadata_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not metadata_csv.is_file():
        raise FileNotFoundError(metadata_csv)
    if not (project_root / "src").is_dir():
        raise FileNotFoundError(f"Missing project src directory: {project_root / 'src'}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(project_root))
    from src.dataset_paired_multicoil_aux_pd_r2 import (  # noqa: PLC0415
        PairedMulticoilAuxPDToPDFSDataset,
    )

    dataset = PairedMulticoilAuxPDToPDFSDataset(
        metadata_csv=str(metadata_csv),
        split=args.split,
        pdfs_acceleration=args.target_acceleration,
        pd_aux_acceleration=args.pd_acceleration,
        patient_ids=None,
        slices_per_patient=None,
        edge_weight=1.0,
    )
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty")

    module_path = Path(inspect.getsourcefile(PairedMulticoilAuxPDToPDFSDataset)).resolve()
    class_signature = str(inspect.signature(PairedMulticoilAuxPDToPDFSDataset))
    source_paths = [module_path]
    for relative_path in ("src/masks.py", "src/fft_utils.py"):
        candidate = (project_root / relative_path).resolve()
        if candidate.is_file() and candidate not in source_paths:
            source_paths.append(candidate)
    source_reports = [
        {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "excerpts": source_excerpts(source_path),
        }
        for source_path in source_paths
    ]
    indices = choose_indices(dataset, args.num_samples, args.indices)

    auxiliary_images: list[np.ndarray] = []
    reference_images: list[np.ndarray] = []
    masks: list[np.ndarray | None] = []
    labels: list[str] = []
    samples: list[dict[str, Any]] = []
    for index in indices:
        if index < 0 or index >= len(dataset):
            raise IndexError(f"Dataset index {index} outside [0, {len(dataset)})")
        item = dataset[index]
        if not isinstance(item, dict):
            raise TypeError(f"Expected dict item, received {type(item).__name__}")
        if "pd_aux_image" not in item:
            raise KeyError(
                f"Dataset item has no pd_aux_image; available keys: {sorted(item)}"
            )
        auxiliary_image = tensor_to_image(item["pd_aux_image"])
        if "pd_target_raw" not in item:
            raise KeyError("Dataset item has no pd_target_raw full-PD reference")
        reference_image = tensor_to_image(item["pd_target_raw"])
        if bool(item.get("pd_flip_lr", False)):
            reference_image = np.flip(reference_image, axis=-1).copy()
        if auxiliary_image.shape != reference_image.shape:
            raise ValueError(
                "R2 auxiliary/reference shape mismatch: "
                f"{auxiliary_image.shape} versus {reference_image.shape}"
            )
        pd_mask = item.get("pd_aux_mask")
        if pd_mask is not None:
            if hasattr(pd_mask, "detach"):
                pd_mask = pd_mask.detach().cpu().numpy()
            pd_mask = np.asarray(pd_mask).reshape(-1)
        record = None
        if hasattr(dataset, "records") and index < len(dataset.records):
            record = dataset.records[index]
        sample = {
            "index": index,
            "record": jsonable(record),
            "item": {key: jsonable(value) for key, value in item.items()},
            "pd_aux_image_statistics": {
                "min": float(np.nanmin(auxiliary_image)),
                "max": float(np.nanmax(auxiliary_image)),
                "mean": float(np.nanmean(auxiliary_image)),
                "std": float(np.nanstd(auxiliary_image)),
                "shape": list(auxiliary_image.shape),
            },
            "r2_zf_vs_full_pd": {
                "relative_l1": float(
                    np.sum(np.abs(auxiliary_image - reference_image))
                    / max(np.sum(np.abs(reference_image)), np.finfo(np.float32).eps)
                ),
                "nmse": float(
                    np.sum(np.square(auxiliary_image - reference_image))
                    / max(np.sum(np.square(reference_image)), np.finfo(np.float32).eps)
                ),
                "correlation": float(
                    np.corrcoef(auxiliary_image.ravel(), reference_image.ravel())[0, 1]
                ),
            },
        }
        patient = None
        if isinstance(record, dict):
            patient = record.get("patient_id")
        label = f"index={index}" + (f", patient={patient}" if patient is not None else "")
        samples.append(sample)
        auxiliary_images.append(auxiliary_image)
        reference_images.append(reference_image)
        masks.append(pd_mask)
        labels.append(label)

    inventory = {
        "dataset_length": len(dataset),
        "selected_indices": indices,
        "samples": samples,
    }
    (output_dir / "pd_aux_inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(
        output_dir / "pd_aux_provenance.md",
        args,
        module_path,
        class_signature,
        source_reports,
        inventory,
    )
    make_contact_sheet(
        auxiliary_images,
        reference_images,
        masks,
        labels,
        output_dir / "pd_aux_r2_contact_sheet.png",
    )

    summary = {
        "status": "ok",
        "dataset_module": str(module_path),
        "dataset_sha256": sha256_file(module_path),
        "audited_source_files": [
            {"path": report["path"], "sha256": report["sha256"]}
            for report in source_reports
        ],
        "dataset_length": len(dataset),
        "selected_indices": indices,
        "outputs": [
            str(output_dir / "pd_aux_provenance.md"),
            str(output_dir / "pd_aux_inventory.json"),
            str(output_dir / "pd_aux_r2_contact_sheet.png"),
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
