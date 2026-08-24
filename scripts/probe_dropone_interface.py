#!/usr/bin/env python3
"""Read-only source/module probe needed before implementing valid drop-one."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "scripts"))
    from evaluate_pd_oracle_stage2a import make_model

    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = make_model(checkpoint["config"], torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    modules = []
    source_files: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        lowered = f"{name} {class_name}".lower()
        if any(term in lowered for term in ("fusion", "controller", "reliability")):
            source_path_text = inspect.getsourcefile(module.__class__)
            source_path = Path(source_path_text).resolve() if source_path_text else None
            try:
                forward_signature = str(inspect.signature(module.forward))
            except (TypeError, ValueError):
                forward_signature = "unavailable"
            try:
                class_source = inspect.getsource(module.__class__)
            except (OSError, TypeError):
                class_source = "unavailable"
            modules.append(
                {
                    "name": name,
                    "class": class_name,
                    "forward_signature": forward_signature,
                    "source_file": str(source_path) if source_path else None,
                    "parameter_count": sum(p.numel() for p in module.parameters()),
                    "class_source": class_source,
                }
            )
            if source_path and source_path.is_file():
                source_files[str(source_path)] = {
                    "sha256": sha256_file(source_path),
                }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config": checkpoint["config"],
        "candidate_modules": modules,
        "source_files": source_files,
        "instruction": (
            "Return this JSON plus the FusionPilotScaleController class source. "
            "Do not implement drop-one by zeroing a whole module output."
        ),
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
