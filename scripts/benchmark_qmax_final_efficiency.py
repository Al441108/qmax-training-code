#!/usr/bin/env python3
from __future__ import annotations

"""Unified forward-only efficiency benchmark on one locked validation slice."""

import argparse
import csv
import gc
import json
import platform
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import numpy as np
import torch
from fastmri.models import VarNet
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_m2_prnf_final_comparison_R8 import (  # noqa: E402
    GAIN_MODEL,
    load_baseline,
    load_fusion,
)
from scripts.evaluate_qmax_counterfactuals import ManifestDataset  # noqa: E402
from scripts.qmax_common import (  # noqa: E402
    IndexedDataset,
    make_dataset,
    prepare_batch,
    set_seed,
    sha256_file,
)
from src.m2_prnf_qmax_compactswin_varnet import (  # noqa: E402
    QMaxCompactSwinAuxPDVarNet,
)
from src.m2_prnf_qmax_varnet import QMaxAuxPDVarNet  # noqa: E402


PROTOCOL_VERSION = "QMax-final-efficiency-v1"
MODEL_ORDER = (
    "single_varnet_r8",
    "m2u_augmented",
    "fifth_arm",
    "qmax_core_stagea_e30",
    "qmax_full_stagea_e60",
    "qmax_full_stageb_e30",
)
MODEL_LABELS = {
    "single_varnet_r8": "Single VarNet R=8",
    "m2u_augmented": "M2-U Augmented",
    "fifth_arm": "Fifth arm",
    "qmax_core_stagea_e30": "QMax-Core (Stage A)",
    "qmax_full_stagea_e60": "QMax-Full (Stage A, final)",
    "qmax_full_stageb_e30": "QMax-Full (CompactSwin)",
}


def install_amp_diagnostic_quantile_compatibility() -> None:
    """Cast detached FP16/BF16 diagnostic quantiles to FP32."""
    current = torch.Tensor.quantile
    if bool(getattr(current, "_qmax_amp_diagnostic_compat", False)):
        return
    original = current

    def compatible_quantile(tensor, *args, **kwargs):
        if tensor.dtype in (torch.float16, torch.bfloat16):
            tensor = tensor.float()
        return original(tensor, *args, **kwargs)

    compatible_quantile._qmax_amp_diagnostic_compat = True
    torch.Tensor.quantile = compatible_quantile


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def torch_load(path: Path) -> Mapping[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Checkpoint is not a mapping: {path}")
    return value


def count_model(model: torch.nn.Module) -> dict[str, int]:
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    return {
        "parameter_count": sum(value.numel() for value in parameters),
        "trainable_parameter_count": sum(
            value.numel() for value in parameters if value.requires_grad
        ),
        "parameter_bytes": sum(
            value.numel() * value.element_size() for value in parameters
        ),
        "buffer_bytes": sum(
            value.numel() * value.element_size() for value in buffers
        ),
    }


def _clean_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        next_key = str(key)
        if next_key.startswith("module."):
            next_key = next_key[len("module.") :]
        if next_key.startswith("model."):
            next_key = next_key[len("model.") :]
        cleaned[next_key] = value
    return cleaned


def _checkpoint_state(checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    for key in ("model_state_dict", "state_dict", "model"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def _nested_config(config: Mapping[str, Any], key: str, default: Any) -> Any:
    if key in config:
        return config[key]
    for nested_key in ("model_kwargs", "model_config", "args"):
        nested = config.get(nested_key)
        if isinstance(nested, Mapping) and key in nested:
            return nested[key]
    return default


def load_single_varnet(args: Any, device: torch.device) -> VarNet:
    checkpoint = torch_load(Path(args.single_checkpoint))
    raw = checkpoint.get("config", {})
    if isinstance(raw, Mapping):
        config = dict(raw)
    elif hasattr(raw, "__dict__"):
        config = vars(raw)
    else:
        config = {}
    kwargs = {
        "num_cascades": int(_nested_config(config, "num_cascades", args.single_num_cascades)),
        "sens_chans": int(_nested_config(config, "sens_chans", args.single_sens_chans)),
        "sens_pools": int(_nested_config(config, "sens_pools", args.single_sens_pools)),
        "chans": int(_nested_config(config, "chans", args.single_chans)),
        "pools": int(_nested_config(config, "pools", args.single_pools)),
    }
    model = VarNet(**kwargs).to(device)
    model.load_state_dict(_clean_state_dict(_checkpoint_state(checkpoint)), strict=True)
    return model


def load_qmax_stagea(path: Path, expected_variant: str) -> torch.nn.Module:
    checkpoint = torch_load(path)
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError(f"QMax checkpoint lacks config: {path}")
    if str(config.get("qmax_variant")) != expected_variant:
        raise RuntimeError(
            f"Expected {expected_variant}, got {config.get('qmax_variant')}"
        )
    kwargs = dict(config["model_kwargs"])
    kwargs.pop("backbone_variant", None)
    model = QMaxAuxPDVarNet(qmax_variant=expected_variant, **kwargs)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def load_qmax_stageb(path: Path) -> torch.nn.Module:
    checkpoint = torch_load(path)
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError("Stage-B checkpoint lacks config")
    if str(config.get("qmax_variant")) != "qmax_full":
        raise RuntimeError("Stage-B efficiency checkpoint is not qmax_full")
    kwargs = dict(config["model_kwargs"])
    declared = kwargs.pop("backbone_variant", config.get("backbone_variant"))
    if declared != "compactswin":
        raise RuntimeError(f"Expected CompactSwin backbone, got {declared}")
    model = QMaxCompactSwinAuxPDVarNet(qmax_variant="qmax_full", **kwargs)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def central_metric_blind_batch(
    metadata: Path, full_clean_manifest: Path, device: torch.device
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    source = IndexedDataset(
        make_dataset(
            str(metadata), "val", acceleration=8, pd_aux_acceleration=2
        )
    )
    dataset = ManifestDataset(source, read_json(full_clean_manifest))
    by_patient: dict[str, list[tuple[int, int]]] = {}
    for local_index, record in enumerate(dataset.records):
        by_patient.setdefault(str(record["patient_id"]), []).append(
            (int(record["slice_idx"]), local_index)
        )
    patients = sorted(by_patient)
    if not patients:
        raise RuntimeError("Locked full-clean validation manifest is empty")
    patient = patients[len(patients) // 2]
    values = sorted(by_patient[patient])
    slice_idx, local_index = values[len(values) // 2]
    batch = next(iter(DataLoader(dataset, batch_size=1, sampler=[local_index])))
    kspace, mask, pd, target, _indices = prepare_batch(batch, device)
    prepared = {
        "kspace": kspace,
        "mask": mask,
        "pd": pd,
        "target": target,
        "available": torch.ones(1, device=device),
        "num_low_frequencies": _num_low_frequencies(batch),
    }
    identity = {
        "selection_rule": (
            "metric-blind: middle patient in sorted locked full-clean cohort; "
            "central available slice"
        ),
        "patient_id": patient,
        "slice_idx": slice_idx,
        "kspace_shape": list(kspace.shape),
        "mask_shape": list(mask.shape),
        "pd_shape": list(pd.shape),
        "target_shape": list(target.shape),
    }
    return prepared, identity


def _num_low_frequencies(batch: Mapping[str, Any]):
    for key in ("pdfs_num_low_frequencies", "num_low_frequencies", "num_low_freqs"):
        if key in batch:
            value = batch[key]
            if torch.is_tensor(value):
                value = value.detach().cpu().flatten()
                if len(value) == 1:
                    return int(value.item())
            return value
    return None


def aux_forward(model: torch.nn.Module, inputs: Mapping[str, Any]):
    return model(
        inputs["kspace"],
        inputs["mask"],
        inputs["pd"],
        inputs["available"],
        return_aux=False,
    )


def single_forward(model: torch.nn.Module, inputs: Mapping[str, Any]):
    return model(
        inputs["kspace"], inputs["mask"], inputs["num_low_frequencies"]
    )


def synchronize() -> None:
    torch.cuda.synchronize()


@torch.no_grad()
def benchmark_model(
    *,
    model_name: str,
    model: torch.nn.Module,
    inputs: Mapping[str, Any],
    forward: Callable[[torch.nn.Module, Mapping[str, Any]], Any],
    checkpoint: Path,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> dict[str, Any]:
    model = model.to(device).eval()
    counts = count_model(model)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        for _ in range(warmup):
            output = forward(model, inputs)
            if isinstance(output, tuple):
                output = output[0]
            if not torch.isfinite(output).all():
                raise RuntimeError(f"Non-finite warm-up output for {model_name}")
    synchronize()
    del output
    torch.cuda.empty_cache()
    synchronize()

    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        output = forward(model, inputs)
        if isinstance(output, tuple):
            output = output[0]
    synchronize()
    if not torch.isfinite(output).all():
        raise RuntimeError(f"Non-finite measured output for {model_name}")
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    timings = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            output = forward(model, inputs)
            if isinstance(output, tuple):
                output = output[0]
        end.record()
        synchronize()
        timings.append(float(start.elapsed_time(end)))
    values = np.asarray(timings, dtype=np.float64)
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    return {
        "model": model_name,
        "model_label": MODEL_LABELS[model_name],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        **counts,
        "warmup_iterations": warmup,
        "timed_iterations": repeats,
        "batch_size": 1,
        "precision": "CUDA AMP FP16",
        "timing_scope": "model forward only; excludes data loading and host-device transfer",
        "latency_median_ms": float(median),
        "latency_q1_ms": float(q1),
        "latency_q3_ms": float(q3),
        "latency_iqr_ms": float(q3 - q1),
        "latency_mean_ms": float(values.mean()),
        "latency_sample_sd_ms": float(values.std(ddof=1)),
        "baseline_allocated_bytes": int(baseline_allocated),
        "peak_allocated_bytes": int(peak_allocated),
        "incremental_forward_peak_bytes": int(peak_allocated - baseline_allocated),
        "baseline_reserved_bytes": int(baseline_reserved),
        "peak_reserved_bytes": int(peak_reserved),
        "output_shape": list(output.shape),
    }


def validate_checkpoint_role(name: str, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if name in {"m2u_augmented", "fifth_arm"} and path.name != "model_best.pt":
        raise RuntimeError(f"{name} must use frozen historical model_best.pt")
    if name == "qmax_core_stagea_e30" and (
        path.name != "model_last.pt" or path.parent.name != "epoch30"
    ):
        raise RuntimeError("StageA-Core must use epoch30/model_last.pt")
    if name == "qmax_full_stagea_e60" and (
        path.name != "model_last.pt" or path.parent.name != "epoch60"
    ):
        raise RuntimeError("Final StageA-Full must use epoch60/model_last.pt")
    if name == "qmax_full_stageb_e30" and (
        path.name != "model_last.pt" or path.parent.name != "epoch30"
    ):
        raise RuntimeError("StageB-Full must use epoch30/model_last.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--full_clean_manifest", required=True)
    parser.add_argument("--robustness_manifest", required=True)
    parser.add_argument("--condition_manifest", required=True)
    parser.add_argument("--single_checkpoint", required=True)
    parser.add_argument("--m2u_augmented_checkpoint", required=True)
    parser.add_argument("--fifth_checkpoint", required=True)
    parser.add_argument("--stagea_core_checkpoint", required=True)
    parser.add_argument("--stagea_full_checkpoint", required=True)
    parser.add_argument("--stageb_full_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--single_num_cascades", type=int, default=12)
    parser.add_argument("--single_chans", type=int, default=18)
    parser.add_argument("--single_pools", type=int, default=4)
    parser.add_argument("--single_sens_chans", type=int, default=8)
    parser.add_argument("--single_sens_pools", type=int, default=4)
    args = parser.parse_args()

    if args.seed != 42 or args.warmup < 5 or args.repeats < 50:
        raise ValueError("Requires seed=42, warmup>=5 and repeats>=50")
    if not torch.cuda.is_available():
        raise RuntimeError("Efficiency benchmark requires CUDA")
    install_amp_diagnostic_quantile_compatibility()
    set_seed(args.seed)
    device = torch.device("cuda")

    paths = {
        key: Path(value).resolve()
        for key, value in {
            "metadata": args.metadata_csv,
            "full_clean": args.full_clean_manifest,
            "robustness": args.robustness_manifest,
            "conditions": args.condition_manifest,
            "single_varnet_r8": args.single_checkpoint,
            "m2u_augmented": args.m2u_augmented_checkpoint,
            "fifth_arm": args.fifth_checkpoint,
            "qmax_core_stagea_e30": args.stagea_core_checkpoint,
            "qmax_full_stagea_e60": args.stagea_full_checkpoint,
            "qmax_full_stageb_e30": args.stageb_full_checkpoint,
        }.items()
    }
    for key in ("metadata", "full_clean", "robustness", "conditions"):
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])
    for name in MODEL_ORDER:
        validate_checkpoint_role(name, paths[name])

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "qmax_final_efficiency.csv"
    audit_path = output_dir / "qmax_final_efficiency_audit.json"
    raw_path = output_dir / "qmax_final_efficiency_raw.json"
    if any(path.exists() for path in (table_path, audit_path, raw_path)):
        raise RuntimeError("Refusing to overwrite existing efficiency results")

    inputs, case_identity = central_metric_blind_batch(
        paths["metadata"], paths["full_clean"], device
    )
    metadata_hash = sha256_file(paths["metadata"])
    clean_hash = sha256_file(paths["full_clean"])
    robustness_hash = sha256_file(paths["robustness"])
    condition_hash = sha256_file(paths["conditions"])

    def loader(name: str) -> tuple[torch.nn.Module, Callable[..., Any]]:
        if name == "single_varnet_r8":
            ns = SimpleNamespace(
                single_checkpoint=str(paths[name]),
                single_num_cascades=args.single_num_cascades,
                single_chans=args.single_chans,
                single_pools=args.single_pools,
                single_sens_chans=args.single_sens_chans,
                single_sens_pools=args.single_sens_pools,
            )
            return load_single_varnet(ns, torch.device("cpu")), single_forward
        if name == "m2u_augmented":
            model, *_ = load_baseline(
                paths[name], "m2u_augmented", torch.device("cpu"),
                clean_hash, robustness_hash, metadata_hash,
            )
            return model, aux_forward
        if name == "fifth_arm":
            model, *_ = load_fusion(
                paths[name], GAIN_MODEL, torch.device("cpu"), clean_hash,
                robustness_hash, condition_hash, metadata_hash,
            )
            return model, aux_forward
        if name == "qmax_core_stagea_e30":
            return load_qmax_stagea(paths[name], "qmax_core"), aux_forward
        if name == "qmax_full_stagea_e60":
            return load_qmax_stagea(paths[name], "qmax_full"), aux_forward
        if name == "qmax_full_stageb_e30":
            return load_qmax_stageb(paths[name]), aux_forward
        raise KeyError(name)

    results = []
    for name in MODEL_ORDER:
        gc.collect()
        torch.cuda.empty_cache()
        model, forward = loader(name)
        results.append(
            benchmark_model(
                model_name=name,
                model=model,
                inputs=inputs,
                forward=forward,
                checkpoint=paths[name],
                warmup=args.warmup,
                repeats=args.repeats,
                device=device,
            )
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    reference = next(row for row in results if row["model"] == "m2u_augmented")
    for row in results:
        row["latency_ratio_vs_m2u_augmented"] = (
            row["latency_median_ms"] / reference["latency_median_ms"]
        )
        row["peak_allocated_ratio_vs_m2u_augmented"] = (
            row["peak_allocated_bytes"] / reference["peak_allocated_bytes"]
        )
    write_csv(table_path, results)
    raw = {
        "protocol_version": PROTOCOL_VERSION,
        "case": case_identity,
        "results": results,
    }
    raw_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    audit = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "passed",
        "scope": "locked validation only; held-out test not accessed",
        "model_order": list(MODEL_ORDER),
        "failed_stageb_core_status": "excluded because formal training failed with non-finite loss",
        "case": case_identity,
        "batch_size": 1,
        "precision": "CUDA AMP FP16",
        "warmup_iterations": args.warmup,
        "timed_iterations": args.repeats,
        "timing_statistic": "median with interquartile range",
        "timing_scope": "model forward only",
        "device": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "input_hashes": {
            "metadata_csv": metadata_hash,
            "full_clean_manifest": clean_hash,
            "robustness_manifest": robustness_hash,
            "condition_manifest": condition_hash,
        },
        "outputs": {
            "table": str(table_path),
            "raw": str(raw_path),
            "audit": str(audit_path),
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
