#!/usr/bin/env python3
from __future__ import annotations

"""Preflight for the v1.5 quality-protected residual-gain pilot."""

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_m2_prnf_quality_gain import (  # noqa: E402
    IndexedDataset,
    ShapeBucketBatchSampler,
    locked_code_hashes,
    make_dataset,
    l1_per_sample,
    prepare_batch,
    residual_scale_override,
    runtime_versions,
    select_patient_ids,
    set_seed,
    sha256_file,
)
from src.fft_utils import center_crop  # noqa: E402
from src.m2_prnf_corruptions import (  # noqa: E402
    CORRUPT_MIXTURE,
    CorruptionConfig,
    HardNegativeSampler,
)
from src.m2_prnf_fusion_pilot_varnet import (  # noqa: E402
    ComplementaryResidualHead,
    M2PRNFFusionPilotVarNet,
    shared_reconstruction_state,
)
from src.m2_prnf_varnet import M2PRNFAuxPDVarNet  # noqa: E402


def model_kwargs(args):
    return dict(
        num_cascades=args.num_cascades,
        sens_chans=args.sens_chans,
        sens_pools=args.sens_pools,
        chans=args.chans,
        pools=args.pools,
        controller_chans=args.controller_chans,
        initial_aux_alpha=args.initial_aux_alpha,
        initial_gate_probability=args.initial_gate_probability,
        initial_need_probability=args.initial_need_probability,
        need_floor=args.need_floor,
    )


def state_difference(reference, candidate):
    if set(reference) != set(candidate):
        return {"keys_identical": False, "maximum_absolute_difference": None}
    maximum = max(
        float((reference[key] - candidate[key]).abs().max().item())
        for key in reference
    )
    return {"keys_identical": True, "maximum_absolute_difference": maximum}


def output_difference(reference, candidate):
    absolute = (reference - candidate).abs()
    denominator = reference.abs().mean().clamp_min(1e-8)
    return {
        "maximum_absolute_difference": float(absolute.max().item()),
        "mean_absolute_difference": float(absolute.mean().item()),
        "relative_mean_absolute_difference": float(
            (absolute.mean() / denominator).item()
        ),
    }


def canonical_json_value(value):
    """Normalise JSON-compatible containers before provenance comparison.

    Dataclass defaults can contain tuples while the persisted v1.3 config has
    the same values represented as JSON lists.  Comparing the raw Python
    containers would therefore report a false mismatch even though their
    serialised protocol values are identical.
    """
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def residual_gradient_test(chans=18):
    set_seed(42)
    head = ComplementaryResidualHead(chans)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    target = torch.randn(2, chans, 16, 16)
    auxiliary = torch.randn_like(target)
    desired = torch.randn_like(target)
    residual, _, _ = head(target, auxiliary)
    initial_exact_zero = float(residual.abs().max().item()) == 0.0
    loss = F.mse_loss(residual, desired)
    loss.backward()
    final_projection_gradient = float(head.residual_out.weight.grad.norm().item())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    residual, _, _ = head(target, auxiliary)
    F.mse_loss(residual, desired).backward()
    context_gradients = [
        float(parameter.grad.norm().item())
        for parameter in head.context.parameters()
        if parameter.grad is not None
    ]
    return {
        "initial_residual_exact_zero": initial_exact_zero,
        "first_step_final_projection_gradient": final_projection_gradient,
        "second_step_context_gradient": sum(context_gradients),
        "passed": initial_exact_zero
        and final_projection_gradient > 0.0
        and sum(context_gradients) > 0.0,
    }


BASE_V13_HASH_PATHS = (
    "src/m2_prnf_varnet.py",
    "src/m2_prnf_corruptions.py",
    "scripts/train_m2_prnf.py",
    "scripts/preflight_m2_prnf.py",
    "scripts/evaluate_m2_prnf_R8.py",
    "src/dataset_paired_multicoil_aux_pd_r2.py",
    "src/fft_utils.py",
    "src/masks.py",
    "FINAL_PROTOCOL_R8.json",
)


def audit_v13_base_hashes(reference_config):
    frozen = reference_config.get("code_hashes", {})
    mismatches = {}
    for relative in BASE_V13_HASH_PATHS:
        path = PROJECT_ROOT / relative
        current = sha256_file(path) if path.is_file() else None
        expected = frozen.get(relative)
        if current != expected:
            mismatches[relative] = {"reference": expected, "current": current}
    return {
        "checked_paths": list(BASE_V13_HASH_PATHS),
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def record_shape(record):
    with h5py.File(record["pdfs_path"], "r") as hf:
        source = (
            hf["reconstruction_rss"]
            if "reconstruction_rss" in hf else hf["kspace"]
        )
        return tuple(int(value) for value in source.shape[-2:])


def build_condition_manifest(dataset, robustness_manifest, seed):
    wanted = {
        (str(patient["patient_id"]), int(slice_index))
        for patient in robustness_manifest["patients"]
        for slice_index in patient["slice_indices"]
    }
    lookup = {
        (str(record["patient_id"]), int(record["slice_idx"])): index
        for index, record in enumerate(dataset.records)
    }
    missing = sorted(wanted - set(lookup))
    if missing:
        raise RuntimeError(f"Robustness entries missing from dataset: {missing[:8]}")
    negative_sampler = HardNegativeSampler(dataset)
    entries = []
    for key in sorted(wanted):
        source_index = int(lookup[key])
        source_record = dataset.records[source_index]
        wrong_slice = negative_sampler.same_patient_wrong_slice(
            source_index,
            random.Random(seed + 1_000_003 * source_index),
        )
        wrong_patient = negative_sampler.wrong_patient_matched_level(
            source_index,
            record_shape(source_record),
            random.Random(seed + 1_000_003 * source_index),
            top_k=8,
        )
        if wrong_slice is None or wrong_patient is None:
            raise RuntimeError(f"No frozen negative candidate for source {source_index}")
        slice_index, slice_delta = wrong_slice
        patient_index, patient_delta = wrong_patient
        slice_record = dataset.records[int(slice_index)]
        patient_record = dataset.records[int(patient_index)]
        entries.append({
            "source_index": source_index,
            "patient_id": key[0],
            "slice_idx": key[1],
            "shift8": {"dy": 0, "dx": 8, "padding_mode": "reflect"},
            "wrong_slice": {
                "replacement_index": int(slice_index),
                "replacement_patient_id": str(slice_record["patient_id"]),
                "replacement_slice_idx": int(slice_record["slice_idx"]),
                "delta_z_norm": float(slice_delta),
            },
            "wrong_patient": {
                "replacement_index": int(patient_index),
                "replacement_patient_id": str(patient_record["patient_id"]),
                "replacement_slice_idx": int(patient_record["slice_idx"]),
                "delta_z_norm": float(patient_delta),
            },
        })
    return {
        "protocol_version": "M2-PRNF-R8-v1.4.1-fusion-pilot-audited",
        "purpose": "Frozen per-slice robustness corruption instances",
        "seed": int(seed),
        "shift8_definition": "horizontal +x 8 pixels with reflect padding",
        "num_entries": len(entries),
        "entries": entries,
    }


def write_or_verify_frozen_manifest(path, manifest):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != manifest:
            raise RuntimeError(
                f"Existing condition manifest differs from regenerated content: {path}"
            )
    else:
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return sha256_file(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--full_clean_manifest", required=True)
    parser.add_argument("--robustness_manifest", required=True)
    parser.add_argument("--condition_manifest", required=True)
    parser.add_argument(
        "--reference_config", required=True,
        help="Completed v1.3 prnf_full config used to audit common presets.",
    )
    parser.add_argument("--acceleration", type=int, default=8)
    parser.add_argument("--pd_aux_acceleration", type=int, default=2)
    parser.add_argument("--formal_batch_size", type=int, default=4)
    parser.add_argument("--formal_max_gpu_memory_gb", type=float, default=94.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_cascades", type=int, default=12)
    parser.add_argument("--chans", type=int, default=18)
    parser.add_argument("--sens_chans", type=int, default=8)
    parser.add_argument("--pools", type=int, default=4)
    parser.add_argument("--sens_pools", type=int, default=4)
    parser.add_argument("--controller_chans", type=int, default=16)
    parser.add_argument("--initial_aux_alpha", type=float, default=0.1)
    parser.add_argument("--initial_gate_probability", type=float, default=0.95)
    parser.add_argument("--initial_need_probability", type=float, default=0.95)
    parser.add_argument("--need_floor", type=float, default=0.25)
    parser.add_argument("--residual_scale", type=float, default=0.1)
    parser.add_argument("--lambda_residual_gain", type=float, default=0.2)
    parser.add_argument("--residual_gain_margin_relative", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.acceleration != 8 or args.pd_aux_acceleration != 2:
        raise ValueError("This frozen pilot preflight requires PD-FS R=8 and PD R=2")
    for path in (
        args.metadata_csv, args.full_clean_manifest, args.robustness_manifest,
        args.reference_config,
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if not 0.0 < args.residual_scale <= 1.0:
        raise ValueError("residual_scale must lie in (0,1]")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference_config = json.loads(
        Path(args.reference_config).read_text(encoding="utf-8")
    )
    expected_common = {
        "acceleration": 8,
        "pd_aux_acceleration": 2,
        "learning_rate": 3e-4,
        "batch_size": 4,
        "grad_accum_steps": 1,
        "num_workers": args.num_workers,
        "num_cascades": args.num_cascades,
        "chans": args.chans,
        "sens_chans": args.sens_chans,
        "pools": args.pools,
        "sens_pools": args.sens_pools,
        "controller_chans": args.controller_chans,
        "initial_aux_alpha": args.initial_aux_alpha,
        "initial_gate_probability": args.initial_gate_probability,
        "initial_need_probability": args.initial_need_probability,
        "need_floor": args.need_floor,
        "lambda_rel": 0.05,
        "lambda_rank": 0.02,
        "aux_loss_ramp_epochs": 5,
        "seed": args.seed,
        "optimizer": {
            "name": "Adam", "betas": [0.9, 0.999], "eps": 1e-8,
            "weight_decay": 0.0,
        },
        "gradient_clip_norm": 10.0,
        "correct_corrupt_loss_weights": {"correct": 0.7, "second_view": 0.3},
        "forward_view_proportions": {"correct": 0.5, "second_view": 0.5},
        "corrupt_view_mixture": dict(CORRUPT_MIXTURE),
        "corruption_config": asdict(CorruptionConfig()),
    }
    preset_mismatches = {
        key: {"expected": expected, "reference": reference_config.get(key)}
        for key, expected in expected_common.items()
        if canonical_json_value(reference_config.get(key))
        != canonical_json_value(expected)
    }
    clean_hash = sha256_file(Path(args.full_clean_manifest))
    robust_hash = sha256_file(Path(args.robustness_manifest))
    if reference_config.get("full_clean_manifest_sha256") != clean_hash:
        preset_mismatches["full_clean_manifest_sha256"] = {
            "expected": clean_hash,
            "reference": reference_config.get("full_clean_manifest_sha256"),
        }
    if reference_config.get("robustness_manifest_sha256") != robust_hash:
        preset_mismatches["robustness_manifest_sha256"] = {
            "expected": robust_hash,
            "reference": reference_config.get("robustness_manifest_sha256"),
        }
    metadata_path = str(Path(args.metadata_csv).resolve())
    metadata_hash = sha256_file(Path(args.metadata_csv))
    if str(Path(reference_config.get("metadata_csv", "")).resolve()) != metadata_path:
        preset_mismatches["metadata_csv"] = {
            "expected": metadata_path,
            "reference": reference_config.get("metadata_csv"),
        }
    historical_metadata_hash = reference_config.get("metadata_sha256")
    metadata_hash_audit = {
        "current_sha256": metadata_hash,
        "historical_sha256": historical_metadata_hash,
        "historical_hash_available": historical_metadata_hash is not None,
        "matches_historical": (
            metadata_hash == historical_metadata_hash
            if historical_metadata_hash is not None else None
        ),
        "limitation": (
            None if historical_metadata_hash is not None else
            "v1.3 did not store metadata_sha256; current content is frozen for v1.4"
        ),
    }
    if historical_metadata_hash is not None and historical_metadata_hash != metadata_hash:
        preset_mismatches["metadata_sha256"] = {
            "expected": historical_metadata_hash,
            "current": metadata_hash,
        }
    base_hash_audit = audit_v13_base_hashes(reference_config)
    if not base_hash_audit["passed"]:
        preset_mismatches["v13_base_code_hashes"] = base_hash_audit["mismatches"]
    preset_audit = {
        "reference_config": str(Path(args.reference_config).resolve()),
        "intentionally_different": {
            "epochs": {"reference": reference_config.get("epochs"), "pilot": 15},
            "variant": {
                "reference": reference_config.get("variant"),
                "pilot": "prnf_no_need",
            },
            "fusion_design": "pilot arm",
        },
        "checked_common_values": expected_common,
        "mismatches": preset_mismatches,
        "passed": not preset_mismatches,
    }
    dataset_args = argparse.Namespace(
        metadata_csv=str(Path(args.metadata_csv).resolve()),
        acceleration=8,
        pd_aux_acceleration=2,
    )
    train_dataset = IndexedDataset(make_dataset(dataset_args, "train"))
    dataset = IndexedDataset(make_dataset(dataset_args, "val"))
    current_train_ids = select_patient_ids(train_dataset, None)
    current_val_ids = select_patient_ids(dataset, None)
    patient_id_audit = {
        "train_matches": current_train_ids == reference_config.get("train_patient_ids"),
        "val_matches": current_val_ids == reference_config.get("val_patient_ids"),
        "current_train_count": len(current_train_ids),
        "current_val_count": len(current_val_ids),
        "reference_train_count": len(reference_config.get("train_patient_ids", [])),
        "reference_val_count": len(reference_config.get("val_patient_ids", [])),
    }
    patient_id_audit["passed"] = (
        patient_id_audit["train_matches"] and patient_id_audit["val_matches"]
    )
    if not patient_id_audit["passed"]:
        preset_mismatches["patient_ids"] = patient_id_audit
    robustness_manifest = json.loads(
        Path(args.robustness_manifest).read_text(encoding="utf-8")
    )
    condition_manifest = build_condition_manifest(
        dataset, robustness_manifest, args.seed
    )
    condition_hash = write_or_verify_frozen_manifest(
        args.condition_manifest, condition_manifest
    )
    preset_audit.update({
        "base_code_hash_audit": base_hash_audit,
        "patient_id_audit": patient_id_audit,
        "metadata_hash_audit": metadata_hash_audit,
        "mismatches": preset_mismatches,
        "passed": not preset_mismatches,
    })
    del train_dataset
    sampler = ShapeBucketBatchSampler(
        dataset, args.formal_batch_size, False, args.seed
    )
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=args.num_workers,
        pin_memory=device.type == "cuda"
    )
    batch = next(iter(loader))
    kspace, mask, pd, target, _ = prepare_batch(batch, device)
    if pd.shape[0] != args.formal_batch_size:
        raise RuntimeError(
            f"First shape bucket has batch {pd.shape[0]}, expected {args.formal_batch_size}"
        )

    kwargs = model_kwargs(args)
    set_seed(args.seed)
    reference_cpu = M2PRNFAuxPDVarNet(variant="m2u_clean", **kwargs)
    reference_parameter_count = sum(
        parameter.numel() for parameter in reference_cpu.parameters()
    )
    reference_state = {
        key: value.detach().clone()
        for key, value in shared_reconstruction_state(reference_cpu).items()
    }
    reference = reference_cpu.to(device).eval()
    with torch.no_grad():
        reference_output = reference(
            kspace[:1], mask[:1], pd[:1], torch.ones(1, device=device)
        )
    del reference, reference_cpu
    if device.type == "cuda":
        torch.cuda.empty_cache()

    equivalence = {}
    for fusion_design in ("global_direct", "hybrid_direct_residual"):
        set_seed(args.seed)
        candidate_cpu = M2PRNFFusionPilotVarNet(
            model_variant="prnf_no_need",
            fusion_design=fusion_design,
            need_scope="residual",
            residual_scale=args.residual_scale,
            **kwargs,
        )
        candidate_parameter_count = sum(
            parameter.numel() for parameter in candidate_cpu.parameters()
        )
        state_report = state_difference(
            reference_state, shared_reconstruction_state(candidate_cpu)
        )
        candidate = candidate_cpu.to(device).eval()
        with torch.no_grad():
            output = candidate(
                kspace[:1], mask[:1], pd[:1], torch.ones(1, device=device),
                reliability_override=1.0, need_override=1.0,
            )
        output_report = output_difference(reference_output, output)
        equivalence[fusion_design] = {
            "parameter_count": candidate_parameter_count,
            "parameters_above_m2u": (
                candidate_parameter_count - reference_parameter_count
            ),
            "relative_parameter_increase": (
                candidate_parameter_count / reference_parameter_count - 1.0
            ),
            "shared_reconstruction_state": state_report,
            "q1_n1_output": output_report,
            "passed": state_report["keys_identical"]
            and state_report["maximum_absolute_difference"] == 0.0
            and output_report["maximum_absolute_difference"] < 1e-5,
        }
        del candidate, candidate_cpu, output
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Separately prove that the legacy arm is exactly the v1.3 no-need model,
    # including controller weights and its unmodified forward output.
    set_seed(args.seed)
    legacy_reference_cpu = M2PRNFAuxPDVarNet(
        variant="prnf_no_need", **kwargs
    )
    legacy_reference_state = {
        key: value.detach().clone()
        for key, value in legacy_reference_cpu.state_dict().items()
    }
    legacy_reference = legacy_reference_cpu.to(device).eval()
    with torch.no_grad():
        legacy_reference_output = legacy_reference(
            kspace[:1], mask[:1], pd[:1], torch.ones(1, device=device)
        )
    del legacy_reference, legacy_reference_cpu
    if device.type == "cuda":
        torch.cuda.empty_cache()
    set_seed(args.seed)
    legacy_candidate_cpu = M2PRNFFusionPilotVarNet(
        model_variant="prnf_no_need",
        fusion_design="legacy_local_direct",
        need_scope="residual",
        residual_scale=args.residual_scale,
        **kwargs,
    )
    legacy_state_report = state_difference(
        legacy_reference_state, legacy_candidate_cpu.state_dict()
    )
    legacy_candidate = legacy_candidate_cpu.to(device).eval()
    with torch.no_grad():
        legacy_candidate_output = legacy_candidate(
            kspace[:1], mask[:1], pd[:1], torch.ones(1, device=device)
        )
    legacy_output_report = output_difference(
        legacy_reference_output, legacy_candidate_output
    )
    equivalence["legacy_vs_v13_prnf_no_need"] = {
        "full_initial_state": legacy_state_report,
        "unmodified_output": legacy_output_report,
        "passed": legacy_state_report["keys_identical"]
        and legacy_state_report["maximum_absolute_difference"] == 0.0
        and legacy_output_report["maximum_absolute_difference"] < 1e-5,
    }
    del (
        legacy_candidate, legacy_candidate_cpu, legacy_candidate_output,
        legacy_reference_output, legacy_reference_state
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    set_seed(args.seed)
    hybrid = M2PRNFFusionPilotVarNet(
        model_variant="prnf_no_need", fusion_design="hybrid_direct_residual",
        need_scope="residual", residual_scale=args.residual_scale, **kwargs
    ).to(device)
    hybrid.eval()
    unavailable = torch.zeros(1, device=device)
    with torch.no_grad():
        missing_a = hybrid(kspace[:1], mask[:1], pd[:1], unavailable)
        missing_b = hybrid(kspace[:1], mask[:1], torch.flip(pd[:1], (-2, -1)), unavailable)
    missing_max_difference = float((missing_a - missing_b).abs().max().item())
    del missing_a, missing_b, reference_output
    if device.type == "cuda":
        torch.cuda.empty_cache()

    hybrid.train()
    optimizer = torch.optim.Adam(hybrid.parameters(), lr=3e-4)
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    paired_kspace = torch.cat([kspace, kspace], dim=0)
    paired_mask = torch.cat([mask, mask], dim=0)
    paired_pd = torch.cat([pd, torch.flip(pd, (-1,))], dim=0)
    paired_target = torch.cat([target, target], dim=0)
    availability = torch.ones(paired_pd.shape[0], device=device)
    with torch.no_grad(), residual_scale_override(hybrid, 0.0):
        direct_reference = hybrid(
            kspace, mask, pd,
            torch.ones(pd.shape[0], device=device),
        )
        direct_reference = center_crop(
            direct_reference, target.shape[-2], target.shape[-1]
        )
        first_direct_l1 = l1_per_sample(direct_reference, target).detach()
    del direct_reference
    prediction, aux = hybrid(
        paired_kspace, paired_mask, paired_pd, availability, return_aux=True
    )
    prediction = center_crop(prediction, paired_target.shape[-2], paired_target.shape[-1])
    reconstruction_loss = F.l1_loss(prediction, paired_target)
    reliability_loss = F.binary_cross_entropy_with_logits(
        aux["q_logits"], torch.ones_like(aux["q_logits"])
    )
    first_clean_l1 = l1_per_sample(prediction[:pd.shape[0]], target)
    first_gain_loss = F.relu(
        first_clean_l1
        - (1.0 - args.residual_gain_margin_relative) * first_direct_l1
    ).mean()
    first_loss = (
        reconstruction_loss + 0.05 * reliability_loss
        + args.lambda_residual_gain * first_gain_loss
    )
    first_loss.backward()
    first_gradients = [
        parameter.grad.detach().norm()
        for parameter in hybrid.parameters()
        if parameter.grad is not None
    ]
    first_gradient_norm = float(torch.stack(first_gradients).norm().item())
    first_loss_finite = bool(torch.isfinite(first_loss).item())
    first_gain_value = float(first_gain_loss.detach().item())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # The inherited VarNet regulariser ends in a zero-initialised out_conv.
    # Consequently, the first full-model backward intentionally updates that
    # final projection before gradients can reach upstream fusion modules.
    # A second real forward/backward is required to audit controller
    # trainability without weakening the four-scale gradient requirement.
    del (
        prediction, aux, reconstruction_loss, reliability_loss, first_loss,
        first_gain_loss, first_clean_l1, first_direct_l1,
    )
    with torch.no_grad(), residual_scale_override(hybrid, 0.0):
        direct_reference = hybrid(
            kspace, mask, pd,
            torch.ones(pd.shape[0], device=device),
        )
        direct_reference = center_crop(
            direct_reference, target.shape[-2], target.shape[-1]
        )
        second_direct_l1 = l1_per_sample(direct_reference, target).detach()
    del direct_reference
    prediction, aux = hybrid(
        paired_kspace, paired_mask, paired_pd, availability, return_aux=True
    )
    prediction = center_crop(
        prediction, paired_target.shape[-2], paired_target.shape[-1]
    )
    reconstruction_loss = F.l1_loss(prediction, paired_target)
    reliability_loss = F.binary_cross_entropy_with_logits(
        aux["q_logits"], torch.ones_like(aux["q_logits"])
    )
    second_clean_l1 = l1_per_sample(prediction[:pd.shape[0]], target)
    second_gain_loss = F.relu(
        second_clean_l1
        - (1.0 - args.residual_gain_margin_relative) * second_direct_l1
    ).mean()
    second_loss = (
        reconstruction_loss + 0.05 * reliability_loss
        + args.lambda_residual_gain * second_gain_loss
    )
    second_loss.backward()
    per_scale_residual_gradients = {
        f"scale_{index + 1}": float(
            controller.complement.residual_out.weight.grad.norm().item()
        )
        for index, controller in enumerate(hybrid.controllers)
    }
    gradients = [
        parameter.grad.detach().norm()
        for parameter in hybrid.parameters()
        if parameter.grad is not None
    ]
    gradient_norm = float(torch.stack(gradients).norm().item())
    optimizer.step()
    peak_memory = (
        torch.cuda.max_memory_allocated() / 1024 ** 3
        if device.type == "cuda" else 0.0
    )
    formal_shape = {
        "anatomical_batch": args.formal_batch_size,
        "forward_batch": int(paired_pd.shape[0]),
        "loss_finite": first_loss_finite and bool(torch.isfinite(second_loss).item()),
        "first_step_loss_finite": first_loss_finite,
        "second_step_loss_finite": bool(torch.isfinite(second_loss).item()),
        "first_step_gradient_norm": first_gradient_norm,
        "second_step_gradient_norm": gradient_norm,
        "first_step_residual_gain_loss": first_gain_value,
        "second_step_residual_gain_loss": float(second_gain_loss.detach().item()),
        "gain_margin_relative": args.residual_gain_margin_relative,
        "lambda_residual_gain": args.lambda_residual_gain,
        "gradient_norm": gradient_norm,
        "per_scale_residual_out_gradient_norm": per_scale_residual_gradients,
        "peak_gpu_memory_gb": peak_memory,
        "maximum_allowed_gpu_memory_gb": args.formal_max_gpu_memory_gb,
        "visible_gpu_total_gb": (
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            if device.type == "cuda" else 0.0
        ),
    }
    formal_shape["passed"] = (
        formal_shape["loss_finite"]
        and gradient_norm > 0.0
        and all(value > 0.0 for value in per_scale_residual_gradients.values())
        and peak_memory <= args.formal_max_gpu_memory_gb
    )

    residual_test = residual_gradient_test(args.chans * 2)
    result = {
        "status": "passed" if preset_audit["passed"] and all(
            item["passed"] for item in equivalence.values()
        ) and missing_max_difference == 0.0 and formal_shape["passed"] \
            and residual_test["passed"] else "failed",
        "protocol_version": "M2-PRNF-R8-v1.5.0-quality-gain-pilot-audited",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "strict_m2u_compatibility": equivalence,
        "v13_preset_compatibility": preset_audit,
        "m2u_reference_parameter_count": reference_parameter_count,
        "missing_invariance": {
            "maximum_output_difference": missing_max_difference,
            "passed": missing_max_difference == 0.0,
        },
        "zero_initialised_residual_gradient": residual_test,
        "formal_shape_preflight": formal_shape,
        "manifest_sha256": {
            "full_clean": sha256_file(Path(args.full_clean_manifest)),
            "robustness": sha256_file(Path(args.robustness_manifest)),
            "conditions": condition_hash,
        },
        "condition_manifest": str(Path(args.condition_manifest).resolve()),
        "metadata_sha256": metadata_hash,
        "code_hashes": locked_code_hashes(),
        "runtime_versions": runtime_versions(),
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
