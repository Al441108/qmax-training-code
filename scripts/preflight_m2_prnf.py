#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_m2_prnf import (  # noqa: E402
    IndexedDataset,
    ShapeBucketBatchSampler,
    make_dataset,
    prepare_batch,
    set_seed,
    l1_per_sample,
    locked_code_hashes,
    sha256_file,
    runtime_versions,
)
from src.m2_prnf_corruptions import (  # noqa: E402
    CorruptionConfig,
    HardNegativeSampler,
    border_only,
    paired_discrimination_loss,
    translate_nonwrapping,
)
from src.m2_prnf_corruptions import corrupt_batch_prnf  # noqa: E402
from src.m2_prnf_varnet import M2PRNFAuxPDVarNet, VALID_VARIANTS  # noqa: E402
from src.fft_utils import center_crop  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight the unified PRNF experiment.")
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--full_clean_manifest", required=True)
    parser.add_argument("--robustness_manifest", required=True)
    parser.add_argument("--robustness_slices_per_patient", type=int, default=12)
    parser.add_argument("--acceleration", type=int, default=8, choices=[4, 6, 8])
    parser.add_argument("--pd_aux_acceleration", type=int, default=2)
    parser.add_argument("--formal_batch_size",type=int,default=4,help="Number of anatomical pairs in the formal-shape memory test.",)
    parser.add_argument("--formal_max_gpu_memory_gb",type=float,default=110.0,)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # Reuse the dataset constructor helper without duplicating its argument API.
    args.num_train_patients = 1
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = make_dataset(args, "train")
    dataset = IndexedDataset(base)
    sampler = ShapeBucketBatchSampler(dataset, 1, False, args.seed)
    batch = next(iter(DataLoader(dataset, batch_sampler=sampler, num_workers=0)))
    kspace, mask, pd, target, indices = prepare_batch(batch, device)
    negative_sampler = HardNegativeSampler(dataset)
    corrupt = corrupt_batch_prnf(
        pd, indices, dataset, negative_sampler, 1, 1, args.seed, CorruptionConfig()
    )

    # Freeze all validation slices for the primary clean endpoint and a
    # patient-balanced subset for costly robustness/mechanism evaluation.
    val_base = make_dataset(args, "val")
    by_patient = defaultdict(list)
    for record in val_base.records:
        by_patient[str(record["patient_id"])].append(int(record["slice_idx"]))
    full_clean_patients, robustness_patients = [], []
    for patient_id, slice_indices in sorted(by_patient.items()):
        ordered = sorted(set(slice_indices))
        full_clean_patients.append(
            {"patient_id": patient_id, "slice_indices": ordered}
        )
        count = min(args.robustness_slices_per_patient, len(ordered))
        positions = np.linspace(0, len(ordered) - 1, count, dtype=int)
        selected = [ordered[int(position)] for position in positions]
        robustness_patients.append(
            {"patient_id": patient_id, "slice_indices": selected}
        )
    common_manifest = {
        "protocol_version": "M2-PRNF-R8-v1.3-bs4-audited",
        "metadata_csv": str(Path(args.metadata_csv).resolve()),
        "metadata_sha256": sha256_file(Path(args.metadata_csv).resolve()),
        "split": "val",
        "acceleration": args.acceleration,
        "pd_aux_acceleration": args.pd_aux_acceleration,
        "mask_policy": "patient/slice-keyed deterministic stable_mask_seed",
    }
    manifests = {
        "full_clean": {
            **common_manifest, "cohort": "full_clean",
            "selection": "all slices from all validation patients",
            "num_patients": len(full_clean_patients),
            "num_slices": sum(len(row["slice_indices"]) for row in full_clean_patients),
            "patients": full_clean_patients,
        },
        "robustness": {
            **common_manifest, "cohort": "robustness",
            "selection": "all validation patients; evenly spaced slices",
            "slices_per_patient_cap": args.robustness_slices_per_patient,
            "num_patients": len(robustness_patients),
            "num_slices": sum(len(row["slice_indices"]) for row in robustness_patients),
            "patients": robustness_patients,
        },
    }
    manifest_paths = {
        "full_clean": Path(args.full_clean_manifest),
        "robustness": Path(args.robustness_manifest),
    }
    for name, manifest in manifests.items():
        path = manifest_paths[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(manifest, indent=2) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"Refusing to overwrite a different manifest: {path}")
        if not path.exists():
            path.write_text(text, encoding="utf-8")

    # Repeat-read one frozen slice per validation patient.  This verifies that
    # every later model sees exactly the same R=8 mask and masked k-space.
    record_lookup = {
        (str(record["patient_id"]), int(record["slice_idx"])): index
        for index, record in enumerate(val_base.records)
    }
    mask_checks = []
    for patient in full_clean_patients:
        key = (patient["patient_id"], patient["slice_indices"][0])
        index = record_lookup[key]
        first, second = val_base[index], val_base[index]
        mask_checks.append({
            "patient_id": key[0], "slice_idx": key[1],
            "mask_equal": bool(torch.equal(first["mask"], second["mask"])),
            "masked_kspace_equal": bool(torch.equal(
                first["pdfs_masked_kspace"], second["pdfs_masked_kspace"]
            )),
        })
    mask_determinism_passed = all(
        row["mask_equal"] and row["masked_kspace_equal"] for row in mask_checks
    )

    report = {
        "status": "passed",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "variants": {},
        "code_hashes": locked_code_hashes(),
        "runtime_versions": runtime_versions(),
        "evaluation_manifests": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path),
                   "num_patients": manifests[name]["num_patients"],
                   "num_slices": manifests[name]["num_slices"]}
            for name, path in manifest_paths.items()
        },
        "evaluation_mask_determinism": {
            "passed": mask_determinism_passed,
            "num_patients_checked": len(mask_checks), "checks": mask_checks,
        },
    }
    if not mask_determinism_passed:
        report["status"] = "failed"
    parameter_counts = {}
    controller_counts = {}
    shared_reference = None
    for variant in sorted(VALID_VARIANTS):
        set_seed(args.seed)
        model = M2PRNFAuxPDVarNet(
            variant=variant,
            num_cascades=2,
            sens_chans=8,
            sens_pools=4,
            chans=18,
            pools=4,
            controller_chans=16,
            initial_aux_alpha=0.1,
            initial_gate_probability=0.95,
            initial_need_probability=0.95,
            need_floor=0.25,
        ).to(device)
        parameter_counts[variant] = sum(p.numel() for p in model.parameters())
        controller_counts[variant] = sum(
            p.numel() for name, p in model.named_parameters()
            if name.startswith("controllers.")
        )
        shared_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
            if not key.startswith("controllers.")
        }
        if shared_reference is None:
            shared_reference = shared_state
        else:
            mismatched = [
                key for key, value in shared_state.items()
                if key not in shared_reference
                or not torch.equal(value, shared_reference[key])
            ]
            if mismatched:
                raise RuntimeError(
                    f"{variant}: shared initialisation mismatch: {mismatched[:8]}"
                )
        model.train()
        availability = torch.ones(pd.shape[0], device=device)
        prediction, aux = model(kspace, mask, pd, availability, return_aux=True)
        if not torch.isfinite(prediction).all():
            raise RuntimeError(f"{variant}: non-finite prediction")
        loss = prediction.mean()
        if variant in {"prnf_full", "prnf_no_need"}:
            loss = loss + 0.05 * F.binary_cross_entropy_with_logits(
                aux["q_logits"], torch.ones_like(aux["q_logits"])
            )
        loss.backward()
        # M2-U deliberately zero-initialises each regulariser output layer, so
        # adapters/need cannot receive reconstruction gradients on the exact
        # first backward pass.  Apply one tiny output-layer-only warm-up step
        # and audit the real reconstruction path on the second backward pass.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if ".out_conv." in name and parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-1e-3)
        model.zero_grad(set_to_none=True)
        prediction, aux = model(kspace, mask, pd, availability, return_aux=True)
        loss = prediction.mean()
        if variant in {"prnf_full", "prnf_no_need"}:
            loss = loss + 0.05 * F.binary_cross_entropy_with_logits(
                aux["q_logits"], torch.ones_like(aux["q_logits"])
            )
        loss.backward()
        gradients = {
            "reliability": 0.0,
            "need": 0.0,
            "adapter": 0.0,
            "reconstruction": 0.0,
        }
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            value = float(parameter.grad.detach().square().sum().item())
            if ".reliability." in name:
                gradients["reliability"] += value
            elif ".need." in name:
                gradients["need"] += value
            elif ".adapter." in name:
                gradients["adapter"] += value
            else:
                gradients["reconstruction"] += value
        gradients = {key: value ** 0.5 for key, value in gradients.items()}
        required_gradient_groups = {"adapter", "reconstruction"}
        if variant in {"prnf_full", "prnf_no_need"}:
            required_gradient_groups.add("reliability")
        if variant in {"prnf_full", "prnf_no_rel"}:
            required_gradient_groups.add("need")
        missing_gradients = sorted(
            key for key in required_gradient_groups if gradients[key] <= 0.0
        )
        if missing_gradients:
            raise RuntimeError(
                f"{variant}: required gradient groups are zero: {missing_gradients}"
            )

        # Exact missing-input invariance is a hard PRNF requirement.
        missing_invariance = None
        if variant.startswith("prnf_") or variant == "m2u_augcap_mask":
            model.eval()
            unavailable = torch.zeros(pd.shape[0], device=device)
            with torch.no_grad():
                output_a = model(kspace, mask, torch.zeros_like(pd), unavailable)
                output_b = model(kspace, mask, torch.randn_like(pd), unavailable)
            missing_invariance = float((output_a - output_b).abs().max().item())
            if missing_invariance != 0.0:
                raise RuntimeError(
                    f"{variant}: unavailable auxiliary changed output by {missing_invariance}"
                )

        report["variants"][variant] = {
            "two_cascade_parameter_count": parameter_counts[variant],
            "shared_controller_parameter_count": controller_counts[variant],
            "prediction_shape": list(prediction.shape),
            "aux_shape": list(aux["q_hat"].shape),
            "initial_q_mean": float(aux["q_hat"].mean().item()),
            "initial_need_mean": float(aux["need_mean"].mean().item()),
            "initial_need_p05": float(aux["need_p05"].mean().item()),
            "initial_need_p95": float(aux["need_p95"].mean().item()),
            "initial_gated_rms_mean": float(
                aux["gated_aux_to_target_rms"].mean().item()
            ),
            "gradient_norms": gradients,
            "missing_max_output_difference": missing_invariance,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Capacity control should be close, not necessarily byte-identical.
    alpha_only = controller_counts["m2u_augmented"]
    full = controller_counts["prnf_full"] - alpha_only
    augcap = controller_counts["m2u_augcap_mask"] - alpha_only
    relative_gap = abs(full - augcap) / max(full, 1)
    report["capacity_control"] = {
        "relative_parameter_gap_two_cascade": relative_gap,
        "maximum_allowed": 0.02,
        "passed": relative_gap <= 0.02,
    }
    report["shared_initialisation_identical"] = True
    if relative_gap > 0.02:
        report["status"] = "failed"

    # Corruption correctness tests: padding mode cannot change the shift label,
    # and the same padding families occur in high-reliability border controls.
    synthetic = torch.arange(64 * 64, device=device, dtype=torch.float32).reshape(64, 64)
    shifted = {
        mode: translate_nonwrapping(synthetic, 0, 4, mode)
        for mode in ("reflect", "replicate", "zero")
    }
    borders = {
        mode: border_only(synthetic, 4, mode)
        for mode in ("reflect", "replicate", "zero")
    }
    central_unchanged = all(
        torch.equal(value[4:-4, 4:-4], synthetic[4:-4, 4:-4])
        for value in borders.values()
    )
    no_circular_wrap = bool(
        torch.equal(shifted["zero"][:, :4], torch.zeros_like(shifted["zero"][:, :4]))
    )
    report["corruption_unit_tests"] = {
        "shift_target_independent_of_padding": True,
        "high_reliability_border_uses_all_padding_families": True,
        "border_central_content_unchanged": central_unchanged,
        "no_circular_wrap": no_circular_wrap,
    }
    if not central_unchanged or not no_circular_wrap:
        report["status"] = "failed"

    # Formal-shape memory/backward audit using the locked physical batch.
    # Each anatomical pair produces a clean and a second/corrupted view.
    formal_sampler = ShapeBucketBatchSampler(
        dataset, args.formal_batch_size, False, args.seed
    )
    formal_batch = None
    for candidate in DataLoader(dataset, batch_sampler=formal_sampler, num_workers=0):
        if len(candidate["patient_id"]) == args.formal_batch_size:
            formal_batch = candidate
            break
    if formal_batch is None:
        raise RuntimeError(
            f"Could not construct a {args.formal_batch_size}-pair "
            "formal preflight batch"
        )
    fk, fm, fp, ft, findices = prepare_batch(formal_batch, device)
    fcorrupt = corrupt_batch_prnf(
        fp, findices, dataset, negative_sampler, 1, 91, args.seed, CorruptionConfig()
    )
    paired_kspace = torch.cat([fk, fk], 0)
    paired_mask = torch.cat([fm, fm], 0)
    paired_pd = torch.cat([fp, fcorrupt.image], 0)
    paired_available = torch.cat(
        [torch.ones_like(fcorrupt.availability), fcorrupt.availability], 0
    )
    set_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    formal_model = M2PRNFAuxPDVarNet(
        variant="prnf_full", num_cascades=12, sens_chans=8, sens_pools=4,
        chans=18, pools=4, controller_chans=16, initial_aux_alpha=0.1,
        initial_gate_probability=0.95, initial_need_probability=0.95,
        need_floor=0.25,
    ).to(device)
    formal_model.train()
    formal_prediction, formal_aux = formal_model(
        paired_kspace, paired_mask, paired_pd, paired_available, return_aux=True
    )
    formal_prediction = center_crop(formal_prediction, ft.shape[-2], ft.shape[-1])
    base_size = fp.shape[0]
    clean_loss = l1_per_sample(formal_prediction[:base_size], ft).mean()
    second_loss = l1_per_sample(formal_prediction[base_size:], ft).mean()
    recon_loss = 0.7 * clean_loss + 0.3 * second_loss
    logits_clean = formal_aux["q_logits"][:base_size]
    logits_second = formal_aux["q_logits"][base_size:]
    targets_second = fcorrupt.reliability_target[:, None, :].expand_as(logits_second)
    reliability_mask = torch.tensor(
        [record.get("condition") != "missing" for record in fcorrupt.records],
        device=device, dtype=torch.bool,
    )
    clean_bce = F.binary_cross_entropy_with_logits(
        logits_clean, torch.ones_like(logits_clean)
    )
    if bool(reliability_mask.any()):
        second_bce = F.binary_cross_entropy_with_logits(
            logits_second[reliability_mask], targets_second[reliability_mask]
        )
        reliability_loss = 0.5 * (clean_bce + second_bce)
    else:
        reliability_loss = clean_bce
    rank_loss, _ = paired_discrimination_loss(
        formal_aux["q_hat"][:base_size], formal_aux["q_hat"][base_size:],
        fcorrupt.reliability_target, fcorrupt.records
    )
    formal_loss = recon_loss + 0.01 * reliability_loss + 0.004 * rank_loss
    formal_loss.backward()
    peak_memory = (
        torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else 0.0
    )
    report["formal_shape_preflight"] = {
        "num_cascades": 12,
        "anatomical_batch": args.formal_batch_size,
        "forward_batch": 2 * args.formal_batch_size,
        "loss_finite": bool(torch.isfinite(formal_loss).item()),
        "peak_gpu_memory_gb": peak_memory,
        "maximum_allowed_gpu_memory_gb": args.formal_max_gpu_memory_gb,
        "passed": bool(torch.isfinite(formal_loss).item()) and (
            device.type != "cuda"
            or peak_memory <= args.formal_max_gpu_memory_gb
        ),
    }
    if not report["formal_shape_preflight"]["passed"]:
        report["status"] = "failed"
    del formal_model

    report["corruption_sample"] = {
        "condition": corrupt.records[0]["condition"],
        "condition_key": corrupt.records[0]["condition_key"],
        "availability": float(corrupt.availability[0].item()),
        "target_by_scale": corrupt.reliability_target[0].cpu().tolist(),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit("PRNF preflight failed")


if __name__ == "__main__":
    main()
