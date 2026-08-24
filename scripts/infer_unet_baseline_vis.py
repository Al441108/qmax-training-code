import sys
import csv
import math
import random
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.metrics import structural_similarity as ssim

# =========================
# Project root / imports
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.dataset_fastmri import FastMRISinglecoilDataset
from src.model_unet import SmallUNet


# =========================
# Utils
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def norm01_np(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    vmin = float(np.min(img))
    vmax = float(np.max(img))
    return (img - vmin) / (vmax - vmin + eps)


def psnr_np(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    mse = np.mean((pred - gt) ** 2)
    if mse < eps:
        return 99.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def ssim_np(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(ssim(pred, gt, data_range=1.0))


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def tensor_to_int(x):
    if isinstance(x, torch.Tensor):
        return int(x.item())
    if isinstance(x, (list, tuple)):
        return int(x[0])
    return int(x)


def tensor_to_str(x):
    if isinstance(x, (list, tuple)):
        return str(x[0])
    return str(x)


def load_model(ckpt_path: Path, device: torch.device) -> SmallUNet:
    model = SmallUNet(in_ch=1, out_ch=1, base_ch=32).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if "model" not in ckpt:
        raise KeyError(f"'model' key not found in checkpoint: {ckpt_path}")

    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def prepare_images_for_metrics(inp, pred, tgt):
    """
    Keep metric logic consistent with your training code:
    normalize each image independently to [0, 1].
    """
    inp_n = norm01_np(inp)
    pred_n = norm01_np(pred)
    tgt_n = norm01_np(tgt)
    err = np.abs(pred_n - tgt_n)
    return inp_n, pred_n, tgt_n, err


def save_four_panel(
    inp: np.ndarray,
    pred: np.ndarray,
    tgt: np.ndarray,
    save_path: Path,
    title: str = "",
    show_colorbar: bool = True,
):
    inp_n, pred_n, tgt_n, err = prepare_images_for_metrics(inp, pred, tgt)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    axes[0].imshow(inp_n, cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(pred_n, cmap="gray")
    axes[1].set_title("Pred")
    axes[1].axis("off")

    axes[2].imshow(tgt_n, cmap="gray")
    axes[2].set_title("Target")
    axes[2].axis("off")

    im = axes[3].imshow(err, cmap="hot")
    axes[3].set_title("Error Map")
    axes[3].axis("off")

    if show_colorbar:
        fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_contact_sheet(records, save_path: Path, max_items: int = 8):
    if len(records) == 0:
        return

    records = records[:max_items]
    n = len(records)

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, rec in enumerate(records):
        inp_n, pred_n, tgt_n, err = prepare_images_for_metrics(
            rec["input_img"], rec["pred_img"], rec["target_img"]
        )

        axes[i, 0].imshow(inp_n, cmap="gray")
        axes[i, 0].set_title("Input")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(pred_n, cmap="gray")
        axes[i, 1].set_title("Pred")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(tgt_n, cmap="gray")
        axes[i, 2].set_title("Target")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(err, cmap="hot")
        axes[i, 3].set_title("Error")
        axes[i, 3].axis("off")

        row_label = (
            f'idx={rec["dataset_idx"]} | '
            f'PSNR={rec["pred_psnr"]:.2f} dB | SSIM={rec["pred_ssim"]:.4f}'
        )
        axes[i, 0].set_ylabel(row_label, fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def run_inference(args):
    set_seed(args.seed)

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[INFO] Using device: {device}")

    root = Path(args.root)
    ckpt_path = PROJECT_ROOT / "run" / "unet_baseline" / args.ckpt_name
    out_dir = PROJECT_ROOT / "run" / "unet_baseline" / args.output_subdir

    ensure_dir(out_dir)

    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[INFO] dataset root = {root}")
    print(f"[INFO] ckpt_path    = {ckpt_path}")
    print(f"[INFO] out_dir      = {out_dir}")

    dataset = FastMRISinglecoilDataset(
        root=str(root),
        split=args.split,
        center_fraction=args.center_fraction,
        acceleration=args.acceleration,
        sample_rate=args.sample_rate,
    )

    print(f"[INFO] Dataset split={args.split}, size={len(dataset)}")

    model = load_model(ckpt_path, device)

    n_available = len(dataset)
    n_pick = min(args.num_examples, n_available)

    rng = np.random.default_rng(args.seed)
    chosen_indices = rng.choice(n_available, size=n_pick, replace=False).tolist()

    print(f"[INFO] Chosen indices: {chosen_indices}")

    summary_rows = []
    records = []

    for vis_id, idx in enumerate(chosen_indices):
        sample = dataset[idx]

        x = sample["input"].unsqueeze(0).to(device)   # [1,1,H,W]
        y = sample["target"].unsqueeze(0).to(device)  # [1,1,H,W]

        pred = model(x)

        x_np = x.squeeze().detach().cpu().numpy()       # [H,W]
        y_np = y.squeeze().detach().cpu().numpy()
        pred_np = pred.squeeze().detach().cpu().numpy()

        # Try to read optional metadata if dataset provides them
        file_name = tensor_to_str(sample["file_name"]) if "file_name" in sample else "unknown_file"
        slice_id = tensor_to_int(sample["slice_idx"]) if "slice_idx" in sample else idx

        inp_n, pred_n, tgt_n, _ = prepare_images_for_metrics(x_np, pred_np, y_np)

        x0_psnr = psnr_np(inp_n, tgt_n)
        pred_psnr = psnr_np(pred_n, tgt_n)
        x0_ssim = ssim_np(inp_n, tgt_n)
        pred_ssim = ssim_np(pred_n, tgt_n)

        improve_psnr = pred_psnr - x0_psnr
        improve_ssim = pred_ssim - x0_ssim

        file_short = Path(file_name).stem[:18]

        filename = f"{vis_id:02d}_idx{idx}_slice{slice_id}_{file_short}.png"
        save_path = out_dir / filename

        title = (
            f"single-contrast | idx={idx} | file={file_short} | slice={slice_id}\n"
            f"x0: PSNR={x0_psnr:.2f} dB, SSIM={x0_ssim:.4f} | "
            f"pred: PSNR={pred_psnr:.2f} dB, SSIM={pred_ssim:.4f}"
        )

        save_four_panel(
            inp=x_np,
            pred=pred_np,
            tgt=y_np,
            save_path=save_path,
            title=title,
            show_colorbar=not args.no_colorbar,
        )

        row = {
            "dataset_idx": idx,
            "vis_id": vis_id,
            "file_name": file_name,
            "slice_id": slice_id,
            "x0_psnr": x0_psnr,
            "pred_psnr": pred_psnr,
            "improve_psnr": improve_psnr,
            "x0_ssim": x0_ssim,
            "pred_ssim": pred_ssim,
            "improve_ssim": improve_ssim,
            "saved_path": str(save_path),
        }
        summary_rows.append(row)

        rec = {
            "dataset_idx": idx,
            "input_img": x_np,
            "pred_img": pred_np,
            "target_img": y_np,
            "pred_psnr": pred_psnr,
            "pred_ssim": pred_ssim,
        }
        records.append(rec)

        print(
            f"[SAVED] {save_path.name} | idx={idx} | "
            f"PSNR {x0_psnr:.2f}->{pred_psnr:.2f} | "
            f"SSIM {x0_ssim:.4f}->{pred_ssim:.4f}"
        )

    summary_csv = out_dir / "summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_idx",
                "vis_id",
                "file_name",
                "slice_id",
                "x0_psnr",
                "pred_psnr",
                "improve_psnr",
                "x0_ssim",
                "pred_ssim",
                "improve_ssim",
                "saved_path",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[INFO] Summary saved to: {summary_csv}")

    if not args.no_contact_sheet:
        make_contact_sheet(
            records,
            save_path=out_dir / "contact_sheet.png",
            max_items=min(len(records), args.num_examples),
        )
        print("[INFO] Contact sheet saved.")

    if len(summary_rows) > 0:
        mean_improve_psnr = np.mean([r["improve_psnr"] for r in summary_rows])
        mean_improve_ssim = np.mean([r["improve_ssim"] for r in summary_rows])
        print(f"[INFO] Mean improvement over selected examples:")
        print(f"       PSNR: {mean_improve_psnr:+.2f} dB")
        print(f"       SSIM: {mean_improve_ssim:+.4f}")

    print("[INFO] Done.")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Visualize single-contrast fastMRI U-Net reconstructions on validation samples."
    )

    parser.add_argument("--root", type=str, default="/rds/general/user/ah725/home/fastmri",
                        help="fastMRI root directory")
    parser.add_argument("--split", type=str, default="val",
                        help="Dataset split to visualize. Default: val")
    parser.add_argument("--num_examples", type=int, default=6,
                        help="Number of random examples to visualize.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling.")
    parser.add_argument("--sample_rate", type=float, default=0.02,
                        help="Dataset sample rate passed to dataset. Match training checkpoint by default.")
    parser.add_argument("--center_fraction", type=float, default=0.08,
                        help="Mask center fraction.")
    parser.add_argument("--acceleration", type=int, default=4,
                        help="Mask acceleration factor.")
    parser.add_argument("--ckpt_name", type=str, default="model_best.pt",
                        help="Checkpoint name under run/unet_baseline/")
    parser.add_argument("--output_subdir", type=str, default="val_vis",
                        help="Output folder under run/unet_baseline/")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device, e.g. cpu or cuda")
    parser.add_argument("--no_colorbar", action="store_true",
                        help="Disable colorbar on error map.")
    parser.add_argument("--no_contact_sheet", action="store_true",
                        help="Do not save contact sheet overview image.")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()