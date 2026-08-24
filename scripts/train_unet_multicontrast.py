import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import csv
import time
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim

from src.dataset_multicontrast import FastMRIMultiContrastDataset
from src.model_unet import SmallUNet


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def psnr_np(pred: np.ndarray, gt: np.ndarray, eps=1e-8) -> float:
    mse = np.mean((pred - gt) ** 2)
    if mse < eps:
        return 99.0
    return 20 * np.log10(1.0 / np.sqrt(mse))


def norm01_np(img: np.ndarray, eps=1e-8) -> np.ndarray:
    mmin = float(np.min(img))
    mmax = float(np.max(img))
    return (img - mmin) / (mmax - mmin + eps)


def ssim_np(pred: np.ndarray, gt: np.ndarray) -> float:
    return ssim(pred, gt, data_range=1.0)


@torch.no_grad()
def evaluate_metrics(model, dl, criterion, device, max_batches=20):
    model.eval()

    val_loss = 0.0

    psnr_x0_pd, psnr_pred_pd = [], []
    ssim_x0_pd, ssim_pred_pd = [], []

    psnr_x0_pdfs, psnr_pred_pdfs = [], []
    ssim_x0_pdfs, ssim_pred_pdfs = [], []

    for bi, batch in enumerate(dl):
        if max_batches is not None and bi >= max_batches:
            break

        x = batch["input"].to(device)    # [B, 2, H, W]
        y = batch["target"].to(device)   # [B, 2, H, W]

        pred = model(x)
        loss = criterion(pred, y)
        val_loss += loss.item()

        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()

        B = x_np.shape[0]
        for b in range(B):
            # channel 0 = PD, channel 1 = PD-FS
            x0_pd = norm01_np(x_np[b, 0])
            pr_pd = norm01_np(pred_np[b, 0])
            gt_pd = norm01_np(y_np[b, 0])

            x0_pdfs = norm01_np(x_np[b, 1])
            pr_pdfs = norm01_np(pred_np[b, 1])
            gt_pdfs = norm01_np(y_np[b, 1])

            psnr_x0_pd.append(psnr_np(x0_pd, gt_pd))
            psnr_pred_pd.append(psnr_np(pr_pd, gt_pd))
            ssim_x0_pd.append(ssim_np(x0_pd, gt_pd))
            ssim_pred_pd.append(ssim_np(pr_pd, gt_pd))

            psnr_x0_pdfs.append(psnr_np(x0_pdfs, gt_pdfs))
            psnr_pred_pdfs.append(psnr_np(pr_pdfs, gt_pdfs))
            ssim_x0_pdfs.append(ssim_np(x0_pdfs, gt_pdfs))
            ssim_pred_pdfs.append(ssim_np(pr_pdfs, gt_pdfs))

    mean_val_loss = val_loss / max(1, min(len(dl), max_batches if max_batches is not None else len(dl)))

    out = {
        "val_loss": mean_val_loss,

        "val_psnr_x0_pd": float(np.mean(psnr_x0_pd)) if psnr_x0_pd else 0.0,
        "val_psnr_pred_pd": float(np.mean(psnr_pred_pd)) if psnr_pred_pd else 0.0,
        "val_ssim_x0_pd": float(np.mean(ssim_x0_pd)) if ssim_x0_pd else 0.0,
        "val_ssim_pred_pd": float(np.mean(ssim_pred_pd)) if ssim_pred_pd else 0.0,

        "val_psnr_x0_pdfs": float(np.mean(psnr_x0_pdfs)) if psnr_x0_pdfs else 0.0,
        "val_psnr_pred_pdfs": float(np.mean(psnr_pred_pdfs)) if psnr_pred_pdfs else 0.0,
        "val_ssim_x0_pdfs": float(np.mean(ssim_x0_pdfs)) if ssim_x0_pdfs else 0.0,
        "val_ssim_pred_pdfs": float(np.mean(ssim_pred_pdfs)) if ssim_pred_pdfs else 0.0,
    }

    model.train()
    return out


def save_ckpt(path, model, optim, epoch, best_score):
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
        },
        path,
    )


def main():
    set_seed(42)

    pairs_json = "/rds/general/user/ah725/home/fastmri_pipeline/metadata/all_pairs_simple.json"
    save_dir = Path("/rds/general/user/ah725/home/fastmri_pipeline/run/unet_multicontrast")
    save_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 8
    num_workers = 4
    epochs = 30
    lr = 1e-3
    val_max_batches = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU name:", torch.cuda.get_device_name(0))

    train_ds = FastMRIMultiContrastDataset(
        pairs_json=pairs_json,
        split="singlecoil_train",
        center_fraction=0.08,
        acceleration=4,
        sample_rate=1.0,
        use_seed=True,
    )

    val_ds = FastMRIMultiContrastDataset(
        pairs_json=pairs_json,
        split="singlecoil_val",
        center_fraction=0.08,
        acceleration=4,
        sample_rate=1.0,
        use_seed=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print("Train size:", len(train_ds))
    print("Val size:", len(val_ds))
    print("Train loader batches:", len(train_loader))
    print("Val loader batches:", len(val_loader))

    model = SmallUNet(in_ch=2, out_ch=2, base_ch=32).to(device)
    criterion = nn.L1Loss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.5)

    log_path = save_dir / "log.csv"
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch",
            "train_loss",
            "val_loss",

            "val_psnr_x0_pd",
            "val_psnr_pred_pd",
            "val_ssim_x0_pd",
            "val_ssim_pred_pd",

            "val_psnr_x0_pdfs",
            "val_psnr_pred_pdfs",
            "val_ssim_x0_pdfs",
            "val_ssim_pred_pdfs",

            "improve_psnr_pd",
            "improve_ssim_pd",
            "improve_psnr_pdfs",
            "improve_ssim_pdfs",
            "mean_improve_psnr",
            "mean_improve_ssim",
        ])

    best_score = -1e9

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            x = batch["input"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)

            pred = model(x)
            loss = criterion(pred, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            running_loss += loss.item()

            if (i + 1) % 20 == 0 or i == 0:
                print(f"Epoch {epoch} | Step {i+1}/{len(train_loader)} | Loss {loss.item():.4f}")

        avg_train_loss = running_loss / max(1, len(train_loader))
        metrics = evaluate_metrics(model, val_loader, criterion, device, max_batches=val_max_batches)

        improve_psnr_pd = metrics["val_psnr_pred_pd"] - metrics["val_psnr_x0_pd"]
        improve_ssim_pd = metrics["val_ssim_pred_pd"] - metrics["val_ssim_x0_pd"]

        improve_psnr_pdfs = metrics["val_psnr_pred_pdfs"] - metrics["val_psnr_x0_pdfs"]
        improve_ssim_pdfs = metrics["val_ssim_pred_pdfs"] - metrics["val_ssim_x0_pdfs"]

        mean_improve_psnr = 0.5 * (improve_psnr_pd + improve_psnr_pdfs)
        mean_improve_ssim = 0.5 * (improve_ssim_pd + improve_ssim_pdfs)

        dt = time.time() - t0

        print(
            f"Epoch {epoch} done | "
            f"train_loss={avg_train_loss:.4f} | val_loss={metrics['val_loss']:.4f} | "
            f"PD PSNR: x0={metrics['val_psnr_x0_pd']:.2f}, pred={metrics['val_psnr_pred_pd']:.2f} ({improve_psnr_pd:+.2f}) | "
            f"PD-FS PSNR: x0={metrics['val_psnr_x0_pdfs']:.2f}, pred={metrics['val_psnr_pred_pdfs']:.2f} ({improve_psnr_pdfs:+.2f}) | "
            f"time={dt:.1f}s"
        )

        save_ckpt(save_dir / "model_last.pt", model, optim, epoch, best_score)

        if mean_improve_psnr > best_score:
            best_score = mean_improve_psnr
            save_ckpt(save_dir / "model_best.pt", model, optim, epoch, best_score)
            print("Saved BEST:", save_dir / "model_best.pt")

        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                epoch,
                avg_train_loss,
                metrics["val_loss"],

                metrics["val_psnr_x0_pd"],
                metrics["val_psnr_pred_pd"],
                metrics["val_ssim_x0_pd"],
                metrics["val_ssim_pred_pd"],

                metrics["val_psnr_x0_pdfs"],
                metrics["val_psnr_pred_pdfs"],
                metrics["val_ssim_x0_pdfs"],
                metrics["val_ssim_pred_pdfs"],

                improve_psnr_pd,
                improve_ssim_pd,
                improve_psnr_pdfs,
                improve_ssim_pdfs,
                mean_improve_psnr,
                mean_improve_ssim,
            ])

        scheduler.step()

    print("Training finished.")
    print("Log saved to:", log_path)
    print("Best checkpoint:", save_dir / "model_best.pt")


if __name__ == "__main__":
    main()