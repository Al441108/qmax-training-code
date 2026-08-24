import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import csv
import time
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim

from src.dataset_fastmri import FastMRISinglecoilDataset
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
def evaluate_metrics(model, dl, criterion, device, max_batches=50):
    model.eval()

    val_loss = 0.0
    psnr_x0_list = []
    psnr_pred_list = []
    ssim_x0_list = []
    ssim_pred_list = []

    for bi, batch in enumerate(dl):
        if max_batches is not None and bi >= max_batches:
            break

        x = batch["input"].to(device)
        y = batch["target"].to(device)

        pred = model(x)
        loss = criterion(pred, y)
        val_loss += loss.item()

        x_np = x.squeeze(1).detach().cpu().numpy()
        pred_np = pred.squeeze(1).detach().cpu().numpy()
        y_np = y.squeeze(1).detach().cpu().numpy()

        B = x_np.shape[0]
        for b in range(B):
            x0_img = norm01_np(x_np[b])
            pr_img = norm01_np(pred_np[b])
            gt_img = norm01_np(y_np[b])

            psnr_x0_list.append(psnr_np(x0_img, gt_img))
            psnr_pred_list.append(psnr_np(pr_img, gt_img))
            ssim_x0_list.append(ssim_np(x0_img, gt_img))
            ssim_pred_list.append(ssim_np(pr_img, gt_img))

    mean_val_loss = val_loss / max(
        1, min(len(dl), max_batches if max_batches is not None else len(dl))
    )
    mean_psnr_x0 = float(np.mean(psnr_x0_list)) if psnr_x0_list else 0.0
    mean_psnr_pred = float(np.mean(psnr_pred_list)) if psnr_pred_list else 0.0
    mean_ssim_x0 = float(np.mean(ssim_x0_list)) if ssim_x0_list else 0.0
    mean_ssim_pred = float(np.mean(ssim_pred_list)) if ssim_pred_list else 0.0

    model.train()
    return mean_val_loss, mean_psnr_x0, mean_psnr_pred, mean_ssim_x0, mean_ssim_pred


def save_ckpt(path, model, optim, epoch, best_val_psnr):
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "epoch": epoch,
            "best_val_psnr": best_val_psnr,
        },
        path,
    )


def main():
    set_seed(42)

    root = "/rds/general/user/ah725/home/fastmri"
    save_dir = Path("/rds/general/user/ah725/home/fastmri_pipeline/run/unet_baseline")
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

    train_ds = FastMRISinglecoilDataset(
        root=root,
        split="train",
        center_fraction=0.08,
        acceleration=4,
        sample_rate=1.0,
    )

    val_ds = FastMRISinglecoilDataset(
        root=root,
        split="val",
        center_fraction=0.08,
        acceleration=4,
        sample_rate=1.0,
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

    model = SmallUNet(in_ch=1, out_ch=1, base_ch=32).to(device)
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
            "val_psnr_x0",
            "val_psnr_pred",
            "val_ssim_x0",
            "val_ssim_pred",
            "improve_psnr",
            "improve_ssim",
        ])

    best_val_psnr = -1e9
    global_step = 0

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
            global_step += 1
            if (i + 1) % 20 == 0 or i == 0:
                print(f"Epoch {epoch} | Step {i+1}/{len(train_loader)} | Loss {loss.item():.4f}")

        avg_train_loss = running_loss / max(1, len(train_loader))

        val_loss, val_psnr_x0, val_psnr_pred, val_ssim_x0, val_ssim_pred = evaluate_metrics(
            model, val_loader, criterion, device, max_batches=val_max_batches
        )

        improve_psnr = val_psnr_pred - val_psnr_x0
        improve_ssim = val_ssim_pred - val_ssim_x0

        dt = time.time() - t0
        print(
            f"Epoch {epoch} done | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"PSNR: x0={val_psnr_x0:.2f}, pred={val_psnr_pred:.2f} (+{improve_psnr:.2f}) | "
            f"SSIM: x0={val_ssim_x0:.4f}, pred={val_ssim_pred:.4f} (+{improve_ssim:.4f}) | "
            f"time={dt:.1f}s"
        )

        save_ckpt(save_dir / "model_last.pt", model, optim, epoch, best_val_psnr)

        if val_psnr_pred > best_val_psnr:
            best_val_psnr = val_psnr_pred
            save_ckpt(save_dir / "model_best.pt", model, optim, epoch, best_val_psnr)
            print("Saved BEST:", save_dir / "model_best.pt")

        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                epoch,
                avg_train_loss,
                val_loss,
                val_psnr_x0,
                val_psnr_pred,
                val_ssim_x0,
                val_ssim_pred,
                improve_psnr,
                improve_ssim,
            ])

        scheduler.step()

    print("Training finished.")
    print("Log saved to:", log_path)
    print("Best checkpoint:", save_dir / "model_best.pt")


if __name__ == "__main__":
    main()
