#!/usr/bin/env python3
"""
=============================================================================
  MGMT Promoter Methylation 3D-CNN Classifier
  ─────────────────────────────────────────────
  Implements the architecture from:
    "Deep learning classification of MGMT status of glioblastomas using
     multiparametric MRI with a novel domain knowledge augmented mask
     fusion approach"  (Koska et al., 2025)

  Adapted for UCSF-PDGM cohort with 128³ tumour-centred crops.
  Input tensor shape: (Batch, 2, 128, 128, 128)
    - Channel 0:  (T1 + FLAIR) / 2    — pre-contrast structural fusion
    - Channel 1:  (T1CE + T2)  / 2    — contrast-enhanced + fluid fusion
=============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# ────────────────────────── logging ──────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("MGMT-3DCNN")


# ============================================================================
#  1.  DATASET
# ============================================================================
class MGMTCropDataset(Dataset):
    """
    Loads per-patient 128³ crops for four MRI sequences and fuses them
    into a 2-channel volume following the domain-knowledge augmented
    mask-fusion strategy.

    Channel mapping:
        ch0 = (T1 + FLAIR) / 2
        ch1 = (T1CE + T2)  / 2

    Parameters
    ----------
    records : list[dict]
        Each dict must contain keys ``patient_id_norm`` and ``label``.
    crop_dir : str | Path
        Root directory containing patient sub-folders.
    """

    SEQUENCES = ("T1_crop.npy", "T2_crop.npy", "FLAIR_crop.npy", "T1c_crop.npy")

    def __init__(self, records: List[Dict], crop_dir: str | Path) -> None:
        self.records = records
        self.crop_dir = Path(crop_dir)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        pid = rec["patient_id_norm"]
        label = rec["label"]

        patient_dir = self.crop_dir / pid

        # Load all four sequences — each is (128,128,128) float32
        t1    = np.load(patient_dir / "T1_crop.npy").astype(np.float32)
        t2    = np.load(patient_dir / "T2_crop.npy").astype(np.float32)
        flair = np.load(patient_dir / "FLAIR_crop.npy").astype(np.float32)
        t1ce  = np.load(patient_dir / "T1c_crop.npy").astype(np.float32)

        # Domain-knowledge augmented mask fusion → 2 channels
        ch0 = (t1 + flair) / 2.0    # pre-contrast structural
        ch1 = (t1ce + t2)  / 2.0    # contrast-enhanced + fluid

        volume = np.stack([ch0, ch1], axis=0)  # (2, 128, 128, 128)

        return (
            torch.from_numpy(volume),
            torch.tensor(label, dtype=torch.float32),
        )


# ============================================================================
#  2.  MODEL  (Koska et al. 2025 — exact architecture)
# ============================================================================
class ConvBlock3D(nn.Module):
    """Conv3D → ReLU → MaxPool3D(2)"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class KoskaMGMTNet(nn.Module):
    """
    Exact 3D-CNN architecture from Koska et al. (2025).

    Input  :  (B, 2, 128, 128, 128)
    Output :  (B, 1)  — sigmoid probability of MGMT methylation.

    Backbone
    --------
    Block 1:  Conv3D(2  →  64, k=3, p=1) → ReLU → MaxPool3D(2)   →  (B,  64, 64, 64, 64)
    Block 2:  Conv3D(64 →  64, k=3, p=1) → ReLU → MaxPool3D(2)   →  (B,  64, 32, 32, 32)
    Block 3:  Conv3D(64 → 128, k=3, p=1) → ReLU → MaxPool3D(2)   →  (B, 128, 16, 16, 16)
    Block 4:  Conv3D(128→ 256, k=3, p=1) → ReLU → MaxPool3D(2)   →  (B, 256,  8,  8,  8)

    Head
    ----
    Global Average Pooling → (B, 256)
    FC(256 → 512) → ReLU
    FC(512 → 256) → ReLU
    FC(256 →   1) → Sigmoid
    """

    def __init__(self) -> None:
        super().__init__()

        # ── Convolutional backbone ──
        self.backbone = nn.Sequential(
            ConvBlock3D(2,   64),   # Block 1
            ConvBlock3D(64,  64),   # Block 2
            ConvBlock3D(64,  128),  # Block 3
            ConvBlock3D(128, 256),  # Block 4
        )

        # ── Global Average Pooling ──
        self.gap = nn.AdaptiveAvgPool3d(1)

        # ── Fully-connected classifier head ──
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)                 # (B, 256, 8, 8, 8)
        x = self.gap(x)                      # (B, 256, 1, 1, 1)
        x = x.view(x.size(0), -1)            # (B, 256)
        x = self.classifier(x)               # (B, 1)
        return x.squeeze(1)                   # (B,)


# ============================================================================
#  3.  DATA LOADING / STRATIFIED SPLITTING
# ============================================================================
def load_cohort(csv_path: str | Path, crop_dir: str | Path) -> List[Dict]:
    """
    Read *cohort_filtered.csv*, map MGMT labels, drop missing, and
    verify that the corresponding crop directory exists on disk.
    """
    csv_path = Path(csv_path)
    crop_dir = Path(crop_dir)

    records: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            mgmt = row.get("MGMT status", "").strip().lower()
            pid_norm = row.get("patient_id_norm", "").strip()

            # Skip rows with missing / unknown MGMT status
            if mgmt not in ("positive", "negative"):
                continue

            # Verify crop directory exists
            if not (crop_dir / pid_norm).is_dir():
                log.warning("Crop directory missing for %s — skipping.", pid_norm)
                continue

            # Verify all required sequences exist
            missing = [
                s for s in MGMTCropDataset.SEQUENCES
                if not (crop_dir / pid_norm / s).exists()
            ]
            if missing:
                log.warning(
                    "Missing sequences for %s: %s — skipping.", pid_norm, missing
                )
                continue

            records.append(
                {
                    "patient_id_norm": pid_norm,
                    "label": 1 if mgmt == "positive" else 0,
                }
            )

    log.info(
        "Loaded %d valid patients  (pos=%d  neg=%d)",
        len(records),
        sum(r["label"] == 1 for r in records),
        sum(r["label"] == 0 for r in records),
    )
    return records


def stratified_split(
    records: List[Dict],
    n_test: int = 100,
    n_val: int = 77,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Three-way stratified split reproducing the Koska et al. holdout scheme:
      - Test :  100 patients  (class-balanced stratified)
      - Val  :   77 patients  (class-balanced stratified)
      - Train:  remaining ≈191 patients
    """
    labels = [r["label"] for r in records]
    indices = list(range(len(records)))

    # First split: carve out the test set
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=n_test,
        stratify=labels,
        random_state=seed,
    )

    # Second split: carve validation from the remainder
    tv_labels = [records[i]["label"] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=n_val,
        stratify=tv_labels,
        random_state=seed,
    )

    train_set = [records[i] for i in train_idx]
    val_set   = [records[i] for i in val_idx]
    test_set  = [records[i] for i in test_idx]

    def _stats(s: List[Dict]) -> str:
        pos = sum(r["label"] for r in s)
        return f"{len(s)} (pos={pos}, neg={len(s)-pos})"

    log.info("Split sizes  —  Train: %s   Val: %s   Test: %s",
             _stats(train_set), _stats(val_set), _stats(test_set))

    return train_set, val_set, test_set


# ============================================================================
#  4.  METRICS  (contingency-table approach)
# ============================================================================
def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute full evaluation suite from the ground truth labels and
    predicted sigmoid probabilities.

    Returns dict with:  accuracy, precision, recall, f1, roc_auc,
                        tp, fp, tn, fn
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Contingency table
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }

    # ROC-AUC  (requires both classes present in y_true)
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["roc_auc"] = float("nan")

    return metrics


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """Pretty-print metrics dictionary."""
    parts = [
        f"Acc={metrics['accuracy']:.4f}",
        f"Prec={metrics['precision']:.4f}",
        f"Rec={metrics['recall']:.4f}",
        f"F1={metrics['f1']:.4f}",
        f"AUC={metrics['roc_auc']:.4f}",
        f"(TP={metrics['tp']} FP={metrics['fp']} "
        f"TN={metrics['tn']} FN={metrics['fn']})",
    ]
    return f"{prefix}{' │ '.join(parts)}"


# ============================================================================
#  5.  TRAINING ENGINE
# ============================================================================
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Run model on a dataloader; return average loss + metrics dict."""
    model.eval()
    running_loss = 0.0
    all_labels: List[float] = []
    all_probs:  List[float] = []

    for volumes, labels in loader:
        volumes = volumes.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        probs = model(volumes)
        loss  = criterion(probs, labels)

        running_loss += loss.item() * volumes.size(0)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    avg_loss = running_loss / len(loader.dataset)
    metrics  = compute_metrics(
        np.array(all_labels), np.array(all_probs)
    )
    return avg_loss, metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch; return average training loss."""
    model.train()
    running_loss = 0.0

    for volumes, labels in loader:
        volumes = volumes.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        probs = model(volumes)
        loss  = criterion(probs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * volumes.size(0)

    return running_loss / len(loader.dataset)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    save_dir: str | Path = ".",
) -> Path:
    """
    Full training loop with early-stopping by best validation accuracy.

    Returns the path to the saved best-model checkpoint.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = save_dir / "best_model_koska_mgmt.pt"

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = -1.0
    history: List[Dict] = []

    log.info("═" * 80)
    log.info("  Training  │  Device: %s  │  Epochs: %d  │  LR: %.1e", device, epochs, lr)
    log.info("═" * 80)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ──
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # ── Validate ──
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0

        # ── Checkpoint on best validation accuracy ──
        is_best = val_metrics["accuracy"] > best_val_acc
        if is_best:
            best_val_acc = val_metrics["accuracy"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": best_val_acc,
                    "val_metrics": val_metrics,
                },
                best_ckpt,
            )

        # ── Logging ──
        star = " ★" if is_best else ""
        log.info(
            "Epoch %3d/%d │ train_loss=%.4f │ val_loss=%.4f │ "
            "val_acc=%.4f │ val_auc=%.4f │ %.1fs%s",
            epoch, epochs, train_loss, val_loss,
            val_metrics["accuracy"], val_metrics["roc_auc"],
            elapsed, star,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )

    # ── Save training history ──
    history_path = save_dir / "training_history.json"
    with open(history_path, "w") as fh:
        json.dump(history, fh, indent=2)
    log.info("Training history saved to %s", history_path)
    log.info("Best validation accuracy: %.4f  (checkpoint: %s)", best_val_acc, best_ckpt)

    return best_ckpt


# ============================================================================
#  6.  TEST EVALUATION
# ============================================================================
def final_test_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    ckpt_path: Path,
) -> Dict[str, float]:
    """Load best checkpoint and evaluate on the held-out test set."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    log.info(
        "Loaded best checkpoint from epoch %d  (val_acc=%.4f)",
        ckpt["epoch"], ckpt["val_accuracy"],
    )

    criterion = nn.BCELoss()
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)

    log.info("═" * 80)
    log.info("  FINAL TEST RESULTS")
    log.info("─" * 80)
    log.info("  Test Loss : %.4f", test_loss)
    log.info("  %s", format_metrics(test_metrics))
    log.info("═" * 80)

    return test_metrics


# ============================================================================
#  7.  CLI ENTRYPOINT
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Koska et al. 2025 3D-CNN for MGMT prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--csv",
        type=str,
        default="cohort_filtered.csv",
        help="Path to the cohort CSV file.",
    )
    p.add_argument(
        "--crop-dir",
        type=str,
        default=".",
        help="Root directory containing patient crop sub-folders.",
    )
    p.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints and logs.",
    )
    p.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    p.add_argument(
        "--n-test", type=int, default=100, help="Number of test-set patients."
    )
    p.add_argument(
        "--n-val", type=int, default=77, help="Number of validation-set patients."
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'cuda', 'cpu', or 'auto'.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Seed everything ──
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Device selection ──
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    log.info("Using device: %s", device)

    # ── Load cohort & split ──
    records = load_cohort(args.csv, args.crop_dir)
    if len(records) == 0:
        log.error("No valid patient records found. Check CSV and crop directory.")
        sys.exit(1)

    train_records, val_records, test_records = stratified_split(
        records, n_test=args.n_test, n_val=args.n_val, seed=args.seed
    )

    # ── Datasets & DataLoaders ──
    train_ds = MGMTCropDataset(train_records, args.crop_dir)
    val_ds   = MGMTCropDataset(val_records,   args.crop_dir)
    test_ds  = MGMTCropDataset(test_records,  args.crop_dir)

    # pin_memory for faster CPU→GPU transfer
    pin = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    # ── Model ──
    model = KoskaMGMTNet().to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        "Model: KoskaMGMTNet  │  Total params: %s  │  Trainable: %s",
        f"{total_params:,}", f"{trainable:,}",
    )

    # ── Train ──
    best_ckpt = train(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        save_dir=args.save_dir,
    )

    # ── Test ──
    test_metrics = final_test_evaluation(model, test_loader, device, best_ckpt)

    # ── Save final test report ──
    report_path = Path(args.save_dir) / "test_report.json"
    with open(report_path, "w") as fh:
        json.dump(test_metrics, fh, indent=2)
    log.info("Test report saved to %s", report_path)


if __name__ == "__main__":
    main()
