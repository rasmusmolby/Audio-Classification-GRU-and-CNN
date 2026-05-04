"""

IMPORTANT!!!! This is ALL 100% CHAT and is not really checked if works!!!
Also, it is kinda broken, so dont actually use it in report!!!


Validation script — compares all available saved models against the test set.

Prints accuracy, per-class precision/recall/F1, parameter count,
file size, and inference speed for each model found in output.save_dir.

Usage:
    python validation.py
"""

import time
import os
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from model import CNNGRU
import data_preprocessing as dp


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_fp32_model(path, cfg, device):
    mc = cfg["model"]
    model = CNNGRU(
        n_mfcc=mc["n_mfcc"], c_cnn=mc["c_cnn"],
        n_classes=mc["n_classes"], gru_state=mc["gru_state"],
        dropout=mc["dropout"],
    )
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    return model.eval().to(device)


def load_int8_model(path):
    # INT8 dynamic quantization runs on CPU only
    return torch.load(path, map_location="cpu", weights_only=False).eval()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(preds, labels, n_classes):
    cm = [[0] * n_classes for _ in range(n_classes)]
    for p, l in zip(preds, labels):
        cm[l][p] += 1

    accuracy = sum(cm[i][i] for i in range(n_classes)) / max(len(labels), 1)

    per_class = {}
    for i in range(n_classes):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(n_classes)) - tp
        fn = sum(cm[i][j] for j in range(n_classes)) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[i] = {"precision": precision, "recall": recall, "f1": f1}

    macro_f1 = sum(v["f1"] for v in per_class.values()) / n_classes
    return accuracy, per_class, macro_f1


def evaluate(model, loader, device):
    all_preds, all_labels = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    elapsed = time.perf_counter() - t0
    ms_per_sample = (elapsed / max(len(all_labels), 1)) * 1000
    return all_preds, all_labels, ms_per_sample


def model_stats(path, model):
    file_size_mb = os.path.getsize(path) / 1e6
    n_params = sum(p.numel() for p in model.parameters())
    return file_size_mb, n_params


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(name, accuracy, per_class, macro_f1, ms_per_sample,
                 file_size_mb, n_params, label_names):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Accuracy:       {accuracy*100:.2f}%")
    print(f"  Macro F1:       {macro_f1*100:.2f}%")
    print(f"  Speed:          {ms_per_sample:.3f} ms/sample")
    print(f"  File size:      {file_size_mb:.2f} MB")
    print(f"  Parameters:     {n_params:,}")
    print(f"\n  {'Class':<18} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*50}")
    for i, class_name in enumerate(label_names):
        m = per_class[i]
        print(f"  {class_name:<18} {m['precision']*100:>9.2f}% {m['recall']*100:>9.2f}% {m['f1']*100:>9.2f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base="1.2", config_path="configs", config_name="default")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    dc = cfg["data"]
    oc = cfg["output"]
    tc = cfg["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path("validations/")
    label_names = list(dc["label_map"].keys())
    n_classes = len(label_names)

    _, _, test_load = dp.get_dataloaders(root_dir=dc["data_dir"], cfg=cfg)
    print(f"Test set: {len(test_load.dataset)} samples")

    # Models to evaluate: (display name, path, loader_type)
    candidates = [
        ("FP32  (best.pt)",        save_dir / "best.pt",        "fp32"),
        ("FP16  (best_fp16.pt)",   save_dir / "best_fp16.pt",   "fp32"),
        ("INT8  (model_int8.pt)",  save_dir / "model_int8.pt",  "int8"),
    ]

    found_any = False
    for name, path, kind in candidates:
        if not path.exists():
            print(f"\n[skip] {name} — not found at {path}")
            continue

        found_any = True
        print(f"\nEvaluating {name} ...")

        if kind == "fp32":
            model = load_fp32_model(path, cfg, device)
            eval_device = device
        else:
            model = load_int8_model(path)
            eval_device = torch.device("cpu")

        # Use a CPU loader for INT8 to avoid moving tensors
        loader = test_load if eval_device.type != "cpu" else _cpu_loader(test_load)

        preds, labels, ms_per_sample = evaluate(model, loader, eval_device)
        accuracy, per_class, macro_f1 = compute_metrics(preds, labels, n_classes)
        file_size_mb, n_params = model_stats(path, model)

        print_report(name, accuracy, per_class, macro_f1,
                     ms_per_sample, file_size_mb, n_params, label_names)

    if not found_any:
        print("\nNo models found. Train a model first (python train.py).")


def _cpu_loader(loader):
    """Wraps a dataloader to ensure tensors stay on CPU."""
    for x, labels in loader:
        yield x.cpu(), labels.cpu()


if __name__ == "__main__":
    main()
