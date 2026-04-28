"""


THIS SCRIPT IS ALL CHAT MADE!



Quantization and precision utilities.

FP16 (mixed precision):
    Imported and used by train.py. Toggle with training.fp16 in config.

INT8 (post-training dynamic quantization):
    Run as a standalone script after training:
        python quantization.py
    Loads best.pt, quantizes Linear + GRU layers to INT8, saves model_int8.pt.

"""

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

from model import CNNGRU


# FP16
def get_amp(device, enabled):
    """Set up AMP components. Returns (scaler, fp16_active)."""
    fp16 = enabled and device.type == "cuda"
    scaler = GradScaler(device=device.type, enabled=fp16)
    print(f"FP16 mixed precision: {'on' if fp16 else 'off'}")
    return scaler, fp16


def training_step(model, x, labels, loss_fn, optimizer, scaler, device, fp16):
    """Forward + backward with optional FP16. Returns loss value."""
    optimizer.zero_grad()
    with autocast(device_type=device.type, enabled=fp16):
        output = model(x)
        loss = loss_fn(output, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()


# INT8

def quantize_int8(model):
    """Dynamically quantize Linear + GRU layers to INT8. Returns CPU model."""
    model = model.cpu().eval()
    # quantize_dynamic is deprecated in PyTorch 2.x but still fully functional.
    # The replacement (torch.export + pt2e) doesn't support GRU well yet.
    return torch.ao.quantization.quantize_dynamic(  # why the fuck is it crossed out?
        model,
        {nn.Linear, nn.GRU},
        dtype=torch.qint8,
    )


@hydra.main(version_base="1.2", config_path="configs", config_name="default")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    mc = cfg["model"]
    oc = cfg["output"]

    save_dir = Path(oc["save_dir"])
    best_ckpt = save_dir / "best.pt"
    int8_path = save_dir / "model_int8.pt"

    if not best_ckpt.exists():
        raise FileNotFoundError(f"No checkpoint found at {best_ckpt}. Train a model first.")

    model = CNNGRU(
        n_mfcc=mc["n_mfcc"],
        c_cnn=mc["c_cnn"],
        n_classes=mc["n_classes"],
        gru_state=mc["gru_state"],
        dropout=mc["dropout"],
    )

    ckpt = torch.load(best_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded {best_ckpt}")

    model_int8 = quantize_int8(model)
    torch.save(model_int8, int8_path)
    print(f"INT8 model saved to {int8_path}")


if __name__ == "__main__":
    main()
