import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
from pathlib import Path
import subprocess
import numpy as np
import random
import yaml


# NOTE PREPROCESSING USED IN CNN_GRU

def load_and_resample(path, cfg):
    """Decode audio to raw PCM via ffmpeg. Faster than soundfile and handles more formats."""
    target_sr = cfg["data"]["sample_rate"]
    result = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", str(path),
         "-f", "f32le", "-acodec", "pcm_f32le",
         "-ar", str(target_sr), "-ac", "1", "pipe:1"],
        capture_output=True,
    )
    if result.returncode != 0 or len(result.stdout) == 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {result.stderr.decode().strip()}")
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    return torch.from_numpy(audio.copy()).unsqueeze(0)  # (1, samples)


def pad_or_trim(waveform, cfg):
    dc = cfg["data"]
    n_samples = int(dc["duration"] * dc["sample_rate"])
    n = waveform.shape[-1]
    if n < n_samples:
        waveform = torch.nn.functional.pad(waveform, (0, n_samples - n))
    else:
        waveform = waveform[..., :n_samples]
    return waveform


def augment_volume(waveform):
    """Random volume jitter in [0.8, 1.2]. Applied during training only."""
    gain = random.uniform(0.8, 1.2)
    return waveform * gain


def waveform_to_mfcc(waveform, mfcc_transform):
    return mfcc_transform(waveform).squeeze(0)  # (n_mfcc, time)


def normalize(spec):
    mean = spec.mean()
    std  = spec.std()
    return (spec - mean) / (std + 1e-8)


# ---------------------------------------------------------------------------
# Build the full sample list before splitting
# ---------------------------------------------------------------------------

def get_all_samples(root_dir, cfg):
    """
    Walk each class folder and return a flat list of (path, label) tuples.
    Capped at max_files_per_class per class.
    """
    dc = cfg["data"]
    root = Path(root_dir)
    MAX_FILES = dc.get("max_files_per_class", 3500)
    samples = []
    for class_name, label in dc["label_map"].items():
        class_dir = root / class_name
        wavs = list(class_dir.glob("*.wav")) + list(class_dir.glob("*.flac"))
        if len(wavs) > MAX_FILES:
            wavs = random.Random(dc["random_seed"]).sample(wavs, MAX_FILES)
        print(f"{class_name}: {len(wavs)} files")
        for wav in wavs:
            samples.append((str(wav), label))
    return samples


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NoiseDataset(Dataset):
    """
    Accepts a pre-split sample list and a mode flag.
    Supports both live processing and .npy cache.
    mode: "train" | "val" | "test"
    """
    def __init__(self, samples, cfg, mode="train"):
        self.samples  = samples
        self.cfg      = cfg
        self.mode     = mode
        self._skip_count = 0

        dc = cfg["data"]
        mc = cfg["mfcc"]
        cache_dir = Path(cfg["output"]["cache_dir"])

        # Cache is usable only if the directory exists and has .npy files
        self.use_cache = (
            dc.get("use_cache", False)
            and cache_dir.exists()
            and any(cache_dir.rglob("*.npy"))
        )

        if self.use_cache:
            print(f"[{mode}] Loading from preprocessed cache...")
        else:
            print(f"[{mode}] No cache / cache disabled — loading raw audio.")
            self.mfcc_transform = T.MFCC(
                sample_rate=dc["sample_rate"],
                n_mfcc=mc["n_mfcc"],
                melkwargs={"n_fft": mc["n_fft"], "hop_length": mc["hop_length"]},
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            if self.use_cache:
                mfcc = torch.from_numpy(np.load(path))
            else:
                waveform = load_and_resample(path, self.cfg)
                waveform = pad_or_trim(waveform, self.cfg)
                # Volume augmentation on training data only
                if self.mode == "train":
                    waveform = augment_volume(waveform)
                mfcc = waveform_to_mfcc(waveform, self.mfcc_transform)
                mfcc = normalize(mfcc)
            return mfcc, label
        except Exception:
            self._skip_count += 1
            if self._skip_count % 50 == 0:
                print(f"[{self.mode}] {self._skip_count} files skipped so far (corrupt/unreadable)")
            return self.__getitem__((idx + 1) % len(self.samples))


# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------

def get_dataloaders(root_dir, cfg):
    tc = cfg["training"]
    dc = cfg["data"]

    # Collect and shuffle all samples with a fixed seed before splitting.
    # This means train/val/test always contain the same files regardless of
    # when or how many times get_dataloaders is called.
    all_samples = get_all_samples(root_dir, cfg)
    random.Random(dc["random_seed"]).shuffle(all_samples)

    n       = len(all_samples)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    # test gets whatever is left — avoids rounding errors dropping samples
    train_samples = all_samples[:n_train]
    val_samples   = all_samples[n_train : n_train + n_val]
    test_samples  = all_samples[n_train + n_val :]

    print(f"Split — train: {len(train_samples)}  val: {len(val_samples)}  test: {len(test_samples)}")

    # Separate Dataset instances so mode-dependent logic (augmentation) works correctly
    train_set = NoiseDataset(train_samples, cfg, mode="train")
    val_set   = NoiseDataset(val_samples,   cfg, mode="val")
    test_set  = NoiseDataset(test_samples,  cfg, mode="test")

    train_loader = DataLoader(train_set, tc["batch_size"], shuffle=True,  num_workers=tc["num_workers"])
    val_loader   = DataLoader(val_set,   tc["batch_size"], shuffle=False, num_workers=tc["num_workers"])
    test_loader  = DataLoader(test_set,  tc["batch_size"], shuffle=False, num_workers=tc["num_workers"])

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
 
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
 
    train_loader, val_loader, test_loader = get_dataloaders(cfg["data"]["data_dir"], cfg)
 
    print("\n--- Batch check ---")
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        mfcc, labels = next(iter(loader))
        print(f"[{split_name}] batch shape: {tuple(mfcc.shape)}  labels: {labels.tolist()}")
        print(f"[{split_name}] min: {mfcc.min():.4f}  max: {mfcc.max():.4f}  "
              f"nan: {mfcc.isnan().any().item()}  inf: {mfcc.isinf().any().item()}")
 
    # Check that augmentation actually does something — two passes over the same
    # training index should return slightly different values due to volume jitter
    print("\n--- Augmentation check (train only) ---")
    dataset = train_loader.dataset
    a, _ = dataset[0]
    b, _ = dataset[0]
    print(f"Same index, two draws — identical: {torch.allclose(a, b)}  (should be False for train)")