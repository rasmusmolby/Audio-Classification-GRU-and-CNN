"""'

QUICK CACHE SYSTEM
Run before train to convert everything into mfcc as to not bottleneck:)
    python preprocess.py
"""

import numpy as np
import subprocess
import torch
import torchaudio.transforms as T
import random
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path


def ffmpeg_load(path, target_sr):
    result = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", str(path),
         "-f", "f32le", "-acodec", "pcm_f32le",
         "-ar", str(target_sr), "-ac", "1", "pipe:1"],
        capture_output=True,
    )
    if result.returncode != 0 or len(result.stdout) == 0:
        raise RuntimeError(result.stderr.decode().strip())
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    return torch.from_numpy(audio.copy()).unsqueeze(0)


def pad_or_trim(waveform, n_samples):
    n = waveform.shape[-1]
    if n < n_samples:
        waveform = torch.nn.functional.pad(waveform, (0, n_samples - n))
    else:
        waveform = waveform[..., :n_samples]
    return waveform


@hydra.main(version_base="1.2", config_path="configs", config_name="default")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    dc = cfg["data"]
    mc = cfg["mfcc"]

    cache_dir = Path(cfg["output"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    target_sr = dc["sample_rate"]
    n_samples = int(dc["duration"] * target_sr)
    max_files = dc.get("max_files_per_class", 3500)

    mfcc_transform = T.MFCC(
        sample_rate=target_sr,
        n_mfcc=mc["n_mfcc"],
        melkwargs={"n_fft": mc["n_fft"], "hop_length": mc["hop_length"]},
    )

    total_saved = 0
    total_skipped = 0
    total_failed = 0

    for class_name in dc["label_map"]:
        class_dir = Path(dc["data_dir"]) / class_name
        out_dir = cache_dir / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        files = list(class_dir.glob("*.wav")) + list(class_dir.glob("*.flac"))
        if len(files) > max_files:
            random.seed(dc["random_seed"])
            files = random.sample(files, max_files)

        print(f"\n{class_name}: {len(files)} files")

        for i, path in enumerate(files):
            npy_path = out_dir / (path.stem + ".npy")

            if npy_path.exists():
                total_skipped += 1
                continue

            try:
                waveform = ffmpeg_load(path, target_sr)
                waveform = pad_or_trim(waveform, n_samples)
                mfcc = mfcc_transform(waveform).squeeze(0)
                mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
                np.save(npy_path, mfcc.numpy())
                total_saved += 1
            except Exception:
                total_failed += 1

            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(files)}")

    print(f"\nDone.")
    print(f"  Saved:   {total_saved}")
    print(f"  Skipped: {total_skipped} (already cached)")
    print(f"  Failed:  {total_failed} (corrupt files, skipped)")
    print(f"\nCache at: {cache_dir.resolve()}")


if __name__ == "__main__":
    main()
