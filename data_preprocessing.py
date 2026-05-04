import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio.transforms as T
from pathlib import Path
import subprocess
import numpy as np
import random


def load_and_resample(path, cfg):
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
    return torch.from_numpy(audio.copy()).unsqueeze(0)  # shape: (1, samples)

def pad_or_trim(waveform, cfg):
    dc = cfg["data"]
    n_samples = int(dc["duration"] * dc["sample_rate"])

    n = waveform.shape[-1]

    if n < n_samples:
        waveform = torch.nn.functional.pad(waveform, (0, (n_samples - n)))

    else: 
        waveform = waveform[..., :n_samples]

    return waveform

'''
# TODO Keeping just if needed later on, but if not remove

# Spectrogram function
mel_transform = T.MelSpectrogram(
    sample_rate = Config.SAMPLE_RATE,
    n_fft = 512,
    hop_length = 160,
    n_mels = 64,
    f_min = 50,
    f_max = 8000,
)
amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)
'''

# Does the actual transformation from waveform to mfcc
def waveform_to_mfcc(waveform, mfcc_transform):
    
    return mfcc_transform(waveform).squeeze(0) # So shape is (39, Time)


def normalize(spec):
    spec_mean = spec.mean()
    spec_std = spec.std()

    return ((spec - spec_mean)/(spec_std + 1e-8)) #1e-8 is added to avoid division with 0


class NoiseDataset(Dataset):
    def __init__(self, root_dir, cfg):
        dc = cfg["data"]
        mc = cfg["mfcc"]
        self.cfg = cfg

        # Cache checker
        cache_dir = Path(cfg["output"]["cache_dir"])
        self.use_cache = cache_dir.exists() and any(cache_dir.rglob("*.npy"))

        if self.use_cache and dc["use_cache"] == True:
            print("Loading from preprocessed cache...")
            self.samples = []
            for class_name, label in dc["label_map"].items():
                npy_files = sorted((cache_dir / class_name).glob("*.npy"))
                for f in npy_files:
                    self.samples.append((str(f), label))
        else:
            print("No cache found or cache disabled in default.yaml, loading raw audio (slow). Run preprocess.py first.")
            self.mfcc_transform = T.MFCC(
                sample_rate=dc["sample_rate"],
                n_mfcc=mc["n_mfcc"],
                melkwargs={"n_fft": mc["n_fft"], "hop_length": mc["hop_length"]},
            )
            self.samples = []
            MAX_FILES = dc.get("max_files_per_class", 3500)
            for class_name, label in dc["label_map"].items():
                class_dir = Path(root_dir) / class_name
                wavs = list(class_dir.glob("*.wav")) + list(class_dir.glob("*.flac"))
                if len(wavs) > MAX_FILES:
                    wavs = random.sample(wavs, MAX_FILES)
                print(f"{class_name}: {len(wavs)} files")
                for wav in wavs:
                    self.samples.append((str(wav), label))

        print(f"Dataset ready: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            if self.use_cache and self.cfg["data"]["use_cache"] == True:
                mfcc = torch.from_numpy(np.load(path))
            else:
                waveform = load_and_resample(path, self.cfg)
                waveform = pad_or_trim(waveform, self.cfg)
                mfcc = waveform_to_mfcc(waveform, self.mfcc_transform)
                mfcc = normalize(mfcc)
            return mfcc, label
        except Exception:
            self._skip_count = getattr(self, "_skip_count", 0) + 1
            if self._skip_count % 50 == 0:
                print(f"[dataset] {self._skip_count} files skipped so far (corrupt/unreadable)")
            return self.__getitem__((idx + 1) % len(self.samples))

def get_dataloaders(root_dir, cfg):
    tc = cfg["training"]
    dataset = NoiseDataset(root_dir, cfg)
    n = len(dataset)
    n_train = int(n*0.70)
    n_val = int(n*0.15)
    n_test = int(n - n_train - n_val)
        
    # Return three dataset splits, explicit generator so the split is always the same regardless of global random state
    generator = torch.Generator().manual_seed(cfg["data"]["random_seed"])
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=generator)

    # Dataloaders:
    train_loader = DataLoader(train_set, tc["batch_size"], shuffle=True,  num_workers=tc["num_workers"])
    val_loader   = DataLoader(val_set,   tc["batch_size"], shuffle=False, num_workers=tc["num_workers"])
    test_loader  = DataLoader(test_set,  tc["batch_size"], shuffle=False, num_workers=tc["num_workers"])

    return train_loader, val_loader, test_loader
