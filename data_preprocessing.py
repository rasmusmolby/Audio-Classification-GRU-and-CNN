import torch
import torchaudio.transforms as T
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
import soundfile as sf
import random
import librosa

def load_and_resample(path, cfg):
# Load and resample  to target sample rate. Returns mono waveform
    waveform, sr = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(waveform).transpose(0, 1)	

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr  != cfg["data"]["sample_rate"]:
        waveform = T.Resample(orig_freq=sr, new_freq=cfg["data"]["sample_rate"])(waveform)

    return waveform # new shape is (1, samples)

def pad_or_trim(waveform, cfg):
    n_samples = int(cfg["data"]["sample_rate"] * cfg["data"]["duration"])

    n = waveform.shape[-1]

    if n < n_samples:
        waveform = torch.nn.functional.pad(waveform, (0, (n_samples - n)))

    else: 
        waveform = waveform[..., :n_samples]

    return waveform

# Spectrogram function
def build_mel_transform(cfg):
    return T.MelSpectrogram(
        sample_rate = cfg["data"]["sample_rate"],
        n_fft = cfg["data"]["n_fft"],
        hop_length = cfg["data"]["hop_length"],
        n_mels = cfg["model"]["n_mels"],
        f_min = cfg["data"]["f_min"],
        f_max = cfg["data"]["f_max"],
    )
#Transformation of amplitude to DB
amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

def augment_volume(waveform):
    gain = random.uniform(0.8, 1.2)
    return waveform * gain


def augment_time_stretch(waveform, rate=1.5):
    """Time-stretch waveform by `rate` using librosa (CPU, numpy bridge)."""
    wav_np = waveform.squeeze(0).numpy()          # (samples,)
    stretched = librosa.effects.time_stretch(wav_np, rate=rate)
    return torch.from_numpy(stretched).unsqueeze(0)  # (1, samples)


def augment_pitch_shift(waveform, cfg, n_steps=2):
    """Shift pitch by `n_steps` semitones."""
    wav_np = waveform.squeeze(0).numpy()
    shifted = librosa.effects.pitch_shift(
        wav_np,
        sr=cfg["data"]["sample_rate"],
        n_steps=n_steps
    )
    return torch.from_numpy(shifted).unsqueeze(0)  # (1, samples)


def augment_additive_noise(waveform):
    """Add zero-mean Gaussian noise; amplitude ~ Uniform(0.005, 0.008)."""
    amplitude = random.uniform(0.005, 0.008)
    noise = torch.randn_like(waveform) * amplitude
    return waveform + noise


def pcen_transform(spec, cfg):
    spec_np = spec.squeeze(0).numpy()  # (64, 201)
    pcen = librosa.pcen(spec_np, 
                        sr=cfg["data"]["sample_rate"], 
                        hop_length=cfg["data"]["hop_length"])  # (64, 201)
    return torch.tensor(pcen, dtype=torch.float32).unsqueeze(0)  # (1, 64, 201)


# Does transformation from amplitude to dB in the Spectrogram 
def mel_to_logmel(spec):
    return amplitude_to_db(spec)


def normalize(spec):
    spec_mean = spec.mean()
    spec_std = spec.std()

    return ((spec - spec_mean)/(spec_std + 1e-8)) # 1e-8 is added to avoid division with 0


def get_all_samples(root_dir, cfg):
    root = Path(root_dir)
    samples = []
    MAX_FILES = cfg["data"]["max_files"]
    for class_name, label in cfg["data"]["label_map"].items():
        class_dir = root / class_name
        wavs = list(class_dir.glob("*.wav")) + list(class_dir.glob("*.flac"))
        if len(wavs) > MAX_FILES:
            wavs = random.Random(42).sample(wavs, MAX_FILES)
        for wav in wavs:
            samples.append((str(wav), label))
    return samples


class NoiseDataset(Dataset):
    '''
    This is the NEW UPDATED VERSION of CARL EMILS SUPER COOL PRE PROCESSING SCRIPT!
    NO VIRUS!

    '''
    def __init__(self, samples, cfg, mode="train"):
        self.samples = samples
        self.mode = mode
        self.cfg = cfg
        self.mel_transform = build_mel_transform(cfg)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            waveform = load_and_resample(path, self.cfg)
            waveform = pad_or_trim(waveform, self.cfg)   # ← trim FIRST to fixed length

            if self.mode == "train":
                # Apply each augmentation randomly, not always
                if random.random() < 0.5:
                    rate = random.uniform(0.9, 1.1)      # ← subtle range, not 1.5
                    waveform = augment_time_stretch(waveform, rate=rate)
                    waveform = pad_or_trim(waveform, self.cfg)  # re-trim after stretch

                if random.random() < 0.5:
                    n_steps = random.choice([-2, -1, 1, 2])
                    waveform = augment_pitch_shift(waveform, self.cfg, n_steps=n_steps)

                if random.random() < 0.5:
                    waveform = augment_volume(waveform)

                if random.random() < 0.3:
                    waveform = augment_additive_noise(waveform)

            spec = self.mel_transform(waveform)

            if self.cfg["data"]["normalization"] == "zscore":
                spec = mel_to_logmel(spec)
                spec = normalize(spec)
            elif self.cfg["data"]["normalization"] == "pcen":
                spec = pcen_transform(spec, self.cfg)

            return spec, label
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))


def get_dataloaders(root_dir, cfg):
    all_samples = get_all_samples(root_dir, cfg)
    random.Random(cfg["data"]["random_seed"]).shuffle(all_samples)  
    
    n = len(all_samples)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train_samples = all_samples[:n_train]
    val_samples   = all_samples[n_train:n_train + n_val]
    test_samples  = all_samples[n_train + n_val:]

    train_set = NoiseDataset(train_samples, cfg, mode="train")
    val_set   = NoiseDataset(val_samples, cfg,   mode="val")
    test_set  = NoiseDataset(test_samples, cfg,  mode="test")

    train_loader = DataLoader(train_set, 
                              batch_size=cfg["training"]["batch_size"], 
                              shuffle=cfg["training"]["shuffle"],  
                              num_workers=cfg["training"]["num_workers"], 
                              pin_memory=cfg["training"]["pin_memory"])
    
    val_loader   = DataLoader(val_set,
                              batch_size=cfg["test"]["batch_size"], 
                              shuffle=cfg["test"]["shuffle"], 
                              num_workers=cfg["test"]["num_workers"], 
                              pin_memory=cfg["test"]["pin_memory"])
    
    test_loader  = DataLoader(test_set,
                              batch_size=cfg["test"]["batch_size"], 
                              shuffle=cfg["test"]["shuffle"], 
                              num_workers=cfg["test"]["num_workers"], 
                              pin_memory=cfg["test"]["pin_memory"])

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    import yaml
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    train_loader, val_loader, test_loader = get_dataloaders(root_dir=cfg["data"]["data_dir"], cfg=cfg)
    specs, labels = next(iter(train_loader))
    print(f"Batch shape: {specs.shape}")
    print(f"Labels: {labels}")
    print(f"NaN: {specs.isnan().any()}, Inf: {specs.isinf().any()}")
    print(f"Min: {specs.min():.4f}, Max: {specs.max():.4f}")