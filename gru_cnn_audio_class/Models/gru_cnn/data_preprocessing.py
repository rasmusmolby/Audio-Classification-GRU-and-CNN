import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
import soundfile as sf
import random


def load_and_resample(path, cfg):
# Load and resample  to target sample rate. Returns mono waveform
    waveform, sr = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(waveform).transpose(0, 1)	

    dc = cfg["data"]

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr  != dc["sample_rate"]:
        waveform = T.Resample(orig_freq=sr, new_freq=dc["sample_rate"])(waveform)

    return waveform # new shape is (1, samples)

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

        self.mfcc_transform = T.MFCC(
            sample_rate=dc["sample_rate"],
            n_mfcc=mc["n_mfcc"],
            melkwargs={"n_fft": mc["n_fft"], "hop_length": mc["hop_length"]},
        )


        self.root = Path(root_dir)
        self.samples = []
        MAX_FILES = dc.get("max_files_per_class", 3500)
        for class_name, label in dc["label_map"].items():
            class_dir = self.root / class_name
            wavs = list(class_dir.glob("*.wav")) + list(class_dir.glob("*.flac"))
            if len(wavs) > MAX_FILES:
                wavs = random.sample(wavs, MAX_FILES)
            print(f"{class_name}: {len(wavs)} filer")
            for wav in wavs:
                self.samples.append((str(wav), label))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            waveform = load_and_resample(path, self.cfg)
            waveform = pad_or_trim(waveform, self.cfg)
            mfcc = waveform_to_mfcc(waveform, self.mfcc_transform)
            mfcc = normalize(mfcc)
            return mfcc, label
        except Exception as e:
            print(f"Warning: failed to load {path}: {e}. Skipping.")
            return self.__getitem__((idx + 1) % len(self.samples))
        


def get_dataloaders(root_dir, cfg):
    tc = cfg["training"]
    dataset = NoiseDataset(root_dir, cfg)
    n = len(dataset)
    n_train = int(n*0.70)
    n_val = int(n*0.15)
    n_test = int(n - n_train - n_val)
        
    # Return three dataset splits:
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    # Dataloaders:
    train_loader = DataLoader(train_set, tc["batch_size"], shuffle=True,  num_workers=tc["num_workers"])
    val_loader   = DataLoader(val_set,   tc["batch_size"], shuffle=False, num_workers=tc["num_workers"])
    test_loader  = DataLoader(test_set,  tc["batch_size"], shuffle=False, num_workers=tc["num_workers"])

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    import yaml
    from pathlib import Path
    cfg_path = Path(__file__).parent / "configs" / "default.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    train_loader, val_loader, test_loader = get_dataloaders(cfg["data"]["data_dir"], cfg=cfg)
    specs, labels = next(iter(train_loader))
    print(f"Batch shape: {specs.shape}")
    print(f"Labels: {labels}")
