import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
import soundfile as sf
import random
import librosa

class Config:
    SAMPLE_RATE = 16000
    CLIP_DURATION = 2.0 #In seconds
    N_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)

    LABEL_MAP = {"disturbing": 0, "transient": 1, "stationary": 2}
    NORMALIZATION = "pcen" # "zscore" or "pcen"


def load_and_resample(path):
# Load and resample  to target sample rate. Returns mono waveform
    waveform, sr = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(waveform).transpose(0, 1)	

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr  != Config.SAMPLE_RATE:
        waveform = T.Resample(orig_freq=sr, new_freq=Config.SAMPLE_RATE)(waveform)

    return waveform # new shape is (1, samples)

def pad_or_trim(waveform):
    n = waveform.shape[-1]

    if n < Config.N_SAMPLES:
        waveform = torch.nn.functional.pad(waveform, (0, (Config.N_SAMPLES - n)))

    else: 
        waveform = waveform[..., :Config.N_SAMPLES]

    return waveform

# Spectrogram function
mel_transform = T.MelSpectrogram(
    sample_rate = Config.SAMPLE_RATE,
    n_fft = 512,
    hop_length = 160,
    n_mels = 64,
    f_min = 50,
    f_max = 8000,
)
#Transformation of amplitude to DB
amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)


# Does the actual transformation from waveform to Mel Spectrogram
def waveform_to_mel(waveform):
    return mel_transform(waveform)


def augment_volume(waveform):
    gain = random.uniform(0.8, 1.2)
    return waveform * gain



def pcen_transform(spec):
    spec_np = spec.squeeze(0).numpy()  # (64, 201)
    pcen = librosa.pcen(spec_np, sr=16000, hop_length=160)  # (64, 201)
    return torch.tensor(pcen, dtype=torch.float32).unsqueeze(0)  # (1, 64, 201)


# Does transformation from amplitude to dB in the Spectrogram 
def mel_to_logmel(spec):
    return amplitude_to_db(spec)


def normalize(spec):
    spec_mean = spec.mean()
    spec_std = spec.std()

    return ((spec - spec_mean)/(spec_std + 1e-8)) #1e-8 is added to avoid division with 0


def get_all_samples(root_dir):
    root = Path(root_dir)
    samples = []
    MAX_FILES = 3500
    for class_name, label in Config.LABEL_MAP.items():
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
    def __init__(self, samples, mode="train"):
        self.samples = samples
        self.mode = mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            waveform = load_and_resample(path)
            waveform = pad_or_trim(waveform)
            if self.mode == "train":
                waveform = augment_volume(waveform)
            spec = waveform_to_mel(waveform)
            if Config.NORMALIZATION == "zscore":
                spec = mel_to_logmel(spec)
                spec = normalize(spec)
            elif Config.NORMALIZATION == "pcen":
                spec = pcen_transform(spec)
            return spec, label
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))


def get_dataloaders(root_dir, cfg):
    all_samples = get_all_samples(root_dir)
    random.Random(cfg["data"]["random_seed"]).shuffle(all_samples)  
    
    n = len(all_samples)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train_samples = all_samples[:n_train]
    val_samples   = all_samples[n_train:n_train + n_val]
    test_samples  = all_samples[n_train + n_val:]

    train_set = NoiseDataset(train_samples, mode="train")
    val_set   = NoiseDataset(val_samples,   mode="val")
    test_set  = NoiseDataset(test_samples,  mode="test")

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
'''
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders("data")
    specs, labels = next(iter(train_loader))
    print(f"Batch shape: {specs.shape}")   
    print(f"Labels: {labels}")


    Config.NORMALIZATION = "pcen"
train_loader, val_loader, test_loader = get_dataloaders("data")
specs, labels = next(iter(train_loader))
print(f"PCEN batch shape: {specs.shape}")
print(f"NaN: {specs.isnan().any()}, Inf: {specs.isinf().any()}")
print(f"Min: {specs.min():.4f}, Max: {specs.max():.4f}")
'''