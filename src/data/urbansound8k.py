from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from src.config import AudioConfig


CLASS_NAMES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]


class UrbanSound8KDataset(Dataset):
    def __init__(self, root: str, folds, audio_cfg: Optional[AudioConfig] = None):
        self.root = Path(root)
        self.audio_cfg = audio_cfg or AudioConfig()

        csv_path = self.root / "metadata" / "UrbanSound8K.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing metadata: {csv_path}")

        df = pd.read_csv(csv_path)
        self.samples = df[df["fold"].isin(list(folds))].reset_index(drop=True)

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.audio_cfg.sample_rate,
            n_fft=self.audio_cfg.n_fft,
            hop_length=self.audio_cfg.hop_length,
            n_mels=self.audio_cfg.n_mels,
            f_min=self.audio_cfg.f_min,
            f_max=self.audio_cfg.f_max,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.samples)

    def _pad_or_cut(self, wav: torch.Tensor) -> torch.Tensor:
        target_len = int(self.audio_cfg.sample_rate * self.audio_cfg.clip_duration)
        length = wav.size(-1)
        if length < target_len:
            wav = torch.nn.functional.pad(wav, (0, target_len - length))
        elif length > target_len:
            wav = wav[..., :target_len]
        return wav

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        audio_path = self.root / "audio" / f"fold{int(row['fold'])}" / row["slice_file_name"]
        wav, sr = torchaudio.load(str(audio_path))

        wav = wav.mean(dim=0, keepdim=True)
        if sr != self.audio_cfg.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.audio_cfg.sample_rate)

        wav = self._pad_or_cut(wav)
        mel = self.to_db(self.mel(wav))
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)

        return mel, int(row["classID"])

