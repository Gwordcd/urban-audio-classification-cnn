"""UrbanSound8K 数据集读取与 Log-Mel 特征提取实现。"""

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from src.config import AudioConfig


CLASS_NAMES = [
    # UrbanSound8K 的 10 个类别名称。
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
    """用于读取 UrbanSound8K 音频并转换为 Log-Mel 频谱的数据集类。"""
    def __init__(self, root: str, folds, audio_cfg: Optional[AudioConfig] = None):
        # 数据集根目录。
        self.root = Path(root)
        # 若未传入音频配置，则使用默认参数。
        self.audio_cfg = audio_cfg or AudioConfig()

        # 元数据文件中记录了每个音频片段的文件名、fold 和类别标签。
        csv_path = self.root / "metadata" / "UrbanSound8K.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing metadata: {csv_path}")

        # 读取全部样本，再根据 fold 进行筛选。
        df = pd.read_csv(csv_path)
        self.samples = df[df["fold"].isin(list(folds))].reset_index(drop=True)

        # Mel 频谱转换模块。
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.audio_cfg.sample_rate,
            n_fft=self.audio_cfg.n_fft,
            hop_length=self.audio_cfg.hop_length,
            n_mels=self.audio_cfg.n_mels,
            f_min=self.audio_cfg.f_min,
            f_max=self.audio_cfg.f_max,
        )
        # 将功率谱转换为分贝尺度。
        self.to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        """返回当前数据子集中的样本数。"""
        return len(self.samples)

    def _pad_or_cut(self, wav: torch.Tensor) -> torch.Tensor:
        """将音频裁剪或补零到固定长度，保证输入尺寸一致。"""
        target_len = int(self.audio_cfg.sample_rate * self.audio_cfg.clip_duration)
        length = wav.size(-1)
        if length < target_len:
            # 长度不足则在尾部补零。
            wav = torch.nn.functional.pad(wav, (0, target_len - length))
        elif length > target_len:
            # 长度过长则直接截断。
            wav = wav[..., :target_len]
        return wav

    def __getitem__(self, idx):
        """读取单条音频样本，返回 Log-Mel 频谱和类别编号。"""
        row = self.samples.iloc[idx]
        # 根据 fold 与文件名拼接出音频路径。
        audio_path = self.root / "audio" / f"fold{int(row['fold'])}" / row["slice_file_name"]
        # 读取音频波形和采样率。
        wav, sr = torchaudio.load(str(audio_path))

        # 转为单声道，减少通道差异影响。
        wav = wav.mean(dim=0, keepdim=True)
        if sr != self.audio_cfg.sample_rate:
            # 若采样率不同，则重采样到统一采样率。
            wav = torchaudio.functional.resample(wav, sr, self.audio_cfg.sample_rate)

        # 统一音频长度。
        wav = self._pad_or_cut(wav)
        # 计算 Mel 频谱并转换为对数尺度。
        mel = self.to_db(self.mel(wav))
        # 对特征进行标准化，提升训练稳定性。
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)

        # 返回模型输入特征与标签编号。
        return mel, int(row["classID"])

