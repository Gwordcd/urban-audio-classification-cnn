"""项目配置文件：集中管理数据路径、音频预处理参数和训练超参数。"""

from dataclasses import dataclass


# 数据集根目录与模型权重保存路径，便于统一修改。
DATASET_ROOT = "src/data/UrbanSound8K"
CHECKPOINT_PATH = "outputs/best.pth"


@dataclass
class AudioConfig:
    """音频特征提取相关参数。"""
    sample_rate: int = 22050
    # 音频裁剪后的固定时长，单位为秒。
    clip_duration: float = 4.0
    # Mel 滤波器个数，决定频谱图在 Mel 维度上的分辨率。
    n_mels: int = 128
    # STFT 的 FFT 点数。
    n_fft: int = 1024
    # STFT 的滑动步长。
    hop_length: int = 512
    # Mel 频率范围下限。
    f_min: float = 20.0
    # Mel 频率范围上限。
    f_max: float = 8000.0


@dataclass
class TrainConfig:
    """训练过程相关参数。"""
    # 每个 batch 中包含的样本数。
    batch_size: int = 32
    # 训练轮数。
    epochs: int = 10
    # 学习率。
    learning_rate: float = 1e-3
    # 权重衰减系数。
    weight_decay: float = 1e-4
    # 数据加载线程数。
    num_workers: int = 0
    # 随机种子，用于保证实验可复现。
    seed: int = 42

