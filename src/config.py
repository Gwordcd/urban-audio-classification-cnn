from dataclasses import dataclass


# Fixed paths and hyper-parameters for this course assignment.
DATASET_ROOT = "src/data/UrbanSound8K"
CHECKPOINT_PATH = "outputs/best.pth"


@dataclass
class AudioConfig:
    sample_rate: int = 22050
    clip_duration: float = 4.0
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 512
    f_min: float = 20.0
    f_max: float = 8000.0


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    seed: int = 42

