import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import AudioConfig, CHECKPOINT_PATH, DATASET_ROOT, TrainConfig
from src.data.urbansound8k import CLASS_NAMES, UrbanSound8KDataset
from src.models.cnn_mel import LightCNNMelClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate UrbanSound8K CNN classifier")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Choose compute device. Defaults to cuda and falls back to cpu if unavailable.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("[Warning] --device cuda requested but CUDA is unavailable. Falling back to cpu.")
        return torch.device("cpu")
    return torch.device(requested)


def main():
    args = parse_args()
    train_cfg = TrainConfig()
    device = resolve_device(args.device)

    if not Path(CHECKPOINT_PATH).exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}. Please run train.py first.")

    ds = UrbanSound8KDataset(DATASET_ROOT, folds=[10], audio_cfg=AudioConfig())
    loader = DataLoader(ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=train_cfg.num_workers)

    model = LightCNNMelClassifier(num_classes=10).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluate", leave=False):
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(y.numpy().tolist())

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    np.savetxt(out_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Device: {device}")
    print(report)
    print(f"Saved report to: {out_dir / 'classification_report.txt'}")
    print(f"Saved confusion matrix to: {out_dir / 'confusion_matrix.csv'}")


if __name__ == "__main__":
    main()

