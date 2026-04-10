import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.config import AudioConfig, CHECKPOINT_PATH, DATASET_ROOT, TrainConfig
from src.data.urbansound8k import UrbanSound8KDataset
from src.models.cnn_mel import LightCNNMelClassifier
from src.train_utils import evaluate, save_checkpoint, seed_everything, train_one_epoch


def parse_args():
    parser = argparse.ArgumentParser(description="Train UrbanSound8K CNN classifier")
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
    seed_everything(train_cfg.seed)
    audio_cfg = AudioConfig()
    device = resolve_device(args.device)

    train_ds = UrbanSound8KDataset(DATASET_ROOT, folds=range(1, 9), audio_cfg=audio_cfg)
    val_ds = UrbanSound8KDataset(DATASET_ROOT, folds=[9], audio_cfg=audio_cfg)
    test_ds = UrbanSound8KDataset(DATASET_ROOT, folds=[10], audio_cfg=audio_cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
    )

    model = LightCNNMelClassifier(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)

    best_val_acc = 0.0
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Device: {device}")

    for epoch in range(1, train_cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:03d}/{train_cfg.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
            f"val_f1={val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            save_checkpoint(model, CHECKPOINT_PATH)
            print(f"Saved best checkpoint with val_acc={best_val_acc:.4f}")

    if Path(CHECKPOINT_PATH).exists():
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    test_metrics = evaluate(model, test_loader, criterion, device)
    print(
        f"Test metrics | loss={test_metrics['loss']:.4f} "
        f"acc={test_metrics['acc']:.4f} f1={test_metrics['f1_macro']:.4f}"
    )


if __name__ == "__main__":
    main()

