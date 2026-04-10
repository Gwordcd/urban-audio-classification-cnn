"""训练入口脚本：负责构建数据集、模型、损失函数与优化器，并完成训练与测试流程。"""

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
    """解析命令行参数，用于指定训练时使用的计算设备。"""
    parser = argparse.ArgumentParser(description="Train UrbanSound8K CNN classifier")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Choose compute device. Defaults to cuda and falls back to cpu if unavailable.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    """将用户指定的设备字符串转换为 torch.device，并在 CUDA 不可用时回退到 CPU。"""
    if requested == "cuda" and not torch.cuda.is_available():
        print("[Warning] --device cuda requested but CUDA is unavailable. Falling back to cpu.")
        return torch.device("cpu")
    return torch.device(requested)


def main():
    """训练主流程：构建数据集、训练模型、保存最优权重并在测试集上评估。"""
    args = parse_args()
    # 读取训练超参数配置，并固定随机种子，保证实验尽量可复现。
    train_cfg = TrainConfig()
    seed_everything(train_cfg.seed)
    # 读取音频预处理参数，用于生成 Mel 频谱特征。
    audio_cfg = AudioConfig()
    # 根据命令行参数选择训练设备。
    device = resolve_device(args.device)

    # 按官方推荐划分数据集：fold1-8 训练，fold9 验证，fold10 测试。
    train_ds = UrbanSound8KDataset(DATASET_ROOT, folds=range(1, 9), audio_cfg=audio_cfg)
    val_ds = UrbanSound8KDataset(DATASET_ROOT, folds=[9], audio_cfg=audio_cfg)
    test_ds = UrbanSound8KDataset(DATASET_ROOT, folds=[10], audio_cfg=audio_cfg)

    # 构建训练、验证和测试数据加载器。
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

    # 构建轻量级 CNN 分类模型，并移动到指定设备上。
    model = LightCNNMelClassifier(num_classes=10).to(device)
    # 使用交叉熵损失进行多分类训练。
    criterion = nn.CrossEntropyLoss()
    # 使用 AdamW 优化器更新模型参数。
    optimizer = AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)

    # 记录验证集最佳准确率，并准备输出目录。
    best_val_acc = 0.0
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 打印本次实验使用的数据集路径和计算设备，便于确认运行环境。
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Device: {device}")

    # 按 epoch 逐轮训练，并在验证集上选择最佳模型。
    for epoch in range(1, train_cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        # 输出当前轮次的训练与验证指标，便于观察收敛情况。
        print(
            f"Epoch {epoch:03d}/{train_cfg.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
            f"val_f1={val_metrics['f1_macro']:.4f}"
        )

        # 当验证集准确率提升时，保存当前模型为最优权重。
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            save_checkpoint(model, CHECKPOINT_PATH)
            print(f"Saved best checkpoint with val_acc={best_val_acc:.4f}")

    # 训练结束后加载最佳权重，再在测试集上做最终评估。
    if Path(CHECKPOINT_PATH).exists():
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    # 输出最终测试集指标。
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(
        f"Test metrics | loss={test_metrics['loss']:.4f} "
        f"acc={test_metrics['acc']:.4f} f1={test_metrics['f1_macro']:.4f}"
    )


if __name__ == "__main__":
    main()

