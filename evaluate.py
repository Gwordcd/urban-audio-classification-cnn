"""评估入口脚本：加载训练好的模型，在测试集上输出分类报告与混淆矩阵。"""

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
    """解析命令行参数，用于指定评估时使用的计算设备。"""
    parser = argparse.ArgumentParser(description="Evaluate UrbanSound8K CNN classifier")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Choose compute device. Defaults to cuda and falls back to cpu if unavailable.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    """将设备字符串转换为 torch.device，并在 CUDA 不可用时回退到 CPU。"""
    if requested == "cuda" and not torch.cuda.is_available():
        print("[Warning] --device cuda requested but CUDA is unavailable. Falling back to cpu.")
        return torch.device("cpu")
    return torch.device(requested)


def main():
    """评估主流程：加载模型、推理测试集并保存分类结果文件。"""
    args = parse_args()
    # 读取训练配置中的 batch_size 和 num_workers，保证训练与评估保持一致。
    train_cfg = TrainConfig()
    # 根据命令行参数选择评估设备。
    device = resolve_device(args.device)

    # 检查训练好的权重是否存在，不存在则提示先完成训练。
    if not Path(CHECKPOINT_PATH).exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}. Please run train.py first.")

    # 只使用 fold10 作为测试集进行最终评估。
    ds = UrbanSound8KDataset(DATASET_ROOT, folds=[10], audio_cfg=AudioConfig())
    loader = DataLoader(ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=train_cfg.num_workers)

    # 构建模型并加载最优权重。
    model = LightCNNMelClassifier(num_classes=10).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    # 收集真实标签与预测标签，用于后续统计指标。
    y_true, y_pred = [], []
    with torch.no_grad():
        # 评估阶段关闭梯度计算，提高推理效率并节省显存。
        for x, y in tqdm(loader, desc="Evaluate", leave=False):
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(y.numpy().tolist())

    # 计算分类报告和混淆矩阵。
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    # 创建输出目录并写入评估结果文件。
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    np.savetxt(out_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    # 打印本次评估使用的数据集路径、设备和统计结果。
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Device: {device}")
    print(report)
    print(f"Saved report to: {out_dir / 'classification_report.txt'}")
    print(f"Saved confusion matrix to: {out_dir / 'confusion_matrix.csv'}")


if __name__ == "__main__":
    main()

