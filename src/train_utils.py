"""训练辅助函数：包含随机种子设置、单轮训练、验证评估和模型保存。"""

import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


def seed_everything(seed: int = 42) -> None:
    """固定 Python、NumPy 和 PyTorch 的随机种子，尽量保证结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device: torch.device) -> Tuple[float, float]:
    """完成一个 epoch 的训练，并返回平均损失和准确率。"""
    model.train()
    losses = []
    y_true, y_pred = [], []

    # 使用 tqdm 展示训练进度。
    for x, y in tqdm(loader, desc="Train", leave=False):
        # 将输入与标签移动到同一设备上。
        x = x.to(device)
        y = y.to(device)

        # 清空上一轮梯度。
        optimizer.zero_grad()
        # 前向传播得到分类结果。
        logits = model(x)
        # 计算损失并反向传播。
        loss = criterion(logits, y)
        loss.backward()
        # 更新参数。
        optimizer.step()

        # 记录当前 batch 的损失、真实标签和预测标签。
        losses.append(loss.item())
        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())

    # 计算整轮训练的准确率。
    acc = accuracy_score(y_true, y_pred)
    return float(np.mean(losses)), float(acc)


def evaluate(model, loader, criterion, device: torch.device) -> Dict[str, float]:
    """在验证集或测试集上评估模型，并返回损失、准确率和宏平均 F1。"""
    model.eval()
    losses = []
    y_true, y_pred = [], []

    # 评估阶段不计算梯度，节省显存并提升速度。
    with torch.no_grad():
        # 使用 tqdm 展示评估进度。
        for x, y in tqdm(loader, desc="Eval", leave=False):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            # 记录 batch 损失以及预测结果。
            losses.append(loss.item())
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    # 返回分类任务常用指标。
    return {
        "loss": float(np.mean(losses)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def save_checkpoint(model, path: str) -> None:
    """保存模型参数到指定路径。"""
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)

