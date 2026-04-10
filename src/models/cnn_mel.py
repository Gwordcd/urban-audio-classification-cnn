"""基于 Log-Mel 频谱的轻量级卷积神经网络模型。"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """卷积块：由两层卷积、归一化、激活函数和池化层组成。"""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            # 第一层卷积，用于提取局部时频特征。
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # 批归一化，稳定训练过程。
            nn.BatchNorm2d(out_channels),
            # ReLU 激活，引入非线性。
            nn.ReLU(inplace=True),
            # 第二层卷积，进一步提炼特征。
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 最大池化，下采样并保留显著响应。
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，直接执行卷积块内部的顺序计算。"""
        return self.block(x)


class LightCNNMelClassifier(nn.Module):
    """用于音频分类的轻量 CNN 模型。"""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # 三个卷积块逐层提取更高层次的时频特征。
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        # 自适应平均池化将特征图压缩为固定尺寸，便于后续分类。
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # 分类头：展平、随机失活和线性输出层。
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：先提取特征，再压缩，再输出类别得分。"""
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

