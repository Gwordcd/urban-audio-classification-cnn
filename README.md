# UrbanSound8K 分类（课程实验）

本项目是一个可直接运行的 UrbanSound8K 十分类基线实现：

- 输入特征：Log-Mel 频谱
- 模型：轻量 CNN（`src/models/cnn_mel.py`）
- 任务：10 类环境声音分类

## 1) 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python train.py --device cuda
python evaluate.py --device cuda
```

如果当前机器没有可用 CUDA，脚本会自动回退到 CPU。

## 2) 数据准备

默认数据目录（见 `src/config.py` 的 `DATASET_ROOT`）：

`src/data/UrbanSound8K`

目录应包含：

```text
src/data/UrbanSound8K/
├─ audio/fold1 ... fold10
└─ metadata/UrbanSound8K.csv
```
数据集下载：http://10.201.201.213/#s/_aLn_dIA

## 3) 环境与依赖

- 推荐 Python 3.9+
- 依赖在 `requirements.txt`
- Windows PowerShell 激活虚拟环境命令：

```powershell
.\.venv\Scripts\Activate.ps1
```

## 4) 训练

默认训练划分：fold1-8 训练，fold9 验证，fold10 测试。

```powershell
python train.py
python train.py --device cpu
python train.py --device cuda
```

训练最优权重保存到：`outputs/best.pth`（见 `src/config.py` 的 `CHECKPOINT_PATH`）。

## 5) 评估

```powershell
python evaluate.py
python evaluate.py --device cpu
python evaluate.py --device cuda
```

评估输出：

- `outputs/classification_report.txt`
- `outputs/confusion_matrix.csv`

## 6) 关键文件说明

- `train.py`：训练入口
- `evaluate.py`：测试与报告导出入口
- `src/config.py`：路径与超参数配置
- `src/data/urbansound8k.py`：数据读取与 Mel 特征处理
- `src/models/cnn_mel.py`：模型定义
- `src/train_utils.py`：训练/验证辅助函数

## 7) 常见问题

- `Checkpoint not found`：先运行 `python train.py` 生成 `outputs/best.pth`
- 设备显示 CPU：检查 CUDA 是否可用，或改用 `--device cpu`
- 路径错误：确认 `src/data/UrbanSound8K/metadata/UrbanSound8K.csv` 存在
