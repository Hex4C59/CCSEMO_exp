# CCSEMO_exp

基于预训练音频模型和音高特征融合的语音情感回归项目。

## 项目简介

本项目用于从语音中预测情感维度（Valence 效价 和 Arousal 唤醒度）。通过结合预训练音频模型（Wav2Vec2、HuBERT、WavLM）与音高特征，使用多头注意力机制进行特征融合，实现情感连续值预测。

**支持数据集：**
- CCSEMO（中文情感语音）
- IEMOCAP（英文多模态情感）
- MSP-PODCAST（大规模播客情感）

**主要评价指标：** CCC (Concordance Correlation Coefficient)

---

## 环境配置

### 1. Python 环境

本项目使用 `uv` 进行依赖管理，需要 Python 3.12+。

```bash
# 安装依赖
uv sync

# 或使用 pip（不推荐）
pip install -e .
```

### 2. 预训练模型准备

将预训练模型放在 `pretrained_model/` 目录下：

```
pretrained_model/
├── wav2vec2-large-robust/
├── hubert-large/
└── wavlm-large/
```

可从 HuggingFace 下载对应模型。

---

## 数据准备

### 1. 标签文件格式

标签文件存放在 `data/labels/<DATASET>/` 目录，格式为 CSV，必须包含以下列：

| 列名 | 说明 |
|------|------|
| `name` | 样本唯一标识 |
| `audio_path` | 音频文件路径 |
| `V` | Valence（效价）标签 |
| `A` | Arousal（唤醒度）标签 |
| `split_set` | 数据集划分（train/dev/test） |

### 2. 音高特征预计算

训练前需要预计算音高特征（使用 Praat/Parselmouth）：

```bash
# 为 CCSEMO 数据集预计算所有划分的音高特征
uv run python scripts/precompute_pitch.py --dataset ccsemo --split all

# 为所有数据集预计算
uv run python scripts/precompute_pitch.py --dataset all --split all
```

音高特征缓存在 `data/processed/pitch_cache_<dataset>/` 目录。

### 3. 生成交叉验证划分（可选）

```bash
# 为 CCSEMO 创建 5 折说话人独立交叉验证
uv run python scripts/make_ccsemo_5fold_speaker_cv.py
```

---

## 训练模型

### 基本训练命令

```bash
# 使用默认配置训练
uv run python src/train.py --config configs/CCSEMO/with_norm.yaml

# 指定预训练模型
uv run python src/train.py \
    --config configs/CCSEMO/with_norm.yaml \
    --pm wav2vec2-large-robust

# 可选预训练模型：
# - wav2vec2-large-robust
# - hubert-large
# - wavlm-large
```

### 配置文件说明

配置文件位于 `configs/<DATASET>/{with_norm,no_norm}.yaml`，主要参数：

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `audio_model_name` | 预训练模型路径 | 见上方预训练模型列表 |
| `task` | 任务模式 | `basemodel`（纯音频）<br>`pitch_and_wav2vec2`（音频+音高） |
| `target` | 预测目标 | `both`（V+A）, `V`, `A` |
| `embedding` | 音高嵌入方式 | `linear`, `cnn` |
| `text_cap_path` | 标签文件路径 | CSV 文件路径 |
| `pitch_cache_dir` | 音高缓存目录 | 默认 `data/processed/pitch_cache_<dataset>` |

---

## 项目结构

```
CCSEMO_exp/
├── configs/                    # 配置文件
│   ├── CCSEMO/
│   ├── IEMOCAP/
│   └── MSP-PODCAST/
├── data/
│   ├── labels/                 # 标签 CSV 文件
│   └── processed/              # 预处理数据缓存
├── pretrained_model/           # 预训练模型
├── scripts/                    # 数据处理脚本
│   ├── precompute_pitch.py     # 音高预计算
│   └── make_ccsemo_5fold_speaker_cv.py
├── src/
│   ├── train.py                # 训练入口
│   ├── model/
│   │   └── iemocap_audio_model.py  # AudioClassifier 模型
│   ├── data/
│   │   └── audio_parsing_dataset.py  # 数据加载器
│   ├── losses/                 # 损失函数
│   ├── metrics/                # 评价指标（CCC, MSE, R²）
│   └── utils/                  # 工具函数
└── runs/                       # 实验结果输出
    └── <dataset>/<model>/<method>/exp_N/
```

---

## 模型架构

**AudioClassifier** 核心组件：

1. **音频编码器：** 预训练模型（Wav2Vec2/HuBERT/WavLM）提取音频特征
2. **音高嵌入：** Linear 或 CNN 将音高特征投影到隐藏维度
3. **多头注意力融合：** 音频与音高特征交互
4. **输出层：** 线性层预测 V/A 连续值

---

## 实验结果

训练结果保存在 `runs/<dataset>/<model>/<method>/exp_N/` 目录：

- `config.yaml` - 实验配置
- `train_log.txt` - 训练日志
- `best_model.pth` - 最佳模型权重
- `metrics.json` - 评价指标

---

