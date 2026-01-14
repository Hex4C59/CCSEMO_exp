# CCSEMO_exp

Speech Emotion Regression via Pretrained Audio Models and Pitch Feature Fusion.

## Overview

This project predicts continuous emotional dimensions (Valence and Arousal) from speech by combining pretrained audio models (Wav2Vec2, HuBERT, WavLM) with pitch features using multi-head attention fusion.

**Supported Datasets:**
- CCSEMO (Chinese emotional speech)
- IEMOCAP (English multimodal emotion)
- MSP-PODCAST (Large-scale podcast emotion)

**Primary Metric:** CCC (Concordance Correlation Coefficient)

---

## Environment Setup

### 1. Python Environment

This project uses `uv` for dependency management and requires Python 3.12+.

```bash
# Install dependencies
uv sync

# Or use pip (not recommended)
pip install -e .
```

### 2. Pretrained Models

Place pretrained models in the `pretrained_model/` directory:

```
pretrained_model/
├── wav2vec2-large-robust/
├── hubert-large/
└── wavlm-large/
```

Download models from HuggingFace.

---

## Data Preparation

### 1. Label File Format

Label files are stored in `data/labels/<DATASET>/` as CSV files with the following columns:

| Column | Description |
|--------|-------------|
| `name` | Unique sample identifier |
| `audio_path` | Path to audio file |
| `V` | Valence label |
| `A` | Arousal label |
| `split_set` | Dataset split (train/dev/test) |

### 2. Precompute Pitch Features

Pitch features must be precomputed before training (using Praat/Parselmouth):

```bash
# Precompute pitch for CCSEMO dataset (all splits)
uv run python scripts/precompute_pitch.py --dataset ccsemo --split all

# Precompute for all datasets
uv run python scripts/precompute_pitch.py --dataset all --split all
```

Pitch features are cached in `data/processed/pitch_cache_<dataset>/`.

### 3. Generate Cross-Validation Splits (Optional)

```bash
# Create 5-fold speaker-independent CV for CCSEMO
uv run python scripts/make_ccsemo_5fold_speaker_cv.py
```

---

## Training

### Basic Training Commands

```bash
# Train with default config
uv run python src/train.py --config configs/CCSEMO/with_norm.yaml

# Specify pretrained model
uv run python src/train.py \
    --config configs/CCSEMO/with_norm.yaml \
    --pm wav2vec2-large-robust

# Available pretrained models:
# - wav2vec2-large-robust
# - hubert-large
# - wavlm-large
```

### Configuration Parameters

Config files are located in `configs/<DATASET>/{with_norm,no_norm}.yaml`. Key parameters:

| Parameter | Description | Options |
|-----------|-------------|---------|
| `audio_model_name` | Path to pretrained model | See above model list |
| `task` | Task mode | `basemodel` (audio only)<br>`pitch_and_wav2vec2` (audio + pitch) |
| `target` | Prediction target | `both` (V+A), `V`, `A` |
| `embedding` | Pitch embedding type | `linear`, `cnn` |
| `text_cap_path` | Path to label CSV | CSV file path |
| `pitch_cache_dir` | Pitch cache directory | Default: `data/processed/pitch_cache_<dataset>` |

---

## Project Structure

```
CCSEMO_exp/
├── configs/                    # Configuration files
│   ├── CCSEMO/
│   ├── IEMOCAP/
│   └── MSP-PODCAST/
├── data/
│   ├── labels/                 # Label CSV files
│   └── processed/              # Preprocessed data cache
├── pretrained_model/           # Pretrained models
├── scripts/                    # Data processing scripts
│   ├── precompute_pitch.py     # Pitch precomputation
│   └── make_ccsemo_5fold_speaker_cv.py
├── src/
│   ├── train.py                # Training entry point
│   ├── model/
│   │   └── iemocap_audio_model.py  # AudioClassifier model
│   ├── data/
│   │   └── audio_parsing_dataset.py  # Data loader
│   ├── losses/                 # Loss functions
│   ├── metrics/                # Metrics (CCC, MSE, R²)
│   └── utils/                  # Utility functions
└── runs/                       # Experiment outputs
    └── <dataset>/<model>/<method>/exp_N/
```

---

## Model Architecture

**AudioClassifier** core components:

1. **Audio Encoder:** Pretrained model (Wav2Vec2/HuBERT/WavLM) extracts audio features
2. **Pitch Embedding:** Linear or CNN projects pitch features to hidden dimension
3. **Multi-Head Attention Fusion:** Audio and pitch feature interaction
4. **Output Layer:** Linear layer predicts continuous V/A values

---

## Experiment Results

Training results are saved in `runs/<dataset>/<model>/<method>/exp_N/`:

- `config.yaml` - Experiment configuration
- `train_log.txt` - Training logs
- `best_model.pth` - Best model checkpoint
- `metrics.json` - Evaluation metrics

