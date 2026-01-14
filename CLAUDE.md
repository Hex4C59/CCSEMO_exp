# CLAUDE.md
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CCSEMO_exp is a deep learning research project for audio emotion regression. It predicts emotional dimensions (Valence and Arousal) from speech by combining pretrained audio models (Wav2Vec2, HuBERT, WavLM) with pitch features using multi-head attention fusion.

## Common Commands

```bash
# Train a model (basic)
uv run python src/train.py --config configs/CCSEMO/with_norm.yaml

# Train with specific pretrained model
uv run python src/train.py --config configs/CCSEMO/with_norm.yaml --pm wav2vec2-large-robust

# Precompute pitch features (required before training)
uv run python scripts/precompute_pitch.py --dataset ccsemo --split all
uv run python scripts/precompute_pitch.py --dataset all --split all  # all datasets

# Create 5-fold cross-validation splits for CCSEMO
uv run python scripts/make_ccsemo_5fold_speaker_cv.py
```

## Architecture

### Core Components

```
src/
├── train.py                    # Main entry point, Trainer class with training loop
├── model/
│   └── iemocap_audio_model.py  # AudioClassifier: Wav2Vec2 + pitch fusion model
├── data/
│   └── audio_parsing_dataset.py # data_loader: audio loading, pitch extraction, batching
├── losses/
│   └── mse_loss.py             # MSE loss for regression
├── metrics/
│   ├── ccc.py                  # Concordance Correlation Coefficient (primary metric)
│   ├── mse.py                  # Mean Squared Error
│   └── r2.py                   # R-squared
└── utils/
    ├── data_loading.py         # load_label_splits(), parse_batch_for_model()
    ├── experiment.py           # Experiment directory setup, result saving
    └── hf_audio.py             # HuggingFace audio processor loading
```

### Model Architecture (AudioClassifier)

- **Backbone**: Pretrained audio models (Wav2Vec2/HuBERT/WavLM) loaded via `AutoModel`
- **Pitch embedding**: Linear or CNN projection of pitch features to hidden dimension
- **Fusion**: Multi-head attention between audio hidden states and pitch embeddings
- **Output**: Linear layer predicting V (Valence) and/or A (Arousal)

### Task Modes

- `basemodel`: Pure audio baseline (no pitch features)
- `pitch_and_wav2vec2`: Audio + pitch fusion with attention

### Data Flow

1. Labels loaded from CSV files in `data/labels/<DATASET>/`
2. Pitch features cached in `data/processed/pitch_cache_<dataset>/`
3. Audio processed by HuggingFace feature extractor
4. Model outputs V/A predictions, evaluated by CCC

## Configuration

YAML configs in `configs/<DATASET>/{with_norm,no_norm}.yaml`:

- `audio_model_name`: Path to pretrained model (e.g., `pretrained_model/wav2vec2-large-robust`)
- `text_cap_path`: Path to labels CSV
- `pitch_cache_dir`: Directory for cached pitch features
- `task`: `basemodel` or `pitch_and_wav2vec2`
- `target`: `both`, `V`, or `A`
- `embedding`: `linear` or `cnn` (pitch embedding type)

## Datasets

- **CCSEMO**: Chinese emotional speech dataset
- **IEMOCAP**: English multimodal emotion dataset
- **MSP-PODCAST**: Large-scale podcast emotion dataset

Labels must include columns: `name`, `V`, `A`, `audio_path`, and optionally `split_set`.

## Key Implementation Details

- **GPU selection**: Auto-selects free GPU via nvidia-smi query (`_find_free_visible_gpu()`)
- **Early stopping**: Based on validation CCC with configurable patience
- **Pitch caching**: Offline precomputation via Praat/Parselmouth for efficiency
- **NaN/Inf handling**: Training loop skips batches with invalid values
- **Experiment tracking**: Results saved to `runs/<dataset>/<model>/<method>/exp_N/`

## Defensive Programming Guidelines for This Project

**Follow lightweight defensive programming** - validate at boundaries, trust PyTorch internals.

### ✅ Good Practices in This Project

**1. Configuration validation at startup:**
```python
# src/train.py: Validate before data loading
if target != "both":
    raise ValueError(f"Only 'both' supported, got '{target}'")
if not os.path.exists(args.audio_model_name):
    raise FileNotFoundError(f"Model not found: {args.audio_model_name}")
```

**2. Data validation at loading time:**
```python
# src/data/audio_parsing_dataset.py: Skip bad data early
if pitch_values is None or len(pitch_values) == 0:
    print(f"Skipping {name}: empty pitch")
    continue
```

**3. Critical assertions in forward pass:**
```python
# src/model/iemocap_audio_model.py
assert last_hidden is not None, "音频骨干模型未返回最后隐藏层"
```

### ❌ Avoid Over-defensive Patterns

**Don't check PyTorch internals:**
```python
# ❌ Bad: PyTorch will report clear errors
if not isinstance(pitch_data, torch.Tensor):
    raise TypeError("...")

# ✅ Good: Let PyTorch handle it
pitch_embeds = self.pitch_embedding(pitch_data)
```

**Don't validate impossible cases:**
```python
# ❌ Bad: Training loop guarantees epoch >= 0
if epoch < 0:
    raise ValueError("Invalid epoch")

# ✅ Good: Trust the loop
for epoch in range(num_epochs):
    train_one_epoch(epoch)
```

**Principle: Fail fast at boundaries (config/data), trust everything else.**
