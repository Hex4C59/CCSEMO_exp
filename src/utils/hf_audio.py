"""HuggingFace 音频预处理装载工具。

根据配置的 `audio_model_name` 自动装载合适的特征提取器（feature_extractor），
并在本地存在可识别的 tokenizer 文件时，尝试装载对应 tokenizer。

目的：
- 对 HuBERT/WavLM/Wav2Vec2 等模型自适应，无需在配置里区分具体类名。
- 优先只加载特征提取器，避免 AutoProcessor 触发 tokenizer 依赖（例如 protobuf）。
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import FeatureExtractionMixin, PreTrainedTokenizerBase


def _has_any_file(directory: str, candidates: tuple[str, ...]) -> bool:
    try:
        return any(os.path.exists(os.path.join(directory, fname)) for fname in candidates)
    except Exception:
        return False


def _should_try_tokenizer(path_or_id: str) -> bool:
    """仅在本地目录且包含常见 tokenizer 文件时才尝试加载 tokenizer。"""
    if not os.path.isdir(path_or_id):
        return False
    return _has_any_file(
        path_or_id,
        (
            "tokenizer.json",
            "vocab.json",
            "spm.model",
            "merges.txt",
        ),
    )


def load_feature_extractor_and_tokenizer(
    name_for_feature_extractor: str,
    name_for_tokenizer: Optional[str] = None,
) -> Tuple["FeatureExtractionMixin", Optional["PreTrainedTokenizerBase"]]:
    """加载音频特征提取器与（可选）tokenizer。

    - 始终优先加载 AutoFeatureExtractor（避免 tokenizer 额外依赖）。
    - 仅当本地目录中检测到典型 tokenizer 文件时，才尝试加载 AutoTokenizer；否则返回 None。
    """
    from transformers import AutoFeatureExtractor  # 延迟导入

    # 1) 加载特征提取器
    feature_extractor = AutoFeatureExtractor.from_pretrained(name_for_feature_extractor)

    # 2) 视情况加载 tokenizer（可选）
    tokenizer = None
    tok_name = name_for_tokenizer or name_for_feature_extractor
    if _should_try_tokenizer(tok_name):
        try:
            from transformers import AutoTokenizer  # 延迟导入

            tokenizer = AutoTokenizer.from_pretrained(tok_name)
        except Exception:
            tokenizer = None

    return feature_extractor, tokenizer

