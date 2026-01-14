"""从 Hugging Face 下载语音表征 / ASR 大模型到本地目录。

当前预置的「特征表征」模型：
- WavLM Large:  microsoft/wavlm-large
- HuBERT Large: facebook/hubert-large-ll60k

预置的「ASR 微调」示例模型：
- HuBERT Large LS-960 ASR（英文）: facebook/hubert-large-ls960-ft
- Wav2Vec2 Large 960h ASR（英文）: facebook/wav2vec2-large-960h
  （注意：该 ID 本身就是 ASR 微调版）
- Wav2Vec2 Large XLSR 中文 ASR: jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn

默认下载到项目根目录下的 ``pretrained_model/`` 目录，按模型名称分别建子目录。
你也可以通过 ``--repo-id`` 自定义任意 Hugging Face 模型进行下载。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_ROOT = PROJECT_ROOT / "pretrained_model"


MODEL_CONFIGS: Dict[str, str] = {
    "wavlm_large": "microsoft/wavlm-large",
    "hubert_large": "facebook/hubert-large-ll60k",
    "wav2vec2-large-lv60": "facebook/wav2vec2-large-lv60",
    "wav2vec2-large-robust": "facebook/wav2vec2-large-robust",
    # ASR 英文微调示例
    "hubert_large_ls960_asr": "facebook/hubert-large-ls960-ft",
    "wav2vec2_large": "facebook/wav2vec2-large-960h",
    "hubert-large-ll60k": "facebook/hubert-large-ll60k",
    
    # ASR 中文微调版
    "chinese-wav2vec2-large": "TencentGameMate/chinese-wav2vec2-large",
    "chinese-hubert-large": "TencentGameMate/chinese-hubert-large",

}


def download_one(model_key: str, hf_id: str, force: bool = False) -> None:
    """下载单个 Hugging Face 模型到本地目录。

    参数
    ----
    model_key:
        本地模型目录名（如 ``wavlm_large``）。
    hf_id:
        Hugging Face 模型 ID（如 ``microsoft/wavlm-large``）。
    force:
        若目标目录已存在，是否强制重新保存。
    """
    target_dir = PRETRAINED_ROOT / model_key
    if target_dir.exists() and not force:
        # snapshot_download 会在已有目录上增量/续传，这里仅做提示，不强制清空
        print(f"[info] {target_dir} 已存在，将在该目录内增量下载（如需强制覆盖可先手动删除或使用 --force）")

    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download] 开始下载 {model_key} ({hf_id}) 到 {target_dir}")

    snapshot_download(
        repo_id=hf_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print(f"[done] {model_key} 下载完成，已保存到 {target_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 Hugging Face 下载 WavLM/HuBERT/Wav2Vec2 Large 模型"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help=(
            "自定义 Hugging Face 模型 ID，"
            "如 facebook/hubert-large-ls960-ft 或任意 WavLM/Wav2Vec2 ASR 微调模型"
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        help="本地保存目录名（仅在指定 --repo-id 时使用；默认使用 repo-id 的最后一段）",
    )
    parser.add_argument(
        "--model",
        "-m",
        choices=["all", *MODEL_CONFIGS.keys()],
        default="all",
        help="要下载的模型（默认: all）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="若目标目录已存在，仍然强制重新保存模型文件",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    PRETRAINED_ROOT.mkdir(parents=True, exist_ok=True)

    # 若显式指定 repo-id，则优先使用自定义模型（适合下载任意 ASR 微调版）
    if args.repo_id:
        local_name = args.name or args.repo_id.split("/")[-1]
        download_one(local_name, args.repo_id, force=bool(args.force))
        return

    if args.model == "all":
        for key, hf_id in MODEL_CONFIGS.items():
            download_one(key, hf_id, force=bool(args.force))
    else:
        hf_id = MODEL_CONFIGS[args.model]
        download_one(args.model, hf_id, force=bool(args.force))


if __name__ == "__main__":
    main()
