"""离线预计算并缓存每条音频的 pitch 序列（与模型无关、可复用）。

功能
----
- 读取与训练相同的配置与数据拆分（train/val/test）。
- 使用 Praat/Parselmouth 提取每条音频的原始 pitch（未归一化）。
- 将结果保存到与数据加载器一致的缓存目录：
  data/processed/pitch_cache_<dataset>/<stem>.npy

参数
----
- 所有路径相关参数直接在脚本顶部写死，可按需修改：
  * DATASET_PRESETS (各数据集的标签路径与缓存目录)
- 运行时支持少量命令行参数：
  * --dataset: iemocap|ccsemo|msp|all，默认 iemocap。
  * --split: train|val|test|all，默认 all。
  * --force: 存在同名缓存时是否强制重算并覆盖。

使用示例
--------
仅提取 IEMOCAP 全部 split:
    uv run python scripts/precompute_pitch.py --dataset iemocap --split all

一次性提取三个数据集的全部 split:
    uv run python scripts/precompute_pitch.py --dataset all --split all
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
import shutil
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from tqdm import tqdm

# 复用现有的数据加载逻辑, 保证与训练阶段的 split 行为一致
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.data_loading import load_label_splits  # noqa: E402


class PitchFeatures:
    """从 single utterance 中提取原始 pitch 轨迹的辅助类.

    这里保留与原 audio_parsing_dataset.py 中相同的实现,
    确保离线提取与旧版在线提取逻辑一致。
    """

    def __init__(self, sound, temp_dir: str = "./tmp_pitch") -> None:
        self.sound = sound
        self.pitch_tiers = None
        self.total_duration = None
        self.pitch_point: list[float] = []
        self.time_point: list[float] = []

        self.pd: list[float] = []
        self.pt: list[float] = []
        self.ps: list[float | None] = []
        self.pr: list[float] = []

        os.makedirs(temp_dir, exist_ok=True)
        self.temp_dir = temp_dir
        self.temp_file = os.path.join(temp_dir, "PitchTier")

    def get_pitch_tiers(self):
        manipulation = call(self.sound, "To Manipulation", 0.01, 75, 600)
        self.pitch_tier = call(manipulation, "Extract pitch tier")
        return self.pitch_tier

    def stylize_pitch(self):
        if self.pitch_tier is not None:
            call(self.pitch_tier, "Stylize...", 2.0, "semitones")
            tmp_pitch_point = self.pitch_point
            tmp_time_point = self.time_point
            self.set_time_and_pitch_point()
            if len(self.pitch_point) == 0:
                self.pitch_point = tmp_pitch_point
                self.time_point = tmp_time_point
        else:
            print("pitch_tier is None")
            return

    def set_total_duration(self):
        total_duration_match = re.search(
            r"Total duration: (\d+(\.\d+)?) seconds",
            str(self.pitch_tier),
        )
        if total_duration_match:
            self.total_duration = float(total_duration_match.group(1))
        else:
            print("Total duration not found.")

    def set_time_and_pitch_point(self):
        self.pitch_tier.save(self.temp_file)
        r_file = open(self.temp_file, "r")

        self.pitch_point = []
        self.time_point = []
        while True:
            line = r_file.readline()
            if not line:
                break

            if "number" in line:
                value = re.sub(r"[^0-9^.]", "", line)

                if value.count(".") > 1:
                    parts = value.split(".")
                    value = parts[0] + "".join(parts[1:])
                if value != "":
                    self.time_point.append(round(float(value), 4))
            elif "value" in line:
                value = re.sub(r"[^0-9^.]", "", line)

                if value.count(".") > 1:
                    parts = value.split(".")
                    value = parts[0] + "".join(parts[1:])
                if value != "":
                    self.pitch_point.append(round(float(value), 4))

        if len(self.pitch_point) == 0:
            while True:
                line = r_file.readline()
                if not line:
                    break
        r_file.close()

    def get_pitchs(self):
        """返回 (pitch_values, time_points) 序列, 与旧实现保持一致."""
        self.pitch_tiers = self.get_pitch_tiers()
        self.set_time_and_pitch_point()
        return (self.pitch_point, self.time_point)


# ===== 在这里修改默认参数 / 数据集预设 =====
# 默认数据集名称; 可通过命令行 --dataset 修改
DEFAULT_DATASET = "iemocap"

# 各数据集的默认标签路径 / 音频根目录 / pitch 缓存目录
DATASET_PRESETS = {
    # IEMOCAP: 使用任意一个 fold*.csv 作为全集标签来源, 仅依赖 audio_path 列.
    "iemocap": {
        "text_cap_path": "data/labels/IEMOCAP/fold1.csv",
        "audio_root": "",
        "pitch_cache_dir": "data/processed/pitch_cache_iemocap",
    },
    # CCSEMO: 使用单一 labels.csv, 已包含 audio_path + split_set.
    "ccsemo": {
        "text_cap_path": "data/labels/CCSEMO/labels.csv",
        "audio_root": "",
        "pitch_cache_dir": "data/processed/pitch_cache_ccsemo",
    },
    # MSP-PODCAST: 使用 MSP-PODCAST 标签; 推荐单独的缓存目录.
    "msp": {
        "text_cap_path": "data/labels/MSP-PODCAST/labels.csv",
        "audio_root": "",
        "pitch_cache_dir": "data/processed/pitch_cache_msp_podcast",
    },
}

# 为兼容旧调用方式保留一套默认参数 (等价于 IEMOCAP 预设)
DEFAULT_TEXT_CAP_PATH = DATASET_PRESETS[DEFAULT_DATASET]["text_cap_path"]
DEFAULT_AUDIO_ROOT = DATASET_PRESETS[DEFAULT_DATASET]["audio_root"]
DEFAULT_PITCH_CACHE_DIR = DATASET_PRESETS[DEFAULT_DATASET]["pitch_cache_dir"]

# 是否启用缓存开关（当前脚本总是会写缓存，此开关仅用于打印提示）
DEFAULT_PITCH_CACHE_ENABLED = True
# ===== 默认参数结束 =====


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预计算 pitch 缓存")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        choices=sorted(list(DATASET_PRESETS.keys()) + ["all"]),
        help=(
            "选择要处理的数据集, 影响标签文件路径与 pitch 缓存目录. "
            "可选: " + ", ".join(sorted(list(DATASET_PRESETS.keys()) + ["all"]))
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test", "all"],
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_wav_path(audio_root: str, split: str, name: str) -> str:
    filename = name if name.endswith(".wav") else f"{name}.wav"
    return os.path.join(audio_root, split, filename)


def cache_paths(cache_root: str, split: str, name: str) -> str:
    # 不再按 train/val/test 子目录划分, 统一缓存在同一数据集根目录下.
    # 保留 split 参数仅为兼容旧调用签名.
    os.makedirs(cache_root, exist_ok=True)
    filename = name if name.endswith(".wav") else f"{name}.wav"
    stem = os.path.splitext(os.path.basename(filename))[0]
    return os.path.join(cache_root, f"{stem}.npy")


def compute_pitch_array(wav_path: str, base_temp_dir: str) -> np.ndarray | None:
    """从单条 wav 提取 pitch 序列, 在独立临时目录中运行 Praat.

    为避免长期复用同一个临时文件导致的文件系统异常，这里为每条样本
    创建一个独立的子目录, 用后立即删除。
    """
    # 为当前样本创建独立临时目录
    os.makedirs(base_temp_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="pitch_", dir=base_temp_dir)
    try:
        try:
            sound = parselmouth.Sound(wav_path)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] Praat 打开失败: {wav_path}, err={e}")
            return None

        intonation = PitchFeatures(sound, temp_dir=temp_dir)
        pitch_tiers, time_points = intonation.get_pitchs()
        if not time_points or not pitch_tiers:
            # 提取不到有效 pitch 时，使用长度为 1 的全零向量占位，
            # 这样仍然会写入缓存并在训练阶段保留该样本。
            return np.zeros((1,), dtype=np.float32)
        return np.asarray(pitch_tiers, dtype=np.float32)
    finally:
        # 清理当前样本的临时目录, 避免堆积大量 PitchTier 文件
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


def run_one_split(
    split: str,
    df: pd.DataFrame,
    args: argparse.Namespace,
    force: bool = False,
) -> Tuple[int, int]:
    # 若 DataFrame 中自带 audio_path 列，则优先使用该列；
    # audio_root 仅作为兼容旧配置的回退。
    audio_root = getattr(args, "audio_path", "")
    cache_root = getattr(
        args,
        "pitch_cache_dir",
        "data/processed/pitch_cache_ccsemo",
    )
    done = 0
    total = len(df)

    # 每次调用各自的临时目录，避免多进程/多次调用冲突
    base_temp_dir = f"./tmp_pitch_precompute_{split}"
    os.makedirs(base_temp_dir, exist_ok=True)

    has_audio_path = "audio_path" in df.columns

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{split}"):
        name = row["name"]
        if has_audio_path and isinstance(row["audio_path"], str) and row["audio_path"]:
            wav = row["audio_path"]
        else:
            wav = build_wav_path(audio_root, split, name)
        out_path = cache_paths(cache_root, split, name)

        if (not force) and os.path.exists(out_path):
            done += 1
            continue

        pitch_arr = compute_pitch_array(wav, base_temp_dir=base_temp_dir)
        # 仅在完全无法打开音频等异常情况下返回 None；占位向量也会被写入缓存。
        if pitch_arr is None:
            continue
        try:
            np.save(out_path, pitch_arr)
            done += 1
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 无法保存: {out_path}, err={e}")

    return done, total


def main() -> None:
    cli = parse_args()

    # 确定需要处理的数据集列表
    if cli.dataset.lower() == "all":
        datasets = sorted(DATASET_PRESETS.keys())
    else:
        datasets = [cli.dataset.lower()]

    for dataset in datasets:
        if dataset not in DATASET_PRESETS:
            raise ValueError(
                f"Unsupported dataset '{dataset}'. "
                f"Supported: {', '.join(sorted(DATASET_PRESETS.keys()))} 或 all"
            )
        preset = DATASET_PRESETS[dataset]

        print(
            f"\n[dataset] {dataset} | labels={preset['text_cap_path']} | "
            f"cache_dir={preset['pitch_cache_dir']}"
        )

        # 将预设封装成与训练时相同字段名的 Namespace，
        # 以复用 load_label_splits / 数据加载器中的逻辑。
        args = argparse.Namespace(
            text_cap_path=preset["text_cap_path"],
            audio_path=preset.get("audio_root", ""),
            pitch_cache_dir=preset["pitch_cache_dir"],
            pitch_cache=DEFAULT_PITCH_CACHE_ENABLED,
        )

        if not getattr(args, "pitch_cache", True):
            print(
                "[info] 配置中关闭了 pitch_cache，但仍然执行离线预计算"
                "（不会影响训练时读取缓存）"
            )

        splits = load_label_splits(args.text_cap_path)
        todo = [cli.split] if cli.split != "all" else ["train", "val", "test"]

        total_done = 0
        total_all = 0
        for sp in todo:
            done, alln = run_one_split(sp, splits[sp], args, force=bool(cli.force))
            print(f"[{dataset}] [split {sp}] cached: {done}/{alln}")
            total_done += done
            total_all += alln

        print(f"[{dataset}] [summary] cached: {total_done}/{total_all}")


if __name__ == "__main__":
    main()
