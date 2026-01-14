
import os
from typing import Dict, Tuple

import pandas as pd


def load_label_splits(text_cap_path: str) -> Dict[str, pd.DataFrame]:
    """读取 train/val/test 的标签 DataFrame, 兼容多种 text_cap_path 配置."""
    tcp = text_cap_path

    if os.path.isdir(tcp):
        labels_path = os.path.join(tcp, "labels.csv")
        if os.path.exists(labels_path):
            data = pd.read_csv(labels_path)
            if "split_set" not in data.columns:
                raise ValueError(
                    "labels.csv 中缺少 split_set 列, 请确保包含 "
                    "audio_path,name,V,A,gender,duration,discrete_emotion,"
                    "split_set 等列。"
                )
            train = data[data["split_set"] == "train"].reset_index(drop=True)
            val = data[data["split_set"] == "val"].reset_index(drop=True)
            test = data[data["split_set"] == "test"].reset_index(drop=True)
        else:
            train = pd.read_csv(os.path.join(tcp, "train.csv"))
            val = pd.read_csv(os.path.join(tcp, "val.csv"))
            test = pd.read_csv(os.path.join(tcp, "test.csv"))
    else:
        # 单文件模式: 统一要求包含 split_set 列, 不再根据 Sesxx 规则划分。
        data = pd.read_csv(tcp)
        if "split_set" not in data.columns:
            raise ValueError(
                "当 text_cap_path 为单个文件时, 该文件必须包含 split_set 列, "
                "用于指示 train/val/test 划分。"
            )

        train = data[data["split_set"] == "train"].reset_index(drop=True)
        val = data[data["split_set"] == "val"].reset_index(drop=True)
        test = data[data["split_set"] == "test"].reset_index(drop=True)

    return {"train": train, "val": val, "test": test}




