from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


def r2_score(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """计算决定系数 R^2.

    R^2 = 1 - SS_res / SS_tot
    SS_res = sum((y - y_hat)^2)
    SS_tot = sum((y - mean(y))^2)
    """
    if observed.size == 0:
        return 0.0

    ss_res = np.sum((observed - predicted) ** 2)
    mean_obs = np.mean(observed)
    ss_tot = np.sum((observed - mean_obs) ** 2)

    if ss_tot == 0:
        # 标签完全一致时 R^2 不好定义，这里返回 0.0 避免除零
        return 0.0

    r2 = 1.0 - ss_res / ss_tot
    return float(r2)


def evaluate_r2(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    target: str = "both",
    save_details_path: Optional[str] = None,
) -> Tuple[float, float]:
    """在一个 dataloader 上计算 R^2.

    仅支持 target="both"，返回 (r2_V, r2_A)。

    Returns:
        Tuple[float, float]: (r2_v, r2_a)
    """
    if target != "both":
        raise ValueError("Only target='both' is supported for R2 evaluation.")

    model.eval()

    logits_v, logits_a = [], []
    v, a = [], []

    with torch.no_grad():
        for _, data in enumerate(tqdm(dataloader, mininterval=10)):
            _, batch_audio, batch_vad = data

            pitch_data = batch_audio[0].to(device)
            audio_data = batch_audio[1].to(device)
            pitch_mask = batch_audio[2].to(device)
            audio_mask = batch_audio[3].to(device)
            batch_vad = batch_vad.to(device)

            pred_logits = model(pitch_data, audio_data, pitch_mask, audio_mask)

            if pred_logits is None:
                continue

            pred_v = pred_logits[:, 0].cpu().numpy()
            pred_a = pred_logits[:, 1].cpu().numpy()

            batch_v = batch_vad[:, 0].cpu().numpy()
            batch_a = batch_vad[:, 1].cpu().numpy()

            logits_v.append(pred_v)
            logits_a.append(pred_a)

            v.append(batch_v)
            a.append(batch_a)

    if not logits_v or not logits_a:
        return 0.0, 0.0

    logits_v = np.concatenate(logits_v)
    logits_a = np.concatenate(logits_a)
    v = np.concatenate(v)
    a = np.concatenate(a)

    r2_v = r2_score(v, logits_v)
    r2_a = r2_score(a, logits_a)

    if save_details_path is not None:
        try:
            import os

            import pandas as pd

            os.makedirs(os.path.dirname(save_details_path), exist_ok=True)
            df = pd.DataFrame(
                {
                    "true_v": v,
                    "pred_v": logits_v,
                    "true_a": a,
                    "pred_a": logits_a,
                }
            )
            df.to_csv(save_details_path, index=False)
        except Exception:
            pass

    return r2_v, r2_a
