from typing import Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


def mean_squared_error(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """计算均方误差 (MSE)."""
    mse = np.mean((observed - predicted) ** 2)
    return float(mse)


def evaluate_mse(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    target: str = "both",
    save_details_path: Optional[str] = None,
) -> Tuple[float, float]:
    """在一个 dataloader 上计算 MSE.

    仅支持 target="both"，返回 (mse_V, mse_A)。

    Returns:
        Tuple[float, float]: (mse_v, mse_a)
    """
    if target != "both":
        raise ValueError("Only target='both' is supported for MSE evaluation.")

    model.eval()
    # 保存模型对V和A的预测值
    logits_v, logits_a = [], []
    # 保存模型对V和A的真实值
    v, a = [], []

    with torch.no_grad():
        for _, data in enumerate(tqdm(dataloader, mininterval=10)):
            batch_audio, batch_vad = data

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

    mse_v = mean_squared_error(v, logits_v)
    mse_a = mean_squared_error(a, logits_a)

    if save_details_path is not None:
        try:
            import os

            import pandas as pd

            os.makedirs(os.path.dirname(save_details_path), exist_ok=True)
            df = pd.DataFrame(
                {
                    "true_v": v,
                    "pred_v": logits_v,
                    "squared_error_v": (v - logits_v) ** 2,
                    "true_a": a,
                    "pred_a": logits_a,
                    "squared_error_a": (a - logits_a) ** 2,
                }
            )
            df.to_csv(save_details_path, index=False)
        except Exception:
            pass

    return mse_v, mse_a
