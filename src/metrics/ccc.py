from typing import Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

def concordance_correlation_coefficient(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """计算 concordance correlation coefficient (CCC)."""
    mean_observed = np.mean(observed)
    mean_predicted = np.mean(predicted)
    # 使用与方差一致的无偏估计（ddof=1），避免混用有偏/无偏统计量
    # [0, 1] 取的是“第一个变量与第二个变量的协方差”。
    # ddof=1 表示用样本协方差（分母是 N-1）
    covariance = np.cov(observed, predicted, ddof=1)[0, 1]
    obs_variance = np.var(observed, ddof=1)
    pred_variance = np.var(predicted, ddof=1)
    ccc = 2 * covariance / (obs_variance + pred_variance + (mean_observed - mean_predicted) ** 2)
    return float(ccc)

def evaluate_ccc(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    save_details_path: Optional[str] = None,
) -> Tuple[float, float]:
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

            pred_v = pred_logits[:, 0].cpu().numpy()
            pred_a = pred_logits[:, 1].cpu().numpy()

            batch_v = batch_vad[:, 0].cpu().numpy()
            batch_a = batch_vad[:, 1].cpu().numpy()

            logits_v.append(pred_v)
            logits_a.append(pred_a)

            v.append(batch_v)
            a.append(batch_a)

    logits_v = np.concatenate(logits_v)
    logits_a = np.concatenate(logits_a)
    v = np.concatenate(v)
    a = np.concatenate(a)

    if save_details_path is not None:
        try:
            import os

            import pandas as pd

            os.makedirs(os.path.dirname(save_details_path), exist_ok=True)
            pd.DataFrame(
                {
                    "true_v": v,
                    "pred_v": logits_v,
                    "true_a": a,
                    "pred_a": logits_a,
                }
            ).to_csv(save_details_path, index=False)
        except Exception:
            pass

    ccc_v = concordance_correlation_coefficient(v, logits_v)
    ccc_a = concordance_correlation_coefficient(a, logits_a)

    return ccc_v, ccc_a

