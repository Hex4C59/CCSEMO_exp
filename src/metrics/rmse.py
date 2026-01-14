from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from utils.data_loading import parse_batch_for_model
from .mse import mean_squared_error


def root_mean_squared_error(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """计算均方根误差 (RMSE)."""
    mse = mean_squared_error(observed, predicted)
    rmse = np.sqrt(mse)
    return float(rmse)


def evaluate_rmse(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    target: str = "both",
    save_details_path: Optional[str] = None,
) -> Tuple[float, float]:
    """在一个 dataloader 上计算 RMSE.

    - target="both": 同时返回 (rmse_V, rmse_A)；
    - target="V": 仅计算 V 的指标，返回 (rmse_V, 0.0)；
    - target="A": 仅计算 A 的指标，返回 (0.0, rmse_A)。

    Returns:
        Tuple[float, float]: (rmse_v, rmse_a)
    """
    model.eval()

    logits_v, logits_a = [], []
    v, a = [], []

    with torch.no_grad():
        for _, data in enumerate(tqdm(dataloader, mininterval=10)):
            batch_audio, batch_vad = data

            pitch_audio, batch_audio, batch_vad = parse_batch_for_model(
                batch_audio,
                batch_vad,
                device,
            )

            pred_logits = model(
                pitch_audio,
                batch_audio,
            )

            if pred_logits is None:
                continue

            if target == "V":
                # 单输出或多输出均 squeeze 到一维
                pred_v = pred_logits.squeeze(-1).cpu().numpy()
                batch_v = batch_vad[:, 0].cpu().numpy()

                logits_v.append(pred_v)
                v.append(batch_v)

            elif target == "A":
                pred_a = pred_logits.squeeze(-1).cpu().numpy()
                batch_a = batch_vad[:, 1].cpu().numpy()

                logits_a.append(pred_a)
                a.append(batch_a)

            else:  # both
                pred_v = pred_logits[:, 0].cpu().numpy()
                pred_a = pred_logits[:, 1].cpu().numpy()

                batch_v = batch_vad[:, 0].cpu().numpy()
                batch_a = batch_vad[:, 1].cpu().numpy()

                logits_v.append(pred_v)
                logits_a.append(pred_a)

                v.append(batch_v)
                a.append(batch_a)

    if target == "V":
        if not logits_v:
            # 没有任何样本时，直接返回 0.0，避免 concatenate 报错
            return 0.0, 0.0

        logits_v = np.concatenate(logits_v)
        v = np.concatenate(v)
        rmse_v = root_mean_squared_error(v, logits_v)

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
                    }
                )
                df.to_csv(save_details_path, index=False)
            except Exception:
                # 不影响主流程；写文件失败时忽略
                pass

        return rmse_v, 0.0

    if target == "A":
        if not logits_a:
            return 0.0, 0.0

        logits_a = np.concatenate(logits_a)
        a = np.concatenate(a)
        rmse_a = root_mean_squared_error(a, logits_a)

        if save_details_path is not None:
            try:
                import os

                import pandas as pd

                os.makedirs(os.path.dirname(save_details_path), exist_ok=True)
                df = pd.DataFrame(
                    {
                        "true_a": a,
                        "pred_a": logits_a,
                        "squared_error_a": (a - logits_a) ** 2,
                    }
                )
                df.to_csv(save_details_path, index=False)
            except Exception:
                pass

        return 0.0, rmse_a

    if not logits_v or not logits_a:
        return 0.0, 0.0

    logits_v = np.concatenate(logits_v)
    logits_a = np.concatenate(logits_a)
    v = np.concatenate(v)
    a = np.concatenate(a)

    rmse_v = root_mean_squared_error(v, logits_v)
    rmse_a = root_mean_squared_error(a, logits_a)

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

    return rmse_v, rmse_a
