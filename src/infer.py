"""推理脚本：使用训练好的权重进行预测，输出带样本名的结果."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import data_loader
from metrics.ccc import concordance_correlation_coefficient
from model import AudioClassifier
from utils.data_loading import load_label_splits


def main():
    parser = argparse.ArgumentParser(description="音频情感回归推理脚本")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--checkpoint", required=True, help="模型权重路径 (.bin)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="使用哪个数据划分进行推理")
    parser.add_argument("--output", default=None, help="输出 CSV 路径，默认为 checkpoint 同目录下的 test_predictions.csv")
    parser.add_argument("--batch_size", type=int, default=None, help="覆盖配置中的 batch_size")
    parser.add_argument("--cuda_device", type=int, default=None, help="指定 GPU 设备")
    args = parser.parse_args()

    # 加载配置
    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = argparse.Namespace(**cfg_dict)

    # 修正标签文件路径（兼容旧配置）
    if hasattr(cfg, "text_cap_path"):
        label_path = cfg.text_cap_path
        if not os.path.exists(label_path):
            # 尝试在 5fold 子目录下查找
            alt_path = label_path.replace("CCSEMO/fold", "CCSEMO/5fold/fold")
            if os.path.exists(alt_path):
                cfg.text_cap_path = alt_path
                print(f"[info] 修正标签路径: {label_path} -> {alt_path}")

    # 确定输出路径
    if args.output is None:
        output_path = os.path.join(os.path.dirname(args.checkpoint), "test_predictions.csv")
    else:
        output_path = args.output

    # 设备选择
    if args.cuda_device is not None:
        device = torch.device(f"cuda:{args.cuda_device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"[info] 使用设备: {device}")

    # 构建模型并加载权重
    print(f"[info] 加载模型权重: {args.checkpoint}")
    model = AudioClassifier(cfg).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 加载数据
    print(f"[info] 加载数据: {cfg.text_cap_path} (split={args.split})")
    splits = load_label_splits(cfg.text_cap_path)
    dataset = data_loader(splits[args.split], cfg, split=args.split)

    batch_size = args.batch_size if args.batch_size else getattr(cfg, "eval_batch_size", 4)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # 推理
    print(f"[info] 开始推理 ({len(dataset)} 样本)...")
    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            batch_names, batch_audio, batch_vad = batch

            pitch_data = batch_audio[0].to(device)
            audio_data = batch_audio[1].to(device)
            pitch_mask = batch_audio[2].to(device)
            audio_mask = batch_audio[3].to(device)
            batch_vad = batch_vad.to(device)

            preds = model(pitch_data, audio_data, pitch_mask, audio_mask)

            for i in range(len(batch_names)):
                results.append({
                    "name": batch_names[i],
                    "true_v": batch_vad[i, 0].item(),
                    "pred_v": preds[i, 0].item(),
                    "true_a": batch_vad[i, 1].item(),
                    "pred_a": preds[i, 1].item(),
                })

    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"[info] 保存 {len(results)} 条预测到: {output_path}")

    # 计算并打印 CCC
    ccc_v = concordance_correlation_coefficient(df["true_v"].values, df["pred_v"].values)
    ccc_a = concordance_correlation_coefficient(df["true_a"].values, df["pred_a"].values)
    ccc_avg = (ccc_v + ccc_a) / 2

    print("\n========== 评估结果 ==========")
    print(f"CCC_V:   {ccc_v:.4f}")
    print(f"CCC_A:   {ccc_a:.4f}")
    print(f"CCC_Avg: {ccc_avg:.4f}")
    print("==============================")


if __name__ == "__main__":
    main()
