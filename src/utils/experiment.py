import os
import shutil
from types import SimpleNamespace
from typing import Tuple


def setup_experiment_directories(args: SimpleNamespace) -> Tuple[str, str, str]:
    """创建唯一实验目录，返回 (experiment_name, load_model_path, logger_path).

    目录层级设计为:
        runs/<dataset>/<pretrained_model>/<method_and_target>/exp_N

    其中:
    - <dataset>       来自配置文件所在子目录 (如 normal_and_patient, iemocap)
    - <pretrained_model> 来自 audio_model_name 的 basename (如 wav2vec2_large)
    - <method_and_target> 由 load_model_path 的 basename + 目标维度组成
                           (如 linear_no_norm_VA)
    - exp_N           在同一方法下按 1,2,3... 递增, 不再使用时间戳
    """
    runs_dir = "runs"
    os.makedirs(runs_dir, exist_ok=True)

    # 提取模型目录/ID的最后一段，并进行轻量清洗，避免路径分隔符/空白
    def _sanitize(name: str) -> str:
        name = os.path.basename(str(name)).strip()
        # 仅保留常见可见字符，其他替换为 '-'
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
        return "".join(ch if ch in allowed else "-" for ch in name)

    # ===== 1) 数据集维度：来自 config 所在子目录 =====
    dataset_raw = "unknown_dataset"
    config_path = getattr(args, "_config_file_path", None)
    if config_path is not None:
        config_dir = os.path.dirname(config_path)
        dataset_candidate = os.path.basename(config_dir) or dataset_raw

        # 兼容 configs/<dataset> 和 configs/<dataset>/<variant>/ 这两种组织方式:
        # - 若当前目录名是 with_norm / no_norm 等「变体」，则再往上一层取数据集名；
        # - 否则直接使用当前目录名作为数据集名。
        parent_dir_name = os.path.basename(os.path.dirname(config_dir))
        if dataset_candidate in {"with_norm", "no_norm"} and parent_dir_name:
            dataset_raw = parent_dir_name
        else:
            dataset_raw = dataset_candidate
    dataset_name = _sanitize(dataset_raw) or "unknown_dataset"

    # ===== 2) 预训练模型维度：来自 audio_model_name =====
    model_name_raw = getattr(args, "audio_model_name", "unknown_model") or "unknown_model"
    model_name = _sanitize(model_name_raw) or "unknown_model"

    # ===== 3) 方法 + target 维度 =====
    base_name = os.path.basename(str(args.load_model_path)).rstrip("/\\")
    if not base_name:
        base_name = os.path.basename(os.path.dirname(str(args.logger_path)))

    # 当前仅支持双维度目标，统一使用 VA 作为后缀
    target_suffix = "VA"

    method_raw = base_name
    # 尝试去掉形如 "<method>_<dataset>" 的后缀, 让方法名在不同数据集下保持一致
    lower_dataset = dataset_name.lower()
    lower_method = method_raw.lower()
    suffix = f"_{lower_dataset}"
    if lower_dataset and lower_method.endswith(suffix):
        method_raw = method_raw[: -len(suffix)]
    method_name = _sanitize(method_raw) or "unknown_method"

    method_with_target = f"{method_name}_{target_suffix}"

    # ===== 4) 组装三级目录 =====
    base_dir = os.path.join(runs_dir, dataset_name, model_name, method_with_target)
    os.makedirs(base_dir, exist_ok=True)

    # ===== 5) 在该方法目录下按照 exp_1, exp_2 ... 递增新建子目录 =====
    existing_runs = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("exp_")
    ]
    max_idx = 0
    for d in existing_runs:
        try:
            idx = int(d.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        else:
            max_idx = max(max_idx, idx)

    next_idx = max_idx + 1
    run_name = f"exp_{next_idx}"
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=False)

    # experiment_name 中保留三维信息 + exp_id, 方便日志/结果文件中快速定位
    experiment_name = f"{dataset_name}_{model_name}_{method_with_target}_{run_name}"

    # 更新 args 内的路径，保持行为与原来一致
    args.experiment_name = experiment_name
    args.load_model_path = run_dir
    args.logger_path = os.path.join(run_dir, "train.log")

    # 尝试保存 config
    if hasattr(args, "_config_file_path"):
        shutil.copy2(args._config_file_path, os.path.join(run_dir, "config.yaml"))

    return experiment_name, args.load_model_path, args.logger_path


def save_experiment_results(
    experiment_name: str,
    best_epoch: int,
    best_score: float,
    ccc_v: float,
    ccc_a: float,
    ccc_avg: float,
    save_dir: str,
    *,
    mse_v: float | None = None,
    mse_a: float | None = None,
    mse_avg: float | None = None,
    r2_v: float | None = None,
    r2_a: float | None = None,
    r2_avg: float | None = None,
) -> str:
    """将实验结果写入指定目录下的文件，返回结果文件路径."""
    results_path = os.path.join(save_dir, "best_results.txt")

    with open(results_path, "w") as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Best Validation Score: {best_score:.4f}\n")
        f.write("\nBest Test Results (CCC, at best epoch):\n")
        f.write(f"V (Valence): {ccc_v:.4f}\n")
        f.write(f"A (Arousal): {ccc_a:.4f}\n")
        f.write(f"Average: {ccc_avg:.4f}\n")

        if (
            mse_v is not None
            and mse_a is not None
            and mse_avg is not None
        ):
            f.write("\nBest Test Results (MSE, at best epoch):\n")
            f.write(f"V (Valence): {mse_v:.4f}\n")
            f.write(f"A (Arousal): {mse_a:.4f}\n")
            f.write(f"Average: {mse_avg:.4f}\n")

        if (
            r2_v is not None
            and r2_a is not None
            and r2_avg is not None
        ):
            f.write("\nBest Test Results (R2, at best epoch):\n")
            f.write(f"V (Valence): {r2_v:.4f}\n")
            f.write(f"A (Arousal): {r2_a:.4f}\n")
            f.write(f"Average: {r2_avg:.4f}\n")

    return results_path
