
import argparse
import copy
import os
import shutil
import subprocess
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from argparse import ArgumentParser, Namespace
from torch._C import device
from torch import nn
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm
import yaml
from model import AudioClassifier
from data import data_loader
from metrics import evaluate_ccc, evaluate_mse, evaluate_r2
from utils import (
    setup_logger,
    set_seed,
    setup_experiment_directories,
    save_experiment_results,
)
from utils.data_loading import load_label_splits


def _find_free_visible_gpu() -> int | None:
    """返回当前 CUDA_VISIBLE_DEVICES 视角下“空闲”的第一个 GPU 编号.

    判定规则:
    - 优先使用 nvidia-smi 查询 compute 进程:
        * 若某个物理 GPU 没有任何 compute 进程, 视为“空闲”;
    - 若无法调用 nvidia-smi, 返回 0 作为回退 (由上层决定是否接受).
    """

    if shutil.which("nvidia-smi") is None:
        print("[warn] 未找到 nvidia-smi, 无法自动检测空闲 GPU, 将回退到 cuda:0")
        return 0

    # 当前可见的物理 GPU ID 列表; 若未设置 CUDA_VISIBLE_DEVICES, 则使用全部 index.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        try:
            visible_global_ids = [
                int(x.strip()) for x in visible.split(",") if x.strip()
            ]
        except ValueError:
            visible_global_ids = None
    else:
        visible_global_ids = None


    q_gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


    index_to_uuid: dict[int, str] = {}
    for line in q_gpu.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        uuid = parts[1]
        index_to_uuid[idx] = uuid


    q_apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    busy_uuids = {
        line.strip()
        for line in q_apps.stdout.strip().splitlines()
        if line.strip() and "No running" not in line
    }


    # 决定要检查的物理 GPU index 列表
    if visible_global_ids is not None:
        candidate_global_ids = [
            gid for gid in visible_global_ids if gid in index_to_uuid
        ]
    else:
        candidate_global_ids = sorted(index_to_uuid.keys())

    if not candidate_global_ids:
        print(
            "[warn] 未在 nvidia-smi 输出中找到任何可见 GPU, "
            "将回退到 cuda:0"
        )
        return 0

    # 在可见的物理卡中找“无 compute 进程”的卡
    free_local_indices: list[int] = []
    for local_idx, global_idx in enumerate(candidate_global_ids):
        uuid = index_to_uuid.get(global_idx)
        if not uuid or uuid not in busy_uuids:
            free_local_indices.append(local_idx)

    if not free_local_indices:
        # 所有可见 GPU 都有正在运行的进程
        return None

    return min(free_local_indices)

def _resolve_device(args) -> torch.device:
    """根据配置和当前 GPU 占用情况, 选择运行设备.

    约定:
    - 若 CUDA 不可用, 直接使用 CPU;
    - 若 args.cuda_device 为非负整数, 视为明确指定, 不做自动选择;
    - 若 args.cuda_device < 0, 触发自动选择逻辑:
        * 如果存在空闲 GPU, 选择第一个空闲 GPU;
        * 若所有可见 GPU 都有进程, 抛出 RuntimeError 并提示用户。
    """

    if not torch.cuda.is_available():
        print("[info] CUDA 不可用, 使用 CPU 运行.")
        return torch.device("cpu")

    cuda_dev = getattr(args, "cuda_device", 0)
    if isinstance(cuda_dev, str):
        try:
            cuda_dev = int(cuda_dev)
        except ValueError:
            # 无法解析为整数时, 统一视为自动模式
            cuda_dev = -1

    # 明确给出非负整数: 尊重用户选择
    if isinstance(cuda_dev, int) and cuda_dev >= 0:
        return torch.device(f"cuda:{cuda_dev}")

    # 自动选择模式: cuda_device < 0
    free_idx = _find_free_visible_gpu()
    if free_idx is None:
        raise RuntimeError(
            "未检测到空闲 GPU (所有可见 CUDA 设备上都有运行中的进程)。\n"
            "请确认至少有一块 GPU 空闲, 或在配置文件中显式设置 cuda_device "
            "为某块 GPU 的索引, 并确保该 GPU 型号满足实验需求。"
        )

    print(f"[info] 自动选择空闲 GPU: cuda:{free_idx}")
    # 更新回 args, 便于后续日志/保存使用
    args.cuda_device = int(free_idx)
    return torch.device(f"cuda:{free_idx}")


class Trainer:
    def __init__(self, args):
        self._check_args(args)
        self.args = args


        self.args.target = "both"

        self.experiment_name, self.args.load_model_path, self.args.logger_path = setup_experiment_directories(self.args)

        # 统一通过 _resolve_device 选择运行设备 (支持自动选择空闲 GPU)
        self.device: device = _resolve_device(self.args)
        print(f"Using device: {self.device}")

        # 记录设备/GPU 型号，方便区分不同实验运行在哪块卡上

        device_index: int = (
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device()
        )
        self.device_name = torch.cuda.get_device_name(device_index)

        print(f"Device name: {self.device_name}")

        # 创建模型
        self.vad_model: AudioClassifier = AudioClassifier(self.args).to(self.device)

        # 统一使用验证集 CCC 作为 early stopping / 最佳模型选择指标：
        # 越大越好，从 -inf 开始。
        self.best_score: float = float("-inf")
        self.best_epoch: int = 0
        self.best_test_metrics = None
        self.save_path: str = self.args.load_model_path
        self.logger_path: str = self.args.logger_path

        # Early stopping 配置（默认开启，可在 YAML 中覆盖）
        self.early_stopping_enabled: bool = bool(
            getattr(self.args, "early_stopping_enabled", True)
        )
        self.early_stopping_min_delta: float = float(
            getattr(self.args, "early_stopping_min_delta", 0.01)
        )
        self.early_stopping_patience: int = int(
            getattr(self.args, "early_stopping_patience", 30)
        )

        self.logger = None

    def _check_args(self, args):
        """基本配置合法性检查."""
        if not hasattr(args, "task") or args.task not in [
            "basemodel",
            "pitch_and_wav2vec2",
        ]:
            raise ValueError(
                f"Unsupported task '{getattr(args, 'task', None)}'. "
                "Only 'basemodel' and 'pitch_and_wav2vec2' are supported."
            )

        # 一些训练必需的配置项：缺失或类型错误时尽早抛出可读错误
        required_fields = {
            "audio_model_name": str,
            "text_cap_path": str,
            "load_model_path": str,
            "logger_path": str,
            "epoch": int,
            "train_batch_size": int,
            "eval_batch_size": int,
            "train_shuffle": bool,
            "learning_rate": (int, float),
            "max_grad_norm": (int, float),
            "seed": int,
            "cuda_device": int,
        }

        missing = []
        wrong_types = []
        for field, expected_type in required_fields.items():
            if not hasattr(args, field):
                missing.append(field)
                continue
            value = getattr(args, field)
            if value is None:
                missing.append(field)
                continue

            # 尝试对数字字段做宽松的字符串转换, 兼容 YAML 将 1e-5 解析为字符串的情况
            if isinstance(value, str):
                try:
                    if expected_type is int:
                        coerced = int(value)
                        setattr(args, field, coerced)
                        value = coerced
                    elif expected_type is float:
                        coerced = float(value)
                        setattr(args, field, coerced)
                        value = coerced
                    elif isinstance(expected_type, tuple) and float in expected_type:
                        coerced = float(value)
                        setattr(args, field, coerced)
                        value = coerced
                except Exception:
                    # 无法转换时保持原值, 交由类型检查报错
                    pass

            if not isinstance(value, expected_type):
                wrong_types.append(
                    f"{field} (got {type(value).__name__}, "
                    f"expected {expected_type})"
                )

        if missing:
            raise ValueError(
                "Missing required config fields: "
                + ", ".join(sorted(missing))
            )

        if wrong_types:
            raise ValueError(
                "Config fields have unexpected types: "
                + "; ".join(wrong_types)
            )

        # target 如果给了就校验取值；当前仅支持 both
        target = getattr(args, "target", None)
        if target is not None and target != "both":
            raise ValueError(
                f"Unsupported target '{target}'. "
                "Supported targets are: ['both']"
            )

    def set_logger(self):
        self.logger = setup_logger(self.logger_path)

    def mse_loss(self, pred_vads, vads):
        loss_fn = nn.MSELoss()
        return loss_fn(pred_vads, vads.float())

    def save_model(self, model, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(model.state_dict(), os.path.join(path, "model.bin"))

    def get_model(self):
        """从 self.save_path 目录下加载 model.bin 权重文件."""
        model_path = os.path.join(self.save_path, "model.bin")
        self.vad_model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.vad_model.eval()

    def _is_better(self, curr_score: float) -> bool:
        """判断当前评分是否在 best_score 基础上有足够提升."""
        return curr_score >= self.best_score + self.early_stopping_min_delta

    def set_parameters(self):
        # 冻结骨干特征提取层参数，避免训练中梯度更新整网
        self.vad_model.freeze_parameters()
        self.training_epochs = self.args.epoch
        # 每个 batch 反向传播后，用 clip_grad_norm_ 把所有参数的梯度范数限制在 max_grad_norm 之内，防止梯度爆炸。
        self.max_grad_norm = self.args.max_grad_norm
        self.lr = self.args.learning_rate

        # scheduler 的步数按「batch 数」来算
        steps_per_epoch = len(self.train_dataloader)
        self.num_training_steps = steps_per_epoch * self.training_epochs

        # ===== warmup 配置 =====
        # 支持两种方式：
        # 1) 显式设置 warmup_steps
        # 2) 设置 warmup_ratio（例如 0.05 或 0.1）
        warmup_steps_cfg = getattr(self.args, "warmup_steps", None)
        warmup_ratio_cfg = getattr(self.args, "warmup_ratio", None)

        if warmup_steps_cfg is not None:
            self.num_warmup_steps = int(warmup_steps_cfg)

        elif warmup_ratio_cfg is not None:
            warmup_ratio = float(warmup_ratio_cfg)
            if not 0.0 <= warmup_ratio <= 1.0:
                raise ValueError(
                    "warmup_ratio 应该在 [0, 1] 范围内，"
                    f"当前为: {warmup_ratio}"
                )
            self.num_warmup_steps = int(
                self.num_training_steps * warmup_ratio
            )
        else:
            # 未配置时默认不开启 warmup
            self.num_warmup_steps = 0

        self.optimizer = torch.optim.AdamW(
            self.vad_model.parameters(),
            lr=float(self.lr),
        )

        # ===== lr scheduler 类型配置 =====
        scheduler_type = getattr(
            self.args,
            "lr_scheduler_type",
            "linear",
        )
        scheduler_type = str(scheduler_type).lower()

        if scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.num_warmup_steps,
                num_training_steps=self.num_training_steps,
            )
        else:
            # 默认使用 linear scheduler
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.num_warmup_steps,
                num_training_steps=self.num_training_steps,
            )

    def _plot_loss_curve(self, train_losses, val_losses):
        """绘制并保存训练/验证损失曲线."""
        if not train_losses or not val_losses:
            return


        epochs = range(1, len(train_losses) + 1)
        fig_path = os.path.join(self.save_path, "loss_curve.png")

        plt.figure()
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Train / Val Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()

        if self.logger is not None:
            self.logger.info("损失曲线已保存到: %s", fig_path)

    def train(self):

        splits = load_label_splits(self.args.text_cap_path)
        train = splits["train"]
        val = splits["val"]
        test = splits["test"]

        self.train_dataset = data_loader(train, self.args, split="train")
        self.dev_dataset = data_loader(val, self.args, split="val")
        self.test_dataset = data_loader(test, self.args, split="test")

        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=self.args.train_shuffle,
            collate_fn=self.train_dataset.collate_fn,
        )
        self.dev_dataloader = DataLoader(
            dataset=self.dev_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,  
            collate_fn=self.dev_dataset.collate_fn,
        )
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,  
            collate_fn=self.test_dataset.collate_fn,
        )

        self.set_logger()
        self.set_parameters()
        # 注意：随机种子已在 main() 函数中设置，此处不再重复设置
        # 避免重置随机数生成器，确保模型初始化和数据加载使用连续的随机序列
        assert self.logger is not None, "Logger 未正确初始化"
        self.logger.info("#############训练阶段#############")
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info(f"Using device: {self.device}")
        # 额外记录具体 GPU/设备型号，方便后续排查和对比实验
        self.logger.info("Device name: %s", self.device_name)
        self.logger.info(
            "Dataset sizes - Train: %d, Val: %d, Test: %d",
            len(self.train_dataset),
            len(self.dev_dataset),
            len(self.test_dataset),
        )
        self.logger.info(
            "Train loss uses MSE; best checkpoint is selected by metric=%s.",
            "CCC",
        )
        self.logger.info(
            "Early stopping: %s (metric=dev_avg CCC, patience=%d, "
            "min_delta=%.4f).",
            "ON" if self.early_stopping_enabled else "OFF",
            self.early_stopping_patience,
            self.early_stopping_min_delta,
        )

        self.best_model_state = None

        train_losses = []
        val_losses = []

        # 记录从当前 best 起，连续多少个 epoch 没有“真正变好”
        epochs_without_improvement = 0

        try:
            for epoch in range(self.training_epochs):
                self.vad_model.train()
                self.logger.info("\n%s", "=" * 80)
                self.logger.info(
                    "Epoch %d/%d - [TRAINING]",
                    epoch + 1,
                    self.training_epochs,
                )
                self.logger.info("%s", "=" * 80)

                total_loss = 0.0
                num_train_samples = 0

                for _, data in enumerate(tqdm(self.train_dataloader, mininterval=10)):
                    # 在每个 batch 开始时清零梯度，确保不会累积上一个 batch 的梯度
                    self.optimizer.zero_grad()

                    _, batch_audio, batch_vad = data

                    pitch_data = batch_audio[0].to(self.device)
                    audio_data = batch_audio[1].to(self.device)
                    pitch_mask = batch_audio[2].to(self.device)
                    audio_mask = batch_audio[3].to(self.device)
                    batch_vad = batch_vad.to(self.device)

                    pred_logits = self.vad_model(pitch_data, audio_data, pitch_mask, audio_mask)

                    loss_val = self.mse_loss(pred_vads=pred_logits, vads=batch_vad)

                    batch_size = batch_vad.size(0)
                    total_loss += loss_val.item() * batch_size
                    num_train_samples += batch_size

                    loss_val.backward()
                    # 梯度裁剪，防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(
                        parameters=self.vad_model.parameters(),
                        max_norm=self.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.scheduler.step()

                avg_loss = total_loss / max(num_train_samples, 1)
                train_losses.append(avg_loss)

                self.vad_model.eval()

                # 计算验证集损失（与训练阶段保持相同的输入格式，支持 pitch mask）
                with torch.no_grad():
                    dev_total_loss = 0.0
                    num_dev_samples = 0
                    for dev_data in self.dev_dataloader:
                        _, batch_audio, batch_vad = dev_data

                        pitch_data = batch_audio[0].to(self.device)
                        audio_data = batch_audio[1].to(self.device)
                        pitch_mask = batch_audio[2].to(self.device)
                        audio_mask = batch_audio[3].to(self.device)
                        batch_vad = batch_vad.to(self.device)

                        dev_logits = self.vad_model(pitch_data, audio_data, pitch_mask, audio_mask)

                        dev_loss = self.mse_loss(pred_vads=dev_logits, vads=batch_vad)
                        batch_size = batch_vad.size(0)
                        dev_total_loss += dev_loss.item() * batch_size
                        num_dev_samples += batch_size


                dev_avg_loss = dev_total_loss / max(num_dev_samples, 1)
                val_losses.append(dev_avg_loss)

                self.logger.info("\n%s", "=" * 80)
                self.logger.info("Epoch %d/%d - [VALIDATION]",epoch + 1, self.training_epochs,)
                self.logger.info("%s", "=" * 80)

                self.logger.info("Train MSE Loss: %.6f", avg_loss)
                self.logger.info("Val   MSE Loss: %.6f", dev_avg_loss)

                # 验证集上统一使用 CCC 作为选择最佳模型的指标
                dev_ccc_v, dev_ccc_a = evaluate_ccc(self.vad_model, self.dev_dataloader, self.device)

                self.logger.info("Validation Results (CCC):")
                dev_avg = (dev_ccc_v + dev_ccc_a) / 2
                self.logger.info("  V (Valence):     %.4f", dev_ccc_v)
                self.logger.info("  A (Arousal):     %.4f", dev_ccc_a)
                self.logger.info("  Average:         %.4f", dev_avg)

                curr_score = dev_avg
                # improved: True 表示相较 best_score 有足够提升, False 表示无提升
                improved = self._is_better(curr_score)
                if improved:
                    self.best_score = curr_score
                    # epoch 是从 0 开始计数的, 保存/日志里通常用从 1 开始的“人类可读”epoch，所以要 epoch + 1。
                    self.best_epoch = epoch + 1
                    # 使用 deepcopy 确保 tensor 也被复制，而非共享内存引用
                    self.best_model_state = copy.deepcopy(self.vad_model.state_dict())

                    self.save_model(self.vad_model, self.save_path)

                    # 一旦有“真正变好”的 epoch，就重置 patience 计数
                    epochs_without_improvement = 0

                    self.logger.info("\n*** New best model found at epoch %d! " "Val score: %.4f ***", self.best_epoch, self.best_score)

                    # 在 Test 集上计算 CCC，记录当前最佳模型的性能（便于中途查看）
                    best_test_metric_v, best_test_metric_a = evaluate_ccc(
                        self.vad_model,
                        self.test_dataloader,
                        self.device,
                    )

                    best_test_avg = (best_test_metric_v + best_test_metric_a) / 2
                    self.best_test_metrics = (best_test_metric_v, best_test_metric_a, best_test_avg)

                    self.logger.info("Best (so far) Test Results (CCC) at epoch %d:", self.best_epoch)
                    self.logger.info("  V (Valence):     %.4f", best_test_metric_v)
                    self.logger.info("  A (Arousal):     %.4f", best_test_metric_a)
                    self.logger.info("  Average:         %.4f", best_test_avg)

                    results_path = save_experiment_results(
                        self.experiment_name,
                        self.best_epoch,
                        self.best_score,
                        best_test_metric_v,
                        best_test_metric_a,
                        best_test_avg,
                        self.args.load_model_path,
                    )
                    self.logger.info("\nResults saved to: %s", results_path)

                else:
                    # 只有在已经有过一次有效 best 的前提下，才开始累计 patience
                    if self.best_model_state is not None:
                        epochs_without_improvement += 1

                    self.logger.info(
                        "验证集分数 %.4f 未在当前最佳 %.4f 的基础上提升至少 %.4f "
                        "(no improvement for %d/%d epochs).",
                        curr_score,
                        self.best_score,
                        self.early_stopping_min_delta,
                        epochs_without_improvement,
                        self.early_stopping_patience,
                    )

                    if (
                        self.early_stopping_enabled
                        and self.best_model_state is not None
                        and epochs_without_improvement
                        >= self.early_stopping_patience
                    ):
                        self.logger.info(
                            "触发 early stopping: 连续 %d 个 epoch dev_avg "
                            "未提升至少 %.4f，相对当前 best=%.4f。",
                            epochs_without_improvement,
                            self.early_stopping_min_delta,
                            self.best_score,
                        )
                        break

        except KeyboardInterrupt:
            # 友好地响应 Ctrl+C，中断训练但保留当前 best_model_state
            self.logger.info("Training interrupted by user (KeyboardInterrupt).")

        # 训练结束后绘制损失曲线（即使是早停或中断）
        self._plot_loss_curve(train_losses, val_losses)

        if self.best_model_state is not None:
            self.logger.info("\n=== Final Best Model ===")
            self.logger.info("Best validation score: %.4f", self.best_score)

            self.vad_model.load_state_dict(self.best_model_state)
            test_details_path = os.path.join(self.save_path, "test_predictions.csv")

            # 1) 在 Test 集上用 CCC 评估，并保存逐样本预测到 CSV
            test_ccc_v, test_ccc_a = evaluate_ccc(
                self.vad_model,
                self.test_dataloader,
                self.device,
                save_details_path=test_details_path,
            )

            test_ccc_avg = (test_ccc_v + test_ccc_a) / 2

            # 2) 同一最佳权重上计算 MSE / R2（仅用于报告，不参与 early stopping）
            test_mse_v, test_mse_a = evaluate_mse(
                self.vad_model,
                self.test_dataloader,
                self.device,
                save_details_path=None,
            )

            test_mse_avg = (test_mse_v + test_mse_a) / 2

            test_r2_v, test_r2_a = evaluate_r2(
                self.vad_model,
                self.test_dataloader,
                self.device,
                save_details_path=None,
            )

            test_r2_avg = (test_r2_v + test_r2_a) / 2

            self.logger.info("\nFinal Test Results (on best validation checkpoint):")
            self.logger.info(
                "  CCC - V: %.4f, A: %.4f, avg: %.4f",
                test_ccc_v,
                test_ccc_a,
                test_ccc_avg,
            )
            self.logger.info(
                "  MSE - V: %.4f, A: %.4f, avg: %.4f",
                test_mse_v,
                test_mse_a,
                test_mse_avg,
            )
            self.logger.info(
                "  R2  - V: %.4f, A: %.4f, avg: %.4f",
                test_r2_v,
                test_r2_a,
                test_r2_avg,
            )
            self.logger.info(
                "测试集逐样本预测结果已保存到: %s",
                test_details_path,
            )

            self.save_model(self.vad_model, self.save_path)

            # 使用最终 Test 集上的 CCC / MSE / R2 结果覆盖写入 best_results.txt
            results_path = save_experiment_results(
                self.experiment_name,
                self.best_epoch,
                self.best_score,
                test_ccc_v,
                test_ccc_a,
                test_ccc_avg,
                self.args.load_model_path,
                mse_v=test_mse_v,
                mse_a=test_mse_a,
                mse_avg=test_mse_avg,
                r2_v=test_r2_v,
                r2_a=test_r2_a,
                r2_avg=test_r2_avg,
            )
            self.logger.info("\nResults saved to: %s", results_path)


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default=None, type=str)
    parser.add_argument(
        "--pm",
        "--pretrained_model",
        dest="pm",
        type=str,
        default=None,
        help=(
            "可选: 覆盖配置文件中的 audio_model_name。示例："
            "--pm chinese-wav2vec2-large (等价于 "
            "pretrained_model/chinese-wav2vec2-large)。"
        ),
    )
    arg_: argparse.Namespace = parser.parse_args()
    if arg_.config is None:
        raise NameError("Include a config file in the argument please.")

    config_path = arg_.config

    with open(file=arg_.config) as config_file:
        args_dict = yaml.safe_load(stream=config_file)

    # 若通过命令行指定了预训练模型, 优先使用命令行参数覆盖配置文件中的 audio_model_name
    if arg_.pm is not None:
        pm = arg_.pm.strip()
        if not pm:
            raise ValueError("--pm / --pretrained_model 不能为空字符串")
        # 若未包含路径前缀, 自动加上 pretrained_model/
        if "/" not in pm:
            pm = f"pretrained_model/{pm}"
        args_dict["audio_model_name"] = pm

    args: Namespace = argparse.Namespace(**args_dict)
    args._config_file_path = config_path

    # 在初始化 Trainer 之前就固定随机种子，确保模型参数和数据划分可复现
    set_seed(args.seed)

    emotion_trainer: Trainer = Trainer(args)
    emotion_trainer.train()


if __name__ == "__main__":
    main()
