from torch.types import Number
from torch import Tensor
from torch.nn.modules.linear import Linear
from torch.nn.modules.activation import MultiheadAttention
from transformers.modeling_outputs import BaseModelOutput
import os
import torch
from transformers import AutoModel
import torch.nn as nn
import torch.nn.functional as F
from typing import Any


class AudioClassifier(torch.nn.Module):
    """Wav2Vec2 + pitch 融合的音频情感回归模型."""

    def __init__(self, args) -> None:
        super().__init__()
    
        self.args = args
        self.a_hidden_size = 0
        # 是否在 cross-attention 前将 pitch 时间轴插值到与 audio hidden 相同长度
        # True: 当前默认行为（一一对齐时间步）
        # False: 保留原始 T_pitch，与 T_audio 可不同（更“纯正”的 cross-attention）
        self.align_pitch_with_audio: bool = bool(
            getattr(self.args, "align_pitch_with_audio", True)
        )


        self.audio_model = AutoModel.from_pretrained(self.args.audio_model_name)
        assert self.audio_model is not None, f"无法加载音频模型：{self.args.audio_model_name}"

        # 仅针对本地 wav2vec2_large 权重修复 masked_spec_embed 的极端数值问题
        audio_model_name: str = str(getattr(self.args, "audio_model_name", ""))
        model_dirname: str = os.path.basename(audio_model_name.rstrip("/"))

        if model_dirname == "wav2vec2_large" and hasattr(self.audio_model, "masked_spec_embed"):
            p = self.audio_model.masked_spec_embed
            with torch.no_grad():
                has_nan_before: Number = torch.isnan(input=p).any().item()
                has_inf_before: Number = torch.isinf(input=p).any().item()
                min_val_before: float = float(p.min())
                max_val_before: float = float(p.max())

            print("masked_spec_embed stats (before fix):", "nan:", has_nan_before, "inf:", has_inf_before, "min:", min_val_before, "max:", max_val_before,)

            nn.init.uniform_(tensor=p, a=-0.1, b=0.1)
    
            with torch.no_grad():
                has_nan_after: Number = torch.isnan(input=p).any().item()
                has_inf_after: Number = torch.isinf(input=p).any().item()
                min_val_after: float = float(p.min())
                max_val_after: float = float(p.max())

            print("masked_spec_embed stats (after fix):", "nan:", has_nan_after, "inf:", has_inf_after, "min:", min_val_after, "max:", max_val_after)
        
        self.a_hidden_size: Any | None = getattr(self.audio_model.config, "hidden_size", None)
        self.num_classes = 2

        assert self.a_hidden_size is not None, "无法获取音频骨干模型的隐藏层维度，请检查模型配置。"
        self.pitch_embedding: Linear = torch.nn.Linear(in_features=1, out_features=self.a_hidden_size)
        self.attention_layer: MultiheadAttention = torch.nn.MultiheadAttention(embed_dim=self.a_hidden_size, num_heads=8, batch_first=True)
        self.fc: Linear = torch.nn.Linear(in_features=self.a_hidden_size, out_features=self.num_classes)

    def freeze_parameters(self) -> None:
        self.audio_model.feature_extractor._freeze_parameters()


    def get_hidden_states(self) -> Any | None:
        return getattr(self, "hidden_states", None)

    def forward(self, pitch_data, audio_data, pitch_mask, audio_mask=None) -> Tensor:

        pitch_padding_mask = pitch_mask
        tf_audio = audio_data
        pitch_values = pitch_data

        # 传入 audio_mask 到 Wav2Vec2 模型
        wav2vec_outputs: BaseModelOutput = self.audio_model(
            tf_audio,
            attention_mask=audio_mask,
        )
        last_hidden = wav2vec_outputs.last_hidden_state

        assert last_hidden is not None, "音频骨干模型未返回最后隐藏层。"        
        pitch_values: torch.Tensor = pitch_values.to(dtype=last_hidden.dtype)
        pitch_embeds = self.pitch_embedding(pitch_values.unsqueeze(dim=-1))
        # 可选：是否将 pitch 的时间轴插值到与骨干相同长度
        if self.align_pitch_with_audio and pitch_embeds.size(1) != last_hidden.size(1):
            # 对齐到与骨干相同的序列长度
            pitch_embeds = pitch_embeds.transpose(1, 2)
            pitch_embeds = F.interpolate(input=pitch_embeds, size=last_hidden.size(1), mode="linear", align_corners=False)
            pitch_embeds = pitch_embeds.transpose(1, 2)

            # 同时调整 mask 的长度
            if pitch_padding_mask is not None:
                pitch_padding_mask = F.interpolate(input=pitch_padding_mask.float().unsqueeze(dim=1), size=last_hidden.size(1), mode="nearest").squeeze(1).bool()

        # 使用 key_padding_mask 来忽略 audio 的 padding
        # 注意：HuggingFace attention_mask (1=有效, 0=padding) 需要转换为
        # PyTorch key_padding_mask (True=padding, False=有效)
        audio_key_padding_mask = None
        if audio_mask is not None:
            audio_mask_aligned = audio_mask
            if audio_mask_aligned.size(1) != last_hidden.size(1):
                audio_mask_aligned = F.interpolate(
                    input=audio_mask_aligned.float().unsqueeze(dim=1),
                    size=last_hidden.size(1),
                    mode="nearest",
                ).squeeze(1)
            audio_key_padding_mask = (audio_mask_aligned == 0)  # 0 转为 True (padding)

        attention_output, _ = self.attention_layer(
            query=pitch_embeds,
            key=last_hidden,
            value=last_hidden,
            key_padding_mask=audio_key_padding_mask,  # 使用 audio 的 padding mask
        )

        # 使用 masked pooling
        if pitch_padding_mask is not None:
            # pitch_padding_mask: (batch, seq_len), True 表示 padding
            # 将 padding 位置的值设为 0，然后只对有效位置求平均
            mask_expanded = (~pitch_padding_mask).unsqueeze(-1).float()  # (batch, seq_len, 1)
            masked_attention = attention_output * mask_expanded
            sum_attention = masked_attention.sum(dim=1)  # (batch, hidden_size)
            valid_lengths = mask_expanded.sum(dim=1)  # (batch, 1)
            pooled_output = sum_attention / (valid_lengths + 1e-9)  # 避免除零
        else:
            pooled_output = attention_output.mean(dim=1)

        output: torch.Tensor = self.fc(pooled_output)

        return output
