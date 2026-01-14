from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
from tqdm import tqdm
import os
from utils.hf_audio import load_feature_extractor_and_tokenizer

def z_score_normalization(pitch_values):
    mean = np.mean(pitch_values)
    std = np.std(pitch_values)
    if not np.isfinite(std) or std < 1e-6:
        return np.zeros_like(pitch_values, dtype=np.float32)
    normalized_pitch = (pitch_values - mean) / (std + 1e-8)
    return normalized_pitch

class data_loader(Dataset):

    def __init__(self, data, args, split=None):
        self.args = args
        self.split = split

        self.audio_model_name = self.args.audio_model_name
        processor_name = getattr(
            self.args,
            "audio_processor_name",
            getattr(self.args, "audio_model_name", None),
        )
        fe, tok = load_feature_extractor_and_tokenizer(
            name_for_feature_extractor=processor_name,
            name_for_tokenizer=getattr(self.args, "tokenizer_name", None),
        )
        self.processor = fe
        self.tokenizer = tok

        max_audio_duration = getattr(self.args, "max_audio_duration", None)
        if max_audio_duration is not None:
            max_audio_duration = float(max_audio_duration)

        self.max_audio_duration = max_audio_duration

        self.saved_data = []

        name = list(data["name"])
        V = list(data["V"])
        A = list(data["A"])

        # 若 labels.csv 中已经包含每条样本的绝对/相对音频路径，则优先使用
        audio_paths = list(data["audio_path"]) if "audio_path" in data.columns else None

        def _get_split_dir() -> str:
            assert self.split is not None
            return self.split

        def _build_wav_path(idx: int, nm: str) -> str:
            return str(audio_paths[idx])

        for i in tqdm(range(len(data)), mininterval=10):
            wav_path = _build_wav_path(idx=i, nm=name[i])
            sr = 16000
            desired_duration = self.max_audio_duration
            waveform, re_sr = self.load_wav(wav_path, desired_duration, sr)
            # 跳过无效数据
            if waveform is None or re_sr is None:
                continue

            # 1) 仅从缓存加载原始 pitch（离线预计算）
            pitch_values = self._load_or_compute_pitch(name=name[i], split=_get_split_dir())
            if pitch_values is None or len(pitch_values) == 0:
                print("pitch list empty")
                continue

            # 2) 连续表示：与原始实现一致，仅按 normalized 决定是否 z-score
            pitch_raw = np.array(pitch_values, dtype=np.float32)
            if self.args.normalized == "ok":
                pitch_cont = z_score_normalization(pitch_values=pitch_raw)
            else:
                pitch_cont = pitch_raw

            # 直接保存原始 pitch 序列 (长度可变)
            # collate_fn 会处理 padding，模型的 pitch_embedding 会处理投影
            pitch_features = torch.tensor(pitch_cont, dtype=torch.float32)

            self.saved_data.append(
                [V[i], A[i], pitch_features, waveform, sr]
            )

        if self.args.normalized == "ok":
            print("pitch was normalized]")
        else:
            print("wasnt normalized")

    def __len__(self) -> int:
        return len(self.saved_data)

    def load_wav(self, wav_path, desired_duration: float | None = 3.0, resample_sr: int = 16000):

        waveform, sample_rate = librosa.load(path=wav_path, sr=resample_sr, mono=True)

        if desired_duration is not None:
            target_len = int(desired_duration * resample_sr)
            cur_len = waveform.shape[0]
            if cur_len < target_len:
                pad_width = target_len - cur_len
                waveform = np.pad(waveform, (0, pad_width), mode="constant")
            elif cur_len > target_len:
                waveform = waveform[:target_len]

        return waveform, sample_rate

    def __getitem__(self, idx: int) -> Any:
        return self.saved_data[idx]

    def collate_fn(self, batch):
        batch_audio = []
        batch_va = []
        batch_raw_audio = []
        for _, row in enumerate(batch):
            (
                V,
                A,
                pitchs,
                raw_audio,
                sr,
            ) = row
            batch_va.append([V, A])
            batch_audio.append(pitchs)
            batch_raw_audio.append(raw_audio)

        padded_pitch = []
        original_lengths = []
        max_len = 0
        for p in batch_audio:
            if not torch.is_tensor(p):
                p = torch.tensor(p, dtype=torch.float32)
            original_lengths.append(p.shape[-1])
            max_len = max(max_len, p.shape[-1])
            padded_pitch.append(p)

        padded_batch = []
        pitch_mask = []
        for p, orig_len in zip(padded_pitch, original_lengths):
            if p.shape[-1] < max_len:
                pad_len = max_len - p.shape[-1]
                pad = torch.zeros(
                    pad_len,
                    dtype=p.dtype,
                    device=p.device,
                )
                p = torch.cat([p, pad], dim=-1)
                mask = torch.cat([
                    torch.zeros(orig_len, dtype=torch.bool),
                    torch.ones(pad_len, dtype=torch.bool),
                ])
            else:
                mask = torch.zeros(max_len, dtype=torch.bool)
            padded_batch.append(p)
            pitch_mask.append(mask)

        batch_raw_audio = [audio.squeeze().flatten() for audio in batch_raw_audio]
        
        processed_audio = self.processor(
            batch_raw_audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True, 
        )

        audio_input_values = processed_audio.input_values  # (batch, audio_seq_len)
        audio_attention_mask = processed_audio.attention_mask  # (batch, audio_seq_len)

        pitch_tensor = torch.stack(padded_batch)
        pitch_mask = torch.stack(pitch_mask)

        return (
            (pitch_tensor, audio_input_values, pitch_mask, audio_attention_mask),
            torch.as_tensor(batch_va, dtype=torch.float32),
        )


    def _cache_root(self) -> str:
        return getattr(self.args,"pitch_cache_dir","data/processed/pitch_cache_ccsemo")

    
    def _cache_path(self, split: str, name: str) -> str:
        base = self._cache_root()
        filename = name if name.endswith(".wav") else f"{name}.wav"
        stem = os.path.splitext(os.path.basename(filename))[0]
        return os.path.join(base, f"{stem}.npy")

    def _load_or_compute_pitch(self, name: str, split: str):
        cache_path = self._cache_path(split, name)
        pitch_arr = np.load(file=cache_path)
        return pitch_arr
