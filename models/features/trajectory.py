"""
models/features/trajectory.py
------------------------------
F2: 轨迹特征提取器（Trajectory Feature Extractor）

从 DDIM 逆向的中间状态序列 {x_{t_k}} 提取轨迹的平滑性和一致性特征。

理论依据：
    - 真实图像（real）在扩散模型流形上，逆向轨迹平滑、步长均匀
    - GAN 生成图像不在该流形上，逆向轨迹早期（小 t）曲率大、步长不均匀
    - 扩散模型生成图像（ADM/SD）虽在流形上，但高频演化模式与真实图像有差异

特征组成（feature_dim = 64）：
    Group 1 - 步长序列（10 维）：
        step_norms[k] = ||x_{t_{k+1}} - x_{t_k}||_F / (H*W*C)
        统计聚合压缩到 10 维（均值、标准差、最大值、最小值 + 插值到 10 维）

    Group 2 - 轨迹曲率（10 维）：
        curvature[k] = ||x_{t_{k+1}} - 2*x_{t_k} + x_{t_{k-1}}||_F
        插值压缩到 10 维

    Group 3 - 全局统计（8 维）：
        total_path_length, path_straightness, step_uniformity (CV),
        max_step_ratio, early_vs_late, curvature_mean, curvature_max, curvature_cv

    Group 4 - 频域演化（16 维）：
        对轨迹的每个中间状态 x_{t_k} 计算平均 PSD，
        观察高频能量随 t 的变化趋势（4 时间段 × 4 频带）

    合计：10 + 10 + 8 + 16 = 44 维，padding 到 64 维
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, '/Users/joces/Downloads/Coding_Campus/Deepfake_Project')

from DRIFT.models.features.base import FeatureExtractor


class TrajectoryFeatureExtractor(FeatureExtractor):
    """
    F2: 从逆向轨迹提取平滑性和一致性特征。

    输入: intermediates = [x_{t_1}, x_{t_2}, ..., x_{t_K}]（K 个中间状态）

    特征 feature_dim = 64。
    """

    # 输出 10 维压缩步长/曲率序列、8 维全局统计、16 维频域演化，pad 到 64
    _COMPRESSED_STEPS = 10   # Group 1/2
    _GLOBAL_STATS = 8        # Group 3
    _FREQ_EVOL = 16          # Group 4
    _RAW_DIM = _COMPRESSED_STEPS * 2 + _GLOBAL_STATS + _FREQ_EVOL  # 44
    _FEATURE_DIM = 64        # padding 到 64

    def __init__(self, num_steps: int = 20) -> None:
        """
        Args:
            num_steps: 中间状态数量 K，默认 20（与 backbone ddim_steps 保持一致）。
        """
        super().__init__()
        self.K = num_steps

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    @property
    def feature_dim(self) -> int:
        return self._FEATURE_DIM

    def extract(
        self,
        x_T: torch.Tensor,
        intermediates: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        从逆向轨迹提取 64 维特征向量。

        Args:
            x_T: 终点噪声张量 ``[B, 3, H, W]``。
              （此处可以为中间状态列表的最后一个元素，或独立的 x_T，
              本方法不直接使用 x_T，仅用其 shape 信息）
            intermediates: list of K tensors，每个形状 ``[B, 3, H, W]``。
              必须提供，不可为 None。

        Returns:
            特征张量 ``[B, 64]``。

        Raises:
            ValueError: 如果 intermediates 为 None。
        """
        if intermediates is None:
            raise ValueError(
                "TrajectoryFeatureExtractor 需要 intermediates，"
                "请用 backbone.invert(x0, return_intermediates=True) 获取中间状态列表。"
            )

        # 确保所有中间状态都在同一设备上（使用 x_T 的设备）
        device = x_T.device
        B = x_T.shape[0]

        # 将 intermediates 转为张量列表，确保设备一致
        states: List[torch.Tensor] = []
        for s in intermediates:
            if isinstance(s, torch.Tensor):
                states.append(s.to(device).float())
            else:
                states.append(torch.tensor(s, device=device, dtype=torch.float32))

        K = len(states)

        if K < 2:
            # 中间状态太少，返回零特征
            features = torch.zeros(B, self._FEATURE_DIM, device=device)
            return features

        # ----------------------------------------------------------------
        # Group 1: 步长序列 (K-1 个步长，压缩到 10 维)
        # ----------------------------------------------------------------
        step_norms = self._compute_step_norms(states, device)  # [B, K-1]
        group1 = self._compress_sequence(step_norms, self._COMPRESSED_STEPS)  # [B, 10]

        # ----------------------------------------------------------------
        # Group 2: 轨迹曲率 (K-2 个曲率，压缩到 10 维)
        # ----------------------------------------------------------------
        if K >= 3:
            curvatures = self._compute_curvatures(states, device)  # [B, K-2]
            group2 = self._compress_sequence(curvatures, self._COMPRESSED_STEPS)  # [B, 10]
        else:
            group2 = torch.zeros(B, self._COMPRESSED_STEPS, device=device)

        # ----------------------------------------------------------------
        # Group 3: 全局统计 (8 维)
        # ----------------------------------------------------------------
        group3 = self._compute_global_stats(
            states, step_norms, group2 if K >= 3 else None, device
        )  # [B, 8]

        # ----------------------------------------------------------------
        # Group 4: 频域演化 (4 时间段 × 4 频带 = 16 维)
        # ----------------------------------------------------------------
        group4 = self._compute_freq_evolution(states, device)  # [B, 16]

        # ----------------------------------------------------------------
        # 拼接并 padding 到 64 维
        # ----------------------------------------------------------------
        raw = torch.cat([group1, group2, group3, group4], dim=1)  # [B, 44]
        pad_size = self._FEATURE_DIM - raw.shape[1]
        if pad_size > 0:
            features = F.pad(raw, (0, pad_size), mode='constant', value=0.0)
        else:
            features = raw[:, :self._FEATURE_DIM]

        # 用 clamp 避免 NaN/Inf
        features = torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
        features = torch.clamp(features, -100.0, 100.0)

        self.validate_output(features)
        return features

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_step_norms(
        self,
        states: List[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """
        计算相邻中间状态的 Frobenius 范数差异，归一化到每像素每通道。

        Returns:
            ``[B, K-1]`` 步长序列张量。
        """
        K = len(states)
        B = states[0].shape[0]
        step_list = []

        for k in range(K - 1):
            diff = states[k + 1] - states[k]           # [B, 3, H, W]
            B_, C, H, W = diff.shape
            norm_val = diff.view(B_, -1).norm(dim=1)    # [B]
            norm_val = norm_val / (C * H * W + 1e-8)   # 归一化
            step_list.append(norm_val)

        return torch.stack(step_list, dim=1)  # [B, K-1]

    def _compute_curvatures(
        self,
        states: List[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """
        计算轨迹曲率：二阶差分的 Frobenius 范数。

        Returns:
            ``[B, K-2]`` 曲率序列张量。
        """
        K = len(states)
        B = states[0].shape[0]
        curv_list = []

        for k in range(1, K - 1):
            second_diff = states[k + 1] - 2.0 * states[k] + states[k - 1]  # [B, 3, H, W]
            B_, C, H, W = second_diff.shape
            curv = second_diff.view(B_, -1).norm(dim=1)  # [B]
            curv = curv / (C * H * W + 1e-8)
            curv_list.append(curv)

        return torch.stack(curv_list, dim=1)  # [B, K-2]

    def _compress_sequence(
        self,
        seq: torch.Tensor,
        target_len: int,
    ) -> torch.Tensor:
        """
        将可变长度序列压缩为固定长度，使用线性插值。

        Args:
            seq: ``[B, L]`` 序列张量。
            target_len: 目标长度。

        Returns:
            ``[B, target_len]`` 压缩后张量。
        """
        B, L = seq.shape
        if L == target_len:
            return seq
        if L == 0:
            return torch.zeros(B, target_len, device=seq.device)

        # 用 F.interpolate 做线性插值
        seq_4d = seq.unsqueeze(1).unsqueeze(-1)  # [B, 1, L, 1]
        compressed = F.interpolate(
            seq_4d, size=(target_len, 1), mode='bilinear', align_corners=False
        )  # [B, 1, target_len, 1]
        return compressed.squeeze(1).squeeze(-1)  # [B, target_len]

    def _compute_global_stats(
        self,
        states: List[torch.Tensor],
        step_norms: torch.Tensor,
        curvatures: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """
        计算 8 维全局统计特征。

        Group 3 维度：
          0: total_path_length
          1: path_straightness = ||x_T - x_0||_F / total_path_length
          2: step_uniformity (CV = std/mean)
          3: max_step_ratio = max / mean
          4: early_vs_late = mean(first_half) / mean(second_half)
          5: curvature_mean
          6: curvature_max
          7: curvature_cv

        Returns:
            ``[B, 8]`` 全局统计张量。
        """
        B = step_norms.shape[0]
        eps = 1e-8

        # --- 步长相关统计 ---
        total_path = step_norms.sum(dim=1)                   # [B]
        step_mean = step_norms.mean(dim=1)                   # [B]
        step_std = step_norms.std(dim=1, unbiased=False)     # [B]
        step_max = step_norms.max(dim=1).values              # [B]

        step_uniformity = step_std / (step_mean + eps)       # CV
        max_step_ratio = step_max / (step_mean + eps)

        # early vs late：前半段 vs 后半段步长均值之比
        K_minus_1 = step_norms.shape[1]
        half = max(1, K_minus_1 // 2)
        early_mean = step_norms[:, :half].mean(dim=1)
        late_mean = step_norms[:, half:].mean(dim=1) if step_norms.shape[1] > half else early_mean
        early_vs_late = early_mean / (late_mean + eps)

        # path_straightness：终点和起点距离 / 总路径长度
        x0 = states[0]   # [B, 3, H, W]
        xT = states[-1]  # [B, 3, H, W]
        B_, C, H, W = x0.shape
        endpoint_dist = (xT - x0).view(B_, -1).norm(dim=1) / (C * H * W + eps)
        path_straightness = endpoint_dist / (total_path + eps)
        path_straightness = torch.clamp(path_straightness, 0.0, 10.0)

        # --- 曲率相关统计 ---
        if curvatures is not None and curvatures.shape[1] > 0:
            curv_mean = curvatures.mean(dim=1)
            curv_max = curvatures.max(dim=1).values
            curv_std = curvatures.std(dim=1, unbiased=False)
            curv_cv = curv_std / (curv_mean + eps)
        else:
            curv_mean = torch.zeros(B, device=device)
            curv_max = torch.zeros(B, device=device)
            curv_cv = torch.zeros(B, device=device)

        global_stats = torch.stack(
            [
                total_path,       # 0
                path_straightness,# 1
                step_uniformity,  # 2
                max_step_ratio,   # 3
                early_vs_late,    # 4
                curv_mean,        # 5
                curv_max,         # 6
                curv_cv,          # 7
            ],
            dim=1,
        )  # [B, 8]

        return global_stats

    def _compute_freq_evolution(
        self,
        states: List[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """
        计算轨迹频域演化特征：4 时间段 × 4 频带 = 16 维。

        对 K 个中间状态，分为 4 段，每段计算平均功率谱，
        再按径向距离划分 4 个频带取对数均值。

        Returns:
            ``[B, 16]`` 频域演化张量。
        """
        K = len(states)
        B = states[0].shape[0]
        H, W = states[0].shape[-2], states[0].shape[-1]

        num_segments = 4
        num_bands = 4

        freq_features = torch.zeros(B, num_segments * num_bands, device=device)

        # 划分时间段
        seg_size = max(1, K // num_segments)

        for seg_idx in range(num_segments):
            seg_start = seg_idx * seg_size
            seg_end = min(seg_start + seg_size, K)
            if seg_start >= K:
                break

            seg_states = states[seg_start:seg_end]

            # 对本时间段内所有状态计算平均 PSD
            # PSD shape: [B, 3, H, W//2+1]
            seg_psd_sum = torch.zeros(B, 3, H, W // 2 + 1, device=device)

            for state in seg_states:
                fft_result = torch.fft.rfft2(state)         # [B, 3, H, W//2+1]
                psd = fft_result.abs() ** 2                  # [B, 3, H, W//2+1]
                seg_psd_sum += psd

            seg_psd_avg = seg_psd_sum / (len(seg_states) + 1e-8)  # [B, 3, H, W//2+1]
            # 三通道平均
            seg_psd = seg_psd_avg.mean(dim=1)  # [B, H, W//2+1]

            # 按径向距离分 num_bands 个频带
            # 构建径向距离矩阵
            H_freq = H
            W_freq = W // 2 + 1
            fy = torch.arange(H_freq, device=device).float()
            fx = torch.arange(W_freq, device=device).float()
            # 将频率范围转换到 [0, 1]（归一化）
            fy = torch.minimum(fy, torch.tensor(H_freq, device=device).float() - fy) / (H_freq / 2.0)
            fx = fx / W_freq
            radial = torch.sqrt(fy.unsqueeze(1) ** 2 + fx.unsqueeze(0) ** 2)  # [H, W//2+1]
            radial = torch.clamp(radial, 0.0, math.sqrt(2.0))

            max_freq = radial.max().item()
            band_width = (max_freq + 1e-8) / num_bands

            for band_idx in range(num_bands):
                band_lo = band_idx * band_width
                band_hi = (band_idx + 1) * band_width
                mask = (radial >= band_lo) & (radial < band_hi)  # [H, W//2+1]

                if mask.sum() == 0:
                    feat_idx = seg_idx * num_bands + band_idx
                    freq_features[:, feat_idx] = 0.0
                    continue

                # [B, H, W//2+1] -> 对 mask 区域取均值
                masked_psd = seg_psd * mask.unsqueeze(0)          # [B, H, W//2+1]
                band_energy = masked_psd.sum(dim=(1, 2)) / (mask.sum().float() + 1e-8)  # [B]
                log_energy = torch.log(band_energy + 1e-8)         # [B]

                feat_idx = seg_idx * num_bands + band_idx
                freq_features[:, feat_idx] = log_energy

        return freq_features  # [B, 16]
