"""
models/features/frequency.py
-----------------------------
综合频域特征提取器（Frequency Feature Extractor）

整合端点 PSD 分析 + VAE 伪影检测 + 相机噪声检测，专门设计来检测不同类型的假图。

理论依据：
    - Branch A（径向 PSD）：GAN 生成图像在频域有特定模式（高频缺失/过多）
    - Branch B（VAE 伪影）：SD/DALL-E 类图像有 8px 周期性伪影（VAE patch 边界）
    - Branch C（相机噪声）：真实照片有传感器 PRNU，生成图像高频段"过于干净"
    - Branch D（相位一致性）：真实图像的相位分布更接近均匀分布

特征组成（feature_dim = 64）：
    Branch A - 径向 PSD 分析（16 维）
    Branch B - VAE 周期性检测（8 维）
    Branch C - 高频相机噪声分析（12 维）
    Branch D - 相位一致性（8 维）

    合计：16 + 8 + 12 + 8 = 44 维，padding 到 64 维
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, '/Users/joces/Downloads/Coding_Campus/Deepfake_Project')

from DRIFT.models.features.base import FeatureExtractor


class FrequencyFeatureExtractor(FeatureExtractor):
    """
    从 x_T 的频域结构提取多层次特征，专门设计来检测不同类型的假图。

    只需 x_T，intermediates 可以为 None。feature_dim = 64。
    """

    _BRANCH_A_DIM = 16   # 径向 PSD 分析
    _BRANCH_B_DIM = 8    # VAE 周期性检测
    _BRANCH_C_DIM = 12   # 高频相机噪声分析
    _BRANCH_D_DIM = 8    # 相位一致性
    _RAW_DIM = _BRANCH_A_DIM + _BRANCH_B_DIM + _BRANCH_C_DIM + _BRANCH_D_DIM  # 44
    _FEATURE_DIM = 64    # padding 到 64

    def __init__(self) -> None:
        super().__init__()

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
        从 x_T 的频域结构提取 64 维特征向量。

        Args:
            x_T: 终点噪声张量 ``[B, 3, H, W]``，值域任意（不要求归一化）。
            intermediates: 此提取器不使用，可以为 None。

        Returns:
            特征张量 ``[B, 64]``。
        """
        device = x_T.device
        x_T = x_T.float()

        # Branch A: 径向 PSD 分析（16 维）
        branch_a = self._branch_a_radial_psd(x_T)     # [B, 16]

        # Branch B: VAE 周期性检测（8 维）
        branch_b = self._branch_b_vae_artifact(x_T)   # [B, 8]

        # Branch C: 高频相机噪声分析（12 维）
        branch_c = self._branch_c_camera_noise(x_T)   # [B, 12]

        # Branch D: 相位一致性（8 维）
        branch_d = self._branch_d_phase(x_T)          # [B, 8]

        # 拼接 44 维
        raw = torch.cat([branch_a, branch_b, branch_c, branch_d], dim=1)  # [B, 44]

        # Padding 到 64 维
        pad_size = self._FEATURE_DIM - raw.shape[1]
        if pad_size > 0:
            features = F.pad(raw, (0, pad_size), mode='constant', value=0.0)
        else:
            features = raw[:, :self._FEATURE_DIM]

        # 防止 NaN/Inf
        features = torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
        features = torch.clamp(features, -100.0, 100.0)

        self.validate_output(features)
        return features

    # ------------------------------------------------------------------
    # Branch A: 径向 PSD 分析
    # ------------------------------------------------------------------

    def _branch_a_radial_psd(self, x_T: torch.Tensor) -> torch.Tensor:
        """
        对 x_T 做 rfft2，按径向距离分 16 个频带，每带取对数平均能量。

        全类型假图的通用特征——GAN 生成图像在频域的能量分布与真实图像不同。

        Returns:
            ``[B, 16]`` 径向 PSD 特征。
        """
        B, C, H, W = x_T.shape
        num_bands = self._BRANCH_A_DIM

        # 计算 PSD：[B, C, H, W//2+1]
        psd = torch.fft.rfft2(x_T).abs() ** 2
        # 三通道平均：[B, H, W//2+1]
        psd_mean = psd.mean(dim=1)

        # 径向距离矩阵
        H_freq = H
        W_freq = W // 2 + 1
        fy = torch.arange(H_freq, device=x_T.device).float()
        fx = torch.arange(W_freq, device=x_T.device).float()
        # 将频率折叠到 [0, 0.5]（归一化频率）
        fy = torch.minimum(fy, torch.tensor(float(H_freq), device=x_T.device) - fy) / float(H_freq // 2 + 1)
        fx = fx / float(W_freq)
        radial = torch.sqrt(fy.unsqueeze(1) ** 2 + fx.unsqueeze(0) ** 2)  # [H, W//2+1]

        max_freq = radial.max().item() + 1e-8
        band_width = max_freq / num_bands

        features = torch.zeros(B, num_bands, device=x_T.device)
        for i in range(num_bands):
            lo = i * band_width
            hi = (i + 1) * band_width
            mask = (radial >= lo) & (radial < hi)
            if mask.sum() == 0:
                continue
            masked = psd_mean * mask.unsqueeze(0)  # [B, H, W//2+1]
            band_energy = masked.sum(dim=(1, 2)) / (mask.sum().float() + 1e-8)
            features[:, i] = torch.log(band_energy + 1e-8)

        return features  # [B, 16]

    # ------------------------------------------------------------------
    # Branch B: VAE 周期性检测
    # ------------------------------------------------------------------

    def _branch_b_vae_artifact(self, x_T: torch.Tensor) -> torch.Tensor:
        """
        专门检测 SD/DALL-E 类假图的 8px VAE 伪影。

        提取 f=k/8（k=1,2,3,4）处的功率谱峰值（针对 H 和 W 方向）。
        SD 来源的假图在这些频率处有显著峰值。

        实现参考：
            H, W = x_T.shape[-2:]
            psd = torch.fft.rfft2(x_T).abs()**2
            h8_idx = H // 8
            peak_h8 = psd[..., h8_idx-1:h8_idx+2, :].mean(dim=-1).mean(dim=-1)

        Returns:
            ``[B, 8]`` VAE 伪影特征（H/W 方向各 4 个峰值）。
        """
        B, C, H, W = x_T.shape

        # 全局 PSD：[B, C, H, W//2+1]
        psd = torch.fft.rfft2(x_T).abs() ** 2
        # 三通道平均：[B, H, W//2+1]
        psd_mean = psd.mean(dim=1)

        # 全局 PSD 的均值和标准差（用于归一化）
        psd_flat = psd_mean.view(B, -1)                   # [B, H*(W//2+1)]
        global_mean = psd_flat.mean(dim=1, keepdim=True)  # [B, 1]
        global_std = psd_flat.std(dim=1, keepdim=True) + 1e-8  # [B, 1]

        h_features = []  # H 方向 4 个峰值
        w_features = []  # W 方向 4 个峰值

        for k in range(1, 5):  # k=1,2,3,4
            # H 方向：f = k/8，对应 H 方向频率 bin = H*k//8
            h_idx = H * k // 8
            h_lo = max(0, h_idx - 1)
            h_hi = min(H, h_idx + 2)

            # PSD[..., H方向频率bin, :] 沿 W 方向取均值
            peak_h = psd_mean[:, h_lo:h_hi, :].mean(dim=(1, 2))  # [B]
            # 归一化为 z-score
            peak_h_z = (peak_h - global_mean.squeeze(1)) / global_std.squeeze(1)
            h_features.append(peak_h_z)

            # W 方向：f = k/8，对应 W 方向频率 bin = (W//2+1)*k//8
            w_idx = (W // 2 + 1) * k // 8
            w_lo = max(0, w_idx - 1)
            w_hi = min(W // 2 + 1, w_idx + 2)

            peak_w = psd_mean[:, :, w_lo:w_hi].mean(dim=(1, 2))  # [B]
            peak_w_z = (peak_w - global_mean.squeeze(1)) / global_std.squeeze(1)
            w_features.append(peak_w_z)

        # H 方向 4 个峰值 + W 方向 4 个峰值 = 8 维
        features = torch.stack(h_features + w_features, dim=1)  # [B, 8]
        return features

    # ------------------------------------------------------------------
    # Branch C: 高频相机噪声分析
    # ------------------------------------------------------------------

    def _branch_c_camera_noise(self, x_T: torch.Tensor) -> torch.Tensor:
        """
        高频相机噪声分析，用于检测 ADM/扩散模型假图。

        理论依据：
            - 真实照片有传感器 PRNU（光响应非均匀性）和去马赛克伪影
            - 这些噪声模式在高频段（f > 0.4 * f_max）表现为特定的空间相关性
            - 生成图像高频段"过于干净"（扩散模型倾向于平滑）

        实现：
            对每个通道：
            1. 提取高频成分的均值和方差（2 个统计量）
            2. 计算高频残差的空间自相关（lag=1,2,4）（3 个统计量）
            3. 高频能量占比（1 个统计量）
            → 3 通道 × 4 统计量 = 12 维

        Returns:
            ``[B, 12]`` 相机噪声特征。
        """
        B, C, H, W = x_T.shape
        device = x_T.device

        features_list = []

        # 构建高频掩码（f > 0.4 * f_max）
        H_freq = H
        W_freq = W // 2 + 1
        fy = torch.arange(H_freq, device=device).float()
        fx = torch.arange(W_freq, device=device).float()
        fy = torch.minimum(fy, torch.tensor(float(H_freq), device=device) - fy) / float(H_freq // 2 + 1)
        fx = fx / float(W_freq)
        radial = torch.sqrt(fy.unsqueeze(1) ** 2 + fx.unsqueeze(0) ** 2)  # [H, W//2+1]
        max_freq = radial.max().item()
        hf_mask = (radial > 0.4 * max_freq)  # [H, W//2+1] 高频掩码
        lf_mask = ~hf_mask

        for c_idx in range(min(C, 3)):
            ch = x_T[:, c_idx, :, :]  # [B, H, W]
            fft_ch = torch.fft.rfft2(ch)  # [B, H, W//2+1]
            psd_ch = fft_ch.abs() ** 2    # [B, H, W//2+1]

            # 1. 高频 PSD 均值和方差
            hf_psd = psd_ch * hf_mask.unsqueeze(0)     # [B, H, W//2+1]
            hf_count = hf_mask.sum().float() + 1e-8
            hf_mean = hf_psd.sum(dim=(1, 2)) / hf_count      # [B]
            hf_var = ((hf_psd - hf_mean.unsqueeze(1).unsqueeze(2)) ** 2 * hf_mask.unsqueeze(0)).sum(dim=(1, 2)) / hf_count
            hf_mean_log = torch.log(hf_mean + 1e-8)
            hf_var_log = torch.log(hf_var + 1e-8)

            # 2. 高频能量占比
            total_energy = psd_ch.sum(dim=(1, 2)) + 1e-8
            hf_energy_ratio = hf_psd.sum(dim=(1, 2)) / total_energy  # [B]

            # 3. 高频残差的空间自相关（lag=1,2,4）
            # 提取高频残差：先做 iFFT 恢复高频空间图像
            hf_fft = fft_ch * hf_mask.unsqueeze(0)     # [B, H, W//2+1]
            hf_spatial = torch.fft.irfft2(hf_fft, s=(H, W))  # [B, H, W]

            # 归一化高频残差
            hf_mean_sp = hf_spatial.view(B, -1).mean(dim=1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]
            hf_std_sp = hf_spatial.view(B, -1).std(dim=1, keepdim=True).unsqueeze(-1) + 1e-8
            hf_norm = (hf_spatial - hf_mean_sp) / hf_std_sp  # [B, H, W]

            # lag=1 水平方向自相关
            autocorr_lag1 = (hf_norm[:, :, :-1] * hf_norm[:, :, 1:]).mean(dim=(1, 2))  # [B]

            # 这个通道的 4 个统计量
            ch_features = torch.stack(
                [hf_mean_log, hf_var_log, hf_energy_ratio, autocorr_lag1],
                dim=1,
            )  # [B, 4]

            features_list.append(ch_features)

        # 3 通道 × 4 = 12 维
        features = torch.cat(features_list, dim=1)  # [B, 12]
        return features

    # ------------------------------------------------------------------
    # Branch D: 相位一致性
    # ------------------------------------------------------------------

    def _branch_d_phase(self, x_T: torch.Tensor) -> torch.Tensor:
        """
        对 x_T 的 rfft2 结果，提取相位分布的熵和均匀性。

        理论：
            - 真实图像的相位分布更接近均匀分布（自然图像先验）
            - 生成图像的相位可能更集中（模型偏好）

        实现：
            - 全局相位均值和标准差（2 维）
            - 相位直方图（8 个 bin）的熵（1 维）
            - 低频段（中央区域）相位均匀性（1 维）
            - 高频段相位均匀性（1 维）

        Returns:
            ``[B, 8]`` 相位一致性特征。
        """
        B, C, H, W = x_T.shape
        device = x_T.device

        # 三通道平均 FFT
        x_T_mean = x_T.mean(dim=1)  # [B, H, W]
        fft_result = torch.fft.rfft2(x_T_mean)   # [B, H, W//2+1]

        # 提取相位角：[-pi, pi]
        phase = torch.angle(fft_result)           # [B, H, W//2+1]

        # 展平相位
        phase_flat = phase.view(B, -1)            # [B, N]

        # 1. 全局相位均值和标准差
        phase_mean = phase_flat.mean(dim=1)       # [B]
        phase_std = phase_flat.std(dim=1)         # [B]

        # 2. 相位直方图熵（8 个 bin 在 [-pi, pi] 上）
        num_bins = 8
        # 将 phase 归一化到 [0, 1]
        phase_norm = (phase_flat + math.pi) / (2 * math.pi + 1e-8)  # [0, 1]
        phase_norm = torch.clamp(phase_norm, 0.0, 1.0 - 1e-6)

        bin_indices = (phase_norm * num_bins).long()  # [B, N]
        bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)

        # 构建 one-hot 并累加
        phase_hist = torch.zeros(B, num_bins, device=device)
        phase_hist.scatter_add_(
            1,
            bin_indices,
            torch.ones_like(bin_indices, dtype=torch.float32),
        )
        # 归一化为概率
        phase_hist = phase_hist / (phase_hist.sum(dim=1, keepdim=True) + 1e-8)

        # 相位熵 H = -sum(p * log(p))
        phase_entropy = -(phase_hist * torch.log(phase_hist + 1e-10)).sum(dim=1)  # [B]

        # 3. 低频和高频区域的相位标准差（各 1 维）
        H_freq = H
        W_freq = W // 2 + 1
        fy = torch.arange(H_freq, device=device).float()
        fx = torch.arange(W_freq, device=device).float()
        fy = torch.minimum(fy, torch.tensor(float(H_freq), device=device) - fy) / float(H_freq // 2 + 1)
        fx = fx / float(W_freq)
        radial = torch.sqrt(fy.unsqueeze(1) ** 2 + fx.unsqueeze(0) ** 2)  # [H, W//2+1]
        max_freq = radial.max().item()

        lf_mask = (radial <= 0.3 * max_freq).unsqueeze(0)  # [1, H, W//2+1]
        hf_mask = (radial > 0.3 * max_freq).unsqueeze(0)   # [1, H, W//2+1]

        # 低频相位标准差
        lf_phase = phase * lf_mask
        lf_count = lf_mask.sum().float() + 1e-8
        lf_mean = lf_phase.sum(dim=(1, 2)) / lf_count         # [B]
        lf_var = ((lf_phase - lf_mean.unsqueeze(1).unsqueeze(2)) ** 2 * lf_mask).sum(dim=(1, 2)) / lf_count
        lf_std = torch.sqrt(lf_var + 1e-8)                    # [B]

        # 高频相位标准差
        hf_phase = phase * hf_mask
        hf_count = hf_mask.sum().float() + 1e-8
        hf_mean = hf_phase.sum(dim=(1, 2)) / hf_count         # [B]
        hf_var = ((hf_phase - hf_mean.unsqueeze(1).unsqueeze(2)) ** 2 * hf_mask).sum(dim=(1, 2)) / hf_count
        hf_std = torch.sqrt(hf_var + 1e-8)                    # [B]

        # 4. 相位最大幅度（频率加权）
        magnitude = fft_result.abs()  # [B, H, W//2+1]
        # 幅度加权相位散布（对相位分布非均匀性的度量）
        mag_sum = magnitude.sum(dim=(1, 2)) + 1e-8
        weighted_phase_abs = (magnitude * phase.abs()).sum(dim=(1, 2)) / mag_sum  # [B]

        # 5. 相位谱中央零频成分占比
        dc_power = fft_result[:, 0, 0].abs() ** 2                           # [B]
        total_power = (fft_result.abs() ** 2).sum(dim=(1, 2)) + 1e-8        # [B]
        dc_ratio = dc_power / total_power                                     # [B]

        features = torch.stack(
            [
                phase_mean,           # 0: 相位均值
                phase_std,            # 1: 相位标准差
                phase_entropy,        # 2: 相位熵
                lf_std,               # 3: 低频相位标准差
                hf_std,               # 4: 高频相位标准差
                weighted_phase_abs,   # 5: 幅度加权相位散布
                dc_ratio,             # 6: DC 成分占比
                hf_std / (lf_std + 1e-8),  # 7: 高/低频相位标准差之比
            ],
            dim=1,
        )  # [B, 8]

        return features
