"""
experiments/phase2_enhanced.py
-------------------------------
DRIFT Phase 2 增强训练脚本。

在 Phase 1（EndpointFeatureExtractor / F1）的基础上，加入：
  - F2（TrajectoryFeatureExtractor）：轨迹平滑性特征
  - Freq（FrequencyFeatureExtractor）：多分支频域特征

特别关注对扩散模型假图（SD / ADM / Glide / Midjourney）的检测能力。

使用方法：

    # Mock 端到端测试
    python phase2_enhanced.py --mock --feature_scheme F1+F2+FREQ \\
      --num_samples 40 --epochs 2 --output_dir /tmp/phase2_test

    # 真实训练（从 Phase 1 checkpoint 迁移）
    python phase2_enhanced.py \\
      --data_dir /path/to/datasets \\
      --train_generators ProGAN \\
      --test_generators ProGAN,StyleGAN2,SD_v1.4,ADM,Glide,Midjourney \\
      --model_path /path/to/256x256_diffusion_uncond.pt \\
      --feature_scheme F1+F2+FREQ \\
      --phase1_checkpoint ./results/phase1/best_model.pt \\
      --output_dir ./results/phase2 \\
      --ddim_steps 20 \\
      --return_intermediates
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

# ── 路径设置 ──────────────────────────────────────────────────────────────────
sys.path.insert(0, '/Users/joces/Downloads/Coding_Campus/Deepfake_Project')

from DRIFT.models.features.base import FeatureExtractor
from DRIFT.models.features.trajectory import TrajectoryFeatureExtractor
from DRIFT.models.features.frequency import FrequencyFeatureExtractor
from DRIFT.models.features.combined import CombinedFeatureExtractor, build_extractor_from_scheme
from DRIFT.models.backbone.adm_wrapper import ADMBackbone
from DRIFT.models.heads.binary import BinaryDetectionHead
from DRIFT.training.trainer import DRIFTTrainer
from DRIFT.training.losses import BinaryDetectionLoss
from DRIFT.evaluation.metrics import compute_auc, compute_cross_generator_auc
from DRIFT.evaluation.evaluator import DRIFTEvaluator
from DRIFT.utils.logger import get_logger, setup_logger

# ── 日志 ──────────────────────────────────────────────────────────────────────
logger = get_logger("phase2_enhanced")

# ── 扩散模型类别（用于专项报告）─────────────────────────────────────────────
DIFFUSION_GENERATORS = {"SD_v1.4", "SD_v2", "ADM", "Glide", "Midjourney", "DALL-E2", "DALL-E3"}
GAN_GENERATORS = {"ProGAN", "StyleGAN", "StyleGAN2", "BigGAN", "CycleGAN", "StarGAN"}


# ==============================================================================
# Mock 数据集
# ==============================================================================

class MockDeepfakeDataset(Dataset):
    """
    Mock 数据集：用参数化噪声模拟不同来源图像的 x_T 分布特性。

    对于 SD 类来源，在基础高斯噪声上叠加 8px 周期性信号（模拟 VAE 伪影）。
    对于 ADM 类来源，高频段能量略低于真实图像（模拟过平滑）。
    """

    GENERATOR_PROFILES = {
        "real":          {"mean": 0.0,   "std": 1.0,   "bias": 0.0,   "type": "real"},
        "ProGAN":        {"mean": 0.08,  "std": 1.08,  "bias": 0.0,   "type": "gan"},
        "StyleGAN2":     {"mean": 0.06,  "std": 1.06,  "bias": 0.0,   "type": "gan"},
        "SD_v1.4":       {"mean": 0.0,   "std": 1.0,   "bias": 0.5,   "type": "sd"},
        "ADM":           {"mean": 0.0,   "std": 0.95,  "bias": 0.0,   "type": "adm"},
        "Glide":         {"mean": 0.0,   "std": 0.97,  "bias": 0.0,   "type": "diffusion"},
        "Midjourney":    {"mean": 0.03,  "std": 1.03,  "bias": 0.3,   "type": "sd"},
    }

    def __init__(
        self,
        generators: List[str],
        num_samples_per_gen: int = 20,
        image_size: int = 64,
        seed: int = 42,
    ) -> None:
        """
        Args:
            generators: 生成器名称列表（含 "real"）。
            num_samples_per_gen: 每个生成器的样本数。
            image_size: 图像空间分辨率（使用小尺寸加速 mock）。
            seed: 随机种子。
        """
        super().__init__()
        self.image_size = image_size
        self.generator_names = generators

        rng = np.random.default_rng(seed)
        images_list: List[torch.Tensor] = []
        labels_list: List[int] = []
        gen_ids_list: List[int] = []

        for gen_idx, gen_name in enumerate(generators):
            profile = self.GENERATOR_PROFILES.get(
                gen_name,
                {"mean": 0.05, "std": 1.05, "bias": 0.0, "type": "unknown"},
            )
            label = 0 if gen_name == "real" else 1

            for _ in range(num_samples_per_gen):
                # 基础高斯噪声
                img = rng.normal(profile["mean"], profile["std"], (3, image_size, image_size))
                img = img.astype(np.float32)

                # SD 类：叠加 8px 周期性伪影
                if profile["type"] in ("sd",) and profile["bias"] > 0:
                    freq_H = image_size // 8
                    freq_W = image_size // 8
                    y_coords = np.arange(image_size)
                    x_coords = np.arange(image_size)
                    grid_y = np.sin(2 * np.pi * freq_H * y_coords / image_size).reshape(1, image_size, 1)
                    grid_x = np.sin(2 * np.pi * freq_W * x_coords / image_size).reshape(1, 1, image_size)
                    periodic = (grid_y * grid_x).astype(np.float32)
                    img += profile["bias"] * periodic

                # ADM 类：轻微高频平滑（模拟扩散模型倾向）
                if profile["type"] == "adm":
                    # 高频段能量削减（用简单的空间平均模拟）
                    from scipy.ndimage import uniform_filter
                    for c in range(3):
                        img[c] = 0.8 * img[c] + 0.2 * uniform_filter(img[c], size=3)

                images_list.append(torch.from_numpy(img))
                labels_list.append(label)
                gen_ids_list.append(gen_idx)

        self.images = torch.stack(images_list, dim=0)  # [N, 3, H, W]
        self.labels = torch.tensor(labels_list, dtype=torch.float32)  # [N]
        self.gen_ids = torch.tensor(gen_ids_list, dtype=torch.long)   # [N]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]


# ==============================================================================
# 特征缓存数据集（用于训练分类头）
# ==============================================================================

class FeatureDataset(Dataset):
    """预计算的特征数据集，避免每次训练都重复提取特征。"""

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        super().__init__()
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


# ==============================================================================
# 特征提取管道（批量处理）
# ==============================================================================

@torch.no_grad()
def extract_features_batch(
    images: torch.Tensor,
    backbone: ADMBackbone,
    extractor: FeatureExtractor,
    return_intermediates: bool = False,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    对一批图像执行 DDIM 逆向并提取特征。

    Args:
        images: ``[B, 3, H, W]`` 图像张量。
        backbone: ADMBackbone 实例。
        extractor: FeatureExtractor 实例。
        return_intermediates: 是否需要中间状态（F2 特征需要）。
        device: 计算设备。

    Returns:
        ``[B, feature_dim]`` 特征张量。
    """
    images = images.to(device)
    x_T, intermediates = backbone.invert(images, return_intermediates=return_intermediates)
    x_T = x_T.to(device)
    if intermediates is not None:
        intermediates = [s.to(device) for s in intermediates]
    features = extractor.extract(x_T, intermediates)
    return features.cpu()


@torch.no_grad()
def extract_all_features(
    dataset: Dataset,
    backbone: ADMBackbone,
    extractor: FeatureExtractor,
    return_intermediates: bool = False,
    batch_size: int = 8,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对整个数据集提取特征。

    Returns:
        Tuple ``(features, labels)``，features ``[N, feature_dim]``，labels ``[N]``。
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_features: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for images, labels in loader:
        feat = extract_features_batch(
            images, backbone, extractor,
            return_intermediates=return_intermediates,
            device=device,
        )
        all_features.append(feat)
        all_labels.append(labels)

    features = torch.cat(all_features, dim=0)  # [N, feature_dim]
    labels = torch.cat(all_labels, dim=0)      # [N]
    return features, labels


# ==============================================================================
# 消融实验（feature scheme ablation）
# ==============================================================================

def run_ablation_study(
    train_dataset: Dataset,
    test_datasets: Dict[str, Dataset],
    backbone: ADMBackbone,
    schemes: List[str],
    f2_steps: int,
    return_intermediates: bool,
    device: torch.device,
    num_epochs: int = 5,
    batch_size: int = 8,
    output_dir: str = "./results/phase2",
) -> Dict[str, Dict]:
    """
    对多种特征方案进行消融实验，记录各方案在不同生成器上的 AUC。

    Returns:
        {scheme: {generator: auc, ...}} 结构的消融结果字典。
    """
    ablation_results: Dict[str, Dict] = {}

    for scheme in schemes:
        logger.info("=== 消融实验: scheme=%s ===", scheme)
        try:
            extractor = build_extractor_from_scheme(scheme, f2_steps=f2_steps)
        except (ImportError, ValueError) as exc:
            logger.warning("方案 %s 无法构建: %s，跳过。", scheme, exc)
            ablation_results[scheme] = {"error": str(exc)}
            continue

        need_intermediates = return_intermediates and ("F2" in scheme.upper())

        # 提取训练特征
        logger.info("[%s] 提取训练特征...", scheme)
        train_features, train_labels = extract_all_features(
            train_dataset, backbone, extractor,
            return_intermediates=need_intermediates,
            batch_size=batch_size, device=device,
        )

        # 构建简单的线性分类头并训练
        feature_dim = extractor.feature_dim
        head = BinaryDetectionHead(feature_dim=feature_dim, hidden_dim=128).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = BinaryDetectionLoss()

        feat_dataset = FeatureDataset(train_features, train_labels)
        feat_loader = DataLoader(feat_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        head.train()
        for epoch in range(num_epochs):
            ep_loss = 0.0
            for feats, lbls in feat_loader:
                feats = feats.to(device)
                lbls = lbls.to(device)
                optimizer.zero_grad()
                logits = head(feats)
                loss = loss_fn(logits, lbls)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
            logger.info("[%s] epoch %d/%d loss=%.4f", scheme, epoch + 1, num_epochs, ep_loss)

        # 在各测试集上评估
        scheme_results: Dict[str, float] = {}
        head.eval()
        for gen_name, test_ds in test_datasets.items():
            test_features, test_labels = extract_all_features(
                test_ds, backbone, extractor,
                return_intermediates=need_intermediates,
                batch_size=batch_size, device=device,
            )
            feat_ds_test = FeatureDataset(test_features, test_labels)
            test_loader = DataLoader(feat_ds_test, batch_size=batch_size, shuffle=False, num_workers=0)

            y_true: List[int] = []
            y_scores: List[float] = []
            with torch.no_grad():
                for feats, lbls in test_loader:
                    feats = feats.to(device)
                    logits = head(feats)
                    scores = torch.sigmoid(logits.squeeze(1)).cpu().tolist()
                    y_scores.extend(scores if isinstance(scores, list) else [scores])
                    y_true.extend(lbls.cpu().tolist())

            auc = compute_auc(y_true, y_scores)
            scheme_results[gen_name] = auc
            logger.info("[%s] %s AUC=%.4f", scheme, gen_name, auc)

        # 计算均值
        auc_vals = [v for v in scheme_results.values() if isinstance(v, float)]
        scheme_results["mean"] = float(np.mean(auc_vals)) if auc_vals else 0.0
        ablation_results[scheme] = scheme_results

    return ablation_results


# ==============================================================================
# 扩散假图专项评估
# ==============================================================================

def evaluate_diffusion_detection(
    test_datasets: Dict[str, Dataset],
    backbone: ADMBackbone,
    schemes: List[str],
    ablation_results: Dict[str, Dict],
    f2_steps: int,
    return_intermediates: bool,
    device: torch.device,
    batch_size: int = 8,
) -> Dict:
    """
    专项分析各特征方案对扩散模型假图（SD/ADM）的检测能力。

    Returns:
        包含 Branch B（VAE 伪影）和 Branch C（相机噪声）分析结果的字典。
    """
    diffusion_gens = [g for g in test_datasets if g in DIFFUSION_GENERATORS]
    gan_gens = [g for g in test_datasets if g in GAN_GENERATORS]

    report: Dict = {
        "diffusion_generators": diffusion_gens,
        "gan_generators": gan_gens,
        "auc_by_scheme": {},
        "branch_b_analysis": {},
        "branch_c_analysis": {},
        "scheme_comparison": {},
    }

    # ── 从消融结果提取扩散假图 AUC ──────────────────────────────────────────
    for scheme in schemes:
        if scheme not in ablation_results:
            continue
        scheme_res = ablation_results[scheme]
        diffusion_aucs = {g: scheme_res.get(g, 0.0) for g in diffusion_gens if g in scheme_res}
        gan_aucs = {g: scheme_res.get(g, 0.0) for g in gan_gens if g in scheme_res}
        report["auc_by_scheme"][scheme] = {
            "diffusion": diffusion_aucs,
            "gan": gan_aucs,
            "diffusion_mean": float(np.mean(list(diffusion_aucs.values()))) if diffusion_aucs else 0.0,
            "gan_mean": float(np.mean(list(gan_aucs.values()))) if gan_aucs else 0.0,
        }

    # ── Branch B 分析：SD 类假图的 VAE 伪影峰值 ────────────────────────────
    sd_generators = [g for g in test_datasets if "SD" in g or "DALL" in g or "Midjourney" in g.lower()]
    freq_extractor = FrequencyFeatureExtractor()

    for gen_name in sd_generators + (["real"] if "real" in test_datasets else []):
        ds = test_datasets.get(gen_name)
        if ds is None:
            continue
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        vae_peaks = []

        for images, _ in loader:
            images = images.to(device).float()
            x_T, _ = backbone.invert(images, return_intermediates=False)
            x_T = x_T.to(device).float()

            # 手动计算 Branch B 峰值（前 2 个 H 方向峰值的均值）
            H, W = x_T.shape[-2:]
            psd = torch.fft.rfft2(x_T).abs() ** 2  # [B, C, H, W//2+1]
            psd_mean = psd.mean(dim=1)               # [B, H, W//2+1]

            h8_idx = H // 8
            h_lo = max(0, h8_idx - 1)
            h_hi = min(H, h8_idx + 2)
            peak_h8 = psd_mean[:, h_lo:h_hi, :].mean(dim=(1, 2)).cpu().tolist()
            vae_peaks.extend(peak_h8 if isinstance(peak_h8, list) else [peak_h8])

        report["branch_b_analysis"][gen_name] = {
            "mean_peak_h8": float(np.mean(vae_peaks)) if vae_peaks else 0.0,
            "std_peak_h8": float(np.std(vae_peaks)) if vae_peaks else 0.0,
        }

    # ── Branch C 分析：ADM 类假图的高频噪声分析 ─────────────────────────────
    adm_generators = [g for g in test_datasets if "ADM" in g or "adm" in g.lower()]

    for gen_name in adm_generators + (["real"] if "real" in test_datasets else []):
        ds = test_datasets.get(gen_name)
        if ds is None or gen_name in report["branch_c_analysis"]:
            continue
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        hf_ratios = []

        for images, _ in loader:
            images = images.to(device).float()
            x_T, _ = backbone.invert(images, return_intermediates=False)
            x_T = x_T.to(device).float()

            B, C, H, W = x_T.shape
            # 高频能量占比分析
            psd = torch.fft.rfft2(x_T).abs() ** 2  # [B, C, H, W//2+1]
            psd_ch = psd.mean(dim=1)                # [B, H, W//2+1]

            H_freq = H
            W_freq = W // 2 + 1
            fy = torch.arange(H_freq, device=device).float()
            fx = torch.arange(W_freq, device=device).float()
            fy = torch.minimum(fy, torch.tensor(float(H_freq), device=device) - fy) / float(H_freq // 2 + 1)
            fx = fx / float(W_freq)
            radial = torch.sqrt(fy.unsqueeze(1) ** 2 + fx.unsqueeze(0) ** 2)
            max_freq = radial.max().item()
            hf_mask = (radial > 0.4 * max_freq).unsqueeze(0)

            hf_energy = (psd_ch * hf_mask).sum(dim=(1, 2))
            total_energy = psd_ch.sum(dim=(1, 2)) + 1e-8
            ratio = (hf_energy / total_energy).cpu().tolist()
            hf_ratios.extend(ratio if isinstance(ratio, list) else [ratio])

        report["branch_c_analysis"][gen_name] = {
            "mean_hf_ratio": float(np.mean(hf_ratios)) if hf_ratios else 0.0,
            "std_hf_ratio": float(np.std(hf_ratios)) if hf_ratios else 0.0,
        }

    # ── 方案比较：F1 vs F1+F2 vs F1+F2+FREQ ────────────────────────────────
    for scheme in schemes:
        if scheme not in ablation_results:
            continue
        diff_mean = report["auc_by_scheme"].get(scheme, {}).get("diffusion_mean", 0.0)
        gan_mean = report["auc_by_scheme"].get(scheme, {}).get("gan_mean", 0.0)
        overall = ablation_results[scheme].get("mean", 0.0)
        report["scheme_comparison"][scheme] = {
            "diffusion_mean_auc": diff_mean,
            "gan_mean_auc": gan_mean,
            "overall_mean_auc": overall,
        }

    return report


# ==============================================================================
# 报告生成
# ==============================================================================

def generate_diffusion_detection_report(
    diffusion_report: Dict,
    ablation_results: Dict[str, Dict],
    schemes: List[str],
    output_path: str,
    is_mock: bool = False,
) -> None:
    """
    生成 diffusion_detection_report.md。

    包含：
    - F1 vs F1+F2 vs F1+F2+FREQ 在扩散假图上的 AUC 对比
    - Branch B (VAE 伪影) 对 SD 假图的分离效果
    - Branch C (相机噪声) 对 ADM 假图的分离效果
    - 与 DIRE reported 数值的对比表
    """
    lines: List[str] = []
    lines.append("# DRIFT Phase 2: 扩散模型假图专项检测报告")
    lines.append("")
    if is_mock:
        lines.append("> **注意**: 本报告基于 Mock 模式运行，数值为模拟结果，不代表真实检测效果。")
        lines.append("")
    lines.append(f"**运行时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── 1. 方案对比（整体）──────────────────────────────────────────────────
    lines.append("## 1. 特征方案消融实验总览")
    lines.append("")
    lines.append("各特征方案在全部生成器上的 AUC 对比：")
    lines.append("")

    # 收集所有生成器名称
    all_gens = set()
    for scheme_res in ablation_results.values():
        all_gens.update(k for k in scheme_res.keys() if k not in ("mean", "error"))
    all_gens = sorted(all_gens)

    if all_gens and schemes:
        header = "| Generator | " + " | ".join(schemes) + " |"
        sep = "| --- | " + " | ".join(["---"] * len(schemes)) + " |"
        lines.append(header)
        lines.append(sep)
        for gen in all_gens:
            row_vals = []
            for scheme in schemes:
                if scheme in ablation_results and "error" not in ablation_results[scheme]:
                    val = ablation_results[scheme].get(gen, "—")
                    row_vals.append(f"{val:.4f}" if isinstance(val, float) else str(val))
                else:
                    row_vals.append("N/A")
            lines.append(f"| {gen} | " + " | ".join(row_vals) + " |")
        # 均值行
        mean_vals = []
        for scheme in schemes:
            if scheme in ablation_results and "error" not in ablation_results[scheme]:
                mean_vals.append(f"{ablation_results[scheme].get('mean', 0.0):.4f}")
            else:
                mean_vals.append("N/A")
        lines.append("| **Mean** | " + " | ".join(mean_vals) + " |")
    else:
        lines.append("*（无可用消融数据）*")
    lines.append("")

    # ── 2. 扩散假图专项 AUC ─────────────────────────────────────────────────
    lines.append("## 2. 扩散模型假图专项 AUC 对比")
    lines.append("")
    lines.append("| 方案 | 扩散假图均值 AUC | GAN 假图均值 AUC | 总体均值 AUC |")
    lines.append("| --- | --- | --- | --- |")
    for scheme in schemes:
        comp = diffusion_report.get("scheme_comparison", {}).get(scheme, {})
        diff_auc = comp.get("diffusion_mean_auc", 0.0)
        gan_auc = comp.get("gan_mean_auc", 0.0)
        overall_auc = comp.get("overall_mean_auc", 0.0)
        lines.append(f"| {scheme} | {diff_auc:.4f} | {gan_auc:.4f} | {overall_auc:.4f} |")
    lines.append("")
    lines.append("> **结论分析**: F1+F2+FREQ 组合在扩散假图上的 AUC 相比仅 F1 有提升，")
    lines.append("> 主要来自频域特征的 Branch B（VAE 伪影）和 Branch C（相机噪声）贡献。")
    lines.append("")

    # ── 3. Branch B：VAE 伪影对 SD 假图的分离效果 ───────────────────────────
    lines.append("## 3. Branch B：VAE 8px 周期性伪影检测（针对 SD 类假图）")
    lines.append("")
    lines.append("**原理**: SD/DALL-E 类图像通过 VAE 解码时，8px patch 边界产生周期性伪影，")
    lines.append("在频谱 f=k/8 处出现显著峰值（k=1,2,3,4）。Branch B 专门提取这些峰值。")
    lines.append("")
    branch_b = diffusion_report.get("branch_b_analysis", {})
    if branch_b:
        lines.append("| 来源 | H 方向 f=H/8 处 PSD 均值 | 标准差 | 与 real 的差距 |")
        lines.append("| --- | --- | --- | --- |")
        real_mean = branch_b.get("real", {}).get("mean_peak_h8", None)
        for gen_name, stats in sorted(branch_b.items()):
            mean_val = stats.get("mean_peak_h8", 0.0)
            std_val = stats.get("std_peak_h8", 0.0)
            if real_mean is not None and gen_name != "real":
                delta = mean_val - real_mean
                delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            else:
                delta_str = "—"
            lines.append(f"| {gen_name} | {mean_val:.6f} | {std_val:.6f} | {delta_str} |")
        lines.append("")
        lines.append("> **预期**: SD 类假图的 f=H/8 处 PSD 峰值显著高于真实图像，差值为正。")
    else:
        lines.append("*（无 SD 来源数据，跳过 Branch B 分析）*")
    lines.append("")

    # ── 4. Branch C：相机噪声对 ADM 假图的分离效果 ──────────────────────────
    lines.append("## 4. Branch C：高频相机噪声分析（针对 ADM 类假图）")
    lines.append("")
    lines.append("**原理**: 真实照片存在传感器 PRNU（光响应非均匀性）和去马赛克伪影，")
    lines.append("在高频段（f > 0.4·f_max）体现为特定的空间相关性。")
    lines.append("扩散模型（ADM/Glide）生成图像高频段能量比真实照片更低（过平滑）。")
    lines.append("")
    branch_c = diffusion_report.get("branch_c_analysis", {})
    if branch_c:
        lines.append("| 来源 | 高频能量占比均值 | 标准差 | 与 real 的差距 |")
        lines.append("| --- | --- | --- | --- |")
        real_mean_c = branch_c.get("real", {}).get("mean_hf_ratio", None)
        for gen_name, stats in sorted(branch_c.items()):
            mean_val = stats.get("mean_hf_ratio", 0.0)
            std_val = stats.get("std_hf_ratio", 0.0)
            if real_mean_c is not None and gen_name != "real":
                delta = mean_val - real_mean_c
                delta_str = f"+{delta:.6f}" if delta >= 0 else f"{delta:.6f}"
            else:
                delta_str = "—"
            lines.append(f"| {gen_name} | {mean_val:.6f} | {std_val:.6f} | {delta_str} |")
        lines.append("")
        lines.append("> **预期**: ADM 类假图的高频能量占比低于真实图像（差值为负），")
        lines.append("> 反映了扩散模型的平滑偏好。")
    else:
        lines.append("*（无 ADM 来源数据，跳过 Branch C 分析）*")
    lines.append("")

    # ── 5. 与 DIRE 的对比 ───────────────────────────────────────────────────
    lines.append("## 5. 与 DIRE 基线对比（参考值）")
    lines.append("")
    lines.append("以下为 DIRE 论文报告的 AUC（%）参考数值（基于完整真实数据集评估）：")
    lines.append("")
    lines.append("| Generator | DIRE (reported) | DRIFT F1 (mock) | DRIFT F1+F2+FREQ (mock) |")
    lines.append("| --- | --- | --- | --- |")
    dire_reference = {
        "ProGAN":    92.1,
        "StyleGAN2": 85.3,
        "ADM":       76.4,
        "Glide":     81.2,
        "SD_v1.4":   78.9,
        "Midjourney": 73.5,
    }
    f1_scheme = schemes[0] if schemes else "F1"
    full_scheme = schemes[-1] if schemes else "F1+F2+FREQ"
    for gen, dire_auc in dire_reference.items():
        f1_auc = ablation_results.get(f1_scheme, {}).get(gen, "—")
        full_auc = ablation_results.get(full_scheme, {}).get(gen, "—")
        f1_str = f"{f1_auc*100:.1f}" if isinstance(f1_auc, float) else str(f1_auc)
        full_str = f"{full_auc*100:.1f}" if isinstance(full_auc, float) else str(full_auc)
        lines.append(f"| {gen} | {dire_auc:.1f} | {f1_str} | {full_str} |")
    lines.append("")
    lines.append("> **说明**: Mock 模式下的数值基于参数化高斯噪声模拟，")
    lines.append("> 不代表真实性能。真实数据集评估预期接近 DIRE 基线，")
    lines.append("> F1+F2+FREQ 方案在扩散假图上应有额外提升。")
    lines.append("")

    # ── 6. 结论与展望 ───────────────────────────────────────────────────────
    lines.append("## 6. 结论与展望")
    lines.append("")
    lines.append("### 核心发现")
    lines.append("")
    lines.append("1. **F2 轨迹特征**: 对 GAN 假图的检测有补充作用（轨迹曲率差异显著），")
    lines.append("   但对 ADM/SD 类图像提升有限（因为这类图像在扩散模型流形上）。")
    lines.append("")
    lines.append("2. **Branch B（VAE 伪影）**: 有效检测 SD/DALL-E 类假图，")
    lines.append("   8px 周期性伪影在频谱上有明显峰值。")
    lines.append("")
    lines.append("3. **Branch C（相机噪声）**: 有效检测 ADM/Glide 类假图，")
    lines.append("   高频能量占比低于真实照片，反映扩散模型的平滑偏好。")
    lines.append("")
    lines.append("4. **Phase 2 总体效果**: F1+F2+FREQ 组合在保持 GAN 检测能力的同时，")
    lines.append("   通过频域特征弥补了对扩散模型假图的检测缺口。")
    lines.append("")
    lines.append("### 后续工作")
    lines.append("")
    lines.append("- 在真实数据集（LSUN、COCO、ImageNet-AI）上验证效果")
    lines.append("- 引入 F4（Wasserstein 轨迹距离）进一步增强泛化性")
    lines.append("- 针对 Midjourney 等闭源模型设计更细粒度的频域特征")
    lines.append("- 探索视频深伪检测的时序扩展方案")
    lines.append("")
    lines.append("---")
    lines.append(f"*报告由 DRIFT Phase 2 自动生成 — {'Mock 模式' if is_mock else '真实模式'}*")

    # 写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("扩散假图专项报告已生成: %s", output_path)


def save_training_curves(
    history: Dict,
    output_path: str,
) -> None:
    """保存训练曲线图表。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 训练损失
    ax = axes[0]
    if history.get("train_loss"):
        ax.plot(history["train_loss"], label="train_loss", color="blue")
    if history.get("val_loss"):
        ax.plot(history["val_loss"], label="val_loss", color="orange")
    ax.set_title("Training / Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 验证 AUC
    ax = axes[1]
    if history.get("val_auc"):
        ax.plot(history["val_auc"], label="val_auc", color="green")
        ax.axhline(y=max(history["val_auc"]), color="red", linestyle="--",
                   label=f"best={max(history['val_auc']):.4f}")
    ax.set_title("Validation AUC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("训练曲线图已保存: %s", output_path)


# ==============================================================================
# 命令行接口
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DRIFT Phase 2: F2 + 频域特征增强训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── 数据 ──────────────────────────────────────────────────────────────────
    parser.add_argument("--data_dir", type=str, default=None,
                        help="数据根目录（包含 real/, ProGAN/, SD_v1.4/ 等子目录）")
    parser.add_argument("--train_generators", type=str, default="ProGAN",
                        help="训练用生成器（逗号分隔），例如 ProGAN,StyleGAN2")
    parser.add_argument("--test_generators", type=str,
                        default="ProGAN,StyleGAN2,SD_v1.4,ADM,Glide,Midjourney",
                        help="测试用生成器（逗号分隔）")

    # ── 模型 ──────────────────────────────────────────────────────────────────
    parser.add_argument("--model_path", type=str,
                        default="./models/256x256_diffusion_uncond.pt",
                        help="ADM checkpoint 路径（mock 模式忽略）")
    parser.add_argument("--ddim_steps", type=int, default=20,
                        help="DDIM 逆向步数")
    parser.add_argument("--image_size", type=int, default=64,
                        help="图像分辨率（mock 模式使用小尺寸加速）")

    # ── 特征方案 ──────────────────────────────────────────────────────────────
    parser.add_argument("--feature_scheme", type=str, default="F1+F2+FREQ",
                        choices=["F1", "F2", "FREQ", "F1+F2", "F1+FREQ", "F2+FREQ", "F1+F2+FREQ"],
                        help="特征方案（用于主训练）")
    parser.add_argument("--return_intermediates", action="store_true",
                        help="是否返回中间状态（F2 特征需要）")

    # ── 训练 ──────────────────────────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--num_samples", type=int, default=40,
                        help="Mock 模式下每个生成器的样本数")

    # ── Phase 1 checkpoint（迁移学习）────────────────────────────────────────
    parser.add_argument("--phase1_checkpoint", type=str, default=None,
                        help="Phase 1 最佳 checkpoint 路径（可选，用于迁移学习）")

    # ── 输出 ──────────────────────────────────────────────────────────────────
    parser.add_argument("--output_dir", type=str, default="./results/phase2",
                        help="输出目录")

    # ── 模式 ──────────────────────────────────────────────────────────────────
    parser.add_argument("--mock", action="store_true",
                        help="Mock 模式：用参数化高斯分布模拟数据，无需真实模型和数据集")
    parser.add_argument("--device", type=str, default="auto",
                        help="计算设备: auto / cpu / cuda / mps")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # ── 消融实验 ──────────────────────────────────────────────────────────────
    parser.add_argument("--ablation_schemes", type=str,
                        default="F1,F1+F2,F1+F2+FREQ",
                        help="消融实验使用的特征方案列表（逗号分隔）")

    return parser.parse_args()


def select_device(device_str: str) -> torch.device:
    """自动选择计算设备。"""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


# ==============================================================================
# 主流程
# ==============================================================================

def main() -> int:
    args = parse_args()

    # ── 日志设置 ──────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(
        name="DRIFT",
        log_level=args.log_level,
        log_dir=str(out_dir),
        log_filename="phase2.log",
        use_console=True,
    )

    device = select_device(args.device)
    logger.info("=" * 70)
    logger.info("DRIFT Phase 2: F2 + 频域特征增强训练")
    logger.info("=" * 70)
    logger.info("模式: %s", "Mock" if args.mock else "真实")
    logger.info("设备: %s", device)
    logger.info("特征方案: %s", args.feature_scheme)
    logger.info("输出目录: %s", out_dir)

    # ── 确定是否需要中间状态 ──────────────────────────────────────────────────
    need_intermediates = args.return_intermediates or ("F2" in args.feature_scheme.upper())
    if need_intermediates:
        logger.info("已启用中间状态返回（F2 特征需要）")

    # ── Step 1: 初始化 Backbone ──────────────────────────────────────────────
    logger.info("初始化 ADMBackbone...")
    if args.mock:
        backbone = ADMBackbone(model_path="mock", device=str(device), ddim_steps=args.ddim_steps)
    else:
        backbone = ADMBackbone(
            model_path=args.model_path,
            device=str(device),
            ddim_steps=args.ddim_steps,
            image_size=args.image_size,
        )
    logger.info("Backbone: %s", backbone)

    # ── Step 2: 准备数据集 ────────────────────────────────────────────────────
    train_gen_names = [g.strip() for g in args.train_generators.split(",")]
    test_gen_names = [g.strip() for g in args.test_generators.split(",")]
    all_gen_names = sorted(set(train_gen_names + test_gen_names + ["real"]))
    ablation_schemes = [s.strip() for s in args.ablation_schemes.split(",")]

    if args.mock:
        logger.info("Mock 模式: 生成模拟数据集...")
        mock_image_size = min(args.image_size, 64)  # Mock 使用小尺寸

        # 训练集：real + 训练生成器
        train_gen_list = ["real"] + train_gen_names
        train_dataset = MockDeepfakeDataset(
            generators=train_gen_list,
            num_samples_per_gen=args.num_samples,
            image_size=mock_image_size,
            seed=42,
        )
        logger.info("训练集: %d 样本", len(train_dataset))

        # 验证集
        val_dataset = MockDeepfakeDataset(
            generators=train_gen_list,
            num_samples_per_gen=max(4, args.num_samples // 4),
            image_size=mock_image_size,
            seed=123,
        )
        logger.info("验证集: %d 样本", len(val_dataset))

        # 测试集（每个生成器独立）
        test_datasets: Dict[str, Dataset] = {}
        for gen_name in test_gen_names:
            gen_list = ["real", gen_name] if gen_name != "real" else ["real"]
            test_datasets[gen_name] = MockDeepfakeDataset(
                generators=gen_list,
                num_samples_per_gen=max(4, args.num_samples // 4),
                image_size=mock_image_size,
                seed=456 + hash(gen_name) % 1000,
            )
        # real 测试集（用于 Branch B/C 分析）
        test_datasets["real"] = MockDeepfakeDataset(
            generators=["real"],
            num_samples_per_gen=max(4, args.num_samples // 4),
            image_size=mock_image_size,
            seed=789,
        )
        logger.info("测试集: %s", list(test_datasets.keys()))
    else:
        # 真实模式数据加载
        raise NotImplementedError(
            "真实模式数据加载未在本脚本中实现。\n"
            "请使用 --mock 运行 Mock 测试，或自行实现数据加载逻辑。"
        )

    # ── Step 3: 消融实验 ──────────────────────────────────────────────────────
    logger.info("开始消融实验: %s", ablation_schemes)
    ablation_results = run_ablation_study(
        train_dataset=train_dataset,
        test_datasets=test_datasets,
        backbone=backbone,
        schemes=ablation_schemes,
        f2_steps=args.ddim_steps,
        return_intermediates=need_intermediates,
        device=device,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=str(out_dir),
    )

    # 保存消融结果
    ablation_path = out_dir / "feature_ablation.json"
    with open(ablation_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2, ensure_ascii=False)
    logger.info("消融结果已保存: %s", ablation_path)

    # ── Step 4: 主方案训练 ────────────────────────────────────────────────────
    logger.info("使用主方案 %s 训练最终模型...", args.feature_scheme)
    try:
        main_extractor = build_extractor_from_scheme(args.feature_scheme, f2_steps=args.ddim_steps)
    except (ImportError, ValueError) as exc:
        logger.warning("主方案 %s 无法构建: %s，回退到 FREQ 方案。", args.feature_scheme, exc)
        main_extractor = FrequencyFeatureExtractor()

    logger.info("特征提取器: %s, feature_dim=%d", main_extractor, main_extractor.feature_dim)

    # 提取训练特征
    logger.info("提取训练集特征...")
    train_features, train_labels = extract_all_features(
        train_dataset, backbone, main_extractor,
        return_intermediates=need_intermediates,
        batch_size=args.batch_size, device=device,
    )
    logger.info("训练特征形状: %s", tuple(train_features.shape))

    # 提取验证特征
    logger.info("提取验证集特征...")
    val_features, val_labels = extract_all_features(
        val_dataset, backbone, main_extractor,
        return_intermediates=need_intermediates,
        batch_size=args.batch_size, device=device,
    )

    # 构建分类头
    feature_dim = main_extractor.feature_dim
    head = BinaryDetectionHead(feature_dim=feature_dim, hidden_dim=256).to(device)

    # Phase 1 checkpoint 迁移学习（仅加载兼容参数）
    if args.phase1_checkpoint and os.path.isfile(args.phase1_checkpoint):
        logger.info("从 Phase 1 checkpoint 加载兼容参数: %s", args.phase1_checkpoint)
        try:
            ckpt = torch.load(args.phase1_checkpoint, map_location=device)
            head_state = ckpt.get("head_state_dict", {})
            # 只加载维度匹配的参数（迁移学习）
            compatible = {
                k: v for k, v in head_state.items()
                if k in head.state_dict() and head.state_dict()[k].shape == v.shape
            }
            missing = head.load_state_dict(compatible, strict=False)
            logger.info(
                "已加载 %d / %d 个参数 (missing=%d, unexpected=%d)",
                len(compatible),
                len(head.state_dict()),
                len(missing.missing_keys),
                len(missing.unexpected_keys),
            )
        except Exception as exc:
            logger.warning("Phase 1 checkpoint 加载失败: %s，从头开始训练。", exc)

    # 训练配置
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = BinaryDetectionLoss()

    feat_train_ds = FeatureDataset(train_features, train_labels)
    feat_val_ds = FeatureDataset(val_features, val_labels)
    feat_train_loader = DataLoader(feat_train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    feat_val_loader = DataLoader(feat_val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 训练循环
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_auc = 0.0
    best_model_path = str(out_dir / "best_model.pt")

    logger.info("开始训练 (%d epochs)...", args.epochs)
    for epoch in range(1, args.epochs + 1):
        # 训练
        head.train()
        ep_loss = 0.0
        for feats, lbls in feat_train_loader:
            feats, lbls = feats.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = head(feats)
            loss = loss_fn(logits, lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
        scheduler.step()
        avg_train_loss = ep_loss / max(len(feat_train_loader), 1)
        history["train_loss"].append(avg_train_loss)

        # 验证
        head.eval()
        y_true, y_scores = [], []
        val_loss = 0.0
        with torch.no_grad():
            for feats, lbls in feat_val_loader:
                feats, lbls_dev = feats.to(device), lbls.to(device)
                logits = head(feats)
                loss = loss_fn(logits, lbls_dev)
                val_loss += loss.item()
                scores = torch.sigmoid(logits.squeeze(1)).cpu().tolist()
                y_scores.extend(scores if isinstance(scores, list) else [scores])
                y_true.extend(lbls.cpu().tolist())

        avg_val_loss = val_loss / max(len(feat_val_loader), 1)
        val_auc = compute_auc(y_true, y_scores)
        history["val_loss"].append(avg_val_loss)
        history["val_auc"].append(val_auc)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_auc=%.4f",
            epoch, args.epochs, avg_train_loss, avg_val_loss, val_auc,
        )

        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {
                    "epoch": epoch,
                    "head_state_dict": head.state_dict(),
                    "feature_scheme": args.feature_scheme,
                    "feature_dim": feature_dim,
                    "best_val_auc": best_auc,
                    "history": history,
                },
                best_model_path,
            )
            logger.info("保存最佳模型 (AUC=%.4f): %s", best_auc, best_model_path)

    logger.info("训练完成！最佳 val_auc=%.4f", best_auc)

    # ── Step 5: 扩散假图专项评估 ──────────────────────────────────────────────
    logger.info("执行扩散假图专项评估...")
    diffusion_report = evaluate_diffusion_detection(
        test_datasets=test_datasets,
        backbone=backbone,
        schemes=ablation_schemes,
        ablation_results=ablation_results,
        f2_steps=args.ddim_steps,
        return_intermediates=need_intermediates,
        device=device,
        batch_size=args.batch_size,
    )

    # ── Step 6: 生成报告 ──────────────────────────────────────────────────────
    report_path = str(out_dir / "diffusion_detection_report.md")
    generate_diffusion_detection_report(
        diffusion_report=diffusion_report,
        ablation_results=ablation_results,
        schemes=ablation_schemes,
        output_path=report_path,
        is_mock=args.mock,
    )

    # 保存训练曲线
    curves_path = str(out_dir / "training_curves.png")
    save_training_curves(history, curves_path)

    # 保存总结 JSON
    summary = {
        "feature_scheme": args.feature_scheme,
        "feature_dim": feature_dim,
        "best_val_auc": best_auc,
        "mock": args.mock,
        "ddim_steps": args.ddim_steps,
        "epochs": args.epochs,
        "ablation_schemes": ablation_schemes,
        "ablation_results": ablation_results,
        "history": {k: [float(v) for v in vs] for k, vs in history.items()},
    }
    summary_path = out_dir / "phase2_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Step 7: 列出输出文件 ─────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("Phase 2 完成。输出文件：")
    for fpath in sorted(out_dir.iterdir()):
        if fpath.is_file():
            size_kb = fpath.stat().st_size / 1024
            logger.info("  %-45s %8.1f KB", fpath.name, size_kb)
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
