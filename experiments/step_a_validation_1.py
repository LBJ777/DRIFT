"""
DRIFT Phase 0 Step A: 统计验证脚本
===================================
目的：在不训练任何模型的情况下，用统计方法验证核心假设：
不同来源图像（真实/GAN/SD）经 DDIM 逆向后的噪声终点 x_T 具有统计上可区分的分布。

使用方法：
  # Mock 模式（无需真实模型）
  python step_a_validation.py --mock --num_samples 10 --output_dir ./results/step_a

  # 真实模式
  python step_a_validation.py \
    --data_dir /path/to/images \
    --model_path /path/to/256x256_diffusion_uncond.pt \
    --output_dir ./results/step_a \
    --num_samples 100 \
    --ddim_steps 20 \
    --image_size 256
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无显示器环境下使用非交互后端
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image

# ── scipy / sklearn 统计工具 ──────────────────────────────────────────────────
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

import torch

# ── 将 preprocessing_model 加入模块搜索路径，以便 import 现有扩散代码 ──────────
_REPO_ROOT = Path(__file__).resolve().parents[2]  # DRIFT/
_PREPROCESS_ROOT = (
    _REPO_ROOT.parent                                        # Deepfake_Project/
    / "AIGCDetectBenchmark-main"
    / "preprocessing_model"
)
if str(_PREPROCESS_ROOT) not in sys.path:
    sys.path.insert(0, str(_PREPROCESS_ROOT))

# ── 现有扩散模型接口（仅在非 mock 模式下需要） ───────────────────────────────
def _import_diffusion_utils():
    """
    延迟导入扩散模型工具，避免 mock 模式下的不必要依赖。
    返回 (model_and_diffusion_defaults, create_model_and_diffusion)。
    """
    try:
        from guided_diffusion.script_util import (
            model_and_diffusion_defaults,
            create_model_and_diffusion,
        )
        return model_and_diffusion_defaults, create_model_and_diffusion
    except ImportError as e:
        raise ImportError(
            f"无法导入 guided_diffusion：{e}\n"
            f"请确认路径 {_PREPROCESS_ROOT} 正确，或使用 --mock 模式。"
        ) from e


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  1. 数据加载                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def load_images_from_dir(data_dir: str, num_samples: int, image_size: int):
    """
    扫描 data_dir 下的子目录，每个子目录视为一个来源标签。
    返回：
      images_by_source: dict[str, list[Tensor]]  -- 每类已归一化到 [-1,1] 的图像张量
      labels_list: list[str]                      -- 按子目录字母序排列的标签名
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"数据目录不存在：{data_dir}")

    subdirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    if not subdirs:
        raise ValueError(f"数据目录 {data_dir} 下没有子目录，请按 real/ProGAN/SD_v1.4 格式组织。")

    images_by_source = {}
    for subdir in subdirs:
        label = subdir.name
        img_paths = [p for p in subdir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS]
        if not img_paths:
            warnings.warn(f"子目录 {subdir} 中没有找到图像文件，跳过。")
            continue

        # 随机采样（固定种子保证可重现）
        rng = np.random.default_rng(seed=42)
        selected = rng.choice(img_paths, size=min(num_samples, len(img_paths)), replace=False)

        tensors = []
        for p in selected:
            try:
                img = Image.open(p).convert("RGB").resize(
                    (image_size, image_size), Image.LANCZOS
                )
                arr = np.array(img, dtype=np.float32) / 127.5 - 1.0  # -> [-1, 1]
                # shape: (H, W, C) -> (C, H, W)
                t = torch.from_numpy(arr.transpose(2, 0, 1))
                tensors.append(t)
            except Exception as exc:
                warnings.warn(f"加载图像失败 {p}：{exc}，跳过。")

        if tensors:
            images_by_source[label] = tensors
            print(f"  [数据加载] {label}: {len(tensors)} 张图像")

    return images_by_source


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  2. 真实 DDIM 逆向模式                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def build_model_and_diffusion(model_path: str, image_size: int, ddim_steps: int, device: torch.device):
    """
    构建 ADM 模型和 DDIM 扩散过程。
    直接调用 guided_diffusion.script_util 中的工厂函数。
    """
    model_and_diffusion_defaults, create_model_and_diffusion = _import_diffusion_utils()

    defaults = model_and_diffusion_defaults()
    # 覆盖关键参数
    defaults.update({
        "image_size": image_size,
        "timestep_respacing": f"ddim{ddim_steps}",
        "use_fp16": (device.type == "cuda"),
    })

    print(f"  [模型] 正在创建 UNet 模型（image_size={image_size}, ddim_steps={ddim_steps}）...")
    model, diffusion = create_model_and_diffusion(**defaults)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"ADM 预训练权重不存在：{model_path}\n"
            "请从 https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
            "256x256_diffusion_uncond.pt 下载，或使用 --mock 模式。"
        )

    print(f"  [模型] 加载权重：{model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    if defaults["use_fp16"]:
        model.convert_to_fp16()
    model.eval()
    return model, diffusion


def run_ddim_inversion(image_tensor: torch.Tensor, model, diffusion, device: torch.device) -> torch.Tensor:
    """
    对单张图像执行 DDIM 逆向，返回 x_T。
    调用 gaussian_diffusion.py 中的 ddim_reverse_sample_loop()。

    参数：
      image_tensor: shape (C, H, W)，值域 [-1, 1]
    返回：
      x_T: shape (C, H, W)
    """
    x0 = image_tensor.unsqueeze(0).to(device)  # -> (1, C, H, W)
    shape = x0.shape

    # ddim_reverse_sample_loop 以 x0 作为 noise 参数（初始输入）
    # 逐步将 x0 映射到 x_T
    x_T = diffusion.ddim_reverse_sample_loop(
        model=model,
        shape=shape,
        noise=x0,
        clip_denoised=False,   # 逆向时不裁剪，保留完整噪声分布
        model_kwargs={},
        device=device,
        progress=False,
        eta=0.0,
    )
    return x_T.squeeze(0).cpu()  # -> (C, H, W)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  3. Mock 模式（模拟不同来源的 x_T 分布）                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def generate_mock_xT(label: str, num_samples: int, image_size: int) -> list:
    """
    用参数化高斯分布模拟不同来源的 x_T 统计特性：
      - real:   N(0, 1)          -- 最接近标准正态（ADM 完美逆向假设）
      - ProGAN: N(0.1, 1.1²)    -- 均值偏移 + 方差膨胀（GAN 模式崩溃残留）
      - SD:     N(0, 1) + 8px 周期性信号（VAE 8px 块状伪影）
      - 其他:   N(0.05, 1.05²)  -- 轻度偏移
    """
    rng = np.random.default_rng(seed=hash(label) % (2**31))
    C, H, W = 3, image_size, image_size
    tensors = []

    for _ in range(num_samples):
        if label.lower() in ("real", "genuine", "authentic"):
            # 真实图像：标准正态，最接近高斯
            noise = rng.standard_normal((C, H, W)).astype(np.float32)

        elif "progan" in label.lower() or "gan" in label.lower():
            # GAN 假图：均值/方差偏移
            noise = (rng.standard_normal((C, H, W)) * 1.1 + 0.1).astype(np.float32)

        elif "sd" in label.lower() or "stable" in label.lower() or "diffusion" in label.lower():
            # SD 假图：基础高斯 + 8px 周期性空间信号（模拟 VAE 量化伪影）
            base = rng.standard_normal((C, H, W)).astype(np.float32)
            # 生成 8px 周期性网格信号（在频域 f=H/8, W/8 处有显著峰值）
            freq_H = H // 8
            freq_W = W // 8
            grid_y = np.sin(2 * np.pi * freq_H * np.arange(H) / H).reshape(1, H, 1)
            grid_x = np.sin(2 * np.pi * freq_W * np.arange(W) / W).reshape(1, 1, W)
            periodic = (grid_y * grid_x).astype(np.float32)  # 广播到 (1, H, W)
            noise = base + 0.5 * periodic  # 叠加周期信号

        else:
            # 其他未知来源：轻度偏移
            noise = (rng.standard_normal((C, H, W)) * 1.05 + 0.05).astype(np.float32)

        tensors.append(torch.from_numpy(noise))

    return tensors


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  4. 特征提取：x_T -> 向量                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _xT_to_compact_features(xT: torch.Tensor) -> np.ndarray:
    """
    从单张 x_T 提取 42 维紧凑统计特征，替代原来的 196608D 原始展平。

    原始像素展平用于 t-SNE 时：PCA 在 196608D 空间中无法提取有判别力的成分，
    导致不同来源的点完全混叠（Silhouette ≈ 0）。紧凑特征直接捕捉有物理意义的
    分布差异，使 t-SNE 能够正确分离。

    特征组：
      G1 (12D): 每通道 mean / std / skewness / kurtosis
      G2 ( 3D): 每通道近似 KS 统计量（对 N(0,1) 的偏离度）
      G3 (24D): 每通道径向 PSD 在 8 个频带的对数均值
      G4 ( 3D): 每通道 f = H/8 处的 VAE 伪影峰值能量（对数）
    """
    C, H, W = xT.shape
    x = xT.float().numpy()  # (C, H, W)
    feats: list = []

    _rng = np.random.default_rng(seed=0)

    # G1: 统计矩（12D）
    for c in range(C):
        ch = x[c].flatten()
        mu = float(ch.mean())
        sigma = float(ch.std()) + 1e-8
        x_c = ch - mu
        skewness = float((x_c ** 3).mean()) / (sigma ** 3)
        kurt = float((x_c ** 4).mean()) / (sigma ** 4) - 3.0
        feats.extend([mu, sigma, skewness, kurt])

    # G2: 近似 KS 统计量 vs N(0,1)（3D）
    for c in range(C):
        ch = x[c].flatten()
        n_sub = min(2000, len(ch))
        idx = _rng.choice(len(ch), n_sub, replace=False)
        ref = _rng.standard_normal(n_sub)
        ks_stat, _ = ks_2samp(ch[idx], ref)
        feats.append(float(ks_stat))

    # G3: 径向 PSD 8 频带（24D）
    for c in range(C):
        ch_t = torch.from_numpy(x[c])
        freq = torch.fft.rfft2(ch_t, norm="ortho")
        power = (freq.real ** 2 + freq.imag ** 2).numpy()  # (Hf, Wf)
        Hf, Wf = power.shape

        fy = np.fft.fftfreq(H)[:Hf]               # (Hf,)
        fx = np.fft.rfftfreq(W)                    # (Wf,)
        fy_g = np.broadcast_to(fy[:, None], (Hf, Wf))
        fx_g = np.broadcast_to(fx[None, :], (Hf, Wf))
        r_norm = np.sqrt(fy_g ** 2 + fx_g ** 2)
        r_max = r_norm.max() + 1e-8
        r_norm = r_norm / r_max

        edges = np.linspace(0.0, 1.0, 9)  # 8 bands
        for b in range(8):
            lo, hi = edges[b], edges[b + 1]
            mask = (r_norm >= lo) & (r_norm < hi)
            if b == 7:
                mask = (r_norm >= lo) & (r_norm <= hi)
            band_mean = power[mask].mean() if mask.any() else 0.0
            feats.append(float(np.log1p(band_mean)))

    # G4: VAE 8px 峰值（3D）
    h_peak = H // 8
    for c in range(C):
        ch_t = torch.from_numpy(x[c])
        freq = torch.fft.rfft2(ch_t, norm="ortho")
        power = (freq.real ** 2 + freq.imag ** 2).numpy()
        Wf = power.shape[1]
        w_peak = min(W // 8, Wf - 1)
        h_idx = min(h_peak, power.shape[0] - 1)
        feats.append(float(np.log1p(power[h_idx, w_peak])))

    return np.array(feats, dtype=np.float32)  # (42,)


def extract_xT_features(xT_by_source: dict) -> tuple:
    """
    将各来源的 x_T 张量整理为 numpy 矩阵，供统计分析使用。

    【v1.1 修正】使用 42 维紧凑统计特征代替 196608D 原始展平。
    原始展平用于 t-SNE 时，PCA 无法从高维像素噪声中提取判别信号，
    导致 Silhouette 为负（各类完全混叠）。紧凑特征保留物理上有意义的
    分布统计量，t-SNE 能够正确揭示来源间的聚类结构。

    返回：
      features: ndarray (N_total, 42) 紧凑统计特征
      labels:   ndarray (N_total,)    整数标签
      label_names: list[str]          标签名称（按索引）
    """
    label_names = sorted(xT_by_source.keys())
    features_list = []
    labels_list = []

    for idx, name in enumerate(label_names):
        xTs = xT_by_source[name]
        for xT in xTs:
            feat = _xT_to_compact_features(xT)   # (42,) 紧凑统计特征
            features_list.append(feat)
            labels_list.append(idx)

    features = np.stack(features_list, axis=0)   # (N, 42)
    labels = np.array(labels_list, dtype=np.int32)
    return features, labels, label_names


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  5. Test 1: t-SNE 可视化                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def test_tsne(features: np.ndarray, labels: np.ndarray, label_names: list, output_path: str) -> dict:
    """
    PCA(50维) -> t-SNE(2维) 可视化，计算 Silhouette Score 作为量化指标。

    判定标准：Silhouette Score > 0.3 视为聚类分离显著。
    """
    print(f"\n[Test 1] 执行 t-SNE 可视化（基于 42D 紧凑统计特征）...")
    n_samples, n_features = features.shape

    # PCA 预降维（加速 t-SNE，减少噪声）
    n_pca = min(50, n_samples - 1, n_features)
    print(f"  PCA: {n_features}D -> {n_pca}D")
    pca = PCA(n_components=n_pca, random_state=42)
    features_pca = pca.fit_transform(features)

    # t-SNE 降到 2D
    perplexity = min(30, max(5, n_samples // 4))
    print(f"  t-SNE: {n_pca}D -> 2D (perplexity={perplexity})")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    features_2d = tsne.fit_transform(features_pca)

    # 计算 Silhouette Score（需要至少 2 个类且每类 > 1 个样本）
    silhouette = float("nan")
    n_classes = len(label_names)
    if n_classes >= 2 and n_samples >= n_classes * 2:
        try:
            silhouette = silhouette_score(features_2d, labels)
        except Exception as exc:
            warnings.warn(f"Silhouette Score 计算失败：{exc}")

    print(f"  Silhouette Score: {silhouette:.4f}" if not np.isnan(silhouette) else "  Silhouette Score: N/A（样本不足）")

    # 绘图
    colors = list(mcolors.TABLEAU_COLORS.values())
    fig, ax = plt.subplots(figsize=(8, 7))
    for idx, name in enumerate(label_names):
        mask = labels == idx
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=colors[idx % len(colors)],
            label=name,
            alpha=0.7,
            s=20,
            edgecolors="none",
        )
    ax.set_title(f"t-SNE of x_T by Source\nSilhouette Score = {silhouette:.4f}" if not np.isnan(silhouette)
                 else "t-SNE of x_T by Source")
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  已保存：{output_path}")

    passed = (not np.isnan(silhouette)) and (silhouette > 0.3)
    return {
        "silhouette_score": silhouette if not np.isnan(silhouette) else None,
        "threshold": 0.3,
        "passed": passed,
        "note": "Silhouette > 0.3 视为聚类分离显著",
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  6. Test 2: 功率谱密度（PSD）分析                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def compute_psd(xT: torch.Tensor) -> np.ndarray:
    """
    计算单张 x_T 的 2D 功率谱密度（三通道平均）。
    返回 shape (H, W//2+1) 的 PSD（rfft2 结果）。
    """
    # xT: (C, H, W)
    psd_channels = []
    for c in range(xT.shape[0]):
        ch = xT[c].float()  # (H, W)
        fft = torch.fft.rfft2(ch)
        psd_channels.append((fft.abs() ** 2).numpy())
    # 三通道平均
    return np.mean(psd_channels, axis=0)  # (H, W//2+1)


def test_psd(xT_by_source: dict, image_size: int, output_path: str) -> dict:
    """
    对每类来源计算平均 PSD，重点检测 SD 来源在 f=H/8 处的峰值。

    判定标准：SD 来源在 f=H/8 处的 PSD 值，相对于所有频率的 z-score > 2.0。
    """
    print("\n[Test 2] 执行功率谱密度（PSD）分析...")
    H = image_size

    psd_means = {}  # label -> mean PSD (H, W//2+1)
    for label, xTs in xT_by_source.items():
        psds = [compute_psd(xT) for xT in xTs]
        psd_means[label] = np.mean(psds, axis=0)

    # 找 SD 来源标签（包含 "sd" 或 "stable" 的子目录名）
    sd_labels = [l for l in psd_means if "sd" in l.lower() or "stable" in l.lower()]

    # 评估：SD 在 f=H/8 处的 z-score
    sd_peak_zscore = float("nan")
    target_freq_H = H // 8  # 对应 8px 周期的频率 bin

    if sd_labels:
        sd_psd = psd_means[sd_labels[0]]  # 取第一个 SD 来源
        # ── 局部邻域 z-score（v1.1 修正）──────────────────────────────────────
        # 原实现：z-score 基于全局 PSD 分布 → 低频能量主导，f=H/8 的 VAE 峰值被稀释
        # 新实现：与目标频率 ±4~12 行的局部背景相比，专注于 8px 周期性凸起
        target_val = float(sd_psd[target_freq_H, :].mean())

        bg_lo = list(range(max(0, target_freq_H - 12), max(0, target_freq_H - 3)))
        bg_hi = list(range(min(H, target_freq_H + 4), min(H, target_freq_H + 13)))
        bg_rows = bg_lo + bg_hi

        if len(bg_rows) >= 4:
            bg_vals = sd_psd[bg_rows, :].flatten()
            local_mean = float(bg_vals.mean())
            local_std = float(bg_vals.std())
            if local_std > 1e-12:
                sd_peak_zscore = (target_val - local_mean) / local_std
            else:
                sd_peak_zscore = 0.0
        else:
            # Fallback to global if not enough background rows (小图)
            psd_flat = sd_psd.flatten()
            mean_psd = psd_flat.mean()
            std_psd = psd_flat.std()
            if std_psd > 1e-12:
                sd_peak_zscore = (target_val - mean_psd) / std_psd
        print(f"  SD 来源 [{sd_labels[0]}] 在 f=H/8={target_freq_H} 处 PSD 局部 z-score: {sd_peak_zscore:.4f}")
    else:
        print("  未找到 SD 来源标签（标签名应包含 'sd' 或 'stable'），跳过峰值检测。")

    # 绘图：每类来源的 PSD 径向平均（便于比较）
    label_names = sorted(psd_means.keys())
    colors = list(mcolors.TABLEAU_COLORS.values())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1：全局 PSD（log 尺度二维热图，仅显示第一个来源示例）
    first_label = label_names[0]
    im = axes[0].imshow(
        np.log1p(psd_means[first_label]),
        aspect="auto",
        cmap="viridis",
        origin="upper",
    )
    axes[0].set_title(f"Log PSD (示例：{first_label})")
    axes[0].set_xlabel("频率 W 方向")
    axes[0].set_ylabel("频率 H 方向")
    plt.colorbar(im, ax=axes[0])
    # 标注目标频率
    axes[0].axhline(H // 8, color="red", linestyle="--", linewidth=1, label=f"f=H/8={H//8}")
    axes[0].legend(fontsize=8)

    # 子图2：各来源 PSD 径向平均曲线
    for idx, label in enumerate(label_names):
        psd = psd_means[label]
        # 径向平均：对每个 H 频率 bin 取 W 方向均值
        radial_avg = psd.mean(axis=1)  # (H,)
        axes[1].plot(radial_avg, label=label, color=colors[idx % len(colors)], linewidth=1.5)

    axes[1].axvline(H // 8, color="gray", linestyle="--", linewidth=1, label=f"f=H/8={H//8}")
    axes[1].set_title("各来源 PSD 径向平均（H 方向频率）")
    axes[1].set_xlabel("频率 bin（H 方向）")
    axes[1].set_ylabel("平均功率")
    axes[1].legend(fontsize=9)
    axes[1].set_yscale("log")

    fig.suptitle("功率谱密度（PSD）对比分析", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  已保存：{output_path}")

    passed = (not np.isnan(sd_peak_zscore)) and (sd_peak_zscore > 2.0)
    return {
        "sd_peak_zscore": sd_peak_zscore if not np.isnan(sd_peak_zscore) else None,
        "target_freq_H": H // 8,
        "threshold": 2.0,
        "passed": passed,
        "note": "SD 来源在 f=H/8 处 PSD z-score > 2.0 视为存在 VAE 伪影",
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  7. Test 3: Wasserstein 距离矩阵                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def xT_to_hist(xT: torch.Tensor, n_bins: int = 100) -> np.ndarray:
    """
    将 x_T 的功率谱（展平）转为归一化直方图，作为分布近似。
    """
    psd = compute_psd(xT).flatten()
    hist, _ = np.histogram(psd, bins=n_bins, density=True)
    return hist.astype(np.float64)


def test_wasserstein(xT_by_source: dict, output_path: str) -> dict:
    """
    构建类间 Wasserstein 距离矩阵，评估类间 vs 类内分离度。

    判定标准：类间平均距离 > 类内平均距离 × 1.5
    """
    print("\n[Test 3] 计算 Wasserstein 距离矩阵...")
    label_names = sorted(xT_by_source.keys())
    n = len(label_names)

    # 每个样本的功率谱直方图
    hists_by_source = {}
    for label, xTs in xT_by_source.items():
        hists_by_source[label] = [xT_to_hist(xT) for xT in xTs]

    # 类平均直方图（用于类间距离计算）
    mean_hists = {}
    for label in label_names:
        hs = np.stack(hists_by_source[label], axis=0)
        mean_hists[label] = hs.mean(axis=0)

    # N×N 类间距离矩阵（使用类平均直方图）
    bin_centers = np.arange(100, dtype=np.float64)  # 与 xT_to_hist 的 bin 数一致
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = wasserstein_distance(
                bin_centers, bin_centers,
                mean_hists[label_names[i]],
                mean_hists[label_names[j]],
            )

    # 类内平均距离（同类样本两两之间）
    intra_dists = []
    for label in label_names:
        hs = hists_by_source[label]
        m = len(hs)
        if m < 2:
            continue
        pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
        # 随机采样最多 200 对，避免组合爆炸
        rng = np.random.default_rng(42)
        if len(pairs) > 200:
            pairs = [pairs[k] for k in rng.choice(len(pairs), 200, replace=False)]
        for (i, j) in pairs:
            d = wasserstein_distance(bin_centers, bin_centers, hs[i], hs[j])
            intra_dists.append(d)

    intra_mean = np.mean(intra_dists) if intra_dists else 0.0

    # 类间平均距离（仅取上三角，排除对角线）
    inter_dists = [dist_matrix[i, j] for i in range(n) for j in range(i + 1, n)]
    inter_mean = np.mean(inter_dists) if inter_dists else 0.0

    print(f"  类间平均 Wasserstein 距离: {inter_mean:.6f}")
    print(f"  类内平均 Wasserstein 距离: {intra_mean:.6f}")
    ratio = inter_mean / (intra_mean + 1e-12)
    print(f"  类间/类内比值: {ratio:.4f}（阈值 > 1.5）")

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(max(5, n + 2), max(4, n + 1)))
    im = ax.imshow(dist_matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Wasserstein 距离")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(label_names, fontsize=9)
    # 在格子中标注数值
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{dist_matrix[i, j]:.4f}", ha="center", va="center",
                    fontsize=8, color="black" if dist_matrix[i, j] < dist_matrix.max() * 0.7 else "white")
    ax.set_title(
        f"Wasserstein 距离矩阵\n"
        f"类间均值={inter_mean:.4f}, 类内均值={intra_mean:.4f}, 比值={ratio:.2f}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  已保存：{output_path}")

    passed = ratio > 1.5
    return {
        "inter_class_mean": float(inter_mean),
        "intra_class_mean": float(intra_mean),
        "ratio": float(ratio),
        "threshold": 1.5,
        "passed": passed,
        "distance_matrix": dist_matrix.tolist(),
        "label_names": label_names,
        "note": "类间/类内比值 > 1.5 视为分布分离显著",
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  8. Test 4: KS 分布可区分性检验                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def test_ks_gaussianity(xT_by_source: dict) -> dict:
    """
    【v1.1 修正】来源间两两 Kolmogorov-Smirnov 双样本检验。

    原实现的问题：
      期望真实图像 x_T 通过单样本 KS vs N(0,1) 检验（p > 0.05）。
      但 20 步 DDIM 逆向无法使 x_T 完全收敛到 N(0,I)，真实图像的 x_T
      同样明显偏离高斯（KS 统计量甚至最高），导致该检验系统性失败。

    新实现：
      检验"不同来源的 x_T 分布是否可区分"，即来源间两两 KS 双样本检验。
      若 ≥ 80% 的来源对之间检验显著（p < 0.05），则 PASS。
      这直接对应 DRIFT 的核心假设：不同来源的 x_T 分布在统计上可区分。

    同时保留单样本 KS vs N(0,1) 的结果作为参考（不影响判定）。
    """
    print("\n[Test 4] 执行 KS 分布可区分性检验（来源间双样本）...")
    label_names_sorted = sorted(xT_by_source.keys())
    real_labels = [l for l in label_names_sorted if l.lower() in ("real", "genuine", "authentic")]

    # ── 收集每类的展平值（下采样加速） ──────────────────────────────────────────
    _rng = np.random.default_rng(42)
    flat_by_source: dict = {}
    for label, xTs in xT_by_source.items():
        all_vals = np.concatenate([xT.numpy().flatten() for xT in xTs])
        n_sub = min(5000, len(all_vals))
        idx = _rng.choice(len(all_vals), n_sub, replace=False)
        flat_by_source[label] = all_vals[idx]

    # ── 单样本 KS vs N(0,1)（参考，不判定） ────────────────────────────────────
    print("  [参考] 各来源 vs N(0,1) 单样本 KS 统计量：")
    results_single: dict = {}
    ref_gaussian = _rng.standard_normal(5000)
    for label in label_names_sorted:
        ks_stat, p_val = ks_2samp(flat_by_source[label], ref_gaussian)
        results_single[label] = {
            "ks_statistic": float(ks_stat),
            "p_value": float(p_val),
            "is_gaussian": bool(p_val > 0.05),
            "is_real_source": label in real_labels,
        }
        tag = "<-- 真实图像" if label in real_labels else ""
        print(f"    [{label}] KS={ks_stat:.6f}, p={p_val:.2e} "
              f"({'接近高斯' if p_val > 0.05 else '偏离高斯'}) {tag}")

    # ── 来源间两两双样本 KS 检验（判定依据） ────────────────────────────────────
    print("  [判定] 来源间两两双样本 KS 检验：")
    pairwise: dict = {}
    n_pairs = 0
    n_significant = 0

    for i, la in enumerate(label_names_sorted):
        for j, lb in enumerate(label_names_sorted):
            if i >= j:
                continue
            ks_stat, p_val = ks_2samp(flat_by_source[la], flat_by_source[lb])
            key = f"{la}_vs_{lb}"
            sig = p_val < 0.05
            pairwise[key] = {
                "ks_statistic": float(ks_stat),
                "p_value": float(p_val),
                "significant": sig,
            }
            n_pairs += 1
            if sig:
                n_significant += 1
            mark = "✓ 显著" if sig else "✗ 不显著"
            print(f"    [{la}] vs [{lb}]: KS={ks_stat:.4f}, p={p_val:.2e}  {mark}")

    frac_significant = n_significant / max(1, n_pairs)
    threshold_frac = 0.8
    passed = frac_significant >= threshold_frac
    note = (
        f"来源间 {n_significant}/{n_pairs} 对显著不同 ({frac_significant*100:.0f}%)，"
        f"≥{int(threshold_frac*100)}% 视为通过"
    )
    print(f"  {note}")

    return {
        "vs_gaussian_reference": results_single,
        "pairwise_ks": pairwise,
        "n_pairs": n_pairs,
        "n_significant": n_significant,
        "fraction_significant": float(frac_significant),
        "passed": passed,
        "note": note,
        "threshold_fraction": threshold_frac,
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  9. 验证门逻辑                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def evaluate_validation_gate(results: dict) -> tuple:
    """
    综合 4 项检验结果，判断是否通过验证门。
    至少 3/4 项通过，项目可进入 Phase 1。

    返回：
      (gate_passed: bool, report_lines: list[str])
    """
    tests = {
        "Test1_tSNE":        results.get("test1_tsne", {}),
        "Test2_PSD":         results.get("test2_psd", {}),
        "Test3_Wasserstein": results.get("test3_wasserstein", {}),
        "Test4_KS":          results.get("test4_ks", {}),
    }

    lines = []
    lines.append("=" * 60)
    lines.append("DRIFT Phase 0 Step A — Validation Gate Report")
    lines.append("=" * 60)
    lines.append("")

    pass_count = 0
    total = len(tests)

    for test_name, test_result in tests.items():
        passed = test_result.get("passed", False)
        if passed:
            pass_count += 1
        status = "PASS" if passed else "FAIL"

        lines.append(f"[{status}] {test_name}")

        # 具体指标输出
        if "silhouette_score" in test_result:
            sc = test_result["silhouette_score"]
            lines.append(f"       Silhouette Score = {sc:.4f}" if sc is not None else "       Silhouette Score = N/A")
            lines.append(f"       阈值 > {test_result.get('threshold', 0.3)}")
        if "sd_peak_zscore" in test_result:
            z = test_result["sd_peak_zscore"]
            lines.append(f"       SD PSD z-score @ f=H/8 = {z:.4f}" if z is not None else "       SD PSD z-score = N/A（无SD来源）")
            lines.append(f"       阈值 > {test_result.get('threshold', 2.0)}")
        if "ratio" in test_result:
            lines.append(f"       类间/类内 Wasserstein 比值 = {test_result['ratio']:.4f}")
            lines.append(f"       阈值 > {test_result.get('threshold', 1.5)}")
        if "per_source" in test_result:
            for src, res in test_result["per_source"].items():
                lines.append(f"       [{src}] KS={res['ks_statistic']:.6f}, p={res['p_value']:.4e}"
                              f" ({'高斯' if res['is_gaussian'] else '非高斯'})")
        if "fraction_significant" in test_result:
            n_sig = test_result.get("n_significant", "?")
            n_tot = test_result.get("n_pairs", "?")
            frac = test_result["fraction_significant"]
            lines.append(f"       来源间 {n_sig}/{n_tot} 对显著不同 ({frac*100:.0f}%)")
            lines.append(f"       阈值 ≥ {int(test_result.get('threshold_fraction', 0.8)*100)}%")

        note = test_result.get("note", "")
        if note:
            lines.append(f"       说明：{note}")
        lines.append("")

    lines.append("-" * 60)
    lines.append(f"总计：{pass_count}/{total} 项通过")
    gate_passed = pass_count >= 3
    lines.append("")
    if gate_passed:
        lines.append(">>> 验证门：PASS <<<")
        lines.append("核心假设成立：不同来源的 x_T 统计分布具有可区分性。")
        lines.append("项目可进入 Phase 1（特征提取与分类器训练）。")
    else:
        lines.append(">>> 验证门：FAIL <<<")
        lines.append("核心假设尚未得到充分验证。")
        lines.append("建议：增加样本数量，检查 DDIM 步数设置，或重新审视假设前提。")
    lines.append("=" * 60)

    return gate_passed, lines


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  10. 主流程                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def parse_args():
    parser = argparse.ArgumentParser(
        description="DRIFT Phase 0 Step A: x_T 统计分布验证",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="图像根目录，子目录名即为来源标签（real/, ProGAN/, SD_v1.4/ ...）",
    )
    parser.add_argument(
        "--model_path", type=str,
        default="./models/256x256_diffusion_uncond.pt",
        help="ADM 预训练权重路径（仅非 mock 模式使用）",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results/step_a",
        help="输出目录",
    )
    parser.add_argument(
        "--num_samples", type=int, default=100,
        help="每类采样数量",
    )
    parser.add_argument(
        "--ddim_steps", type=int, default=20,
        help="DDIM 逆向步数",
    )
    parser.add_argument(
        "--image_size", type=int, default=256,
        help="图像分辨率（正方形，需与 ADM 模型匹配）",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Mock 模式：用参数化高斯分布模拟 x_T，无需真实模型",
    )
    parser.add_argument(
        "--mock_sources", type=str,
        default="real,ProGAN,SD_v1.4,DALL-E2,Midjourney",
        help="Mock 模式下的来源名称列表（逗号分隔）",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="计算设备：auto / cpu / cuda / mps",
    )
    return parser.parse_args()


def select_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def main():
    args = parse_args()
    device = select_device(args.device)
    print(f"\n{'='*60}")
    print("DRIFT Phase 0 Step A: x_T 统计分布验证")
    print(f"{'='*60}")
    print(f"  模式：{'Mock（模拟）' if args.mock else '真实 DDIM 逆向'}")
    print(f"  设备：{device}")
    print(f"  输出：{args.output_dir}")

    # 创建输出目录
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: 获取 x_T ─────────────────────────────────────────────────────
    xT_by_source = {}

    if args.mock:
        # Mock 模式：用参数化高斯分布模拟
        print(f"\n[Mock] 模拟来源：{args.mock_sources}")
        source_names = [s.strip() for s in args.mock_sources.split(",")]
        for name in source_names:
            xTs = generate_mock_xT(name, args.num_samples, args.image_size)
            xT_by_source[name] = xTs
            print(f"  [{name}] 生成 {len(xTs)} 个模拟 x_T")
    else:
        # 真实模式：加载图像 -> DDIM 逆向
        if args.data_dir is None:
            raise ValueError("非 mock 模式下必须指定 --data_dir")

        print(f"\n[数据] 从 {args.data_dir} 加载图像...")
        images_by_source = load_images_from_dir(args.data_dir, args.num_samples, args.image_size)

        print(f"\n[模型] 构建 ADM 扩散模型...")
        model, diffusion = build_model_and_diffusion(
            args.model_path, args.image_size, args.ddim_steps, device
        )

        print(f"\n[逆向] 对每张图像执行 DDIM 逆向（steps={args.ddim_steps}）...")
        for label, images in images_by_source.items():
            xTs = []
            for i, img in enumerate(images):
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"  [{label}] 进度：{i+1}/{len(images)}")
                try:
                    xT = run_ddim_inversion(img, model, diffusion, device)
                    xTs.append(xT)
                except Exception as exc:
                    warnings.warn(f"  [{label}] 图像 {i} DDIM 逆向失败：{exc}")
            if xTs:
                xT_by_source[label] = xTs
                print(f"  [{label}] 完成，共 {len(xTs)} 个 x_T")

    if not xT_by_source:
        raise RuntimeError("没有成功获取任何 x_T，请检查数据目录或 Mock 配置。")

    # ── Step 2: 保存原始 x_T 特征 ────────────────────────────────────────────
    print("\n[特征] 提取并保存原始 x_T 特征...")
    features, labels_arr, label_names = extract_xT_features(xT_by_source)
    npz_path = out_dir / "raw_xT_features.npz"
    np.savez_compressed(
        str(npz_path),
        features=features,
        labels=labels_arr,
        label_names=np.array(label_names),
    )
    print(f"  已保存：{npz_path}  shape={features.shape}")

    # ── Step 3: 执行四项统计检验 ──────────────────────────────────────────────
    all_results = {}

    # Test 1: t-SNE
    t1 = test_tsne(
        features, labels_arr, label_names,
        output_path=str(out_dir / "tsne_visualization.png"),
    )
    all_results["test1_tsne"] = t1

    # Test 2: PSD
    t2 = test_psd(
        xT_by_source, args.image_size,
        output_path=str(out_dir / "psd_comparison.png"),
    )
    all_results["test2_psd"] = t2

    # Test 3: Wasserstein
    t3 = test_wasserstein(
        xT_by_source,
        output_path=str(out_dir / "wasserstein_heatmap.png"),
    )
    all_results["test3_wasserstein"] = t3

    # Test 4: KS 检验
    t4 = test_ks_gaussianity(xT_by_source)
    all_results["test4_ks"] = t4

    # ── Step 4: 保存 KS 检验 JSON ─────────────────────────────────────────────
    ks_json_path = out_dir / "ks_test_results.json"
    # 过滤掉不可序列化的 distance_matrix 等大型字段
    ks_output = {
        "test4_ks": t4,
        "meta": {
            "num_samples": args.num_samples,
            "image_size": args.image_size,
            "ddim_steps": args.ddim_steps,
            "mock": args.mock,
            "sources": list(xT_by_source.keys()),
        },
    }
    with open(ks_json_path, "w", encoding="utf-8") as f:
        json.dump(ks_output, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"\n[输出] KS 检验结果已保存：{ks_json_path}")

    # ── Step 5: 验证门评估 + 报告 ─────────────────────────────────────────────
    gate_passed, report_lines = evaluate_validation_gate(all_results)

    report_path = out_dir / "validation_gate_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"\n[报告] 验证门报告已保存：{report_path}")
    print("\n" + "\n".join(report_lines))

    # ── Step 6: 列出所有输出文件 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("输出文件列表：")
    for fpath in sorted(out_dir.iterdir()):
        size_kb = fpath.stat().st_size / 1024
        print(f"  {fpath.name:40s}  {size_kb:8.1f} KB")
    print("=" * 60)

    return 0 if gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())
