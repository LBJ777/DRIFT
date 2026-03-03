# DRIFT Phase 0 Step A — x_T 统计分布验证

**目的**：在不训练任何分类器的前提下，用统计方法验证 DRIFT 核心假设：真实图像与 AI 生成图像经 DDIM 逆向后的噪声终点 x_T 具有统计上可区分的分布差异。

---

## 环境依赖

Python >= 3.9，使用如下命令安装依赖：

```bash
pip install torch torchvision
pip install scipy scikit-learn matplotlib pillow numpy tqdm
```

| 包 | 用途 |
|---|---|
| `torch` | DDIM 逆向计算 / 张量操作 |
| `scipy` | Wasserstein 距离、KS 检验 |
| `scikit-learn` | PCA、t-SNE、Silhouette Score |
| `matplotlib` | 可视化图像保存 |
| `pillow` | 图像加载与预处理 |
| `numpy` | 数值计算与特征保存 |
| `tqdm` | 逆向进度条（被现有代码依赖） |

---

## 如何下载 ADM 预训练模型

本脚本使用 OpenAI 的 Ablated Diffusion Model (ADM) 执行 DDIM 逆向。

**256×256 无条件模型（推荐）**：

```bash
mkdir -p ./models
wget -c "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt" \
     -O ./models/256x256_diffusion_uncond.pt
```

文件大小约 2.0 GB。下载后通过 `--model_path` 参数指定路径。

---

## 快速验证（Mock 模式）

Mock 模式无需下载模型，使用参数化高斯分布模拟各来源的 x_T，可立即验证脚本逻辑：

```bash
cd /Users/joces/Downloads/Coding_Campus/Deepfake_Project/DRIFT/experiments

# 最小验证（10 个样本/类，5 类来源）
python step_a_validation.py \
  --mock \
  --num_samples 10 \
  --output_dir ./results/step_a_mock

# 自定义 Mock 来源名称
python step_a_validation.py \
  --mock \
  --num_samples 50 \
  --mock_sources "real,ProGAN,SD_v1.4,DALL-E2,Midjourney" \
  --output_dir ./results/step_a_mock
```

Mock 模式下各来源的模拟统计特性：

| 来源标签 | 模拟分布 | 说明 |
|---|---|---|
| `real` | N(0, 1) | 完美高斯，模拟 ADM 完美逆向 |
| `ProGAN` | N(0.1, 1.1²) | 均值/方差偏移，模拟 GAN 伪影 |
| `SD_*` / `stable*` | N(0,1) + 8px 周期信号 | 叠加 VAE 块状伪影 |
| 其他 | N(0.05, 1.05²) | 轻度偏移 |

---

## 真实模式（含 ADM 模型）

数据目录组织方式（子目录名即为来源标签）：

```
/path/to/images/
├── real/          # 真实图像
├── ProGAN/        # ProGAN 生成
├── SD_v1.4/       # Stable Diffusion v1.4 生成
├── DALL-E2/       # DALL-E 2 生成
└── Midjourney/    # Midjourney 生成
```

运行命令：

```bash
python step_a_validation.py \
  --data_dir /path/to/images \
  --model_path ./models/256x256_diffusion_uncond.pt \
  --output_dir ./results/step_a \
  --num_samples 100 \
  --ddim_steps 20 \
  --image_size 256
```

---

## 输出文件说明

```
results/step_a/
├── tsne_visualization.png       # t-SNE 散点图：颜色区分来源
├── psd_comparison.png           # PSD 热图 + 各来源径向平均曲线
├── wasserstein_heatmap.png      # N×N Wasserstein 距离热力图
├── ks_test_results.json         # KS 检验数值结果（含 p 值）
├── validation_gate_report.txt   # 验证门报告（最重要）
└── raw_xT_features.npz          # 原始 x_T 特征矩阵（供后续 Phase 使用）
```

---

## 如何解读 validation_gate_report.txt

报告结构示例：

```
============================================================
DRIFT Phase 0 Step A — Validation Gate Report
============================================================

[PASS] Test1_tSNE
       Silhouette Score = 0.4823
       阈值 > 0.3
       说明：Silhouette > 0.3 视为聚类分离显著

[PASS] Test2_PSD
       SD PSD z-score @ f=H/8 = 3.1254
       阈值 > 2.0
       说明：SD 来源在 f=H/8 处 PSD z-score > 2.0 视为存在 VAE 伪影

[PASS] Test3_Wasserstein
       类间/类内 Wasserstein 比值 = 2.3847
       阈值 > 1.5
       说明：类间/类内比值 > 1.5 视为分布分离显著

[FAIL] Test4_KS
       [real] KS=0.012345, p=0.4512 (高斯)
       [ProGAN] KS=0.089234, p=0.0012 (非高斯)
       说明：real 来源 p-value > 0.05（高斯），其他来源 p-value < 0.05（非高斯）

------------------------------------------------------------
总计：3/4 项通过

>>> 验证门：PASS <<<
核心假设成立：不同来源的 x_T 统计分布具有可区分性。
项目可进入 Phase 1（特征提取与分类器训练）。
============================================================
```

### 通过标准（4 选 3）

| 检验 | 指标 | 通过阈值 | 含义 |
|---|---|---|---|
| Test 1: t-SNE | Silhouette Score | > 0.3 | 不同来源在 2D 嵌入中有视觉可分离的聚类 |
| Test 2: PSD | SD 来源在 f=H/8 处 z-score | > 2.0 | SD/VAE 的 8px 量化伪影在频域显著可见 |
| Test 3: Wasserstein | 类间/类内距离比值 | > 1.5 | x_T 功率谱分布在类间差异大于类内差异 |
| Test 4: KS | real p>0.05，其他 p<0.05 | 方向正确 | 真实图像最接近高斯噪声，AI 图像偏离高斯 |

**结论**：3/4 或 4/4 项通过 → PASS，项目可进入 Phase 1；否则 → FAIL，需要重新审视数据或参数设置。

---

## 故障排查

**ImportError: 无法导入 guided_diffusion**
- 检查路径：`AIGCDetectBenchmark-main/preprocessing_model/guided_diffusion/` 目录是否完整
- 使用 Mock 模式绕过：`--mock`

**CUDA out of memory**
- 减少 `--num_samples`，或在 CPU 上运行：`--device cpu`

**t-SNE 运行过慢**
- 减少 `--num_samples`（建议 Mock 模式用 ≤ 50）

**Silhouette Score 为 N/A**
- 每类至少需要 2 个样本，增加 `--num_samples`

---

## 与 DRIFT 整体架构的关系

```
Phase 0 Step A (本脚本)  →  验证 x_T 分布可区分性
Phase 0 Step B           →  特征工程：选择最优 x_T 表示
Phase 1                  →  训练轻量分类器（基于 x_T 特征）
Phase 2                  →  端到端评估与对抗鲁棒性测试
```
