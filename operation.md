# DRIFT · Step A 验证实验 · 操作手册

> **目标**：验证"不同来源的图像经 DDIM 逆向后，终点噪声 x_T 的统计分布是否可区分"
> **执行时间估计**：Mock 测试 5 分钟 / 真实实验 视数据量和 GPU 而定（100张/类约 30~60 分钟）
> **结果交付**：将 `results/step_a/` 整个目录发回即可

---

## 第零步：确认环境

```bash
# 检查 Python 版本（需要 3.9+）
python --version

# 检查 PyTorch 和 GPU
python -c "import torch; print('PyTorch:', torch.__version__); print('MPS可用:', torch.backends.mps.is_available()); print('CUDA可用:', torch.cuda.is_available())"

# 检查必要包（缺少的用 pip install 安装）
python -c "import scipy, sklearn, matplotlib, PIL, tqdm; print('依赖包全部OK')"
```

如果缺少包，运行：

```bash
pip install scipy scikit-learn matplotlib pillow tqdm numpy
```

---

## 第一步：定位项目根目录

所有命令都在以下目录执行：

```bash
cd /Users/joces/Downloads/Coding_Campus/Deepfake_Project
```

验证目录结构：

```bash
ls DRIFT/experiments/
# 应该看到：step_a_validation.py  README_step_a.md  phase1_binary.py  phase2_enhanced.py
```

---

## 第二步：Mock 模式验证（不需要真实模型，先确认脚本能跑通）

**这一步用随机生成的模拟数据测试脚本本身是否正常工作。**

```bash
python DRIFT/experiments/step_a_validation.py \
  --mock \
  --num_samples 15 \
  --output_dir ./results/step_a_mock
```

预期输出（约 30 秒）：

```
[Mock] 模拟来源：real, ProGAN, SD_v1.4, DALL-E2, Midjourney
...
[Test 1] 执行 t-SNE 可视化...  → [PASS] 或 [FAIL]
[Test 2] 功率谱密度分析...    → [PASS] 或 [FAIL]
[Test 3] Wasserstein 距离矩阵... → [PASS] 或 [FAIL]
[Test 4] 高斯性检验...        → [PASS] 或 [FAIL]

总计：X/4 项通过
>>> 验证门：PASS <<<   ← 3/4 即通过
```

如果 Mock 模式也报错，请截图错误信息并停止，等待老师回来确认。

---

## 第三步：准备真实数据

### 3.1 数据目录结构

在项目目录下创建如下结构（子目录名即为来源标签）：

```
data/
├── real/           ← 真实图像（FFHQ / LSUN / ImageNet 等）
│   ├── 000001.jpg
│   ├── 000002.png
│   └── ...
├── ProGAN/         ← ProGAN 生成图像
│   └── ...
├── StyleGAN2/      ← StyleGAN2 生成图像（可选）
│   └── ...
├── SD_v1.4/        ← Stable Diffusion v1.4 生成图像
│   └── ...
└── ADM/            ← ADM 生成图像（可选）
    └── ...
```

**最低要求**：`real/` 和至少一个假图目录，每类至少 50 张图像。
**推荐**：每类 100 张，覆盖 real + ProGAN + SD_v1.4 三类（实验结果最有说服力）。

图像格式：`.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp` 均可，256×256 或以上分辨率。

### 3.2 检查数据是否就绪

```bash
# 查看每个子目录的图像数量
for d in data/*/; do
  echo "$d: $(ls $d | grep -c '\\.')"
done
```

---

## 第四步：准备 ADM 预训练模型

ADM 模型文件：`256x256_diffusion_uncond.pt`

**确认模型路径**（和老师确认这个文件在哪里）：

```bash
# 如果不知道路径，搜索一下
find / -name "256x256_diffusion_uncond.pt" 2>/dev/null
```

如果没有，从官方下载（约 2GB，需要稳定网络）：

```bash
wget -c "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt" \
     -O ./models/256x256_diffusion_uncond.pt
```

---

## 第五步：运行真实 Step A 实验

### 命令（标准配置）

```bash
python DRIFT/experiments/step_a_validation.py \
  --data_dir ./data \
  --model_path /path/to/256x256_diffusion_uncond.pt \
  --output_dir ./results/step_a \
  --num_samples 100 \
  --ddim_steps 20 \
  --image_size 256
```

**把 `/path/to/256x256_diffusion_uncond.pt` 替换为实际路径。**

### 如果显存不足，使用以下精简配置

```bash
python DRIFT/experiments/step_a_validation.py \
  --data_dir ./data \
  --model_path /path/to/256x256_diffusion_uncond.pt \
  --output_dir ./results/step_a \
  --num_samples 50 \
  --ddim_steps 10 \
  --image_size 256
```

### 运行过程说明

```
初始化扩散模型...           # 加载 ADM 模型（约 30 秒）
[real] 处理第 1/100 张...   # 每张图像做 20 次 UNet 前向
[real] 处理第 2/100 张...   # 每张约 5~15 秒（GPU）
...
[特征] 提取并保存 x_T...
[Test 1] 执行 t-SNE...      # 约 1~2 分钟
[Test 2] PSD 分析...
[Test 3] Wasserstein 矩阵...
[Test 4] KS 检验...
```

进度条会持续刷新，属于正常现象。不要中途关闭终端。

---

## 第六步：查看和记录结果

实验完成后，输出目录包含：

```
results/step_a/
├── validation_gate_report.txt   ← 【最重要】PASS/FAIL 判定报告
├── tsne_visualization.png       ← t-SNE 可视化（不同来源的聚类图）
├── psd_comparison.png           ← 功率谱密度对比图
├── wasserstein_heatmap.png      ← 生成器之间的距离热力图
├── ks_test_results.json         ← 每类来源的高斯性检验数值
└── raw_xT_features.npz          ← 原始 x_T 特征（供后续分析）
```

### 如何读 validation_gate_report.txt

```bash
cat results/step_a/validation_gate_report.txt
```

报告格式示例：

```
============================================================
DRIFT Phase 0 — Step A 验证报告
============================================================
[PASS] Test1_tSNE
       Silhouette Score: 0.412（> 0.3 阈值）
       说明：不同来源的 x_T 在 t-SNE 空间中有明显聚类分离

[PASS] Test2_PSD
       SD 来源在 f=1/8 处的 z-score: 3.21（> 2.0 阈值）
       说明：SD-VAE 的 8px 周期性伪影在 x_T 频域中可见

[FAIL] Test3_Wasserstein
       类间/类内比值: 1.32（< 1.5 阈值）
       说明：Wasserstein 距离尚未达到阈值，需更多样本

[PASS] Test4_KS
       real: p=0.51（高斯）
       ProGAN: p=0.001（非高斯）
       SD_v1.4: p=0.031（非高斯）

总计：3/4 项通过
>>> 验证门：PASS <<<   （3/4 即通过）
```

---

## 第七步：需要记录并反馈的内容

请将以下内容整理好，等老师回来分析：

### 必须提供

- [ ] `validation_gate_report.txt` 的完整内容（复制粘贴或截图）
- [ ] `tsne_visualization.png`（看图片：不同颜色的点是否分开了？）
- [ ] `psd_comparison.png`（看图片：SD 那条线在 1/8 处是否有明显峰值？）
- [ ] `wasserstein_heatmap.png`（热力图：对角线颜色浅，非对角线颜色深 = 好结果）
- [ ] `ks_test_results.json` 中的数值（每类来源的 KS 统计量和 p 值）

### 补充记录（越详细越好）

- 实际使用的图像类别和每类数量（如：real=100张，ProGAN=100张，SD=80张）
- 实验使用的 GPU 型号和每张图像的处理时间
- 是否有任何报错或警告信息

### 反馈模板（直接填写）

```
实验时间：____
GPU：____
图像类别及数量：____
验证门结果：__ / 4 项通过，总体 [PASS/FAIL]

Test 1 (t-SNE): [PASS/FAIL], Silhouette Score = ____
Test 2 (PSD):   [PASS/FAIL], SD z-score = ____
Test 3 (WD):    [PASS/FAIL], 类间/类内比 = ____
Test 4 (KS):    [PASS/FAIL], real p值 = ____, ProGAN p值 = ____

主观观察：
- t-SNE图：不同来源的点 [是/否/部分] 分开了
- PSD图：SD 在 f=1/8 处 [有/没有] 明显峰值
- 其他备注：____
```

---

## 常见问题

| 问题 | 原因 | 解决方法 |
|------|------|---------|
| `ModuleNotFoundError: guided_diffusion` | 路径问题 | 确认从项目根目录运行 |
| `RuntimeError: out of memory` | GPU 显存不足 | 加 `--batch_size 1` 参数 |
| `FileNotFoundError: model` | 模型路径错误 | 检查 `--model_path` 参数 |
| `PIL.Image` 相关错误 | Pillow 版本 | `pip install --upgrade pillow` |
| t-SNE 非常慢 | 样本量大 | 临时改为 `--num_samples 30` |
| 脚本卡住不动 | 第一张图像慢 | ADM 加载需要时间，等待 60 秒 |

---

## 实验结束后的文件备份

```bash
# 把结果打包，方便传输
zip -r step_a_results.zip results/step_a/

# 同时备份原始特征（供老师做进一步分析）
# raw_xT_features.npz 包含所有图像的 x_T，很有价值，务必保留
```

---

*文档版本 v1.0 · 2026.03 · DRIFT 项目*
