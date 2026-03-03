# DRIFT · 代码库理解文档

> **阅读目标**：理解 DRIFT 项目在做什么、每个代码文件的职责、Step A 的验证逻辑
> **适合读者**：熟悉 Python/PyTorch 基础，了解深度学习基本概念
> **阅读时间**：约 30~40 分钟

---

## 一、研究背景（用一张图理解）

```
传统方法 DIRE 的做法：

  真实图像 x₀ ─→ [添加噪声] ─→ x_T ─→ [去噪重建] ─→ x₀'
                                              ↓
                                  重建误差 = |x₀ - x₀'|
                                  真图误差小，假图误差大

问题：需要前向+反向各20次 UNet = 共40次，极慢；只能判断真/假，无法识别是哪个生成器。

─────────────────────────────────────────────────────────────

我们的方法 DRIFT 的做法：

  任意图像 x₀ ─→ [DDIM单向逆向，仅10~20次] ─→ 逆向终点 x_T
                                                       ↓
                             提取 x_T 的统计特征（频域/轨迹/分布）
                                                       ↓
                             → 真/假检测  AND  → 生成器归因

核心假设：不同来源的图像逆向到噪声空间后，x_T 的统计特征不同。
Step A 的目标：用实验验证这个假设是否成立。
```

---

## 二、项目目录结构（带注释）

```
Deepfake_Project/
├── AIGCDetectBenchmark-main/          # 已有代码库1：9种检测方法集成
│   └── preprocessing_model/
│       └── guided_diffusion/          # ← DRIFT 复用这里的 ADM 扩散模型代码
│           ├── gaussian_diffusion.py  # DDIM 逆向核心算法（直接复用，不修改）
│           ├── unet.py               # ADM 的 UNet 神经网络
│           └── script_util.py        # 模型创建工具函数
│
├── DFFreq-main-main/                  # 已有代码库2：频域检测方法
│   └── networks/FreqLC.py            # 频域分析参考（DRIFT 参考其思路）
│
├── SDAIE-main/                        # 已有代码库3：自监督EXIF检测
│   └── oc_funs.py                    # GMM 多类别检测（后续 Wave 3 复用）
│
└── DRIFT/                             # ← 我们的新方法代码库
    ├── configs/
    │   └── drift_default.yaml        # 所有超参数的默认配置
    │
    ├── data/
    │   ├── dataloader.py             # 数据加载器（兼容已有数据集格式）
    │   └── transforms.py            # 图像预处理（裁剪/归一化）
    │
    ├── models/
    │   ├── backbone/
    │   │   └── adm_wrapper.py       # ★ ADM封装：提供 .invert(x0) 接口
    │   ├── features/
    │   │   ├── base.py              # 特征提取器抽象基类
    │   │   ├── endpoint.py          # F1：x_T 端点统计特征（128维）
    │   │   ├── trajectory.py        # F2：逆向轨迹特征（64维）
    │   │   ├── frequency.py         # 频域特征（64维，针对扩散模型假图）
    │   │   └── combined.py          # 组合提取器（F1+F2+FREQ）
    │   ├── heads/
    │   │   ├── binary.py            # 真/假二分类头
    │   │   └── attribution.py       # 生成器归因头（Phase 3）
    │   └── preprocessing/
    │       └── feature_cache.py     # x_T 预计算缓存（加速训练）
    │
    ├── training/
    │   ├── trainer.py               # 通用训练循环
    │   └── losses.py                # 损失函数（BCE + 归因损失）
    │
    ├── evaluation/
    │   ├── metrics.py               # AUC/AP/归因ACC/推理时间
    │   └── evaluator.py             # 评估器（生成论文格式表格）
    │
    ├── utils/
    │   ├── logger.py                # 日志
    │   ├── visualization.py         # t-SNE/PSD/Wasserstein 可视化
    │   └── checkpointing.py         # 模型保存/加载
    │
    ├── experiments/
    │   ├── step_a_validation.py     # ★ 当前任务：Step A 统计验证
    │   ├── phase1_binary.py         # Phase 1：训练二分类检测器
    │   └── phase2_enhanced.py       # Phase 2：增强特征+扩散假图检测
    │
    ├── docs/
    │   └── theory_framework.md      # 数学理论框架（含LaTeX推导）
    ├── operation.md                 # 操作手册（本文档的姊妹文档）
    └── codebase.md                  # 本文档
```

---

## 三、核心概念：DDIM 逆向是什么？

### 3.1 扩散模型的正向过程（加噪）

扩散模型训练时，把真实图像 x₀ 逐步加噪声，最终变成纯噪声 x_T：

```
x₀（真实图像）→ x₁ → x₂ → ... → x_T（近似高斯噪声）
              +噪声  +噪声         最终 ≈ N(0, I)

数学公式：q(x_t | x_0) = N(√ᾱ_t · x_0, (1-ᾱ_t) · I)

ᾱ_t：控制噪声程度的系数，t越大，ᾱ_t越小，图像信息越少
```

### 3.2 DDIM 逆向过程（去噪，用于生成）

生成图像时，从纯噪声 x_T 开始，UNet 预测每步的噪声，逐步去噪恢复图像：

```
x_T（随机噪声）→ x_{T-1} → ... → x₁ → x₀（生成图像）
              -噪声(UNet)   -噪声(UNet)

DDIM 更新公式：
x_{t-1} = √ᾱ_{t-1} · x̂₀(x_t) + √(1-ᾱ_{t-1}) · ε_θ(x_t, t)

其中 x̂₀(x_t) 是 UNet 对原图的预测
```

### 3.3 DRIFT 的逆向方向（与生成方向相反）

DRIFT 把一张未知图像 x₀（真实或假的）反向映射回噪声：

```
x₀（输入图像）→ x_t1 → x_t2 → ... → x_T（终点噪声）
             UNet×K次（K≈10~20）

关键问题：这个终点 x_T 和图像的"来源"有关系吗？
Step A 就是在回答这个问题。
```

---

## 四、Step A 验证脚本详解

**文件**：`DRIFT/experiments/step_a_validation.py`（830行）

### 4.1 整体流程

```python
main():
    1. 解析命令行参数（--data_dir, --model_path, --num_samples, --mock...）
    2. 初始化 ADM 模型（真实模型 或 mock 随机数）
    3. 遍历 data_dir 的每个子目录（子目录名 = 来源标签）
    4. 对每张图像做 DDIM 逆向，获取 x_T
    5. 保存 raw_xT_features.npz
    6. 执行 4 项统计检验
    7. 评估验证门（3/4 通过则 PASS）
    8. 生成报告和可视化图
```

### 4.2 DDIM 逆向部分（第 130~200 行）

```python
# 导入已有代码（不重写，直接用 DIRE 的实现）
from guided_diffusion.gaussian_diffusion import ...
from guided_diffusion.script_util import create_model_and_diffusion

def run_ddim_inversion(model, diffusion, img_tensor, ddim_steps):
    """
    输入：img_tensor [1, 3, 256, 256]，值域 [-1, 1]
    输出：x_T [1, 3, 256, 256]，逆向终点噪声
    """
    # 调用已有的 ddim_reverse_sample_loop 函数
    # 这个函数在 gaussian_diffusion.py 第 720-754 行已经实现
    x_T = diffusion.ddim_reverse_sample_loop(
        model,
        shape=(1, 3, H, W),
        noise=img_tensor,    # 以输入图像作为"起点"
        eta=0.0,             # 确定性逆向（不加额外随机性）
    )
    return x_T
```

### 4.3 四项统计检验

**Test 1：t-SNE 可视化（第 260~320 行）**

```python
def test_tsne(features, labels, output_path):
    """
    features: [N, D]  N张图像的 x_T 展平后的特征
    labels:   [N]     每张图像的来源标签（0=real, 1=ProGAN...）

    步骤：
    1. PCA 降维：D维 → 50维（加速后续 t-SNE）
    2. t-SNE：50维 → 2维（用于可视化）
    3. 计算 Silhouette Score（衡量聚类质量）
       - 接近 1.0：聚类完美分开
       - 接近 0.0：聚类重叠
       - 负数：聚类混乱
    4. 阈值：Silhouette > 0.3 则 PASS
    """

# 判断标准：不同来源的点在2D图中是否自然分开成团？
```

**Test 2：功率谱密度分析（第 330~400 行）**

```python
def test_psd(xT_by_source, output_path):
    """
    对每类来源的 x_T 计算功率谱密度（PSD = |FFT|²）

    关键：检测 SD/Stable Diffusion 的 8px 周期性伪影

    原因：SD 使用 VAE 做 8× 空间下采样
          解码后的图像在 8px 间隔处有轻微"格子"伪影
          这个伪影在频域中表现为 f=1/8（即 H//8, W//8 处）的峰值

    使用 torch.fft.rfft2(x_T) 计算2D频谱
    在 H//8 ± 1 和 W//8 ± 1 位置检测能量是否显著高于周围
    用 z-score 量化显著性（z-score > 2.0 则 PASS）
    """
```

**Test 3：Wasserstein 距离矩阵（第 410~480 行）**

```python
def test_wasserstein(xT_by_source, output_path):
    """
    计算每对来源之间的 Wasserstein 距离（分布差异度量）

    好结果：类间距离（不同来源之间）>> 类内距离（同一来源内部）
    衡量：ratio = 类间均值 / 类内均值
          ratio > 1.5 则 PASS

    可视化为 N×N 热力图：
    - 对角线（同类）：颜色浅（距离小）
    - 非对角线（不同类）：颜色深（距离大）

    这是最直观的生成器归因指标：
    如果热力图对角线最浅，说明每个生成器都有独特的 x_T 指纹
    """
```

**Test 4：高斯性检验（第 490~550 行）**

```python
def test_ks_gaussianity(xT_by_source, output_path):
    """
    Kolmogorov-Smirnov 检验：x_T 是否符合标准高斯分布 N(0,1)?

    理论预测：
    - 真实图像：在 ADM 流形上，逆向后 x_T ≈ N(0,1)，p-value 应该大
    - GAN 假图：不在流形上，逆向后 x_T 有偏差，p-value 应该小
    - SD 假图：VAE 伪影导致轻微偏差，p-value 介于中间

    判断标准：
    - real 的 p-value > 0.05（接受高斯假设）
    - 其他来源的 p-value < 0.05（拒绝高斯假设）
    → 同时满足则 PASS
    """
```

### 4.4 验证门逻辑（第 560~600 行）

```python
def evaluate_validation_gate(test_results):
    """
    4项检验中至少3项 PASS → 总体 PASS → 项目可进入 Phase 1

    这个设计允许1项失败：
    - Test 3 (Wasserstein) 对样本量最敏感，少样本容易失败
    - 主要看 Test 1 (t-SNE) 和 Test 4 (KS)，最有说服力
    """
```

---

## 五、关键文件速查

### 5.1 ADM 封装（`models/backbone/adm_wrapper.py`）

```python
class ADMBackbone:
    def invert(self, x0: Tensor, return_intermediates=False):
        """
        输入：x0 [B, 3, H, W]，值域 [-1, 1]
        输出：x_T [B, 3, H, W]
              intermediates（可选）：逆向过程中的K个中间状态

        内部调用：gaussian_diffusion.ddim_reverse_sample_loop()
        这是 DIRE 项目已实现的函数，DRIFT 直接复用
        """

    def mock_invert(self, x0):
        """当 model_path="mock" 时调用，返回随机 tensor，用于测试"""
```

### 5.2 F1 端点特征（`models/features/endpoint.py`）

```python
class EndpointFeatureExtractor(FeatureExtractor):
    """
    feature_dim = 128

    5组特征：
    G1 (12维) - 统计矩：每通道的 mean/std/skewness/kurtosis
                用途：检测 x_T 的分布偏移

    G2  (3维) - 高斯性偏差：每通道的 KS 统计量
                用途：量化偏离 N(0,1) 的程度

    G3  (9维) - 空间自相关：lag=1,2,4 的水平/垂直自相关
                用途：检测周期性结构（GAN的checkerboard）

    G4 (24维) - 径向 PSD：8个频带的能量分布
                用途：频域指纹

    G5 (12维) - VAE 峰值检测：f=1/8 处的能量
                用途：专门检测 SD/DALL-E 的8px VAE伪影
    """
```

### 5.3 F2 轨迹特征（`models/features/trajectory.py`）

```python
class TrajectoryFeatureExtractor(FeatureExtractor):
    """
    feature_dim = 64
    需要 intermediates（中间状态序列）

    4组特征：
    G1 (10维) - 步长序列：每步 ||x_{t+1} - x_t|| 的分布
                真实图像轨迹步长均匀；GAN假图初期步长大

    G2 (10维) - 轨迹曲率：二阶差分（轨迹"弯曲程度"）
                GAN假图曲率高（score function 预测不稳定）

    G3  (8维) - 全局统计：路径长度/直线度/变异系数等

    G4 (16维) - 频域演化：不同时间段的频域变化趋势
    """
```

---

## 六、数据流图（完整流程）

```
图像文件（jpg/png）
      │
      ▼
transforms.py          → 缩放到 256×256，归一化到 [-1, 1]
      │
      ▼
adm_wrapper.py         → 调用 gaussian_diffusion.ddim_reverse_sample_loop()
.invert(x0)               UNet 做 K=20 次前向推断
      │
      ├──→ x_T（终点噪声）[B, 3, 256, 256]
      └──→ intermediates（可选，K个中间状态）
                │
                ▼
endpoint.py            → 统计矩 + 高斯性 + 自相关 + PSD + VAE峰值
.extract(x_T)             输出：128维特征向量
                │
                ▼
(Step A) 统计检验：      → t-SNE + PSD分析 + Wasserstein + KS检验
test_tsne / test_psd...    输出：4张图 + 1份报告 + PASS/FAIL判定
```

---

## 七、Mock 模式 vs 真实模式的区别

| | Mock 模式 | 真实模式 |
|---|---|---|
| ADM 模型 | 不需要，用随机 tensor | 需要加载 2GB 权重 |
| 数据来源 | 用不同均值/方差的高斯分布模拟 | 实际的图像文件 |
| x_T 来源 | 直接生成随机数（模拟不同来源） | ADM 对真实图像做 DDIM 逆向 |
| 运行时间 | ~30 秒 | 取决于GPU，每张图5~15秒 |
| 结果意义 | 只验证脚本逻辑，数值无科学意义 | 真正的科学验证 |
| 用途 | 调试、确认脚本能跑通 | 得到实验结论 |

---

## 八、如何确认代码在正确运行

### 正常运行的标志

```
✓ 看到 "[real] 处理第 X/100 张..." 字样且持续递增
✓ GPU 利用率在 50%~100%（用 nvidia-smi 或 istat 查看）
✓ 4项 Test 逐一完成，每项显示 [PASS] 或 [FAIL]
✓ 最后打印 ">>> 验证门：PASS/FAIL <<<"
✓ results/step_a/ 目录有6个文件
```

### 异常情况处理

```
问题：脚本卡在"初始化扩散模型..."超过2分钟
原因：模型权重正在从磁盘加载（2GB，正常）
处理：继续等待

问题：CUDA out of memory / MPS out of memory
原因：GPU 显存不足
处理：增加参数 --batch_size 1（每次只处理一张图）

问题：KeyError 或 AttributeError
原因：代码版本不匹配
处理：记录完整错误信息，等待老师处理

问题：t-SNE 运行超过10分钟
原因：样本量过多，正常（100样本×5类=500点的t-SNE）
处理：继续等待，或减小 --num_samples 到 50
```

---

## 九、结果如何对应研究假设

| Test | 假设 | PASS 说明 | FAIL 说明 |
|------|------|-----------|-----------|
| t-SNE | 不同来源的 x_T 自然分簇 | 假设成立！可以用聚类做归因 | 需要更多步数或不同特征 |
| PSD | SD-VAE 伪影在 x_T 中可见 | SD 假图可以被频域检测 | 伪影被 ADM 逆向"抹平"了 |
| Wasserstein | 不同生成器的 x_T 分布有距离 | 生成器归因在信息论上可行 | 需要更多样本或更强特征 |
| KS | 真图 x_T 更接近高斯 | 流形分离假设成立 | 真图也偏离高斯，需重新分析 |

**3/4 PASS → 研究可以继续（进入 Phase 1 训练）**
**2/4 或以下 PASS → 需要回到老师讨论，可能需要调整策略**

---

## 十、本次任务之外的代码（了解即可，不需要运行）

| 文件 | 用途 | 何时用 |
|------|------|--------|
| `phase1_binary.py` | 训练 F1 特征的二分类检测器 | Step A PASS 后 |
| `phase2_enhanced.py` | 训练 F1+F2+频域的增强检测器 | Phase 1 验证后 |
| `models/heads/attribution.py` | 生成器归因（S3 两阶段） | Wave 3 |
| `docs/theory_framework.md` | 数学理论推导（LaTeX） | 写论文时 |

---

*文档版本 v1.0 · 2026.03 · DRIFT 项目*
*如有疑问，记录下来等待老师回来讨论*
