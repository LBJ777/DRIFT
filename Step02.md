# Step02: DRIFT 方法构建 · 对话文档
> 基于 Step01.md 代码审计结论 · 目标: CVPR/ICCV/NeurIPS 投稿
> 方法名: **DRIFT** (Diffusion Reverse Inversion Fingerprint Tracking)

---

## 一句话问题定义

> **DIRE 通过"先加噪再去噪"检测重建误差，效率极低且只能二分类；我们通过"单向逆向到噪声"提取图像的流形指纹，用一次前向完成真假检测与生成器归因。**

---

## Section 1 · 研究动机：从 DIRE 的弱点出发

### 1.1 DIRE 做了什么（代码确认）

```
输入 x₀  →  [加噪: x₀→x_T, UNet×20次]  →  [去噪: x_T→x₀', UNet×20次]  →  DIRE = |x₀ - x₀'|  →  ResNet分类
```

- 代码路径: `compute_dire.py:93-114`
- UNet调用: **40次**（20次逆向 + 20次重建）
- 输出: 像素级误差图，信息极度压缩
- 缺陷1: **效率极低** — 40次UNet前向，工业不可用
- 缺陷2: **只能二分类** — 无法区分生成器类型
- 缺陷3: **加噪破坏信息** — 添加噪声本身就改变了输入图像

### 1.2 我们的洞察

**问题**: 为什么要"先加噪再去噪"才能检测？我们的假设是**加噪这一步是多余的**。

**新思路**: 不加噪，直接用 DDIM 把图像逆向到噪声空间：

```
输入 x₀  →  [DDIM逆向: x₀→x_T, UNet×K次, K≈10~15]  →  提取轨迹+终点特征  →  检测 + 归因
```

这条"逆向轨迹"本身就是图像来源的指纹。

---

## Section 2 · 理论框架：为什么逆向轨迹是指纹？

### 2.1 概率流 ODE 视角

DDIM 逆向等价于求解**概率流 ODE (Probability Flow ODE)**:

```
dx = -½ · β(t) · [x + ∇_x log p_θ(x, t)] dt
```

其中 `∇_x log p_θ(x, t)` 是扩散模型学到的**得分函数 (score function)**，代表"当前 x_t 最可能向哪个方向移动以回到数据流形"。

### 2.2 假设 A — 流形分离性

| 图像来源 | 与 ADM 流形的关系 | DDIM 逆向轨迹 | 终点 x_T 分布 |
|---------|-----------------|--------------|--------------|
| **真实图像** | 在流形上（ADM 在真实图像上训练） | 平滑，沿 ODE 确定性路径 | ≈ N(0, I) |
| **GAN 假图** (ProGAN/StyleGAN) | 不在流形上（GAN 伪影是 OOD） | 轨迹异常，Score 预测不一致 | ≠ N(0, I)，有结构性偏差 |
| **扩散模型假图** (SD/ADM) | 部分在流形上，但 VAE 解码引入伪影 | 轨迹较平滑，但终点有频域签名 | ≈ N(0, I) 但有 8px 周期伪影 |

**形式化**（待实验验证）:

对于真实图像 x₀ ~ q_real：

```
x_T = DDIM_inv(x₀; ADM) ≈ ε,  ε ~ N(0, I)
```

对于 GAN 假图 x₀ ~ q_fake^GAN：

```
x_T = DDIM_inv(x₀; ADM)  →  E[||x_T||²_spectral] > E_real
（频域能量分布偏离高斯分布）
```

### 2.3 假设 B — 生成器指纹性

不同生成器的伪影在 DDIM 逆向后映射到**不同的噪声模式**：

```
W₂(P(x_T^real), P(x_T^GAN)) > δ₁     # 真图 vs GAN
W₂(P(x_T^SD),   P(x_T^GAN)) > δ₂     # SD 假图 vs GAN 假图
W₂(P(x_T^SD),   P(x_T^real)) > δ₃    # SD 假图 vs 真图（最关键）
```

**δ₃ 为什么 > 0（SD假图为何可被检测）**:

SD 使用 VAE 做 **8× 下采样**（像素空间 → 潜在空间 → 像素空间），
VAE 解码器在图像中引入**8像素空间周期性伪影**（f = 1/8 的功率谱峰值）。

这个伪影在经过 ADM DDIM 逆向后**保留在 x_T 中**，
因为 ADM 的 score function 不认识 VAE 伪影，无法"修正"它。

```python
# 可验证性: 在 x_T 的功率谱密度中检测
x_T_psd = torch.fft.rfft2(x_T).abs() ** 2
# SD 假图: psd[:, :, H//8, W//8] 有显著峰值
# 真实图像: 该频率无显著峰值
```

---

## Section 3 · 算法设计：四种特征提取方案

> **策略**: 先在小数据集上做消融，确认哪种方案最优，再全量实验。

### 方案 F1 — 终点统计特征（最高效，优先验证）

**输入**: 终点噪声 x_T
**特征维度**:

| 特征 | 计算方式 | 目标检测对象 |
|------|---------|------------|
| 高斯性度量 | KL(P(x_T) ‖ N(0,I)) | 全类型（偏离高斯即异常） |
| 功率谱密度 (PSD) | rfft2(x_T).abs()² | SD 的 8px VAE 伪影 |
| 高阶矩 | skewness, kurtosis | GAN 的分布偏移 |
| 空间自相关 | xcorr(x_T, shifts) | 周期性结构 |

**效率**: K 次 UNet 前向（K=10~15），**无需重建**
**代码复用**: `FreqLC.py:140-179` 的 FFT 分析模块

---

### 方案 F2 — 轨迹曲率特征（中等效率，专攻GAN）

**输入**: 稀疏时间步 {x_{t_k}}_{k=1}^K
**特征**:

```python
# 轨迹平滑性指标
smoothness = Σ ||x_{t_{k+1}} - x_{t_k}||₂ / K        # 总路径长
curvature  = Σ ||x_{t_{k+1}} - 2x_{t_k} + x_{t_{k-1}}||₂ / K  # 曲率

# 步长方差（真实图像轨迹更均匀）
step_var = Var(||x_{t_{k+1}} - x_{t_k}||₂)
```

**理论**: GAN 假图在逆向初期（t 小）轨迹变化剧烈，后期（t 大）趋于平稳。
**代码复用**: `gaussian_diffusion.py:756-805` 的 `ddim_reverse_sample_loop_progressive()`

---

### 方案 F3 — Score 一致性特征（可解释性最强）

**输入**: 每步 UNet 的噪声预测 ε_θ(x_{t_k}, t_k)
**特征**:

```python
# 预测一致性：真实图像的 score 预测更稳定
consistency = Var(ε_θ(x_{t_k}, t_k) for k in 1..K)

# 预测误差累积
pred_error = Σ ||x_{t_{k+1}} - DDIM_step(x_{t_k}, ε_θ(...))||₂
```

**可解释性**: Score function 的不确定性 = 图像"脱离流形"的程度
**代码复用**: `unet.py:634-663` 的 forward，提取中间 hs 列表

---

### 方案 F4 — 多分支融合（最高精度，最终方案）

```
x₀ → DDIM逆向(K步) → {x_{t_k}, ε_θ(x_{t_k})}_{k=1}^K + x_T
                              ↓
               ┌──────────────────────────────┐
               │  Branch A: x_T 频域统计       │ → f_end (32维)
               │  Branch B: 轨迹曲率序列       │ → f_traj (K×4维)
               │  Branch C: Score 一致性       │ → f_score (K×C维)
               └──────────────────────────────┘
                              ↓
               轻量级 Transformer (2层，4头) 或 MLP
                              ↓
               分类头1: 真/假 (Binary CE)
               分类头2: 生成器归因 (可选，Multi-class CE)
```

---

## Section 4 · 与现有方法的对比矩阵

| 维度 | DIRE | SDAIE | DFFreq | **DRIFT（我们）** |
|------|------|-------|--------|-----------------|
| 逆向方向 | 双向（加噪→去噪） | 无扩散 | 无扩散 | **单向（图像→噪声）** |
| UNet 调用次数 | **40次** | 0 | 0 | **10~15次** |
| 学习范式 | 有监督 | 自监督→有监督 | 有监督 | **灵活（一类/有监督）** |
| 生成器归因 | ✗ | ✗ | ✗ | **✓（端点指纹）** |
| SD 类假图检测 | ✓ (弱) | ✓ | ✓ | **✓（频域 VAE 检测）** |
| 可解释性 | 中（重建误差图） | 低 | 中（频谱） | **高（x_T 可视化 + 流形距离）** |
| 训练数据需求 | 真+假标注 | 仅真实 | 真+假标注 | **可选（见 Section 5）** |
| 理论基础 | 重建误差 | EXIF 先验 | 频域假设 | **概率流 ODE** |

---

## Section 5 · 训练策略（开放问题 1）

> **当前待决策**: 生成器归因的监督信号来源

### 策略 S1 — 纯无监督（训练数据效率最高）

```
只用真实图像:
1. 对真实图像集做 DDIM 逆向，建立 x_T 的"真实分布"参考
2. 测试图像的 x_T 与参考分布的距离 → 真/假判断
3. 对假图的 x_T 做聚类（K-means/GMM）→ 自动发现生成器指纹
```

- 优势: 无需假图标注数据，迁移性最强
- 风险: 归因精度依赖聚类质量
- 可参考: `SDAIE-main/oc_funs.py:127-150` 的 GMM 框架

### 策略 S2 — 有监督（精度最高）

```
用真+假标注数据:
- Head1: BCELoss (真/假)
- Head2: CrossEntropyLoss (生成器类型，15+类)
```

- 优势: 精度高，直接可比 SDAIE
- 风险: 需要各类型假图标注，新型生成器需要重训

### 策略 S3 — 两阶段（推荐备选）

```
Phase 1: 仅用真实图像 + 无监督聚类发现 x_T 的自然簇
Phase 2: 用少量标注数据将聚类对齐到生成器标签
```

- 优势: 均衡数据效率和精度
- 天然支持"发现未知生成器"的能力（高价值应用场景）

---

## Section 6 · 实验设计

### 6.1 数据集

| 阶段 | 数据集 | 用途 |
|------|--------|------|
| 训练 | ProGAN (FFHQ/LSUN) | 与 CNNSpot/DIRE 保持一致 |
| 跨生成器测试 | 17 个生成器（见下） | 泛化性测试（核心指标） |
| 消融 | ProGAN 子集 (10%) | 快速方案筛选 |

**17 个生成器**（来自 SDAIE 代码库 `oc_funs.py:25-34`）:
ProGAN · StyleGAN · StyleGAN2 · BigGAN · CycleGAN · StarGAN · GauGAN · ADM · Glide · Stable Diffusion v1.4 · SD v1.5 · VQDM · Wukong · DALLE2 · Midjourney · SDXL · StyleGAN3

### 6.2 消融实验顺序

```
Step A: 先验证假设（优先级最高）
  → 对不同来源图像做 DDIM 逆向，可视化 x_T 的功率谱
  → 确认不同生成器的 x_T 是否有统计差异（t-test / Wasserstein）
  → 此步骤无需训练，纯统计验证

Step B: 特征方案对比（F1→F4）
  → 同等计算预算下，哪种特征方案 AUC 最高？
  → 稀疏采样步数 K 的影响（K=5,10,15,20）

Step C: 训练策略对比（S1/S2/S3）
  → 在跨生成器测试集上的泛化性差异

Step D: 全量实验 + 对比 DIRE/SDAIE/DFFreq
```

### 6.3 核心评估指标

| 指标 | 说明 | 对标基线 |
|------|------|---------|
| AUC (binary) | 真/假二分类 | DIRE, CNNSpot, UnivFD |
| 跨生成器平均 AUC | 17 生成器平均 | SDAIE (reported) |
| 推理时间 (ms/img) | GPU, batch=1 | DIRE (慢) vs CNNSpot (快) |
| 生成器归因 ACC | 新指标，SOTA 无对比 | — |
| 参数量 (M) | 检测头的参数 | 轻量化声明 |

---

## Section 7 · 顶会创新点攻击覆盖矩阵

> 类比 Step01 方法论的"攻击覆盖矩阵"：每个创新点必须消解至少一个现有方法的理论弱点。

| 现有方法的弱点 | DIRE | SDAIE | DFFreq | DRIFT 的对应创新点 |
|--------------|------|-------|--------|-----------------|
| 效率极低（40次UNet） | ✗ | — | — | **C1: 单向逆向，10次即可** |
| 无生成器归因能力 | ✗ | ✗ | ✗ | **C2: x_T 端点指纹** |
| 对 SD/AIGC 假图弱 | △ | △ | △ | **C3: 频域 VAE 伪影检测** |
| 无理论解释（为什么有效） | ✗ | ✗ | △ | **C4: 概率流 ODE 框架** |
| 需要大量假图标注 | ✗ | ✓ | ✗ | **C5: 可选一类检测模式** |

> **填写规则**: ✓ = 解决；△ = 部分解决；✗ = 已知弱点

---

## Section 8 · 可复用代码原子（直接对应 Step01 技术模块）

| 模块 | 来源 | 文件路径 | 在 DRIFT 中的用途 |
|------|------|---------|-----------------|
| DDIM 逆向循环 | AIGCDetectBenchmark | `gaussian_diffusion.py:720-805` | 核心逆向过程 |
| 稀疏时间步采样 | AIGCDetectBenchmark | `respace.py:7-60` | K 步稀疏采样 |
| UNet 中间层特征 | AIGCDetectBenchmark | `unet.py:634-663` (hs 列表) | F3 方案特征提取 |
| FFT 频域分析 | DFFreq | `FreqLC.py:140-179` | x_T 的功率谱密度 |
| GMM 多类别检测 | SDAIE | `oc_funs.py:127-150` | S1 无监督聚类归因 |
| 对比蒸馏正则化 | SDAIE | `bc_train.py:52-107` | 训练稳定性（可选） |
| MPNCOV 协方差池化 | SDAIE | `MPNCOV.py:14-143` | F4 融合的特征池化（可选） |

---

## Section 9 · 所有决策状态（已全部关闭）

| 决策项 | 状态 | 决定 |
|--------|------|------|
| 骨干模型 | ✅ | ADM (guided-diffusion) |
| 逆向方向 | ✅ | 单向 x₀→x_T，K=10~15 稀疏步 |
| 归因策略 | ✅ | S3 两阶段（Phase1 GMM + Phase2 200张/类对齐） |
| 稀疏采样 | ✅ | 均匀采样为主实验，前段密集为消融 |
| 理论深度 | ✅ | 并行推进（实验驱动 + 附录理论框架） |
| 聚类算法 | ✅ | GMM（复用 oc_funs.py:127-150） |
| 对比基线 | ✅ | DIRE · SDAIE · UnivFD · CNNSpot · DFFreq · Gram-Net |
| DFFreq 创新性 | ✅ | 无冲突（作用对象不同：x₀ vs x_T），cite 并比较 |
| 投稿目标 | ✅ | 首个统一检测+生成器归因框架，CVPR/ICCV/NeurIPS |

---

## Section 10 · 完整实验路径规划（Step A → 投稿）

> **最终目标**: 首个统一 Deepfake 检测 + 生成器归因框架，投顶会

---

### Phase 0 — 假设验证（无需训练，1~2 周）

**Step A: x_T 分布差异统计验证** ← 当前位置，立即启动

```
使用工具: gaussian_diffusion.py:720-805 (现有代码，无需修改)
输入: 各 100 张来自 [real/FFHQ, ProGAN, StyleGAN2, SD_v1.4, ADM] 的图像
操作: DDIM 逆向（K=20步），提取 x_T

验证指标:
  1. t-SNE 可视化 x_T → 不同来源是否自然分开？
  2. 对每类 x_T 计算功率谱密度 → SD 在 f=1/8 处是否有峰值？
  3. Wasserstein 距离矩阵（5×5）→ 类间距 vs 类内距
  4. 高斯性检验（Kolmogorov-Smirnov test）→ 真图 x_T 最接近 N(0,I)？
```

**验证门**: 所有4项指标中至少3项显示统计显著差异（p < 0.05），才进入 Phase 1。

**两种可能结果及对策**:

| 结果 | 对策 |
|------|------|
| x_T 直接可分（t-SNE 有明显簇） | 直接用 F1 特征，论文叙事最简洁 |
| x_T 不可分，但轨迹统计可分 | 以 F2/F3 为主，x_T 降为辅助特征 |

---

### Phase 1 — 最小可行原型（2~3 周）

**目标**: 证明 DRIFT 在二分类上优于或匹配 DIRE，同时效率提升 3x

**Step 1.1: 实现 DRIFT-F1（端点统计特征 + 二分类头）**

```python
# 复用: gaussian_diffusion.py 的 ddim_reverse_sample_loop
# 新增: 端点特征提取器（FFT + 统计矩）
# 复用: util.py:86-94 的 ResNet50 分类头（替换输入）

class DRIFT_F1(nn.Module):
    def __init__(self, diffusion, model):
        self.ddim_inv = diffusion.ddim_reverse_sample_loop  # 现有函数
        self.freq_extractor = FreqEndpointExtractor()        # 新写，~30行
        self.classifier = ResNet50_head()                    # 复用现有
```

**Step 1.2: 第一轮对比实验**

| 方法 | 数据集 | 指标 |
|------|--------|------|
| DIRE | Wang2020 (ProGAN训练) | AUC, ms/img |
| CNNSpot | 同上 | AUC, ms/img |
| **DRIFT-F1** | 同上 | AUC, ms/img |

**验证门（Layer 4 Level 3）**: DRIFT-F1 的 AUC ≥ DIRE-0.5%（精度相当），ms/img ≤ DIRE × 0.4（效率提升2.5x+），才继续 Phase 2。

---

### Phase 2 — 特征增强 + SD 类假图检测（2~3 周）

**目标**: 解决 SD/AIGC 类假图检测 gap，加入轨迹特征

**Step 2.1: 验证并集成 SD-VAE 频域检测**

```python
# 在 Phase 0 验证通过后:
# 在 F1 特征提取器中加入 SD-VAE 伪影检测分支
# 对 x_T 做 rfft2，提取 f=1/8 频率区域的功率谱能量作为额外特征维度
```

**Step 2.2: 实现 F2 轨迹平滑性特征**

```python
# 复用: ddim_reverse_sample_loop_progressive (返回中间状态)
# 新增: 轨迹曲率计算（~20行）
# 组合: DRIFT-F1+F2 联合特征
```

**Step 2.3: 跨生成器 17 类泛化测试（核心指标）**

| 方法 | 跨17生成器平均AUC | 目标 |
|------|-----------------|------|
| DIRE | ~70% (reported) | — |
| SDAIE | ~80% (reported) | 需要超越 |
| UnivFD | ~75% (reported) | 需要超越 |
| **DRIFT-F1+F2** | ? | > 80% |

**验证门**: 跨生成器平均 AUC > SDAIE reported，才进入 Phase 3（生成器归因）。

---

### Phase 3 — 生成器归因系统 S3（3~4 周）

**目标**: 实现首个 Deepfake 检测 + 生成器归因统一框架

**Step 3.1: Phase 1 — 无监督 GMM 聚类**

```python
# 复用: SDAIE-main/oc_funs.py:127-150 的 train_GMM_sklearn()
# 输入: 收集各类生成器图像的 x_T 特征（F1特征向量）
# 训练: 不使用任何生成器标签
# 验证: 聚类纯度（Purity）/ 调整互信息（AMI）

# 新增实验: 用 t-SNE 可视化 GMM 发现的簇 vs 真实生成器标签
# 如果 Purity > 0.7 → S3 Phase1 成立
```

**Step 3.2: Phase 2 — 少样本标签对齐**

```python
# 每个生成器类只用 200 张带标注图像
# 在 GMM 特征空间上训练轻量线性分类器
# 测试: 留出 5 个未见过的生成器（zero-shot 归因能力验证）
```

**Step 3.3: 归因结果对比**

| 方法 | 支持生成器归因 | 归因 ACC | 数据需求 |
|------|-------------|---------|---------|
| DIRE | ✗ | — | — |
| SDAIE | ✗ | — | — |
| DFFreq | ✗ | — | — |
| **DRIFT-S3** | **✓ (首个)** | ? | 仅 200张/类 |

---

### Phase 4 — 完整系统 + 全量消融（3~4 周）

**Step 4.1: F4 融合（Transformer 轨迹编码）**

```python
# 用轻量 Transformer (2层, 4头) 融合 {x_{t_k}, ε_θ(x_{t_k})} 序列
# 与 F1、F1+F2 做消融对比
# 目标: F4 在 AUC 上 vs F1+F2 提升 ≥ 1%（否则不加，保持简洁）
```

**Step 4.2: 消融实验矩阵**

| 消融维度 | 实验设置 |
|---------|---------|
| 采样步数 K | K = 5, 10, 15, 20 |
| 时间步策略 | 均匀 vs 前段密集（t=0~100） |
| 特征方案 | F1 / F1+F2 / F1+F2+F3 / F4 |
| 归因策略 | S1纯无监督 / S2有监督 / S3两阶段 |
| 聚类数量 | GMM K = 5, 10, 15, 20 |

**Step 4.3: 全量基线对比**

| 基线 | 二分类 AUC | 跨生成器 AUC | 归因 ACC | 推理时间 |
|------|-----------|------------|---------|---------|
| CNNSpot | baseline | baseline | ✗ | 快 |
| DIRE | better | better | ✗ | 极慢 |
| UnivFD | better | best | ✗ | 快 |
| SDAIE | similar | best | ✗ | 中 |
| DFFreq | better(freq) | similar | ✗ | 中 |
| Gram-Net | baseline | baseline | ✗ | 快 |
| **DRIFT** | **目标SOTA** | **目标SOTA** | **首个** | **中** |

---

### Phase 5 — 理论框架 + 论文写作（并行，4~5 周）

**Step 5.1: 理论推导（与 Phase 2-4 并行）**

```
命题1（轨迹偏离定理）:
  对 q_fake ⊥ q_real，DDIM 逆向的 score 预测误差随 t 单调累积
  证明工具: score function 的 Lipschitz 常数 + 总变差距离

命题2（端点指纹性）:
  x_T^k 的频域特征满足: W₂(P_k, P_j) > δ_{kj}
  证明工具: VAE 解码器的频率响应函数 + 信息守恒论证

注: 实验结果优先，理论放附录（CVPR/ICCV 风格）
```

**Step 5.2: 论文写作结构**

```
Abstract (200词)
  → 问题: deepfake 检测 + 归因
  → 方法: DDIM 单向逆向 + 流形指纹
  → 结果: SOTA 检测 + 首个归因框架

Introduction
  → Motivation: DIRE 的 3 个弱点（代码确认）
  → Insight: 逆向终点即指纹
  → Contributions: C1(单向逆向) C2(端点指纹) C3(统一框架) C4(效率)

Related Work
  → Deepfake Detection (DIRE, CNNSpot, SDAIE, DFFreq)
  → Diffusion Model Inversion (DDIM, DPM-Solver)
  → Generator Attribution (forensics 方向)

Method
  → 3.1 DDIM 逆向基础（cite DDIM 原文）
  → 3.2 DRIFT 特征提取（F1+F2，F4 可选）
  → 3.3 S3 生成器归因

Experiments
  → 4.1 Step A 验证（x_T 分布差异，可视化）← 说服审稿人的关键
  → 4.2 二分类检测对比
  → 4.3 跨生成器泛化
  → 4.4 生成器归因结果（新任务，无对手）
  → 4.5 效率对比（DIRE vs DRIFT ms/img）
  → 4.6 消融研究

Conclusion
Appendix（理论推导）
```

---

### 完整时间线（参考）

```
Week 1-2   Phase 0: Step A 假设验证（当前）
Week 3-5   Phase 1: DRIFT-F1 最小原型 + 初步对比
Week 6-8   Phase 2: 特征增强 + 跨生成器测试
Week 9-12  Phase 3: S3 归因系统
Week 13-16 Phase 4: 全量实验 + 消融
Week 5-16  Phase 5: 理论 + 写作（并行）
──────────────────────────────────────────────
CVPR deadline 通常在 11 月中旬，按此倒推 16 周即为启动时间点
```

---

### 关键验证门汇总（Layer 4 清单对应）

| 阶段 | 验证门 | 未通过的对策 |
|------|--------|------------|
| Phase 0 | x_T 统计差异 p<0.05（3/4项） | 改用轨迹特征为主，x_T 为辅 |
| Phase 1 | DRIFT-F1 AUC ≥ DIRE，效率 ≥ 2.5x | 增加 K 步数或换特征方案 |
| Phase 2 | 跨生成器 AUC > SDAIE reported | 加入 F3 Score 一致性特征 |
| Phase 3 | GMM 聚类纯度 > 0.7 | 换 Normalizing Flow 聚类 |
| Phase 4 | F4 比 F1+F2 提升 ≥ 1% | 保持 F1+F2，不加 Transformer |
