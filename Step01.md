审计结论摘要

  AIGCDetectBenchmark（9种方法集成基准）

  - 范式：全部有监督二分类（BCELoss + ResNet-50/CLIP）
  - 覆盖方法：空域（CNNSpot/Gram-Net/PatchCraft）、频域（FreDect-DCT）、预处理特征（LGrad梯度/LNP去噪/DIRE扩散逆向）、迁移（UnivFD-CLIP冻结骨干）
  - 最强泛化：UnivFD 冻结 CLIP 骨干，仅微调 FC 层 → zero-shot 能力最好
  - 代码确认弱点：DIRE 每张图像需 40次 UNet 前向（极慢）；无对抗鲁棒性测试；无域适应

  DFFreq（IEEE TIFS 2026）

  - 范式：有监督二分类
  - 核心创新：FreqLC.py:134-179，DWT(60%) + FFT相位注意力(40%)，置于 ResNet 之前
  - 代码确认问题：频域权重 0.6/0.4 硬编码；ResNet 只用前两层；pretrained=False 无预训练初始化
  - 迁移性：代码层面未见 zero-shot 或 domain adaptation 设计，跨生成器泛化未量化

  SDAIE（IEEE TPAMI 2025）

  - 范式：三阶段——①自监督（EXIF 14任务）→ ②无监督（GMM异常检测）→ ③有监督微调（冻结预训练+蒸馏）
  - 核心创新：首创用 EXIF 相机元数据作为自监督信号；HPF(30个SRM) → MPNCOV协方差池化 → Transformer 聚合
  - 代码确认弱点：GMM 5个组件硬编码；补丁聚合无显式策略；while not success 循环有无限循环风险

  ---
  计划：生成 Step01.md

  文件结构：
  1. 三个 codebase 的逐一解析（含代码路径引用）
  2. 方法维度对比矩阵（学习范式/骨干/频域建模/迁移性/效率）
  3. 可复用技术原子清单（6个核心模块，含 file:line）
  4. 研究空白清单（5个，直接从代码弱点提炼，非推测，为 Step02 idea 构建服务）

⏺ User approved Claude's plan
  ⎿  Plan saved to: ~/.claude/plans/optimized-gliding-squirrel.md · /plan to edit
     Step01.md 生成计划

     Context（背景与目标）

     用户的研究方向转向深度伪造（Deepfake/AIGC）检测。项目目录中有三个已发表高水平论文的代码库：
     - AIGCDetectBenchmark-main（集成9种检测方法的综合基准）
     - DFFreq-main-main（发表于 IEEE TIFS 2026 的双频域检测框架）
     - SDAIE-main（发表于 IEEE TPAMI 2025 的自监督EXIF元数据检测框架）

     目标：通过深度代码审计，生成 Step01.md 文件，为 Step02.md 中构建新 idea 提供充分的技术基础。

     ---
     最终输出文件

     路径: /Users/joces/Downloads/Coding_Campus/Deepfake_Project/Step01.md

     ---
     Step01.md 文档结构设计

     第一部分：三个 Codebase 的详细解析矩阵

     按照研究方法论文档的"比较框架设计"（Layer 3），从以下维度逐一拆解：

     1. AIGCDetectBenchmark（综合基准 · 9种方法集成）

     拆解维度：
     - 检测策略分类：空域（CNNSpot/Gram-Net/PatchCraft）、频域（FreDect）、预处理特征（LGrad/LNP/DIRE）、迁移特征（UnivFD）
     - 监督范式：全部为有监督二分类，BCEWithLogitsLoss
     - 骨干多样性：ResNet-50（主流）、CLIP/ViT（UnivFD）
     - 迁移性分析：16个数据集跨生成器评估；UnivFD 冻结骨干的 zero-shot 能力最强
     - 数据处理：统一的 augment（JPEG/blur），方法特定的预处理（DCT/梯度/去噪/DDIM逆向）
     - 代码审计确认的弱点：无对抗测试、DIRE 效率极低、无域适应

     2. DFFreq（双频域框架 · IEEE TIFS 2026）

     拆解维度：
     - 核心机制：FreqLCBlock = DWT(60%) + FFT相位注意力(40%)，置于 ResNet 之前作为前置预处理
     - 关键创新：保留幅度谱，用 LocalAttention（8×8窗口，4头）学习相位谱
     - 监督范式：有监督二分类，BCEWithLogitsLoss
     - 骨干：修改版 ResNet-50（只用 layer1/layer2，512维特征）
     - 频域处理细节：DWT Haar小波1级分解；FFT rfft2 + 相位自注意力 + irfft2
     - 迁移性：JPEG/blur 增强；代码层面未见 zero-shot 或 domain adaptation 设计
     - 弱点：权重 0.6/0.4 硬编码；ResNet 仅用前两层；无预训练初始化（pretrained=False）

     3. SDAIE（自监督 EXIF 元数据框架 · IEEE TPAMI 2025）

     拆解维度：
     - 核心机制：三阶段流水线
       - Stage1: 自监督预训练（EXIF 14任务多任务学习）
       - Stage2: 一类检测（GMM on 真实特征）
       - Stage3: 二元分类（冻结预训练 + 随机可训练 + 蒸馏正则化 α=0.05）
     - 监督范式：Stage1自监督（EXIF伪标签）+ Stage2无监督（GMM）+ Stage3有监督
     - 骨干：HPF(30个SRM滤波器) → 多尺度卷积 → MPNCOV协方差池化 → Transformer(2层,4头) → 528维
     - 迁移性：仅在真实图像上预训练，17个生成器跨域评估；通过特征蒸馏保持表示对齐
     - 弱点：补丁聚合无显式策略；GMM组件数硬编码；无显式 domain adaptation

     ---
     第二部分：方法维度对比矩阵

     按照方法论文档的"攻击覆盖矩阵"思路，设计 5×N 对比矩阵：

     ┌──────────────┬──────────────────────┬───────────────────┬─────────────────────┬─────────────────┬───────────────────┐
     │     维度     │ AIGCDetect (CNNSpot) │ AIGCDetect (DIRE) │ AIGCDetect (UnivFD) │     DFFreq      │       SDAIE       │
     ├──────────────┼──────────────────────┼───────────────────┼─────────────────────┼─────────────────┼───────────────────┤
     │ 学习范式     │ 有监督               │ 有监督            │ 有监督+zero-shot    │ 有监督          │ 自监督→有监督     │
     ├──────────────┼──────────────────────┼───────────────────┼─────────────────────┼─────────────────┼───────────────────┤
     │ 骨干         │ ResNet-50            │ ResNet-50         │ CLIP/ViT            │ ResNet-50(修改) │ HPF+MPNCOV+ViT    │
     ├──────────────┼──────────────────────┼───────────────────┼─────────────────────┼─────────────────┼───────────────────┤
     │ 特征空间     │ 空域卷积             │ DDIM差异          │ 语义特征            │ 双频域          │ 高频+协方差       │
     ├──────────────┼──────────────────────┼───────────────────┼─────────────────────┼─────────────────┼───────────────────┤
     │ 跨生成器迁移 │ 弱                   │ 中                │ 强                  │ 未知            │ 强（17生成器）    │
     ├──────────────┼──────────────────────┼───────────────────┼─────────────────────┼─────────────────┼───────────────────┤
     │ 训练数据需求 │ 真+假标注            │ 真+假标注         │ 真+假标注（少）     │ 真+假标注       │ 仅真实图像→再微调 │
     ├──────────────┼──────────────────────┼───────────────────┼─────────────────────┼─────────────────┼───────────────────┤
     │ 计算效率     │ 高                   │ 极低              │ 高                  │ 中              │ 中                │
     ├──────────────┼──────────────────────┼───────────────────┼─────────────────────┼─────────────────┼───────────────────┤
     │ 频域建模     │ 否                   │ 否                │ 否                  │ 是（核心）      │ 间接（HPF）       │
     └──────────────┴──────────────────────┴───────────────────┴─────────────────────┴─────────────────┴───────────────────┘

     ---
     第三部分：关键技术模块深度拆解

     精准记录后续 idea 构建所需的"可复用技术原子"：

     1. 频域处理模块（DFFreq）
       - 实现路径：DFFreq-main-main/networks/FreqLC.py:89-179
       - API：FreqLCBlock(in_channels, window_size) → forward(x) → x_idwt*0.6 + x_ifft*0.4
       - 可改进点：权重可学习化；DWT多级；窗口大小自适应
     2. 自监督 EXIF 预训练（SDAIE）
       - 实现路径：SDAIE-main/backbone_train.py:39-92，utils.py:104-220
       - 核心：14任务多任务头，EXIF属性差异用 Error 函数转概率
       - 可改进点：利用更多EXIF属性；与其他自监督结合（SimCLR/MAE）
     3. MPNCOV 二阶特征池化（SDAIE）
       - 实现路径：SDAIE-main/Classifier/MPNCOV.py:14-143
       - 输出：528维上三角协方差向量，比 GAP 捕捉更多统计信息
       - 可改进点：替换为 Gram-Net 的 Gram 矩阵对比
     4. 特征蒸馏正则化（SDAIE）
       - 实现路径：SDAIE-main/bc_train.py:52-107
       - 设计：冻结预训练网络 + 可训练随机网络 + MSE蒸馏 α=0.05
       - 可改进点：对比蒸馏（contrastive KD）；动态 α
     5. 多尺度 Patch 融合 + NMS 选择（AIGCDetectBenchmark PatchCraft）
       - 实现路径：AIGCDetectBenchmark-main/networks/Patch5Model.py:43-165
       - 设计：NMS在7×7特征图选Patch → 映射回原图 → 自注意力聚合（3层SA_layer）
       - 可改进点：Patch 选择加入频域信息
     6. 一类 GMM 检测（SDAIE）
       - 实现路径：SDAIE-main/oc_funs.py:127-179
       - 设计：5组件GMM拟合真实特征，2%分位数阈值
       - 可改进点：替换 GMM 为 Normalizing Flow；自适应阈值

     ---
     第四部分：研究空白与潜在创新方向（为 Step02 做铺垫）

     从代码审计发现的共同弱点中提炼研究空白：

     1. 频域 + 语义的统一建模
       - DFFreq 做频域但无语义；UnivFD 做语义但无频域
       - 空白：二者的协同建模
     2. 迁移性机制不明确
       - SDAIE 的迁移性依赖隐式 EXIF 学习，无理论保证
       - DFFreq 的泛化性未量化验证
       - 空白：明确的 domain-invariant 特征提取
     3. 无监督/自监督方向几乎空白
       - AIGCDetectBenchmark 全部有监督；DFFreq 有监督；SDAIE 自监督但在Stage3依然微调
       - 空白：纯自监督或无监督的检测框架
     4. 攻击覆盖不完整
       - 无一方法做了对抗攻击鲁棒性测试
       - 空白：面向对抗攻击的鲁棒检测
     5. 效率-精度 Pareto 边界
       - DIRE 精度高但极慢；CNNSpot 快但泛化差
       - 空白：轻量级但强泛化的检测框架

     ---
     实施步骤

     1. 依据上述结构，生成 Step01.md 文件，保存到 /Users/joces/Downloads/Coding_Campus/Deepfake_Project/Step01.md
     2. 文件组织：
       - 每个 Codebase 有独立的二级标题
       - 包含代码路径引用（file:line）
       - 包含对比矩阵
       - 包含"研究空白"分析
       - 不超过 1000 行（确保可读性）
     3. Step01.md 完成后，用户可在 Step02.md 中基于这些信息进行 idea 构建对话

     ---
     验证方式

     - 确认 Step01.md 文件存在于项目根目录
     - 每个技术点均有具体 file:line 引用（代码审计已确认）
     - 研究空白与代码弱点有直接对应关系（非推测）