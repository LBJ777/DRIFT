# DRIFT: Theoretical Framework

**DRIFT — Diffusion Reverse Inversion Fingerprint Tracking**

*Mathematical foundations for deepfake detection and generator attribution via DDIM inversion trajectories.*

---

## 1. Theoretical Premises and Notation

### 1.1 Distributions and Models

| Symbol | Definition |
|--------|-----------|
| $p_\theta$ | A pretrained ADM (Ablated Diffusion Model) trained on the real-image distribution $q_\text{real}$ |
| $q_\text{real}$ | The real-image distribution (e.g., ImageNet or FFHQ) |
| $q_k^\text{fake}$ | The image distribution produced by the $k$-th generative model, $k = 1, \ldots, K$ |
| $\bar{\alpha}_t$ | Cumulative noise schedule: $\bar{\alpha}_t = \prod_{s=1}^{t}(1 - \beta_s)$, with $\bar{\alpha}_0 = 1$, $\bar{\alpha}_T \approx 0$ |
| $\varepsilon_\theta(x_t, t)$ | The noise prediction network (UNet) of $p_\theta$ |
| $\hat{x}_0(x_t, t)$ | The one-step denoised estimate: $\hat{x}_0 = \bigl(x_t - \sqrt{1 - \bar{\alpha}_t}\,\varepsilon_\theta(x_t, t)\bigr) / \sqrt{\bar{\alpha}_t}$ |

### 1.2 The DDIM Forward (Noising) Process

The marginal distribution of the forward process at time $t$ is:

$$q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\; \sqrt{\bar{\alpha}_t}\,x_0,\; (1 - \bar{\alpha}_t)\,I\right)$$

### 1.3 The DDIM Inversion (Reverse) Step

Given a deterministic DDIM scheduler, the **inversion** step from $x_t$ to $x_{t+1}$ (i.e., stepping *toward* higher noise levels) is:

$$x_{t+1} = \sqrt{\bar{\alpha}_{t+1}}\,\hat{x}_0(x_t, t) + \sqrt{1 - \bar{\alpha}_{t+1}}\;\varepsilon_\theta(x_t, t) \tag{1}$$

This corresponds to integrating the **Probability Flow ODE** in the reverse-time direction:

$$dx = \left[-\tfrac{1}{2}\beta(t)\,x - \tfrac{1}{2}\beta(t)\,\nabla_x \log p_\theta(x, t)\right]dt \tag{2}$$

with $\nabla_x \log p_\theta(x_t, t) \approx -\varepsilon_\theta(x_t, t)/\sqrt{1 - \bar{\alpha}_t}$ (the score function approximation).

### 1.4 The DRIFT Inversion Map and Trajectory

**Definition 1 (Inversion Map).** For a given image $x_0 \in \mathbb{R}^d$, the *DDIM inversion map* $\Phi_T : \mathbb{R}^d \to \mathbb{R}^d$ is defined by iterating equation (1) from $t = 0$ to $t = T$:

$$\Phi_T(x_0) := x_T$$

where $x_T$ is the result of applying $T$ deterministic inversion steps starting from $x_0$.

**Definition 2 (Inversion Trajectory).** Let $0 = t_0 < t_1 < \cdots < t_N = T$ be a sequence of selected timesteps. The *inversion trajectory* of $x_0$ is:

$$\tau(x_0) := \{x_{t_1}, x_{t_2}, \ldots, x_{t_N}, x_T\}$$

where each $x_{t_k}$ is the intermediate state produced by iterating equation (1) up to step $t_k$.

### 1.5 The Data Manifold Assumption

**Assumption 1 (Support Alignment).** Because $p_\theta$ is trained on $q_\text{real}$, the learned score function $\varepsilon_\theta$ is a good approximation of the true score $\nabla_x \log q_t^\text{real}$ for images drawn from $q_\text{real}$. Formally:

$$\mathbb{E}_{x_0 \sim q_\text{real}}\left[\left\|\varepsilon_\theta(x_t, t) + \sqrt{1 - \bar{\alpha}_t}\,\nabla_{x_t}\log q_t^\text{real}(x_t)\right\|_2^2\right] \leq \delta_\text{train}$$

where $\delta_\text{train}$ is the diffusion model training loss, typically small. This assumption states that the model's score estimate is reliable on in-distribution inputs.

**Assumption 2 (OOD Gap).** For any fake generator $k$, the distribution $q_k^\text{fake}$ is not identical to $q_\text{real}$. Specifically, we assume a non-trivial Total Variation distance:

$$\text{TV}(q_k^\text{fake},\; q_\text{real}) > 0$$

This is a mild assumption: any imperfect generative model satisfies it.

---

## 2. Proposition 1: Trajectory Deviation Theorem

### 2.1 Intuition

When $x_0 \sim q_\text{real}$, the DDIM inversion follows a deterministic ODE path that is consistent with $p_\theta$. The score function $\varepsilon_\theta(x_t, t)$ predicts accurately along this path, so the ODE solution is smooth and well-behaved.

When $x_0 \sim q_k^\text{fake}$, the image lies in a distribution that differs from $q_\text{real}$. The score function $\varepsilon_\theta$ was never trained on this distribution, so it produces systematic errors. These errors accumulate across the $T$ inversion steps, causing the trajectory $\tau(x_0)$ to deviate from the paths taken by real images.

### 2.2 Formal Statement

**Proposition 1** *(Trajectory Deviation Theorem).*

Let $\varepsilon_\theta^*(x_t, t) := -\sqrt{1 - \bar{\alpha}_t}\,\nabla_{x_t}\log q_t(x_t)$ denote the *true* score (under the image distribution $q$ from which $x_0$ was drawn). Define the per-step score prediction error as:

$$\delta_t(x_0) := \varepsilon_\theta(x_t, t) - \varepsilon_\theta^*(x_t, t)$$

Let $x_t^\text{real}$ and $x_t^\text{fake}$ denote the inversion states at step $t$ for $x_0^\text{real} \sim q_\text{real}$ and $x_0^\text{fake} \sim q_k^\text{fake}$, respectively. Under Assumptions 1 and 2, the expected squared trajectory deviation satisfies:

$$\mathbb{E}\!\left[\left\|x_t^\text{fake} - x_t^\text{real}\right\|_2^2\right] \;\geq\; \Omega\!\left(\text{TV}(q_k^\text{fake}, q_\text{real})^2 \cdot t\right) \quad \text{as } t \to T \tag{3}$$

Furthermore, the expected per-step score error for fake images admits a lower bound:

$$\mathbb{E}_{x_0 \sim q_k^\text{fake}}\!\left[\|\delta_t(x_0)\|_2^2\right] \;\geq\; c_t \cdot \text{TV}(q_k^\text{fake}, q_\text{real})^2 \tag{4}$$

for some constant $c_t > 0$ that depends on the smoothness of the score function.

### 2.3 Proof Sketch

*Proof sketch.*

**Step 1: Score error lower bound via TV distance.**

By the data-processing inequality and properties of the TV distance, for any measurable function $f$ and any two distributions $P$, $Q$:

$$\|f\|_{L^1(P)} - \|f\|_{L^1(Q)} \leq \|f\|_{L^\infty} \cdot \text{TV}(P, Q)$$

The score functions $\nabla_x \log q_t^\text{real}$ and $\nabla_x \log q_t^\text{fake}$ are related through the convolution structure of the forward process. Because $q_t(x_t) = \int q(x_t \mid x_0)\,q(x_0)\,dx_0$, a difference in $q(x_0)$ propagates to a difference in $q_t(x_t)$. Under mild regularity (e.g., bounded second moments and Lipschitz score functions), one can show:

$$\left\|\nabla_{x_t}\log q_t^\text{fake}(x_t) - \nabla_{x_t}\log q_t^\text{real}(x_t)\right\|_2 = O\!\left(\text{TV}(q_k^\text{fake}, q_\text{real})\right)$$

The model $\varepsilon_\theta$ approximates the real score by Assumption 1, so the systematic error on fake inputs is:

$$\|\delta_t(x_0^\text{fake})\|_2 = \left\|\varepsilon_\theta(x_t, t) - \varepsilon_\theta^{*,\text{fake}}(x_t, t)\right\|_2 = O\!\left(\text{TV}(q_k^\text{fake}, q_\text{real})\right)$$

This establishes inequality (4). $\square_\text{step1}$

**Step 2: Error accumulation across timesteps.**

The DDIM update (equation 1) is a deterministic linear map in $\hat{x}_0$ and $\varepsilon_\theta$. The deviation $\Delta_t = x_t^\text{fake} - x_t^\text{real}$ evolves according to:

$$\Delta_{t+1} = \sqrt{\bar{\alpha}_{t+1}}\,\Delta_t^{\hat{x}_0} + \sqrt{1 - \bar{\alpha}_{t+1}}\,\Delta_t^\varepsilon$$

where $\Delta_t^{\hat{x}_0}$ and $\Delta_t^\varepsilon$ are the deviations in the $\hat{x}_0$ and score estimates. Because the per-step error has a systematic bias of order $\text{TV}(q_k^\text{fake}, q_\text{real})$ (from Step 1), and successive errors are not perfectly canceling (they share a common source: the OOD nature of $x_0^\text{fake}$), the squared deviation grows at least linearly in $t$:

$$\mathbb{E}\!\left[\|\Delta_t\|_2^2\right] \geq c \cdot t \cdot \text{TV}(q_k^\text{fake}, q_\text{real})^2$$

This establishes inequality (3). $\square$

**Remark on strictness.** The above is a proof sketch. Step 1 uses an informal application of the TV-score relationship; making it rigorous requires additional regularity conditions on $q_k^\text{fake}$ (e.g., absolute continuity with respect to $q_\text{real}$ in the diffused space, or bounded density ratios). Step 2 assumes the per-step errors do not cancel; this is plausible but not proven for all pairs $(q_k^\text{fake}, q_\text{real})$.

### 2.4 Corollary: Deviation Accumulates Over Time

**Corollary 1.** Under the conditions of Proposition 1, the trajectory deviation is monotonically increasing in expectation:

$$\mathbb{E}\!\left[\|\Delta_{t+1}\|_2^2\right] \geq \mathbb{E}\!\left[\|\Delta_t\|_2^2\right] + c \cdot \text{TV}(q_k^\text{fake}, q_\text{real})^2$$

This means that analyzing $x_T$ (the endpoint) captures the maximum accumulated deviation, while intermediate trajectory statistics $\{x_{t_k}\}$ capture the time-resolved deviation profile.

### 2.5 Measurable Quantities and Experimental Correspondence

The following statistics correspond directly to Proposition 1 in experiments:

| Theoretical Quantity | Experimental Measurement |
|---------------------|--------------------------|
| $\|\Delta_T\|_2^2 = \|x_T^\text{fake} - \mathbb{E}[x_T^\text{real}]\|_2^2$ | MSE of $x_T$ from a reference Gaussian $\mathcal{N}(0, I)$ |
| $\|\delta_t(x_0)\|_2$ at each timestep | Norm of $\varepsilon_\theta(x_t, t)$ during inversion |
| Trajectory deviation profile $\{\|\Delta_{t_k}\|_2\}_{k=1}^N$ | Intermediate state norms along the inversion path |
| $\text{TV}(q_k^\text{fake}, q_\text{real})$ | Proxy: FID score of generator $k$ on the training domain |

> **中文注释 — 实验验证方式**
>
> 命题 1 的核心预测是：假图在 DDIM 逆向过程中积累的轨迹偏差与生成器的 OOD 程度正相关。
>
> 验证方法：
> 1. **轨迹范数曲线**：对每张输入图像，记录各时间步 $\|x_{t_k}\|_2$ 和 $\|\varepsilon_\theta(x_{t_k}, t_k)\|_2$，比较真实图像与不同生成器假图的均值曲线。预测：假图的曲线应系统性偏高，且偏差随 $t$ 单调增加。
> 2. **端点统计检验**：对真实图像，$x_T$ 应近似服从 $\mathcal{N}(0, I)$；对假图，$x_T$ 的均值或方差应显著偏离。可用 KL 散度或 Wasserstein 距离量化。
> 3. **与 FID 的相关性**：计算各生成器的 FID 分数，检验其与轨迹偏差量 $\mathbb{E}[\|\Delta_T\|_2^2]$ 的 Pearson 相关系数，预测应显著正相关（$r > 0.7$）。

---

## 3. Proposition 2: Endpoint Fingerprint Separability

### 3.1 Intuition

Different generative models introduce different systematic artifacts in their outputs. These artifacts arise from specific architectural choices: VAE compression in latent diffusion models (e.g., Stable Diffusion), spectral biases in GAN discriminators, and upsampling patterns in various decoders.

When an image $x_0$ containing such artifacts is subjected to DDIM inversion, the artifacts are not removed — rather, they are re-encoded in the noise space as structured deviations from a pure Gaussian. Crucially, different generators produce artifacts at different spatial frequencies, leading to distinguishable fingerprints in the frequency spectrum of $x_T$.

### 3.2 Formal Statement

**Proposition 2** *(Endpoint Fingerprint Separability).*

For $k, l \in \{1, \ldots, K\} \cup \{\text{real}\}$ with $k \neq l$, let $\mu_k := \Phi_T\#q_k$ denote the pushforward distribution of $q_k$ under the DDIM inversion map $\Phi_T$. Then:

$$W_2(\mu_k,\; \mu_l) > 0 \tag{5}$$

where $W_2$ denotes the 2-Wasserstein distance.

Moreover, in the frequency domain, the power spectral densities of $\mu_k$ and $\mu_l$ exhibit generator-specific peaks. Define the power spectral density of $x_T$ as $S_k(f) := \mathbb{E}_{x_T \sim \mu_k}[|\hat{x}_T(f)|^2]$ where $\hat{x}_T$ is the 2D DFT of $x_T$. Then there exist frequency bands $\mathcal{F}_k \subset [0, f_s/2]^2$ such that:

$$S_k(f) \gg S_l(f) \quad \forall f \in \mathcal{F}_k,\; l \neq k \tag{6}$$

### 3.3 Mathematical Modeling of SD-VAE Frequency Artifacts

Stable Diffusion uses a Variational Autoencoder with an $8\times$ spatial downsampling factor. This is the primary source of a characteristic periodic artifact.

**Setup.** Let $x \in \mathbb{R}^{H \times W \times 3}$ be the input image. The VAE encoder compresses it to a latent code $z = \text{Enc}(x) \in \mathbb{R}^{(H/8) \times (W/8) \times 4}$. Diffusion is performed in this latent space. At decode time:

$$\hat{x} = \text{Dec}(z)$$

**Frequency Response of the VAE Decoder.** The VAE decoder performs learned upsampling (typically via nearest-neighbor or bilinear upsampling followed by convolution). Denote the decoder's effective spatial frequency response as $H(f)$. Due to the $8\times$ upsampling, the decoder inherently creates periodic replication artifacts at multiples of the sub-sampling frequency $f_s/8$. In 2D:

$$H(f_x, f_y) \text{ has local maxima at } f_x, f_y \in \left\{0,\; \frac{f_s}{8},\; \frac{2f_s}{8},\; \ldots\right\} \tag{7}$$

where $f_s$ is the pixel-domain sampling frequency (= 1 cycle/pixel in normalized units).

**The Artifact Signal.** The reconstructed image can be written as:

$$\hat{x} = x + \eta_\text{VAE}$$

where $\eta_\text{VAE}$ is the VAE reconstruction error (the artifact term). By the frequency analysis above, $\eta_\text{VAE}$ is not white noise — it has elevated power at spatial periods of 8 pixels (and harmonics):

$$\mathbb{E}\!\left[|\hat{\eta}_\text{VAE}(f)|^2\right] \text{ is elevated at } |f| = \frac{1}{8} \text{ cycles/pixel} \tag{8}$$

In practice, the artifact period is approximately **8 pixels** in each spatial dimension, which is directly visible as a grid-like pattern in the Fourier spectrum of SD-generated images.

**Propagation Through DDIM Inversion.** Now consider how $\eta_\text{VAE}$ appears in $x_T = \Phi_T(x_0)$.

Let $x_0 = x_0^\text{clean} + \eta_\text{VAE}$, where $x_0^\text{clean}$ is a hypothetical artifact-free image. The DDIM inversion of $x_0$ is:

$$x_T = \Phi_T(x_0^\text{clean} + \eta_\text{VAE})$$

Since the DDIM inversion map $\Phi_T$ is approximately linear in the low-noise regime (small perturbations around a given trajectory), we can write:

$$\Phi_T(x_0^\text{clean} + \eta_\text{VAE}) \approx \Phi_T(x_0^\text{clean}) + J_{\Phi_T}(x_0^\text{clean})\,\eta_\text{VAE} + O(\|\eta_\text{VAE}\|^2) \tag{9}$$

where $J_{\Phi_T}$ is the Jacobian of the inversion map. The key insight is that $J_{\Phi_T}$ is not a frequency-scrambling operator — it approximately preserves the frequency content of its input perturbation (since each DDIM step is a smooth function of $x_t$, dominated by linear terms in the low-noise regime).

Therefore, the frequency artifact $\eta_\text{VAE}$ with its 8px period is **approximately preserved** in $x_T$:

$$S_\text{SD}(f) = \mathbb{E}\!\left[|\hat{x}_T^\text{SD}(f)|^2\right] \gg \mathbb{E}\!\left[|\hat{x}_T^\text{real}(f)|^2\right] \quad \text{at } |f| = \frac{1}{8} \text{ cycles/pixel} \tag{10}$$

**Remark on linearization.** Equation (9) is a first-order approximation. Its validity depends on $\|\eta_\text{VAE}\|$ being small relative to the curvature of $\Phi_T$. In practice, VAE artifacts are subtle (PSNR $> 30$ dB), so this approximation is reasonable. A non-linear analysis would require bounding higher-order terms in the Taylor expansion of the ODE solution map, which we leave to future work.

### 3.4 Mathematical Modeling of GAN Frequency Artifacts

GANs produce characteristic **checkerboard artifacts** arising from transposed convolution (deconvolution) layers with stride $s$. These artifacts appear at spatial frequencies:

$$f_\text{GAN} = \frac{f_s}{s} \quad \text{(and harmonics)} \tag{11}$$

For a GAN with final stride-2 transposed convolution, the dominant artifact frequency is $f_s/2$ (the Nyquist frequency), manifesting as a high-frequency checkerboard pattern. For stride-4 upsampling, the frequency is $f_s/4$ (4px period).

By an analogous argument to the SD-VAE case, these GAN artifacts in $x_0$ are approximately preserved in $x_T$ after DDIM inversion, contributing to a distinct peak at $f_\text{GAN}$ in $S_\text{GAN}(f)$.

The fingerprint separation condition (6) then holds because:

$$\mathcal{F}_\text{SD} = \left\{f : |f| \approx \frac{1}{8}\right\}, \quad \mathcal{F}_\text{GAN} = \left\{f : |f| \approx \frac{1}{s}\right\}$$

and these frequency bands are generically distinct for SD ($s=8$) and typical GAN architectures ($s=2$ or $s=4$).

### 3.5 Proof Sketch of Proposition 2

*Proof sketch.*

**Existence of $W_2 > 0$ (equation 5).** We prove by contradiction. Suppose $W_2(\mu_k, \mu_l) = 0$. Then $\mu_k = \mu_l$ as measures. Since $\Phi_T$ is a diffeomorphism (the deterministic DDIM map is invertible under mild regularity conditions on the score function), its pushforward is injective: $\Phi_T\#q_k = \Phi_T\#q_l$ implies $q_k = q_l$. But we assumed $q_k \neq q_l$ (distinct generators), giving a contradiction.

**Remark.** The injectivity of $\Phi_T$ relies on the ODE solution map being a diffeomorphism, which holds when the score function is Lipschitz and the ODE has a unique solution. This is a standard assumption in the ODE theory of diffusion models (see Song et al., 2021).

**Frequency separation (equation 6).** This follows from the frequency artifact analysis in sections 3.3 and 3.4: different generators introduce artifact energy at distinct frequency bands, and DDIM inversion approximately preserves this frequency structure (equation 9). $\square$

> **中文注释 — 实验验证方式**
>
> 命题 2 的核心预测是：$x_T$ 的频谱具有生成器特异性峰值。
>
> 验证方法：
> 1. **频谱分析**：对大量真实图像和各生成器假图，分别计算 $x_T$ 的平均 2D 功率谱密度（PSD）。预测：SD 生成图像的 $x_T$ 的 PSD 在 $f = f_s/8$（即8像素周期）处有显著峰值；GAN 假图在对应于其步幅的频率处有峰值；真实图像则无明显人工峰值。
> 2. **Wasserstein 距离矩阵**：在频域特征（如对各频率分箱的能量直方图）上计算两两生成器之间的 $W_2$ 距离矩阵，应形成一个块对角结构（同类图像内距离小，跨类距离大）。
> 3. **归因实验**：以 $x_T$ 的频域特征向量（PSD 在各关键频率的值）作为分类器输入，验证其能区分不同生成器来源（预测准确率应显著高于基线随机分类）。
> 4. **消融实验**：对比使用 $x_T$ 整体特征 vs. 仅使用特定频率带特征的归因准确率，以验证频率峰值的判别价值。

---

## 4. Proposition 3 (Optional): The Unified Framework Theorem

### 4.1 Overview

Propositions 1 and 2 address detection and attribution separately. This section presents a unified framework that subsumes both tasks under a single geometric picture: *measuring the distance from $x_T$ to the model's latent noise manifold*.

### 4.2 Formal Statement

**Proposition 3** *(Unified Framework Theorem).*

Let $\mathcal{N}_\text{real} := \Phi_T\#q_\text{real}$ denote the "real noise manifold" (the distribution of $x_T$ when $x_0 \sim q_\text{real}$), and let $\mathcal{N}_k := \Phi_T\#q_k^\text{fake}$ denote the noise manifold for generator $k$.

**(a) Detection.** An image $x_0$ is classified as fake if and only if:

$$D_\text{detect}(x_0) := W_2\!\left(\delta_{\Phi_T(x_0)},\; \mathcal{N}(0, I)\right) > \tau_\text{detect}$$

where $\delta_{\Phi_T(x_0)}$ is the Dirac measure at $x_T = \Phi_T(x_0)$, $\mathcal{N}(0, I)$ is the standard Gaussian (the ideal inversion endpoint for a perfectly-trained model and a real image), and $\tau_\text{detect}$ is a threshold.

In practice, $D_\text{detect}(x_0)$ is approximated by scalar statistics of $x_T$: its deviation from zero mean, its per-channel variance, or its kurtosis.

**(b) Attribution.** Given that $x_0$ is detected as fake, it is attributed to generator $k^*$ via:

$$k^* = \arg\min_{k \in \{1, \ldots, K\}} D_\text{attr}\!\left(\Phi_T(x_0),\; \mathcal{N}_k\right) \tag{12}$$

where $D_\text{attr}$ is a distribution distance (e.g., estimated by a nearest-neighbor distance in an embedding space, or the Mahalanobis distance under the covariance of $\mathcal{N}_k$).

### 4.3 Geometric Interpretation

The space of inversion endpoints $\mathbb{R}^d$ is partitioned into regions corresponding to the noise manifolds of different generators:

$$\mathbb{R}^d \approx \mathcal{N}_\text{real} \cup \mathcal{N}_1 \cup \cdots \cup \mathcal{N}_K$$

(with overlaps possible but assumed to be small by Proposition 2). Detection asks: "Is $x_T$ near $\mathcal{N}_\text{real}$?" Attribution asks: "Which $\mathcal{N}_k$ is $x_T$ nearest to?"

This geometric view suggests that **any metric-based classifier** (e.g., $k$-NN, SVM, or a learned metric network) operating on $x_T$ embeddings can perform both tasks simultaneously.

### 4.4 Conditions for Proposition 3 to Hold

Proposition 3 relies on both Propositions 1 and 2. Specifically:

- Proposition 1 ensures that $\mathcal{N}_\text{real}$ and $\mathcal{N}_k$ are well-separated ($W_2 > 0$), enabling detection.
- Proposition 2 ensures that different $\mathcal{N}_k$ are mutually well-separated, enabling attribution.
- An additional condition is that the decision threshold $\tau_\text{detect}$ can be set to achieve acceptable false positive/negative rates, which depends on the overlap between $\mathcal{N}_\text{real}$ and $\mathcal{N}_k^\text{fake}$.

> **中文注释 — 实验验证方式**
>
> 命题 3 提供了一个统一的分类器设计原则。
>
> 验证方法：
> 1. **统一特征实验**：使用同一套从 $x_T$ 提取的特征（如均值、方差、峰度、PSD 关键频率能量），同时训练检测分类器和归因分类器，观察两个任务的准确率是否均优于各自的特化基线。
> 2. **聚类可视化**：对 $x_T$ 用 t-SNE 或 UMAP 降维，可视化不同来源图像对应的 $x_T$ 分布。预测：真实图像与各生成器假图形成可分离的聚类，支持命题 3 的几何图像。
> 3. **Mahalanobis 距离归因**：用训练集估计各 $\mathcal{N}_k$ 的均值和协方差，测试集用 Mahalanobis 距离做最近邻归因，报告 Top-1 和 Top-3 准确率。

---

## 5. Theoretical Relationship to DIRE

### 5.1 What DIRE Measures

DIRE (Diffusion Reconstruction Error; Wang et al., 2023) computes the **reconstruction error** of an image after a round-trip through the diffusion model:

$$\text{DIRE}(x_0) := \left\|x_0 - \text{Dec}\!\left(\text{Enc}(x_0)\right)\right\|_2 \tag{13}$$

where $\text{Enc}$ corresponds to DDIM inversion (forward, $x_0 \to x_T$) and $\text{Dec}$ corresponds to DDIM sampling (backward, $x_T \to \hat{x}_0$). Concretely:

$$\text{DIRE}(x_0) = \left\|x_0 - \hat{x}_0\right\|_2, \quad \hat{x}_0 = \Phi_T^{-1}(\Phi_T(x_0)) \tag{14}$$

### 5.2 What DRIFT Measures

DRIFT discards the backward pass entirely and measures statistics of the **inversion endpoint**:

$$\text{DRIFT}(x_0) := f\!\left(\Phi_T(x_0)\right) \tag{15}$$

where $f$ extracts scalar statistics from $x_T$ (e.g., deviation from $\mathcal{N}(0, I)$, frequency-domain features, or a learned classifier applied to $x_T$).

### 5.3 Information-Theoretic Comparison

**Theorem (Informal).** DRIFT preserves strictly more information about $x_0$ than DIRE.

*Argument.*

DIRE collapses the inversion map $\Phi_T$ (an injective function of $x_0$) to a single scalar (or low-dimensional vector) via the norm $\|x_0 - \Phi_T^{-1}(\Phi_T(x_0))\|_2$. This is a lossy operation: many different $x_T$ configurations can yield the same reconstruction error.

Formally, let $I(\cdot; \cdot)$ denote mutual information. DIRE computes a sufficient statistic only for the magnitude of the reconstruction error, so:

$$I\!\left(\text{DIRE}(x_0);\; x_0\right) \leq I\!\left(\Phi_T(x_0);\; x_0\right) \tag{16}$$

DRIFT, by retaining $x_T = \Phi_T(x_0)$ in full (or via rich feature extraction), approximately achieves the upper bound $I(\Phi_T(x_0); x_0)$, which equals $H(x_0)$ if $\Phi_T$ is injective (i.e., DRIFT is lossless in the information-theoretic sense).

**Practical implication.** DIRE cannot distinguish between two fake images that happen to have the same reconstruction error but different artifact structures. DRIFT can, because the frequency-domain content of $x_T$ retains the generator-specific fingerprints demonstrated in Proposition 2.

**Computational implication.** DRIFT requires only one forward pass through $\varepsilon_\theta$ (the DDIM inversion), whereas DIRE requires two passes (inversion + reconstruction). DRIFT is therefore approximately $2\times$ faster at inference time.

### 5.4 Summary Table

| Property | DIRE | DRIFT |
|----------|------|-------|
| Passes through $\varepsilon_\theta$ | 2 (inversion + reconstruction) | 1 (inversion only) |
| Output | Scalar reconstruction error $\|x_0 - \hat{x}_0\|_2$ | Full vector $x_T$ + trajectory $\tau(x_0)$ |
| Information retained | Low (scalar summary) | High (full inversion endpoint) |
| Task supported | Detection only | Detection + attribution |
| Generator-specific fingerprints | Partially (implicit in reconstruction error) | Explicit (frequency domain of $x_T$) |
| Computational cost | $2T$ NFEs | $T$ NFEs |

> NFE = Number of Function Evaluations (calls to $\varepsilon_\theta$).

---

## 6. Limitations and Honest Assessment

### 6.1 What Is Rigorously Proven vs. Conjectured

| Claim | Status |
|-------|--------|
| DDIM inversion map $\Phi_T$ is injective (diffeomorphism) | Known result from ODE theory of diffusion models; see Song et al. (2021) |
| $W_2(\mu_k, \mu_l) > 0$ for $k \neq l$ (equation 5) | Proof sketch (relies on injectivity; see section 3.5) |
| Score error lower bound in terms of TV (equation 4) | Informal argument; requires additional regularity conditions |
| Linear error accumulation in $t$ (equation 3) | Heuristic; full proof requires bounding correlation of per-step errors |
| Linearization of $\Phi_T$ around artifact perturbation (equation 9) | First-order approximation; higher-order terms not bounded |
| Frequency peak at $f_s/8$ for SD-VAE artifacts | Empirical observation + qualitative frequency analysis; not a formal theorem |
| DRIFT preserves more information than DIRE (equation 16) | Informal information-theoretic argument; not a formal proof |

### 6.2 Necessary Conditions and Assumptions

The theoretical guarantees in this document hold under the following conditions, which may not always be satisfied:

1. **Assumption 1 (Support Alignment)** requires that $p_\theta$ is well-trained on $q_\text{real}$. If the pretrained ADM has high training loss or significant mode dropping, the score function may be unreliable even on real images.

2. **Assumption 2 (OOD Gap)** requires that $\text{TV}(q_k^\text{fake}, q_\text{real}) > 0$. This fails for a hypothetically perfect generator. As generative models improve (particularly future diffusion-based generators), this gap may shrink, reducing DRIFT's detection power.

3. **Injectivity of $\Phi_T$** requires that the DDIM ODE has a unique solution, which holds under Lipschitz continuity of the score function. This may fail near singularities of the score or in degenerate cases.

4. **Linearization validity** (equation 9) assumes that VAE artifacts are small perturbations relative to the curvature of $\Phi_T$. For low-quality or highly compressed VAE outputs, the artifact may be large enough to invalidate the linear approximation.

5. **Frequency artifact persistence** (equation 10) assumes that DDIM inversion approximately preserves frequency structure. This is an empirical observation; theoretical support requires a more careful analysis of how the ODE solution map acts on frequency components.

### 6.3 Potential Counterexamples

**Counterexample to detection (Proposition 1):** A generator that exactly matches $q_\text{real}$ would produce $\text{TV} = 0$ and thus zero trajectory deviation. While no current generator achieves this, future models may approach this limit.

**Counterexample to attribution (Proposition 2):** Two different GAN architectures with the same stride (and thus the same checkerboard frequency) would produce overlapping $\mathcal{N}_k$ fingerprints in the frequency domain. Attribution between such generators would rely on subtler statistics beyond the dominant frequency peak.

**Robustness concern:** JPEG compression, resizing, or other post-processing operations applied to fake images before DDIM inversion could alter the artifact structure, potentially reducing the distinctiveness of frequency fingerprints.

**Model mismatch:** If the pretrained ADM $p_\theta$ is trained on a different domain than the test images (e.g., ADM trained on ImageNet is used to analyze FFHQ-domain images), the score function errors may be systematically biased even for real images in the test domain, increasing false positives.

### 6.4 An Alternative Analysis Direction

If the linearization approach for frequency artifact propagation (equation 9) proves too weak, an alternative analysis can be pursued:

**Empirical characterization approach.** Rather than proving frequency preservation analytically, one can directly measure $S_k(f) = \mathbb{E}[|\hat{x}_T(f)|^2]$ for each generator class empirically and verify that condition (6) holds. This converts Proposition 2 from a theoretical guarantee into an empirically-validated claim, which is still scientifically valid and practically useful.

---

## 7. Proofs Pending Completion

The following items in this document are proof sketches or informal arguments that require completion to achieve the status of formal theorems. These are candidates for rigorous treatment in the final paper or supplementary material:

1. **[Prop. 1, Step 1]** Formalize the relationship between TV distance and score function deviation. Required: regularity conditions on $q_k^\text{fake}$ (e.g., existence of density, Poincare inequality), and a bound of the form $\|\nabla \log q_t^A - \nabla \log q_t^B\|_{L^2} \leq C \cdot \text{TV}(q_0^A, q_0^B)$.

2. **[Prop. 1, Step 2]** Prove that per-step score errors accumulate rather than cancel. Required: show that $\mathbb{E}[\delta_t \cdot \delta_{t'}] \geq 0$ for $t \neq t'$ (positive error correlation across timesteps), or find an alternative lower bound argument that does not require this.

3. **[Prop. 2, Freq. analysis]** Bound the higher-order terms in equation (9). Required: an upper bound on the operator norm of the Hessian of $\Phi_T$ applied to directions aligned with the artifact $\eta_\text{VAE}$.

4. **[Prop. 2, Freq. analysis]** Provide a formal characterization of the VAE decoder's frequency response $H(f)$ (equation 7). This may require analyzing specific VAE architectures (e.g., the Stable Diffusion KL-regularized VAE) and computing the DFT of its upsampling kernel.

5. **[Prop. 3]** Provide a rigorous formulation of the unified detection-attribution decision rule (equations in section 4.2) with finite-sample guarantees. Required: concentration inequalities for the empirical Wasserstein distance, and a sample complexity bound for estimating $\mathcal{N}_k$.

6. **[Section 5.3]** Formalize the information-theoretic argument (equation 16). Required: a precise statement of the data processing inequality applied to $\Phi_T$, and conditions under which equality approximately holds.

---

*Document version: draft 0.1 — March 2026.*
*Intended use: supplementary theoretical appendix for DRIFT paper submission.*
*Status: Propositions 1–2 have proof sketches with identified gaps; Proposition 3 is a framework pending empirical support; all formal proofs are pending peer-level rigor.*
