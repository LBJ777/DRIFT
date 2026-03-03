"""
models/features/combined.py
-----------------------------
组合特征提取器（Combined Feature Extractor）

将 F1（EndpointFeatureExtractor） + F2（TrajectoryFeatureExtractor）
+ Frequency（FrequencyFeatureExtractor）融合为 Phase 2 的完整特征向量。

可配置性：
    - use_f1: 是否使用端点特征（F1）
    - use_f2: 是否使用轨迹特征（F2，需要 intermediates）
    - use_freq: 是否使用频域特征

    feature_dim = sum(各启用模块的 feature_dim)

注意：
    - EndpointFeatureExtractor 由 Phase 1 的 Agent 在 endpoint.py 中创建。
    - 如果 endpoint.py 不存在，会优雅地处理 ImportError 并提示用户先运行 Phase 1。
    - 如果 use_f2=True 但 intermediates=None，抛出清晰的错误信息。
"""

from __future__ import annotations

from typing import List, Optional

import torch

import sys
sys.path.insert(0, '/Users/joces/Downloads/Coding_Campus/Deepfake_Project')

from DRIFT.models.features.base import FeatureExtractor
from DRIFT.models.features.trajectory import TrajectoryFeatureExtractor
from DRIFT.models.features.frequency import FrequencyFeatureExtractor

# ── 尝试导入 Phase 1 的端点特征提取器（可能不存在）──────────────────────────
try:
    from DRIFT.models.features.endpoint import EndpointFeatureExtractor
    _ENDPOINT_AVAILABLE = True
except ImportError:
    _ENDPOINT_AVAILABLE = False


class CombinedFeatureExtractor(FeatureExtractor):
    """
    Phase 2 使用的完整特征提取器。

    组合 EndpointFeatureExtractor(F1) + TrajectoryFeatureExtractor(F2)
    + FrequencyFeatureExtractor，输出拼接后的固定长度特征向量。

    Args:
        use_f1: 是否启用端点特征（F1）。需要 endpoint.py 存在。
        use_f2: 是否启用轨迹特征（F2）。需要 intermediates 输入。
        use_freq: 是否启用频域特征。
        f2_steps: F2 轨迹特征提取器的中间状态数量，默认 20。

    Raises:
        ImportError: 若 use_f1=True 但 EndpointFeatureExtractor 不可用。
        ValueError: 若所有特征都被禁用（use_f1=False, use_f2=False, use_freq=False）。

    Example::

        extractor = CombinedFeatureExtractor(use_f1=True, use_f2=True, use_freq=True)
        x_T, intermediates = backbone.invert(x0, return_intermediates=True)
        features = extractor.extract(x_T, intermediates)  # [B, feature_dim]
    """

    def __init__(
        self,
        use_f1: bool = True,
        use_f2: bool = True,
        use_freq: bool = True,
        f2_steps: int = 20,
    ) -> None:
        super().__init__()

        self.use_f1 = use_f1
        self.use_f2 = use_f2
        self.use_freq = use_freq

        self._extractors: List[FeatureExtractor] = []
        self._extractor_names: List[str] = []

        # ── F1: 端点特征 ─────────────────────────────────────────────────────
        if use_f1:
            if not _ENDPOINT_AVAILABLE:
                raise ImportError(
                    "EndpointFeatureExtractor (F1) 不可用：\n"
                    "  请先运行 Phase 1 的 Agent 以创建 "
                    "DRIFT/models/features/endpoint.py。\n"
                    "  或者设置 use_f1=False 跳过端点特征。"
                )
            self._f1 = EndpointFeatureExtractor()
            self._extractors.append(self._f1)
            self._extractor_names.append("F1_endpoint")
        else:
            self._f1 = None

        # ── F2: 轨迹特征 ─────────────────────────────────────────────────────
        if use_f2:
            self._f2 = TrajectoryFeatureExtractor(num_steps=f2_steps)
            self._extractors.append(self._f2)
            self._extractor_names.append("F2_trajectory")
        else:
            self._f2 = None

        # ── Freq: 频域特征 ────────────────────────────────────────────────────
        if use_freq:
            self._freq = FrequencyFeatureExtractor()
            self._extractors.append(self._freq)
            self._extractor_names.append("Freq_frequency")
        else:
            self._freq = None

        if not self._extractors:
            raise ValueError(
                "CombinedFeatureExtractor: 至少需要启用一种特征提取器。\n"
                "请设置 use_f1=True, use_f2=True 或 use_freq=True 中至少一个。"
            )

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    @property
    def feature_dim(self) -> int:
        """返回所有启用模块的 feature_dim 之和。"""
        return sum(e.feature_dim for e in self._extractors)

    def extract(
        self,
        x_T: torch.Tensor,
        intermediates: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        提取并拼接各模块的特征向量。

        Args:
            x_T: 终点噪声张量 ``[B, 3, H, W]``。
            intermediates: 中间状态列表。若 use_f2=True，则必须提供；
                否则会抛出 ValueError。

        Returns:
            拼接后的特征张量 ``[B, feature_dim]``。

        Raises:
            ValueError: 若 use_f2=True 但 intermediates 为 None。
        """
        # 检查 F2 依赖
        if self.use_f2 and intermediates is None:
            raise ValueError(
                "CombinedFeatureExtractor: use_f2=True 时需要提供 intermediates。\n"
                "请使用 backbone.invert(x0, return_intermediates=True) 获取中间状态列表，\n"
                "或将 use_f2 设置为 False（仅使用 F1 和频域特征）。"
            )

        feature_parts: List[torch.Tensor] = []

        # F1: 端点特征
        if self.use_f1 and self._f1 is not None:
            f1_feat = self._f1.extract(x_T, intermediates)    # [B, f1_dim]
            feature_parts.append(f1_feat)

        # F2: 轨迹特征
        if self.use_f2 and self._f2 is not None:
            f2_feat = self._f2.extract(x_T, intermediates)    # [B, f2_dim]
            feature_parts.append(f2_feat)

        # Freq: 频域特征
        if self.use_freq and self._freq is not None:
            freq_feat = self._freq.extract(x_T, intermediates)  # [B, freq_dim]
            feature_parts.append(freq_feat)

        # 拼接
        features = torch.cat(feature_parts, dim=1)  # [B, feature_dim]

        self.validate_output(features)
        return features

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def active_modules(self) -> List[str]:
        """返回当前启用的特征模块名称列表。"""
        return list(self._extractor_names)

    @property
    def module_dims(self) -> dict:
        """返回各模块的 feature_dim 字典。"""
        return {
            name: extractor.feature_dim
            for name, extractor in zip(self._extractor_names, self._extractors)
        }

    def __repr__(self) -> str:
        dims = ", ".join(
            f"{name}={extractor.feature_dim}"
            for name, extractor in zip(self._extractor_names, self._extractors)
        )
        return (
            f"CombinedFeatureExtractor("
            f"use_f1={self.use_f1}, "
            f"use_f2={self.use_f2}, "
            f"use_freq={self.use_freq}, "
            f"feature_dim={self.feature_dim} [{dims}]"
            f")"
        )


# ── 工厂函数：根据 scheme 字符串创建组合提取器 ────────────────────────────────

def build_extractor_from_scheme(
    scheme: str,
    f2_steps: int = 20,
) -> CombinedFeatureExtractor:
    """
    根据特征方案字符串创建 CombinedFeatureExtractor。

    支持的 scheme：
        - ``"F1"``         — 仅端点特征
        - ``"F1+F2"``      — 端点 + 轨迹特征
        - ``"F1+FREQ"``    — 端点 + 频域特征
        - ``"F1+F2+FREQ"`` — 全特征（Phase 2 推荐）
        - ``"F2"``         — 仅轨迹特征
        - ``"FREQ"``       — 仅频域特征
        - ``"F2+FREQ"``    — 轨迹 + 频域特征

    Args:
        scheme: 特征方案字符串（大小写不敏感）。
        f2_steps: F2 特征提取器的 ddim 步数。

    Returns:
        配置好的 ``CombinedFeatureExtractor`` 实例。

    Raises:
        ValueError: 若 scheme 字符串无效。

    Example::

        extractor = build_extractor_from_scheme("F1+F2+FREQ", f2_steps=20)
    """
    scheme_upper = scheme.upper().strip()
    valid_schemes = {
        "F1",
        "F2",
        "FREQ",
        "F1+F2",
        "F1+FREQ",
        "F2+FREQ",
        "F1+F2+FREQ",
    }

    if scheme_upper not in valid_schemes:
        raise ValueError(
            f"无效的特征方案: '{scheme}'。\n"
            f"支持的方案: {sorted(valid_schemes)}"
        )

    use_f1 = "F1" in scheme_upper
    use_f2 = "F2" in scheme_upper
    use_freq = "FREQ" in scheme_upper

    return CombinedFeatureExtractor(
        use_f1=use_f1,
        use_f2=use_f2,
        use_freq=use_freq,
        f2_steps=f2_steps,
    )
