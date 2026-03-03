"""
evaluation/evaluator.py
-----------------------
Unified DRIFT evaluator that produces paper-style result tables.

Responsibility:
    Orchestrate the full evaluation pipeline across all generators,
    aggregate results into a structured pandas DataFrame, and provide
    comparison utilities for contrasting DRIFT against baselines.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .metrics import (
    compute_auc,
    compute_ap,
    compute_accuracy,
    compute_attribution_accuracy,
    measure_inference_time,
    compute_cross_generator_auc,
)

logger = logging.getLogger(__name__)


class DRIFTEvaluator:
    """Unified evaluator for DRIFT experiments.

    Produces evaluation tables compatible with the format used in the paper
    (Table 1: cross-generator AUC/AP, Table 2: attribution accuracy).

    Args:
        device: PyTorch device string (``"cuda"`` or ``"cpu"``).
        threshold: Decision threshold used for accuracy computation (default
            0.5).
        output_dir: If set, results CSV files and Markdown tables are written
            here.

    Example::

        evaluator = DRIFTEvaluator(device="cuda", output_dir="./results")
        df = evaluator.run_full_evaluation(pipeline, test_loaders)
        print(df.to_string())
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        threshold: float = 0.5,
        output_dir: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device)
        self.threshold = threshold
        self.output_dir = output_dir
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def _collect_scores(
        self,
        model: Any,
        dataloader: DataLoader,
    ) -> tuple[List[int], List[float]]:
        """Run *model* over *dataloader* and collect ground-truth labels and
        predicted scores.

        Args:
            model: Callable accepting ``images`` and returning a score tensor.
            dataloader: DataLoader returning ``(image, label)`` tuples.

        Returns:
            Tuple ``(y_true, y_scores)`` as Python lists.
        """
        y_true: List[int] = []
        y_scores: List[float] = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                scores = model(images)

                if isinstance(scores, torch.Tensor):
                    scores = scores.squeeze().cpu()
                    if scores.ndim == 0:
                        scores = scores.unsqueeze(0)
                    scores = scores.tolist()
                else:
                    scores = list(scores)

                labels_list = (
                    labels.cpu().tolist()
                    if isinstance(labels, torch.Tensor)
                    else list(labels)
                )

                y_true.extend(labels_list)
                y_scores.extend(scores)

        return y_true, y_scores

    def evaluate_generator(
        self,
        model: Any,
        dataloader: DataLoader,
        generator_name: str = "unknown",
    ) -> Dict[str, float]:
        """Evaluate *model* on a single generator's test loader.

        Args:
            model: Inference callable (images → scores).
            dataloader: DataLoader for one generator's test data.
            generator_name: Human-readable name (used only for logging).

        Returns:
            Dict with keys ``"auc"``, ``"ap"``, ``"acc"``.
        """
        y_true, y_scores = self._collect_scores(model, dataloader)
        metrics = {
            "auc": compute_auc(y_true, y_scores),
            "ap": compute_ap(y_true, y_scores),
            "acc": compute_accuracy(y_true, y_scores, threshold=self.threshold),
        }
        logger.info(
            "[%s] AUC=%.4f AP=%.4f ACC=%.4f",
            generator_name,
            metrics["auc"],
            metrics["ap"],
            metrics["acc"],
        )
        return metrics

    def run_full_evaluation(
        self,
        model: Any,
        test_loaders: Dict[str, DataLoader],
        compute_speed: bool = True,
    ) -> pd.DataFrame:
        """Run a complete cross-generator evaluation and return a summary table.

        Produces a DataFrame with one row per generator and columns:
        ``Generator``, ``AUC``, ``AP``, ``ACC``.  An additional ``Mean`` row
        contains macro-averages.  Optionally appends a ``ms/img`` column if
        *compute_speed* is ``True``.

        Args:
            model: Inference callable (images → score tensor).
            test_loaders: Dict mapping generator name → DataLoader.
            compute_speed: If ``True``, run ``measure_inference_time()`` using
                the first test loader and append a ``ms/img`` column.

        Returns:
            A ``pd.DataFrame`` in the style of paper Table 1.
        """
        rows: List[Dict[str, Any]] = []
        logger.info("Running full evaluation on %d generators ...", len(test_loaders))

        for gen_name, loader in test_loaders.items():
            metrics = self.evaluate_generator(model, loader, generator_name=gen_name)
            rows.append(
                {
                    "Generator": gen_name,
                    "AUC": round(metrics["auc"] * 100, 2),
                    "AP": round(metrics["ap"] * 100, 2),
                    "ACC": round(metrics["acc"] * 100, 2),
                }
            )

        if not rows:
            logger.warning("No test loaders provided — returning empty DataFrame.")
            return pd.DataFrame(columns=["Generator", "AUC", "AP", "ACC"])

        df = pd.DataFrame(rows)

        # Append mean row
        mean_row = {
            "Generator": "Mean",
            "AUC": round(df["AUC"].mean(), 2),
            "AP": round(df["AP"].mean(), 2),
            "ACC": round(df["ACC"].mean(), 2),
        }

        # Optionally add inference speed
        if compute_speed and test_loaders:
            first_loader = next(iter(test_loaders.values()))
            ms_per_img = measure_inference_time(
                model, first_loader, device=self.device
            )
            df["ms/img"] = "—"
            mean_row["ms/img"] = round(ms_per_img, 2)

        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

        logger.info("Evaluation complete.\n%s", df.to_string(index=False))

        if self.output_dir is not None:
            csv_path = os.path.join(self.output_dir, "evaluation_results.csv")
            df.to_csv(csv_path, index=False)
            logger.info("Results saved to '%s'.", csv_path)

        return df

    # ------------------------------------------------------------------
    # Attribution evaluation
    # ------------------------------------------------------------------

    def evaluate_attribution(
        self,
        attribution_model: Any,
        test_loaders: Dict[str, DataLoader],
    ) -> pd.DataFrame:
        """Evaluate generator attribution accuracy.

        Calls *attribution_model* with image batches and expects it to return
        predicted generator name strings.

        Args:
            attribution_model: Callable accepting images and returning a list
                or array of predicted generator name strings.
            test_loaders: Dict mapping true generator name → DataLoader.

        Returns:
            DataFrame with columns ``Generator`` and ``Attribution Accuracy (%)``.
        """
        rows: List[Dict[str, Any]] = []

        with torch.no_grad():
            for true_gen, loader in test_loaders.items():
                y_true: List[str] = []
                y_pred: List[str] = []

                for images, _ in loader:
                    images = images.to(self.device)
                    preds = attribution_model(images)
                    if isinstance(preds, np.ndarray):
                        preds = preds.tolist()
                    y_pred.extend(preds)
                    y_true.extend([true_gen] * len(preds))

                acc = compute_attribution_accuracy(y_true, y_pred)
                rows.append(
                    {
                        "Generator": true_gen,
                        "Attribution Accuracy (%)": round(acc * 100, 2),
                    }
                )

        df = pd.DataFrame(rows)
        if not df.empty:
            mean_acc = df["Attribution Accuracy (%)"].mean()
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [{"Generator": "Mean", "Attribution Accuracy (%)": round(mean_acc, 2)}]
                    ),
                ],
                ignore_index=True,
            )
        return df

    # ------------------------------------------------------------------
    # Baseline comparison
    # ------------------------------------------------------------------

    def compare_with_baselines(
        self,
        drift_results: pd.DataFrame,
        baseline_results: Dict[str, pd.DataFrame],
    ) -> str:
        """Generate a Markdown comparison table of DRIFT vs. baselines.

        Args:
            drift_results: DataFrame returned by ``run_full_evaluation()``.
                Must contain a ``Generator`` column and at least ``AUC``.
            baseline_results: Dict mapping baseline method name → DataFrame
                with the same schema as *drift_results*.

        Returns:
            A Markdown-formatted string containing a side-by-side AUC
            comparison table suitable for pasting into a paper/report.

        Example::

            md = evaluator.compare_with_baselines(
                drift_results=drift_df,
                baseline_results={"CNNSpot": cnnspot_df, "DIRE": dire_df},
            )
            print(md)
        """
        # Extract AUC column from each result frame
        methods: Dict[str, Dict[str, float]] = {}

        # DRIFT
        methods["DRIFT (ours)"] = dict(
            zip(
                drift_results["Generator"].tolist(),
                drift_results["AUC"].tolist(),
            )
        )

        # Baselines
        for method_name, df in baseline_results.items():
            if "Generator" not in df.columns or "AUC" not in df.columns:
                logger.warning(
                    "Baseline '%s' is missing 'Generator' or 'AUC' columns — skipped.",
                    method_name,
                )
                continue
            methods[method_name] = dict(
                zip(df["Generator"].tolist(), df["AUC"].tolist())
            )

        # Collect all generator names (excluding the Mean row)
        all_gens = sorted(
            {
                g
                for m_dict in methods.values()
                for g in m_dict.keys()
                if g != "Mean"
            }
        )

        method_names = list(methods.keys())

        # Build Markdown
        header = "| Generator | " + " | ".join(method_names) + " |"
        separator = "| --- | " + " | ".join(["---"] * len(method_names)) + " |"
        lines = [header, separator]

        for gen in all_gens:
            row_vals = [f"{methods[m].get(gen, '—')}" for m in method_names]
            lines.append(f"| {gen} | " + " | ".join(row_vals) + " |")

        # Mean row
        mean_vals = [f"{methods[m].get('Mean', '—')}" for m in method_names]
        lines.append("| **Mean** | " + " | ".join(mean_vals) + " |")

        md_table = "\n".join(lines)

        if self.output_dir is not None:
            md_path = os.path.join(self.output_dir, "comparison_table.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("# DRIFT vs. Baselines — AUC (%)\n\n")
                f.write(md_table + "\n")
            logger.info("Comparison table saved to '%s'.", md_path)

        return md_table
