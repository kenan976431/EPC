"""
metrics.py

Aggregate metrics from Section IV-B:
  VDR  - violation detection rate
  BDS  - behavioral deviation severity
  OTR  - oracle trigger rate, per layer L1-L4
  TD   - trajectory deviation
  PCD  - perturbation coverage diversity (via clustering over violation
         feature vectors: triggered dimension, oracle layer, time window)
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

from .oracle import OracleResult, OracleLayer, behavioral_deviation_score


@dataclass
class ViolationRecord:
    dimension: str
    layer: OracleLayer
    severity_score: float
    time_window: int  # coarse step index bucket, for PCD clustering


@dataclass
class MetricsAccumulator:
    total_cases: int = 0
    violated_cases: int = 0
    violations: List[ViolationRecord] = field(default_factory=list)
    bds_scores: List[float] = field(default_factory=list)
    layer_trigger_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    dimension_totals: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    dimension_violations: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record_case(self, dimension: str, results: List[OracleResult], time_window: int = 0) -> bool:
        self.total_cases += 1
        self.dimension_totals[dimension] += 1
        any_violation = any(not r.passed for r in results)
        if any_violation:
            self.violated_cases += 1
            self.dimension_violations[dimension] += 1
            for r in results:
                if not r.passed:
                    self.layer_trigger_counts[r.layer.value] += 1
                    self.violations.append(
                        ViolationRecord(dimension, r.layer, r.severity_score, time_window)
                    )
        self.bds_scores.append(behavioral_deviation_score(results))
        return any_violation

    @property
    def vdr(self) -> float:
        """Violation Detection Rate."""
        if self.total_cases == 0:
            return 0.0
        return self.violated_cases / self.total_cases

    @property
    def mean_bds(self) -> float:
        """Mean Behavioral Deviation Score over all cases (0 for non-violating)."""
        if not self.bds_scores:
            return 0.0
        return float(np.mean(self.bds_scores))

    def otr(self, layer: OracleLayer) -> float:
        """Oracle Trigger Rate for a given layer."""
        if self.total_cases == 0:
            return 0.0
        return self.layer_trigger_counts.get(layer.value, 0) / self.total_cases

    def vdr_by_dimension(self) -> Dict[str, float]:
        """Exact per-dimension VDR using tracked per-dimension denominators."""
        return {
            dim: self.dimension_violations.get(dim, 0) / max(total, 1)
            for dim, total in self.dimension_totals.items()
        }

    def perturbation_coverage_diversity(self, n_clusters: int = None) -> float:
        """
        PCD = unique violation clusters per 100 detected violations
        (Section IV-E). Uses simple k-means-style clustering over a
        (dimension_hash, layer_hash, time_window) feature space; swap in
        scikit-learn's SpectralClustering for closer fidelity to the paper.
        """
        if not self.violations:
            return 0.0

        dims = sorted({v.dimension for v in self.violations})
        layers = [l.value for l in OracleLayer]
        dim_idx = {d: i for i, d in enumerate(dims)}
        layer_idx = {l: i for i, l in enumerate(layers)}

        feats = np.array(
            [
                [dim_idx[v.dimension], layer_idx[v.layer.value], v.time_window]
                for v in self.violations
            ],
            dtype=np.float32,
        )
        # Normalize columns.
        feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)

        k = n_clusters or max(1, min(len(dims) * len(layers), len(feats)))
        clusters = _simple_kmeans(feats, k)
        n_unique = len(set(clusters))
        return 100.0 * n_unique / len(self.violations)


def _simple_kmeans(x: np.ndarray, k: int, n_iter: int = 25, seed: int = 0) -> np.ndarray:
    """Minimal dependency-free k-means (no sklearn requirement)."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    k = min(k, n)
    centers = x[rng.choice(n, k, replace=False)]
    assign = np.zeros(n, dtype=int)

    for _ in range(n_iter):
        dists = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
        new_assign = dists.argmin(axis=1)
        if np.array_equal(new_assign, assign):
            break
        assign = new_assign
        for c in range(k):
            members = x[assign == c]
            if len(members) > 0:
                centers[c] = members.mean(axis=0)
    return assign
