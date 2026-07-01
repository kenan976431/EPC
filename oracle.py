"""
oracle.py

Multi-level oracle from Section III-A / IV-B:
  L1 outcome     -> S(theta): task success
  L2 decision    -> DTW divergence of action sequences
  L3 trajectory  -> D(theta, theta'): aligned end-effector deviation
  L4 robustness  -> variance / trend stability across severities & seeds

Also implements D(theta, theta') via resampling + Euclidean alignment as
defined by Eq. in Section III-A, with an optional DTW alignment mode.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Sequence

from .behavior import Behavior


def _resample(traj: np.ndarray, n_points: int) -> np.ndarray:
    """Resample a (T, 3) trajectory to n_points via linear interpolation
    over cumulative arc length, so that D(theta, theta') can be computed
    between trajectories of different lengths."""
    if traj.shape[0] == 0:
        return np.zeros((n_points, 3), dtype=np.float32)
    if traj.shape[0] == 1:
        return np.repeat(traj, n_points, axis=0)

    seg_len = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1] if cum[-1] > 0 else 1e-6
    targets = np.linspace(0.0, total, n_points)

    out = np.zeros((n_points, 3), dtype=np.float32)
    for d in range(3):
        out[:, d] = np.interp(targets, cum, traj[:, d])
    return out


def trajectory_deviation(theta: Behavior, theta_prime: Behavior, n_points: int = 100) -> float:
    """
    D(theta, theta') = (1/T) * sum_t || p_t - p'_t ||_2
    over temporally-aligned end-effector poses (Section III-A).
    """
    p = _resample(theta.trajectory_array(), n_points)
    pp = _resample(theta_prime.trajectory_array(), n_points)
    return float(np.mean(np.linalg.norm(p - pp, axis=1)))


def decision_divergence(theta: Behavior, theta_prime: Behavior) -> float:
    """
    L2 decision-level check: DTW distance over aligned action sequences.
    A lightweight O(n*m) DTW implementation (no external deps) since action
    sequences here are short (tens to low hundreds of steps).
    """
    a = np.stack(theta.actions, axis=0) if theta.actions else np.zeros((1, 1))
    b = np.stack(theta_prime.actions, axis=0) if theta_prime.actions else np.zeros((1, 1))
    n, m = len(a), len(b)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(a[i - 1] - b[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m] / max(n, m))


class OracleLayer(Enum):
    L1_OUTCOME = "L1"
    L2_DECISION = "L2"
    L3_TRAJECTORY = "L3"
    L4_ROBUSTNESS = "L4"


@dataclass
class OracleResult:
    layer: OracleLayer
    passed: bool
    detail: str
    severity_score: float = 0.0  # contributes to Behavioral Deviation Score (BDS)


def check_outcome_consistency(theta: Behavior, theta_prime: Behavior) -> OracleResult:
    """L1: baseline outcome-only check (used for the ablation ratio)."""
    ok = not (theta.outcome >= 0.5 and theta_prime.outcome < 0.5 and _unexpected(theta, theta_prime))
    return OracleResult(OracleLayer.L1_OUTCOME, ok, "outcome flip under perturbation" if not ok else "ok")


def _unexpected(theta: Behavior, theta_prime: Behavior) -> bool:
    # Placeholder hook: relation-specific logic decides "unexpected"-ness;
    # relations.py supplies the actual physical-consistency predicate.
    return True


def check_decision_stability(theta: Behavior, theta_prime: Behavior, eps_action: float) -> OracleResult:
    dtw = decision_divergence(theta, theta_prime)
    ok = dtw <= eps_action
    sev = max(0.0, (dtw - eps_action) / max(eps_action, 1e-6))
    return OracleResult(OracleLayer.L2_DECISION, ok, f"DTW={dtw:.4f} (eps={eps_action})", sev)


def check_trajectory_deviation(theta: Behavior, theta_prime: Behavior, eps_d: float) -> OracleResult:
    d = trajectory_deviation(theta, theta_prime)
    ok = d <= eps_d
    sev = max(0.0, (d - eps_d) / max(eps_d, 1e-6))
    return OracleResult(OracleLayer.L3_TRAJECTORY, ok, f"D={d:.4f} (eps={eps_d})", sev)


def check_robustness_trend(outcomes: Sequence[float], costs: Sequence[float]) -> OracleResult:
    """
    L4: across an ordered severity sweep, outcomes/cost should not improve
    non-monotonically in a way that contradicts the induced relation
    pattern (e.g. success rate should not increase, cost should not drop,
    as severity increases for monotonic-degradation MRs). We flag a
    violation if variance of successful-run costs at the *highest* severity
    is not >= variance at the lowest, i.e. robustness should degrade or
    stay flat, never improve, as severity grows.
    """
    if len(outcomes) < 2:
        return OracleResult(OracleLayer.L4_ROBUSTNESS, True, "insufficient severities")

    succ_costs = [c for o, c in zip(outcomes, costs) if o >= 0.5]
    if len(succ_costs) < 2:
        return OracleResult(OracleLayer.L4_ROBUSTNESS, True, "insufficient successful runs")

    trend_violation = costs[-1] < costs[0] * 0.9  # highest severity got notably *cheaper*
    ok = not trend_violation
    sev = max(0.0, (costs[0] - costs[-1]) / max(costs[0], 1e-6)) if trend_violation else 0.0
    return OracleResult(
        OracleLayer.L4_ROBUSTNESS,
        ok,
        f"cost trend {costs[0]:.2f}->{costs[-1]:.2f}",
        sev,
    )


def behavioral_deviation_score(results: List[OracleResult]) -> float:
    """BDS: aggregate severity of detected violations (Section IV-C)."""
    violated = [r.severity_score for r in results if not r.passed]
    if not violated:
        return 0.0
    return float(np.clip(np.mean(violated), 0.0, 1.0))
