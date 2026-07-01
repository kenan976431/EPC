"""
relations.py

Implements MR1-MR9 from Section III-B as executable predicates
Phi_p(B(E), B(E')) -> {Pass, Violation}, each combining the L1-L4 oracle
primitives from oracle.py. Every relation returns a list of OracleResult
(one per applicable layer) so callers can compute VDR / BDS / OTR_L1..L4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .behavior import Behavior
from .oracle import (
    OracleLayer,
    OracleResult,
    check_decision_stability,
    check_trajectory_deviation,
)
from .perturbations import PCDimension


@dataclass
class MRConfig:
    eps_action: float = 0.5     # epsilon_c analogue for decision-level DTW
    eps_cost: float = 2.0       # epsilon_c: allowed cost delta for invariance/boundary MRs
    eps_traj: float = 0.05      # epsilon_d: allowed trajectory deviation (meters)


def _outcome_ok(theta: Behavior, theta_prime: Behavior) -> bool:
    return theta.outcome >= 0.5 and theta_prime.outcome >= 0.5


# ---------------------------------------------------------------------------
# Monotonic degradation: MR1 (illuminance), MR4 (occlusion), MR8 (mass mag.)
# Stronger perturbation must not yield strictly better outcome/cost/deviation.
# ---------------------------------------------------------------------------
def mr_monotonic_degradation(
    theta_low: Behavior, theta_high: Behavior, cfg: MRConfig, name: str
) -> List[OracleResult]:
    results: List[OracleResult] = []

    if _outcome_ok(theta_low, theta_high):
        cost_ok = theta_high.cost() >= theta_low.cost() - cfg.eps_cost
        sev = max(0.0, (theta_low.cost() - theta_high.cost()) / max(theta_low.cost(), 1e-6))
        results.append(
            OracleResult(
                layer=OracleLayer.L1_OUTCOME,
                passed=cost_ok,
                detail=f"[{name}] stronger perturbation cost {theta_high.cost():.2f} "
                       f"vs weaker {theta_low.cost():.2f}",
                severity_score=sev,
            )
        )
    else:
        # Outcome-level violation: task got *easier* under stronger perturbation.
        outcome_regressed = theta_high.outcome > theta_low.outcome
        results.append(
            OracleResult(
                layer=OracleLayer.L1_OUTCOME,
                passed=not outcome_regressed,
                detail=f"[{name}] outcome low={theta_low.outcome} high={theta_high.outcome}",
                severity_score=1.0 if outcome_regressed else 0.0,
            )
        )

    results.append(check_decision_stability(theta_low, theta_high, cfg.eps_action))
    results.append(check_trajectory_deviation(theta_low, theta_high, cfg.eps_traj))
    return results


# ---------------------------------------------------------------------------
# Invariance: MR2 (texture), MR3 (perspective), MR5 (depth)
# Follow-up must preserve success, comparable cost, small trajectory drift.
# ---------------------------------------------------------------------------
def mr_invariance(theta: Behavior, theta_prime: Behavior, cfg: MRConfig, name: str) -> List[OracleResult]:
    results: List[OracleResult] = []

    outcome_ok = (theta.outcome >= 0.5) == (theta_prime.outcome >= 0.5)
    results.append(
        OracleResult(
            OracleLayer.L1_OUTCOME,
            outcome_ok,
            f"[{name}] outcome {theta.outcome} -> {theta_prime.outcome}",
            severity_score=0.0 if outcome_ok else 1.0,
        )
    )

    cost_delta = abs(theta.cost() - theta_prime.cost())
    cost_ok = cost_delta <= cfg.eps_cost
    sev = max(0.0, (cost_delta - cfg.eps_cost) / max(cfg.eps_cost, 1e-6))
    results.append(OracleResult(OracleLayer.L2_DECISION, cost_ok, f"[{name}] |dCost|={cost_delta:.2f}", sev))

    results.append(check_trajectory_deviation(theta, theta_prime, cfg.eps_traj))
    return results


# ---------------------------------------------------------------------------
# Boundary stability: MR6 (friction), MR7 (rigidity)
# Within the same physical regime, behavior must stay stable.
# ---------------------------------------------------------------------------
def mr_boundary_stability(
    theta_a: Behavior, theta_b: Behavior, cfg: MRConfig, name: str, same_regime: bool
) -> List[OracleResult]:
    if not same_regime:
        return [OracleResult(OracleLayer.L1_OUTCOME, True, f"[{name}] different regime, no constraint")]

    results = []
    cost_delta = abs(theta_a.cost() - theta_b.cost())
    cost_ok = cost_delta <= cfg.eps_cost
    sev = max(0.0, (cost_delta - cfg.eps_cost) / max(cfg.eps_cost, 1e-6))
    results.append(OracleResult(OracleLayer.L2_DECISION, cost_ok, f"[{name}] |dCost| within regime", sev))
    results.append(check_trajectory_deviation(theta_a, theta_b, cfg.eps_traj))
    return results


# ---------------------------------------------------------------------------
# Cross-embodiment consistency: MR9 (mass distribution)
# Sign of cost delta should agree across semantically equivalent embodiments.
# ---------------------------------------------------------------------------
def mr_cross_embodiment_consistency(
    delta_costs_by_embodiment: Sequence[float], name: str = "MR9"
) -> OracleResult:
    if len(delta_costs_by_embodiment) < 2:
        return OracleResult(OracleLayer.L4_ROBUSTNESS, True, f"[{name}] insufficient embodiments")

    signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in delta_costs_by_embodiment]
    consistent = len(set(s for s in signs if s != 0)) <= 1
    return OracleResult(
        OracleLayer.L4_ROBUSTNESS,
        consistent,
        f"[{name}] sign(deltaC) per embodiment = {signs}",
        severity_score=0.0 if consistent else 1.0,
    )


# Dispatch table: dimension -> which MR family + a human-readable MR id,
# matching the taxonomy in Table I / Section III-B.
MR_REGISTRY = {
    PCDimension.PC1A_ILLUMINANCE: ("monotonic_degradation", "MR1"),
    PCDimension.PC1B_TEXTURE: ("invariance", "MR2"),
    PCDimension.PC2_PERSPECTIVE: ("invariance", "MR3"),
    PCDimension.PC3A_OCCLUSION: ("monotonic_degradation", "MR4"),
    PCDimension.PC3B_DEPTH: ("invariance", "MR5"),
    PCDimension.PC4A_FRICTION: ("boundary_stability", "MR6"),
    PCDimension.PC4B_RIGIDITY: ("boundary_stability", "MR7"),
    PCDimension.PC5A_MASS_MAGNITUDE: ("monotonic_degradation", "MR8"),
    PCDimension.PC5B_MASS_DISTRIBUTION: ("cross_embodiment_consistency", "MR9"),
}
