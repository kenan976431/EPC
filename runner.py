"""
runner.py

Orchestrates the full EPC pipeline (Section III/IV) against a live EBench
EvalClient:

  1. Record a source behavior B(E) for the seed episode (no perturbation).
  2. For each generated follow-up test case x' = delta_k(x; lambda):
       a. Apply the perturbation via SceneMutator (server-side seam).
       b. Reset the same seed episode and record B(E').
       c. Route (B(E), B(E')) to the MR family implied by the dimension.
       d. Feed results into MetricsAccumulator.
  3. For monotonic-degradation dimensions, also run an L4 robustness check
     across the full severity sweep for that dimension.
  4. For PC5-b, run the cross-embodiment consistency check across embodiments.

This module intentionally has no Isaac-Sim-specific imports: `client` must
satisfy the EvalClient interface (`reset()`, `step(actions)`), and
`scene_mutator` must satisfy the SceneMutator interface. In a fully wired
deployment, `scene_mutator.apply(...)` runs on the server (or via an admin
RPC you add) before `client.reset()` re-instantiates the episode.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .behavior import Behavior, BehaviorRecorder
from .metrics import MetricsAccumulator
from .oracle import OracleResult, check_robustness_trend
from .perturbations import PCDimension, PerturbationSpec, SceneMutator
from .relations import (
    MRConfig,
    mr_boundary_stability,
    mr_cross_embodiment_consistency,
    mr_invariance,
    mr_monotonic_degradation,
    MR_REGISTRY,
)
from .test_generator import TestCase, TestSuite, generate_suite


@dataclass
class CaseOutcome:
    case: TestCase
    behavior: Behavior
    results: List[OracleResult]
    violated: bool


@dataclass
class RunReport:
    task_id: str
    seed_behavior: Behavior
    case_outcomes: List[CaseOutcome]
    metrics: MetricsAccumulator
    l4_results: List[OracleResult]
    mr9_result: Optional[OracleResult]


class EPCRunner:
    def __init__(
        self,
        client,
        scene_mutator: SceneMutator,
        worker_id: str,
        policy_fn: Callable[[Dict[str, Any]], Any],
        mr_config: MRConfig = None,
        max_steps: int = 500,
    ):
        self.client = client
        self.scene_mutator = scene_mutator
        self.worker_id = worker_id
        self.policy_fn = policy_fn
        self.mr_config = mr_config or MRConfig()
        self.max_steps = max_steps
        self.recorder = BehaviorRecorder(client, worker_id)

    def _record(self, reset_fn: Optional[Callable[[], None]] = None) -> Behavior:
        return self.recorder.run_episode(self.policy_fn, reset_fn=reset_fn, max_steps=self.max_steps)

    def run_task(
        self,
        task_id: str,
        seed_episode_id: str,
        dimensions: List[PCDimension] = None,
        severities=None,
    ) -> RunReport:
        suite: TestSuite = generate_suite(
            task_id, seed_episode_id, dimensions=dimensions,
            **({"severities": severities} if severities else {}),
        )

        # 1. Source behavior, unperturbed.
        seed_behavior = self._record()

        metrics = MetricsAccumulator()
        outcomes: List[CaseOutcome] = []
        by_dimension: Dict[PCDimension, List[CaseOutcome]] = defaultdict(list)
        embodiment_delta_costs: List[float] = []

        for i, case in enumerate(suite.cases):
            def reset_fn(case=case):
                self.scene_mutator.apply(case.perturbation)

            follow_up = self._record(reset_fn=reset_fn)

            family, mr_name = MR_REGISTRY[case.dimension]
            if family == "monotonic_degradation":
                results = mr_monotonic_degradation(seed_behavior, follow_up, self.mr_config, mr_name)
            elif family == "invariance":
                results = mr_invariance(seed_behavior, follow_up, self.mr_config, mr_name)
            elif family == "boundary_stability":
                # Same physical regime := same qualitative severity band, i.e.
                # the perturbation did not cross a friction/rigidity phase
                # boundary (e.g. static->kinetic slip, rigid->deformable).
                results = mr_boundary_stability(
                    seed_behavior, follow_up, self.mr_config, mr_name, same_regime=True
                )
            elif family == "cross_embodiment_consistency":
                embodiment_delta_costs.append(follow_up.cost() - seed_behavior.cost())
                results = []
            else:
                results = []

            violated = metrics.record_case(case.dimension.value, results, time_window=i) if results else False
            outcome = CaseOutcome(case=case, behavior=follow_up, results=results, violated=violated)
            outcomes.append(outcome)
            by_dimension[case.dimension].append(outcome)

        # L4 robustness: per monotonic-degradation dimension, check the
        # ordered severity sweep trend.
        l4_results: List[OracleResult] = []
        for dim, cases in by_dimension.items():
            family, mr_name = MR_REGISTRY[dim]
            if family != "monotonic_degradation":
                continue
            ordered = sorted(cases, key=lambda c: c.case.perturbation.severity.magnitude)
            outcomes_seq = [seed_behavior.outcome] + [c.behavior.outcome for c in ordered]
            costs_seq = [seed_behavior.cost()] + [c.behavior.cost() for c in ordered]
            l4 = check_robustness_trend(outcomes_seq, costs_seq)
            l4_results.append(l4)
            metrics.record_case(dim.value, [l4], time_window=len(suite.cases))

        # MR9: cross-embodiment sign consistency for PC5-b.
        mr9_result = None
        if embodiment_delta_costs:
            mr9_result = mr_cross_embodiment_consistency(embodiment_delta_costs)
            metrics.record_case(
                PCDimension.PC5B_MASS_DISTRIBUTION.value, [mr9_result], time_window=len(suite.cases) + 1
            )

        return RunReport(
            task_id=task_id,
            seed_behavior=seed_behavior,
            case_outcomes=outcomes,
            metrics=metrics,
            l4_results=l4_results,
            mr9_result=mr9_result,
        )


def summarize_report(report: RunReport) -> str:
    m = report.metrics
    lines = [
        f"Task: {report.task_id}",
        f"Seed outcome: {report.seed_behavior.outcome:.2f}  cost: {report.seed_behavior.cost():.2f}",
        f"Cases: {m.total_cases}   Violated: {m.violated_cases}   VDR: {m.vdr:.3f}",
        f"Mean BDS: {m.mean_bds:.3f}",
        f"PCD: {m.perturbation_coverage_diversity():.2f} unique clusters / 100 violations",
        "OTR by layer:",
    ]
    from .oracle import OracleLayer
    for layer in OracleLayer:
        lines.append(f"  {layer.value}: {m.otr(layer):.3f}")
    lines.append("VDR by dimension:")
    for dim, rate in sorted(m.vdr_by_dimension().items()):
        lines.append(f"  {dim}: {rate:.3f}")
    if report.mr9_result is not None:
        lines.append(f"MR9 (cross-embodiment): {'PASS' if report.mr9_result.passed else 'VIOLATION'} "
                     f"- {report.mr9_result.detail}")
    return "\n".join(lines)
