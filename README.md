# EPC — Embodied Physical Compliance

A metamorphic testing framework for evaluating **physical consistency** in embodied AI systems (manipulation policies, VLA models, etc.), built on top of [EBench / GenManip](https://github.com/InternRobotics/EBench).

EPC treats a target policy as a **black box**: instead of asking "did the robot succeed?", it asks "does the robot's behavior change in a *physically sensible* way when the scene is perturbed?" — e.g. success should not improve when lighting gets worse, trajectories should stay stable when only the object's texture changes, and behavior should agree across semantically equivalent robot embodiments.

## How it works

1. **Record a seed behavior** `B(E) = (S, Π, Γ)` — outcome, action sequence, and end-effector trajectory — for an unperturbed episode.
2. **Generate follow-up test cases** `x' = δ_k(x; λ)` by applying a physical perturbation along one of nine dimensions, at a chosen severity.
3. **Apply the perturbation** to the simulated scene and re-run the same seed episode to record a follow-up behavior `B(E')`.
4. **Check a metamorphic relation** `Φ_p(B(E), B(E')) → {Pass, Violation}` appropriate to that dimension, using a **multi-level oracle** (outcome → decision → trajectory → robustness).
5. **Aggregate metrics** across all cases (violation rate, deviation severity, per-layer trigger rate, perturbation coverage diversity).

## The nine perturbation dimensions

| Dimension | What it perturbs | Relation family | MR |
|---|---|---|---|
| PC1-a Illuminance Intensity | Scene lighting | Monotonic degradation | MR1 |
| PC1-b Surface Texture | Object materials | Invariance | MR2 |
| PC2 Perspective Shift | Camera viewpoint | Invariance | MR3 |
| PC3-a Occlusion Ratio | Object visibility | Monotonic degradation | MR4 |
| PC3-b Spatial Depth | Object depth placement | Invariance | MR5 |
| PC4-a Contact Friction | Surface friction | Boundary stability | MR6 |
| PC4-b Structural Rigidity | Soft/rigid body stiffness | Boundary stability | MR7 |
| PC5-a Mass Magnitude | Object mass | Monotonic degradation | MR8 |
| PC5-b Mass Distribution | Center of mass / cross-embodiment | Cross-embodiment consistency | MR9 |

**Relation families:**
- **Monotonic degradation** — a stronger perturbation must not yield a strictly better outcome, lower cost, or reduced trajectory deviation than a weaker one.
- **Invariance** — outcome, cost, and trajectory should stay essentially unchanged (within tolerance) since the perturbation shouldn't affect task-relevant physics.
- **Boundary stability** — behavior should stay stable as long as the perturbation keeps the object within the same qualitative physical regime (e.g. still rigid, still static friction).
- **Cross-embodiment consistency** — the *sign* of the behavioral change should agree across different robot embodiments facing an equivalent perturbation.

## Multi-level oracle

| Layer | Checks |
|---|---|
| L1 — Outcome | Task success/failure flip |
| L2 — Decision | DTW divergence between action sequences |
| L3 — Trajectory | Arc-length-aligned end-effector deviation `D(θ, θ')` |
| L4 — Robustness | Trend stability across a full severity sweep / across embodiments |

## Metrics

- **VDR** — Violation Detection Rate (overall and per dimension)
- **BDS** — Behavioral Deviation Score (mean severity of detected violations)
- **OTR** — Oracle Trigger Rate, per layer (L1–L4)
- **PCD** — Perturbation Coverage Diversity (unique violation clusters per 100 violations, via lightweight k-means over dimension/layer/time-window features)

## Project layout

```
epc_project/
├── epc/
│   ├── __init__.py
│   ├── behavior.py         # Behavior dataclass + EvalClient episode recorder
│   ├── perturbations.py    # 9 perturbation dimensions + SceneMutator seam
│   ├── oracle.py           # L1-L4 oracle primitives (DTW, trajectory alignment)
│   ├── relations.py        # MR1-MR9 metamorphic relations
│   ├── test_generator.py   # Builds severity-sweep test suites per seed task
│   ├── metrics.py          # VDR / BDS / OTR / PCD aggregation
│   └── runner.py           # End-to-end orchestration against EvalClient
├── configs/
│   └── epc_config.yaml     # Severities, thresholds, dimensions, embodiments
├── scripts/
│   └── run_epc.py          # CLI entry point
├── tests/                  # Offline/unit tests (dependency-free, no live server)
└── requirements.txt
```

## Integration boundary

EBench's public client (`genmanip_client.eval_client.EvalClient`) only exposes a black-box observation/action loop — it does not expose scene authoring. Applying the physical perturbations (lighting, friction, mass, camera pose, etc.) requires a live USD stage handle inside the Isaac Sim / GenManip server process.

`epc/perturbations.py` defines this seam explicitly via the `SceneMutator` class: each `_apply_*` method documents which USD/Isaac Sim API it should call (`pxr.UsdPhysics`, `pxr.UsdLux`, `pxr.UsdShade`, `omni.replicator`), but is left unwired by default. Everything **above** this seam — test generation, the oracle, and metrics — is stage-agnostic and works against any backend that honors the `SceneMutator` interface. Dimensions left unwired raise a clear `RuntimeError` rather than silently no-op, so you always know what's actually being tested.

## Installation

```bash
pip install -r requirements.txt
```

`requirements.txt` covers EPC's own dependencies (`numpy`, `PyYAML`). EBench's `genmanip_client` package must be installed separately — see the [EBench repo](https://github.com/InternRobotics/EBench) and its [custom-model integration guide](https://internrobotics.github.io/EBench-doc/evaluation/custom-model/).

## Usage

Preview the generated test suite without contacting a live server:

```bash
python scripts/run_epc.py --config configs/epc_config.yaml --policy my_policies.my_module:build_policy_fn --dry-run
```

Run the full suite against a live EBench server:

```bash
python scripts/run_epc.py \
  --config configs/epc_config.yaml \
  --server-ip 127.0.0.1 --server-port 8080 \
  --policy my_policies.my_module:build_policy_fn \
  --output epc_runs/report.json
```

`--policy` takes a `module:factory_function` spec — a zero-argument factory that returns a `policy_fn(obs) -> action` (or `action_chunk`) callable, matching EBench's "Integrate Your Own Model" contract. This keeps your target model fully swappable and out of this repository.

The run prints a summary (VDR, mean BDS, PCD, per-layer OTR, per-dimension VDR, MR9 cross-embodiment result) and writes a full JSON report with per-case details.

## Configuration

`configs/epc_config.yaml` controls:
- **severities** — magnitude bands for slight/moderate/severe perturbations
- **thresholds** — `eps_action` (DTW), `eps_cost` (execution-cost delta), `eps_traj` (trajectory deviation)
- **dimensions** — which of the nine PC dimensions to include in a run
- **embodiments** — robot embodiments used for the PC5-b cross-embodiment check
- **eval_client** / **run** — worker ID, step budget, chunking, task/episode IDs, output directory

## Status

This is a research prototype. The test generation, oracle, and metrics layers are fully implemented and testable offline. The `SceneMutator` perturbation methods are stubs describing the intended USD/Isaac Sim calls and need to be wired to your specific GenManip server fork (or an admin RPC you add alongside it) before physical perturbations actually take effect in simulation.
