#!/usr/bin/env python3
"""
run_epc.py

CLI entry point for the EPC metamorphic test suite.

Usage:
    python scripts/run_epc.py \\
        --config configs/epc_config.yaml \\
        --server-ip 127.0.0.1 --server-port 8080 \\
        --policy my_policies.my_module:build_policy_fn \\
        --output epc_runs/report.json

`--policy` names a zero-arg factory `module:function` that returns a
`policy_fn(obs) -> action` callable (or `action_chunk` list), matching the
EBench "Integrate Your Own Model" contract. This keeps the target model
fully swappable and out of this repository.

Note: applying scene-level perturbations (illuminance, friction, mass,
etc.) requires a `SceneMutator` bound to a live USD stage handle inside the
GenManip/Isaac Sim server process -- see epc/perturbations.py. This script
wires the seam; you must supply that handle (e.g. via a small in-process
admin RPC on your server fork) for those dimensions to actually mutate the
scene. Dimensions left unwired will raise a clear RuntimeError rather than
silently no-op.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from epc.perturbations import PCDimension, SceneMutator
from epc.relations import MRConfig
from epc.runner import EPCRunner, summarize_report
from epc.perturbations import Severity


DIM_BY_TAG = {d.value: d for d in PCDimension}


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_policy_fn(spec: str):
    """spec = 'module.submodule:factory_function'"""
    module_name, func_name = spec.split(":")
    module = importlib.import_module(module_name)
    factory = getattr(module, func_name)
    return factory()


def build_eval_client(server_ip: str, server_port: int):
    """
    Constructs EBench's documented EvalClient. Import is local so this
    script still runs (e.g. --dry-run) in environments without the
    genmanip_client package installed.
    """
    from genmanip_client.eval_client import EvalClient

    return EvalClient(server_ip=server_ip, server_port=server_port)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the EPC metamorphic test suite against EBench.")
    ap.add_argument("--config", default="configs/epc_config.yaml")
    ap.add_argument("--server-ip", default="127.0.0.1")
    ap.add_argument("--server-port", type=int, default=8080)
    ap.add_argument("--policy", required=True, help="module:factory_function returning policy_fn(obs)->action")
    ap.add_argument("--task-id", default=None, help="Override run.task_id from config")
    ap.add_argument("--seed-episode-id", default=None, help="Override run.seed_episode_id from config")
    ap.add_argument("--output", default=None, help="Path to write JSON report (default: <output_dir>/report.json)")
    ap.add_argument("--dry-run", action="store_true", help="Build the suite and print it without contacting a server")
    args = ap.parse_args()

    cfg = load_config(args.config)
    task_id = args.task_id or cfg["run"]["task_id"]
    seed_episode_id = args.seed_episode_id or cfg["run"]["seed_episode_id"]
    dims = [DIM_BY_TAG[tag] for tag in cfg["dimensions"]]

    from epc.test_generator import generate_suite
    suite = generate_suite(task_id, seed_episode_id, dimensions=dims)

    if args.dry_run:
        print(f"Generated {len(suite.cases)} follow-up cases for task '{task_id}':")
        for case in suite.cases:
            print(f"  [{case.dimension.value}] severity={case.perturbation.severity.value} "
                  f"embodiment={case.embodiment} params={case.perturbation.params}")
        return

    policy_fn = load_policy_fn(args.policy)
    client = build_eval_client(args.server_ip, args.server_port)

    # SceneMutator needs a live USD stage handle from inside the server
    # process; see module docstring. Left unbound here -- physical
    # perturbations will raise until you wire that handle.
    scene_mutator = SceneMutator(stage=None)

    mr_config = MRConfig(
        eps_action=cfg["thresholds"]["eps_action"],
        eps_cost=cfg["thresholds"]["eps_cost"],
        eps_traj=cfg["thresholds"]["eps_traj"],
    )

    runner = EPCRunner(
        client=client,
        scene_mutator=scene_mutator,
        worker_id=cfg["eval_client"]["worker_id"],
        policy_fn=policy_fn,
        mr_config=mr_config,
        max_steps=cfg["eval_client"]["max_steps"],
    )

    report = runner.run_task(task_id, seed_episode_id, dimensions=dims)
    print(summarize_report(report))

    output_path = args.output or str(Path(cfg["run"]["output_dir"]) / f"{task_id}_report.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "task_id": report.task_id,
                "seed_outcome": report.seed_behavior.outcome,
                "vdr": report.metrics.vdr,
                "mean_bds": report.metrics.mean_bds,
                "pcd": report.metrics.perturbation_coverage_diversity(),
                "vdr_by_dimension": report.metrics.vdr_by_dimension(),
                "cases": [
                    {
                        "dimension": co.case.dimension.value,
                        "severity": co.case.perturbation.severity.value,
                        "embodiment": co.case.embodiment,
                        "violated": co.violated,
                        "outcome": co.behavior.outcome,
                        "cost": co.behavior.cost(),
                        "details": [r.detail for r in co.results],
                    }
                    for co in report.case_outcomes
                ],
            },
            f,
            indent=2,
        )
    print(f"\nWrote report to {output_path}")


if __name__ == "__main__":
    main()
