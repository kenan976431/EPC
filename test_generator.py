"""
test_generator.py

Given a seed task x = <E, B> (Section III), generates the family of
follow-up cases x' = delta_k(x; lambda) across EPC dimensions and severity
levels, without altering the task instruction (task semantics held fixed
per Section III's test-case representation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .perturbations import PCDimension, PerturbationSpec, Severity, build_severity_sweep


DEFAULT_PARAMS: Dict[PCDimension, dict] = {
    PCDimension.PC1A_ILLUMINANCE: {"target": "key_light"},
    PCDimension.PC1B_TEXTURE: {"target": "manip_object", "texture_set": "retexture_pool_A"},
    PCDimension.PC2_PERSPECTIVE: {"camera": "front_view", "max_angle_deg": 20},
    PCDimension.PC3A_OCCLUSION: {"occluder": "auto", "camera": "front_view"},
    PCDimension.PC3B_DEPTH: {"target": "manip_object", "axis": "camera_z"},
    PCDimension.PC4A_FRICTION: {"target": "manip_object"},
    PCDimension.PC4B_RIGIDITY: {"target": "manip_object"},
    PCDimension.PC5A_MASS_MAGNITUDE: {"target": "manip_object"},
    PCDimension.PC5B_MASS_DISTRIBUTION: {"target": "manip_object", "embodiments": ["r5a", "lift2"]},
}


@dataclass
class TestCase:
    task_id: str
    seed_episode_id: str
    dimension: PCDimension
    perturbation: PerturbationSpec
    embodiment: str = "default"


@dataclass
class TestSuite:
    task_id: str
    seed_episode_id: str
    cases: List[TestCase] = field(default_factory=list)


def generate_suite(
    task_id: str,
    seed_episode_id: str,
    dimensions: List[PCDimension] = None,
    severities=(Severity.SLIGHT, Severity.MODERATE, Severity.SEVERE),
) -> TestSuite:
    """
    Builds a full EPC follow-up suite for one seed scenario: for every
    requested dimension, a severity sweep of PerturbationSpecs, matching
    the paper's "five severity levels per dimension" design (simplified
    here to the paper's three qualitative bands: Slight/Moderate/Severe).
    """
    dims = dimensions or list(PCDimension)
    suite = TestSuite(task_id=task_id, seed_episode_id=seed_episode_id)

    for dim in dims:
        if dim == PCDimension.PC5B_MASS_DISTRIBUTION:
            continue  # handled below with paired embodiments
        base_params = DEFAULT_PARAMS[dim]
        for spec in build_severity_sweep(dim, base_params, severities):
            suite.cases.append(
                TestCase(
                    task_id=task_id,
                    seed_episode_id=seed_episode_id,
                    dimension=dim,
                    perturbation=spec,
                )
            )

    # PC5-b needs paired cases across embodiments for cross-embodiment MR9.
    if PCDimension.PC5B_MASS_DISTRIBUTION in dims:
        embodiments = DEFAULT_PARAMS[PCDimension.PC5B_MASS_DISTRIBUTION]["embodiments"]
        for emb in embodiments:
            for sev in severities:
                spec = PerturbationSpec(
                    dimension=PCDimension.PC5B_MASS_DISTRIBUTION,
                    severity=sev,
                    params={**DEFAULT_PARAMS[PCDimension.PC5B_MASS_DISTRIBUTION], "magnitude": sev.magnitude},
                )
                suite.cases.append(
                    TestCase(
                        task_id=task_id,
                        seed_episode_id=seed_episode_id,
                        dimension=PCDimension.PC5B_MASS_DISTRIBUTION,
                        perturbation=spec,
                        embodiment=emb,
                    )
                )

    return suite
