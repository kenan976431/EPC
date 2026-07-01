"""
perturbations.py

Implements the physical perturbation operator T_p : E -> E' (Section II-B)
across the nine EPC dimensions from Section III-B:

  PC1-a Illuminance Intensity   PC1-b Surface Texture
  PC2   Perspective Shift
  PC3-a Occlusion Ratio         PC3-b Spatial Depth
  PC4-a Contact Friction        PC4-b Structural Rigidity
  PC5-a Mass Magnitude          PC5-b Mass Distribution

IMPORTANT / integration boundary:
EBench's public client (`genmanip_client.eval_client.EvalClient`) exposes a
black-box obs/action loop only -- it does not expose scene authoring. Scene
mutation (lights, materials, friction, mass, camera pose) happens on the
Isaac Sim / USD stage inside the GenManip server process. `SceneMutator`
below defines the seam: on the server side (or via a companion RPC you add
to `ray_eval_server.py`), implement each `_apply_*` method against the USD
stage using `pxr.UsdPhysics`, `pxr.UsdLux`, `pxr.UsdShade`, and
`omni.replicator` APIs. Everything above this seam (test generation, oracle,
metrics) is stage-agnostic and works against any backend that honors this
interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class Severity(Enum):
    SLIGHT = "slight"      # <=10%
    MODERATE = "moderate"  # 10-30%
    SEVERE = "severe"      # >30%

    @property
    def magnitude(self) -> float:
        return {"slight": 0.10, "moderate": 0.25, "severe": 0.45}[self.value]


class PCDimension(Enum):
    PC1A_ILLUMINANCE = "PC1-a"
    PC1B_TEXTURE = "PC1-b"
    PC2_PERSPECTIVE = "PC2"
    PC3A_OCCLUSION = "PC3-a"
    PC3B_DEPTH = "PC3-b"
    PC4A_FRICTION = "PC4-a"
    PC4B_RIGIDITY = "PC4-b"
    PC5A_MASS_MAGNITUDE = "PC5-a"
    PC5B_MASS_DISTRIBUTION = "PC5-b"


# Relation pattern each dimension induces (Section III-B), used by the oracle
# to select which physical consistency constraint Phi_p to check.
RELATION_PATTERN = {
    PCDimension.PC1A_ILLUMINANCE: "monotonic_degradation",
    PCDimension.PC1B_TEXTURE: "invariance",
    PCDimension.PC2_PERSPECTIVE: "invariance",
    PCDimension.PC3A_OCCLUSION: "monotonic_degradation",
    PCDimension.PC3B_DEPTH: "invariance",
    PCDimension.PC4A_FRICTION: "boundary_stability",
    PCDimension.PC4B_RIGIDITY: "boundary_stability",
    PCDimension.PC5A_MASS_MAGNITUDE: "monotonic_degradation",
    PCDimension.PC5B_MASS_DISTRIBUTION: "cross_embodiment_consistency",
}


@dataclass
class PerturbationSpec:
    """A single physical perturbation operator delta_k(.; lambda)."""

    dimension: PCDimension
    severity: Severity
    params: Dict[str, Any]

    @property
    def relation_pattern(self) -> str:
        return RELATION_PATTERN[self.dimension]


class SceneMutator:
    """
    Server-side seam for applying PerturbationSpec to an Isaac Sim / USD
    scene while preserving task semantics (per Section III's definition of
    T_p). Each _apply_* method is a stub describing the intended USD/Isaac
    Sim call; wire these into your GenManip server fork's scene-setup hook
    (e.g. before `client.reset()` on the server side) or expose them over
    an admin RPC alongside the eval server.
    """

    def __init__(self, stage=None, physics_scene=None):
        # `stage` would be a pxr.Usd.Stage handle obtained inside the
        # Isaac Sim process; kept optional so this module imports cleanly
        # outside of Isaac Sim (e.g. for unit tests / offline planning).
        self.stage = stage
        self.physics_scene = physics_scene

    def apply(self, spec: PerturbationSpec) -> None:
        dispatch = {
            PCDimension.PC1A_ILLUMINANCE: self._apply_illuminance,
            PCDimension.PC1B_TEXTURE: self._apply_texture,
            PCDimension.PC2_PERSPECTIVE: self._apply_perspective_shift,
            PCDimension.PC3A_OCCLUSION: self._apply_occlusion,
            PCDimension.PC3B_DEPTH: self._apply_depth_shift,
            PCDimension.PC4A_FRICTION: self._apply_friction,
            PCDimension.PC4B_RIGIDITY: self._apply_rigidity,
            PCDimension.PC5A_MASS_MAGNITUDE: self._apply_mass_magnitude,
            PCDimension.PC5B_MASS_DISTRIBUTION: self._apply_mass_distribution,
        }
        dispatch[spec.dimension](spec)

    # -- PC1: photometric --------------------------------------------------
    def _apply_illuminance(self, spec: PerturbationSpec) -> None:
        """Scale UsdLux intensity on the dome/key lights by -magnitude."""
        # e.g. light_prim.GetAttribute("inputs:intensity").Set(base * (1 - m))
        self._require_stage("illuminance")

    def _apply_texture(self, spec: PerturbationSpec) -> None:
        """Swap UsdShade material bindings on target prims (retexture only;
        keep collision geometry / mass unchanged -> preserves invariance)."""
        self._require_stage("texture")

    # -- PC2: viewpoint -----------------------------------------------------
    def _apply_perspective_shift(self, spec: PerturbationSpec) -> None:
        """Perturb camera prim's local transform (pos/orient) by a bounded
        delta that keeps the manipulation target observable."""
        self._require_stage("perspective")

    # -- PC3: geometric occlusion / depth ------------------------------------
    def _apply_occlusion(self, spec: PerturbationSpec) -> None:
        """Insert/scale an occluder prim between camera and target so the
        projected occlusion ratio matches spec.params['occlusion_ratio']."""
        self._require_stage("occlusion")

    def _apply_depth_shift(self, spec: PerturbationSpec) -> None:
        """Translate target object along the camera viewing axis by
        spec.params['depth_delta_m'], keeping reachability class fixed."""
        self._require_stage("depth")

    # -- PC4: mechanical ------------------------------------------------------
    def _apply_friction(self, spec: PerturbationSpec) -> None:
        """Set UsdPhysics.MaterialAPI static/dynamic friction coefficients
        on the manipulated object's physics material."""
        self._require_stage("friction")

    def _apply_rigidity(self, spec: PerturbationSpec) -> None:
        """Adjust deformable/soft-body stiffness (or switch rigid<->soft
        body schema) via PhysxSchema deformable body properties."""
        self._require_stage("rigidity")

    # -- PC5: dynamic mass-inertia ----------------------------------------------
    def _apply_mass_magnitude(self, spec: PerturbationSpec) -> None:
        """Scale UsdPhysics.MassAPI mass attribute on the target rigid body."""
        self._require_stage("mass magnitude")

    def _apply_mass_distribution(self, spec: PerturbationSpec) -> None:
        """Redistribute mass across sub-links (or shift center of mass) via
        MassAPI centerOfMass / diagonalInertia while holding total mass fixed."""
        self._require_stage("mass distribution")

    def _require_stage(self, what: str) -> None:
        if self.stage is None:
            raise RuntimeError(
                f"SceneMutator has no USD stage bound; cannot apply {what} "
                "perturbation. Run inside the Isaac Sim / GenManip server "
                "process, or supply a stage handle."
            )


def build_severity_sweep(
    dimension: PCDimension,
    base_params: Dict[str, Any],
    severities=(Severity.SLIGHT, Severity.MODERATE, Severity.SEVERE),
) -> list:
    """Generate a severity sweep for one dimension, matching the paper's
    five-level (here simplified to three-tier) severity design (Table I)."""
    specs = []
    for sev in severities:
        params = dict(base_params)
        params["magnitude"] = sev.magnitude
        specs.append(PerturbationSpec(dimension=dimension, severity=sev, params=params))
    return specs
