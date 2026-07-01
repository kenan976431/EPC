"""
behavior.py

Implements B(E) = (S, Pi, Gamma) from Section II-B of the paper:
  S     -> task outcome (success / failure, or fractional score)
  Pi    -> decision / action sequence
  Gamma -> execution trajectory (end-effector poses over time)

BehaviorRecorder drives an EBench EvalClient episode end-to-end and returns
a structured Behavior object that downstream oracle code can compare across
source / follow-up executions.
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Behavior:
    """Structured behavior observation B(E) = (S, Pi, Gamma)."""

    outcome: float                                  # S(theta): 0/1 or fractional score
    actions: List[np.ndarray] = field(default_factory=list)     # Pi: raw action vectors
    ee_trajectory: List[np.ndarray] = field(default_factory=list)  # Gamma: (x,y,z) EE positions
    ee_quat_trajectory: List[np.ndarray] = field(default_factory=list)
    step_count: int = 0
    wall_time_s: float = 0.0
    energy_proxy: float = 0.0                        # sum of |delta action|, cheap energy proxy
    metadata: Dict[str, Any] = field(default_factory=dict)

    def cost(self) -> float:
        """
        Execution efficiency C(theta) (Section III-A).
        Combines step count and an energy proxy so that longer / jerkier
        trajectories are penalized, matching the paper's "step count, time,
        or energy" definition.
        """
        return float(self.step_count) + 0.1 * self.energy_proxy

    def trajectory_array(self) -> np.ndarray:
        if not self.ee_trajectory:
            return np.zeros((0, 3), dtype=np.float32)
        return np.stack(self.ee_trajectory, axis=0)


class BehaviorRecorder:
    """
    Wraps genmanip_client.eval_client.EvalClient to record a full episode
    into a Behavior object, using the documented obs/action schema:

        obs[worker_id]["obs"] = {
            "instruction": str,
            "state.joints": np.ndarray (12,),
            "state.gripper": np.ndarray (4,),
            "state.base": np.ndarray (3,),
            "state.ee_pose": [[pos, quat], [pos, quat]],  # left/right EE
            "video.{camera}_view": np.ndarray (H, W, 3) uint8,
            "timestep": int,
            "reset": bool,
        }

    A policy_fn(obs) -> action_dict (or action_chunk list) supplies the
    controller under test; it is treated as a black box, consistent with
    the paper's claim that EPC applies to black-box embodied agents.
    """

    def __init__(self, client, worker_id: str, chunk_mode: bool = True):
        self.client = client
        self.worker_id = worker_id
        self.chunk_mode = chunk_mode

    def run_episode(
        self,
        policy_fn: Callable[[Dict[str, Any]], Any],
        reset_fn: Optional[Callable[[], None]] = None,
        max_steps: int = 500,
    ) -> Behavior:
        t0 = time.time()
        obs = self.client.reset()
        if reset_fn is not None:
            reset_fn()

        behavior = Behavior(outcome=0.0)
        prev_action_vec: Optional[np.ndarray] = None
        done = False
        step = 0

        while not done and step < max_steps:
            worker_obs = obs[self.worker_id]["obs"]

            if worker_obs.get("reset", False) and reset_fn is not None and step > 0:
                # Server switched episodes mid-loop (shouldn't happen for a
                # single-episode recorder, but guard against it anyway).
                reset_fn()

            self._log_state(behavior, worker_obs, prev_action_vec)

            action = policy_fn(obs)

            if self.chunk_mode and isinstance(action, list):
                obs, done = self.client.step(action)
                step += len(action)
            else:
                obs, done = self.client.step(action)
                step += 1

            act_dict = action[0] if isinstance(action, list) else action
            prev_action_vec = self._action_to_vector(act_dict.get(self.worker_id, {}))
            behavior.actions.append(prev_action_vec)

        # Final outcome comes from server-reported episode result if present,
        # otherwise fall back to the last observation's success flag.
        final_obs = obs[self.worker_id]
        behavior.outcome = float(final_obs.get("success", final_obs.get("score", 0.0)))
        behavior.step_count = step
        behavior.wall_time_s = time.time() - t0
        return behavior

    @staticmethod
    def _action_to_vector(action_entry: Dict[str, Any]) -> np.ndarray:
        arm = np.asarray(action_entry.get("action", np.zeros(16)), dtype=np.float32)
        base = np.asarray(action_entry.get("base_motion", np.zeros(3)), dtype=np.float32)
        return np.concatenate([arm, base])

    @staticmethod
    def _log_state(behavior: Behavior, worker_obs: Dict[str, Any], prev_action_vec) -> None:
        ee_pose = worker_obs.get("state.ee_pose")
        if ee_pose:
            # Track the right-hand (or single) end-effector by convention;
            # extend to both arms if bimanual trajectory deviation is needed.
            pos, quat = ee_pose[0]
            behavior.ee_trajectory.append(np.asarray(pos, dtype=np.float32))
            behavior.ee_quat_trajectory.append(np.asarray(quat, dtype=np.float32))
        if prev_action_vec is not None:
            behavior.energy_proxy += float(np.abs(prev_action_vec).sum())
