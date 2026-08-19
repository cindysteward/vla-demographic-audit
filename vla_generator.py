"""
The real generator, needs a CUDA GPU and huggingface.co access, run on
Kaggle, not locally. Verified against source of both simpler-env/SimplerEnv
and DelinQu/SimplerEnv-OpenVLA.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

TRAJ_DIR = Path("./trajectories")
TRAJ_DIR.mkdir(exist_ok=True)

_LAYOUT_RE = re.compile(r"^\[\[LAYOUT:(\d+)\]\](.*)$")

_MODEL = None
_ENV_CACHE = {}


def _get_model():
    global _MODEL
    if _MODEL is None:
        from simpler_env.policies.openvla.openvla_model import OpenVLAInference
        _MODEL = OpenVLAInference(saved_model_path="openvla/openvla-7b", policy_setup="google_robot")
    return _MODEL


def _get_env(task_name: str):
    if task_name not in _ENV_CACHE:
        import simpler_env
        _ENV_CACHE[task_name] = simpler_env.make(task_name)
    return _ENV_CACHE[task_name]


def run_one_trial(prompt: str, trial_id: str, task_name: str = "google_robot_move_near") -> str:
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    match = _LAYOUT_RE.match(prompt)
    if not match:
        raise ValueError(f"prompt did not carry a [[LAYOUT:n]] tag: {prompt!r}")
    seed = int(match.group(1))
    recipient_clause = match.group(2)

    env = _get_env(task_name)
    model = _get_model()

    obs, reset_info = env.reset(seed=seed)
    base_instruction = env.get_language_instruction()
    instruction = base_instruction + recipient_clause

    model.reset(instruction)
    image = get_image_from_maniskill2_obs_dict(env, obs)

    world_vectors, rot_axangles, grippers = [], [], []
    predicted_terminated, success, truncated = False, False, False
    n_steps = 0
    max_steps = 80

    while not (predicted_terminated or truncated) and n_steps < max_steps:
        raw_action, action = model.step(image, instruction)
        predicted_terminated = bool(action["terminate_episode"][0] > 0)
        world_vectors.append(action["world_vector"])
        rot_axangles.append(action["rot_axangle"])
        grippers.append(action["gripper"])

        obs, reward, success, truncated, info = env.step(
            np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
        )
        image = get_image_from_maniskill2_obs_dict(env, obs)
        n_steps += 1

    traj_path = TRAJ_DIR / f"{trial_id}.npz"
    np.savez(traj_path, world_vector=np.array(world_vectors), rot_axangle=np.array(rot_axangles), gripper=np.array(grippers))

    return json.dumps({
        "trial_id": trial_id, "seed": seed, "recipient_clause": recipient_clause,
        "base_instruction": base_instruction, "success": bool(success),
        "n_steps": n_steps, "traj_path": str(traj_path),
    })


def make_generator(task_name: str = "google_robot_move_near"):
    def _generator(prompt: str) -> str:
        match = _LAYOUT_RE.match(prompt)
        seed, clause = match.group(1), match.group(2).strip().replace(" ", "_") or "none"
        trial_id = f"L{seed}_{clause}"
        return run_one_trial(prompt, trial_id, task_name)
    return _generator
