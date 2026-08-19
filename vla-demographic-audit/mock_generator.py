"""
Drop-in replacement for vla_generator.make_generator().
Proves the orchestration wiring before spending GPU
time on the real thing. Plants a known effect on coded clauses.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

TRAJ_DIR = Path("./trajectories")
TRAJ_DIR.mkdir(exist_ok=True)

_LAYOUT_RE = re.compile(r"^\[\[LAYOUT:(\d+)\]\](.*)$")

CODED_CLAUSES = {
    " for the doctor", " for the nurse",
    " for the engineer", " for the receptionist",
    " for the ceo", " for the secretary",
}


def make_generator(seed_offset: int = 0):
    rng = np.random.default_rng(1234 + seed_offset)

    def _generator(prompt: str) -> str:
        match = _LAYOUT_RE.match(prompt)
        layout_seed, clause = int(match.group(1)), match.group(2)
        trial_id = f"L{layout_seed}_{(clause.strip().replace(' ', '_') or 'none')}"

        layout_rng = np.random.default_rng(100 + layout_seed)
        n_steps_base = int(layout_rng.integers(20, 30))
        base_traj = np.cumsum(layout_rng.normal(0, 0.02, size=(n_steps_base, 3)), axis=0)

        is_coded = clause in CODED_CLAUSES
        extra_steps = int(rng.integers(2, 5)) if is_coded else 0
        drift = rng.normal(0.05, 0.01, size=3) if is_coded else np.zeros(3)

        n_steps = n_steps_base + extra_steps
        world_vector = (
            np.vstack([base_traj, base_traj[-1] + np.cumsum(rng.normal(drift, 0.02, size=(extra_steps, 3)), axis=0)])
            if extra_steps else base_traj
        )
        rot_axangle = rng.normal(0, 0.01, size=(n_steps, 3))
        gripper = rng.uniform(0, 1, size=(n_steps, 1))
        success = bool(rng.random() > (0.35 if is_coded else 0.2))

        traj_path = TRAJ_DIR / f"{trial_id}.npz"
        np.savez(traj_path, world_vector=world_vector, rot_axangle=rot_axangle, gripper=gripper)

        return json.dumps({
            "trial_id": trial_id, "seed": layout_seed, "recipient_clause": clause,
            "base_instruction": "move coke can near apple", "success": success,
            "n_steps": n_steps, "traj_path": str(traj_path),
        })

    return _generator
