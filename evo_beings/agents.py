"""
Agents module: beings with minimal drives.
No scripted jobs—only movement, harvest, and intrinsic bookkeeping.
"""
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Set
import numpy as np
from .world import World

@dataclass
class Agent:
    """
    Minimal agent:
      - pos: location on the grid
      - energy: internal store
      - inventory: resources carried
      - visited: set of tiles for exploration measurement
    Decision policy is intentionally simple for MVP; evolution will prefer useful tendencies.
    """
    pos: Tuple[int, int]
    energy: float = 3.0
    inventory: int = 0
    visited: Set[Tuple[int, int]] = field(default_factory=set)

    def act(self, world: World) -> Dict[str, Any]:
        obs = world.sense(self.pos, radius=2)
        decision = self._policy(obs)
        self._apply(decision, world)
        return decision

    def _policy(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        mats = obs["materials"]
        center = (mats.shape[0] // 2, mats.shape[1] // 2)
        dy, dx = 0, 0
        ys, xs = np.where(mats == 1)
        if len(ys) > 0:
            idx = np.argmin(np.abs(ys - center[0]) + np.abs(xs - center[1]))
            ty, tx = ys[idx], xs[idx]
            dy = int(np.sign(ty - center[0]))
            dx = int(np.sign(tx - center[1]))
        else:
            choices = [(0,1),(0,-1),(1,0),(-1,0)]
            dy, dx = choices[np.random.randint(0, len(choices))]
        use = 1 if (dy == 0 and dx == 0) else 0
        return {"move": (dy, dx), "use": use}

    def _apply(self, decision: Dict[str, Any], world: World) -> None:
        dy, dx = decision["move"]
        ny, nx = world.in_bounds(self.pos[0] + dy, self.pos[1] + dx)
        self.pos = (ny, nx)
        self.visited.add(self.pos)
        if decision["use"] == 1:
            self.inventory += world.harvest(self.pos, kind=1)
            self.energy += 0.2
        self.energy -= 0.01
