"""
Agents: simple beings that move, harvest, deposit at pantry, and broadcast a tiny message vector.
No scripted jobs; pantry return is a bias when carrying food.
"""
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Set, List
import numpy as np
from .world import World

@dataclass
class Agent:
    pos: Tuple[int, int]
    energy: float = 3.0
    inventory: int = 0
    visited: Set[Tuple[int, int]] = field(default_factory=set)
    last_msg: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))

    def act(self, world: World, inbox: List[np.ndarray] = None) -> Dict[str, Any]:
        if inbox is None:
            inbox = []
        obs = world.sense(self.pos, radius=2)
        decision = self._policy(world, obs, inbox)
        self._apply(decision, world)
        return decision

    def _policy(self, world: World, obs: Dict[str, Any], inbox: List[np.ndarray]) -> Dict[str, Any]:
        mats = obs["materials"]
        h, w = mats.shape
        cy, cx = h // 2, w // 2

        # tiny neighbor bias
        msg_bias = 0.0
        if inbox:
            m = np.stack(inbox).mean(0)
            msg_bias = float(m.mean()) - 0.5  # [-0.5, +0.5]

        # pantry homing when carrying
        toward_pantry = None
        if self.inventory > 0 and world.pantry is not None:
            # in local obs coords, our tile is center; we only need direction
            py, px = world.pantry
            # approximate direction by comparing absolute positions
            dy = int(np.sign(py - (py - (py - cy))))  # keep as unit step
            dx = int(np.sign(px - (px - (px - cx))))
            # simpler: move one step toward absolute pantry by comparing world coords
            # (dy, dx) will be recomputed below using our absolute pos
            ay, ax = self.pos
            dy = int(np.sign(world.pantry[0] - ay))
            dx = int(np.sign(world.pantry[1] - ax))
            toward_pantry = (dy, dx)

        ys, xs = np.where(mats == 1)  # look for nearby food
        move_decided = False
        dy = dx = 0

        if toward_pantry is not None and np.random.rand() < 0.8:
            dy, dx = toward_pantry
            move_decided = True

        if not move_decided:
            if len(ys) > 0 and np.random.rand() > (0.25 + 0.5 * max(0.0, -msg_bias)):
                idx = np.argmin(np.abs(ys - cy) + np.abs(xs - cx))
                ty, tx = int(ys[idx]), int(xs[idx])
                dy = int(np.sign(ty - cy)); dx = int(np.sign(tx - cx))
            else:
                dy, dx = [(0,1),(0,-1),(1,0),(-1,0)][np.random.randint(0,4)]

        on_food = (mats[h//2, w//2] == 1)
        use = 1 if on_food or (dy == 0 and dx == 0) else 0

        dens = float((mats == 1).mean())
        msg = np.clip(np.random.rand(4) * (0.5 + dens), 0, 1).astype(np.float32)
        return {"move": (dy, dx), "use": use, "msg": msg}

    def _apply(self, decision: Dict[str, Any], world: World) -> None:
        dy, dx = decision["move"]
        ny, nx = world.in_bounds(self.pos[0] + dy, self.pos[1] + dx)
        self.pos = (ny, nx)
        self.visited.add(self.pos)

        # harvest if staying
        if decision["use"] == 1:
         gained = world.harvest(self.pos, kind=1)
         if gained > 0:
            self.inventory += gained
            self.energy = min(self.energy + 0.5, 5.0)  # better refuel, capped

        # deposit at pantry
        y, x = self.pos
        if world.materials[y, x] == 3 and self.inventory > 0:
            world.shared_store += self.inventory
            self.inventory = 0
            self.energy = min(self.energy + 0.2, 5.0)  # rest at base

        self.energy -= 0.005
        self.last_msg = decision["msg"]
