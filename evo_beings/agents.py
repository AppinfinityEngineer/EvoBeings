"""
Agents module: beings with minimal drives.
No scripted jobs—only movement, harvest, and intrinsic bookkeeping.
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

    def act(self, world: "World", inbox: List[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform one action.
        inbox: list of neighbor messages, or None for single-agent runs.
        """
        if inbox is None:
            inbox = []
        obs = world.sense(self.pos, radius=2)
        decision = self._policy(obs, inbox)
        self._apply(decision, world)
        return decision

    def _policy(self, obs: Dict[str, Any], inbox: List[np.ndarray]) -> Dict[str, Any]:
        mats = obs["materials"]
        center = (mats.shape[0] // 2, mats.shape[1] // 2)

        # very small influence from messages: bias explore vs. stay (unscripted semantics)
        msg_bias = 0.0
        if inbox:
            m = np.stack(inbox).mean(0)               # aggregate neighbor signal
            msg_bias = float(m.mean()) - 0.5          # -0.5..+0.5

        ys, xs = np.where(mats == 1)
        if len(ys) > 0 and np.random.rand() > (0.2 + 0.5*max(0.0, -msg_bias)):
            # move toward nearest resource (unless messages bias pausing)
            center = (mats.shape[0] // 2, mats.shape[1] // 2)
            idx = np.argmin(np.abs(ys - center[0]) + np.abs(xs - center[1]))
            ty, tx = ys[idx], xs[idx]
            dy = int(np.sign(ty - center[0])); dx = int(np.sign(tx - center[1]))
        else:
            dy, dx = [(0,1),(0,-1),(1,0),(-1,0)][np.random.randint(0,4)]

        use = 1 if (dy == 0 and dx == 0) else 0

        # emit a tiny random message with slight coupling to local materials density
        dens = float((mats == 1).mean())
        msg = np.clip(np.random.rand(4) * (0.5 + dens), 0, 1).astype(np.float32)
        return {"move": (dy, dx), "use": use, "msg": msg}

    def _apply(self, decision: Dict[str, Any], world: "World") -> None:
        dy, dx = decision["move"]
        ny, nx = world.in_bounds(self.pos[0] + dy, self.pos[1] + dx)
        self.pos = (ny, nx)
        self.visited.add(self.pos)
        if decision["use"] == 1:
            self.inventory += world.harvest(self.pos, kind=1)
            self.energy += 0.2
        self.energy -= 0.01
        self.last_msg = decision["msg"]