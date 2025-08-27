"""
Agents: simple beings that move, harvest multiple resources, craft a small 'tool',
occasionally place structures, deposit at pantry, and broadcast a tiny message vector.
No scripted jobs; pantry return is a bias when carrying food; crafting/building emerge from inventory state.
"""
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Set, List
import numpy as np
from .world import World

@dataclass
class Agent:
    pos: Tuple[int, int]
    energy: float = 3.0
    inventory: Dict[str, int] = field(default_factory=lambda: {"food": 0, "wood": 0, "fiber": 0, "stone": 0})
    visited: Set[Tuple[int, int]] = field(default_factory=set)
    last_msg: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    tool_level: int = 0  # emergent 'profession' support: boosts harvest/efficiency a bit

    def act(self, world: World, inbox: List[np.ndarray] = None) -> Dict[str, Any]:
        if inbox is None:
            inbox = []
        obs = world.sense(self.pos, radius=2)
        decision = self._policy(world, obs, inbox)
        self._apply(decision, world)
        return decision

    # ---------------------------------------------------------------------

    def _policy(self, world: World, obs: Dict[str, Any], inbox: List[np.ndarray]) -> Dict[str, Any]:
        mats = obs["materials"]
        h, w = mats.shape
        cy, cx = h // 2, w // 2

        # tiny neighbor bias (proximity awareness)
        msg_bias = 0.0
        if inbox:
            m = np.stack(inbox).mean(0)
            msg_bias = float(m.mean()) - 0.5  # [-0.5, +0.5]

        # pantry homing when carrying food
        toward_pantry = None
        if self.inventory["food"] > 0 and world.pantry is not None:
            ay, ax = self.pos
            dy = int(np.sign(world.pantry[0] - ay))
            dx = int(np.sign(world.pantry[1] - ax))
            toward_pantry = (dy, dx)

        # Perception: nearest of a given kind within local window
        def nearest_of(val: int):
            ys, xs = np.where(mats == val)
            if len(ys) == 0:
                return None
            idx = np.argmin(np.abs(ys - cy) + np.abs(xs - cx))
            ty, tx = int(ys[idx]), int(xs[idx])
            return (int(np.sign(ty - cy)), int(np.sign(tx - cx)))

        # Goal preferences (unscripted but state-driven):
        # 1) If on food -> harvest now.
        on_food = (mats[cy, cx] == 1)

        # 2) If carrying food -> strongly go to pantry.
        if toward_pantry is not None and np.random.rand() < 0.85:
            dy, dx = toward_pantry
        else:
            # 3) If we lack a tool, prefer harvesting wood/fiber to craft
            move = None
            if self.tool_level == 0:
                move = nearest_of(4) or nearest_of(5)  # tree or fiber
            # 4) Otherwise prefer food if visible
            if move is None:
                move = nearest_of(1)
            # 5) Explore if nothing salient, with slight anti-crowding bias
            if move is None or np.random.rand() < (0.15 + 0.2 * max(0.0, -msg_bias)):
                move = [(0,1),(0,-1),(1,0),(-1,0)][np.random.randint(0,4)]
            dy, dx = move

        # Use (harvest) if standing on something harvestable
        use = 1 if on_food or (dy == 0 and dx == 0) else 0

        # Broadcast: random, nudged by local food density (placeholder channel)
        dens = float((mats == 1).mean())
        msg = np.clip(np.random.rand(4) * (0.5 + dens), 0, 1).astype(np.float32)
        return {"move": (dy, dx), "use": use, "msg": msg}

    # ---------------------------------------------------------------------

    def _apply(self, decision: Dict[str, Any], world: World) -> None:
        dy, dx = decision["move"]
        ny, nx = world.in_bounds(self.pos[0] + dy, self.pos[1] + dx)
        self.pos = (ny, nx)
        self.visited.add(self.pos)

        y, x = self.pos
        tile = int(world.materials[y, x])

        # HARVEST
        if decision["use"] == 1:
            # harvest whatever we are standing on if valid
            if tile in (1, 4, 5, 6):
                gained = world.harvest(self.pos, kind=tile)
                if gained > 0:
                    if tile == 1:
                        self.inventory["food"] += gained
                        # eating/foraging replenishes energy
                        self.energy = min(self.energy + (0.5 + 0.1 * self.tool_level), 5.0)
                    elif tile == 4:
                        self.inventory["wood"] += gained
                    elif tile == 5:
                        self.inventory["fiber"] += gained
                    elif tile == 6:
                        self.inventory["stone"] += gained

        # CRAFT (unscripted trigger: if we have enough mats and no tool)
        if self.tool_level == 0 and self.inventory["wood"] >= 2 and self.inventory["fiber"] >= 1:
            # consume materials to craft a basic tool that increases efficiency slightly
            self.inventory["wood"] -= 2
            self.inventory["fiber"] -= 1
            self.tool_level = 1

        # PLACE STRUCTURE (unscripted marker when carrying lots of raw mats)
        if world.materials[y, x] == 0 and (self.inventory["stone"] >= 1 or self.inventory["wood"] >= 2):
            if np.random.rand() < 0.03:  # rare
                if world.place_structure(y, x):
                    if self.inventory["stone"] >= 1:
                        self.inventory["stone"] -= 1
                    else:
                        self.inventory["wood"] = max(0, self.inventory["wood"] - 2)

        # DEPOSIT at pantry (and rest bonus)
        if world.materials[y, x] == 3 and self.inventory["food"] > 0:
            world.shared_store += self.inventory["food"]
            self.inventory["food"] = 0
            self.energy = min(self.energy + 0.2, 5.0)

        # ENERGY drain
        self.energy -= 0.005

        # remember last message
        self.last_msg = decision["msg"]
