"""
Agents: gather multiple resources, prefer pantry when carrying food, craft a simple tool,
occasionally build roads/caches/beacons, use caches as intermediate depots, and broadcast a tiny message vector.
Still no scripted jobs; behaviors stem from local state + world affordances.
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
    tool_level: int = 0  # boosts efficiency a bit

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

        # neighbor bias (proximity signal)
        msg_bias = 0.0
        if inbox:
            m = np.stack(inbox).mean(0)
            msg_bias = float(m.mean()) - 0.5

        # pantry homing when carrying food
        toward_pantry = None
        if self.inventory["food"] > 0 and world.pantry is not None:
            ay, ax = self.pos
            toward_pantry = (int(np.sign(world.pantry[0] - ay)), int(np.sign(world.pantry[1] - ax)))

        # find nearest of a type in local window
        def nearest_of(val: int):
            ys, xs = np.where(mats == val)
            if len(ys) == 0:
                return None
            idx = np.argmin(np.abs(ys - cy) + np.abs(xs - cx))
            ty, tx = int(ys[idx]), int(xs[idx])
            return (int(np.sign(ty - cy)), int(np.sign(tx - cx)))

        # priorities:
        # 1) if on food, harvest
        on_food = (mats[cy, cx] == 1)

        # 2) if carrying food, go home most of the time
        if toward_pantry is not None and np.random.rand() < 0.85:
            dy, dx = toward_pantry
        else:
            move = None
            # 3) if no tool, seek wood/fiber to craft
            if self.tool_level == 0:
                move = nearest_of(4) or nearest_of(5)
            # 4) else prefer food if visible
            if move is None:
                move = nearest_of(1)
            # 5) otherwise explore, with tiny anti-crowding bias
            if move is None or np.random.rand() < (0.15 + 0.2 * max(0.0, -msg_bias)):
                move = [(0,1),(0,-1),(1,0),(-1,0)][np.random.randint(0,4)]
            dy, dx = move

        # harvest/use if standing or staying
        use = 1 if on_food or (dy == 0 and dx == 0) else 0

        # broadcast (placeholder channel)
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
            if tile in (1, 4, 5, 6):  # food/wood/fiber/stone
                gained = world.harvest(self.pos, kind=tile)
                if gained > 0:
                    if tile == 1:
                        self.inventory["food"] += gained
                        self.energy = min(self.energy + (0.5 + 0.1 * self.tool_level), 5.0)
                    elif tile == 4:
                        self.inventory["wood"] += gained
                    elif tile == 5:
                        self.inventory["fiber"] += gained
                    elif tile == 6:
                        self.inventory["stone"] += gained

        # CRAFT simple tool
        if self.tool_level == 0 and self.inventory["wood"] >= 2 and self.inventory["fiber"] >= 1:
            self.inventory["wood"] -= 2
            self.inventory["fiber"] -= 1
            self.tool_level = 1

        # BUILD occasionally (emergent world-shaping)
        if world.materials[y, x] == 0:
            r = np.random.rand()
            if self.inventory["stone"] >= 1 and r < 0.03:
                if world.place_road(y, x):
                    self.inventory["stone"] -= 1
            elif self.inventory["wood"] >= 3 and r < 0.05:
                if world.place_cache(y, x):
                    # start the cache with some of our wood turned into food? no — just empty;
                    # but if we have food, deposit to cache below.
                    pass
            elif self.inventory["fiber"] >= 3 and r < 0.02:
                if world.place_beacon(y, x):
                    self.inventory["fiber"] -= 3

        # CACHE interaction: deposit if carrying food; withdraw if empty and cache has stock
        if world.materials[y, x] == 8:
            if self.inventory["food"] > 0:
                world.cache_deposit(y, x, self.inventory["food"])
                self.inventory["food"] = 0
            elif self.inventory["food"] == 0 and world.caches.get((y, x), 0) > 0:
                taken = world.cache_take(y, x, 1)
                self.inventory["food"] += taken

        # PANTRY deposit (and rest bonus)
        if world.materials[y, x] == 3 and self.inventory["food"] > 0:
            world.shared_store += self.inventory["food"]
            self.inventory["food"] = 0
            self.energy = min(self.energy + 0.2, 5.0)

        # ENERGY drain (roads are cheaper)
        move_cost = 0.005
        if tile == 7:  # road
            move_cost *= 0.6
        self.energy -= move_cost

        # remember last message
        self.last_msg = decision["msg"]
