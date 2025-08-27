"""
Agents: online learners. They haul food home, reinforce the path they used,
prefer roads (and learn to prefer them more), and build roads when desire is high.
They can still craft tools, use caches, and drop beacons.

No scripted jobs; preferences evolve from experience.
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
    tool_level: int = 0                # boosts efficiency a bit
    road_bias: float = 0.1             # learned preference for stepping onto roads
    carry_path: List[Tuple[int, int]] = field(default_factory=list)  # path since picking up food
    knowledge: Dict[str, int] = field(default_factory=lambda: {"deposits": 0, "roads_built": 0, "caches_used": 0})

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

        # tiny neighbor bias (social proximity)
        msg_bias = 0.0
        if inbox:
            m = np.stack(inbox).mean(0)
            msg_bias = float(m.mean()) - 0.5  # [-0.5,+0.5]

        # Pantry homing when carrying
        carrying_food = self.inventory["food"] > 0
        dy = dx = 0

        # Candidate moves (4-neighborhood)
        moves = [(0,1),(0,-1),(1,0),(-1,0)]
        rng = np.random.random()

        def score_move(dy: int, dx: int) -> float:
            # estimate: prefer moving closer to pantry when carrying; otherwise explore/seek resources
            ay, ax = self.pos
            ny, nx = ay + dy, ax + dx
            ny, nx = world.in_bounds(ny, nx)
            # pantry distance
            dist_term = 0.0
            if world.pantry is not None:
                py, px = world.pantry
                d_now = abs(py - ay) + abs(px - ax)
                d_next = abs(py - ny) + abs(px - nx)
                dist_term = float(d_now - d_next)  # positive if closer
            # road bonus if stepping onto a road, scaled by learned bias
            tile_next = world.materials[ny, nx]
            road_bonus = self.road_bias if tile_next == 7 else 0.0
            # small random exploration & anti-crowding from messages
            jitter = (np.random.rand() - 0.5) * 0.1 + 0.05 * max(0.0, -msg_bias)
            if carrying_food:
                return 1.0 * dist_term + 1.0 * road_bonus + jitter
            else:
                # when not carrying, prefer visible food; else small wander
                # look for food in local obs and bias toward its direction
                ys, xs = np.where(mats == 1)
                food_term = 0.0
                if len(ys) > 0:
                    idx = np.argmin(np.abs(ys - cy) + np.abs(xs - cx))
                    ty, tx = int(ys[idx]), int(xs[idx])
                    dy_to = int(np.sign(ty - cy)); dx_to = int(np.sign(tx - cx))
                    if dy_to == dy and dx_to == dx:
                        food_term = 1.0
                return 0.6 * food_term + 0.2 * road_bonus + jitter

        # choose best move among the 4
        scored = [(score_move(my, mx), (my, mx)) for (my, mx) in moves]
        scored.sort(reverse=True, key=lambda t: t[0])
        dy, dx = scored[0][1]

        # harvest/use if standing or on food
        on_food = (mats[cy, cx] == 1)
        use = 1 if on_food or (dy == 0 and dx == 0) else 0

        # broadcast placeholder
        dens = float((mats == 1).mean())
        msg = np.clip(np.random.rand(4) * (0.5 + dens), 0, 1).astype(np.float32)
        return {"move": (dy, dx), "use": use, "msg": msg}

    # ---------------------------------------------------------------------

    def _apply(self, decision: Dict[str, Any], world: World) -> None:
        dy, dx = decision["move"]
        ay, ax = self.pos
        ny, nx = world.in_bounds(ay + dy, ax + dx)
        self.pos = (ny, nx)
        self.visited.add(self.pos)
        y, x = self.pos
        tile = int(world.materials[y, x])

        # HARVEST
        just_picked_food = False
        if decision["use"] == 1 and tile in (1, 4, 5, 6):
            gained = world.harvest(self.pos, kind=tile)
            if gained > 0:
                if tile == 1:
                    before = self.inventory["food"]
                    self.inventory["food"] += gained
                    just_picked_food = (before == 0)
                    self.energy = min(self.energy + (0.5 + 0.1 * self.tool_level), 5.0)
                elif tile == 4:
                    self.inventory["wood"] += gained
                elif tile == 5:
                    self.inventory["fiber"] += gained
                elif tile == 6:
                    self.inventory["stone"] += gained

        # start tracking carry path when we first pick food
        if just_picked_food:
            self.carry_path = [self.pos]
        elif self.inventory["food"] > 0:
            # continue tracking while carrying
            self.carry_path.append(self.pos)

        # CRAFT simple tool
        if self.tool_level == 0 and self.inventory["wood"] >= 2 and self.inventory["fiber"] >= 1:
            self.inventory["wood"] -= 2
            self.inventory["fiber"] -= 1
            self.tool_level = 1

        # BUILD ROADS where desire is high
        if world.materials[y, x] == 0:
            desire = world.road_desire[y, x]
            if desire > 3.0 and (self.inventory["stone"] >= 1 or self.inventory["wood"] >= 2):
                if world.place_road(y, x):
                    if self.inventory["stone"] >= 1:
                        self.inventory["stone"] -= 1
                    else:
                        self.inventory["wood"] -= 2
                    self.knowledge["roads_built"] += 1
            else:
                # occasional caches/beacons still possible
                r = np.random.rand()
                if self.inventory["wood"] >= 3 and r < 0.01:
                    world.place_cache(y, x)
                elif self.inventory["fiber"] >= 3 and r < 0.005:
                    world.place_beacon(y, x); self.inventory["fiber"] -= 3

        # CACHE interaction: simple in/out
        if world.materials[y, x] == 8:
            if self.inventory["food"] > 0:
                world.cache_deposit(y, x, self.inventory["food"])
                self.inventory["food"] = 0
                self.knowledge["caches_used"] += 1
            elif self.inventory["food"] == 0 and world.caches.get((y, x), 0) > 0:
                taken = world.cache_take(y, x, 1)
                self.inventory["food"] += taken

        # PANTRY deposit (reinforce learning)
        if world.materials[y, x] == 3 and self.inventory["food"] > 0:
            world.shared_store += self.inventory["food"]
            self.inventory["food"] = 0
            self.energy = min(self.energy + 0.2, 5.0)
            self.knowledge["deposits"] += 1

            # learning: reinforce the path we used while carrying
            if self.carry_path:
                world.reinforce_path(self.carry_path, amount=1.0)
                # update personal road preference toward fraction of steps on roads
                on_road = 0
                for (py, px) in self.carry_path:
                    if world.materials[py, px] == 7:
                        on_road += 1
                frac = on_road / max(1, len(self.carry_path))
                alpha = 0.2
                self.road_bias = float((1 - alpha) * self.road_bias + alpha * frac)
                self.carry_path = []

        # ENERGY drain (roads are cheaper)
        move_cost = 0.005
        if tile == 7:  # road
            move_cost *= 0.6
        self.energy -= move_cost

        # remember last message
        self.last_msg = decision["msg"]
