"""
Agents: anti-congestion & learning (with correct auto-harvest).
- Auto-harvest AFTER moving (food/wood/fiber/stone) so carrying state is reliable.
- Avoid pantry zone when NOT carrying (push outward).
- If pantry is crowded when carrying, offload to caches (or place one).
- Learn road preferences from successful long hauls.
- Build roads primarily away from the pantry where desire is high.
"""
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Set, List
import numpy as np
from .world import World

HOME_R = 4          # inside this radius of pantry counts as "home"
CROWD_N = 3         # if >= this many neighbors (from inbox) we treat it as congestion
MIN_PATH_REINF = 6  # only reinforce if haul path length >= this
ROAD_MIN_DIST = 5   # only build roads when at least this far from pantry


@dataclass
class Agent:
    pos: Tuple[int, int]
    energy: float = 3.0
    inventory: Dict[str, int] = field(default_factory=lambda: {"food": 0, "wood": 0, "fiber": 0, "stone": 0})
    visited: Set[Tuple[int, int]] = field(default_factory=set)
    last_msg: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    tool_level: int = 0
    road_bias: float = 0.1
    carry_path: List[Tuple[int, int]] = field(default_factory=list)
    knowledge: Dict[str, int] = field(default_factory=lambda: {"deposits": 0, "roads_built": 0, "caches_used": 0})
    last_inbox_n: int = 0  # track local crowding from the last act()

    def act(self, world: World, inbox: List[np.ndarray] = None) -> Dict[str, Any]:
        if inbox is None:
            inbox = []
        self.last_inbox_n = len(inbox)
        obs = world.sense(self.pos, radius=2)
        decision = self._policy(world, obs, inbox)
        self._apply(decision, world)
        return decision

    # ---------------------------------------------------------------------

    def _policy(self, world: World, obs: Dict[str, Any], inbox: List[np.ndarray]) -> Dict[str, Any]:
        mats = obs["materials"]
        h, w = mats.shape
        cy, cx = h // 2, w // 2

        # social proximity
        msg_bias = 0.0
        if inbox:
            m = np.stack(inbox).mean(0)
            msg_bias = float(m.mean()) - 0.5

        # pantry vector
        ay, ax = self.pos
        if world.pantry is not None:
            py, px = world.pantry
            dist_home = abs(py - ay) + abs(px - ax)
        else:
            dist_home = 999

        carrying = self.inventory["food"] > 0

        # candidate moves
        moves = [(0,1),(0,-1),(1,0),(-1,0)]

        def score_move(dy: int, dx: int) -> float:
            ny, nx = world.in_bounds(ay + dy, ax + dx)
            tile_next = world.materials[ny, nx]

            # distance change to pantry
            d_now = dist_home
            d_next = abs((world.pantry[0] if world.pantry else ay) - ny) + abs((world.pantry[1] if world.pantry else ax) - nx)
            toward_home = float(d_now - d_next)  # positive if moving closer

            # road bonus
            road_bonus = self.road_bias if tile_next == 7 else 0.0

            # exploration / anti-crowding
            jitter = (np.random.rand() - 0.5) * 0.1 + 0.08 * max(0.0, -msg_bias)

            if carrying:
                # prefer moving closer to pantry; roads help too
                return 1.0 * toward_home + 1.0 * road_bonus + jitter
            else:
                # if in home zone, bias outward (away from pantry) and toward non-food resources
                outward = 0.0
                if dist_home <= HOME_R:
                    outward = -toward_home  # positive if moving away

                # prefer wood/fiber/stone if visible in local obs
                ys_f, xs_f = np.where(mats == 1)
                ys_w, xs_w = np.where((mats == 4) | (mats == 5) | (mats == 6))

                toward_food = 0.0
                toward_other = 0.0
                if len(ys_f) > 0:
                    idx = np.argmin(np.abs(ys_f - cy) + np.abs(xs_f - cx))
                    dyf = int(np.sign(int(ys_f[idx]) - cy)); dxf = int(np.sign(int(xs_f[idx]) - cx))
                    if dyf == dy and dxf == dx: toward_food = 0.4
                if len(ys_w) > 0:
                    idx = np.argmin(np.abs(ys_w - cy) + np.abs(xs_w - cx))
                    dyw = int(np.sign(int(ys_w[idx]) - cy)); dxw = int(np.sign(int(xs_w[idx]) - cx))
                    if dyw == dy and dxw == dx: toward_other = 0.8  # stronger pull to non-food

                # if congested near pantry, push outward harder
                crowd_push = 0.3 if (dist_home <= HOME_R and self.last_inbox_n >= CROWD_N) else 0.0

                return outward + crowd_push + toward_other + 0.2 * road_bonus + 0.2 * toward_food + jitter

        dy, dx = max(moves, key=lambda mv: score_move(*mv))

        # harvest/use flag is only for "stay" actions now; harvesting is auto after move
        use = 1 if (dy == 0 and dx == 0) else 0

        # broadcast
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

        # ---------- AUTO-HARVEST AFTER MOVE (fix) ----------
        just_picked_food = False
        if tile in (1, 4, 5, 6):
            gained = world.harvest(self.pos, kind=tile)
            if gained > 0:
                if tile == 1:
                    before = self.inventory["food"]
                    self.inventory["food"] += gained
                    just_picked_food = (before == 0)
                    self.energy = min(self.energy + (0.5 + 0.1 * self.tool_level), 5.0)
                elif tile == 4: self.inventory["wood"]  += gained
                elif tile == 5: self.inventory["fiber"] += gained
                elif tile == 6: self.inventory["stone"] += gained

        # start/continue tracking carry path
        if just_picked_food:
            self.carry_path = [self.pos]
        elif self.inventory["food"] > 0:
            self.carry_path.append(self.pos)

        # CRAFT simple tool
        if self.tool_level == 0 and self.inventory["wood"] >= 2 and self.inventory["fiber"] >= 1:
            self.inventory["wood"] -= 2
            self.inventory["fiber"] -= 1
            self.tool_level = 1

        # BUILD ROADS away from pantry where desire is high
        dist_home = abs(world.pantry[0] - y) + abs(world.pantry[1] - x) if world.pantry else 999
        if world.materials[y, x] == 0 and dist_home >= ROAD_MIN_DIST:
            desire = world.road_desire[y, x]
            if desire > 3.0 and (self.inventory["stone"] >= 1 or self.inventory["wood"] >= 2):
                if world.place_road(y, x):
                    if self.inventory["stone"] >= 1: self.inventory["stone"] -= 1
                    else: self.inventory["wood"] -= 2
                    self.knowledge["roads_built"] += 1

        # CACHE interaction
        if world.materials[y, x] == 8:
            if self.inventory["food"] > 0:
                world.cache_deposit(y, x, self.inventory["food"])
                self.inventory["food"] = 0
                self.knowledge["caches_used"] += 1
            elif self.inventory["food"] == 0 and world.caches.get((y, x), 0) > 0:
                taken = world.cache_take(y, x, 1)
                self.inventory["food"] += taken

        # If on pantry and it's crowded, try side cache first
        if world.materials[y, x] == 3 and self.inventory["food"] > 0 and self.last_inbox_n >= CROWD_N:
            nbrs = [(y, x+1),(y, x-1),(y+1, x),(y-1, x)]
            placed = False
            for ny2, nx2 in nbrs:
                if 0 <= ny2 < world.cfg.height and 0 <= nx2 < world.cfg.width:
                    if world.materials[ny2, nx2] == 8:
                        world.cache_deposit(ny2, nx2, self.inventory["food"])
                        self.inventory["food"] = 0
                        self.knowledge["caches_used"] += 1
                        placed = True
                        break
            if not placed:
                for ny2, nx2 in nbrs:
                    if 0 <= ny2 < world.cfg.height and 0 <= nx2 < world.cfg.width:
                        if world.materials[ny2, nx2] == 0 and self.inventory["wood"] >= 3:
                            if world.place_cache(ny2, nx2):
                                world.cache_deposit(ny2, nx2, self.inventory["food"])
                                self.inventory["food"] = 0
                                self.inventory["wood"] -= 3
                                self.knowledge["caches_used"] += 1
                                placed = True
                                break

        # PANTRY deposit (normal)
        if world.materials[y, x] == 3 and self.inventory["food"] > 0:
            world.shared_store += self.inventory["food"]
            self.inventory["food"] = 0
            self.energy = min(self.energy + 0.2, 5.0)
            self.knowledge["deposits"] += 1
            # learning: reinforce haul path only if meaningful
            if len(self.carry_path) >= MIN_PATH_REINF:
                world.reinforce_path(self.carry_path, amount=1.0)
                on_road = sum(1 for (py, px) in self.carry_path if world.materials[py, px] == 7)
                frac = on_road / max(1, len(self.carry_path))
                alpha = 0.2
                self.road_bias = float((1 - alpha) * self.road_bias + alpha * frac)
            self.carry_path = []

        # ENERGY drain (roads cheaper)
        move_cost = 0.005
        if tile == 7:  # road
            move_cost *= 0.6
        self.energy -= move_cost

        self.last_msg = decision["msg"]
