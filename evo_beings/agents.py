"""
Agents with age-based intelligence, exploration drive, wander missions, and
runtime innovation that can name structures and bootstrap resources.

New:
- Per-tick intelligence also gets boosted by nearby structures with iq_aura.
- Innovation can add emit (wood/fiber/stone/food) and iq_aura.
- Names are descriptive: e.g., "Relay Depot Grove", "Quarry Waystation", "Farm Workshop".
"""
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Set, List, Optional
import numpy as np
from .world import World

HOME_R = 4
CROWD_N = 3
MIN_PATH_REINF = 6
ROAD_MIN_DIST = 5

INNOVATE_IQ = 0.8
INNOVATE_PROB = 0.02

# exploration / boredom
BORED_TICKS = 160
LONG_NO_DEPOSIT = 400
WANDER_LIFE = 700


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

    age: int = 0
    intelligence: float = 0.1

    knowledge: Dict[str, int] = field(default_factory=lambda: {
        "deposits": 0, "roads_built": 0, "caches_used": 0, "innovations": 0
    })
    last_inbox_n: int = 0

    # exploration state
    bored_counter: int = 0
    ticks_since_deposit: int = 0
    wander_target: Optional[Tuple[int, int]] = None
    wander_until: int = 0

    def act(self, world: World, inbox: List[np.ndarray] = None) -> Dict[str, Any]:
        if inbox is None: inbox = []
        self.last_inbox_n = len(inbox)
        self.age += 1
        # base growth + aura boost from nearby structures
        self.intelligence = min(2.0, self.intelligence + 0.0007 + world.iq_aura_at(*self.pos))
        self.ticks_since_deposit += 1

        # boredom logic (only when not carrying)
        if world.pantry:
            py, px = world.pantry
            if self.inventory["food"] == 0 and (abs(self.pos[0]-py)+abs(self.pos[1]-px) <= HOME_R):
                self.bored_counter += 1
            else:
                self.bored_counter = max(0, self.bored_counter - 2)

        # (re)start a wander mission if bored or too long without deposits
        if (self.wander_target is None or world.tick >= self.wander_until) and \
           (self.bored_counter > BORED_TICKS or self.ticks_since_deposit > LONG_NO_DEPOSIT):
            self._choose_wander_target(world)

        obs = world.sense(self.pos, radius=2)
        decision = self._policy(world, obs)
        self._apply(decision, world)
        return decision

    # ---------------------------------------------------------------------

    def _choose_wander_target(self, world: World) -> None:
        H, W = world.cfg.height, world.cfg.width
        py, px = world.pantry if world.pantry else self.pos

        cand: List[Tuple[int, int, float]] = []
        for _ in range(400):
            y = int(np.random.randint(0, H))
            x = int(np.random.randint(0, W))
            d = abs(y - py) + abs(x - px)
            if d < max(6, (H+W)//8):
                continue
            cand.append((y, x, float(world.explore[y, x])))

        if not cand:
            return
        cand.sort(key=lambda t: t[2])
        pick = cand[: max(1, len(cand)//5)]
        y, x, _ = pick[int(np.random.randint(0, len(pick)))]
        self.wander_target = (y, x)
        self.wander_until = world.tick + WANDER_LIFE

    # ---------------------------------------------------------------------

    def _policy(self, world: World, obs: Dict[str, Any]) -> Dict[str, Any]:
        mats = obs["materials"]; h, w = mats.shape
        cy, cx = h // 2, w // 2

        ay, ax = self.pos
        H, W = world.cfg.height, world.cfg.width

        if world.pantry is not None:
            py, px = world.pantry
            dist_home = abs(py - ay) + abs(px - ax)
        else:
            dist_home = 999

        carrying = self.inventory["food"] > 0
        moves = [(0,1),(0,-1),(1,0),(-1,0)]

        def edge_dist(y, x): 
            return min(y, x, H - 1 - y, W - 1 - x)
        curr_edge = edge_dist(ay, ax)

        def score_move(dy: int, dx: int) -> float:
            ny, nx = world.in_bounds(ay + dy, ax + dx)
            tile_next = int(world.materials[ny, nx])

            if ny == ay and nx == ax:
                return -1e6  # avoid no-op

            d_now = dist_home
            d_next = abs((world.pantry[0] if world.pantry else ay) - ny) + \
                     abs((world.pantry[1] if world.pantry else ax) - nx)
            toward_home = float(d_now - d_next)

            has_move = 1.0 if (tile_next == 7 or "move_mult" in world.tile_props(tile_next)) else 0.0
            road_bonus = self.road_bias * has_move

            exp_here = world.explore[ny, nx]
            explore_gain = -float(exp_here) * 0.02  # prefer lower explore

            next_edge = edge_dist(ny, nx)
            edge_inward = max(0.0, next_edge - curr_edge)

            to_target = 0.0
            if not carrying and self.wander_target is not None:
                ty, tx = self.wander_target
                d_now_t = abs(ay - ty) + abs(ax - tx)
                d_next_t = abs(ny - ty) + abs(nx - tx)
                to_target = float(d_now_t - d_next_t)

            novelty = 1.0 if (ny, nx) not in self.visited else 0.0
            jitter = (np.random.rand() - 0.5) * 0.02

            if carrying:
                return (1.0 + 0.5 * min(1.0, self.intelligence)) * toward_home \
                       + 1.0 * road_bonus + 0.10 * edge_inward + 0.02 * novelty + jitter
            else:
                outward = -toward_home if dist_home <= HOME_R else 0.0
                ys_f, xs_f = np.where(mats == 1)
                ys_w, xs_w = np.where((mats == 4) | (mats == 5) | (mats == 6))
                toward_food = 0.0; toward_other = 0.0
                if len(ys_f) > 0:
                    idx = np.argmin(np.abs(ys_f - cy) + np.abs(xs_f - cx))
                    if (int(np.sign(int(ys_f[idx]) - cy)) == dy) and (int(np.sign(int(xs_f[idx]) - cx)) == dx):
                        toward_food = 0.35
                if len(ys_w) > 0:
                    idx = np.argmin(np.abs(ys_w - cy) + np.abs(xs_w - cx))
                    if (int(np.sign(int(ys_w[idx]) - cy)) == dy) and (int(np.sign(int(xs_w[idx]) - cx)) == dx):
                        toward_other = 0.8

                return outward + 0.30 * edge_inward + toward_other + 0.25 * road_bonus \
                       + 0.60 * to_target + explore_gain + 0.02 * novelty + jitter

        dy, dx = max(moves, key=lambda mv: score_move(*mv))

        # semantic message
        local_food = float((mats == 1).mean())
        dist_norm = min(1.0, (dist_home if world.pantry else 10) / 10.0)
        msg = np.array([
            1.0 if carrying else 0.0,
            local_food,
            dist_norm,
            min(1.0, self.intelligence / 2.0),
        ], dtype=np.float32)

        return {"move": (dy, dx), "use": 0, "msg": msg}

    # ---------------------------------------------------------------------

    def _apply(self, decision: Dict[str, Any], world: World) -> None:
        dy, dx = decision["move"]
        ay, ax = self.pos
        ny, nx = world.in_bounds(ay + dy, ax + dx)
        self.pos = (ny, nx)
        self.visited.add(self.pos)
        y, x = self.pos
        code_here = int(world.materials[y, x])

        # mark exploration
        world.mark_visit(y, x, amount=0.6)

        # AUTO-HARVEST after move
        just_picked_food = False
        if code_here in (1, 4, 5, 6):
            gained = world.harvest(self.pos, kind=code_here)
            if gained > 0:
                if code_here == 1:
                    before = self.inventory["food"]; self.inventory["food"] += gained
                    just_picked_food = (before == 0)
                    self.energy = min(self.energy + (0.5 + 0.1 * self.tool_level), 5.0)
                elif code_here == 4: self.inventory["wood"]  += gained
                elif code_here == 5: self.inventory["fiber"] += gained
                elif code_here == 6: self.inventory["stone"] += gained

        if just_picked_food:
            self.carry_path = [self.pos]
            self.wander_target = None
        elif self.inventory["food"] > 0:
            self.carry_path.append(self.pos)

        # craft simple tool
        if self.tool_level == 0 and self.inventory["wood"] >= 2 and self.inventory["fiber"] >= 1:
            self.inventory["wood"] -= 2
            self.inventory["fiber"] -= 1
            self.tool_level = 1

        # innovate occasionally
        if self.intelligence >= INNOVATE_IQ and np.random.rand() < INNOVATE_PROB:
            self._maybe_innovate(world)

        # build roads away from pantry if desire high
        dist_home = abs(world.pantry[0] - y) + abs(world.pantry[1] - x) if world.pantry else 999
        if code_here == 0 and dist_home >= ROAD_MIN_DIST:
            desire = world.road_desire[y, x]
            if desire > 3.0 and (self.inventory["stone"] >= 1 or self.inventory["wood"] >= 2):
                if world.place(y, x, 7):
                    if self.inventory["stone"] >= 1: self.inventory["stone"] -= 1
                    else: self.inventory["wood"] -= 2
                    self.knowledge["roads_built"] += 1
                    self.intelligence = min(2.0, self.intelligence + 0.01)

        # cache interaction (base or dynamic)
        props = world.tile_props(int(world.materials[y, x]))
        if props.get("cache", False):
            if self.inventory["food"] > 0:
                world.cache_deposit(y, x, self.inventory["food"]); self.inventory["food"] = 0
                self.knowledge["caches_used"] += 1
            elif self.inventory["food"] == 0 and world.caches.get((y, x), 0) > 0:
                take = world.cache_take(y, x, 1); self.inventory["food"] += take

        # pantry deposit
        if int(world.materials[y, x]) == 3 and self.inventory["food"] > 0:
            world.shared_store += self.inventory["food"]; self.inventory["food"] = 0
            self.energy = min(self.energy + 0.2, 5.0)
            self.knowledge["deposits"] += 1
            self.intelligence = min(2.0, self.intelligence + 0.02)
            self.ticks_since_deposit = 0
            self.bored_counter = 0
            # reinforce haul path
            if len(self.carry_path) >= MIN_PATH_REINF:
                world.reinforce_path(self.carry_path, amount=1.0)
                on_road = sum(1 for (py, px) in self.carry_path if int(world.materials[py, px]) == 7)
                frac = on_road / max(1, len(self.carry_path))
                alpha = 0.2
                self.road_bias = float((1 - alpha) * self.road_bias + alpha * frac)
            self.carry_path = []

        # move energy cost (roads/dynamic movers cheaper)
        move_cost = 0.005
        move_mult = float(world.tile_props(int(world.materials[y, x])).get("move_mult", 1.0))
        move_cost *= move_mult
        self.energy -= move_cost

        self.last_msg = decision["msg"]

    # ---------------------------------------------------------------------

    def _maybe_innovate(self, world: World) -> None:
        y, x = self.pos
        if int(world.materials[y, x]) != 0:
            return
        have_wood = self.inventory["wood"] >= 3
        have_fiber = self.inventory["fiber"] >= 2
        have_stone = self.inventory["stone"] >= 1
        if not (have_wood or have_fiber or have_stone):
            return

        py, px = world.pantry if world.pantry else (y, x)
        dist_home = abs(py - y) + abs(px - x)

        props: Dict[str, Any] = {}
        name_parts = []

        # mobility far from home
        if dist_home >= 6 and (have_stone or have_wood):
            props["move_mult"] = 0.7 if have_stone else 0.8
            name_parts.append("Waystation")

        # comms hub
        if np.random.rand() < 0.45 and have_fiber:
            props["comms_bonus"] = int(2 + np.random.randint(0, 3))  # 2..4
            name_parts.append("Relay")

        # depot/cache
        if np.random.rand() < 0.55 and have_wood:
            props["cache"] = True
            name_parts.append("Depot")

        # food growth
        if np.random.rand() < 0.30:
            props["food_boost"] = round(0.1 + 0.3 * np.random.rand(), 2)
            name_parts.append("Grove")

        # resource emit (elders can bootstrap resources)
        if self.intelligence > 1.1 and np.random.rand() < 0.5:
            emit: Dict[int, float] = {}
            # pick 1–2 kinds to emit
            kinds = [4, 5, 6, 1]  # wood, fiber, stone, food
            self_rng = np.random.choice(kinds, size=int(1 + np.random.randint(0, 2)), replace=False)
            for k in self_rng:
                emit[int(k)] = round(0.10 + 0.20 * np.random.rand(), 2)  # per-attempt probability
                # add a name part
                if k == 4: name_parts.append("Forester")
                if k == 5: name_parts.append("Loom")
                if k == 6: name_parts.append("Quarry")
                if k == 1: name_parts.append("Farm")
            props["emit"] = emit
            props["emit_radius"] = 1

        # iq aura: make nearby workers learn faster
        if self.intelligence > 1.0 and np.random.rand() < 0.4:
            props["iq_aura"] = round(0.001 + 0.003 * np.random.rand(), 4)
            name_parts.append("Workshop")

        if not props:
            props["move_mult"] = 0.85
            name_parts.append("Marker")

        # final name
        name = " ".join(name_parts) if name_parts else "Node"

        code = world.add_dynamic_structure_type(name=name, props=props)
        if world.place(y, x, code):
            if props.get("cache"): self.inventory["wood"] = max(0, self.inventory["wood"] - 3)
            if "comms_bonus" in props: self.inventory["fiber"] = max(0, self.inventory["fiber"] - 2)
            if "move_mult" in props and self.inventory["stone"] > 0: self.inventory["stone"] -= 1
            self.knowledge["innovations"] += 1
            self.intelligence = min(2.0, self.intelligence + 0.03)
