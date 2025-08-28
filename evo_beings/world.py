"""
World with: resources, pantry, basic & dynamic structures, road-desire learning,
a global exploration field, dynamic effects (food_boost, emit, iq_aura, comms),
and automatic map expansion when agents reach the frontier.

Material codes:
  0 empty, 1 food, 2 seed, 3 pantry, 4 tree, 5 fiber, 6 rock, 7 road, 8 cache, 9 beacon
Dynamic invented structures: 10, 11, ... (runtime)
"""
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
from collections import deque
import numpy as np


@dataclass
class WorldConfig:
    width: int = 64
    height: int = 40
    max_energy: float = 10.0
    resource_density: float = 0.0
    season_period: int = 400
    seed: int = 7


class World:
    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.tick = 0

        # core fields
        self.materials = np.zeros((cfg.height, cfg.width), dtype=np.int16)
        self.energy = np.zeros((cfg.height, cfg.width), dtype=np.float32)
        self.temp = np.zeros((cfg.height, cfg.width), dtype=np.float32)

        # economy/state
        self.shared_store: int = 0
        self.pantry: Tuple[int, int] | None = None
        self.caches: Dict[Tuple[int, int], int] = {}

        # learning / navigation fields
        self.road_desire = np.zeros((cfg.height, cfg.width), dtype=np.float32)
        self.explore = np.zeros((cfg.height, cfg.width), dtype=np.float32)

        # dynamic structure registry:
        # code -> { name, color, cache?, move_mult?, comms_bonus?, food_boost?, iq_aura?, emit{mat->p}, emit_radius? }
        self.struct_defs: Dict[int, Dict[str, Any]] = {}
        self.next_dyn_code: int = 10

        # expansion cooldown
        self._last_expand_tick: int = -10_000

        # rolling event log (e.g., builds)
        self.events = deque(maxlen=800)

        self._seed_resources()

    # ---------- event logging ----------
    def log_event(self, kind: str, info: Dict[str, Any]) -> None:
        self.events.append({"tick": self.tick, "kind": kind, **info})

    def log_build(self, code: int, y: int, x: int) -> None:
        names = {7: "Road", 8: "Cache", 9: "Beacon"}
        if code >= 10:
            nm = self.struct_defs.get(code, {}).get("name", f"type{code}")
        else:
            nm = names.get(code, f"type{code}")
        self.log_event("build", {"code": int(code), "y": int(y), "x": int(x), "name": nm})

    # ---------- helpers ----------
    def _seed_resources(self) -> None:
        if self.cfg.resource_density > 0:
            mask = self.rng.random(self.materials.shape) < self.cfg.resource_density
            self.materials[mask] = 1
        self.energy[:] = self.cfg.max_energy * 0.25

    def in_bounds(self, y: int, x: int) -> Tuple[int, int]:
        y = int(np.clip(y, 0, self.cfg.height - 1))
        x = int(np.clip(x, 0, self.cfg.width - 1))
        return y, x

    def sense(self, pos: Tuple[int, int], radius: int = 2) -> Dict[str, Any]:
        y, x = pos
        ys = slice(max(0, y - radius), min(self.cfg.height, y + radius + 1))
        xs = slice(max(0, x - radius), min(self.cfg.width, x + radius + 1))
        return {
            "materials": self.materials[ys, xs].copy(),
            "energy": self.energy[ys, xs].copy(),
            "temp": self.temp[ys, xs].copy(),
            "tick": np.array([self.tick], dtype=np.int32),
        }

    # ---------- dynamics ----------
    def step(self) -> None:
        self.tick += 1
        # tiny seasonal temperature oscillation
        self.temp += 0.01 * np.sin(2 * np.pi * self.tick / self.cfg.season_period)
        # slow energy decay
        self.energy *= 0.999

        # seeds ripen into food
        if self.tick % 6 == 0:
            self.materials[self.materials == 2] = 1

        # learning/exploration fade
        self.road_desire *= 0.9995
        self.explore *= 0.9997

        # dynamic structure effects periodically
        if self.tick % 10 == 0:
            self._apply_dynamic_effects()

    def _apply_dynamic_effects(self) -> None:
        """Apply effects of invented structures:
           - food_boost: small chance to spawn food in radius 2
           - emit: spawn specific materials near the structure
        """
        H, W = self.materials.shape
        for code, props in self.struct_defs.items():
            ys, xs = np.where(self.materials == code)
            if len(ys) == 0:
                continue

            # Food boost (garden/grove)
            fb = float(props.get("food_boost", 0.0))
            if fb > 0:
                for (y, x) in zip(ys, xs):
                    for _ in range(2):
                        dy, dx = int(self.rng.integers(-2, 3)), int(self.rng.integers(-2, 3))
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and self.materials[ny, nx] == 0:
                            if self.rng.random() < min(0.25, fb):
                                self.materials[ny, nx] = 1  # food

            # Emit resources (workshop/foundry/quarry/farm)
            emit: Dict[int, float] = props.get("emit", {})
            if emit:
                r = int(props.get("emit_radius", 1))
                for (y, x) in zip(ys, xs):
                    for _ in range(2):  # a few attempts per tick per site
                        dy, dx = int(self.rng.integers(-r, r + 1)), int(self.rng.integers(-r, r + 1))
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and self.materials[ny, nx] == 0:
                            for mat, p in emit.items():
                                if self.rng.random() < float(p):
                                    self.materials[ny, nx] = int(mat)
                                    break

    def grow_food_near_pantry(self, radius: int = 4, k: int = 4) -> int:
        if self.pantry is None:
            return 0
        py, px = self.pantry
        H, W = self.materials.shape
        cand: List[Tuple[int, int]] = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if abs(dy) + abs(dx) > radius:
                    continue
                y, x = py + dy, px + dx
                if 0 <= y < H and 0 <= x < W and self.materials[y, x] == 0:
                    cand.append((y, x))
        self.rng.shuffle(cand)
        for (y, x) in cand[:k]:
            self.materials[y, x] = 1
        return min(k, len(cand))

    def grow_food_ring(self, r_min: int = 5, r_max: int = 9, k: int = 6) -> int:
        if self.pantry is None:
            return 0
        py, px = self.pantry
        H, W = self.materials.shape
        cand: List[Tuple[int, int]] = []
        for dy in range(-r_max, r_max + 1):
            for dx in range(-r_max, r_max + 1):
                d = abs(dy) + abs(dx)
                if d < r_min or d > r_max:
                    continue
                y, x = py + dy, px + dx
                if 0 <= y < H and 0 <= x < W and self.materials[y, x] == 0:
                    cand.append((y, x))
        self.rng.shuffle(cand)
        for (y, x) in cand[:k]:
            self.materials[y, x] = 1
        return min(k, len(cand))

    def harvest(self, pos: Tuple[int, int], kind: int) -> int:
        y, x = pos
        if self.materials[y, x] == kind:
            self.materials[y, x] = 0
            if (y, x) in self.caches:
                del self.caches[(y, x)]
            return 1
        return 0

    # ---------- building & protection ----------
    def add_pantry(self, y: int, x: int) -> None:
        self.materials[y, x] = 3
        self.pantry = (y, x)

    def ensure_pantry(self, y: int, x: int) -> None:
        # simple “auto-repair” — always restore pantry
        self.add_pantry(y, x)

    def place(self, y: int, x: int, code: int) -> bool:
        if self.materials[y, x] == 0:
            self.materials[y, x] = code
            if code == 8 or (code in self.struct_defs and self.struct_defs[code].get("cache", False)):
                self.caches[(y, x)] = 0
            # log builds for non-empty structural tiles
            if code >= 7:
                self.log_build(code, y, x)
            return True
        return False

    def erase(self, y: int, x: int) -> None:
        """Erase a tile — but NEVER the pantry."""
        if self.materials[y, x] == 3:
            return
        if (y, x) in self.caches:
            del self.caches[(y, x)]
        self.materials[y, x] = 0

    # observer brushes
    def place_seed(self, y: int, x: int) -> bool:   return self.place(y, x, 2)
    def place_tree(self, y: int, x: int) -> bool:   return self.place(y, x, 4)
    def place_fiber(self, y: int, x: int) -> bool:  return self.place(y, x, 5)
    def place_rock(self, y: int, x: int) -> bool:   return self.place(y, x, 6)
    def place_road(self, y: int, x: int) -> bool:   return self.place(y, x, 7)
    def place_cache(self, y: int, x: int) -> bool:  return self.place(y, x, 8)
    def place_beacon(self, y: int, x: int) -> bool: return self.place(y, x, 9)

    # ---------- dynamic registry ----------
    def add_dynamic_structure_type(self, name: str, props: Dict[str, Any]) -> int:
        code = self.next_dyn_code
        self.next_dyn_code += 1

        # pleasant random color if not provided
        hue = float(self.rng.random())

        def hsv_to_hex(h: float) -> str:
            i = int(h * 6) % 6
            f = h * 6 - int(h * 6)
            q = 1 - f
            rgb = [(1, f, 0), (q, 1, 0), (0, 1, f), (0, q, 1), (f, 0, 1), (1, 0, q)][i]
            r, g, b = [int(255 * (0.6 + 0.4 * v)) for v in rgb]
            return f"#{r:02x}{g:02x}{b:02x}"

        color = props.pop("color", hsv_to_hex(hue))
        self.struct_defs[code] = {**props, "name": name, "color": color}
        return code

    def tile_props(self, code: int) -> Dict[str, Any]:
        if code == 7:  # road
            return {"move_mult": 0.6}
        if code == 9:  # beacon
            return {"comms_bonus": 3}
        if code == 8:  # cache
            return {"cache": True}
        if code in self.struct_defs:
            return self.struct_defs[code]
        return {}

    # ---------- caches ----------
    def cache_deposit(self, y: int, x: int, n: int) -> int:
        code = int(self.materials[y, x])
        if not self.tile_props(code).get("cache", False):
            return 0
        self.caches[(y, x)] = self.caches.get((y, x), 0) + n
        return n

    def cache_take(self, y: int, x: int, n: int) -> int:
        code = int(self.materials[y, x])
        if not self.tile_props(code).get("cache", False):
            return 0
        have = self.caches.get((y, x), 0)
        take = min(have, n)
        self.caches[(y, x)] = have - take
        return take

    # ---------- learning hooks ----------
    def reinforce_path(self, path, amount: float = 1.0) -> None:
        if not path:
            return
        for (y, x) in path:
            if 0 <= y < self.cfg.height and 0 <= x < self.cfg.width:
                self.road_desire[y, x] = min(self.road_desire[y, x] + amount, 50.0)

    def mark_visit(self, y: int, x: int, amount: float = 1.0) -> None:
        self.explore[y, x] = min(self.explore[y, x] + amount, 100.0)

    # ---------- comms / learning auras ----------
    def comm_bonus_at(self, y: int, x: int) -> int:
        r = 2
        y0, y1 = max(0, y - r), min(self.cfg.height, y + r + 1)
        x0, x1 = max(0, x - r), min(self.cfg.width, x + r + 1)
        patch = self.materials[y0:y1, x0:x1]
        bonus = 0
        if np.any(patch == 9):  # beacons
            bonus = max(bonus, 3)
        for c, props in self.struct_defs.items():
            if props.get("comms_bonus", 0) > 0 and np.any(patch == c):
                bonus = max(bonus, int(props["comms_bonus"]))
        return bonus

    def iq_aura_at(self, y: int, x: int) -> float:
        """Return learning boost from nearby dynamic structures (capped per tick)."""
        r = 2
        y0, y1 = max(0, y - r), min(self.cfg.height, y + r + 1)
        x0, x1 = max(0, x - r), min(self.cfg.width, x + r + 1)
        patch = self.materials[y0:y1, x0:x1]
        boost = 0.0
        for c, props in self.struct_defs.items():
            aura = float(props.get("iq_aura", 0.0))
            if aura > 0 and np.any(patch == c):
                boost += aura
        return min(boost, 0.01)

    def neighbor_messages(self, agents: List["Agent"], idx: int, radius: int = 3):
        y, x = agents[idx].pos
        r_eff = radius + self.comm_bonus_at(y, x)
        msgs = []
        for j, a in enumerate(agents):
            if j == idx:
                continue
            ay, ax = a.pos
            if abs(ay - y) + abs(ax - x) <= r_eff:
                msgs.append(getattr(a, "last_msg", np.zeros(4, dtype=np.float32)))
        return msgs

    # ---------- auto expansion ----------
    def expand_if_needed(self, agents: List["Agent"], margin: int = 8, pad: int = 32) -> int:
        """
        Pad the world on all sides if any agent/structure approaches the edge.
        Expansion is throttled (cooldown) and hard-capped by MAX size.
        Returns the pad amount (0 if no expansion).
        """
        # cooldown (~400 ticks between expansions)
        if self.tick - self._last_expand_tick < 400:
            return 0

        H, W = self.materials.shape

        # absolute cap to keep memory sane
        MAX_H, MAX_W = 512, 512
        if H >= MAX_H or W >= MAX_W:
            return 0

        # trigger if any agent is close to an edge
        need = any(
            (a.pos[0] < margin) or (a.pos[1] < margin) or
            (H - 1 - a.pos[0] < margin) or (W - 1 - a.pos[1] < margin)
            for a in agents
        )
        # also trigger if any non-empty cell is inside the margin band
        if not need and margin > 0:
            if (self.materials[:margin, :].any() or self.materials[-margin:, :].any() or
                self.materials[:, :margin].any() or self.materials[:, -margin:].any()):
                need = True
        if not need:
            return 0

        # clamp pad so we do not exceed MAX
        pad_h = min(pad, max(0, (MAX_H - H) // 2))
        pad_w = min(pad, max(0, (MAX_W - W) // 2))
        pad_use = int(min(pad_h, pad_w))
        if pad_use <= 0:
            return 0

        # grow arrays
        self.materials = np.pad(self.materials, ((pad_use, pad_use), (pad_use, pad_use)), mode="constant")
        self.energy = np.pad(self.energy, ((pad_use, pad_use), (pad_use, pad_use)), mode="constant")
        self.temp = np.pad(self.temp, ((pad_use, pad_use), (pad_use, pad_use)), mode="constant")
        self.road_desire = np.pad(self.road_desire, ((pad_use, pad_use), (pad_use, pad_use)), mode="constant")
        self.explore = np.pad(self.explore, ((pad_use, pad_use), (pad_use, pad_use)), mode="constant")

        # shift caches & pantry
        self.caches = {(y + pad_use, x + pad_use): v for (y, x), v in self.caches.items()}
        if self.pantry is not None:
            py, px = self.pantry
            self.pantry = (py + pad_use, px + pad_use)

        # update config (new size)
        self.cfg = WorldConfig(
            width=W + 2 * pad_use,
            height=H + 2 * pad_use,
            max_energy=self.cfg.max_energy,
            resource_density=self.cfg.resource_density,
            season_period=self.cfg.season_period,
            seed=self.cfg.seed,
        )

        self._last_expand_tick = self.tick
        return pad_use
