"""
World: discrete grid with resources, seeds, pantry/base, and simple structures.
Material codes:
  0 = empty
  1 = food/resource
  2 = seed (ripens to food)
  3 = pantry/base (deposit here to grow the colony)
  4 = tree (yields wood)
  5 = fiber bush (yields fiber)
  6 = rock (yields stone)
  7 = road (reduced move cost)
  8 = cache (local food depot)
  9 = beacon (extends comms range)
"""
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
import numpy as np

@dataclass
class WorldConfig:
    width: int = 64
    height: int = 40
    max_energy: float = 10.0
    resource_density: float = 0.0      # start empty; only observer/agents add stuff
    season_period: int = 400
    seed: int = 7

class World:
    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.tick = 0

        self.materials = np.zeros((cfg.height, cfg.width), dtype=np.int8)  # 0..9
        self.energy = np.zeros((cfg.height, cfg.width), dtype=np.float32)
        self.temp = np.zeros((cfg.height, cfg.width), dtype=np.float32)

        # economy/state
        self.shared_store: int = 0                  # pantry food
        self.pantry: Tuple[int, int] | None = None
        self.caches: Dict[Tuple[int, int], int] = {}  # per-tile food stores for caches

        self._seed_resources()

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
        self.temp += 0.01 * np.sin(2 * np.pi * self.tick / self.cfg.season_period)
        self.energy *= 0.999
        # seeds ripen every 6 ticks
        if self.tick % 6 == 0:
            self.materials[self.materials == 2] = 1

    def grow_food_near_pantry(self, radius: int = 4, k: int = 4) -> int:
        """Try to spawn up to k new food tiles near pantry on empty cells."""
        if self.pantry is None:
            return 0
        py, px = self.pantry
        candidates: List[Tuple[int, int]] = []
        H, W = self.materials.shape
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if abs(dy) + abs(dx) > radius:
                    continue
                y, x = py + dy, px + dx
                if 0 <= y < H and 0 <= x < W and self.materials[y, x] == 0:
                    candidates.append((y, x))
        if not candidates:
            return 0
        self.rng.shuffle(candidates)
        placed = 0
        for (y, x) in candidates[:k]:
            self.materials[y, x] = 1
            placed += 1
        return placed

    def harvest(self, pos: Tuple[int, int], kind: int) -> int:
        """Remove a material of 'kind' from this cell and return +1 if successful."""
        y, x = pos
        if self.materials[y, x] == kind:
            self.materials[y, x] = 0
            return 1
        return 0

    # ---------- building & comms ----------
    def add_pantry(self, y: int, x: int) -> None:
        self.materials[y, x] = 3
        self.pantry = (y, x)

    def place(self, y: int, x: int, code: int) -> bool:
        if self.materials[y, x] == 0:
            self.materials[y, x] = code
            if code == 8:  # cache
                self.caches[(y, x)] = 0
            return True
        return False

    def erase(self, y: int, x: int) -> None:
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

    # caches API
    def cache_deposit(self, y: int, x: int, n: int) -> int:
        if self.materials[y, x] != 8:
            return 0
        self.caches[(y, x)] = self.caches.get((y, x), 0) + n
        return n

    def cache_take(self, y: int, x: int, n: int) -> int:
        if self.materials[y, x] != 8:
            return 0
        have = self.caches.get((y, x), 0)
        take = min(have, n)
        self.caches[(y, x)] = have - take
        return take

    # comms helpers
    def near_beacon(self, y: int, x: int, r: int = 1) -> bool:
        y0, y1 = max(0, y - r), min(self.cfg.height, y + r + 1)
        x0, x1 = max(0, x - r), min(self.cfg.width, x + r + 1)
        return np.any(self.materials[y0:y1, x0:x1] == 9)

    def neighbor_messages(self, agents: List["Agent"], idx: int, radius: int = 3):
        """Collect neighbor messages; beacon near receiver increases range."""
        y, x = agents[idx].pos
        extra = 3 if self.near_beacon(y, x) else 0
        r_eff = radius + extra
        msgs = []
        for j, a in enumerate(agents):
            if j == idx:
                continue
            ay, ax = a.pos
            if abs(ay - y) + abs(ax - x) <= r_eff:
                msgs.append(getattr(a, "last_msg", np.zeros(4, dtype=np.float32)))
        return msgs
