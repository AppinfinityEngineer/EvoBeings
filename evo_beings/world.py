"""
World: discrete grid with resources, seeds, and a pantry/base.
Material codes:
  0 = empty
  1 = food/resource
  2 = seed (ripens to food)
  3 = pantry/base (deposit here to grow the colony)
"""
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
import numpy as np

@dataclass
class WorldConfig:
    width: int = 64
    height: int = 40
    max_energy: float = 10.0
    resource_density: float = 0.0      # start with NO food on the map
    season_period: int = 400
    seed: int = 7

class World:
    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.tick = 0

        self.materials = np.zeros((cfg.height, cfg.width), dtype=np.int8)  # 0..3
        self.energy = np.zeros((cfg.height, cfg.width), dtype=np.float32)
        self.temp = np.zeros((cfg.height, cfg.width), dtype=np.float32)

        self.shared_store: int = 0
        self.pantry: Tuple[int, int] | None = None

        self._seed_resources()

    # ---------- helpers ----------
    def _seed_resources(self) -> None:
        """Only place initial food if resource_density > 0 (we default to 0)."""
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
        # seeds ripen every 10 ticks
        if self.tick % 6 == 0:
            self.materials[self.materials == 2] = 1

    def harvest(self, pos: Tuple[int, int], kind: int = 1) -> int:
        y, x = pos
        if self.materials[y, x] == kind:
            self.materials[y, x] = 0
            return 1
        return 0

    # ---------- building & comms ----------
    def add_pantry(self, y: int, x: int) -> None:
        self.materials[y, x] = 3
        self.pantry = (y, x)

    def place_seed(self, y: int, x: int) -> bool:
        if self.materials[y, x] == 0:
            self.materials[y, x] = 2
            return True
        return False

    def neighbor_messages(self, agents: List["Agent"], idx: int, radius: int = 3):
        y, x = agents[idx].pos
        msgs = []
        for j, a in enumerate(agents):
            if j == idx:
                continue
            ay, ax = a.pos
            if abs(ay - y) + abs(ax - x) <= radius:
                msgs.append(getattr(a, "last_msg", np.zeros(4, dtype=np.float32)))
        return msgs
