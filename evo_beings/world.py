"""
World module: discrete grid with energy, resources, and simple seasonality.
No jobs, no scripts—only state updates and conservation constraints.
"""
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np

@dataclass
class WorldConfig:
    width: int = 48
    height: int = 32
    max_energy: float = 10.0
    resource_density: float = 0.08
    season_period: int = 400
    seed: int = 7

class World:
    """
    Discrete 2D world with:
      - materials: 0 = empty, 1 = food/resource
      - energy field (ambient)
      - temperature with a simple seasonal sinusoid
    """
    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.tick = 0
        self.materials = np.zeros((cfg.height, cfg.width), dtype=np.int8)
        self.energy = np.zeros((cfg.height, cfg.width), dtype=np.float32)
        self.temp = np.zeros((cfg.height, cfg.width), dtype=np.float32)
        self._seed_resources()

    def _seed_resources(self) -> None:
        mask = self.rng.random(self.materials.shape) < self.cfg.resource_density
        self.materials[mask] = 1
        self.energy[:] = self.cfg.max_energy * 0.25

    def sense(self, pos: Tuple[int, int], radius: int = 2) -> Dict[str, np.ndarray]:
        y, x = pos
        ys = slice(max(0, y - radius), min(self.cfg.height, y + radius + 1))
        xs = slice(max(0, x - radius), min(self.cfg.width, x + radius + 1))
        return {
            "materials": self.materials[ys, xs].copy(),
            "energy": self.energy[ys, xs].copy(),
            "temp": self.temp[ys, xs].copy(),
            "tick": np.array([self.tick], dtype=np.int32),
        }

    def step(self) -> None:
        self.tick += 1
        self.temp += 0.01 * np.sin(2 * np.pi * self.tick / self.cfg.season_period)
        self.energy *= 0.999

    def harvest(self, pos: Tuple[int, int], kind: int = 1) -> int:
        y, x = pos
        if self.materials[y, x] == kind:
            self.materials[y, x] = 0
            return 1
        return 0

    def in_bounds(self, y: int, x: int) -> Tuple[int, int]:
        y = int(np.clip(y, 0, self.cfg.height - 1))
        x = int(np.clip(x, 0, self.cfg.width - 1))
        return y, x
    
    # inside World class
    def neighbor_messages(self, agents, idx: int, radius: int = 3):
        y, x = agents[idx].pos
        msgs = []
        for j, a in enumerate(agents):
            if j == idx:
                continue
            ay, ax = a.pos
            if abs(ay - y) + abs(ax - x) <= radius:
                msgs.append(getattr(a, "last_msg", np.zeros(4, dtype=np.float32)))
        return msgs

        
