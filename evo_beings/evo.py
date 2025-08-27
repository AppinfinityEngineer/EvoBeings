"""
Evolution helpers: simple diversity-first archive (tiny MAP-Elites style).
"""
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class Genome:
    vec: np.ndarray

def random_genome(dim: int, rng: np.random.Generator) -> Genome:
    return Genome(vec=rng.normal(0, 0.5, size=(dim,)).astype(np.float32))

def mutate(g: Genome, rng: np.random.Generator, sigma: float = 0.1, p: float = 0.05) -> Genome:
    v = g.vec.copy()
    mask = rng.random(v.shape) < p
    v[mask] += rng.normal(0, sigma, size=mask.sum()).astype(v.dtype)
    return Genome(v)

def descriptor(stats) -> np.ndarray:
    explore = np.clip(stats["unique_tiles"] / 120.0, 0, 1)
    gather  = np.clip(stats["harvests"] / 30.0, 0, 1)
    survive = np.clip(stats["ticks_alive"] / 600.0, 0, 1)
    return np.array([explore, gather, survive], dtype=np.float32)

class Archive:
    """
    Minimal 3D grid archive on [explore, gather, survive]
    """
    def __init__(self, bins: Tuple[int,int,int] = (8,8,8)):
        self.bins = np.array(bins)
        self.grid_fit = np.full(bins, -np.inf, dtype=np.float32)
        self.grid_gen = np.empty(bins, dtype=object)

    def _idx(self, bd: np.ndarray) -> Tuple[int,int,int]:
        i = (bd * self.bins).astype(int)
        i = np.minimum(i, self.bins - 1)
        return tuple(i.tolist())

    def consider(self, g: Genome, stats) -> bool:
        bd = descriptor(stats)
        fit = stats["fitness"]
        idx = self._idx(bd)
        if fit > self.grid_fit[idx]:
            self.grid_fit[idx] = fit
            self.grid_gen[idx] = g
            return True
        return False

    def sample(self, rng: np.random.Generator, k: int = 16) -> List[Genome]:
        coords = np.argwhere(np.isfinite(self.grid_fit))
        if len(coords) == 0:
            return []
        sel = rng.choice(len(coords), size=min(k, len(coords)), replace=False)
        return [self.grid_gen[tuple(c)] for c in coords[sel]]
