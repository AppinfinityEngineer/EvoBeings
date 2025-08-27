"""
Runner: headless simulation + tiny evolution loop.
This is the single entrypoint you can call from a script or notebook.
"""
from typing import Dict
import numpy as np
from tqdm import trange
from .world import World, WorldConfig
from .agents import Agent
from .evo import Genome, random_genome, mutate, Archive

def simulate(genome: Genome, seed: int = 42) -> Dict[str, float]:
    rng = np.random.default_rng(seed + int(abs(genome.vec.sum())*1e6)%100000)
    world = World(WorldConfig(seed=seed))
    pos = (rng.integers(0, world.cfg.height), rng.integers(0, world.cfg.width))
    agent = Agent(pos=pos, energy=3.0)
    stats = {"ticks_alive": 0, "harvests": 0, "unique_tiles": 0}
    for _ in range(600):
        decision = agent.act(world)
        if decision["use"] == 1:
            stats["harvests"] += 1
        stats["unique_tiles"] = len(agent.visited)
        world.step()
        stats["ticks_alive"] += 1
        if agent.energy <= 0:
            break
    stats["fitness"] = 1.0*stats["ticks_alive"] + 1.0*stats["harvests"] + 0.5*stats["unique_tiles"]
    return stats

def evolve(generations: int = 20, pop: int = 64, dim: int = 64, seed: int = 7) -> Archive:
    rng = np.random.default_rng(seed)
    arc = Archive()
    pool = [random_genome(dim, rng) for _ in range(pop)]
    for g in pool:
        arc.consider(g, simulate(g, seed))
    for _ in trange(generations, desc="evolve"):
        parents = arc.sample(rng, k=pop//2) or pool
        children = [mutate(p, rng) for p in parents]
        for c in children:
            arc.consider(c, simulate(c, seed))
    return arc
