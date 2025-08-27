"""
Live viewer for the MVP sandbox using matplotlib.
Shows:
  - green squares = resources
  - gray = empty
  - yellow dot = agent
"""

import time
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from .world import World, WorldConfig
from .agents import Agent

def _frame(world: World, agent: Agent):
    grid = np.zeros_like(world.materials, dtype=np.int8)
    grid[world.materials == 1] = 1
    ay, ax = agent.pos
    grid[ay, ax] = 2
    return grid

def run_live(width: int = 48, height: int = 32, steps: int = 600, seed: int = 11, fps: int = 30):
    cfg = WorldConfig(width=width, height=height, seed=seed)
    world = World(cfg)
    rng = np.random.default_rng(seed)
    start = (rng.integers(0, height), rng.integers(0, width))
    agent = Agent(pos=start)

    cmap = colors.ListedColormap(["#2b2b2b", "#2e7d32", "#fdd835"])  # empty, resource, agent
    norm = colors.BoundaryNorm([0,1,2,3], cmap.N)

    fig, ax = plt.subplots(figsize=(width/8, height/8))
    fig.canvas.manager.set_window_title("Evo Beings — Live MVP")
    img = ax.imshow(_frame(world, agent), cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.pause(0.001)

    delay = 1.0 / max(1, fps)
    for _ in range(steps):
        agent.act(world)
        world.step()
        img.set_data(_frame(world, agent))
        plt.pause(0.001)
        time.sleep(delay)

    # keep window open at the end until closed by user
    plt.show()

if __name__ == "__main__":
    run_live()
