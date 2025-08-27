"""
Live viewers for the MVP sandbox using matplotlib.

- run_live(...)        -> single-agent viewer
- run_live_multi(...)  -> multi-agent viewer with simple message "chatter" readout

Colors:
  empty     = dark gray
  resource  = green
  agent(s)  = colored dots
"""

import time
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from .world import World, WorldConfig
from .agents import Agent


def _frame_single(world: World, agent: Agent) -> np.ndarray:
    """
    Build a small integer grid for imshow:
      0 = empty, 1 = resource, 2 = agent
    """
    grid = np.zeros_like(world.materials, dtype=np.int8)
    grid[world.materials == 1] = 1
    ay, ax = agent.pos
    grid[ay, ax] = 2
    return grid


def run_live(
    width: int = 48,
    height: int = 32,
    steps: int = 600,
    seed: int = 11,
    fps: int = 30,
) -> None:
    """
    Single-agent live viewer.
    """
    cfg = WorldConfig(width=width, height=height, seed=seed)
    world = World(cfg)

    rng = np.random.default_rng(seed)
    start = (rng.integers(0, height), rng.integers(0, width))
    agent = Agent(pos=start)

    cmap = colors.ListedColormap(["#2b2b2b", "#2e7d32", "#fdd835"])  # empty, resource, agent
    norm = colors.BoundaryNorm([0, 1, 2, 3], cmap.N)

    fig, ax = plt.subplots(figsize=(width / 8, height / 8))
    try:
        fig.canvas.manager.set_window_title("Evo Beings — Live MVP")
    except Exception:
        pass
    img = ax.imshow(_frame_single(world, agent), cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.pause(0.001)

    delay = 1.0 / max(1, fps)
    for _ in range(steps):
        agent.act(world)  # inbox defaults to []
        world.step()
        img.set_data(_frame_single(world, agent))
        plt.pause(0.001)
        time.sleep(delay)

    plt.show()


def run_live_multi(
    n_agents: int = 10,
    width: int = 64,
    height: int = 40,
    steps: int = 1200,
    seed: int = 21,
    fps: int = 18,
    comm_radius: int = 3,
) -> None:
    """
    Multi-agent live viewer with simple "chatter" readout.

    Notes:
      - Requires World.neighbor_messages(agents, idx, radius) to be present.
      - Agents are rendered as scatter points with distinct colors.
      - 'chatter' is the count of message components > 0.6 across agents (debug proxy).
    """
    cfg = WorldConfig(width=width, height=height, seed=seed)
    world = World(cfg)

    rng = np.random.default_rng(seed)

    # --- spawn with UNIQUE positions to avoid overlap ---
    taken = set()
    agents: List[Agent] = []
    for _ in range(n_agents):
        while True:
            pos = (int(rng.integers(0, height)), int(rng.integers(0, width)))
            if pos not in taken:
                taken.add(pos)
                break
        agents.append(Agent(pos=pos))

    # Base world layer (resources)
    cmap = colors.ListedColormap(["#2b2b2b", "#2e7d32", "#fdd835"])  # empty, resource, (agent color unused here)
    norm = colors.BoundaryNorm([0, 1, 2, 3], cmap.N)

    fig, ax = plt.subplots(figsize=(width / 8, height / 8))
    try:
        fig.canvas.manager.set_window_title("Evo Beings — Live Multi")
    except Exception:
        pass

    img = ax.imshow(world.materials * 1, cmap=cmap, norm=norm, interpolation="nearest")

    # --- distinct COLORS per agent so overlaps are visible ---
    colormap = plt.cm.get_cmap("tab10", n_agents)
    agent_colors = [colormap(i) for i in range(n_agents)]

    ys = [a.pos[0] for a in agents]
    xs = [a.pos[1] for a in agents]
    scat = ax.scatter(xs, ys, s=50, c=agent_colors)  # bigger markers

    # HUD text
    hud = ax.text(2, 1, "", color="white", fontsize=8)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.pause(0.001)

    delay = 1.0 / max(1, fps)
    for _ in range(steps):
        # Act with neighbor inboxes
        for i, a in enumerate(agents):
            inbox = world.neighbor_messages(agents, i, radius=comm_radius) if hasattr(world, "neighbor_messages") else []
            a.act(world, inbox)
        world.step()

        # Update world layer & agents
        img.set_data(world.materials * 1)
        ys = [a.pos[0] for a in agents]
        xs = [a.pos[1] for a in agents]
        scat.set_offsets(np.c_[xs, ys])

        # Simple chatter metric (debug view)
        chatter = sum((getattr(a, "last_msg", np.zeros(4)) > 0.6).sum() for a in agents)
        hud.set_text(f"tick {world.tick} | agents {len(agents)} | chatter {int(chatter)}")

        plt.pause(0.001)
        time.sleep(delay)

    plt.show()
