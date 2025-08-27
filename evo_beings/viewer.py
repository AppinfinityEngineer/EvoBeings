"""
Live viewers using matplotlib.

- run_live(...)        -> single-agent viewer
- run_live_multi(...)  -> multi-agent viewer with seed planting, pantry, and reproduction

Colors:
  empty  = dark gray
  food   = green
  seed   = teal
  pantry = white square
  agents = colored dots (scatter)
"""
import time
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from .world import World, WorldConfig
from .agents import Agent

def run_live(
    width: int = 48,
    height: int = 32,
    steps: int = 600,
    seed: int = 11,
    fps: int = 30,
) -> None:
    cfg = WorldConfig(width=width, height=height, seed=seed, resource_density=0.0)
    world = World(cfg)

    # pantry in center
    base_y, base_x = world.cfg.height // 2, world.cfg.width // 2
    world.add_pantry(base_y, base_x)

    rng = np.random.default_rng(seed)
    start = (rng.integers(0, height), rng.integers(0, width))
    agent = Agent(pos=start)

    cmap = colors.ListedColormap(["#2b2b2b", "#2e7d32", "#00897b", "#ffffff"])
    norm = colors.BoundaryNorm([0,1,2,3,4], cmap.N)

    fig, ax = plt.subplots(figsize=(width/8, height/8))
    try: fig.canvas.manager.set_window_title("Evo Beings — Live MVP")
    except Exception: pass

    img = ax.imshow(world.materials * 1, cmap=cmap, norm=norm, interpolation="nearest")
    dot = ax.scatter([agent.pos[1]], [agent.pos[0]], s=50, c="#fdd835")
    hud = ax.text(2, 1, "", color="white", fontsize=8)

    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.pause(0.001)

    def on_click(ev):
        if ev.inaxes is not ax or ev.xdata is None or ev.ydata is None:
            return
        y = int(round(ev.ydata)); x = int(round(ev.xdata))
        y = max(0, min(world.cfg.height - 1, y))
        x = max(0, min(world.cfg.width - 1, x))
        world.place_seed(y, x)
    fig.canvas.mpl_connect("button_press_event", on_click)

    delay = 1.0 / max(1, fps)
    for _ in range(steps):
        agent.act(world)
        world.step()

        img.set_data(world.materials * 1)
        dot.set_offsets(np.c_[[agent.pos[1]], [agent.pos[0]]])
        hud.set_text(f"tick {world.tick} | store {world.shared_store}")

        plt.pause(0.001); time.sleep(delay)
    plt.show()

def run_live_multi(
    n_agents: int = 1,
    width: int = 64,
    height: int = 40,
    steps: int = 1200,
    seed: int = 21,
    fps: int = 18,
    comm_radius: int = 3,
) -> None:
    cfg = WorldConfig(width=width, height=height, seed=seed, resource_density=0.0)
    world = World(cfg)

    # pantry/base in center
    base_y, base_x = world.cfg.height // 2, world.cfg.width // 2
    world.add_pantry(base_y, base_x)

    rng = np.random.default_rng(seed)

    # unique spawns
    taken = set()
    agents: List[Agent] = []
    for _ in range(n_agents):
        while True:
            pos = (int(rng.integers(0, height)), int(rng.integers(0, width)))
            if pos not in taken and world.materials[pos[0], pos[1]] != 3:
                taken.add(pos)
                break
        agents.append(Agent(pos=pos))

    cmap = colors.ListedColormap(["#2b2b2b", "#2e7d32", "#00897b", "#ffffff"])
    norm = colors.BoundaryNorm([0,1,2,3,4], cmap.N)

    fig, ax = plt.subplots(figsize=(width/8, height/8))
    try: fig.canvas.manager.set_window_title("Evo Beings — Live Multi")
    except Exception: pass

    img = ax.imshow(world.materials * 1, cmap=cmap, norm=norm, interpolation="nearest")

    colormap = plt.cm.get_cmap("tab10", max(n_agents, 10))
    agent_colors = [colormap(i % colormap.N) for i in range(n_agents)]
    ys = [a.pos[0] for a in agents]; xs = [a.pos[1] for a in agents]
    scat = ax.scatter(xs, ys, s=50, c=agent_colors)

    hud = ax.text(2, 1, "", color="white", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.pause(0.001)

    # click-to-plant seeds
    def on_click(ev):
        if ev.inaxes is not ax or ev.xdata is None or ev.ydata is None:
            return
        y = int(round(ev.ydata)); x = int(round(ev.xdata))
        y = max(0, min(world.cfg.height - 1, y))
        x = max(0, min(world.cfg.width - 1, x))
        world.place_seed(y, x)
    fig.canvas.mpl_connect("button_press_event", on_click)

    # reproduction cost: 10, 20, 40, 80, ...
    def cost_for_next(pop: int) -> int:
        return 10 * (2 ** max(0, pop - 1))

    delay = 1.0 / max(1, fps)
    for _ in range(steps):
        # act
        for i, a in enumerate(agents):
            inbox = world.neighbor_messages(agents, i, radius=comm_radius)
            a.act(world, inbox)
        world.step()

        # reproduce
        need = cost_for_next(len(agents))
        if world.shared_store >= need:
            world.shared_store -= need
            # spawn near pantry
            seed_rng = np.random.default_rng(world.tick * 17 + len(agents))
            for _ in range(50):
                oy, ox = int(seed_rng.integers(-2, 3)), int(seed_rng.integers(-2, 3))
                ny = int(np.clip(base_y + oy, 0, world.cfg.height - 1))
                nx = int(np.clip(base_x + ox, 0, world.cfg.width - 1))
                if all(a.pos != (ny, nx) for a in agents) and world.materials[ny, nx] != 3:
                    agents.append(Agent(pos=(ny, nx)))
                    agent_colors.append(colormap(len(agent_colors) % colormap.N))
                    break

        # draw
        img.set_data(world.materials * 1)
        ys = [a.pos[0] for a in agents]; xs = [a.pos[1] for a in agents]
        scat.set_offsets(np.c_[xs, ys]); scat.set_color(agent_colors)

        chatter = sum((getattr(a, "last_msg", np.zeros(4)) > 0.6).sum() for a in agents)
        hud.set_text(f"tick {world.tick} | agents {len(agents)} | store {world.shared_store} | chatter {int(chatter)}")

        plt.pause(0.001); time.sleep(delay)

    plt.show()
