"""
Live viewers using matplotlib.

- run_live(...)        -> single-agent viewer
- run_live_multi(...)  -> multi-agent viewer with seed planting, pantry, and doubling reproduction

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


# ----------------------------- Single-agent ---------------------------------


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
    norm = colors.BoundaryNorm([0, 1, 2, 3, 4], cmap.N)

    fig, ax = plt.subplots(figsize=(width / 8, height / 8))
    try:
        fig.canvas.manager.set_window_title("Evo Beings — Live MVP")
    except Exception:
        pass

    img = ax.imshow(world.materials * 1, cmap=cmap, norm=norm, interpolation="nearest")
    dot = ax.scatter([agent.pos[1]], [agent.pos[0]], s=50, c="#fdd835")
    hud = ax.text(2, 1, "", color="white", fontsize=8)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.pause(0.001)

    # click-to-plant seeds
    def on_click(ev):
        if ev.inaxes is not ax or ev.xdata is None or ev.ydata is None:
            return
        y = int(round(ev.ydata))
        x = int(round(ev.xdata))
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

        plt.pause(0.001)
        time.sleep(delay)

    plt.show()


# ---------------------------- Multi-agent -----------------------------------


def run_live_multi(
    n_agents: int = 1,
    width: int = 64,
    height: int = 40,
    steps: int = 1200,
    seed: int = 21,
    fps: int = 18,
    comm_radius: int = 3,
) -> None:
    """
    Multi-agent viewer with:
      - click-to-plant seeds
      - central pantry (deposit here)
      - **doubling reproduction**: when store >= COST_PER_NEW * pop -> spawn `pop` new agents
    """
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
    norm = colors.BoundaryNorm([0, 1, 2, 3, 4], cmap.N)

    fig, ax = plt.subplots(figsize=(width / 8, height / 8))
    try:
        fig.canvas.manager.set_window_title("Evo Beings — Live Multi")
    except Exception:
        pass

    img = ax.imshow(world.materials * 1, cmap=cmap, norm=norm, interpolation="nearest")

    # distinct colors per agent
    colormap = plt.cm.get_cmap("tab10", max(n_agents, 10))
    agent_colors = [colormap(i % colormap.N) for i in range(n_agents)]

    ys = [a.pos[0] for a in agents]
    xs = [a.pos[1] for a in agents]
    scat = ax.scatter(xs, ys, s=50, c=agent_colors)

    hud = ax.text(2, 1, "", color="white", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.pause(0.001)

    # click-to-plant seeds
    def on_click(ev):
        if ev.inaxes is not ax or ev.xdata is None or ev.ydata is None:
            return
        y = int(round(ev.ydata))
        x = int(round(ev.xdata))
        y = max(0, min(world.cfg.height - 1, y))
        x = max(0, min(world.cfg.width - 1, x))
        world.place_seed(y, x)

    fig.canvas.mpl_connect("button_press_event", on_click)

    # --- reproduction cost (per newborn) ---
    COST_PER_NEW = 10  # tune to taste

    def try_spawn_near_pantry() -> bool:
        """Attempt to spawn one agent on a free non-pantry tile near the pantry."""
        seed_rng = np.random.default_rng(world.tick * 17 + len(agents))
        for _ in range(100):  # try offsets around pantry
            oy, ox = int(seed_rng.integers(-2, 3)), int(seed_rng.integers(-2, 3))
            ny = int(np.clip(base_y + oy, 0, world.cfg.height - 1))
            nx = int(np.clip(base_x + ox, 0, world.cfg.width - 1))
            if world.materials[ny, nx] == 3:
                continue  # don't place on pantry
            if all(a.pos != (ny, nx) for a in agents):
                agents.append(Agent(pos=(ny, nx)))
                agent_colors.append(colormap(len(agent_colors) % colormap.N))
                return True
        return False  # no space found

    delay = 1.0 / max(1, fps)
    for _ in range(steps):
        # act
        for i, a in enumerate(agents):
            inbox = world.neighbor_messages(agents, i, radius=comm_radius)
            a.act(world, inbox)
        world.step()

        # --- batch reproduction: DOUBLE the population when the store allows ---
        # Each doubling costs COST_PER_NEW * current_pop
        doubled = 0
        while True:
            pop = len(agents)
            need = COST_PER_NEW * pop
            if world.shared_store < need:
                break

            # consume resources for this doubling
            world.shared_store -= need

            # spawn `pop` new agents (double pop)
            spawned = 0
            for _ in range(pop):
                if try_spawn_near_pantry():
                    spawned += 1
                else:
                    # couldn't place all; refund unused cost
                    world.shared_store += COST_PER_NEW * (pop - spawned)
                    break

            if spawned == 0:
                # no space at all; stop attempting this tick
                break

            doubled += 1

        # draw frame
        img.set_data(world.materials * 1)
        ys = [a.pos[0] for a in agents]
        xs = [a.pos[1] for a in agents]
        scat.set_offsets(np.c_[xs, ys])
        scat.set_color(agent_colors)

        # simple chatter proxy (if your Agent sets last_msg)
        chatter = sum((getattr(a, "last_msg", np.zeros(4)) > 0.6).sum() for a in agents)
        hud.set_text(
            f"tick {world.tick} | agents {len(agents)} | store {world.shared_store} | doubled x{doubled} | chatter {int(chatter)}"
        )

        plt.pause(0.001)
        time.sleep(delay)

    plt.show()
