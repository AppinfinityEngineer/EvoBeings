"""
Live viewers using matplotlib.

- run_live(...)        -> single-agent viewer
- run_live_multi(...)  -> multi-agent viewer with:
    * seed/tree/fiber/rock planting via hotkeys
    * pantry/base in center
    * agents harvest/craft/place markers/deposit
    * batch doubling reproduction when store threshold is met

Hotkeys / mouse:
  1 = Seed brush (teal)       (click to place)
  2 = Tree brush (wood)       (click to place)
  3 = Fiber brush (purple)    (click to place)
  4 = Rock brush (stone)      (click to place)
  0 = Eraser                  (right-click always erases)
  Left click  = place current brush
  Right click = erase (empty)
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

    cmap = colors.ListedColormap([
        "#2b2b2b",  # 0 empty
        "#2e7d32",  # 1 food
        "#00897b",  # 2 seed
        "#ffffff",  # 3 pantry
        "#8d6e63",  # 4 tree (wood)
        "#7e57c2",  # 5 fiber bush
        "#90a4ae",  # 6 rock
        "#ffab00",  # 7 structure/marker
    ])
    norm = colors.BoundaryNorm(list(range(9)), cmap.N)

    fig, ax = plt.subplots(figsize=(width / 8, height / 8))
    try: fig.canvas.manager.set_window_title("Evo Beings — Live MVP")
    except Exception: pass

    img = ax.imshow(world.materials * 1, cmap=cmap, norm=norm, interpolation="nearest")
    dot = ax.scatter([agent.pos[1]], [agent.pos[0]], s=50, c="#fdd835")
    hud = ax.text(2, 1, "", color="white", fontsize=8)

    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.pause(0.001)

    current_brush = 1  # default: seed

    def on_key(ev):
        nonlocal current_brush
        if ev.key == "1": current_brush = 2  # seed code
        elif ev.key == "2": current_brush = 4  # tree
        elif ev.key == "3": current_brush = 5  # fiber
        elif ev.key == "4": current_brush = 6  # rock
        elif ev.key == "0": current_brush = 0  # eraser

    def on_click(ev):
        if ev.inaxes is not ax or ev.xdata is None or ev.ydata is None:
            return
        y = int(round(ev.ydata)); x = int(round(ev.xdata))
        y = max(0, min(world.cfg.height - 1, y))
        x = max(0, min(world.cfg.width - 1, x))
        if ev.button == 3:  # right click erase
            world.materials[y, x] = 0
            return
        # left click: place according to brush
        if current_brush == 2: world.place_seed(y, x)
        elif current_brush == 4: world.place_tree(y, x)
        elif current_brush == 5: world.place_fiber(y, x)
        elif current_brush == 6: world.place_rock(y, x)

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)

    delay = 1.0 / max(1, fps)
    for _ in range(steps):
        agent.act(world)
        world.step()

        img.set_data(world.materials * 1)
        dot.set_offsets(np.c_[[agent.pos[1]], [agent.pos[0]]])
        hud.set_text(f"tick {world.tick} | store {world.shared_store} | brush {current_brush}")
        plt.pause(0.001); time.sleep(delay)
    plt.show()

# ---------------------------- Multi-agent -----------------------------------

def run_live_multi(
    n_agents: int = 1,
    width: int = 64,
    height: int = 40,
    steps: int = 2000,
    seed: int = 21,
    fps: int = 18,
    comm_radius: int = 3,
) -> None:
    """
    Multi-agent viewer:
      - Seed/Tree/Fiber/Rock brushes via 1/2/3/4 keys, 0=eraser. Left-click to place; right-click to erase.
      - Pantry in center; agents deposit food there.
      - Doubling reproduction: when store >= COST_PER_NEW * pop -> spawn `pop` new agents.
      - Agents may craft a tool (wood+wood+fiber) and sometimes place structures (using stone or wood).
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

    cmap = colors.ListedColormap([
        "#2b2b2b",  # 0 empty
        "#2e7d32",  # 1 food
        "#00897b",  # 2 seed
        "#ffffff",  # 3 pantry
        "#8d6e63",  # 4 tree (wood)
        "#7e57c2",  # 5 fiber bush
        "#90a4ae",  # 6 rock
        "#ffab00",  # 7 structure/marker
    ])
    norm = colors.BoundaryNorm(list(range(9)), cmap.N)

    fig, ax = plt.subplots(figsize=(width / 8, height / 8))
    try: fig.canvas.manager.set_window_title("Evo Beings — Live Multi")
    except Exception: pass

    img = ax.imshow(world.materials * 1, cmap=cmap, norm=norm, interpolation="nearest")

    # distinct colors per agent
    colormap = plt.cm.get_cmap("tab10", max(n_agents, 10))
    agent_colors = [colormap(i % colormap.N) for i in range(n_agents)]
    ys = [a.pos[0] for a in agents]; xs = [a.pos[1] for a in agents]
    scat = ax.scatter(xs, ys, s=50, c=agent_colors)

    hud = ax.text(2, 1, "", color="white", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.pause(0.001)

    # brushes
    current_brush = 2  # seed
    def on_key(ev):
        nonlocal current_brush
        if ev.key == "1": current_brush = 2   # seed
        elif ev.key == "2": current_brush = 4 # tree
        elif ev.key == "3": current_brush = 5 # fiber
        elif ev.key == "4": current_brush = 6 # rock
        elif ev.key == "0": current_brush = 0 # eraser
    def on_click(ev):
        if ev.inaxes is not ax or ev.xdata is None or ev.ydata is None:
            return
        y = int(round(ev.ydata)); x = int(round(ev.xdata))
        y = max(0, min(world.cfg.height - 1, y))
        x = max(0, min(world.cfg.width - 1, x))
        if ev.button == 3:
            world.materials[y, x] = 0
            return
        if current_brush == 2: world.place_seed(y, x)
        elif current_brush == 4: world.place_tree(y, x)
        elif current_brush == 5: world.place_fiber(y, x)
        elif current_brush == 6: world.place_rock(y, x)

    fig.canvas.mpl_connect("key_press_event", on_key)
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
        doubled = 0
        while True:
            pop = len(agents)
            need = COST_PER_NEW * pop
            if world.shared_store < need:
                break
            world.shared_store -= need
            spawned = 0
            for _ in range(pop):
                if try_spawn_near_pantry():
                    spawned += 1
                else:
                    world.shared_store += COST_PER_NEW * (pop - spawned)
                    break
            if spawned == 0:
                break
            doubled += 1

        # draw frame
        img.set_data(world.materials * 1)
        ys = [a.pos[0] for a in agents]; xs = [a.pos[1] for a in agents]
        scat.set_offsets(np.c_[xs, ys]); scat.set_color(agent_colors)

        chatter = sum((getattr(a, "last_msg", np.zeros(4)) > 0.6).sum() for a in agents)
        hud.set_text(
            f"tick {world.tick} | agents {len(agents)} | store {world.shared_store} | doubled x{doubled} | brush {current_brush} | chatter {int(chatter)}"
        )

        plt.pause(0.001); time.sleep(delay)

    plt.show()
