"""
Live viewers using matplotlib.

- run_live(...)        -> single-agent viewer (infinite)
- run_live_multi(...)  -> multi-agent viewer (infinite) with:
    * anti-congestion adaptive auto-growth
    * seed/tree/fiber/rock/road/cache/beacon brushes (1..7,9; 0=erase)
    * batch doubling reproduction when pantry store crosses threshold
"""
import time
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from .world import World, WorldConfig
from .agents import Agent

COST_PER_NEW = 10     # per newborn for doubling
CROWD_CAP = 6         # if >= this many agents near pantry, suppress inner growth


# ----------------------------- Single-agent ---------------------------------

def run_live(
    width: int = 48,
    height: int = 32,
    seed: int = 11,
    fps: int = 30,
) -> None:
    cfg = WorldConfig(width=width, height=height, seed=seed, resource_density=0.0)
    world = World(cfg)

    base_y, base_x = world.cfg.height // 2, world.cfg.width // 2
    world.add_pantry(base_y, base_x)

    rng = np.random.default_rng(seed)
    start = (rng.integers(0, height), rng.integers(0, width))
    agent = Agent(pos=start)

    cmap = colors.ListedColormap([
        "#2b2b2b", "#2e7d32", "#00897b", "#ffffff",
        "#8d6e63", "#7e57c2", "#90a4ae", "#ffab00", "#26c6da", "#ec407a"
    ])
    norm = colors.BoundaryNorm(list(range(11)), cmap.N)

    fig, ax = plt.subplots(figsize=(width / 8, height / 8))
    try: fig.canvas.manager.set_window_title("Evo Beings — Live MVP")
    except Exception: pass

    img = ax.imshow(world.materials * 1, cmap=cmap, norm=norm, interpolation="nearest")
    dot = ax.scatter([agent.pos[1]], [agent.pos[0]], s=50, c="#fdd835")
    hud = ax.text(2, 1, "", color="white", fontsize=8)

    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.pause(0.001)

    current_brush = 2  # seed

    def on_key(ev):
        nonlocal current_brush
        if ev.key == "1": current_brush = 2
        elif ev.key == "2": current_brush = 4
        elif ev.key == "3": current_brush = 5
        elif ev.key == "4": current_brush = 6
        elif ev.key == "5": current_brush = 7
        elif ev.key == "6": current_brush = 8
        elif ev.key == "7": current_brush = 9
        elif ev.key == "0": current_brush = 0

    def on_click(ev):
        if ev.inaxes is not ax or ev.xdata is None or ev.ydata is None:
            return
        y = int(round(ev.ydata)); x = int(round(ev.xdata))
        y = max(0, min(world.cfg.height - 1, y))
        x = max(0, min(world.cfg.width - 1, x))
        if ev.button == 3:
            world.erase(y, x); return
        if   current_brush == 2: world.place_seed(y, x)
        elif current_brush == 4: world.place_tree(y, x)
        elif current_brush == 5: world.place_fiber(y, x)
        elif current_brush == 6: world.place_rock(y, x)
        elif current_brush == 7: world.place_road(y, x)
        elif current_brush == 8: world.place_cache(y, x)
        elif current_brush == 9: world.place_beacon(y, x)

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)

    delay = 1.0 / max(1, fps)
    grow_every = max(1, int(2 * fps))  # ~2 seconds

    # infinite run
    while plt.fignum_exists(fig.number):
        agent.act(world)
        world.step()

        # auto-grow food near pantry (simple, single-agent)
        if world.tick % grow_every == 0:
            world.grow_food_near_pantry(radius=4, k=4)

        img.set_data(world.materials * 1)
        dot.set_offsets(np.c_[[agent.pos[1]], [agent.pos[0]]])
        hud.set_text(f"tick {world.tick} | store {world.shared_store} | brush {current_brush}")
        plt.pause(0.001); time.sleep(delay)

    plt.close(fig)


# ---------------------------- Multi-agent -----------------------------------

def run_live_multi(
    n_agents: int = 1,
    width: int = 64,
    height: int = 40,
    seed: int = 21,
    fps: int = 18,
    comm_radius: int = 3,
) -> None:
    cfg = WorldConfig(width=width, height=height, seed=seed, resource_density=0.0)
    world = World(cfg)

    base_y, base_x = world.cfg.height // 2, world.cfg.width // 2
    world.add_pantry(base_y, base_x)

    rng = np.random.default_rng(seed)

    taken = set()
    agents: List[Agent] = []
    for _ in range(n_agents):
        while True:
            pos = (int(rng.integers(0, height)), int(rng.integers(0, width)))
            if pos not in taken and world.materials[pos[0], pos[1]] != 3:
                taken.add(pos); break
        agents.append(Agent(pos=pos))

    cmap = colors.ListedColormap([
        "#2b2b2b", "#2e7d32", "#00897b", "#ffffff",
        "#8d6e63", "#7e57c2", "#90a4ae", "#ffab00", "#26c6da", "#ec407a"
    ])
    norm = colors.BoundaryNorm(list(range(11)), cmap.N)

    fig, ax = plt.subplots(figsize=(width / 8, height / 8))
    try: fig.canvas.manager.set_window_title("Evo Beings — Live Multi (Adaptive Growth)")
    except Exception: pass

    img = ax.imshow(world.materials * 1, cmap=cmap, norm=norm, interpolation="nearest")

    colormap = plt.cm.get_cmap("tab10", max(n_agents, 10))
    agent_colors = [colormap(i % colormap.N) for i in range(n_agents)]
    ys = [a.pos[0] for a in agents]; xs = [a.pos[1] for a in agents]
    scat = ax.scatter(xs, ys, s=50, c=agent_colors)

    hud = ax.text(2, 1, "", color="white", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.pause(0.001)

    current_brush = 2  # seed
    def on_key(ev):
        nonlocal current_brush
        if ev.key == "1": current_brush = 2
        elif ev.key == "2": current_brush = 4
        elif ev.key == "3": current_brush = 5
        elif ev.key == "4": current_brush = 6
        elif ev.key == "5": current_brush = 7
        elif ev.key == "6": current_brush = 8
        elif ev.key == "7": current_brush = 9
        elif ev.key == "0": current_brush = 0
    def on_click(ev):
        if ev.inaxes is not ax or ev.xdata is None or ev.ydata is None:
            return
        y = int(round(ev.ydata)); x = int(round(ev.xdata))
        y = max(0, min(world.cfg.height - 1, y))
        x = max(0, min(world.cfg.width - 1, x))
        if ev.button == 3:
            world.erase(y, x); return
        if   current_brush == 2: world.place_seed(y, x)
        elif current_brush == 4: world.place_tree(y, x)
        elif current_brush == 5: world.place_fiber(y, x)
        elif current_brush == 6: world.place_rock(y, x)
        elif current_brush == 7: world.place_road(y, x)
        elif current_brush == 8: world.place_cache(y, x)
        elif current_brush == 9: world.place_beacon(y, x)

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)

    # reproduction helper
    def try_spawn_near_pantry() -> bool:
        seed_rng = np.random.default_rng(world.tick * 17 + len(agents))
        for _ in range(200):
            oy, ox = int(seed_rng.integers(-3, 4)), int(seed_rng.integers(-3, 4))
            ny = int(np.clip(base_y + oy, 0, world.cfg.height - 1))
            nx = int(np.clip(base_x + ox, 0, world.cfg.width - 1))
            if world.materials[ny, nx] == 3:
                continue
            if all(a.pos != (ny, nx) for a in agents):
                agents.append(Agent(pos=(ny, nx)))
                agent_colors.append(colormap(len(agent_colors) % colormap.N))
                return True
        return False

    delay = 1.0 / max(1, fps)
    grow_every = max(1, int(2 * fps))  # ~2 seconds

    # infinite run
    while plt.fignum_exists(fig.number):
        # agents act
        for i, a in enumerate(agents):
            inbox = world.neighbor_messages(agents, i, radius=comm_radius)
            a.act(world, inbox)
        world.step()

        # ---- adaptive auto-growth ----
        if world.tick % grow_every == 0:
            # count agents near pantry
            py, px = world.pantry
            near = sum(1 for a in agents if abs(a.pos[0]-py)+abs(a.pos[1]-px) <= 4)
            # if crowded or store already high vs pop, grow in an outer ring
            target = COST_PER_NEW * max(1, len(agents))
            if near >= CROWD_CAP or world.shared_store > target:
                world.grow_food_ring(r_min=5, r_max=9, k=6)
            else:
                world.grow_food_near_pantry(radius=4, k=3)

        # ---- batch reproduction: double pop whenever store allows ----
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

        # draw
        img.set_data(world.materials * 1)
        ys = [a.pos[0] for a in agents]; xs = [a.pos[1] for a in agents]
        scat.set_offsets(np.c_[xs, ys]); scat.set_color(agent_colors)

        # HUD
        py, px = world.pantry
        near = sum(1 for a in agents if abs(a.pos[0]-py)+abs(a.pos[1]-px) <= 4)
        hud.set_text(
            f"tick {world.tick} | agents {len(agents)} | store {world.shared_store} | doubled x{doubled} | nearPantry {near} | brush {current_brush}"
        )

        plt.pause(0.001); time.sleep(delay)

    plt.close(fig)
