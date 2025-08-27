"""
Viewer (infinite run) with:
- adaptive growth & doubling reproduction
- pantry protection/auto-repair (press 'P' to restore)
- dynamic structure legend (press 'L' to toggle)
- HUD includes average agent intelligence again

Controls:
  1=Seed  2=Tree  3=Fiber  4=Rock  5=Road  6=Cache  7=Beacon  0=Eraser
  P=Restore pantry   L=Toggle legend
"""
import time
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from .world import World, WorldConfig
from .agents import Agent

COST_PER_NEW = 10
CROWD_CAP = 6


def _make_cmap(world: World):
    base = [
        "#2b2b2b", "#2e7d32", "#00897b", "#ffffff",
        "#8d6e63", "#7e57c2", "#90a4ae", "#ffab00",
        "#26c6da", "#ec407a"
    ]
    dyn_max = max([9] + list(world.struct_defs.keys()))
    colors_list = list(base)
    for code in range(10, dyn_max + 1):
        colors_list.append(world.struct_defs.get(code, {"color": "#616161"})["color"])
    cmap = colors.ListedColormap(colors_list)
    norm = colors.BoundaryNorm(list(range(len(colors_list) + 1)), cmap.N)
    return cmap, norm


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

    base_y, base_x = height // 2, width // 2
    world.add_pantry(base_y, base_x)

    rng = np.random.default_rng(seed)

    agents: List[Agent] = []
    taken = set()
    for _ in range(n_agents):
        while True:
            pos = (int(rng.integers(0, height)), int(rng.integers(0, width)))
            if pos not in taken and world.materials[pos[0], pos[1]] != 3:
                taken.add(pos); break
        agents.append(Agent(pos=pos))

    cmap, norm = _make_cmap(world)
    fig, ax = plt.subplots(figsize=(width / 8, height / 8))
    try: fig.canvas.manager.set_window_title("Evo Beings — Live Multi (Innovation & Learning)")
    except Exception: pass

    img = ax.imshow(world.materials * 1, cmap=cmap, norm=norm, interpolation="nearest")
    colormap = plt.cm.get_cmap("tab20", max(n_agents, 20))
    agent_colors = [colormap(i % colormap.N) for i in range(n_agents)]
    ys = [a.pos[0] for a in agents]; xs = [a.pos[1] for a in agents]
    scat = ax.scatter(xs, ys, s=50, c=agent_colors)

    hud = ax.text(2, 1, "", color="white", fontsize=8)
    legend_text = ax.text(width - 1, 1, "", color="white", fontsize=7,
                          ha="right", va="top", family="monospace")
    legend_on = True

    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.pause(0.001)

    current_brush = 2  # seed

    def on_key(ev):
        nonlocal current_brush, legend_on
        if ev.key == "1": current_brush = 2
        elif ev.key == "2": current_brush = 4
        elif ev.key == "3": current_brush = 5
        elif ev.key == "4": current_brush = 6
        elif ev.key == "5": current_brush = 7
        elif ev.key == "6": current_brush = 8
        elif ev.key == "7": current_brush = 9
        elif ev.key == "0": current_brush = 0
        elif ev.key in ("p", "P"):
            world.ensure_pantry(base_y, base_x)
        elif ev.key in ("l", "L"):
            legend_on = not legend_on

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

    while plt.fignum_exists(fig.number):
        # ensure pantry present
        if world.pantry is None or world.materials[base_y, base_x] != 3:
            world.ensure_pantry(base_y, base_x)

        # agents act
        for i, a in enumerate(agents):
            inbox = world.neighbor_messages(agents, i, radius=comm_radius)
            a.act(world, inbox)

        world.step()

        # adaptive growth
        if world.tick % grow_every == 0:
            py, px = world.pantry
            near = sum(1 for a in agents if abs(a.pos[0]-py)+abs(a.pos[1]-px) <= 4)
            target = COST_PER_NEW * max(1, len(agents))
            if near >= CROWD_CAP or world.shared_store > target:
                world.grow_food_ring(r_min=5, r_max=9, k=6)
            else:
                world.grow_food_near_pantry(radius=4, k=3)

        # doubling reproduction
        doubled = 0
        while world.shared_store >= COST_PER_NEW * len(agents):
            world.shared_store -= COST_PER_NEW * len(agents)
            spawned = 0
            for _ in range(len(agents)):
                oy, ox = int(np.random.randint(-3, 4)), int(np.random.randint(-3, 4))
                ny = int(np.clip(base_y + oy, 0, world.cfg.height - 1))
                nx = int(np.clip(base_x + ox, 0, world.cfg.width - 1))
                if world.materials[ny, nx] == 3: continue
                if all(a.pos != (ny, nx) for a in agents):
                    agents.append(Agent(pos=(ny, nx)))
                    agent_colors.append(colormap(len(agent_colors) % colormap.N))
                    spawned += 1
            if spawned == 0:
                world.shared_store += COST_PER_NEW * len(agents)
                break
            doubled += 1

        # refresh cmap if new dynamic types appeared
        cmap, norm = _make_cmap(world)
        img.set_cmap(cmap); img.set_norm(norm)
        img.set_data(world.materials * 1)

        ys = [a.pos[0] for a in agents]; xs = [a.pos[1] for a in agents]
        scat.set_offsets(np.c_[xs, ys]); scat.set_color(agent_colors)

        # HUD: avg intelligence back on screen
        mean_iq = np.mean([a.intelligence for a in agents]) if agents else 0.0
        hud.set_text(
            f"tick {world.tick} | agents {len(agents)} | store {world.shared_store} | doubled x{doubled} | dynTypes {len(world.struct_defs)} | avgIQ {mean_iq:.2f} | brush {current_brush}"
        )

        # Legend: invented types with names
        if legend_on:
            lines = ["[structures]"]
            if world.struct_defs:
                for code, props in sorted(world.struct_defs.items()):
                    nm = props.get("name", f"type{code}")
                    parts = []
                    if "move_mult" in props: parts.append("move")
                    if props.get("cache"):  parts.append("cache")
                    if "comms_bonus" in props: parts.append(f"comms+{props['comms_bonus']}")
                    if "food_boost" in props: parts.append(f"food+{props['food_boost']}")
                    if "iq_aura" in props: parts.append(f"iq+{props['iq_aura']}")
                    if "emit" in props: parts.append("emit")
                    lines.append(f"{code}: {nm} ({', '.join(parts)})")
            else:
                lines.append("(none yet)")
            legend_text.set_text("\n".join(lines))
        else:
            legend_text.set_text("")

        plt.pause(0.001); time.sleep(delay)

    plt.close(fig)
