"""
Viewer (infinite run) with:
- adaptive growth & doubling reproduction
- pantry protection/auto-repair (P), legend toggle (L), chatter toggle (C), labels toggle (B)
- HUD: agents, store, dynTypes, avgIQ, chatter, brush
- Camera: WASD/arrows/middle-drag pan, +/- or wheel zoom, follow centroid (F), home (H)
"""
import time
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import NullLocator

from .world import World, WorldConfig
from .agents import Agent

COST_PER_NEW = 10
CROWD_CAP = 6

# UI throttles (higher = lighter)
HUD_PERIOD     = 2
LEGEND_PERIOD  = 12
LABEL_PERIOD   = 12
MAX_LABELS     = 20

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

    # --- viewer "one screen" pad (per side) ---
    PAD_ONE_SCREEN = max(8, min(width, height) // 2)

    cmap, norm = _make_cmap(world)
    fig, ax = plt.subplots(figsize=(width / 8, height / 8))
    try: fig.canvas.manager.set_window_title("Evo Beings — Live Multi (Innovation & Learning)")
    except Exception: pass

    ax.set_facecolor("#2b2b2b")
    ax.set_xscale("linear"); ax.set_yscale("linear")
    ax.xaxis.set_major_locator(NullLocator()); ax.yaxis.set_major_locator(NullLocator())

    img = ax.imshow(world.materials * 1, cmap=cmap, norm=norm,
                    interpolation="nearest", origin="upper")
    H0, W0 = world.materials.shape
    img.set_extent((-0.5, W0 - 0.5, H0 - 0.5, -0.5))

    colormap = plt.cm.get_cmap("tab20", max(n_agents, 20))
    agent_colors = [colormap(i % colormap.N) for i in range(n_agents)]
    ys = [a.pos[0] for a in agents]; xs = [a.pos[1] for a in agents]
    scat = ax.scatter(xs, ys, s=50, c=agent_colors)

    hud = ax.text(2, 1, "", color="white", fontsize=8)
    legend_text = ax.text(width - 1, 1, "", color="white", fontsize=7,
                          ha="right", va="top", family="monospace")
    legend_on = True
    show_chatter = True
    show_build_labels = True

    # camera state
    cam_x, cam_y = float(base_x), float(base_y)
    zoom = 1.0
    follow = False

    dragging = False
    drag_start_xy = (0.0, 0.0)
    drag_start_cam = (0.0, 0.0)
    last_dyn_count = len(world.struct_defs)

    def set_view():
        H, W = world.materials.shape
        vw = max(4.0, width / max(1e-6, zoom))
        vh = max(3.0, height / max(1e-6, zoom))
        half_w, half_h = vw * 0.5, vh * 0.5
        if vw >= W or vh >= H:
            ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
            return (W - 1) * 0.5, (H - 1) * 0.5, (W - 1) * 0.5, (H - 1) * 0.5
        left, right = cam_x - half_w, cam_x + half_w
        top, bottom = cam_y - half_h, cam_y + half_h
        if left < -0.5: right += (-0.5 - left); left = -0.5
        if right > W - 0.5: left -= (right - (W - 0.5)); right = W - 0.5
        if top < -0.5: bottom += (-0.5 - top); top = -0.5
        if bottom > H - 0.5: top -= (bottom - (H - 0.5)); bottom = H - 0.5
        ax.set_xlim(left, right); ax.set_ylim(bottom, top)
        return 0.5 * (left + right), 0.5 * (top + bottom), half_w, half_h

    set_view()
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.pause(0.001)

    current_brush = 2

    def _home_camera():
        nonlocal cam_x, cam_y, zoom, follow
        py, px = world.pantry if world.pantry else (world.cfg.height // 2, world.cfg.width // 2)
        cam_x, cam_y, zoom, follow = float(px), float(py), 1.0, False

    def on_key(ev):
        nonlocal current_brush, legend_on, show_chatter, zoom, cam_x, cam_y, follow, show_build_labels
        step = 5.0 / max(1.0, zoom)
        if ev.key == "1": current_brush = 2
        elif ev.key == "2": current_brush = 4
        elif ev.key == "3": current_brush = 5
        elif ev.key == "4": current_brush = 6
        elif ev.key == "5": current_brush = 7
        elif ev.key == "6": current_brush = 8
        elif ev.key == "7": current_brush = 9
        elif ev.key == "0": current_brush = 0
        elif ev.key in ("p", "P"): world.ensure_pantry(base_y, base_x)
        elif ev.key in ("l", "L"): legend_on = not legend_on
        elif ev.key in ("c", "C"): show_chatter = not show_chatter
        elif ev.key in ("b", "B"): show_build_labels = not show_build_labels
        elif ev.key in ("f", "F"): follow = not follow
        elif ev.key in ("h", "H", "home"): _home_camera()
        elif ev.key in ("+", "="): zoom = min(6.0, zoom * 1.2)
        elif ev.key in ("-", "_"): zoom = max(0.5, zoom / 1.2)
        elif ev.key in ("left", "a"):  cam_x -= step; follow = False
        elif ev.key in ("right","d"):  cam_x += step; follow = False
        elif ev.key in ("up",   "w"):  cam_y -= step; follow = False
        elif ev.key in ("down", "s"):  cam_y += step; follow = False

    def on_click(ev):
        nonlocal dragging, drag_start_xy, drag_start_cam, follow
        if ev.inaxes is not ax or ev.xdata is None or ev.ydata is None: return
        if ev.button == 2:
            dragging = True; drag_start_xy = (float(ev.xdata), float(ev.ydata))
            drag_start_cam = (cam_x, cam_y); follow = False; return
        y = int(np.rint(ev.ydata)); x = int(np.rint(ev.xdata))
        y = max(0, min(world.cfg.height - 1, y)); x = max(0, min(world.cfg.width - 1, x))
        if ev.button == 3: world.erase(y, x); return
        if   current_brush == 2: world.place_seed(y, x)
        elif current_brush == 4: world.place_tree(y, x)
        elif current_brush == 5: world.place_fiber(y, x)
        elif current_brush == 6: world.place_rock(y, x)
        elif current_brush == 7: world.place_road(y, x)
        elif current_brush == 8: world.place_cache(y, x)
        elif current_brush == 9: world.place_beacon(y, x)

    def on_release(ev):
        nonlocal dragging
        if ev.button == 2: dragging = False

    def on_motion(ev):
        nonlocal cam_x, cam_y
        if not dragging or ev.inaxes is not ax or ev.xdata is None or ev.ydata is None: return
        sx, sy = drag_start_xy
        dx = sx - float(ev.xdata); dy = sy - float(ev.ydata)
        cam_x = drag_start_cam[0] + dx; cam_y = drag_start_cam[1] + dy

    def on_scroll(ev):
        nonlocal zoom, follow
        if ev.button == "up":   zoom = min(6.0, zoom * 1.1)
        elif ev.button == "down": zoom = max(0.5, zoom / 1.1)
        follow = False

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    delay = 1.0 / max(1, fps)
    grow_every = max(1, int(2 * fps))

    label_artists: List = []

    while plt.fignum_exists(fig.number):
        if world.pantry is None or world.materials[base_y, base_x] != 3:
            world.ensure_pantry(base_y, base_x)

        # expand by ONE SCREEN per event (viewer-sized), world throttles internally
        pad = world.expand_if_needed(agents, margin=8, pad=PAD_ONE_SCREEN)
        if pad > 0:
            cam_x += pad; cam_y += pad
            H, W = world.materials.shape
            img.set_extent((-0.5, W - 0.5, H - 0.5, -0.5))

        # agents act
        for i, a in enumerate(agents):
            inbox = world.neighbor_messages(agents, i, radius=comm_radius)
            a.act(world, inbox)

        world.step()

        # adaptive food near pantry
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

        # follow centroid
        if follow and agents:
            cy = float(np.mean([a.pos[0] for a in agents]))
            cx = float(np.mean([a.pos[1] for a in agents]))
            cam_x, cam_y = cx, cy

        # rebuild cmap only if new dynamic types appeared
        if len(world.struct_defs) != last_dyn_count:
            cmap, norm = _make_cmap(world)
            img.set_cmap(cmap); img.set_norm(norm)
            last_dyn_count = len(world.struct_defs)

        # update raster + agents
        img.set_data(world.materials * 1)
        ys = [a.pos[0] for a in agents]; xs = [a.pos[1] for a in agents]
        scat.set_offsets(np.c_[xs, ys]); scat.set_color(agent_colors)

        cx, cy, half_w, half_h = set_view()

        # HUD (throttled)
        if world.tick % HUD_PERIOD == 0:
            chatter = sum((getattr(a, "last_msg", np.zeros(4)) > 0.6).sum() for a in agents)
            mean_iq = np.mean([a.intelligence for a in agents]) if agents else 0.0
            hud.set_position((cx - half_w + 2, cy - half_h + 1))
            hud.set_text(
                f"tick {world.tick} | agents {len(agents)} | store {world.shared_store} | doubled x{doubled} "
                f"| dynTypes {len(world.struct_defs)} | avgIQ {mean_iq:.2f} | chatter {int(chatter)} | brush {current_brush}"
            )

        # Legend (throttled)
        if legend_on and world.tick % LEGEND_PERIOD == 0:
            legend_text.set_position((cx + half_w - 1, cy - half_h + 1))
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
        elif not legend_on:
            legend_text.set_text("")

        # Labels (throttled)
        if world.tick % LABEL_PERIOD == 0:
            for art in label_artists:
                try: art.remove()
                except Exception: pass
            label_artists.clear()
            if show_build_labels:
                H, W = world.materials.shape
                x0, x1 = ax.get_xlim(); y1, y0 = ax.get_ylim()
                ix0 = max(0, int(np.floor(min(x0, x1)))); ix1 = min(W - 1, int(np.ceil(max(x0, x1))))
                iy0 = max(0, int(np.floor(min(y0, y1)))); iy1 = min(H - 1, int(np.ceil(max(y0, y1))))
                patch = world.materials[iy0:iy1+1, ix0:ix1+1]
                ys_lab, xs_lab = np.where(patch >= 10)
                coords = list(zip(ys_lab + iy0, xs_lab + ix0))
                coords.sort(key=lambda p: abs(p[0]-cy) + abs(p[1]-cx))
                for (yy, xx) in coords[:MAX_LABELS]:
                    code = int(world.materials[yy, xx])
                    nm = world.struct_defs.get(code, {}).get("name", f"type{code}")
                    label_artists.append(ax.text(xx + 0.2, yy - 0.2, nm, color="white", fontsize=6))

            if show_chatter and agents:
                step = max(1, len(agents) // 30)
                for a in agents[::step]:
                    label_artists.append(ax.text(a.pos[1] + 0.3, a.pos[0] - 0.3, a.call_sign,
                                                 color="w", fontsize=6, alpha=0.7))

        plt.pause(0.001); time.sleep(delay)

    plt.close(fig)
