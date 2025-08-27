from evo_beings.viewer import run_live_multi  # or run_live

if __name__ == "__main__":
    # Watch emergent roads/caches/beacons + auto pantry growth every ~2s.
    run_live_multi(n_agents=1, width=64, height=40, steps=4000, seed=21, fps=18)
