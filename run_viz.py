from evo_beings.viewer import run_live_multi  # or run_live

if __name__ == "__main__":
    # Infinite run (until window closed) with online learning & auto-growth
    run_live_multi(n_agents=1, width=64, height=40, seed=21, fps=18)
