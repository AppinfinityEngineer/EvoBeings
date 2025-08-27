from evo_beings.viewer import run_live_multi  # or run_live

if __name__ == "__main__":
    # Start with one agent; plant seeds/trees/fiber/rocks and watch the colony adapt.
    run_live_multi(n_agents=1, width=64, height=40, steps=3000, seed=21, fps=18)
