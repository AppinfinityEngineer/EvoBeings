from evo_beings.viewer import run_live_multi

if __name__ == "__main__":
    print("Launching MULTI viewer…")
    run_live_multi(n_agents=10, width=64, height=40, steps=1200, seed=21, fps=18)
