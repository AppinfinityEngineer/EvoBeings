"""
Run the live MVP viewer.
"""
from evo_beings.viewer import run_live

if __name__ == "__main__":
    run_live(width=64, height=40, steps=800, seed=13, fps=20)
