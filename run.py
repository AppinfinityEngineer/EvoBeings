"""
CLI entry: run a tiny evolution and print a short report.
"""
from evo_beings.runner import evolve, simulate
from evo_beings.evo import descriptor

if __name__ == "__main__":
    arc = evolve(generations=10, pop=32, dim=32, seed=11)
    print("Archive filled cells:", (arc.grid_fit > -1e9).sum())
