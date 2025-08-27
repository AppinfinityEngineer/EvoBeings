"""
World module: defines the environment where beings live and interact.
"""
class World:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.tick = 0

    def step(self):
        """Advance the world simulation by one tick."""
        self.tick += 1
