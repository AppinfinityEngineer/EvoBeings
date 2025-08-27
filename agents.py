"""
Agents module: defines the beings/organisms.
"""
class Agent:
    def __init__(self, name: str, energy: float = 100.0):
        self.name = name
        self.energy = energy

    def act(self, world):
        """Perform one action in the world (to be expanded)."""
        self.energy -= 1
