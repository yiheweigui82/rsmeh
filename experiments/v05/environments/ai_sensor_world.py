"""AI-like sensor interface: four channels.

Two wider-bandwidth readings of the same shared factors plus TWO private
channels (infrared / high-frequency / network data in the draft's language).
Extra private bandwidth is the key difference from the human interface: more
resolution on things the other embodiments cannot see at all.
"""

from __future__ import annotations

from .universe import sensor_slice


class AISensorWorld:
    name = "ai"

    def __init__(self, universe):
        self.universe = universe
        self.obs_dim = universe.obs_dims["ai"]
        assert self.obs_dim == 4, self.obs_dim

    def observe(self, episode, t):
        return episode["obs"]["ai"][t]

    @staticmethod
    def slices():
        return sensor_slice("ai")

    def describe(self):
        return ("infrared / ultraviolet / high-frequency / network: two readings "
                "of the shared factors plus two private channels")
