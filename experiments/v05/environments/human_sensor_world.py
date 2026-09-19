"""Human-like sensor interface: three channels.

Two of them are a coarse linear reading of the universe's shared factors (the
"perceivable band"); one is a private channel no other embodiment receives.
"""

from __future__ import annotations

import numpy as np

from .universe import EMBODIMENTS, sensor_slice


class HumanSensorWorld:
    name = "human"

    def __init__(self, universe):
        self.universe = universe
        self.obs_dim = universe.obs_dims["human"]
        assert self.obs_dim == 3, self.obs_dim

    def observe(self, episode, t):
        """The human interface's reading of the state at time t."""
        return episode["obs"]["human"][t]

    @staticmethod
    def slices():
        return sensor_slice("human")

    def describe(self):
        return ("RGB vision / audio / touch: two coarse linear readings of the "
                "shared factors plus one private (touch) channel")
