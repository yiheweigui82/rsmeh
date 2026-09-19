"""Abstract sensor interface: three channels, nonlinearly mixed.

The circulated draft gave this embodiment the hidden factors themselves
(`hidden_vector_1`, `hidden_vector_2`, `causal_structure`), which would have made
"the abstract agent recovers the hidden factors best" true by construction.
Here it receives a **nonlinear** (tanh) mixing of the same two shared directions
plus one abstract private variable -- a genuinely different interface, strictly
partial like the others.
"""

from __future__ import annotations

from .universe import sensor_slice


class AbstractSensorWorld:
    name = "abstract"

    def __init__(self, universe):
        self.universe = universe
        self.obs_dim = universe.obs_dims["abstract"]
        assert self.obs_dim == 3, self.obs_dim

    def observe(self, episode, t):
        return episode["obs"]["abstract"][t]

    @staticmethod
    def slices():
        return sensor_slice("abstract")

    def describe(self):
        return ("mathematical state variables: a tanh mixing of two shared "
                "directions plus one abstract private variable")
