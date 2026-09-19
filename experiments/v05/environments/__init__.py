"""Sensor interfaces -- three embodiments of one universe.

Each file below exposes `observe(state)` in the spirit of the circulated draft,
but the observation is a **slice of the universe's already-generated stream**, so
that all three embodiments provably see one causal process and none of them sees
the shared factors directly.

    human    3 sensors: two coarse linear readings of the shared factors + touch
    ai       4 sensors: two wider-range readings of the shared factors + two
                       private high-bandwidth channels
    abstract 3 sensors: a NONLINEAR (tanh) mixing of two shared directions + one
                       abstract state variable
"""
