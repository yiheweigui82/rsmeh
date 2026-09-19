"""Regression tests for v0.5 (multi-embodiment reality).  stdlib only.

    python -m unittest discover -s tests -v

Pins the claims the protocol rests on: one causal process behind every modality,
no embodiment observing the shared factors directly, a clean CCA value range, an
error helper that pairs t with t+1 (flattened order silently paired neighbours),
and the shared-vs-disjoint contrast that the whole design depends on.
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments", "v05"))

from compare_reality import cca_corrs, r2_multi, translation   # noqa: E402
from environments.universe import Universe                      # noqa: E402
from train_v05 import norm_pred_err, train_all, collect         # noqa: E402


class UniverseStructure(unittest.TestCase):
    def test_shared_mode_has_one_block_for_every_modality(self):
        ep = Universe("shared", seed=3).rollout(24, 24, np.random.default_rng(3))
        for name in ("human", "ai", "abstract"):
            self.assertTrue(np.allclose(ep["blocks"][name], ep["shared_factors"]))

    def test_disjoint_mode_has_no_common_block(self):
        ep = Universe("disjoint", seed=3).rollout(24, 24, np.random.default_rng(3))
        same = [np.allclose(ep["blocks"][n], ep["shared_factors"])
                for n in ("human", "ai", "abstract")]
        self.assertFalse(all(same))

    def test_no_embodiment_observes_the_factors_directly(self):
        ep = Universe("shared", seed=3).rollout(24, 24, np.random.default_rng(3))
        s = ep["shared_factors"].reshape(-1, 3)
        for name in ("human", "ai", "abstract"):
            obs = ep["obs"][name].reshape(-1, ep["obs"][name].shape[-1])
            self.assertLess(r2_multi(obs, s), 0.98, f"{name} sees the factors")

    def test_interface_dims(self):
        u = Universe("shared", seed=0)
        self.assertEqual(list(u.obs_dims.values()), [3, 4, 3])


class Metrics(unittest.TestCase):
    def test_cca_lives_in_unit_interval(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(400, 6))
        Y = X * 2 + rng.normal(size=(400, 6)) * 0.1
        c = cca_corrs(X, Y)
        self.assertTrue(np.all(c <= 1 + 1e-6) and np.all(c >= -1e-6))

    def test_cca_floor_is_low_for_independent_inputs(self):
        rng = np.random.default_rng(2)
        self.assertLess(float(cca_corrs(rng.normal(size=(400, 6)),
                                        rng.normal(size=(400, 6))).mean()), 0.25)

    def test_error_helper_pairs_each_step_with_the_next(self):
        """A flattened roll paired (t,b) with (t,b+1) and reported 1.78."""
        o = np.zeros((4, 1, 1))
        o[1, 0, 0] = 1.0
        self.assertAlmostEqual(norm_pred_err(o, np.zeros((4, 1, 1))), 16 / 9, places=9)

    def test_translation_rejects_a_shuffled_target(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 5))
        Y = np.concatenate([X, rng.normal(size=(300, 2))], axis=1)
        self.assertGreater(translation(X, Y), 0.3)
        self.assertLess(translation(X, Y[rng.permutation(len(Y))]), 0.02)


class Falsifier(unittest.TestCase):
    """The contrast the design stands on, at toy scale (~30 s)."""

    def test_shared_aligns_and_disjoint_does_not(self):
        out = {}
        for mode in ("shared", "disjoint"):
            u, agents = train_all(mode=mode, iters=90, seed=1)
            rec = collect(u, agents, batch=64, seq=30, seed=8)
            flat = lambda a: a.reshape(-1, a.shape[-1])
            out[mode] = translation(flat(rec["z"]["human"]), flat(rec["z"]["ai"]))
        self.assertGreater(out["shared"], 0.05)
        self.assertLess(out["disjoint"], 0.02)


if __name__ == "__main__":
    unittest.main()
