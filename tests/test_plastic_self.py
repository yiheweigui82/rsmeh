"""Regression tests for v0.4 (plastic self).  stdlib only -- no pytest.

    python -m unittest discover -s tests -v

Each test pins a claim the docs make, so a break here means a claim broke:

  * the three design conditions that make the second-order target non-redundant:
      (a) the hidden modulator shapes my FUTURE prediction,
      (b) it leaves my CURRENT prediction untouched,
      (c) without the fast pathway it is irrelevant (the causal control);
  * the world modes mean what they say (`static` m == 0, `visible` exposes m);
  * the fast-weight pathway is stable (the first version diverged to 1e17);
  * the metric helpers refuse to score a constant target;
  * a short end-to-end run reproduces the repair's qualitative relations.
"""

import os
import sys
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments", "v04"))

from agent.plastic_agent import NoFastAgent, PlasticAgent      # noqa: E402
from agent.plastic_metrics import _r2                           # noqa: E402
from environment.plastic_world import PlasticWorld              # noqa: E402
from train_v04 import evaluate, train                           # noqa: E402


def rollout(gain, ctor=PlasticAgent, T=12, B=4):
    torch.manual_seed(0)
    model = ctor(1, 16)
    obs = torch.as_tensor(np.random.default_rng(0).normal(size=(T, B, 1)) * 0.2,
                          dtype=torch.float32)
    with torch.no_grad():
        return model.rollout(obs, torch.zeros(T, B), torch.full((T, B), float(gain)))


class DesignConditions(unittest.TestCase):
    """(a) future, (b) not present, (c) unobservable + causally required."""

    def test_modulator_leaves_the_current_prediction_untouched(self):
        a, b = rollout(1.0), rollout(3.0)
        self.assertTrue(torch.allclose(a["pred"][0], b["pred"][0]),
                        "condition (b) violated: the modulator changed p_t")

    def test_modulator_shapes_the_future_prediction(self):
        a, b = rollout(1.0), rollout(3.0)
        self.assertFalse(torch.allclose(a["pred"][-1], b["pred"][-1]),
                         "condition (a) violated: the modulator changed nothing")

    def test_without_the_fast_pathway_the_modulator_is_irrelevant(self):
        a, b = rollout(1.0, NoFastAgent), rollout(3.0, NoFastAgent)
        self.assertTrue(torch.allclose(a["pred"][-1], b["pred"][-1]),
                        "the control must be insensitive to m")


class Worlds(unittest.TestCase):
    def test_static_has_no_modulator(self):
        w = PlasticWorld("static")
        w.reset(4, np.random.default_rng(0))
        self.assertTrue(np.allclose(w.plasticity_gain(), 1.0))

    def test_visible_exposes_the_modulator(self):
        self.assertEqual(PlasticWorld("visible").reset(2, np.random.default_rng(0)).shape[1], 2)
        self.assertEqual(PlasticWorld("plastic").reset(2, np.random.default_rng(0)).shape[1], 1)

    def test_modulator_stays_bounded(self):
        w = PlasticWorld("plastic")
        w.reset(256, np.random.default_rng(0))
        for _ in range(60):
            w.step(np.zeros(256))
        self.assertLess(float(np.abs(w.modulator).max()), 3.0)


class Metrics(unittest.TestCase):
    def test_constant_target_is_undefined(self):
        self.assertNotEqual(_r2(np.arange(8.0)[:, None], np.ones(8)),
                            _r2(np.arange(8.0)[:, None], np.ones(8)))

    def test_fast_weights_do_not_diverge(self):
        """The first implementation reached 1e17 in 40 iterations."""
        model = PlasticAgent(1, 16)
        T, B = 40, 8
        obs = torch.as_tensor(np.random.default_rng(1).normal(size=(T, B, 1)) * 0.2,
                              dtype=torch.float32)
        mod = torch.full((T, B), 2.5)
        with torch.no_grad():
            rec = model.rollout(obs, torch.zeros(T, B), mod)
        self.assertTrue(torch.isfinite(rec["pred"]).all())
        self.assertLess(float(rec["pred"].abs().max()), 10.0)


class EndToEnd(unittest.TestCase):
    """The repair, at toy scale: runs in ~20 s and must reproduce its relations."""

    def test_tiny_run_reproduces_the_relations(self):
        world = PlasticWorld("plastic", seed=1)
        torch.manual_seed(1)
        model = PlasticAgent(world.obs_dim, 32)
        train(world, model, iters=120, batch=32, T=30, seed=1)
        m = evaluate(world, model, np.random.default_rng(7), batch=64, T=30)
        self.assertTrue(np.isfinite(m["worldErr"]))
        self.assertLess(m["selfPredErr"], m["trivialSelfErr"],
                        "the second-order head must beat 'predict the same again'")
        self.assertGreater(m["incrR2"], 3 * max(m["incrR2_null"], 1e-3))
        self.assertGreater(m["mInfo"], 0.02)
        self.assertGreater(m["mInfo"], 5 * max(m["mInfo_null"], 1e-3))


if __name__ == "__main__":
    unittest.main()
