#   Copyright (c) 2026, Signaloid.
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to
#   deal in the Software without restriction, including without limitation the
#   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#   sell copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#   DEALINGS IN THE SOFTWARE.

"""Tests of the asymptotic Wasserstein-1 distance distribution and of the
checks on the central limit theorem behind it."""

import unittest

import numpy as np
from scipy.stats import norm  # type: ignore

from signaloid.distributional.distributional import DistributionalValue
from signaloid_benchmarking.config import CltAssumptions
from signaloid_benchmarking.types import TaggedDistributionalValue
from signaloid_benchmarking.equivalent_mc.equivalent_mc_utils import (
    _compute_asymptotic_wasserstein_distribution_mean,
    _compute_wasserstein_1_asymptotic_distribution_brownian_bridge,
    _ground_truth_cdf_knots,
    _scalar_asymptotic_statistics,
    _wasserstein_1_clt_tail_diagnostics,
)


def _weighted(positions: list[float], masses: list[float]) -> TaggedDistributionalValue:
    return TaggedDistributionalValue(
        dv=DistributionalValue.from_weighted_samples(positions, masses)
    )


def _samples(samples: np.ndarray) -> TaggedDistributionalValue:
    return TaggedDistributionalValue(dv=DistributionalValue.from_samples(samples))


class TestGroundTruthCdfKnots(unittest.TestCase):
    """The step CDF is read off the ground truth exactly, then thinned."""

    def test_atoms_are_merged_and_the_cdf_is_cumulative_mass(self) -> None:
        positions, cdf = _ground_truth_cdf_knots(
            _weighted([2.0, 1.0, 1.0, 3.0], [0.25, 0.25, 0.25, 0.25]), max_knots=10
        )

        np.testing.assert_array_equal(positions, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(cdf, [0.5, 0.75, 1.0])

    def test_thinning_keeps_the_ends_and_the_last_cdf_value(self) -> None:
        samples = np.linspace(0.0, 1.0, 101)

        positions, cdf = _ground_truth_cdf_knots(_samples(samples), max_knots=11)

        self.assertEqual(len(positions), 11)
        self.assertEqual(positions[0], 0.0)
        self.assertEqual(positions[-1], 1.0)
        self.assertEqual(cdf[-1], 1.0)
        self.assertTrue(np.all(np.diff(positions) > 0))


class TestWasserstein1AsymptoticDistribution(unittest.TestCase):
    """The simulated ``∫ |B(F(x))| dx`` matches closed forms."""

    def test_two_atom_ground_truth_matches_the_closed_form(self) -> None:
        """With atoms at 1 and 3 of mass p and 1 - p the integral is
        ``2 |B(p)|``, whose mean is ``2 sqrt(2 p (1 - p) / pi)`` and whose
        95th percentile is ``2 * 1.96 sqrt(p (1 - p))``."""
        p = 0.3
        ground_truth = _weighted([1.0, 3.0], [p, 1 - p])
        exact_mean = 2.0 * np.sqrt(2.0 * p * (1 - p) / np.pi)
        exact_q95 = 2.0 * norm.ppf(0.975) * np.sqrt(p * (1 - p))

        self.assertAlmostEqual(
            _compute_asymptotic_wasserstein_distribution_mean(ground_truth, 100),
            exact_mean,
            places=12,
        )
        samples = _compute_wasserstein_1_asymptotic_distribution_brownian_bridge(
            ground_truth, num_points=100, num_samples=20_000, seed=1
        ).positions
        standard_error = samples.std() / np.sqrt(len(samples))
        self.assertLess(abs(samples.mean() - exact_mean), 4 * standard_error)
        self.assertAlmostEqual(np.quantile(samples, 0.95) / exact_q95, 1.0, delta=0.04)

    def test_uniform_ground_truth_mean_is_sqrt_pi_over_2_over_4(self) -> None:
        """For the uniform distribution ``∫ sqrt(F (1 - F)) dx = pi / 8``."""
        ground_truth = _samples(np.linspace(0.0, 1.0, 4001))

        mean = _compute_asymptotic_wasserstein_distribution_mean(ground_truth, 4001)

        self.assertAlmostEqual(mean, np.sqrt(np.pi / 2) / 4, delta=1e-3)

    def test_normal_ground_truth_is_not_biased_by_its_tails(self) -> None:
        """A density-based quantile derivative over-weighted the extreme
        quantile. The CDF form must land on the exact mean, 1.2884 for a
        standard normal, within simulation and thinning error."""
        samples = np.random.default_rng(0).normal(size=100_000)
        ground_truth = _samples(samples)

        simulated = _compute_wasserstein_1_asymptotic_distribution_brownian_bridge(
            ground_truth, num_points=10_000, num_samples=2000, seed=0
        ).positions

        self.assertAlmostEqual(simulated.mean() / 1.2884, 1.0, delta=0.03)

    def test_simulation_is_reproducible_for_a_seed(self) -> None:
        ground_truth = _weighted([0.0, 1.0, 2.0], [0.2, 0.3, 0.5])

        first = _compute_wasserstein_1_asymptotic_distribution_brownian_bridge(
            ground_truth, num_points=10, num_samples=50, seed=7
        ).positions
        second = _compute_wasserstein_1_asymptotic_distribution_brownian_bridge(
            ground_truth, num_points=10, num_samples=50, seed=7
        ).positions
        other = _compute_wasserstein_1_asymptotic_distribution_brownian_bridge(
            ground_truth, num_points=10, num_samples=50, seed=8
        ).positions

        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, other))


class TestTailDiagnostics(unittest.TestCase):
    """The tail diagnostics separate ground truths that meet the CLT
    condition from those that do not."""

    def test_infinite_variance_is_flagged(self) -> None:
        """A Pareto tail with exponent 2 has infinite variance."""
        rng = np.random.default_rng(0)
        ground_truth = _samples(rng.pareto(2.0, size=200_000) + 1.0)

        tail_index, tail_share = _wasserstein_1_clt_tail_diagnostics(ground_truth)

        self.assertLessEqual(tail_index, 2.3)
        self.assertGreater(tail_share, CltAssumptions.TAIL_SHARE_LIMIT)

    def test_normal_tails_pass(self) -> None:
        rng = np.random.default_rng(0)
        ground_truth = _samples(rng.normal(size=200_000))

        tail_index, tail_share = _wasserstein_1_clt_tail_diagnostics(ground_truth)

        self.assertGreater(tail_index, 4.0)
        self.assertLess(tail_share, 0.05)


class TestScalarAsymptoticStatistics(unittest.TestCase):
    """The scalar limit is a folded normal that keeps the fitted mean."""

    def test_unbiased_fit_is_half_normal(self) -> None:
        mean, q95, q99, mean_quantile = _scalar_asymptotic_statistics(0.0, 1.0)

        self.assertAlmostEqual(mean, np.sqrt(2 / np.pi), places=6)
        self.assertAlmostEqual(q95, norm.ppf(0.975), places=6)
        self.assertAlmostEqual(q99, norm.ppf(0.995), places=6)
        self.assertTrue(0.0 < mean_quantile < 1.0)

    def test_bias_widens_the_quantiles(self) -> None:
        _, unbiased_q95, _, _ = _scalar_asymptotic_statistics(0.0, 1.0)
        _, biased_q95, _, _ = _scalar_asymptotic_statistics(1.0, 1.0)

        self.assertGreater(biased_q95, unbiased_q95)


if __name__ == "__main__":
    unittest.main()
