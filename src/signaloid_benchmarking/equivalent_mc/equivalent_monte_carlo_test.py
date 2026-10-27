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

import contextlib
import io
import os
import tempfile
import unittest

from signaloid.distributional.distributional import DistributionalValue

from signaloid_benchmarking.config import (
    BenchmarkingVariables,
    EquivMC,
    ReportingMethods,
    VariableTypes,
)
from signaloid_benchmarking.types import (
    BenchmarkingVariable,
    TaggedDistributionalValue,
)
from signaloid_benchmarking.equivalent_mc.equivalent_monte_carlo import (
    EquivalentMonteCarlo,
)


def _make_emcc_with_distances(
    distances_by_count: dict[int, list[float]],
) -> EquivalentMonteCarlo:
    """Build an EquivalentMonteCarlo with only ``_adversary_distances``
    populated, bypassing __init__ for focused unit-testing of the
    closest-key fallback. The plot methods read ``self.n_adversaries``
    via ``math.log2``, so seed it too."""
    emcc = object.__new__(EquivalentMonteCarlo)
    emcc._adversary_distances = distances_by_count
    emcc.n_adversaries = 5
    return emcc


def _scalar_tagged(samples: list[float], mc_count: int) -> TaggedDistributionalValue:
    """A scalar adversary/ground-truth carrier, as produced by the scalar
    loaders in ``load.py`` (DistributionalValue.from_samples + mc_count)."""
    return TaggedDistributionalValue(
        dv=DistributionalValue.from_samples(samples),
        mc_count=mc_count,
    )


class TestFoldedNormalPlot(unittest.TestCase):
    """The scalar comparison plot copes with a deterministic scalar."""

    def test_zero_scale_skips_the_density_curve(self) -> None:
        emcc = _make_emcc_with_distances({1000: [0.0, 0.0, 0.0]})
        emcc.variable = BenchmarkingVariable(
            name="x", description="x variable", type=VariableTypes.SCALAR
        )
        asymptotic = emcc.variable.asymptotic_distribution
        asymptotic.mean = 0.0
        asymptotic.scale = 0.0
        asymptotic.location = 0.0
        emcc.ground_truth = _scalar_tagged([1.0], 1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            prefix = os.path.join(tmp_dir, "plot")
            emcc._plot_equiv_mc_vs_folded_normal(mc_count=1000, prefix=prefix)
            self.assertTrue(any(name.endswith(".png") for name in os.listdir(tmp_dir)))


class TestResolveAdversaryDistances(unittest.TestCase):
    """``_resolve_adversary_distances`` falls back to the closest sampled key."""

    def test_resolve_adversary_distances_exact_match(self) -> None:
        """An exact key returns its list unchanged."""
        emcc = _make_emcc_with_distances({100: [1.0, 2.0], 500: [3.0]})

        self.assertEqual(emcc._resolve_adversary_distances(100), [1.0, 2.0])
        self.assertEqual(emcc._resolve_adversary_distances(500), [3.0])

    def test_resolve_adversary_distances_falls_back_to_closest_key(self) -> None:
        """A missing key resolves to the closest sampled adversary size.

        Regression for the bug surfaced on the Rendering Importance
        Sampling demo: a predicted EMCC of 1363 raised ``KeyError`` because
        the adaptive adversary stepping never sampled exactly 1363. The
        helper must pick the nearest sampled key instead.
        """
        emcc = _make_emcc_with_distances({1000: [1.0], 1500: [2.0], 2000: [3.0]})

        # 1363 is closer to 1500 than to 1000 or 2000.
        self.assertEqual(emcc._resolve_adversary_distances(1363), [2.0])

    def test_resolve_adversary_distances_ties_to_lower_key(self) -> None:
        """``min`` with ``abs`` keys ties to the first key encountered."""
        emcc = _make_emcc_with_distances({100: [1.0], 200: [2.0]})

        # 150 is equidistant from 100 and 200.
        # min() takes the first seen.
        result = emcc._resolve_adversary_distances(150)
        self.assertIn(result, ([1.0], [2.0]))

    def test_resolve_adversary_distances_empty_returns_none(self) -> None:
        """No adversaries have been computed yet → None, no exception."""
        emcc = _make_emcc_with_distances({})

        self.assertIsNone(emcc._resolve_adversary_distances(1))
        self.assertIsNone(emcc._resolve_adversary_distances(50000))


class TestPlotSkipsWhenNoDistances(unittest.TestCase):
    """Plot helpers bail before matplotlib work when no distances exist."""

    def test_plot_equiv_mc_vs_brownian_bridge_skips_when_no_distances(self) -> None:
        """``_plot_equiv_mc_vs_brownian_bridge`` must not raise when
        ``_adversary_distances`` is empty — it should bail before any
        matplotlib work."""
        emcc = _make_emcc_with_distances({})

        emcc._plot_equiv_mc_vs_brownian_bridge(
            mc_count=1363,
            brownian_bridge_integrals=[0.0, 1.0, 2.0],
            prefix="unused",
        )

    def test_plot_equiv_mc_vs_folded_normal_skips_when_no_distances(self) -> None:
        """``_plot_equiv_mc_vs_folded_normal`` must not raise when
        ``_adversary_distances`` is empty."""
        emcc = _make_emcc_with_distances({})

        emcc._plot_equiv_mc_vs_folded_normal(mc_count=1363, prefix="unused")


class TestComputeDistanceDataScalar(unittest.TestCase):
    """Scalar variables with a single-sample adversary must not crash."""

    def test_compute_distance_data_scalar_single_sample_adversary_no_raise(
        self,
    ) -> None:
        """Regression: a scalar variable whose ``adversary_mc[0]`` carries a
        single sample must not crash ``compute_distance_data``.

        A single-sample adversary makes ``adversary_size_max`` collapse to
        ``len(positions) - 1 == 0``. The scalar branch of
        ``_compute_adversary_distances`` never reads the adversary-size array,
        but the array was still generated unconditionally — driving the
        adaptive-step path into ``np.log(0)`` (``math.ceil(-inf)`` raises
        ``OverflowError``). ``_generate_adversary_size_array`` now short-circuits
        to an empty array for non-distribution variables.
        """
        variable = BenchmarkingVariable(
            name="scalarOutput",
            description="Scalar Output",
            type=VariableTypes.SCALAR,
        )
        emcc = EquivalentMonteCarlo(
            ground_truth=_scalar_tagged([1.0], mc_count=1),
            uxhw_data=[_scalar_tagged([1.1], mc_count=1)],
            variable=variable,
            # adversary_mc[0] carries a single sample => adversary_size_max == 0
            adversary_mc=[
                _scalar_tagged([1.2], mc_count=1),
                _scalar_tagged([0.9, 1.1], mc_count=2),
            ],
            n_processes=1,
            n_adversaries=1,
        )

        # adversary_size_max is the degenerate value that previously broke the
        # adaptive-step log so the guard must tolerate it.
        self.assertEqual(emcc.adversary_size_max, 0)

        # Adaptive steps is the path that previously hit log(0).
        emcc.compute_distance_data(use_adaptive_steps=True)

        # The scalar branch ignores the size array (empty for scalars).
        self.assertEqual(len(emcc.adversary_size_array), 0)
        # The scalar distance computation still ran against each adversary.
        self.assertTrue(emcc._adversary_distances)


if __name__ == "__main__":
    unittest.main()


def _make_emcc_for_selection(
    distances_by_count: dict[int, list[float]],
    rows: list[dict],
    ground_truth_samples: int = 0,
) -> EquivalentMonteCarlo:
    """Build an EquivalentMonteCarlo with adversary distances, a variable
    carrying ``rows`` of EMCC data and, when asked, a ground truth of
    ``ground_truth_samples`` positions for the scaling check."""
    emcc = _make_emcc_with_distances(distances_by_count)
    emcc.use_clt = False
    emcc.use_binned_uxhw = False
    emcc.variable = BenchmarkingVariable(name="x", description="x variable")
    emcc.variable.emcc_results.emcc_data = rows
    if ground_truth_samples:
        emcc.ground_truth = TaggedDistributionalValue(
            dv=DistributionalValue.from_samples(
                list(range(ground_truth_samples)),
            )
        )
    return emcc


def _row(distance: float, method: str) -> dict:
    return {
        BenchmarkingVariables.UXHW_CONF: "Athens-32",
        BenchmarkingVariables.UXHW_DISTANCE: distance,
        EquivMC.REPORTING_METHOD: method,
    }


_HALVING_DISTANCES = {
    100: [1.0, 1.1, 0.9],
    200: [0.7, 0.72, 0.68],
    400: [0.5, 0.52, 0.48],
    800: [0.35, 0.36, 0.34],
    1600: [0.25, 0.26, 0.24],
    3200: [0.18, 0.17, 0.19],
}


class TestDetermineEquivMcCounts(unittest.TestCase):
    """The measured EMCC is the smallest adversary size that beats UxHw, and
    the proportion column counts adversaries that beat the UxHw distance."""

    def test_smallest_beating_size_is_selected(self) -> None:
        rows = [
            _row(0.3, ReportingMethods.MEAN),
            _row(0.3, ReportingMethods.QUANTILE_95),
        ]
        emcc = _make_emcc_for_selection(_HALVING_DISTANCES, rows)

        with contextlib.redirect_stdout(io.StringIO()):
            emcc.determine_equiv_mc_counts()

        self.assertEqual(rows[0][EquivMC.EMCC], 1600)
        self.assertEqual(rows[1][EquivMC.EMCC], 1600)

    def test_proportion_compares_adversaries_with_the_uxhw_distance(self) -> None:
        rows = [_row(0.3, ReportingMethods.MEAN), _row(0.255, ReportingMethods.MEAN)]
        emcc = _make_emcc_for_selection(_HALVING_DISTANCES, rows)

        with contextlib.redirect_stdout(io.StringIO()):
            emcc.determine_equiv_mc_counts()

        self.assertAlmostEqual(rows[0][EquivMC.PERCENTAGE_MC_BEATS_UXHW], 1.0)
        self.assertEqual(rows[1][EquivMC.EMCC], 1600)
        self.assertAlmostEqual(rows[1][EquivMC.PERCENTAGE_MC_BEATS_UXHW], 2 / 3)

    def test_no_beating_size_falls_back_to_one(self) -> None:
        rows = [_row(0.1, ReportingMethods.MEAN)]
        emcc = _make_emcc_for_selection(_HALVING_DISTANCES, rows)

        with contextlib.redirect_stdout(io.StringIO()):
            emcc.determine_equiv_mc_counts()

        self.assertEqual(rows[0][EquivMC.EMCC], 1)

    def test_unusable_binned_distance_falls_back_to_the_plain_one(self) -> None:
        """
        With the binned distance asked for, a binned distance of zero would
        let no adversary beat UxHw. The plain distance is used instead, as
        the EMCC prediction and the run summary do.
        """
        rows = [
            {
                **_row(0.3, ReportingMethods.MEAN),
                BenchmarkingVariables.UXHW_BINNED_DISTANCE: 0.0,
            }
        ]
        emcc = _make_emcc_for_selection(_HALVING_DISTANCES, rows)
        emcc.use_binned_uxhw = True

        with contextlib.redirect_stdout(io.StringIO()):
            emcc.determine_equiv_mc_counts()

        self.assertEqual(rows[0][EquivMC.EMCC], 1600)


class TestSqrtNScalingCheck(unittest.TestCase):
    """``sqrt(N)`` times the mean adversary distance must be constant."""

    def test_constant_scaling_passes(self) -> None:
        emcc = _make_emcc_for_selection(
            {100: [1.0, 1.0], 400: [0.5, 0.5]}, [], ground_truth_samples=20_000
        )

        emcc._check_sqrt_n_scaling()

        ratio = emcc.variable.asymptotic_distribution.scaling_ratio
        assert ratio is not None
        self.assertAlmostEqual(ratio, 1.0)
        self.assertEqual(emcc.variable.clt_warnings(), [])

    def test_broken_scaling_is_flagged(self) -> None:
        emcc = _make_emcc_for_selection(
            {100: [1.0], 400: [0.7]}, [], ground_truth_samples=20_000
        )

        with contextlib.redirect_stdout(io.StringIO()) as output:
            emcc._check_sqrt_n_scaling()

        ratio = emcc.variable.asymptotic_distribution.scaling_ratio
        assert ratio is not None
        self.assertAlmostEqual(ratio, 1.4)
        self.assertIn("sqrt(N)", output.getvalue())
        self.assertEqual(len(emcc.variable.clt_warnings()), 1)

    def test_sizes_near_the_ground_truth_size_are_ignored(self) -> None:
        """Above a tenth of the ground truth the ground truth's own error
        inflates the distances, so such sizes do not take part."""
        emcc = _make_emcc_for_selection(
            {100: [1.0], 400: [0.5], 5000: [0.5]}, [], ground_truth_samples=20_000
        )

        emcc._check_sqrt_n_scaling()

        ratio = emcc.variable.asymptotic_distribution.scaling_ratio
        assert ratio is not None
        self.assertAlmostEqual(ratio, 1.0)

    def test_deterministic_ground_truth_counts_as_constant_scaling(self) -> None:
        """Every adversary distance is 0 for a deterministic ground truth, so
        0 / 0 must read as a ratio of 1 rather than raise."""
        emcc = _make_emcc_for_selection(
            {100: [0.0, 0.0], 400: [0.0]}, [], ground_truth_samples=20_000
        )

        emcc._check_sqrt_n_scaling()

        self.assertEqual(emcc.variable.asymptotic_distribution.scaling_ratio, 1.0)
        self.assertEqual(emcc.variable.clt_warnings(), [])

    def test_positive_over_zero_is_flagged(self) -> None:
        emcc = _make_emcc_for_selection(
            {100: [0.0], 400: [0.5]}, [], ground_truth_samples=20_000
        )

        with contextlib.redirect_stdout(io.StringIO()):
            emcc._check_sqrt_n_scaling()

        self.assertEqual(
            emcc.variable.asymptotic_distribution.scaling_ratio, float("inf")
        )
        self.assertEqual(len(emcc.variable.clt_warnings()), 1)

    def test_sizes_too_close_together_are_not_compared(self) -> None:
        emcc = _make_emcc_for_selection(
            {100: [1.0], 200: [0.5]}, [], ground_truth_samples=20_000
        )

        emcc._check_sqrt_n_scaling()

        self.assertIsNone(emcc.variable.asymptotic_distribution.scaling_ratio)
