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

"""
Unit tests for what ``compute_uxhw_distances`` loads for each variable.

A variable with its own analytic reference reads it from the weighted
samples table, whatever the run-wide ground-truth type, and each variable
loads the UxHw results of its own matrix. The loaders are replaced with
mocks that record what each variable asked for.
"""

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from signaloid_benchmarking.automation.analysis import compute_uxhw_distances
from signaloid_benchmarking.config import DistanceMetrics, RepresentationTypes
from signaloid_benchmarking.types import BenchmarkingVariable

ANALYSIS = "signaloid_benchmarking.automation.analysis"


class TestComputeUxhwDistancesPerVariable(unittest.TestCase):
    """The ground-truth table and the UxHw matrix of each variable."""

    def setUp(self) -> None:
        self.load_ground_truth = self._patch("_load_ground_truth")
        self.load_uxhw = self._patch("_load_uxhw_distributions", return_value=[])
        self._patch("write_uxhw_distance_file")

        analytic = BenchmarkingVariable(name="analytic", description="analytic")
        analytic.analytic_ground_truth = "/app/ground-truth.py"
        analytic.representation_types = ["Athens"]
        analytic.representation_sizes = [8]
        monte_carlo = BenchmarkingVariable(name="monte_carlo", description="mc")
        all_died = BenchmarkingVariable(name="all_died", description="died")
        all_died.skipped_configurations = {
            "Jupiter-32": "its tracing run was killed by signal 9 (SIGKILL), or exited with code 137",
            "Jupiter-64": "its tracing run was killed by signal 9 (SIGKILL), or exited with code 137",
        }

        compute_uxhw_distances(
            benchmarking_variables=[analytic, monte_carlo, all_died],
            distance_type=DistanceMetrics.WASSERSTEIN_1,
            ground_truth_db_path="ground-truth.db",
            ground_truth_type=RepresentationTypes.MONTE_CARLO,
            tracing_db_path="tracing.db",
            representation_types=["Jupiter"],
            representation_sizes=[32, 64],
            correlations=["Disabled"],
            uxhw_distance_file="uxhw-distances.csv",
        )

    def _patch(self, name: str, **kwargs: Any) -> MagicMock:
        """
        Replace one name in the analysis module for this test.

        Args:
            name: The name to replace.
            **kwargs: Passed to the mock.

        Returns:
            The mock.
        """
        patcher = patch(f"{ANALYSIS}.{name}", **kwargs)
        mock: MagicMock = patcher.start()
        self.addCleanup(patcher.stop)

        return mock

    def _ground_truth_request(self, name: str) -> dict[str, Any]:
        """
        Find the ground-truth load one variable made.

        Args:
            name: The variable's name.

        Returns:
            The keyword arguments of that load.
        """
        return next(
            dict(call.kwargs)
            for call in self.load_ground_truth.call_args_list
            if call.kwargs["target_expression"] == name
        )

    def _uxhw_request(self, name: str) -> dict[str, Any]:
        """
        Find the UxHw load one variable made.

        Args:
            name: The variable's name.

        Returns:
            The keyword arguments of that load.
        """
        return next(
            dict(call.kwargs)
            for call in self.load_uxhw.call_args_list
            if call.kwargs["target_expr"] == name
        )

    def test_a_variable_whose_every_run_died_loads_nothing(self) -> None:
        """It has no rows in the tracing database, and loading them raises."""
        loaded = [
            call.kwargs["target_expression"]
            for call in self.load_ground_truth.call_args_list
        ] + [call.kwargs["target_expr"] for call in self.load_uxhw.call_args_list]

        self.assertNotIn("all_died", loaded)

    def test_an_analytic_variable_reads_weighted_samples(self) -> None:
        """Its reference is weighted samples, not Monte Carlo."""
        request = self._ground_truth_request("analytic")

        self.assertEqual(request["table"], RepresentationTypes.WEIGHTED_SAMPLES)
        self.assertFalse(request["monte_carlo"])

    def test_other_variables_read_the_run_wide_table(self) -> None:
        """A variable without its own reference is read as before."""
        request = self._ground_truth_request("monte_carlo")

        self.assertEqual(request["table"], RepresentationTypes.MONTE_CARLO)
        self.assertTrue(request["monte_carlo"])

    def test_each_variable_loads_its_own_matrix(self) -> None:
        """The UxHw load asks for each variable's types and sizes."""
        self.assertEqual(
            (
                self._uxhw_request("analytic")["ur_types"],
                self._uxhw_request("analytic")["ur_sizes"],
            ),
            (["Athens"], [8]),
        )
        self.assertEqual(
            (
                self._uxhw_request("monte_carlo")["ur_types"],
                self._uxhw_request("monte_carlo")["ur_sizes"],
            ),
            (["Jupiter"], [32, 64]),
        )


if __name__ == "__main__":
    unittest.main()
