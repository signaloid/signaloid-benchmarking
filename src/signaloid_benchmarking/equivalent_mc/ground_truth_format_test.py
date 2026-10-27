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
Unit tests for the ground truth ``load_data_and_compute_equivalent_mc``
asks for.

The function is public, so a caller may name its ground-truth table
anything. The table name picks the table and the ground-truth type picks
the sample format, except that a variable with its own analytic reference
reads weighted samples. The ground-truth load is replaced with a mock
that records its arguments and then stops the run.
"""

import unittest
from typing import Any
from unittest.mock import patch

from signaloid_benchmarking.config import RepresentationTypes
from signaloid_benchmarking.equivalent_mc.equivalent_mc_main import (
    load_data_and_compute_equivalent_mc,
)
from signaloid_benchmarking.types import BenchmarkingVariable


class _Stop(Exception):
    """Ends the run once the ground truth has been asked for."""


class TestGroundTruthFormat(unittest.TestCase):
    """The table and sample format of each variable's ground truth."""

    def _requested(self, variable: BenchmarkingVariable) -> dict[str, Any]:
        """
        Ask for one variable's ground truth from a table named ReferenceMC.

        Args:
            variable: The benchmarking variable.

        Returns:
            The arguments the ground truth was loaded with.
        """
        requested: dict[str, Any] = {}

        def record(**kwargs: Any) -> None:
            requested.update(kwargs)
            raise _Stop

        with patch(
            "signaloid_benchmarking.equivalent_mc.equivalent_mc_main._load_ground_truth",
            side_effect=record,
        ):
            with self.assertRaises(_Stop):
                load_data_and_compute_equivalent_mc(
                    {  # type: ignore[typeddict-item]
                        "n_processes": 1,
                        "n_adversaries": 1,
                        "reporting_methods": [],
                        "plots_dir": "plots",
                        "distance_type": "Wasserstein-1",
                        "auto_prefix": False,
                        "benchmarking_variables": [variable],
                        "ground_truth_database_path": "ground-truth.db",
                        "ground_truth_table_name": "ReferenceMC",
                        "ground_truth_type": RepresentationTypes.MONTE_CARLO,
                    }
                )

        return requested

    def test_a_custom_table_keeps_the_types_sample_format(self) -> None:
        """A Monte Carlo reference in a table named ReferenceMC is read as one."""
        requested = self._requested(BenchmarkingVariable(name="a", description="a"))

        self.assertEqual(requested["table"], "ReferenceMC")
        self.assertTrue(requested["monte_carlo"])

    def test_an_analytic_variable_reads_weighted_samples(self) -> None:
        """Its own reference overrides both the table and the format."""
        variable = BenchmarkingVariable(
            name="a", description="a", analytic_ground_truth="/app/ground-truth.py"
        )

        requested = self._requested(variable)

        self.assertEqual(requested["table"], RepresentationTypes.WEIGHTED_SAMPLES)
        self.assertFalse(requested["monte_carlo"])


if __name__ == "__main__":
    unittest.main()
