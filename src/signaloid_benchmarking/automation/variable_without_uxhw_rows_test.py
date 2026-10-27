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
Unit tests for the analysis steps when one variable has no UxHw rows.

A variable whose every UxHw run died has no traced data. Equivalent Monte
Carlo leaves it out, since it has nothing to compare. Reloading the EMCC
data must not happen on its account, or every other variable's rows would
be added a second time.
"""

import contextlib
import io
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from signaloid_benchmarking.automation.analysis import (
    compute_equivalent_mc,
    load_emcc_data,
)
from signaloid_benchmarking.config import BenchmarkingVariables, EquivMC
from signaloid_benchmarking.types import BenchmarkingVariable


def _variables() -> tuple[BenchmarkingVariable, BenchmarkingVariable]:
    """
    Build a variable whose every run died and one that kept a row.

    Returns:
        The two variables, the empty one first.
    """
    all_died = BenchmarkingVariable(name="a", description="a")
    all_died.skipped_configurations = {
        "Athens-8": "its tracing run was killed by signal 9 (SIGKILL), or exited with code 137",
    }
    measured = BenchmarkingVariable(name="b", description="b")
    measured.emcc_results.emcc_data = [
        {
            BenchmarkingVariables.VARIABLE_DESCRIPTION: "b",
            BenchmarkingVariables.UXHW_CONF: "Athens-8",
            EquivMC.EMCC: 100,
        }
    ]

    return all_died, measured


class TestLoadEmccData(unittest.TestCase):
    """The CSV is read only when no variable holds data in memory."""

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.output_data_file = os.path.join(tmp_dir.name, "output_data.csv")
        all_died, measured = _variables()
        pd.DataFrame(measured.emcc_results.emcc_data).to_csv(
            self.output_data_file, index=False
        )
        self.all_died = all_died
        self.measured = measured

    def _load(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            load_emcc_data(
                benchmarking_variables=[self.all_died, self.measured],
                output_data_file=self.output_data_file,
            )

    def test_an_empty_first_variable_does_not_add_rows_twice(self) -> None:
        self._load()

        self.assertEqual(len(self.measured.emcc_results.emcc_data), 1)
        self.assertEqual(self.all_died.emcc_results.emcc_data, [])
        self.assertEqual(self.measured.emcc_results.equiv_mc_list, [100])
        self.assertEqual(self.all_died.emcc_results.equiv_mc_list, [])

    def test_the_csv_is_read_when_no_variable_holds_data(self) -> None:
        """A run resumed from its results files still loads them."""
        self.measured.emcc_results.emcc_data = []

        self._load()

        self.assertEqual(len(self.measured.emcc_results.emcc_data), 1)


class TestComputeEquivalentMc(unittest.TestCase):
    """Equivalent Monte Carlo runs only for variables with a configuration."""

    def test_a_variable_whose_every_run_died_is_left_out(self) -> None:
        all_died, measured = _variables()

        with (
            patch("signaloid_benchmarking.automation.analysis.load_asymptotic_dist"),
            patch(
                "signaloid_benchmarking.automation.analysis."
                "load_data_and_compute_equivalent_mc"
            ) as compute,
        ):
            compute_equivalent_mc(
                benchmarking_variables=[all_died, measured],
                output_data_file="unused.csv",
                asymptotic_dist_file="unused.csv",
                ground_truth_db_path="unused.db",
                ground_truth_type="MonteCarlo",
                adversary_db_path="unused.db",
                tracing_db_path="unused.db",
                representation_types=["Athens"],
                representation_sizes=[8],
                correlations=["Disabled"],
                num_parallel_workers=1,
                n_adversaries=1,
                distance_type="Wasserstein-1",
                use_binned_uxhw=False,
                use_clt=False,
                plots_dir="unused",
            )

        self.assertEqual(
            compute.call_args.args[0]["benchmarking_variables"], [measured]
        )


if __name__ == "__main__":
    unittest.main()
