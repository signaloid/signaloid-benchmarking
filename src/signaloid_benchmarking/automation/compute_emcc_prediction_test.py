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
import unittest

from signaloid_benchmarking.automation.analysis import _get_equiv_mc_list
from signaloid_benchmarking.automation.benchmarking_utils import (
    compute_emcc_prediction,
)
from signaloid_benchmarking.config import EquivMC
from signaloid_benchmarking.types import BenchmarkingVariable


class TestComputeEmccPrediction(unittest.TestCase):
    """The predicted count is the smallest N whose scaled statistic beats the
    distance, so the square ratio is rounded up."""

    def _predict(self, quantity: float | None, distance: float) -> int | None:
        with contextlib.redirect_stdout(io.StringIO()):
            return compute_emcc_prediction(quantity, distance)

    def test_exact_square_is_kept(self) -> None:
        self.assertEqual(self._predict(10.0, 1.0), 100)

    def test_fractional_square_is_rounded_up(self) -> None:
        self.assertEqual(self._predict(10.5, 1.0), 111)
        self.assertEqual(self._predict(10.0, 1.0000001), 100)

    def test_missing_statistic_gives_no_prediction(self) -> None:
        """No asymptotic statistic means no prediction, not the search floor."""
        self.assertIsNone(self._predict(None, 1.0))

    def test_invalid_inputs_fall_back_to_one(self) -> None:
        self.assertEqual(self._predict(10.0, 0.0), 1)
        self.assertEqual(self._predict(10.0, float("inf")), 1)
        self.assertEqual(self._predict(1e200, 1e-200), 1)


class TestEquivMcListFallback(unittest.TestCase):
    """The size list built from predictions skips rows without one."""

    def test_absent_predictions_contribute_nothing(self) -> None:
        variable = BenchmarkingVariable(name="x", description="x variable")
        variable.emcc_results.emcc_data = [
            {EquivMC.EMCC_PREDICTED: 100},
            {EquivMC.EMCC_PREDICTED: None},
            {EquivMC.EMCC_PREDICTED: float("nan")},
            {EquivMC.EMCC_PREDICTED: 50},
        ]

        _get_equiv_mc_list(benchmarking_variables=[variable])

        self.assertEqual(variable.emcc_results.equiv_mc_list, [50, 100])

    def test_no_prediction_at_all_gives_an_empty_list(self) -> None:
        variable = BenchmarkingVariable(name="x", description="x variable")
        variable.emcc_results.emcc_data = [{EquivMC.EMCC_PREDICTED: None}]

        _get_equiv_mc_list(benchmarking_variables=[variable])

        self.assertEqual(variable.emcc_results.equiv_mc_list, [])


if __name__ == "__main__":
    unittest.main()
