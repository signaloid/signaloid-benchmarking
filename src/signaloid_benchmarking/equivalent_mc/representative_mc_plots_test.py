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
The representative Monte Carlo plots are saved under the name the report
looks them up by.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from signaloid_benchmarking.automation.report_writer import _get_triptych_image_names
from signaloid_benchmarking.config import BenchmarkingVariables, EquivMC
from signaloid_benchmarking.equivalent_mc.equivalent_mc_utils import (
    _generate_representative_mc_plots,
)
from signaloid_benchmarking.types import BenchmarkingVariable


class TestRepresentativeMcPlotNames(unittest.TestCase):
    """The writer and the report agree on the adversary plot's path."""

    def test_punctuated_description_matches_the_report(self) -> None:
        """A description with a slash and brackets still gives one file name."""
        variable = BenchmarkingVariable(
            name="margin", description="Margin (qcrit/qlocal)"
        )
        variable.emcc_results.emcc_data = [
            {
                EquivMC.EMCC: 10,
                BenchmarkingVariables.UXHW_DISTANCE: 0.1,
                BenchmarkingVariables.UXHW_BINNED_DISTANCE: 0.1,
            }
        ]
        emcc = SimpleNamespace(variable=variable, use_binned_uxhw=False)
        module = "signaloid_benchmarking.equivalent_mc.equivalent_mc_utils"

        with (
            patch(
                f"{module}._generate_progressive_samples",
                return_value=np.linspace(0.0, 1.0, 10),
            ),
            patch(f"{module}.plot_wrapper.plot") as plot,
        ):
            _generate_representative_mc_plots(
                emcc, "Wasserstein-1", plots_dir="plots"  # type: ignore[arg-type]
            )

        expected = _get_triptych_image_names(
            variable=variable, best_config="Athens-8", best_emcc=10, plots_dir="plots"
        )[0]
        self.assertEqual(plot.call_args.kwargs["path"], expected)
        self.assertNotIn("/", expected.removeprefix("plots/"))


if __name__ == "__main__":
    unittest.main()
