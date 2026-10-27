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
Unit tests for the per-variable keys ``Benchmark.load_yaml`` reads.

The benchmarking template lets each ``BenchmarkingVariables`` entry declare
``RepresentationTypes``, ``RepresentationSizes``, ``CorrelationTracking`` and
``AnalyticGroundTruth``, each falling back to its run-wide flag when absent.
"""

import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml

from signaloid_benchmarking.automation.benchmark import Benchmark
from signaloid_benchmarking.config import RepresentationTypes

RUN_WIDE_TYPES = ["Jupiter"]
RUN_WIDE_SIZES = [32, 64]
RUN_WIDE_CORRELATIONS = ["Disabled"]

# Both outputs of the benchmarking template. The first declares every
# per-variable key, as the template does, and the second declares none.
DECLARING_ENTRY = {
    "VariableName": "outputVariables[0]",
    "VariableDescription": "First Variable",
    "OutputObject": "Distribution",
    "CommandLineArguments": "-S 0",
    "RepresentationTypes": ["Athens"],
    "RepresentationSizes": [8, 16, 32, 64, 128],
    "CorrelationTracking": ["Autocorrelation"],
    "AnalyticGroundTruth": "ground-truth.py",
}
SILENT_ENTRY = {
    "VariableName": "outputVariables[1]",
    "VariableDescription": "Second Variable (mean)",
    "OutputObject": "Scalar",
    "CommandLineArguments": "-S 1",
}


def _write_application(directory: Path, entries: list[dict[str, Any]]) -> None:
    """
    Write a minimal application with the given ``BenchmarkingVariables``.

    Args:
        directory: The application root.
        entries: The ``BenchmarkingVariables`` entries.
    """
    signaloid_yaml = {
        "TraceVariables": [
            {
                "File": "main.c",
                "LineNumber": 50,
                "Expression": "outputVariables[0:1]",
            }
        ],
        "BenchmarkingVariables": entries,
        "BenchmarkingAllOutputs": [{"CommandLineArguments": "-S 2"}],
    }
    (directory / "signaloid.yaml").write_text(
        yaml.safe_dump(signaloid_yaml, sort_keys=False),
        encoding="utf-8",
    )
    (directory / "ground-truth.py").write_text("", encoding="utf-8")


class TestLoadYamlPerVariableKeys(unittest.TestCase):
    """The four per-variable keys of a BenchmarkingVariables entry."""

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.application = Path(tmp_dir.name)

    def _load(self, entries: list[dict[str, Any]]) -> Benchmark:
        """
        Load ``entries`` through a Benchmark with known run-wide flags.

        Args:
            entries: The ``BenchmarkingVariables`` entries.

        Returns:
            The Benchmark after ``load_yaml``.
        """
        _write_application(self.application, entries)
        benchmark = Benchmark(
            path_to_application=str(self.application),
            representation_types=list(RUN_WIDE_TYPES),
            representation_sizes=list(RUN_WIDE_SIZES),
            correlations=list(RUN_WIDE_CORRELATIONS),
        )
        benchmark.load_yaml()

        return benchmark

    def _matrix(self, benchmark: Benchmark, index: int) -> tuple:
        """
        Resolve one variable's matrix against the run-wide flags.

        Args:
            benchmark: A Benchmark after ``load_yaml``.
            index: The variable's position.

        Returns:
            The variable's types, sizes and correlation modes.
        """
        return benchmark.benchmarking_variables[index].representation_matrix(
            representation_types=benchmark.representation_types,
            representation_sizes=benchmark.representation_sizes,
            correlations=benchmark.correlations,
        )

    def test_declared_keys_replace_the_run_wide_flags(self) -> None:
        """The template's first output is measured on what it declares."""
        benchmark = self._load([DECLARING_ENTRY, SILENT_ENTRY])

        self.assertEqual(
            self._matrix(benchmark, 0),
            (["Athens"], [8, 16, 32, 64, 128], ["Autocorrelation"]),
        )

    def test_absent_keys_leave_the_run_wide_flags_in_force(self) -> None:
        """An entry that declares nothing behaves as before this change."""
        benchmark = self._load([DECLARING_ENTRY, SILENT_ENTRY])

        silent = benchmark.benchmarking_variables[1]
        self.assertIsNone(silent.representation_types)
        self.assertIsNone(silent.representation_sizes)
        self.assertIsNone(silent.correlations)
        self.assertIsNone(silent.analytic_ground_truth)
        self.assertEqual(
            self._matrix(benchmark, 1),
            (RUN_WIDE_TYPES, RUN_WIDE_SIZES, RUN_WIDE_CORRELATIONS),
        )

    def test_each_axis_falls_back_on_its_own(self) -> None:
        """Declaring one axis leaves the other two run-wide."""
        entry = dict(SILENT_ENTRY, RepresentationSizes=[8])
        benchmark = self._load([DECLARING_ENTRY, entry])

        self.assertEqual(
            self._matrix(benchmark, 1),
            (RUN_WIDE_TYPES, [8], RUN_WIDE_CORRELATIONS),
        )

    def test_analytic_reference_resolves_against_the_application(self) -> None:
        """The script is named relative to the application root."""
        benchmark = self._load([DECLARING_ENTRY, SILENT_ENTRY])

        self.assertEqual(
            benchmark.benchmarking_variables[0].analytic_ground_truth,
            os.path.abspath(self.application / "ground-truth.py"),
        )

    def test_analytic_reference_implies_weighted_samples(self) -> None:
        """Only the variable with the reference reads weighted samples."""
        benchmark = self._load([DECLARING_ENTRY, SILENT_ENTRY])

        declaring, silent = benchmark.benchmarking_variables
        self.assertEqual(
            declaring.ground_truth_table(RepresentationTypes.MONTE_CARLO),
            RepresentationTypes.WEIGHTED_SAMPLES,
        )
        self.assertEqual(
            silent.ground_truth_table(RepresentationTypes.MONTE_CARLO),
            RepresentationTypes.MONTE_CARLO,
        )

    def test_run_wide_flags_are_left_untouched(self) -> None:
        """
        Reading the keys changes no run-wide flag, so a second read sees
        the same defaults and gives the same result.
        """
        benchmark = self._load([DECLARING_ENTRY, SILENT_ENTRY])
        first = [self._matrix(benchmark, index) for index in (0, 1)]

        benchmark.load_yaml()

        self.assertEqual(benchmark.representation_types, RUN_WIDE_TYPES)
        self.assertEqual(benchmark.representation_sizes, RUN_WIDE_SIZES)
        self.assertEqual(benchmark.correlations, RUN_WIDE_CORRELATIONS)
        self.assertFalse(benchmark.has_analytic_ground_truth)
        self.assertEqual([self._matrix(benchmark, index) for index in (0, 1)], first)

    def test_invalid_values_are_refused_by_name(self) -> None:
        """A bad value stops the load, naming the variable and the key."""
        cases = {
            "unknown type": ("RepresentationTypes", ["Saturn"]),
            "empty types": ("RepresentationTypes", []),
            "types not a list": ("RepresentationTypes", "Athens"),
            "types given no value": ("RepresentationTypes", None),
            "non-integer size": ("RepresentationSizes", [8, "16"]),
            "boolean size": ("RepresentationSizes", [8, True]),
            "unknown correlation": ("CorrelationTracking", ["Partial"]),
        }
        for label, (key, value) in cases.items():
            with self.subTest(label):
                entry = dict(SILENT_ENTRY, **{key: value})
                with self.assertRaises(ValueError) as caught:
                    self._load([DECLARING_ENTRY, entry])
                message = str(caught.exception)
                self.assertIn(key, message)
                self.assertIn("outputVariables[1]", message)

    def test_analytic_reference_must_name_an_existing_script(self) -> None:
        """A typo'd script stops the load rather than the ground truth."""
        entry = dict(SILENT_ENTRY, AnalyticGroundTruth="ground-trut.py")

        with self.assertRaises(ValueError) as caught:
            self._load([DECLARING_ENTRY, entry])

        message = str(caught.exception)
        self.assertIn("AnalyticGroundTruth", message)
        self.assertIn("outputVariables[1]", message)
        self.assertIn("ground-trut.py", message)

    def test_analytic_reference_must_be_a_name(self) -> None:
        """An empty or non-string value is refused."""
        for value in ("", "  ", ["ground-truth.py"], None):
            with self.subTest(value=value):
                entry = dict(SILENT_ENTRY, AnalyticGroundTruth=value)
                with self.assertRaises(ValueError):
                    self._load([DECLARING_ENTRY, entry])


if __name__ == "__main__":
    unittest.main()
