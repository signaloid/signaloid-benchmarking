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
Unit tests for ``distance_uxhw_vs_mc``.

The end-to-end tests run ``main`` on small synthetic databases. The ground
truth is a Gaussian Monte Carlo sample pool, and each UxHw database holds a
quantised Gaussian per configuration.
"""

import math
import os
import tempfile
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

import numpy as np

from signaloid_benchmarking.automation.database_generator import (
    generate_database_from_mc_samples,
)
from signaloid_benchmarking.differential_accuracy.distance_uxhw_vs_mc import (
    benchmarking_variables,
    compute_distances,
    main,
    relative_difference_percentage,
    scoring,
)
from signaloid_benchmarking.equivalent_mc.conftest import (
    SyntheticUxhwConfig,
    SyntheticUxhwExpression,
    build_uxhw_database,
)
from signaloid_benchmarking.config import VariableTypes
from signaloid_benchmarking.types import BenchmarkingVariable
from signaloid.distributional.dirac_delta import DiracDelta
from signaloid.distributional.distributional import DistributionalValue

_TABLE = "TracingTable"
_CENTRE = 100.0
_SCALE = 15.0

_BENCHMARKING_VARIABLES_YAML = """
BenchmarkingVariables:
    - VariableName: output
      VariableDescription: Output
      OutputObject: Distribution
"""


def _configs(*sizes: int) -> list[SyntheticUxhwConfig]:
    """
    Athens configurations of the given sizes, with correlation tracking off.

    Args:
        sizes: Representation sizes.

    Returns:
        One configuration per size.
    """
    return [SyntheticUxhwConfig("Athens", size, "Disabled") for size in sizes]


def _read_report(path: str) -> list[dict[str, str]]:
    """
    Parse the GitHub-flavoured markdown table the tool writes.

    Args:
        path: Path of the report.

    Returns:
        One dictionary per table row, keyed by column header.
    """
    with open(path) as file:
        lines = file.read().splitlines()
    headers = [cell.strip() for cell in lines[0].strip("|").split("|")]
    return [
        dict(zip(headers, [cell.strip() for cell in line.strip("|").split("|")]))
        for line in lines[2:]
    ]


class TestDistanceUxhwVsMcEndToEnd(unittest.TestCase):
    """Runs of ``main`` on synthetic databases."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.directory = self._directory.name
        self.report = os.path.join(self.directory, "report.md")

    def tearDown(self) -> None:
        self._directory.cleanup()

    def _ground_truth(self, names: list[str]) -> str:
        """
        Write a Monte Carlo ground truth database.

        Args:
            names: Expression names to write samples for.

        Returns:
            Path of the database.
        """
        rng = np.random.default_rng(20261026)
        variables = []
        for index, name in enumerate(names):
            variable = BenchmarkingVariable(
                name=name, description=name, value_id=f"value-{index}"
            )
            variable.distribution_samples.set_values(
                [float(sample) for sample in rng.normal(_CENTRE, _SCALE, 2_000)]
            )
            variables.append(variable)
        path = os.path.join(self.directory, "ground-truth.db")
        generate_database_from_mc_samples(path, variables)
        return path

    def _uxhw(
        self,
        file_name: str,
        configs: list[SyntheticUxhwConfig],
        names: list[str],
        centre: float = _CENTRE,
    ) -> str:
        """
        Write a UxHw database.

        Args:
            file_name: Name of the database file.
            configs: Configurations to write.
            names: Expression names to write.
            centre: Centre of the quantised Gaussian.

        Returns:
            Path of the database.
        """
        expressions = [
            SyntheticUxhwExpression(name, f"value-{index}", centre, _SCALE)
            for index, name in enumerate(names)
        ]
        path = os.path.join(self.directory, file_name)
        build_uxhw_database(path, _TABLE, expressions, configs)
        return path

    def _yaml(self, contents: str) -> str:
        """
        Write a signaloid.yaml.

        Args:
            contents: The yaml text.

        Returns:
            Path of the file.
        """
        path = os.path.join(self.directory, "signaloid.yaml")
        with open(path, "w") as file:
            file.write(contents)
        return path

    def _main(
        self,
        ground_truth: str,
        baseline: str,
        undertest: str,
        sizes: list[int],
        yaml_path: str | None,
        baseline_types: tuple[str, ...] = ("Athens",),
        undertest_types: tuple[str, ...] = ("Athens",),
    ) -> None:
        """
        Run ``main``, silencing stdout.

        Args:
            ground_truth: Ground truth database path.
            baseline: Baseline UxHw database path.
            undertest: Under-test UxHw database path.
            sizes: Representation sizes to filter on, or empty to omit -s.
            yaml_path: signaloid.yaml path, or None to omit the option.
            baseline_types: Representation types to filter the baseline on.
            undertest_types: Representation types to filter the under-test on.
        """
        argv = [
            "--ground-truth-database",
            ground_truth,
            "--uxhw-baseline-database",
            baseline,
            "--uxhw-under-test-database",
            undertest,
            "--uxhw-tables",
            _TABLE,
            "--baseline-uncertainty-representation-type",
            *baseline_types,
            "--under-test-uncertainty-representation-type",
            *undertest_types,
            "--benchmark",
            "synthetic",
            "-o",
            self.report,
        ]
        if sizes:
            argv += ["-s", *[str(size) for size in sizes]]
        if yaml_path is not None:
            argv += ["-y", yaml_path]
        with redirect_stdout(StringIO()):
            main(argv)

    def test_self_comparison_reports_no_change(self) -> None:
        """Comparing a database with itself keeps every row and scores it ⚪."""
        ground_truth = self._ground_truth(["output"])
        uxhw = self._uxhw("uxhw.db", _configs(8, 16), ["output"])

        self._main(
            ground_truth, uxhw, uxhw, [8, 16], self._yaml(_BENCHMARKING_VARIABLES_YAML)
        )

        rows = _read_report(self.report)
        self.assertEqual(
            [row["Baseline Conf"] for row in rows], ["Athens-8", "Athens-16"]
        )
        for row in rows:
            self.assertEqual(row["Baseline Conf"], row["Under-test Conf"])
            self.assertEqual(float(row["WD Diff"]), 0.0)
            self.assertEqual(row["Scoring"], "⚪")
            self.assertEqual(row["Binned Scoring"], "⚪")

    def test_closer_under_test_scores_improvement(self) -> None:
        """An under-test run centred on the ground truth beats an offset one."""
        ground_truth = self._ground_truth(["output"])
        baseline = self._uxhw(
            "baseline.db", _configs(16), ["output"], centre=_CENTRE + 10.0
        )
        undertest = self._uxhw("undertest.db", _configs(16), ["output"])

        self._main(
            ground_truth,
            baseline,
            undertest,
            [16],
            self._yaml(_BENCHMARKING_VARIABLES_YAML),
        )

        (row,) = _read_report(self.report)
        self.assertLess(float(row["WD Under-test"]), float(row["WD Baseline"]))
        self.assertLess(float(row["WD Diff"]), 0.0)
        self.assertEqual(row["Scoring"], "🟢")

    def test_configurations_of_different_sizes_are_not_paired(self) -> None:
        """
        Equal counts of configurations do not mean they match. Athens-16 must
        never be compared against Athens-32.
        """
        ground_truth = self._ground_truth(["output"])
        baseline = self._uxhw("baseline.db", _configs(8, 16), ["output"])
        undertest = self._uxhw("undertest.db", _configs(8, 32), ["output"])

        with self.assertRaises(SystemExit) as raised:
            self._main(
                ground_truth,
                baseline,
                undertest,
                [8, 16, 32],
                self._yaml(_BENCHMARKING_VARIABLES_YAML),
            )

        self.assertNotEqual(raised.exception.code, 0)
        self.assertFalse(os.path.exists(self.report))

    def test_different_types_are_paired_by_size(self) -> None:
        """
        Representation types may differ between the two sides. Pairing goes by
        size, so Athens-16 meets Jupiter-16 and Atlas-32 meets Atlas-32.
        """
        ground_truth = self._ground_truth(["output"])
        baseline = self._uxhw(
            "baseline.db",
            [
                SyntheticUxhwConfig("Athens", 16, "Disabled"),
                # Non-Athens sizes come from UR_Order_CoreLibrary, which the
                # builder sets to N + 3.
                SyntheticUxhwConfig("Atlas", 32 - 3, "Disabled"),
            ],
            ["output"],
        )
        undertest = self._uxhw(
            "undertest.db",
            [
                SyntheticUxhwConfig("Atlas", 32 - 3, "Disabled"),
                SyntheticUxhwConfig("Jupiter", 16 - 3, "Disabled"),
            ],
            ["output"],
        )

        self._main(
            ground_truth,
            baseline,
            undertest,
            [16, 32],
            self._yaml(_BENCHMARKING_VARIABLES_YAML),
            baseline_types=("Athens", "Atlas"),
            undertest_types=("Atlas", "Jupiter"),
        )

        rows = _read_report(self.report)
        self.assertEqual(
            [(row["Baseline Conf"], row["Under-test Conf"]) for row in rows],
            [("Athens-16", "Jupiter-16"), ("Atlas-32", "Atlas-32")],
        )

    def test_trace_variables_fallback_reads_every_entry(self) -> None:
        """Every TraceVariables entry is measured, not only the last one."""
        names = ["first", "second[0]", "second[1]"]
        ground_truth = self._ground_truth(names)
        uxhw = self._uxhw("uxhw.db", _configs(8), names)
        yaml_path = self._yaml("""
TraceVariables:
    - File: main.c
      LineNumber: 1
      Expression: first
    - File: main.c
      LineNumber: 2
      Expression: second[0:1]
""")

        self._main(ground_truth, uxhw, uxhw, [8], yaml_path)

        rows = _read_report(self.report)
        self.assertEqual([row["Expression"] for row in rows], names)

    def test_missing_yaml_is_a_usage_error(self) -> None:
        """Omitting --yaml-signaloid fails in argument parsing."""
        ground_truth = self._ground_truth(["output"])
        uxhw = self._uxhw("uxhw.db", _configs(8), ["output"])

        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
            self._main(ground_truth, uxhw, uxhw, [8], None)

        self.assertEqual(raised.exception.code, 2)

    def test_missing_size_filter_is_a_usage_error(self) -> None:
        """
        The loaders need at least one size, so omitting -s fails in argument
        parsing.
        """
        ground_truth = self._ground_truth(["output"])
        uxhw = self._uxhw("uxhw.db", _configs(8), ["output"])
        yaml_path = self._yaml(_BENCHMARKING_VARIABLES_YAML)

        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
            self._main(ground_truth, uxhw, uxhw, [], yaml_path)

        self.assertEqual(raised.exception.code, 2)


class TestBenchmarkingVariables(unittest.TestCase):
    """The variables both tools read from a signaloid.yaml."""

    def test_legacy_trace_variables_are_distributions(self) -> None:
        """Older TraceVariables files expand ranges and default to Distribution."""
        yaml_dict = {"TraceVariables": [{"Expression": "out[0:1]"}]}
        self.assertEqual(
            benchmarking_variables(yaml_dict),
            [("out[0]", "Distribution"), ("out[1]", "Distribution")],
        )

    def test_benchmarking_variables_keep_output_type(self) -> None:
        """BenchmarkingVariables entries keep their declared OutputObject."""
        yaml_dict = {
            "BenchmarkingVariables": [
                {"VariableName": "a", "OutputObject": "Scalar"},
                {"VariableName": "b", "OutputObject": "Distribution"},
            ]
        }
        self.assertEqual(
            benchmarking_variables(yaml_dict),
            [("a", "Scalar"), ("b", "Distribution")],
        )


class TestScoring(unittest.TestCase):
    """The relative difference and the traffic light it maps to."""

    def test_scores(self) -> None:
        """Each (baseline, under-test) pair gets the expected score."""
        cases = [
            (1.0, 0.5, "🟢"),
            (1.0, 1.0, "⚪"),
            (1.0, 1.0 + 1e-9, "🟡"),
            (1.0, 1.0 + 1e-4, "🟠"),
            (1.0, 2.0, "🔴"),
            (0.0, 0.0, "⚪"),
            (0.0, 1.0, "🔴"),
            (math.nan, 1.0, "🟢"),
            (math.inf, 1.0, "🟢"),
            (1.0, math.nan, "🔴"),
            # An exact baseline that turns into NaN is a regression.
            (0.0, math.nan, "🔴"),
        ]
        for baseline, undertest, expected in cases:
            with self.subTest(baseline=baseline, undertest=undertest):
                percentage = relative_difference_percentage(baseline, undertest)
                self.assertEqual(scoring(percentage, baseline, undertest), expected)

    def test_zero_baseline_does_not_divide_by_zero(self) -> None:
        """A zero baseline gives 0 or a signed infinity."""
        self.assertEqual(relative_difference_percentage(0.0, 0.0), 0.0)
        self.assertEqual(relative_difference_percentage(0.0, 1.0), math.inf)
        self.assertEqual(relative_difference_percentage(0.0, -1.0), -math.inf)


def _distribution(*deltas: tuple[float, float]) -> DistributionalValue:
    """
    Build a distribution from (position, mass) pairs.

    Args:
        deltas: The (position, mass) pairs.

    Returns:
        The distribution.
    """
    return DistributionalValue(
        dirac_deltas=[
            DiracDelta(position=position, mass=mass) for position, mass in deltas
        ]
    )


class TestComputeDistancesBlowUp(unittest.TestCase):
    """
    A degraded side gets infinite distances instead of aborting the command.
    """

    def _compute(
        self,
        expression_type: str,
        baseline: DistributionalValue,
        undertest: DistributionalValue,
        ground_truth: DistributionalValue,
    ) -> tuple[float, float, float, float]:
        """
        Run ``compute_distances`` and expect a blow-up warning.

        Args:
            expression_type: Type of the benchmarking variable.
            baseline: Baseline UxHw distribution.
            undertest: Under-test UxHw distribution.
            ground_truth: Ground truth distribution.

        Returns:
            The distances ``compute_distances`` returns.
        """
        with self.assertWarns(UserWarning):
            return compute_distances(
                expression_type=expression_type,
                uxhw_baseline_dist=baseline,
                uxhw_undertest_dist=undertest,
                ground_truth=ground_truth,
            )

    def test_degraded_under_test_scores_regression(self) -> None:
        """A NaN carrying mass on the under-test side is a regression."""
        ground_truth = _distribution((1.0, 0.5), (2.0, 0.5))
        baseline = _distribution((1.0, 0.5), (2.5, 0.5))
        undertest = _distribution((1.0, 0.5), (math.nan, 0.5))

        baseline_distance, undertest_distance, baseline_binned, undertest_binned = (
            self._compute(VariableTypes.DISTRIBUTION, baseline, undertest, ground_truth)
        )

        self.assertTrue(math.isfinite(baseline_distance))
        self.assertTrue(math.isfinite(baseline_binned))
        self.assertEqual(undertest_distance, math.inf)
        self.assertEqual(undertest_binned, math.inf)
        percentage = relative_difference_percentage(
            baseline_distance, undertest_distance
        )
        self.assertEqual(
            scoring(percentage, baseline_distance, undertest_distance), "🔴"
        )

    def test_degraded_baseline_scores_recovery(self) -> None:
        """An infinity carrying mass on the baseline side makes a recovery."""
        ground_truth = _distribution((1.0, 0.5), (2.0, 0.5))
        baseline = _distribution((1.0, 0.5), (math.inf, 0.5))
        undertest = _distribution((1.0, 0.5), (2.5, 0.5))

        baseline_distance, undertest_distance, baseline_binned, undertest_binned = (
            self._compute(VariableTypes.DISTRIBUTION, baseline, undertest, ground_truth)
        )

        self.assertEqual(baseline_distance, math.inf)
        self.assertEqual(baseline_binned, math.inf)
        self.assertTrue(math.isfinite(undertest_distance))
        self.assertTrue(math.isfinite(undertest_binned))
        percentage = relative_difference_percentage(
            baseline_distance, undertest_distance
        )
        self.assertEqual(
            scoring(percentage, baseline_distance, undertest_distance), "🟢"
        )

    def test_non_finite_scalar_is_degraded(self) -> None:
        """A NaN scalar gets infinite distances."""
        ground_truth = _distribution((2.0, 1.0))
        baseline = _distribution((2.5, 1.0))
        undertest = _distribution((math.nan, 1.0))

        baseline_distance, undertest_distance, _, undertest_binned = self._compute(
            VariableTypes.SCALAR, baseline, undertest, ground_truth
        )

        self.assertTrue(math.isfinite(baseline_distance))
        self.assertEqual(undertest_distance, math.inf)
        self.assertEqual(undertest_binned, math.inf)

    def test_large_finite_scalar_is_not_degraded(self) -> None:
        """The magnitude check is off for scalars, so 1e200 is scored."""
        ground_truth = _distribution((2.0, 1.0))
        large = _distribution((1e200, 1.0))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            distances = compute_distances(
                expression_type=VariableTypes.SCALAR,
                uxhw_baseline_dist=large,
                uxhw_undertest_dist=large,
                ground_truth=ground_truth,
            )

        for distance in distances:
            self.assertTrue(math.isfinite(distance))


if __name__ == "__main__":
    unittest.main()
