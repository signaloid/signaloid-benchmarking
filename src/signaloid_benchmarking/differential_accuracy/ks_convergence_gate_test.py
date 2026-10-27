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
Unit tests for ``ks_convergence_gate``.

These tests cover the either-side decision, the choice of the largest size,
the grouping by correlation mode and occurrence, the precomputed CDF, the
tie-aware statistic and bootstrap, the handling of non-finite mass and of
scalar variables, the yaml handling and the report filtering. The end-to-end
tests run ``main`` on a report that ``distance_uxhw_vs_mc`` writes from small
synthetic databases.
"""

import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

import numpy as np
from scipy.stats import ks_1samp, norm  # type: ignore

from signaloid_benchmarking.automation.database_generator import (
    generate_database_from_mc_samples,
)
from signaloid_benchmarking.config import Correlations
from signaloid_benchmarking.differential_accuracy import distance_uxhw_vs_mc
from signaloid_benchmarking.differential_accuracy import (
    ks_convergence_gate as gate,
)
from signaloid_benchmarking.equivalent_mc.conftest import (
    SyntheticUxhwConfig,
    SyntheticUxhwExpression,
    build_uxhw_database,
)
from signaloid_benchmarking.types import (
    BenchmarkingVariable,
    TaggedDistributionalValue,
)
from signaloid.distributional.dirac_delta import DiracDelta
from signaloid.distributional.distributional import DistributionalValue

ALL_SIZES = [2, 4, 8, 16, 32, 64, 128, 256, 512]


def normal_ground_truth(count: int = 50_000) -> DistributionalValue:
    samples = np.random.default_rng(7).standard_normal(count)
    return DistributionalValue.from_samples(samples)


def normal_comb(size: int, shift: float = 0.0) -> DistributionalValue:
    """Equal-mass quantisation of a normal, optionally shifted."""
    quantiles = (np.arange(size) + 0.5) / size
    return DistributionalValue(
        dirac_deltas=[
            DiracDelta(position=float(norm.ppf(q) + shift), mass=1 / size)
            for q in quantiles
        ]
    )


def tagged(dv: DistributionalValue, size: int, mode: str) -> TaggedDistributionalValue:
    return TaggedDistributionalValue(
        dv=dv,
        representation_type="Athens",
        representation_size=size,
        correlation_tracking=mode,
    )


def present(result: gate.SideResult | None) -> gate.SideResult:
    """The side result, which the test expects to exist."""
    assert result is not None
    return result


def side(passed: bool, size: int = 512) -> gate.SideResult:
    return gate.SideResult(size=size, passed=passed, p_value=0.5)


class DecideTest(unittest.TestCase):
    def test_both_converge(self) -> None:
        decision, note = gate.decide(side(True), side(True))
        self.assertEqual(decision, gate.MEASURED)
        self.assertEqual(note, "")

    def test_only_baseline_converges_is_measured_and_noted(self) -> None:
        decision, note = gate.decide(side(True), side(False))
        self.assertEqual(decision, gate.MEASURED_ONE_SIDE)
        self.assertIn("baseline converges", note)

    def test_only_under_test_converges_is_measured_and_noted(self) -> None:
        decision, note = gate.decide(side(False), side(True))
        self.assertEqual(decision, gate.MEASURED_ONE_SIDE)
        self.assertIn("under-test converges", note)

    def test_neither_converges(self) -> None:
        decision, note = gate.decide(side(False), side(False))
        self.assertEqual(decision, gate.NOT_MEASURED)
        self.assertEqual(note, "neither SDK converges")

    def test_missing_side_counts_as_not_converging(self) -> None:
        self.assertEqual(gate.decide(None, side(True))[0], gate.MEASURED_ONE_SIDE)
        self.assertEqual(gate.decide(None, None)[0], gate.NOT_MEASURED)


class StepCdfTest(unittest.TestCase):
    def test_matches_distributional_value_cdf(self) -> None:
        rng = np.random.default_rng(3)
        positions = rng.normal(size=200)
        masses = rng.uniform(0.1, 1.0, size=200)
        dv = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=float(p), mass=float(m))
                for p, m in zip(positions, masses)
            ]
        )
        cdf = gate.step_cdf(dv)
        points = rng.normal(size=100)
        np.testing.assert_allclose(cdf(points), dv.cdf(points))

    def test_rejects_zero_mass(self) -> None:
        dv = DistributionalValue(dirac_deltas=[DiracDelta(position=1.0, mass=0.0)])
        with self.assertRaises(ValueError):
            gate.step_cdf(dv)

    def test_rejects_mass_at_infinity_rather_than_renormalising(self) -> None:
        dv = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.9),
                DiracDelta(position=float("inf"), mass=0.1),
            ]
        )
        with self.assertRaises(ValueError):
            gate.step_cdf(dv)


class KolmogorovSmirnovStatisticTest(unittest.TestCase):
    def test_matches_scipy_on_continuous_ground_truth(self) -> None:
        cdf = gate.step_cdf(normal_ground_truth())
        sample = np.random.default_rng(11).standard_normal(100)
        self.assertAlmostEqual(
            gate.kolmogorov_smirnov_statistic(sample, cdf),
            ks_1samp(sample, cdf).statistic,
            places=12,
        )

    def test_atom_the_sample_lands_on_is_not_a_gap(self) -> None:
        """
        Six tenths of the mass at zero, the rest spread out. A sample drawn
        from that same law must score small, where scipy scores the atom.
        """
        rng = np.random.default_rng(5)
        tail = rng.uniform(1.0, 2.0, 4000)
        ground_truth = DistributionalValue.from_samples(
            np.concatenate([np.zeros(6000), tail])
        )
        cdf = gate.step_cdf(ground_truth)
        sample = np.where(rng.uniform(size=200) < 0.6, 0.0, rng.uniform(1.0, 2.0, 200))
        self.assertLess(gate.kolmogorov_smirnov_statistic(sample, cdf), 0.15)
        self.assertGreater(ks_1samp(sample, cdf).statistic, 0.5)


class BootstrappedKsTest(unittest.TestCase):
    def test_exact_discrete_representation_passes(self) -> None:
        atoms = [(0.0, 0.6), (1.0, 0.3), (2.0, 0.1)]
        dv = DistributionalValue(
            dirac_deltas=[DiracDelta(position=p, mass=m) for p, m in atoms]
        )
        passed, p_value = gate.bootstrapped_ks(
            dv,
            gate.step_cdf(dv),
            sample_size=128,
            number_of_bootstraps=10,
            significance_level=0.05,
            rng=np.random.default_rng(0),
        )
        self.assertTrue(passed, p_value)

    def test_rejects_non_finite_positive_mass(self) -> None:
        dv = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=0.5, mass=0.5),
                DiracDelta(position=float("inf"), mass=0.5),
            ]
        )
        with self.assertRaises(ValueError):
            gate.bootstrapped_ks(
                dv,
                gate.step_cdf(normal_ground_truth()),
                sample_size=8,
                number_of_bootstraps=3,
                significance_level=0.05,
                rng=np.random.default_rng(0),
            )

    def test_ignores_zero_mass_placeholder(self) -> None:
        dv = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=-0.5, mass=0.5),
                DiracDelta(position=0.5, mass=0.5),
                DiracDelta(position=float("nan"), mass=0.0),
            ]
        )
        passed, p_value = gate.bootstrapped_ks(
            dv,
            gate.step_cdf(normal_ground_truth()),
            sample_size=8,
            number_of_bootstraps=3,
            significance_level=0.05,
            rng=np.random.default_rng(0),
        )
        self.assertIsInstance(passed, bool)
        self.assertTrue(0.0 <= p_value <= 1.0)


class HasNonFiniteMassTest(unittest.TestCase):
    def test_zero_mass_placeholder_does_not_count(self) -> None:
        dv = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=1.0),
                DiracDelta(position=float("nan"), mass=0.0),
            ]
        )
        self.assertFalse(gate.has_non_finite_mass(dv))

    def test_mass_at_infinity_counts(self) -> None:
        dv = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.9),
                DiracDelta(position=float("inf"), mass=0.1),
            ]
        )
        self.assertTrue(gate.has_non_finite_mass(dv))

    def test_non_finite_mass_value_counts(self) -> None:
        dv = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.5),
                DiracDelta(position=2.0, mass=float("nan")),
            ]
        )
        self.assertTrue(gate.has_non_finite_mass(dv))


class SupportSpreadTest(unittest.TestCase):
    def test_point_mass_has_zero_spread(self) -> None:
        dv = DistributionalValue.from_samples([3.0] * 10)
        self.assertEqual(gate.support_spread(dv), 0.0)

    def test_ignores_zero_mass_placeholders(self) -> None:
        dv = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=0.0, mass=0.5),
                DiracDelta(position=2.0, mass=0.5),
                DiracDelta(position=float("inf"), mass=0.0),
            ]
        )
        self.assertEqual(gate.support_spread(dv), 2.0)


class SizePassesTest(unittest.TestCase):
    """
    The test is random, so these assert what holds across seeds rather
    than one verdict: a perfect representation at size 512 passes for
    nearly every seed, and a representation one standard deviation off
    fails at every size from 64 up.
    """

    def setUp(self) -> None:
        self.cdf = gate.step_cdf(normal_ground_truth())
        self.parameters = gate.TestParameters()

    def test_perfect_representation_passes_at_the_largest_size(self) -> None:
        passes = [
            gate.size_passes(
                normal_comb(512), self.cdf, 512, gate.TestParameters(seed=seed)
            )[0]
            for seed in range(10)
        ]
        self.assertGreaterEqual(sum(passes), 8, passes)

    def test_shifted_representation_fails_at_large_sizes(self) -> None:
        for size in [64, 128, 256, 512]:
            passed, p_value = gate.size_passes(
                normal_comb(size, shift=1.0),
                self.cdf,
                size,
                self.parameters,
            )
            self.assertFalse(passed, (size, p_value))

    def test_seed_makes_verdict_reproducible(self) -> None:
        first = gate.size_passes(normal_comb(32), self.cdf, 32, self.parameters)
        second = gate.size_passes(normal_comb(32), self.cdf, 32, self.parameters)
        self.assertEqual(first, second)


class GateExpressionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = gate.TestParameters()
        self.ground_truth = normal_ground_truth()

    def test_zero_spread_ground_truth_is_not_measured(self) -> None:
        values = [tagged(normal_comb(8), 8, Correlations.DISABLED)]
        decisions = gate.gate_expression(
            "x",
            DistributionalValue.from_samples([1.0] * 5),
            values,
            values,
            self.parameters,
        )
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].decision, gate.NOT_MEASURED)
        self.assertEqual(decisions[0].note, "ground truth has zero spread")

    def test_non_finite_ground_truth_mass_is_not_measured(self) -> None:
        """
        A tenth of the ground truth at infinity. Testing the finite nine
        tenths alone would let a representation that misses it pass.
        """
        ground_truth = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=0.0, mass=0.9),
                DiracDelta(position=float("inf"), mass=0.1),
            ]
        )
        values = [tagged(normal_comb(8), 8, Correlations.DISABLED)]
        decisions = gate.gate_expression(
            "x", ground_truth, values, values, self.parameters
        )
        self.assertEqual(decisions[0].decision, gate.NOT_MEASURED)
        self.assertEqual(decisions[0].note, "ground truth has non-finite mass")

    def test_modes_are_decided_separately(self) -> None:
        sizes = [16, 32, 64, 128, 256]
        baseline = [
            tagged(normal_comb(size), size, Correlations.DISABLED) for size in sizes
        ]
        under_test = [
            tagged(normal_comb(size, shift=2.0), size, Correlations.AUTOCORRELATION)
            for size in sizes
        ]
        decisions = {
            decision.correlation_tracking: decision
            for decision in gate.gate_expression(
                "x",
                self.ground_truth,
                baseline,
                under_test,
                self.parameters,
            )
        }
        self.assertEqual(
            set(decisions), {Correlations.DISABLED, Correlations.AUTOCORRELATION}
        )
        disabled = decisions[Correlations.DISABLED]
        self.assertEqual(disabled.decision, gate.MEASURED_ONE_SIDE)
        self.assertIsNone(disabled.under_test)
        self.assertEqual(present(disabled.baseline).size, 256)
        self.assertEqual(disabled.size_label, "256")
        autocorrelation = decisions[Correlations.AUTOCORRELATION]
        self.assertEqual(autocorrelation.decision, gate.NOT_MEASURED)
        self.assertIsNone(autocorrelation.baseline)
        self.assertFalse(present(autocorrelation.under_test).passed)
        self.assertEqual(present(autocorrelation.under_test).size, 256)

    def test_only_the_largest_size_is_tested(self) -> None:
        """Off by two std at 16, exact at 32: only size 32 counts."""
        values = [
            tagged(normal_comb(16, shift=2.0), 16, Correlations.DISABLED),
            tagged(normal_comb(32), 32, Correlations.DISABLED),
        ]
        decision = gate.gate_expression(
            "x", self.ground_truth, values, values, self.parameters
        )[0]
        self.assertEqual(present(decision.baseline).size, 32)
        self.assertTrue(present(decision.baseline).passed)
        self.assertEqual(decision.decision, gate.MEASURED)

    def test_occurrences_are_decided_separately(self) -> None:
        """
        One expression name at two places, so two values per size with the
        same configuration string. The first converges, the second is two
        standard deviations off, and each must get its own decision in the
        order the comparison tool pairs them.
        """
        sizes = [16, 32, 64, 128, 256]
        values = []
        for size in sizes:
            values.append(tagged(normal_comb(size), size, Correlations.DISABLED))
            values.append(
                tagged(normal_comb(size, shift=2.0), size, Correlations.DISABLED)
            )
        decisions = gate.gate_expression(
            "x", self.ground_truth, values, values, self.parameters
        )
        self.assertEqual(
            [(d.correlation_tracking, d.occurrence) for d in decisions],
            [(Correlations.DISABLED, 0), (Correlations.DISABLED, 1)],
        )
        self.assertEqual(decisions[0].decision, gate.MEASURED)
        self.assertTrue(present(decisions[0].baseline).passed)
        self.assertEqual(present(decisions[0].baseline).size, 256)
        self.assertEqual(decisions[1].decision, gate.NOT_MEASURED)
        self.assertFalse(present(decisions[1].baseline).passed)
        self.assertEqual(present(decisions[1].baseline).size, 256)

    def test_unsampleable_representation_fails(self) -> None:
        empty = DistributionalValue(dirac_deltas=[])
        values = [
            tagged(normal_comb(16), 16, Correlations.DISABLED),
            tagged(empty, 32, Correlations.DISABLED),
        ]
        decision = gate.gate_expression(
            "x", self.ground_truth, values, [], self.parameters
        )[0]
        self.assertEqual(present(decision.baseline).size, 32)
        self.assertFalse(present(decision.baseline).passed)
        self.assertTrue(np.isnan(present(decision.baseline).p_value))


AC = "Athens-32-Autocorrelation"
REPORT = "\n".join(
    [
        "| Benchmark | Expression | Baseline Conf | Under-test Conf | WD Diff |",
        "|---|---|---|---|---|",
        "| demo | out[0] | Athens-32 | Athens-32 | 0 |",
        f"| demo | out[0] | {AC} | {AC} | 0.1 |",
        "| demo | out[1] | Athens-32 | Athens-32 | 0 |",
    ]
)


class FilterReportTest(unittest.TestCase):
    def test_drops_only_excluded_rows(self) -> None:
        excluded = {("out[0]", Correlations.AUTOCORRELATION, 0)}
        text, dropped = gate.filter_report(REPORT, excluded, "empty")
        self.assertEqual(dropped, 1)
        self.assertNotIn("Autocorrelation", text)
        self.assertIn("| out[1]", text)
        self.assertEqual(text.count("\n"), 4)

    def test_keeps_everything_without_exclusions(self) -> None:
        text, dropped = gate.filter_report(REPORT, set(), "empty")
        self.assertEqual(dropped, 0)
        self.assertEqual(text, REPORT + "\n")

    def test_empty_table_becomes_note(self) -> None:
        excluded = {
            ("out[0]", Correlations.DISABLED, 0),
            ("out[0]", Correlations.AUTOCORRELATION, 0),
            ("out[1]", Correlations.DISABLED, 0),
        }
        text, dropped = gate.filter_report(REPORT, excluded, "empty")
        self.assertEqual(dropped, 3)
        self.assertEqual(text, "empty\n")

    def test_lines_outside_tables_pass_through(self) -> None:
        text, _ = gate.filter_report("prose\n" + REPORT, set(), "empty")
        self.assertTrue(text.startswith("prose\n| Benchmark"))

    def test_occurrence_selects_among_rows_with_the_same_conf(self) -> None:
        """
        The same expression and configuration twice, as the comparison tool
        writes when a name is traced at two places. Only the second is
        excluded, and the rows must be told apart by position.
        """
        report = "\n".join(
            [
                "| B | Expression | Base | Under-test Conf | WD |",
                "|---|---|---|---|---|",
                "| demo | out[0] | Athens-32 | Athens-32 | 0.1 |",
                "| demo | out[0] | Athens-32 | Athens-32 | 0.2 |",
                "| demo | out[1] | Athens-32 | Athens-32 | 0.3 |",
            ]
        )
        excluded = {("out[0]", Correlations.DISABLED, 1)}
        text, dropped = gate.filter_report(report, excluded, "empty")
        self.assertEqual(dropped, 1)
        self.assertIn("| 0.1 |", text)
        self.assertNotIn("| 0.2 |", text)
        self.assertIn("| 0.3 |", text)


class ConfigurationCorrelationTest(unittest.TestCase):
    def test_suffix_selects_autocorrelation(self) -> None:
        self.assertEqual(
            gate.configuration_correlation("Athens-32-Autocorrelation"),
            Correlations.AUTOCORRELATION,
        )
        self.assertEqual(
            gate.configuration_correlation("Jupiter-16"),
            Correlations.DISABLED,
        )


class GateTableTest(unittest.TestCase):
    def test_lists_every_decision(self) -> None:
        decisions = [
            gate.GateDecision(
                "a",
                Correlations.DISABLED,
                side(True),
                side(False, size=256),
                *gate.decide(side(True), side(False, size=256)),
            ),
            gate.GateDecision(
                "b",
                Correlations.DISABLED,
                None,
                None,
                gate.NOT_MEASURED,
                "ground truth has zero spread",
            ),
            gate.GateDecision(
                "c", "", None, None, gate.MEASURED, "scalar output, not gated"
            ),
        ]
        table = gate.gate_table(decisions, "demo", "Athens")
        lines = table.splitlines()
        self.assertEqual(len(lines), 5)
        self.assertIn("Occurrence", lines[0])
        self.assertIn("| Size ", lines[0])
        self.assertIn("256 / 512", lines[2])
        self.assertIn("pass (p=0.50)", lines[2])
        self.assertIn(gate.MEASURED_ONE_SIDE, lines[2])
        self.assertIn("n/a", lines[3])
        self.assertIn(gate.NOT_MEASURED, lines[3])
        self.assertIn("| Athens ", lines[4])
        self.assertIn("scalar output, not gated", lines[4])
        self.assertTrue(table.endswith("\n"))

    def test_no_decisions_gives_empty_string(self) -> None:
        self.assertEqual(gate.gate_table([], "demo", "Athens"), "")


_TABLE = "TracingTable"
_BENCHMARKING_VARIABLES_YAML = """
BenchmarkingVariables:
    - VariableName: good
      VariableDescription: good
      OutputObject: Distribution
    - VariableName: bad
      VariableDescription: bad
      OutputObject: Distribution
"""


class MainTest(unittest.TestCase):
    """
    ``main`` on a report that ``distance_uxhw_vs_mc`` wrote. Both runs place
    ``good`` on the ground truth and ``bad`` ten standard deviations off it.
    """

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        directory = self._directory.name
        self.report = os.path.join(directory, "report.md")
        self.measured = os.path.join(directory, "measured.md")
        self.gate_report = os.path.join(directory, "gate.md")

        # Evenly spaced quantiles rather than random draws, so the ground
        # truth is the normal itself and the verdict on ``good`` does not
        # hinge on sampling noise.
        quantiles = (np.arange(2_000) + 0.5) / 2_000
        samples = [float(sample) for sample in norm.ppf(quantiles, 100.0, 15.0)]
        variables = []
        for index, name in enumerate(["good", "bad"]):
            variable = BenchmarkingVariable(
                name=name, description=name, value_id=f"value-{index}"
            )
            variable.distribution_samples.set_values(samples)
            variables.append(variable)
        ground_truth = os.path.join(directory, "ground-truth.db")
        generate_database_from_mc_samples(ground_truth, variables)

        uxhw = os.path.join(directory, "uxhw.db")
        build_uxhw_database(
            uxhw,
            _TABLE,
            [
                SyntheticUxhwExpression("good", "value-0", 100.0, 15.0),
                SyntheticUxhwExpression("bad", "value-1", 250.0, 15.0),
            ],
            [SyntheticUxhwConfig("Athens", 64, Correlations.DISABLED)],
        )

        yaml_path = os.path.join(directory, "signaloid.yaml")
        with open(yaml_path, "w") as file:
            file.write(_BENCHMARKING_VARIABLES_YAML)

        # The same argument list goes to both tools, as callers pass it.
        self.arguments = [
            "--ground-truth-database",
            ground_truth,
            "--uxhw-baseline-database",
            uxhw,
            "--uxhw-under-test-database",
            uxhw,
            "--uxhw-tables",
            _TABLE,
            "--baseline-uncertainty-representation-type",
            "Athens",
            "--under-test-uncertainty-representation-type",
            "Athens",
            "-s",
            "64",
            "-y",
            yaml_path,
            "--benchmark",
            "synthetic",
        ]
        with redirect_stdout(StringIO()):
            distance_uxhw_vs_mc.main(self.arguments + ["-o", self.report])

    def tearDown(self) -> None:
        self._directory.cleanup()

    def _gate(self, *extra: str) -> int:
        """
        Run the gate's ``main`` with the shared arguments, silencing stdout.

        Args:
            extra: Gate-only arguments.

        Returns:
            The exit status ``main`` returns.
        """
        with redirect_stdout(StringIO()):
            return gate.main(self.arguments + list(extra))

    def test_drops_the_configuration_that_converges_on_neither_side(self) -> None:
        """Only the row of ``bad`` leaves the report."""
        status = self._gate(
            "--comparison-report",
            self.report,
            "--output",
            self.measured,
            "--gate-output",
            self.gate_report,
        )

        self.assertEqual(status, 0)
        with open(self.report) as file:
            report = file.read()
        self.assertIn("| bad ", report)
        with open(self.measured) as file:
            measured = file.read()
        self.assertIn("| good ", measured)
        self.assertNotIn("| bad ", measured)

        with open(self.gate_report) as file:
            rows = {
                line.split("|")[2].strip(): line
                for line in file.read().splitlines()[2:]
            }
        self.assertIn(gate.MEASURED, rows["good"])
        self.assertIn(gate.NOT_MEASURED, rows["bad"])
        self.assertIn("neither SDK converges", rows["bad"])

    def test_gate_report_alone_leaves_no_filtered_report(self) -> None:
        """Without a comparison report the gate only writes its table."""
        self._gate("--gate-output", self.gate_report)

        self.assertTrue(os.path.exists(self.gate_report))
        self.assertFalse(os.path.exists(self.measured))

    def test_comparison_report_needs_output(self) -> None:
        """--comparison-report without --output is a usage error."""
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
            self._gate(
                "--comparison-report",
                self.report,
                "--gate-output",
                self.gate_report,
            )

        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
