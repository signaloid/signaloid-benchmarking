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
Bootstrapped Kolmogorov-Smirnov gate for distance_uxhw_vs_mc reports.

Tests whether each UxHw configuration converges to the ground truth, and drops
the rows of the configurations that converge on neither side from the
comparison report. README.md in this directory explains the test, the
decisions, and why the statistic is computed here rather than by scipy.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import chi2, kstwo  # type: ignore
from tabulate import tabulate

from signaloid_benchmarking.config import (
    Correlations,
    RepresentationTypes,
    VariableTypes,
)
from signaloid_benchmarking.differential_accuracy.distance_uxhw_vs_mc import (
    benchmarking_variables,
    pairing_key,
)
from signaloid_benchmarking.equivalent_mc.load import (
    _load_ground_truth,
    _load_uxhw_distributions,
)
from signaloid_benchmarking.types import TaggedDistributionalValue
from signaloid.distributional.distributional import DistributionalValue

DEFAULT_SIGNIFICANCE_LEVEL = 0.05
DEFAULT_NUMBER_OF_BOOTSTRAPS = 10
DEFAULT_SEED = 0

MEASURED = "✅ measured"
MEASURED_ONE_SIDE = "⚠️ measured"
NOT_MEASURED = "⛔ not measured"

EXPRESSION_COLUMN = "Expression"
UNDER_TEST_CONF_COLUMN = "Under-test Conf"


@dataclass
class TestParameters:
    """Bootstrapped KS parameters shared by every call of one run."""

    significance_level: float = DEFAULT_SIGNIFICANCE_LEVEL
    number_of_bootstraps: int = DEFAULT_NUMBER_OF_BOOTSTRAPS
    seed: int = DEFAULT_SEED


@dataclass
class StepCdf:
    """
    Right-continuous step CDF with its left limit, both vectorised.

    Attributes:
        sorted_positions: Support in ascending order.
        cumulative: Mass at or below each position, with a leading zero.
    """

    sorted_positions: np.ndarray
    cumulative: np.ndarray

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.cumulative[np.searchsorted(self.sorted_positions, x, side="right")]

    def left_limit(self, x: np.ndarray) -> np.ndarray:
        return self.cumulative[np.searchsorted(self.sorted_positions, x, side="left")]


@dataclass
class SideResult:
    """Verdict of one database for one configuration at its largest size."""

    size: int
    passed: bool
    p_value: float

    def summary(self) -> str:
        verdict = "pass" if self.passed else "fail"
        return f"{verdict} (p={self.p_value:.2f})"


@dataclass
class GateDecision:
    """Verdict for one expression, correlation mode and occurrence."""

    expression: str
    correlation_tracking: str
    baseline: SideResult | None
    under_test: SideResult | None
    decision: str
    note: str
    occurrence: int = 0

    @property
    def size_label(self) -> str:
        sizes = sorted(
            {side.size for side in (self.baseline, self.under_test) if side is not None}
        )
        return " / ".join(str(size) for size in sizes)


def decide(
    baseline: SideResult | None,
    under_test: SideResult | None,
) -> tuple[str, str]:
    """
    Combine the two sides into a decision and a note for the report.

    A configuration is measured when either side passes. If only one side
    does, the row is still measured but marked, since that difference is
    itself a regression or an improvement.

    Args:
        baseline: Verdicts of the baseline database, None without rows.
        under_test: Verdicts of the under-test database, None without rows.

    Returns:
        The decision label and a note.
    """
    baseline_converges = baseline is not None and baseline.passed
    under_test_converges = under_test is not None and under_test.passed
    if baseline_converges and under_test_converges:
        return MEASURED, ""
    if baseline_converges:
        return MEASURED_ONE_SIDE, "baseline converges, under-test does not"
    if under_test_converges:
        return MEASURED_ONE_SIDE, "under-test converges, baseline does not"
    return NOT_MEASURED, "neither SDK converges"


def _positive_mass_support(
    distribution: DistributionalValue,
) -> tuple[np.ndarray, np.ndarray]:
    """Positions and masses of the entries that carry mass."""
    positions = np.asarray(distribution.positions, dtype=float)
    masses = np.asarray(distribution.masses, dtype=float)
    keep = masses > 0.0
    return positions[keep], masses[keep]


def has_non_finite_mass(distribution: DistributionalValue) -> bool:
    """
    True when mass sits at NaN or infinity, or a mass is not itself finite.

    Zero-mass entries at special values are placeholders the parser may
    carry and do not count. Mass at a special value has no place on the
    real line, so neither a CDF nor a distance is defined for it.

    Args:
        distribution: Any distribution.

    Returns:
        Whether the distribution is untestable for that reason.
    """
    masses = np.asarray(distribution.masses, dtype=float)
    if not np.all(np.isfinite(masses)):
        return True
    positions, _ = _positive_mass_support(distribution)
    return not bool(np.all(np.isfinite(positions)))


def support_spread(distribution: DistributionalValue) -> float:
    """
    Width of the positive-mass support.

    Only meaningful for a distribution without non-finite mass, see
    has_non_finite_mass.

    Args:
        distribution: Ground truth distribution.

    Returns:
        Largest minus smallest position, zero for a point mass.
    """
    positions, _ = _positive_mass_support(distribution)
    if positions.size == 0:
        return 0.0
    return float(positions.max() - positions.min())


def step_cdf(distribution: DistributionalValue) -> StepCdf:
    """
    Step CDF of the positive-mass support, sorted once.

    Matches DistributionalValue.cdf. The test evaluates the CDF ten times
    per size and a ground truth holds up to a million positions, so
    sorting on every call is too slow.

    Args:
        distribution: Ground truth distribution.

    Returns:
        The CDF with its left limit.

    Raises:
        ValueError: when the distribution has non-finite mass, or no
            positive mass at all.
    """
    if has_non_finite_mass(distribution):
        raise ValueError("ground truth has mass at a non-finite position")
    positions, masses = _positive_mass_support(distribution)
    total_mass = float(masses.sum())
    if total_mass <= 0.0:
        raise ValueError("ground truth has no positive mass")
    order = np.argsort(positions)
    return StepCdf(
        sorted_positions=positions[order],
        cumulative=np.concatenate(([0.0], np.cumsum(masses[order]) / total_mass)),
    )


def kolmogorov_smirnov_statistic(sample: np.ndarray, cdf: StepCdf) -> float:
    """
    Largest gap between a sample's empirical CDF and a step CDF.

    Above the step CDF the gap is checked at each sample point. Below it
    the gap is checked just before each sample point, against the CDF's
    left limit, so an atom of the CDF that the sample lands on does not
    count as a gap. scipy's ks_1samp uses the CDF's value at the point on
    that side, which is only right for a continuous CDF.

    Args:
        sample: Observed values, any order.
        cdf: Reference step CDF.

    Returns:
        The two-sided statistic in [0, 1].
    """
    ordered = np.sort(np.asarray(sample, dtype=float))
    count = len(ordered)
    at_or_below = np.searchsorted(ordered, ordered, side="right") / count
    below = np.searchsorted(ordered, ordered, side="left") / count
    above_gap = np.max(at_or_below - cdf(ordered))
    below_gap = np.max(cdf.left_limit(ordered) - below)
    return float(max(above_gap, below_gap, 0.0))


def bootstrapped_ks(
    representation: DistributionalValue,
    cdf: StepCdf,
    sample_size: int,
    number_of_bootstraps: int,
    significance_level: float,
    rng: np.random.Generator,
) -> tuple[bool, float]:
    """
    Bootstrapped KS goodness-of-fit test of a representation.

    Same procedure as the library's bootstrapped_kolmogorov_smirnov: draw
    sample_size values from the representation, test them against the
    ground truth, repeat, and combine the p-values with Fisher's method.
    Only the statistic differs, see kolmogorov_smirnov_statistic.

    Args:
        representation: Discrete distribution to sample from.
        cdf: Ground truth step CDF.
        sample_size: Values drawn per bootstrap.
        number_of_bootstraps: Bootstrap repetitions.
        significance_level: Combined p-value below which the test fails.
        rng: Seeded generator.

    Returns:
        Whether the test passed, and the combined p-value.

    Raises:
        ValueError: when the representation has no positive mass, or a
            positive-mass position that is not finite.
    """
    if has_non_finite_mass(representation):
        raise ValueError("representation has mass at a non-finite position")
    positions, masses = _positive_mass_support(representation)
    if masses.sum() <= 0.0:
        raise ValueError("representation has no positive mass to sample")
    probabilities = masses / masses.sum()

    p_values = np.empty(number_of_bootstraps)
    for index in range(number_of_bootstraps):
        sample = rng.choice(positions, size=sample_size, p=probabilities)
        statistic = kolmogorov_smirnov_statistic(sample, cdf)
        p_values[index] = kstwo.sf(statistic, sample_size)

    fisher = -2.0 * float(np.sum(np.log(np.maximum(p_values, np.finfo(float).tiny))))
    combined = 1.0 - float(chi2.cdf(fisher, df=2 * number_of_bootstraps))
    return combined >= significance_level, combined


def size_passes(
    representation: DistributionalValue,
    true_cdf: StepCdf,
    size: int,
    parameters: TestParameters,
) -> tuple[bool, float]:
    """
    Run the bootstrapped KS test for one representation size.

    Draws N/2 samples per bootstrap as the retired tool did, so the test
    cannot resolve the representation's own quantisation.

    Args:
        representation: UxHw distribution at this size.
        true_cdf: Ground truth CDF.
        size: Representation size N.
        parameters: Significance level, bootstrap count and seed.

    Returns:
        Whether the test passed, and the combined p-value.

    Raises:
        ValueError: when the representation cannot be sampled.
    """
    return bootstrapped_ks(
        representation,
        true_cdf,
        sample_size=max(size // 2, 1),
        number_of_bootstraps=parameters.number_of_bootstraps,
        significance_level=parameters.significance_level,
        rng=np.random.default_rng(parameters.seed),
    )


def numbered_values(
    values: list[TaggedDistributionalValue],
) -> list[tuple[TaggedDistributionalValue, str, int]]:
    """
    Values in comparison-tool order, each with its mode and occurrence.

    distance_uxhw_vs_mc sorts both sides with pairing_key and pairs them in
    that order, so values whose configuration string collides (one
    expression name traced at several places) are paired by position. The
    same stable sort and running count reproduces that position.

    Args:
        values: UxHw distributions of one expression from one database.

    Returns:
        (value, correlation mode, occurrence) triples.
    """
    occurrences: dict[str, int] = {}
    numbered = []
    for value in sorted(values, key=pairing_key):
        configuration = repr(value)
        occurrence = occurrences.get(configuration, 0)
        occurrences[configuration] = occurrence + 1
        mode = value.correlation_tracking or Correlations.DISABLED
        numbered.append((value, mode, occurrence))
    return numbered


def test_side(
    values: list[TaggedDistributionalValue],
    true_cdf: StepCdf,
    parameters: TestParameters,
) -> dict[tuple[str, int], SideResult]:
    """
    Test the largest size of every configuration of one database.

    A representation the test cannot sample (empty, or with non-finite
    positive mass) fails with a NaN p-value.

    Args:
        values: UxHw distributions of one expression from one database.
        true_cdf: Ground truth CDF.
        parameters: Significance level, bootstrap count and seed.

    Returns:
        Verdicts keyed by correlation-tracking mode and occurrence.
    """
    largest: dict[tuple[str, int], TaggedDistributionalValue] = {}
    for value, mode, occurrence in numbered_values(values):
        key = (mode, occurrence)
        size = int(value.representation_size or 0)
        if key not in largest or _size_of(largest[key]) < size:
            largest[key] = value

    results: dict[tuple[str, int], SideResult] = {}
    for key, value in largest.items():
        size = _size_of(value)
        value.dv.drop_zero_mass_positions()
        try:
            passed, p_value = size_passes(value.dv, true_cdf, size, parameters)
        except ValueError:
            passed, p_value = False, float("nan")
        results[key] = SideResult(size, bool(passed), float(p_value))
    return results


def _size_of(value: TaggedDistributionalValue) -> int:
    return int(value.representation_size or 0)


def gate_expression(
    expression: str,
    ground_truth: DistributionalValue,
    baseline_values: list[TaggedDistributionalValue],
    under_test_values: list[TaggedDistributionalValue],
    parameters: TestParameters,
) -> list[GateDecision]:
    """
    Decide every correlation-tracking mode of one expression.

    Args:
        expression: Benchmarking variable name.
        ground_truth: Monte Carlo ground truth of the expression.
        baseline_values: Baseline distributions at every size and mode.
        under_test_values: Under-test distributions at every size and mode.
        parameters: Significance level, bootstrap count and seed.

    Returns:
        One decision per mode and occurrence present on either side.
    """
    keys = sorted(
        {
            (mode, occurrence)
            for side in (baseline_values, under_test_values)
            for _, mode, occurrence in numbered_values(side)
        }
    )
    reason = ""
    if has_non_finite_mass(ground_truth):
        reason = "ground truth has non-finite mass"
    elif support_spread(ground_truth) == 0.0:
        reason = "ground truth has zero spread"
    if reason:
        return [
            GateDecision(expression, mode, None, None, NOT_MEASURED, reason, occurrence)
            for mode, occurrence in keys
        ]
    true_cdf = step_cdf(ground_truth)
    baseline = test_side(baseline_values, true_cdf, parameters)
    under_test = test_side(under_test_values, true_cdf, parameters)
    decisions = []
    for mode, occurrence in keys:
        baseline_side = baseline.get((mode, occurrence))
        under_test_side = under_test.get((mode, occurrence))
        decision, note = decide(baseline_side, under_test_side)
        decisions.append(
            GateDecision(
                expression,
                mode,
                baseline_side,
                under_test_side,
                decision,
                note,
                occurrence,
            )
        )
    return decisions


def configuration_correlation(configuration: str) -> str:
    """
    Correlation-tracking mode encoded in a report configuration string.

    The comparison tool writes Athens-32 for disabled tracking and
    Athens-32-Autocorrelation otherwise.

    Args:
        configuration: Value of a Conf column.

    Returns:
        The mode keyword.
    """
    if configuration.endswith("-" + Correlations.AUTOCORRELATION):
        return Correlations.AUTOCORRELATION
    return Correlations.DISABLED


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_separator(line: str) -> bool:
    if not line.lstrip().startswith("|"):
        return False
    cells = _split_row(line)
    return bool(cells) and all(cell and set(cell) <= set("-: ") for cell in cells)


def filter_report(
    text: str, excluded: set[tuple[str, str, int]], note: str
) -> tuple[str, int]:
    """
    Drop the rows of a comparison report whose configuration is excluded.

    Rows are matched on the Expression column, the correlation mode of the
    Under-test Conf column, and the occurrence, which is the row's rank
    among the rows of its table with the same expression and configuration
    string, see numbered_values. A table left without rows is replaced by
    the note. Lines outside tables pass through.

    Args:
        text: Markdown written by distance_uxhw_vs_mc.
        excluded: (expression, correlation mode, occurrence) triples.
        note: Replacement for a table left empty.

    Returns:
        The filtered text and the number of rows dropped.
    """
    lines = text.splitlines()
    out: list[str] = []
    dropped = 0
    expression_index: int | None = None
    conf_index: int | None = None
    header_start: int | None = None
    rows_kept = 0
    seen: dict[tuple[str, str], int] = {}

    def close_table() -> None:
        nonlocal header_start
        if header_start is not None and rows_kept == 0:
            del out[header_start:]
            out.append(note)
        header_start = None

    for index, line in enumerate(lines):
        is_header = (
            line.lstrip().startswith("|")
            and index + 1 < len(lines)
            and _is_separator(lines[index + 1])
        )
        if is_header:
            close_table()
            header = _split_row(line)
            expression_index = header.index(EXPRESSION_COLUMN)
            conf_index = header.index(UNDER_TEST_CONF_COLUMN)
            header_start = len(out)
            rows_kept = 0
            seen = {}
            out.append(line)
            continue
        if not line.lstrip().startswith("|"):
            close_table()
            out.append(line)
            continue
        if _is_separator(line) or expression_index is None:
            out.append(line)
            continue
        cells = _split_row(line)
        configuration = cells[conf_index or 0]
        occurrence = seen.get((cells[expression_index], configuration), 0)
        seen[(cells[expression_index], configuration)] = occurrence + 1
        key = (
            cells[expression_index],
            configuration_correlation(configuration),
            occurrence,
        )
        if key in excluded:
            dropped += 1
            continue
        rows_kept += 1
        out.append(line)
    close_table()
    return "\n".join(out) + "\n", dropped


def gate_table(
    decisions: list[GateDecision], benchmark: str, representation: str
) -> str:
    """
    Render the decisions as a markdown table, one row per configuration.

    Args:
        decisions: Decisions of every expression and mode.
        benchmark: Demo name for the first column.
        representation: Representation pair label, such as Athens.

    Returns:
        A github-flavoured table ending in a newline, or an empty string.
    """
    if not decisions:
        return ""
    rows = [
        {
            "Benchmark": benchmark,
            "Expression": decision.expression,
            "Configuration": (
                f"{representation}-{decision.correlation_tracking}"
                if decision.correlation_tracking
                else representation
            ),
            "Occurrence": str(decision.occurrence),
            "Size": decision.size_label,
            "Baseline": (decision.baseline.summary() if decision.baseline else "n/a"),
            "Under-test": (
                decision.under_test.summary() if decision.under_test else "n/a"
            ),
            "Decision": decision.decision,
            "Note": decision.note,
        }
        for decision in decisions
    ]
    return (
        tabulate(rows, headers="keys", tablefmt="github", disable_numparse=True) + "\n"
    )


def load_largest_size(
    db_path: str,
    tables: list[str],
    expression: str,
    ur_types: list[str],
    sizes: list[int],
) -> list[TaggedDistributionalValue]:
    """
    Values at the largest of the requested sizes the database holds.

    Sizes are tried from the largest down and the first with rows wins, so
    the smaller sizes are never read or parsed.

    Args:
        db_path: UxHw database.
        tables: Tables to read.
        expression: Benchmarking variable name.
        ur_types: Representation keywords to match.
        sizes: Requested representation sizes.

    Returns:
        Every value at that size, any mode and occurrence.

    Raises:
        ValueError: when no requested size has rows.
    """
    error: ValueError | None = None
    for size in sorted(set(sizes), reverse=True):
        try:
            values = _load_uxhw_distributions(
                db_path=db_path,
                tables=tables,
                target_expr=expression,
                ur_types=ur_types,
                ur_sizes=[size],
            )
        except ValueError as loader_error:
            error = loader_error
            continue
        if values:
            return values
    raise ValueError(
        f"{db_path}: no rows for {expression} at any of the sizes "
        f"{sorted(set(sizes))}"
    ) from error


def build_parser() -> argparse.ArgumentParser:
    """
    Argument parser. The database options mirror distance_uxhw_vs_mc so a
    caller can pass the same argument list to both.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        prog="ks_convergence_gate",
        description=(
            "Decide with a bootstrapped KS test which configurations of a "
            "benchmark converge to the ground truth, and drop the others "
            "from its accuracy comparison report."
        ),
    )
    parser.add_argument(
        "--ground-truth-database",
        dest="ground_truth_database_path",
        required=True,
    )
    parser.add_argument(
        "--ground-truth-table",
        dest="ground_truth_table",
        default=RepresentationTypes.MONTE_CARLO,
    )
    parser.add_argument(
        "--uxhw-baseline-database",
        dest="uxhw_baseline_database_path",
        required=True,
    )
    parser.add_argument(
        "--uxhw-under-test-database",
        dest="uxhw_undertest_database_path",
        required=True,
    )
    parser.add_argument("--uxhw-tables", dest="uxhw_tables", nargs="+", required=True)
    parser.add_argument(
        "--baseline-uncertainty-representation-type",
        dest="uxhw_baseline_ur_type",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--under-test-uncertainty-representation-type",
        dest="uxhw_undertest_ur_type",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--uncertainty-representation-sizes",
        dest="ur_sizes",
        type=int,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "-y", "--yaml-signaloid", dest="signaloid_yaml_path", required=True
    )
    parser.add_argument("--benchmark", dest="benchmark", default="")
    parser.add_argument(
        "--comparison-report",
        dest="comparison_report_path",
        default=None,
        help="Report written by distance_uxhw_vs_mc to filter.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        default=None,
        help="Where to write the filtered comparison report.",
    )
    parser.add_argument(
        "--gate-output",
        dest="gate_output_path",
        required=True,
        help="Where to write the gate table.",
    )
    parser.add_argument(
        "--significance-level",
        type=float,
        default=DEFAULT_SIGNIFICANCE_LEVEL,
    )
    parser.add_argument(
        "--number-of-bootstraps",
        type=int,
        default=DEFAULT_NUMBER_OF_BOOTSTRAPS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.comparison_report_path is None) != (args.output_path is None):
        parser.error("--comparison-report and --output go together")

    parameters = TestParameters(
        significance_level=args.significance_level,
        number_of_bootstraps=args.number_of_bootstraps,
        seed=args.seed,
    )
    yaml_dict = yaml.safe_load(Path(args.signaloid_yaml_path).read_text())
    monte_carlo = args.ground_truth_table == RepresentationTypes.MONTE_CARLO
    representation = "/".join(args.uxhw_baseline_ur_type)
    if args.uxhw_undertest_ur_type != args.uxhw_baseline_ur_type:
        representation += "→" + "/".join(args.uxhw_undertest_ur_type)

    decisions: list[GateDecision] = []
    for expression, expression_type in benchmarking_variables(yaml_dict):
        if expression_type != VariableTypes.DISTRIBUTION:
            decisions.append(
                GateDecision(
                    expression, "", None, None, MEASURED, "scalar output, not gated"
                )
            )
            print(f"[gate] {expression}: scalar output, not gated")
            continue
        ground_truth = _load_ground_truth(
            db_path=args.ground_truth_database_path,
            table=args.ground_truth_table,
            target_expression=expression,
            expression_type=expression_type,
            monte_carlo=monte_carlo,
        ).dv
        baseline_values = load_largest_size(
            args.uxhw_baseline_database_path,
            args.uxhw_tables,
            expression,
            args.uxhw_baseline_ur_type,
            args.ur_sizes,
        )
        under_test_values = load_largest_size(
            args.uxhw_undertest_database_path,
            args.uxhw_tables,
            expression,
            args.uxhw_undertest_ur_type,
            args.ur_sizes,
        )
        for decision in gate_expression(
            expression,
            ground_truth,
            baseline_values,
            under_test_values,
            parameters,
        ):
            label = f"{expression} {representation}-"
            label += decision.correlation_tracking
            if decision.occurrence:
                label += f" #{decision.occurrence}"
            for name, side in (
                ("baseline", decision.baseline),
                ("under-test", decision.under_test),
            ):
                if side is not None:
                    print(
                        f"[gate] {label} {name} at size {side.size}: "
                        f"{side.summary()}"
                    )
            print(f"[gate] {label}: {decision.decision} {decision.note}")
            decisions.append(decision)

    Path(args.gate_output_path).write_text(
        gate_table(decisions, args.benchmark, representation)
    )
    if args.comparison_report_path is not None:
        excluded = {
            (
                decision.expression,
                decision.correlation_tracking,
                decision.occurrence,
            )
            for decision in decisions
            if decision.decision == NOT_MEASURED
        }
        filtered, dropped = filter_report(
            Path(args.comparison_report_path).read_text(),
            excluded,
            note=(
                f"No configurations measured for {args.benchmark} "
                f"({representation}): all failed the KS convergence gate."
            ),
        )
        Path(args.output_path).write_text(filtered)
        print(f"[gate] dropped {dropped} row(s) from the comparison report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
