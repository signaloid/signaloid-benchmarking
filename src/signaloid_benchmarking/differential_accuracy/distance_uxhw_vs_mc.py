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
Compare the distances of a baseline and an under-test UxHw configuration to
the same Monte Carlo ground truth.
"""

from argparse import ArgumentParser
import sys
import warnings
import yaml
import pandas as pd
from tabulate import tabulate
import numpy as np

# The loaders are only exposed as private helpers, so we import them by their
# underscored names.
from signaloid_benchmarking.equivalent_mc.load import (
    _load_ground_truth,
    _load_uxhw_distributions,
)
from signaloid_benchmarking.automation.benchmarking_utils import (
    expand_array_expressions,
)
from signaloid_benchmarking.distribution_helpers.representation_health import (
    _representation_blow_up_reason,
)
from signaloid_benchmarking.types import TaggedDistributionalValue
from signaloid_benchmarking.config import (
    RepresentationTypes,
    SignaloidYaml,
    VariableTypes,
)
from signaloid.distributional.distributional import DistributionalValue
from signaloid.distributional_distance.wasserstein import (
    wasserstein_1_uxhw_wrapper,
)
from signaloid.distributional_distance.binned_wasserstein import (
    binned_wasserstein_1_uxhw_wrapper,
)
from signaloid.distributional_distance.scalar import relative_error_uxhw_wrapper


def distributions_identical(a: DistributionalValue, b: DistributionalValue) -> bool:
    """
    Check if two distributions have identical positions and masses.
    """
    if len(a.positions) != len(b.positions) or len(a.masses) != len(b.masses):
        return False
    return bool((a.positions == b.positions).all() and (a.masses == b.masses).all())


def pairing_key(uxhw: TaggedDistributionalValue) -> tuple[int, str, str]:
    """
    Sort key that lines up baseline and under-test configurations for pairing.

    Size and correlation tracking come first because a pair must match on
    both. The representation type may differ between the two sides, so
    sorting on it first could pair e.g. Athens-16 with Atlas-32. The
    configuration string breaks ties deterministically.

    Args:
        uxhw: A loaded UxHw configuration.

    Returns:
        A tuple of (size, correlation tracking, configuration string).
    """
    assert uxhw.representation_size is not None
    assert uxhw.correlation_tracking is not None
    return (uxhw.representation_size, uxhw.correlation_tracking, repr(uxhw))


def benchmarking_variables(yaml_dict: dict) -> list[tuple[str, str]]:
    """
    List the (expression, output type) pairs a signaloid.yaml declares.

    Args:
        yaml_dict: Parsed signaloid.yaml, in either the BenchmarkingVariables
            or the older TraceVariables format.

    Returns:
        Expressions with their output type.
    """
    if SignaloidYaml.BENCHMARKING_VARIABLES in yaml_dict:
        return [
            (
                variable[SignaloidYaml.VARIABLE_NAME],
                variable[SignaloidYaml.OUTPUT_OBJECT],
            )
            for variable in yaml_dict[SignaloidYaml.BENCHMARKING_VARIABLES]
        ]
    return [
        (traced[SignaloidYaml.EXPRESSION], VariableTypes.DISTRIBUTION)
        for traced in expand_array_expressions(yaml_dict[SignaloidYaml.TRACE_VARIABLES])
    ]


def relative_difference_percentage(
    baseline_distance: float,
    undertest_distance: float,
) -> float:
    """
    Relative difference of the under-test distance with respect to the baseline,
    expressed as a percentage.

    A zero baseline distance means the baseline matched the ground truth exactly,
    so there is no scale to normalise against.  Rather than dividing by zero, we
    report:

    * ``0.0``   when the under-test distance is also zero (both are exact, so
      there is no change to report), and
    * ``+/-inf`` when the under-test distance is non-zero (any regression away
      from an exact baseline is unboundedly bad in relative terms), signed by
      the direction of the change.
    * ``NaN`` when the under-test distance is NaN, since there is no direction
      to sign an infinity with.

    ``scoring`` maps ``0.0`` to ⚪, and ``+inf`` and ``NaN`` to 🔴, which is the
    intent in each case.

    Args:
        baseline_distance:  Distance from the baseline UxHw result to ground truth.
        undertest_distance: Distance from the under-test UxHw result to ground truth.

    Returns:
        The relative difference as a percentage.
    """
    difference = undertest_distance - baseline_distance
    if baseline_distance == 0:
        if np.isnan(difference):
            return float("nan")
        if difference == 0:
            return 0.0
        return np.inf if difference > 0 else -np.inf
    return difference / baseline_distance * 100


def scoring(
    distance_diff_percentage: float,
    baseline_distance: float,
    undertest_distance: float,
) -> str:
    # If baseline was nan/inf but undertest is finite, that's an improvement
    if (np.isnan(baseline_distance) or np.isinf(baseline_distance)) and np.isfinite(
        undertest_distance
    ):
        return "🟢"
    if distance_diff_percentage < 0:
        return "🟢"
    elif np.isclose(distance_diff_percentage, 0, atol=1e-14):
        return "⚪"
    elif distance_diff_percentage <= 1e-6:
        return "🟡"
    elif distance_diff_percentage <= 1e-1:
        return "🟠"
    else:
        return "🔴"


def distances_to_ground_truth(
    expression_type: str,
    uxhw_dist: DistributionalValue,
    ground_truth: DistributionalValue,
    side: str,
) -> tuple[float, float]:
    """
    Compute the distance and binned distance from one UxHw result to ground truth.

    A degraded representation gets infinite distances instead of reaching the
    finite-only distance wrappers, which would raise and abort the command.
    This follows ``signaloid_benchmarking.automation.analysis``. The overflow
    magnitude check only applies to distribution outputs, because a large
    finite scalar is legitimate.

    Args:
        expression_type: Type of the benchmarking variable (distribution or scalar).
        uxhw_dist:       UxHw distribution to evaluate.
        ground_truth:    Reference distribution or value to measure distance against.
        side:            Which UxHw result this is, used in the warning.

    Returns:
        A tuple of (distance, binned_distance).
    """
    is_distribution = expression_type == VariableTypes.DISTRIBUTION
    blow_up_reason = _representation_blow_up_reason(
        uxhw_dist, check_magnitude=is_distribution
    )
    if blow_up_reason is not None:
        warnings.warn(
            f"Representation blow-up for {side} UxHw: {blow_up_reason}. "
            f"Setting its distances to inf."
        )
        return float("inf"), float("inf")

    if is_distribution:
        return (
            wasserstein_1_uxhw_wrapper(uxhw_dist, ground_truth),
            binned_wasserstein_1_uxhw_wrapper(uxhw_dist, ground_truth),
        )

    # No binned distance for scalar distributions, so set to same as non-binned distance for consistency in results table
    distance = relative_error_uxhw_wrapper(uxhw_dist, ground_truth)
    return distance, distance


def compute_distances(
    expression_type: str,
    uxhw_baseline_dist: DistributionalValue,
    uxhw_undertest_dist: DistributionalValue,
    ground_truth: DistributionalValue,
) -> tuple[float, float, float, float]:
    """Compute Wasserstein distances from baseline and undertest distributions to ground truth.

    For distributional outputs, computes both standard and binned Wasserstein-1 distances.
    For scalar outputs, computes scalar distances (binned distances are set equal to
    non-binned for consistency).

    Each side is guarded independently. A degraded side gets infinite distances,
    so a regression or a recovery is still scored against the other side.

    If the baseline and undertest distributions are identical, distance calculations
    are reused to avoid redundant computation.

    Args:
        expression_type:     Type of the benchmarking variable (distribution or scalar).
        uxhw_baseline_dist:  Baseline UxHw distribution to evaluate.
        uxhw_undertest_dist: Undertest UxHw distribution to evaluate.
        ground_truth:        Reference distribution or value to measure distance against.

    Returns:
        A tuple of (baseline_distance, undertest_distance,
                     baseline_binned_distance, undertest_binned_distance).
    """
    distance_baseline, binned_distance_baseline = distances_to_ground_truth(
        expression_type, uxhw_baseline_dist, ground_truth, "baseline"
    )

    # Check if undertest and baseline are identical, if they are, can half the number of distance calculations
    if distributions_identical(uxhw_baseline_dist, uxhw_undertest_dist):
        distance_undertest = distance_baseline
        binned_distance_undertest = binned_distance_baseline
    else:
        distance_undertest, binned_distance_undertest = distances_to_ground_truth(
            expression_type, uxhw_undertest_dist, ground_truth, "under-test"
        )

    return (
        distance_baseline,
        distance_undertest,
        binned_distance_baseline,
        binned_distance_undertest,
    )


def write_report(results: list[dict], benchmark: str | None, output_path: str) -> None:
    """
    Write every compared configuration to a markdown table.

    All rows are kept, including those whose baseline and under-test distances
    are identical, because the absolute distances are what characterise a
    representation and a run may deliberately compare an SDK against itself.
    Consumers that only care about what changed (the CI job summary, say) filter
    the table themselves rather than having the data discarded here.

    Args:
        results: One row per compared UxHw configuration.
        benchmark: Name of the evaluated benchmark, used in the empty report.
        output_path: Path of the markdown file to write.
    """
    df = pd.DataFrame(results)
    with open(output_path, "w") as file:
        if results:
            file.write(
                tabulate(
                    df,  # type: ignore[arg-type]
                    headers="keys",
                    tablefmt="github",
                    showindex=False,
                )
            )
        else:
            file.write(f"No configurations compared for {benchmark}.\n")


def main(argv: list[str] | None = None) -> None:
    parser = ArgumentParser(
        prog="distance_uxhw_vs_mc",
        description="Compare Wasserstein distances of baseline vs under-test "
        "UxHw configurations against a Monte Carlo ground truth.",
    )

    # Ground Truth data args
    parser.add_argument(
        "--ground-truth-database",
        dest="ground_truth_database_path",
        default=None,
        required=True,
        help="Path to database containing the ground truth Monte Carlo samples.",
    )

    parser.add_argument(
        "--ground-truth-table",
        dest="ground_truth_table",
        default=RepresentationTypes.MONTE_CARLO,
        help="Database table containing the ground truth Monte Carlo samples.",
    )

    # UxHw data args
    parser.add_argument(
        "--uxhw-baseline-database",
        dest="uxhw_baseline_database_path",
        default=None,
        required=True,
        help="Path to database containing the baseline UxHw data.",
    )

    parser.add_argument(
        "--uxhw-under-test-database",
        dest="uxhw_undertest_database_path",
        default=None,
        required=True,
        help="Path to database containing the under-test UxHw data.",
    )

    parser.add_argument(
        "--uxhw-tables",
        dest="uxhw_tables",
        nargs="+",
        action="extend",
        required=True,
        help="Database table containing the UxHw data.",
    )

    # UxHw configurations. These are the values matched against the DB UR_Type
    # column.
    representation_type_choices = [
        RepresentationTypes.ATHENS,
        RepresentationTypes.JUPITER,
        RepresentationTypes.ATLAS,
        RepresentationTypes.EUROPA,
    ]
    parser.add_argument(
        "--baseline-uncertainty-representation-type",
        dest="uxhw_baseline_ur_type",
        default=[],
        required=True,
        choices=representation_type_choices,
        action="extend",
        nargs="+",
        help="Uncertain Representation type to look for in the baseline UxHw data.",
    )

    parser.add_argument(
        "--under-test-uncertainty-representation-type",
        dest="uxhw_undertest_ur_type",
        default=[],
        required=True,
        choices=representation_type_choices,
        action="extend",
        nargs="+",
        help="Uncertain Representation type to look for in the under-test UxHw data.",
    )

    parser.add_argument(
        "-s",
        "--uncertainty-representation-sizes",
        dest="ur_sizes",
        type=int,
        default=[],
        required=True,
        action="extend",
        nargs="+",
        help="Uncertain Representation sizes to look for in the UxHw data",
    )

    parser.add_argument(
        "-y",
        "--yaml-signaloid",
        dest="signaloid_yaml_path",
        default=None,
        required=True,
        help="Source Signaloid yaml file.",
    )

    parser.add_argument(
        "--benchmark",
        dest="benchmark",
        default=None,
        help="Evaluated benchmark.",
    )

    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        default="output.md",
        help="Path to write the results markdown table to (default: output.md).",
    )

    args = parser.parse_args(argv)

    with open(args.signaloid_yaml_path, "r") as yaml_file:
        yaml_dict = yaml.safe_load(yaml_file)

    results: list[dict] = []

    # Loop over all target expressions
    for target_expr, expression_type in benchmarking_variables(yaml_dict):
        print(target_expr)
        # Load ground truth
        ground_truth: TaggedDistributionalValue
        monte_carlo = (
            True
            if args.ground_truth_table == RepresentationTypes.MONTE_CARLO
            else False
        )
        ground_truth = _load_ground_truth(
            db_path=args.ground_truth_database_path,
            table=args.ground_truth_table,
            target_expression=target_expr,
            expression_type=expression_type,
            monte_carlo=monte_carlo,
        )

        uxhw_baseline_list = _load_uxhw_distributions(
            db_path=args.uxhw_baseline_database_path,
            tables=args.uxhw_tables,
            target_expr=target_expr,
            ur_types=args.uxhw_baseline_ur_type,
            ur_sizes=args.ur_sizes,
        )
        uxhw_undertest_list = _load_uxhw_distributions(
            db_path=args.uxhw_undertest_database_path,
            tables=args.uxhw_tables,
            target_expr=target_expr,
            ur_types=args.uxhw_undertest_ur_type,
            ur_sizes=args.ur_sizes,
        )

        if len(uxhw_baseline_list) != len(uxhw_undertest_list):
            print(
                "UxHw baseline and UxHw under-test differ in configurations. Exiting."
            )
            sys.exit(-1)

        # Sort both lists so the zip pairs matching configurations. Without
        # sorting, groupby ordering from different databases may not match,
        # causing e.g. CORRELATION_OFF to be compared against COMMON_ANCESTOR.
        uxhw_baseline_list.sort(key=pairing_key)
        uxhw_undertest_list.sort(key=pairing_key)

        # Loop over UxHw configurations and compute distances to ground truth, then compare the distances between baseline and under-test UxHw configurations
        for uxhw_baseline, uxhw_undertest in zip(
            uxhw_baseline_list, uxhw_undertest_list
        ):
            # Equal counts do not mean the sorted lists line up. Representation
            # types may differ on purpose, but sizes and correlation tracking
            # must match, or e.g. Athens-16 is compared against Athens-32.
            if (
                uxhw_baseline.representation_size != uxhw_undertest.representation_size
                or uxhw_baseline.correlation_tracking
                != uxhw_undertest.correlation_tracking
            ):
                print(
                    f"Cannot pair UxHw baseline {uxhw_baseline!r} with UxHw "
                    f"under-test {uxhw_undertest!r}. Exiting."
                )
                sys.exit(-1)

            # The loaders return a TaggedDistributionalValue whose ``.dv`` is
            # already a parsed DistributionalValue ready for the distance
            # wrappers. Drop any zero-mass special-value (NaN/±inf) Diracs the
            # parser may carry so the finite-only distance validators accept the
            # values (mirrors signaloid_benchmarking.automation.analysis).
            uxhw_baseline_dist = uxhw_baseline.dv
            uxhw_undertest_dist = uxhw_undertest.dv
            if expression_type == VariableTypes.DISTRIBUTION:
                uxhw_baseline_dist.drop_zero_mass_positions()
                uxhw_undertest_dist.drop_zero_mass_positions()
            (
                distance_baseline,
                distance_undertest,
                binned_distance_baseline,
                binned_distance_undertest,
            ) = compute_distances(
                expression_type=expression_type,
                uxhw_baseline_dist=uxhw_baseline_dist,
                uxhw_undertest_dist=uxhw_undertest_dist,
                ground_truth=ground_truth.dv,
            )

            # Difference of WD distances
            distance_diff = distance_undertest - distance_baseline

            distance_diff_percentage = relative_difference_percentage(
                baseline_distance=distance_baseline,
                undertest_distance=distance_undertest,
            )

            # Difference of Binned WD distances
            binned_distance_diff = binned_distance_undertest - binned_distance_baseline

            binned_distance_diff_percentage = relative_difference_percentage(
                baseline_distance=binned_distance_baseline,
                undertest_distance=binned_distance_undertest,
            )

            # Store result
            results.append(
                {
                    "Benchmark": args.benchmark,
                    "Expression": target_expr,
                    "Baseline Conf": repr(uxhw_baseline),
                    "Under-test Conf": repr(uxhw_undertest),
                    "WD Baseline": distance_baseline,
                    "WD Under-test": distance_undertest,
                    "WD Diff": distance_diff,
                    "Relative Diff Percentage": distance_diff_percentage,
                    "Scoring": scoring(
                        distance_diff_percentage, distance_baseline, distance_undertest
                    ),
                    "Binned WD Baseline": binned_distance_baseline,
                    "Binned WD Under-test": binned_distance_undertest,
                    "Binned Diff": binned_distance_diff,
                    "Relative Binned Diff Percentage": binned_distance_diff_percentage,
                    "Binned Scoring": scoring(
                        binned_distance_diff_percentage,
                        binned_distance_baseline,
                        binned_distance_undertest,
                    ),
                }
            )

    write_report(results, args.benchmark, args.output_path)


if __name__ == "__main__":
    main()
