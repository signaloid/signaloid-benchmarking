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

import os
from typing import NotRequired, TypedDict
from tabulate import tabulate
import signaloid.distributional_information_plotting.plot_wrapper as plot_wrapper
from signaloid.distributional_information_plotting.plot_histogram_dirac_deltas import (
    PlotData,
)
from signaloid_benchmarking.equivalent_mc.equivalent_monte_carlo import (
    EquivalentMonteCarlo,
)
from signaloid_benchmarking.equivalent_mc.load import (
    _load_uxhw_distributions,
    _load_ground_truth,
)
from signaloid_benchmarking.equivalent_mc.equivalent_mc_utils import (
    _generate_adversary_list,
    _generate_equivalent_mc_plots,
    _compute_shared_xlim,
)
from signaloid_benchmarking.config import (
    RepresentationTypes,
)

from signaloid_benchmarking.types import (
    BenchmarkingVariable,
    TaggedDistributionalValue,
)
from signaloid_benchmarking.distribution_helpers.representation_health import (
    _representation_blow_up_reason,
)

# Printed in the Mean / Variance columns for a blown-up (degraded)
# representation, in place of the overflowing statistics.
BLOW_UP_TABLE_TOKEN = "blow-up / excluded"


class LoadDataComputeEquivalentMCArgs(TypedDict):
    """
    Bundle of inputs for :func:`load_data_and_compute_equivalent_mc`: database
    locations/tables, the UxHw sweep parameters, distance/reporting settings,
    and plotting toggles.
    """

    benchmarking_variables: list[BenchmarkingVariable]
    ground_truth_database_path: str
    ground_truth_table_name: str
    adversary_database_path: str
    adversary_table_name: str
    adversary_size_step: int
    adversary_size_min: int
    adversary_size_max: int | None
    uxhw_database_path: str
    uxhw_table_names: list[str]
    uxhw_ur_types: list[str]
    uxhw_ur_sizes: list[int]
    correlations: list[str]
    distance_type: str
    n_processes: int
    n_adversaries: int
    plot_comparison_distributions: bool
    use_clt: bool
    ground_truth_type: str
    plot_adversary_distances: bool
    use_adaptive_steps: bool
    plot_distributions: bool
    use_binned_uxhw: bool
    plots_dir: str
    # Ignored. Kept optional so callers that still set them type-check.
    output_file: NotRequired[str | None]
    auto_prefix: NotRequired[bool]
    reporting_methods: NotRequired[list[str]]


def _print_uxhw_table(uxhw: list[TaggedDistributionalValue]) -> None:
    """
    Print a table of the loaded UxHw configurations and their mean/variance.

    Blown-up representations are shown as ``"blow-up / excluded"`` instead of
    their (overflowing) statistics.

    Args:
        uxhw: The loaded UxHw configurations to tabulate.
    """
    table_data = [["UR_type", "UR_order", "Correlation Tracking", "Mean", "Variance"]]
    alignments = ["left", "right", "left", "left", "left"]
    # representation_type / correlation_tracking come from the carrier metadata
    # (never off a Distribution). Numeric stats come from the carried dv.
    for uxhw_conf in uxhw:
        # The dv is freshly re-loaded here (it does not carry the Step-6
        # inf-distance flag), so re-detect a blow-up. A blown representation
        # parks mass at ~1e303, where reading .mean / .variance overflows and
        # crashes the sweep. Emit a "blow-up / excluded" row instead.
        if _representation_blow_up_reason(uxhw_conf.dv) is not None:
            mean_cell: str = BLOW_UP_TABLE_TOKEN
            variance_cell: str = BLOW_UP_TABLE_TOKEN
        else:
            mean_cell = str(uxhw_conf.dv.mean)
            variance_cell = str(uxhw_conf.dv.variance)
        table_data.append(
            [
                str(uxhw_conf.representation_type),
                str(uxhw_conf.dv.UR_order),
                str(uxhw_conf.correlation_tracking),
                mean_cell,
                variance_cell,
            ]
        )
    print(
        tabulate(table_data, headers="firstrow", tablefmt="simple", colalign=alignments)
    )
    print()


def _resolve_plot_path(filename: str, plots_dir: str) -> str:
    """
    Prepend ``plots_dir`` to a plot filename when set.

    Args:
        filename: The plot filename.
        plots_dir: Directory to place the plot in. Unchanged when empty.

    Returns:
        The joined path, or ``filename`` unchanged when ``plots_dir`` is empty.
    """
    if plots_dir:
        return os.path.join(plots_dir, filename)
    return filename


def _generate_distribution_plots(
    variable: BenchmarkingVariable,
    ground_truth: TaggedDistributionalValue,
    uxhw_from_table: list[TaggedDistributionalValue],
    plots_dir: str,
    xlim: tuple[float, float] | None = None,
) -> None:
    """
    Generate the ground-truth plot and a plot per UxHw configuration.

    Args:
        variable: The benchmarking variable being plotted.
        ground_truth: The ground-truth distribution.
        uxhw_from_table: The UxHw configurations loaded from the database.
        plots_dir: Directory to write the plots into. Defaults to the current working directory when empty.
        xlim: Limits applied to the x-axis for every plot. ``None`` falls back to per-plot auto-scaling.
    """
    try:
        print("Generating Ground Truth Plot.")
        gt_plot_path = _resolve_plot_path(
            f"{variable.formatted_description}-ground-truth.png", plots_dir
        )
        plot_wrapper.plot(
            plot_data=PlotData(ground_truth.dv),
            path=gt_plot_path,
            save=True,
            xlim=xlim,
        )
    except Exception as e:
        print(
            f"Failed to generate ground truth plot for "
            f"{variable.description} with error: {e}"
        )

    # The per-config filename embeds the carrier's config string (its repr,
    # e.g. "Athens-16" or "Athens-16-Autocorrelation") so it matches exactly
    # what report_writer reconstructs from the results table.
    for dist in uxhw_from_table:
        assert dist.correlation_tracking is not None
        plot_name = _resolve_plot_path(
            f"{variable.formatted_description}-{dist}.png",
            plots_dir,
        )
        try:
            plot_wrapper.plot(
                plot_data=PlotData(dist.dv),
                path=plot_name,
                save=True,
                xlim=xlim,
            )
        except Exception as e:
            print(f"Failed to generate plot {plot_name} with error: {e}")


def load_data_and_compute_equivalent_mc(args: LoadDataComputeEquivalentMCArgs) -> None:
    """
    Run the full equivalent-MC analysis for each benchmarking variable.

    For every variable: loads the ground truth, adversaries, and UxHw
    distributions from the databases, prints the UxHw table, computes and
    reports the EMCC, and generates the configured plots.

    Args:
        args: The equivalent-MC arguments (database locations, sweep
            parameters, distance/reporting settings, and plotting toggles).
    """
    optional_args = {}
    n_processes = args["n_processes"]
    n_adversaries = args["n_adversaries"]
    plots_dir = args["plots_dir"]
    distance_type = args["distance_type"]

    # Loop through variables
    for variable in args["benchmarking_variables"]:
        print("=================================================================")
        print("Traced expression:", variable.name)
        print("Expression description:", variable.description)
        print("Expression type:", variable.type)
        print()

        # Load ground truth. A variable with its own analytic reference
        # reads weighted samples whatever the run-wide table and type are.
        # Otherwise the table name only picks the table and the sample
        # format follows the ground-truth type, since this function is
        # public and a caller may give its table any name.
        ground_truth = _load_ground_truth(
            db_path=args["ground_truth_database_path"],
            table=variable.ground_truth_table(args["ground_truth_table_name"]),
            target_expression=variable.name,
            expression_type=variable.type,
            monte_carlo=(
                variable.ground_truth_table(args["ground_truth_type"])
                == RepresentationTypes.MONTE_CARLO
            ),
        )

        # Load adversaries
        adversary_list: list[TaggedDistributionalValue] = _generate_adversary_list(
            args=args, target_expr=variable.name, expr_type=variable.type
        )
        optional_args["adversary_mc"] = adversary_list

        # Get all distributions from databases
        variable_types, variable_sizes, _ = variable.representation_matrix(
            representation_types=args["uxhw_ur_types"],
            representation_sizes=args["uxhw_ur_sizes"],
            correlations=args["correlations"],
        )
        uxhw = _load_uxhw_distributions(
            db_path=args["uxhw_database_path"],
            tables=args["uxhw_table_names"],
            target_expr=variable.name,
            ur_types=variable_types,
            ur_sizes=variable_sizes,
        )

        # A single x-axis domain shared by the ground-truth, UxHw, and
        # representative MC plots. The adversary distributions contribute an
        # outlier-robust range so Monte Carlo tails do not blow up the domain.
        shared_xlim = None
        if args["plot_distributions"]:
            shared_xlim = _compute_shared_xlim(
                ground_truth=ground_truth,
                uxhw=uxhw,
                adversaries=adversary_list,
            )
            _generate_distribution_plots(
                variable, ground_truth, uxhw, plots_dir, xlim=shared_xlim
            )

        _print_uxhw_table(uxhw=uxhw)

        # Compute
        emcc = EquivalentMonteCarlo(
            ground_truth=ground_truth,
            uxhw_data=uxhw,
            variable=variable,
            distance_type=distance_type,
            use_binned_uxhw=args["use_binned_uxhw"],
            n_processes=n_processes,
            n_adversaries=n_adversaries,
            use_clt=args["use_clt"],
            adversary_size_step=args["adversary_size_step"],
            adversary_size_min=args["adversary_size_min"],
            adversary_size_max=args["adversary_size_max"],
            **optional_args,
        )

        emcc.compute_distance_data(use_adaptive_steps=args["use_adaptive_steps"])
        emcc.compute_and_report_emmc()

        # Generate plots
        _generate_equivalent_mc_plots(
            emcc=emcc,
            args=args,
            target_expr=variable.name,
            distance_type=distance_type,
            xlim=shared_xlim,
        )
