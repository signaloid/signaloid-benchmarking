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
from typing import TYPE_CHECKING
import numpy as np
from scipy.stats import anderson, foldnorm  # type: ignore
from signaloid_benchmarking.types import TaggedDistributionalValue
from signaloid.distributional.distributional import DistributionalValue
from signaloid_benchmarking.config import (
    DistanceMetrics,
    VariableTypes,
    EquivMC,
    BenchmarkingVariables,
    RepresentationTypes,
    ReportingNumbers,
    CltAssumptions,
    uses_binned_distance,
)
import signaloid.distributional_information_plotting.plot_wrapper as plot_wrapper
from signaloid.distributional_information_plotting.plot_histogram_dirac_deltas import (
    PlotData,
)
from signaloid.distributional_distance.wasserstein import wasserstein_p_distance
from numpy.typing import ArrayLike
import pandas as pd
import math
from heapq import merge
from signaloid_benchmarking.equivalent_mc.load import (
    _load_mc_scalar,
    _load_mc,
)
import numpy.typing as npt

if TYPE_CHECKING:
    from signaloid_benchmarking.equivalent_mc.equivalent_monte_carlo import (
        EquivalentMonteCarlo,
    )
    from signaloid_benchmarking.equivalent_mc.equivalent_mc_main import (
        LoadDataComputeEquivalentMCArgs,
    )
import matplotlib.pyplot as plt

# Quantile bounds used to derive an outlier-robust plotting range for
# sample-based distributions (e.g. Monte Carlo adversaries), so that a few
# extreme samples in the tails do not stretch the shared x-axis domain.
_ROBUST_RANGE_LOWER_QUANTILE = 0.005
_ROBUST_RANGE_UPPER_QUANTILE = 0.995

# Brownian-bridge paths are simulated in chunks of at most this many values,
# which bounds the working memory of the asymptotic-distance simulation.
_BROWNIAN_BRIDGE_CHUNK_ELEMENTS = 5_000_000


def _distribution_range(
    dv: DistributionalValue, robust: bool
) -> tuple[float, float] | None:
    """
    Compute the x-axis range spanned by a distribution.

    Args:
        dv: The distribution whose support range to measure.
        robust: When True, clip the range to the
            [``_ROBUST_RANGE_LOWER_QUANTILE``, ``_ROBUST_RANGE_UPPER_QUANTILE``]
            quantiles so extreme outliers (common in Monte Carlo samples) do not
            stretch the domain. When False, use the full support.

    Returns:
        The ``(low, high)`` range, or ``None`` if it cannot be determined
        (e.g. empty distribution or non-finite bounds).
    """
    treat_as_samples = (
        getattr(dv, "representation_type", None) == RepresentationTypes.SAMPLES
    )
    lower, upper = (
        (_ROBUST_RANGE_LOWER_QUANTILE, _ROBUST_RANGE_UPPER_QUANTILE)
        if robust
        else (0.0, 1.0)
    )
    try:
        low, high = dv.inverse_cdf(
            np.array([lower, upper]), treat_as_samples=treat_as_samples
        )
    except Exception:
        return None
    if not (math.isfinite(low) and math.isfinite(high)) or high < low:
        return None
    return float(low), float(high)


def _compute_shared_xlim(
    ground_truth: TaggedDistributionalValue,
    uxhw: list[TaggedDistributionalValue],
    adversaries: list[TaggedDistributionalValue],
    padding_fraction: float = 0.05,
) -> tuple[float, float] | None:
    """
    Compute a single x-axis domain shared by the ground-truth, UxHw, and
    adversary/Monte-Carlo distribution plots so they are directly comparable.

    The ground-truth and UxHw distributions contribute their full support (they
    are trustworthy binned representations), while the adversary distributions
    contribute an outlier-robust range so that Monte Carlo tails do not blow up
    the domain. The union of these ranges, padded by ``padding_fraction`` on
    each side (matching the padding ``plot_wrapper.plot`` applies when it
    auto-scales), is returned.

    Args:
        ground_truth: The ground-truth distribution.
        uxhw: The UxHw configurations to be plotted.
        adversaries: The adversary distributions the Monte Carlo plots are
            drawn from.
        padding_fraction: Fraction of the total range to pad on each side.

    Returns:
        The shared ``(min, max)`` x-limits, or ``None`` if no range could be
        determined (callers then fall back to per-plot auto-scaling).
    """
    ranges: list[tuple[float, float]] = []

    gt_range = _distribution_range(ground_truth.dv, robust=False)
    if gt_range is not None:
        ranges.append(gt_range)

    for dist in uxhw:
        dist_range = _distribution_range(dist.dv, robust=False)
        if dist_range is not None:
            ranges.append(dist_range)

    for adversary in adversaries:
        adversary_range = _distribution_range(adversary.dv, robust=True)
        if adversary_range is not None:
            ranges.append(adversary_range)

    if not ranges:
        return None

    min_x = min(low for low, _ in ranges)
    max_x = max(high for _, high in ranges)
    range_spacing = padding_fraction * (max_x - min_x)
    return (min_x - range_spacing, max_x + range_spacing)


def _compute_wasserstein_1_asymptotic_distribution_brownian_bridge(
    ground_truth: TaggedDistributionalValue,
    num_points: int,
    num_samples: int,
    seed: int = 0,
) -> DistributionalValue:
    """
    Simulate the distribution of ``∫ |B(F(x))| dx``, where ``B`` is a
    standard Brownian bridge and ``F`` is the ground truth's CDF.

    By Theorem 2.1 of del Barrio, Giné and Matrán (1999) this is the limit in
    law of ``sqrt(N) * W1(F_N, F)`` for an N-sample empirical distribution
    ``F_N``, provided ``∫ sqrt(F (1 - F)) dx`` is finite. The integral is
    evaluated on the knots of the ground truth's own step CDF, so it needs no
    density estimate and no cutoff, and atoms and gaps in the support are
    handled exactly.

    Args:
        ground_truth: The ground-truth distribution.
        num_points: Upper bound on the number of CDF knots used.
        num_samples: Number of Brownian-bridge paths to simulate.
        seed: Seed of the random number generator, so a run is reproducible.

    Returns:
        The asymptotic Wasserstein-1 distance distribution.
    """
    positions, cdf = _ground_truth_cdf_knots(ground_truth, num_points)
    samples = _simulate_wasserstein_1_asymptotic_samples(
        positions, cdf, num_samples, seed
    )
    dist = DistributionalValue.from_samples(samples)
    # Tag as SAMPLES so a downstream consumer deriving `treat_as_samples` from
    # `representation_type` takes the samples branch (`np.quantile` inverse-CDF
    # / auto-binned pdf).
    setattr(dist, "representation_type", RepresentationTypes.SAMPLES)

    return dist


def _ground_truth_cdf_knots(
    ground_truth: TaggedDistributionalValue, max_knots: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduce the ground truth to the knots of its step CDF.

    Sorts the support, merges coincident positions into one atom and thins
    the result to at most ``max_knots`` knots spaced evenly in rank, always
    keeping the first and the last. The CDF returned is that of the thinned
    ground truth itself, which is a mass-preserving quantisation of the
    original.

    Args:
        ground_truth: The ground-truth distribution.
        max_knots: Upper bound on the number of knots returned.

    Returns:
        A tuple ``(positions, cdf)`` of equal length, with ``positions``
        strictly increasing and ``cdf[-1]`` equal to 1.
    """
    positions = np.asarray(ground_truth.dv.positions, dtype=float)
    masses = np.asarray(ground_truth.dv.masses, dtype=float)
    order = np.argsort(positions, kind="stable")
    positions = positions[order]
    cdf = np.cumsum(masses[order])
    cdf = cdf / cdf[-1]

    # The CDF value of an atom is the cumulative mass at the last of its
    # coincident positions. ``np.unique`` reports first occurrences, so look
    # them up on the reversed array.
    positions, reversed_index = np.unique(positions[::-1], return_index=True)
    cdf = cdf[len(cdf) - 1 - reversed_index]

    if len(positions) > max_knots:
        keep = np.unique(np.linspace(0, len(positions) - 1, max_knots).astype(int))
        positions = positions[keep]
        cdf = cdf[keep]
    cdf[-1] = 1.0

    return positions, cdf


def _simulate_brownian_bridges(
    times: np.ndarray, num_paths: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Simulate standard Brownian bridges on an increasing grid of times.

    Args:
        times: Increasing times in (0, 1]. The last time must be 1, where
            every bridge is pinned to 0.
        num_paths: Number of independent bridges.
        rng: The random number generator to draw the increments from.

    Returns:
        An array of shape ``(num_paths, len(times))`` of bridge values.
    """
    increments = np.diff(times, prepend=0.0)
    wiener = np.cumsum(
        rng.normal(size=(num_paths, len(times))) * np.sqrt(increments), axis=1
    )

    return wiener - np.outer(wiener[:, -1], times)


def _simulate_wasserstein_1_asymptotic_samples(
    positions: np.ndarray, cdf: np.ndarray, num_samples: int, seed: int
) -> np.ndarray:
    """
    Draw samples of ``∫ |B(F(x))| dx`` for the step CDF ``F`` given by knots.

    ``F`` is constant between knots, so the integral is exactly the sum of
    ``|B(F(x_j))| (x_{j+1} - x_j)`` over the knots. Paths are simulated in
    chunks to bound memory.

    Args:
        positions: Strictly increasing knot positions.
        cdf: CDF value at each knot. The last value must be 1.
        num_samples: Number of paths to simulate.
        seed: Seed of the random number generator.

    Returns:
        One integral value per simulated path.
    """
    widths = np.diff(positions)
    rng = np.random.default_rng(seed)
    samples = np.empty(num_samples)
    chunk = max(1, _BROWNIAN_BRIDGE_CHUNK_ELEMENTS // len(cdf))
    for start in range(0, num_samples, chunk):
        count = min(chunk, num_samples - start)
        bridges = _simulate_brownian_bridges(cdf, count, rng)
        samples[start : start + count] = np.abs(bridges[:, :-1]) @ widths

    return samples


def _compute_asymptotic_wasserstein_distribution_mean(
    ground_truth: TaggedDistributionalValue, num_points: int
) -> float:
    """
    Compute ``sqrt(2 / pi) * ∫ sqrt(F(x) (1 - F(x))) dx`` for the ground
    truth's step CDF ``F``, on the same knots the Brownian-bridge simulation
    uses. This is the exact mean of ``∫ |B(F(x))| dx``, since
    ``E|B(t)| = sqrt(2 t (1 - t) / pi)``. Divided by sqrt(N) it is the mean
    Wasserstein-1 distance between the ground truth and an N-sample Monte
    Carlo simulation of it.

    Args:
        ground_truth: The ground-truth distribution.
        num_points: Upper bound on the number of CDF knots used.

    Returns:
        The value of the integral.
    """
    positions, cdf = _ground_truth_cdf_knots(ground_truth, num_points)
    integrand = np.sqrt(cdf[:-1] * (1.0 - cdf[:-1]))

    return float(np.sqrt(2.0 / np.pi) * np.dot(integrand, np.diff(positions)))


def _wasserstein_1_clt_tail_diagnostics(
    ground_truth: TaggedDistributionalValue,
) -> tuple[float, float]:
    """
    Diagnose whether the ground truth meets the Wasserstein-1 CLT condition.

    Theorem 2.1 of del Barrio, Giné and Matrán (1999) states that
    ``sqrt(N) * W1(F_N, F)`` has the Brownian-bridge limit if and only if
    ``∫ sqrt(F (1 - F)) dx`` is finite. That holds when ``E|X|^(2 + δ)`` is
    finite for some ``δ > 0`` and fails when the variance is infinite. A
    finite sample always has a finite integral, so the condition is judged
    from the tails of the sample instead.

    Args:
        ground_truth: The ground-truth distribution.

    Returns:
        A tuple ``(tail_index, tail_share)``. ``tail_index`` is a
        mass-weighted Hill estimate of the exponent ``α`` in
        ``P(|X - median| > t) ~ t^-α``, from the outer
        ``CltAssumptions.HILL_MASS_FRACTION`` of mass. A value at or below 2
        means infinite variance. ``tail_share`` is the fraction of
        ``∫ sqrt(F (1 - F)) dx`` carried by the outer
        ``CltAssumptions.TAIL_SHARE_MASS_FRACTION`` of mass on each side. A
        large share means the limit is approached slowly.
    """
    positions, cdf = _ground_truth_cdf_knots(
        ground_truth, len(ground_truth.dv.positions)
    )
    masses = np.diff(cdf, prepend=0.0)

    # Hill estimate on the largest deviations from the median.
    median = positions[np.searchsorted(cdf, 0.5)]
    deviations = np.abs(positions - median)
    order = np.argsort(deviations)[::-1]
    deviations = deviations[order]
    tail_masses = masses[order]
    num_tail = int(
        np.searchsorted(np.cumsum(tail_masses), CltAssumptions.HILL_MASS_FRACTION)
    )
    num_tail = min(max(num_tail, CltAssumptions.HILL_MIN_POINTS), len(deviations) - 1)
    threshold = deviations[num_tail]
    top_mass = tail_masses[:num_tail].sum()
    if num_tail < 1 or threshold <= 0.0 or top_mass <= 0.0:
        tail_index = float("inf")
    else:
        log_excess = np.dot(
            tail_masses[:num_tail], np.log(deviations[:num_tail] / threshold)
        )
        tail_index = float("inf") if log_excess <= 0.0 else float(top_mass / log_excess)

    # Share of the CLT integral carried by the outer mass on each side.
    contributions = np.sqrt(cdf[:-1] * (1.0 - cdf[:-1])) * np.diff(positions)
    total = contributions.sum()
    fraction = CltAssumptions.TAIL_SHARE_MASS_FRACTION
    inner = (cdf[:-1] >= fraction) & (cdf[:-1] <= 1.0 - fraction)
    tail_share = (
        0.0 if total <= 0.0 else float(1.0 - contributions[inner].sum() / total)
    )

    return tail_index, tail_share


def _scalar_asymptotic_statistics(
    location: float, scale: float
) -> tuple[float, float, float, float]:
    """
    Statistics of the folded normal ``|N(location, scale)|``.

    For a scalar output the central limit theorem gives
    ``sqrt(N) * (scalar(N) - ground truth) ~ N(location, scale)``, so the
    distance is folded normal. Keeping ``location`` rather than assuming 0
    accounts for a biased estimator.

    Args:
        location: Mean of the underlying normal.
        scale: Standard deviation of the underlying normal.

    Returns:
        A tuple ``(mean, quantile_95, quantile_99, mean_quantile)`` where
        ``mean_quantile`` is the CDF value at the mean.
    """
    if scale <= 0.0:
        # A deterministic scalar has no spread, so every statistic is the
        # absolute bias.
        return abs(location), abs(location), abs(location), 1.0

    dist = foldnorm(c=abs(location) / scale, scale=scale)
    mean = float(dist.mean())

    return (
        mean,
        float(dist.ppf(ReportingNumbers.QUANTILE_95)),
        float(dist.ppf(ReportingNumbers.QUANTILE_99)),
        float(dist.cdf(mean)),
    )


def _generate_representative_mc_plots(
    emcc: "EquivalentMonteCarlo",
    distance_type: str,
    plots_dir: str = "",
    xlim: tuple[float, float] | None = None,
) -> None:
    """
    Generate MC plots from representative samples that approximate the UxHw
    distribution's distance to the ground truth.

    Builds up Monte Carlo samples progressively in chunks until the distance
    between the MC samples and the ground truth approximates the UxHw-to-ground
    truth distance:

    1. For each row in the results DataFrame, extract the target EMCC count.
    2. Compute the target distance (UxHw vs ground truth).
    3. Build MC samples progressively in chunks to match that target distance.
    4. Fall back to the best attempt if convergence fails within the attempt
       limit.

    Args:
        emcc: EquivalentMonteCarlo object holding the results DataFrame,
            adversary array, and ground truth.
        distance_type: Distance metric to use.
        plots_dir: Directory to write the plots into. Defaults to the current working directory when empty.
        xlim: Limits applied to the x-axis so the MC plots use the same domain
            as the ground-truth and UxHw plots. ``None`` falls back to per-plot
            auto-scaling.
    """
    # The name the report looks the plots up by.
    expr_string = emcc.variable.formatted_description

    # Maximum MC size we generate
    max_mc_size = 1_000_000

    count_list = []
    # Main loop over UxHw configurations
    df = pd.DataFrame(emcc.variable.emcc_results.emcc_data)
    for _, row in df.iterrows():
        try:
            # Skip if the EMCC column does not exist.
            if EquivMC.EMCC not in df.columns:
                continue

            # Get equivalent Monte Carlo count if it exists
            equiv_mc_count = row[EquivMC.EMCC]
            if pd.isna(equiv_mc_count) or not isinstance(equiv_mc_count, (int, float)):
                continue
            equiv_mc_count = int(equiv_mc_count)

            # Some UxHw results will have same equivMC counts so we can skip repeats
            if equiv_mc_count in count_list:
                continue
            # Set upper limit on MC size to avoid memory related issues
            if equiv_mc_count > max_mc_size:
                equiv_mc_count = max_mc_size
            count_list.append(equiv_mc_count)

            # The distance the EMCC was measured on.
            binned = row.get(BenchmarkingVariables.UXHW_BINNED_DISTANCE)
            target_distance = (
                binned
                if uses_binned_distance(binned, emcc.use_binned_uxhw)
                else row[BenchmarkingVariables.UXHW_DISTANCE]
            )
            selected_samples = _generate_progressive_samples(
                emcc, equiv_mc_count, target_distance, distance_type
            )
            adversary_dist = DistributionalValue.from_samples(selected_samples)
            adv_plot_path = f"{expr_string}-adversary-{equiv_mc_count}.png"
            if plots_dir:
                adv_plot_path = os.path.join(plots_dir, adv_plot_path)
            plot_wrapper.plot(
                plot_data=PlotData(adversary_dist),
                path=adv_plot_path,
                save=True,
                xlim=xlim,
            )
        except Exception as e:
            print(
                f"Failed to generate representative MC plot for {expr_string} with an equivalent MC count of {equiv_mc_count}: {e}"
            )
            continue


def _distance_func(
    distance_type: str,
    u_values: ArrayLike,
    v_values: ArrayLike,
    u_weights: ArrayLike | None = None,
    v_weights: ArrayLike | None = None,
) -> float:
    """
    Compute the distance between two weighted distributions.

    Only used for computing the equivalent Monte Carlo counts.

    Args:
        distance_type: Distance metric (Wasserstein-1 or Wasserstein-2).
        u_values: Support points for the first distribution.
        v_values: Support points for the second distribution.
        u_weights: Weights for the first distribution.
        v_weights: Weights for the second distribution.

    Returns:
        The distance between the two distributions.

    Raises:
        ValueError: If ``distance_type`` is not a supported metric.
    """

    if distance_type == DistanceMetrics.WASSERSTEIN_1:
        return wasserstein_p_distance(u_values, u_weights, v_values, v_weights, p=1)
    elif distance_type == DistanceMetrics.WASSERSTEIN_2:
        return wasserstein_p_distance(u_values, u_weights, v_values, v_weights, p=2)
    else:
        raise ValueError(f"Unknown distance type: {distance_type}")


def _generate_adversary_list(
    args: "LoadDataComputeEquivalentMCArgs", target_expr: str, expr_type: str
) -> list[TaggedDistributionalValue]:
    """
    Load the adversary distributions for Monte Carlo comparison.

    Distribution variables load a single distribution (wrapped in a list).
    Scalar variables load several directly.

    Args:
        args: Config holding the adversary database path and table name.
        target_expr: Variable name to load from the database.
        expr_type: Expression type (distribution or scalar).

    Returns:
        The adversary distributions loaded from the database.
    """
    adversary_list: list[TaggedDistributionalValue] = []
    if expr_type == VariableTypes.DISTRIBUTION:
        adversary_list.append(
            _load_mc(
                db_path=args["adversary_database_path"],
                table=args["adversary_table_name"],
                target_expression=target_expr,
            )
        )
    else:
        adversary_list = _load_mc_scalar(
            db_path=args["adversary_database_path"],
            table=args["adversary_table_name"],
            target_expression=target_expr,
        )
    return adversary_list


def _generate_comparison_plots(emcc: "EquivalentMonteCarlo", prefix: str) -> None:
    """
    Plot the distance distribution (scaled by sqrt(MC_count)) against the
    asymptotic distribution (e.g. from a Brownian-bridge simulation).

    Args:
        emcc: The Equivalent Monte Carlo object.
        prefix: Prefix for the figure name.
    """

    for mc_count in emcc.variable.emcc_results.equiv_mc_list:
        if mc_count > 0 and mc_count < emcc.adversary_size_max:
            if emcc.variable.type == VariableTypes.DISTRIBUTION:
                samples = emcc.variable.asymptotic_distribution.samples
                assert samples is not None, (
                    "asymptotic_distribution.samples must be populated "
                    "for distribution-typed variables before plotting"
                )
                emcc._plot_equiv_mc_vs_brownian_bridge(
                    mc_count,
                    samples,
                    prefix=prefix,
                )
            else:
                emcc._plot_equiv_mc_vs_folded_normal(mc_count, prefix=prefix)


def _plot_adversary_distances(emcc: "EquivalentMonteCarlo", prefix: str) -> None:
    """
    Plot the distance distribution for each adversary.

    Args:
        emcc: The Equivalent Monte Carlo object.
        prefix: Prefix for the figure name.
    """
    for i, (k, lst) in enumerate(emcc._adversary_distances.items()):
        plt.scatter(
            np.log(k),
            np.log(np.mean(lst)),
            marker="x",
            color="r",
            label="Measured Adversary Distances" if i == 0 else None,
        )

    plt.xlabel("Log of Adversary Size")
    plt.ylabel("Log of Mean Wasserstein Distance")
    plt.legend()
    # Tag the ground truth in the filename by its representation type (from the
    # carrier metadata, never off a Distribution). Only WeightedSamples counts
    # as "weighted". MonteCarlo / Samples are "unweighted".
    string = (
        "unweighted"
        if emcc.ground_truth.representation_type in ("Samples", "MonteCarlo")
        else "weighted"
    )
    plt.savefig(
        f"{prefix}-adversary_distances-{emcc.ground_truth.dv.UR_order}_{string}_ground_truth_samples-{emcc.n_adversaries}_adversaries.png",
        dpi=500,
    )
    plt.close()


def _generate_equivalent_mc_plots(
    emcc: "EquivalentMonteCarlo",
    args: "LoadDataComputeEquivalentMCArgs",
    target_expr: str,
    distance_type: str,
    xlim: tuple[float, float] | None = None,
) -> None:
    """
    Generate the configured equivalent Monte Carlo analysis plots.

    Args:
        emcc: The Equivalent Monte Carlo object.
        args: Config specifying which plots to generate.
        target_expr: Variable name used for plot file naming.
        distance_type: Distance metric used (e.g. Wasserstein-1).
        xlim: Limits applied to the x-axis for the representative MC plots, so they use
            the same domain as the ground-truth and UxHw plots. ``None`` falls
            back to per-plot auto-scaling.
    """
    plots_dir = args["plots_dir"]
    prefixed_expr = os.path.join(plots_dir, target_expr) if plots_dir else target_expr

    if args["plot_adversary_distances"]:
        _plot_adversary_distances(emcc=emcc, prefix=prefixed_expr)

    if args["plot_comparison_distributions"]:
        _generate_comparison_plots(emcc=emcc, prefix=prefixed_expr)
    # Plot equivalent MC run with "similar" distance to UxHw
    if args["plot_distributions"] and emcc.variable.type == VariableTypes.DISTRIBUTION:
        print("Generating MC plots...")
        _generate_representative_mc_plots(
            emcc, distance_type, plots_dir=plots_dir, xlim=xlim
        )


def _attempt_convergence(
    emcc: "EquivalentMonteCarlo",
    selected_samples: np.ndarray,
    sample_size_step: int,
    sample_size: int,
    target_constant: float,
    distance_type: str,
    threshold: float = 0.05,
) -> tuple[bool, np.ndarray, float]:
    """
    Draw one more chunk of samples and check convergence to the target distance.

    Args:
        emcc: The Equivalent Monte Carlo object.
        selected_samples: Samples accumulated so far (sorted).
        sample_size_step: Number of new samples to draw this attempt.
        sample_size: Total sample size the distance is scaled against.
        target_constant: Target value of distance * sqrt(sample_size).
        distance_type: Distance metric to use.
        threshold: Max proportional difference from the target to count as
            converged.

    Returns:
        A tuple of (converged, merged samples, proportional difference from
        the target).
    """
    new_samples = np.random.choice(emcc._adversary_array, size=sample_size_step)
    new_samples.sort()

    # Merge sort with previously selected samples in O(n) time
    samples = np.array(list(merge(selected_samples, new_samples)))

    mc_distance = _distance_func(
        distance_type,
        u_values=samples,
        v_values=emcc.ground_truth.dv.positions,
        v_weights=emcc.ground_truth.dv.masses,
    )

    # Check convergence using the target constant
    attempt_constant = mc_distance * np.sqrt(sample_size)
    proportion_diff = np.abs(attempt_constant - target_constant) / target_constant

    converged = proportion_diff < threshold
    return converged, samples, proportion_diff


def _generate_mc_samples_for_plotting(
    emcc: "EquivalentMonteCarlo",
    selected_samples: np.ndarray,
    sample_size: int,
    sample_size_step: int,
    target_constant: float,
    distance_type: str,
    equiv_mc_count: int,
) -> np.ndarray:
    """
    Generate samples for one target size, retrying until convergence.

    Repeatedly calls :func:`_attempt_convergence`, keeping the best attempt as
    a fallback if none converge within the attempt limit.

    Args:
        emcc: The Equivalent Monte Carlo object.
        selected_samples: Samples accumulated so far (sorted).
        sample_size: Total sample size the distance is scaled against.
        sample_size_step: Number of new samples to draw per attempt.
        target_constant: Target value of distance * sqrt(sample_size).
        distance_type: Distance metric to use.
        equiv_mc_count: Equivalent MC count, used to bound the attempt count.

    Returns:
        The converged samples, or the best-effort fallback samples.
    """
    max_attempts = max(5, min(1000, math.ceil(10_000_000 / equiv_mc_count)))
    num_attempts = 0
    smallest_diff = np.inf
    fallback_samples = np.array([])

    while num_attempts < max_attempts:
        num_attempts += 1

        converged, samples, proportion_diff = _attempt_convergence(
            emcc,
            selected_samples,
            sample_size_step,
            sample_size,
            target_constant,
            distance_type,
        )

        # Update fallback option with best result so far
        if proportion_diff < smallest_diff:
            smallest_diff = proportion_diff
            fallback_samples = samples.copy()

        if converged:
            return samples

    # Return fallback if no convergence
    return fallback_samples


def _generate_progressive_samples(
    emcc: "EquivalentMonteCarlo",
    equiv_mc_count: int,
    target_distance: float,
    distance_type: str,
) -> np.ndarray:
    """
    Build samples up to ``equiv_mc_count`` in chunks, converging each chunk.

    Args:
        emcc: The Equivalent Monte Carlo object.
        equiv_mc_count: Target total number of samples.
        target_distance: UxHw-to-ground-truth distance to match.
        distance_type: Distance metric to use.

    Returns:
        The accumulated samples.
    """
    target_constant = target_distance * np.sqrt(equiv_mc_count)
    sample_sizes, sample_size_steps = _calculate_sample_sizes(equiv_mc_count)

    selected_samples = np.array([])

    # Progressive sampling: build up samples in chunks
    for sample_size, sample_size_step in zip(sample_sizes, sample_size_steps):
        selected_samples = _generate_mc_samples_for_plotting(
            emcc,
            selected_samples,
            sample_size,
            sample_size_step,
            target_constant,
            distance_type,
            equiv_mc_count,
        )

    return selected_samples


def _calculate_sample_sizes(
    equiv_mc_count: int, sample_size_step: int = 100_000
) -> tuple[list[int], list[int]]:
    """
    Calculate cumulative sample sizes and step sizes for progressive sampling.

    Args:
        equiv_mc_count: Target total number of samples.
        sample_size_step: Chunk size for each progressive step.

    Returns:
        A tuple of (cumulative sample sizes, per-step increments).
    """
    if equiv_mc_count <= sample_size_step:
        sample_sizes = [equiv_mc_count]
    else:
        sample_sizes = list(
            range(sample_size_step, equiv_mc_count, sample_size_step)
        ) + [equiv_mc_count]

    sample_size_steps = [sample_sizes[0]] + [
        sample_sizes[i] - sample_sizes[i - 1] for i in range(1, len(sample_sizes))
    ]

    return sample_sizes, sample_size_steps


def _compute_asymptotic_distribution_scalar_empirical(
    samples: npt.ArrayLike, sample_size: int, ground_truth_value: float
) -> tuple[float, float, bool]:
    """
    Compute the asymptotic distribution of N samples, fitting a Gaussian to
    ``samples * sqrt(N)`` centred on the ground truth.

    Args:
        samples: Input samples.
        sample_size: Number of samples.
        ground_truth_value: Ground-truth value to centre samples around.

    Returns:
        A tuple of (mean, standard deviation, is_normal) for the centred and
        scaled samples, where ``is_normal`` is the Anderson-Darling verdict.
    """
    samples_array = (np.asarray(samples, dtype=float) - ground_truth_value) * np.sqrt(
        sample_size
    )

    mean = np.mean(samples_array)
    std = np.std(samples_array, ddof=1)

    is_normal = anderson_darling_test(samples_array)

    return mean.item(), std.item(), is_normal


def anderson_darling_test(samples: npt.ArrayLike) -> bool:
    """
    Anderson-Darling test for normality.

    Args:
        samples: Samples to test.

    Returns:
        ``True`` if the samples appear normal at the 15% significance level.
    """
    result = anderson(samples, dist="norm")

    if result.statistic < result.critical_values[0]:
        print("✓ Data appears normally distributed at 15% level")
        return True
    else:
        print("✗ Data does NOT appear normally distributed at 15% level")
        return False
