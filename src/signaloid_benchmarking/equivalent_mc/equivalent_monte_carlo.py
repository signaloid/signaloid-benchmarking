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

from numpy.typing import ArrayLike
from typing import Any
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from threading import Thread
from tabulate import tabulate
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from signaloid_benchmarking.equivalent_mc.adversary_distance import (
    _adversary_distance_loop,
)
from signaloid_benchmarking.config import (
    ReportingMethods,
    ReportingNumbers,
    VariableTypes,
    DistanceMetrics,
    EquivMC,
    BenchmarkingVariables,
    CltAssumptions,
    uses_binned_distance,
)
from signaloid_benchmarking.types import (
    BenchmarkingVariable,
    TaggedDistributionalValue,
)
from scipy.stats import foldnorm  # type: ignore


# Configure the computation using this class
class EquivalentMonteCarlo:
    """
    Compute the equivalent Monte Carlo count (EMCC) for a benchmarking variable.

    Measures how many Monte Carlo samples an adversary needs before its distance
    to the ground truth beats each UxHw configuration. Adversary distances are
    computed in parallel over a range of sample sizes (optionally adaptively
    stepped). The resulting EMCC values and the proportion of adversaries that
    beat each configuration are written back onto the variable's EMCC results
    and reported in a table.
    """

    def __init__(
        self,
        ground_truth: TaggedDistributionalValue,
        uxhw_data: list[TaggedDistributionalValue],
        variable: BenchmarkingVariable,
        adversary_mc: list[TaggedDistributionalValue],
        distance_type: str = DistanceMetrics.WASSERSTEIN_1,
        n_processes: int = 1,
        n_adversaries: int = 1,
        use_clt: bool = False,
        adversary_size_step: int = 100,
        adversary_size_min: int = 1,
        adversary_size_max: int | None = None,
        use_binned_uxhw: bool = True,
    ) -> None:
        """
        Initialise the equivalent Monte Carlo computation.

        Args:
            ground_truth: The ground-truth distribution to measure against.
            uxhw_data: The UxHw representations under test. Must be non-empty.
            variable: The benchmarking variable whose EMCC results are populated.
            adversary_mc: Adversary Monte Carlo samples. The pool is subsampled at
                each requested size.
            distance_type: Distance metric (Wasserstein-1, Binned Wasserstein-1,
                or Wasserstein-2).
            n_processes: Number of worker processes for the adversary-distance
                computation.
            n_adversaries: Number of adversary repetitions per sample size.
            use_clt: Use the predicted (CLT) EMCC instead of measuring it
                explicitly.
            adversary_size_step: Fixed step between adversary sizes in the
                non-adaptive mode.
            adversary_size_min: Smallest adversary sample size.
            adversary_size_max: Largest adversary sample size. Defaults to
                ``len(adversary array) - 1``.
            use_binned_uxhw: Use the binned Wasserstein-1 UxHw distance.

        Raises:
            ValueError: If ``uxhw_data`` is empty.
            RuntimeError: If ``distance_type`` is not a supported metric.
        """
        self.ground_truth: TaggedDistributionalValue = ground_truth
        self.adversary_mc: list[TaggedDistributionalValue] = adversary_mc
        if len(uxhw_data) == 0:
            raise ValueError("uxhw_data must contain at least one entry.")
        self.variable = variable
        self.use_binned_uxhw = use_binned_uxhw

        # Order of the Wasserstein distance the adversaries are scored with.
        self.wasserstein_order: int
        if distance_type in [
            DistanceMetrics.WASSERSTEIN_1,
            DistanceMetrics.BINNED_WASSERSTEIN_1,
        ]:
            self.wasserstein_order = 1
        elif distance_type == DistanceMetrics.WASSERSTEIN_2:
            self.wasserstein_order = 2
        else:
            raise RuntimeError(
                f"Invalid distance type configuration "
                f"{distance_type}. Check py for valid options"
            )

        self.n_processes = n_processes
        self.n_adversaries = n_adversaries
        self.use_clt = use_clt

        # Maps an adversary sample count to the distances that different
        # subsamplings of that size achieved against the ground truth. For
        # example, `self._adversary_distances[420] == [1, 0.6, 1.2]` means three
        # subsamplings of 420 samples scored distances 1, 0.6, and 1.2.
        self._adversary_distances: dict[int, list[float]] = dict()

        # Create single large adversary array for distributional variables
        self._adversary_array: np.ndarray = np.array(adversary_mc[0].dv.positions)

        self.adversary_size_step = adversary_size_step
        self.adversary_size_min = adversary_size_min
        self.adversary_size_max = (
            adversary_size_max
            if adversary_size_max is not None
            else len(self._adversary_array) - 1
        )

    def compute_distance_data(self, use_adaptive_steps: bool = True) -> None:
        """
        Compute adversary distances across the generated sample-size array.

        Args:
            use_adaptive_steps: Grow the adversary-size step for larger sizes.
        """
        # Compute the adversary size array an step size array
        self.adversary_size_array = self._generate_adversary_size_array(
            use_adaptive_steps
        )
        # Compute the adversary distances
        self._compute_adversary_distances()

    def _generate_adversary_size_array(
        self, use_adaptive_steps: bool = True
    ) -> np.ndarray:
        """
        Generate the array of Monte Carlo sample sizes for distance measurements.

        There are two ways to compute the equivalent MC count:
            1. Estimate the values first, then use EquivalentMonteCarlo to verify.
            2. Compute distances across an array of candidate sizes and read the
               count off those measurements.

        Args:
            use_adaptive_steps: Grow the adversary-size step for larger sizes.

        Returns:
            The array of adversarial MC sizes.
        """
        # Only the distribution branch of `_compute_adversary_distances` uses
        # this array. The scalar branch iterates `self.adversary_mc` directly.
        # For scalars `adversary_size_max` can be 0, which would make the
        # adaptive-step `np.log(adversary_size_max)` below hit `log(0)`, so
        # short-circuit here.
        if self.variable.type != VariableTypes.DISTRIBUTION:
            return np.array([])

        predicted_size_array = np.array([])
        if len(self.variable.emcc_results.equiv_mc_list) > 0:
            predicted_size_array = np.array(self.variable.emcc_results.equiv_mc_list)
            predicted_max_value: int = int(np.max(predicted_size_array))

            # If using predictions then we can reduce the maximum adversary size
            # to the largest prediction (if smaller)
            if self.use_clt:
                self.adversary_size_max = min(
                    self.adversary_size_max, predicted_max_value
                )

            if self.variable.type == VariableTypes.DISTRIBUTION:
                if len(self._adversary_array) < predicted_max_value:
                    print(
                        f"Warning! Predicted maximum adversary size {predicted_max_value} greater than size of adversary array {len(self._adversary_array)}!"
                    )

            # Only consider values below the max
            predicted_size_array = predicted_size_array[
                predicted_size_array < self.adversary_size_max
            ]

        # Return array of sizes
        if self.use_clt:
            return predicted_size_array

        else:
            # Compute EMCC explicitly
            if use_adaptive_steps:
                # Space the adversary sizes so each step is roughly a constant
                # fraction (alpha) of the MC count, keeping precision consistent
                # while minimizing computational effort.
                alpha = 0.01
                num_steps = math.ceil(
                    -np.log(self.adversary_size_max) / np.log(1 - alpha)
                )
                array = np.geomspace(
                    self.adversary_size_min, self.adversary_size_max, num=num_steps
                )
                size_array = np.unique(np.ceil(array).astype(int))
            else:
                size_array = np.arange(
                    self.adversary_size_min,
                    self.adversary_size_max,
                    self.adversary_size_step,
                )

            return np.asarray(size_array)

    def _compute_adversary_distances(self) -> None:
        print(
            "Computing adversary choices distances with "
            + f"n_processes={self.n_processes} and "
            + f"n_adversaries={self.n_adversaries}"
        )

        if self.variable.type == VariableTypes.DISTRIBUTION:
            total_iterations = len(self.adversary_size_array) * self.n_adversaries
            results = []
            # Extract only needed attributes once
            gt_positions = self.ground_truth.dv.positions
            gt_masses = self.ground_truth.dv.masses

            with Manager() as manager:
                progress_queue = manager.Queue()

                with tqdm(
                    total=total_iterations, desc="Processing", unit="step"
                ) as progress_bar:
                    progress_updater = Thread(
                        target=_update_progress, args=(progress_queue, progress_bar)
                    )
                    progress_updater.start()

                    with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
                        futures = [
                            executor.submit(
                                _adversary_distance_loop,
                                self.adversary_size_array,
                                self._adversary_array,
                                gt_positions,
                                progress_queue,
                                gt_masses,
                                self.wasserstein_order,
                            )
                            for _ in range(self.n_adversaries)
                        ]
                        for future in futures:
                            result = future.result()
                            results.append(result)
                    progress_queue.put(None)
                    progress_updater.join()
        else:
            results = []
            # adversary.mc_count is the per-adversary scalar MC sample count,
            # read off the carrier metadata (never off a Distribution).
            for adversary in self.adversary_mc:
                if adversary.mc_count is not None:
                    # Scalar distance in basis points: relative error from the
                    # ground truth scaled by 10,000 (BASIS_POINT_CONVERSION_FACTOR,
                    # since 1 bp = 0.01%).
                    distance_array = (
                        np.abs(
                            adversary.dv.positions - self.ground_truth.dv.positions[0]
                        )
                        * EquivMC.BASIS_POINT_CONVERSION_FACTOR
                        / np.abs(self.ground_truth.dv.positions[0])
                    )
                    distance_list = distance_array.tolist()
                    results.append([(adversary.mc_count, distance_list)])

        for result in results:
            for adversary_size, distance_lst in result:
                self._adversary_distances.setdefault(adversary_size, []).extend(
                    distance_lst
                )

        if self.variable.type == VariableTypes.DISTRIBUTION:
            self._check_sqrt_n_scaling()

    def _check_sqrt_n_scaling(self) -> None:
        """
        Check that sqrt(N) times the mean adversary distance is constant.

        Under the central limit theorem behind the EMCC prediction it is. The
        product is compared between the smallest measured size at or above
        ``CltAssumptions.SCALING_MIN_SIZE`` and the largest one at or below a
        tenth of the ground truth's support, above which the ground truth's
        own sampling error inflates the distances. Nothing is checked unless
        the two sizes are at least a factor four apart. The ratio is stored on
        the variable's asymptotic distribution and reported when it is further
        than ``CltAssumptions.SCALING_RATIO_TOLERANCE`` from 1.
        """
        largest = len(self.ground_truth.dv.positions) // 10
        sizes = sorted(
            size
            for size in self._adversary_distances
            if CltAssumptions.SCALING_MIN_SIZE <= size <= largest
        )
        if len(sizes) < 2 or sizes[-1] < 4 * sizes[0]:
            return

        def scaled_mean(size: int) -> float:
            return float(np.sqrt(size) * np.mean(self._adversary_distances[size]))

        smallest = scaled_mean(sizes[0])
        largest_mean = scaled_mean(sizes[-1])
        if smallest == 0.0:
            # A deterministic ground truth gives every adversary a distance of
            # 0, so the scaling is trivially constant. A positive distance at
            # the large size only cannot follow sqrt(N) at all.
            ratio = 1.0 if largest_mean == 0.0 else float("inf")
        else:
            ratio = largest_mean / smallest
        self.variable.asymptotic_distribution.scaling_ratio = ratio
        if abs(ratio - 1.0) > CltAssumptions.SCALING_RATIO_TOLERANCE:
            print(
                f"Warning: sqrt(N) * mean adversary distance changed by a factor "
                f"{ratio:.2f} between N={sizes[0]} and N={sizes[-1]} for "
                f"{self.variable.name}. The sqrt(N) scaling assumed by the EMCC "
                "prediction does not hold at these sizes."
            )

    def _resolve_adversary_distances(self, mc_count: int) -> list[float] | None:
        """
        Return the adversary distance list for ``mc_count``.

        The adversary-distance computation can use adaptive stepping, so
        arbitrary integers (for example predicted-EMCC values that were not
        sampled) are not guaranteed to be keys in ``self._adversary_distances``.
        When ``mc_count`` is absent, fall back to the closest sampled key.

        Args:
            mc_count: Monte Carlo count to look up.

        Returns:
            The list of distances at ``mc_count`` (exact match) or at
            the closest sampled MC count. ``None`` only when no
            adversary distances have been computed yet.
        """
        if not self._adversary_distances:
            return None
        if mc_count in self._adversary_distances:
            return self._adversary_distances[mc_count]
        closest = min(
            self._adversary_distances.keys(),
            key=lambda k: abs(k - mc_count),
        )
        return self._adversary_distances[closest]

    def compute_and_report_emmc(self) -> None:
        """
        Determine the EMCC for each UxHw configuration and print the report.
        """
        self.determine_equiv_mc_counts()
        self._report_emmc()

    def determine_equiv_mc_counts(self) -> None:
        """
        Compute the EMCC for each UxHw configuration and store it on the
        variable's EMCC data.

        For each configuration, finds the smallest adversary MC count whose
        distance to the ground truth beats the UxHw distance (or uses the
        predicted count under ``use_clt``), and records the proportion of
        adversaries that beat the configuration.
        """
        # Loop through UxHw configurations
        for emcc_dic in self.variable.emcc_results.emcc_data:
            binned = emcc_dic.get(BenchmarkingVariables.UXHW_BINNED_DISTANCE)
            distance = (
                binned
                if uses_binned_distance(binned, self.use_binned_uxhw)
                else emcc_dic[BenchmarkingVariables.UXHW_DISTANCE]
            )

            if self.use_clt:
                mc_count = emcc_dic[EquivMC.EMCC_PREDICTED]

            else:
                # Find all MC adversaries that beat UxHw configuraion
                mc_beats_uxhw = [
                    (k, lst)
                    for k, lst in self._adversary_distances.items()
                    if (
                        _comparison_statistic(lst, emcc_dic[EquivMC.REPORTING_METHOD])
                        < distance
                    )
                ]

                # We want to find the smallest possible Monte Carlo count such that MC beats UxHw
                mc_beats_uxhw.sort(key=lambda x: x[0])

                if not mc_beats_uxhw:
                    # This means there was no Adversary whose average distance was less than UxHw.
                    mc_count = 1
                    print(
                        f"Warning: Adversary Monte Carlo data not enough for {emcc_dic[BenchmarkingVariables.UXHW_CONF]}"
                    )
                else:
                    mc_count = min(mc_beats_uxhw, key=lambda x: x[0])[0]

                # If mc_count is non-positive then set to 1
                mc_count = max(1, mc_count)

            emcc_dic[EquivMC.EMCC] = mc_count

            distances = self._resolve_adversary_distances(mc_count)
            if distances is not None:
                distance_samples = np.array(distances)
                proportion_mc_beats_uxhw = float(
                    np.sum(distance_samples < distance)
                ) / float(len(distance_samples))
                emcc_dic[EquivMC.PERCENTAGE_MC_BEATS_UXHW] = proportion_mc_beats_uxhw

    def _report_emmc(self) -> None:
        """
        Print the variable's EMCC results as a table.
        """
        print("Report: Equivalent Monte Carlo Count")
        print(f"expression={self.variable.name}")
        df = pd.DataFrame(self.variable.emcc_results.emcc_data)

        # Use the column name directly, not the index
        if BenchmarkingVariables.UXHW_CONF in df.columns:
            df[BenchmarkingVariables.UXHW_CONF] = df[
                BenchmarkingVariables.UXHW_CONF
            ].apply(repr)

        print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))  # type: ignore

    def _draw_scaled_adversary_distances(self, data: list[float], mean: float) -> None:
        """
        Draw the predicted asymptotic mean and the histogram of the scaled
        adversary distances.

        Args:
            data: Adversary distances scaled by ``sqrt(mc_count)``.
            mean: Predicted mean of the asymptotic distribution.
        """
        plt.axvline(
            x=mean,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Predicted Asymptotic Mean",
        )
        plt.hist(
            data,
            bins=max(1, int(1 + math.log2(self.n_adversaries))),
            density=True,
            color="b",
            alpha=0.5,
            label="Adversarial MC",
        )

    def _save_adversary_plot(self, mc_count: int, prefix: str) -> None:
        """
        Title and label the current figure, save it and close it.

        Args:
            mc_count: Adversary MC count the figure shows.
            prefix: Prefix for the saved figure name.
        """
        plt.title(
            f"{self.ground_truth.dv.UR_order} Ground Truth, {mc_count} MC Count for {self.n_adversaries} Adversaries"
        )
        plt.xlabel(r"Wasserstein Distance $\times\ \sqrt{\text{MC Count}}$")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.savefig(
            f"{prefix}-{self.ground_truth.dv.UR_order}_ground_truth-{mc_count}_mc_count-{self.n_adversaries}-adversaries.png",
            dpi=500,
        )
        plt.close()

    def _plot_equiv_mc_vs_brownian_bridge(
        self, mc_count: int, brownian_bridge_integrals: ArrayLike, prefix: str
    ) -> None:
        """
        Plot the scaled adversary distances at ``mc_count`` against the
        Brownian-bridge asymptotic prediction, and save the figure.

        Args:
            mc_count: Adversary MC count whose distances are plotted.
            brownian_bridge_integrals: Simulated Brownian-bridge integrals for
                the asymptotic reference histogram.
            prefix: Prefix for the saved figure name.
        """
        distances = self._resolve_adversary_distances(mc_count)
        if distances is None:
            return
        data = [dist * np.sqrt(mc_count) for dist in distances]
        mean = self.variable.asymptotic_distribution.mean
        assert (
            mean is not None
        ), "asymptotic_distribution.mean must be populated before plotting"
        self._draw_scaled_adversary_distances(data, mean)
        plt.hist(
            brownian_bridge_integrals,
            bins=max(1, int(1 + math.log2(len(np.asarray(brownian_bridge_integrals))))),
            density=True,
            alpha=0.5,
            color="r",
            label="Brownian Bridge Simulation",
        )
        self._save_adversary_plot(mc_count, prefix)

    def _plot_equiv_mc_vs_folded_normal(self, mc_count: int, prefix: str) -> None:
        """
        Plot the scaled adversary distances at ``mc_count`` against the
        folded-normal asymptotic prediction (for scalar variables), and save
        the figure.

        Args:
            mc_count: Adversary MC count whose distances are plotted.
            prefix: Prefix for the saved figure name.
        """
        distances = self._resolve_adversary_distances(mc_count)
        if distances is None:
            return
        data = [dist * np.sqrt(mc_count) for dist in distances]
        asymptotic = self.variable.asymptotic_distribution
        assert (
            asymptotic.mean is not None
            and asymptotic.scale is not None
            and asymptotic.location is not None
        ), "asymptotic_distribution.mean, .scale and .location must be populated before plotting"
        self._draw_scaled_adversary_distances(data, asymptotic.mean)
        # A deterministic scalar has no spread, so its asymptotic distribution
        # is the point mass already marked by the mean line above.
        if asymptotic.scale > 0.0:
            x = np.linspace(0, max(data), 100)
            folded_normal_values = foldnorm.pdf(
                x,
                c=abs(asymptotic.location) / asymptotic.scale,
                scale=asymptotic.scale,
            )
            plt.plot(
                x, folded_normal_values, color="red", label="Asymptotic Distribution"
            )
        self._save_adversary_plot(mc_count, prefix)


def _comparison_statistic(lst: list[float] | np.ndarray, method: str) -> float:
    """
    Reduce a list of distances to a single value by the reporting method.

    Args:
        lst: Distances to reduce.
        method: One of ``ReportingMethods.MEAN``, ``QUANTILE_95``, or
            ``QUANTILE_99``.

    Returns:
        The mean or requested quantile of ``lst``.

    Raises:
        ValueError: If ``method`` is not a supported reporting method.
    """
    methods = {
        ReportingMethods.MEAN: np.mean,
        ReportingMethods.QUANTILE_95: lambda x: np.quantile(
            x, ReportingNumbers.QUANTILE_95
        ),
        ReportingMethods.QUANTILE_99: lambda x: np.quantile(
            x, ReportingNumbers.QUANTILE_99
        ),
    }

    arr = np.asarray(lst)

    if method not in methods:
        raise ValueError(f"Reporting statistic '{method}' is not supported.")

    return methods[method](arr)  # type: ignore


def _update_progress(progress_queue: Any, progress_bar: Any) -> None:
    """
    Advance ``progress_bar`` by counts read from ``progress_queue`` until a
    ``None`` sentinel is received.

    Args:
        progress_queue: Queue yielding step counts, then ``None`` to stop.
        progress_bar: The tqdm progress bar to advance.
    """
    while True:
        progress = progress_queue.get()
        if progress is None:
            break
        progress_bar.update(progress)
