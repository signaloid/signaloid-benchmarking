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

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from signaloid.distributional.distributional import DistributionalValue

from signaloid_benchmarking.config import (
    CltAssumptions,
    ReportingMethods,
    RepresentationTypes,
    VariableTypes,
    configuration_string,
)

# Only BenchmarkingVariable is part of the public API (it is the element type
# of LoadDataComputeEquivalentMCArgs["benchmarking_variables"] that callers
# construct). The remaining value/result types are populated on / read off a
# BenchmarkingVariable rather than constructed by callers, so they stay
# importable by explicit path but out of the advertised surface.
__all__ = [
    "BenchmarkingVariable",
]


@dataclass
class TaggedDistributionalValue:
    """
    A :class:`DistributionalValue` paired with the loader metadata the
    benchmarking surface needs.

    The metadata fields are populated directly from the SQLite columns by the
    ``equivalent_mc/load.py`` loaders, so nothing here reads them off a
    ``Distribution`` instance. ``dv`` is always a base
    :class:`DistributionalValue` (never the analyses-side ``Distribution``
    subclass), keeping this surface free of that dependency.

    Attributes:
        dv: The numeric distribution (positions, masses, mean, ...).
        representation_type: Uncertain-representation type string (e.g.
            ``"Athens"``, ``"MonteCarlo"``, ``"WeightedSamples"``).
        representation_size: Build size N for the representation. For Athens this
            is ``UR_Order``. Otherwise ``UR_Order_CoreLibrary``.
        correlation_tracking: Correlation-tracking status string from the DB
            (e.g. ``"Disabled"``, ``"Autocorrelation"``). ``None`` for
            ground-truth and adversary loads that carry no correlation metadata.
        mc_count: Monte Carlo sample count for scalar loads, ``None``
            otherwise.
    """

    dv: DistributionalValue
    representation_type: str | None = None
    representation_size: int | None = None
    correlation_tracking: str | None = None
    mc_count: int | None = None

    def __repr__(self) -> str:
        """
        Reproduce ``Distribution.__repr__`` so the config string (the
        ``UXHW_CONF`` key written to the EMCC CSV and used to join timing data)
        stays byte-identical.
        """
        return configuration_string(
            representation_type=str(self.representation_type),
            representation_size=self.representation_size,  # type: ignore[arg-type]
            correlation=str(self.correlation_tracking),
        )


@dataclass
class DistributionSamples:
    """
    Per-variable distribution sample positions/weights and scalar Monte Carlo
    outputs. Owned by ``BenchmarkingVariable.distribution_samples``.
    """

    values: list[float] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    scalar_output_dict: dict[int, list[float]] = field(default_factory=dict)

    def set_values(self, values: list[float]) -> None:
        """
        Set distribution sample positions. Weights unchanged.

        Args:
            values: List of sample position values.
        """
        self.values = values

    def set_weighted_values(
        self,
        values: list[float],
        weights: list[float],
    ) -> None:
        """
        Set distribution sample positions and weights.

        Args:
            values: List of sample position values.
            weights: List of corresponding sample weights.
        """
        self.values = values
        self.weights = weights

    def empty_values(self) -> None:
        """
        Clear stored sample data (positions, weights, scalar outputs).
        """
        self.values = []
        self.weights = []
        self.scalar_output_dict = {}


@dataclass
class TimingMeasurements:
    """
    Per-variable timing measurements collected during the UxHw/MC/native
    pipeline phases. Owned by ``BenchmarkingVariable.timing_measurements``.
    """

    measurement_dict: dict[str, dict[str, float | None]] = field(default_factory=dict)

    def append(
        self,
        *,
        config: str,
        time: float,
        e2e_time: float,
        pin_dyn_inst_count: float | None = None,
        db_time: float = 0.0,
        db_dyn_inst_count: float | None = None,
    ) -> None:
        """
        Record a timing measurement for a given configuration.

        Args:
            config: Configuration key (e.g. representation type string).
            time: In-application elapsed time in seconds.
            e2e_time: End-to-end elapsed time in seconds.
            pin_dyn_inst_count: Dynamic instruction count from PIN, or
                ``None`` when the run was made without an Intel PIN kit.
                Intel PIN is optional and off by default.
            db_time: Database-access time in seconds.
            db_dyn_inst_count: Emulated dynamic instruction count from the
                UxHw statistics database, or ``None``. Native builds record
                none, so it is ``None`` on every current run.
        """
        dictionary: dict[str, float | None] = {}
        dictionary["In Application Time"] = time
        dictionary["Database Time"] = db_time
        dictionary["End-to-End Time"] = e2e_time
        dictionary["Database Dyn. Inst. Count"] = db_dyn_inst_count
        dictionary["PIN Dyn. Inst. Count"] = pin_dyn_inst_count

        self.measurement_dict[config] = dictionary


@dataclass
class EmccResults:
    """
    Per-variable EMCC (Equivalent Monte Carlo Count) analysis state. Owned by
    ``BenchmarkingVariable.emcc_results``.
    """

    equiv_mc_list: list[int] = field(default_factory=list)
    emcc_data: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class UxhwDistanceRecord:
    """
    One UxHw configuration vs. ground-truth Wasserstein distance.

    ``uxhw_conf`` is a union: the producer pipeline
    (``analysis.compute_uxhw_distances``) stores a
    ``TaggedDistributionalValue``, while the CSV-round-tripped loader path
    (``measurement_loader.load_uxhw_distances``) stores its ``repr()`` string.
    Consumers handle both shapes.

    ``blow_up_reason`` marks a degraded (blown-up) representation: when not
    ``None`` the distances are set to ``inf`` and the config is reported as
    "blow-up / excluded" rather than dropped, so it stays in the full matrix
    (see ``BenchmarkingVariables.BLOW_UP_REASON``). In-memory only, not
    persisted to the CSVs.
    """

    uxhw_conf: "TaggedDistributionalValue | str"
    uxhw_distance: float
    uxhw_binned_distance: float | None = None
    blow_up_reason: str | None = None


@dataclass
class UxhwDistances:
    """
    Per-variable UxHw-distance records. Owned by
    ``BenchmarkingVariable.uxhw_distances``.
    """

    records: list[UxhwDistanceRecord] = field(default_factory=list)


@dataclass
class AsymptoticDistribution:
    """
    Per-variable asymptotic distribution data. Owned by
    ``BenchmarkingVariable.asymptotic_distribution``.

    Field shape varies by variable type:
    - Distribution-typed variables populate ``mean``,
      ``quantile_95``, ``quantile_99``, ``samples``, ``tail_index``,
      ``tail_share`` and, after the adversary runs, ``scaling_ratio``.
    - Scalar-typed variables populate ``mean``, ``quantile_95``,
      ``quantile_99``, ``is_normal``, ``scale`` and ``location``.

    All fields default to ``None`` so a freshly-constructed instance
    can be filled in incrementally by the generator
    (``Benchmark.generate_asymptotic_distance_distributions``) or
    the loader (``measurement_loader.load_asymptotic_dist``).
    """

    mean: float | None = None
    quantile_95: float | None = None
    quantile_99: float | None = None
    # CDF value at ``mean``. Serialized to the asymptotic-distance
    # CSV and round-tripped back by ``load_asymptotic_dist``, but no
    # consumer reads this field at runtime.
    mean_quantile: float | None = None
    # Distribution-typed variables only.
    samples: np.ndarray | None = None
    # Scalar-typed variables only. ``location`` and ``scale`` are the fitted
    # mean and standard deviation of the centred, scaled scalar errors.
    is_normal: bool | None = None
    scale: float | None = None
    location: float | None = None
    # Distribution-typed variables only. Tail diagnostics of the Wasserstein-1
    # CLT condition, see ``_wasserstein_1_clt_tail_diagnostics``.
    tail_index: float | None = None
    tail_share: float | None = None
    # Distribution-typed variables only. Ratio of sqrt(N) times the mean
    # adversary distance at the largest and the smallest measured sizes. Set
    # by the equivalent Monte Carlo stage and kept in memory only.
    scaling_ratio: float | None = None

    def value_for(self, reporting_method: str) -> float | None:
        """
        Look up the value associated with a ReportingMethods string.

        Supports the dynamic, reporting-method-keyed access pattern in
        ``compute_emcc_predictions``.

        Args:
            reporting_method: One of ``ReportingMethods.MEAN``,
                ``ReportingMethods.QUANTILE_95``, or
                ``ReportingMethods.QUANTILE_99``.

        Returns:
            The stored float for the requested method, or ``None`` if
            the field has not been populated yet.

        Raises:
            KeyError: If ``reporting_method`` is not a known method, so a
                typo does not silently return ``None``.
        """
        try:
            return {
                ReportingMethods.MEAN: self.mean,
                ReportingMethods.QUANTILE_95: self.quantile_95,
                ReportingMethods.QUANTILE_99: self.quantile_99,
            }[reporting_method]
        except KeyError as exc:
            raise KeyError(
                f"Unknown reporting method {reporting_method!r}; "
                f"expected one of {{Mean, Quantile-95, Quantile-99}}."
            ) from exc


class BenchmarkingVariable:
    """
    Holds all data for a single benchmarked variable.
    """

    def __init__(
        self,
        name: str,
        description: str,
        value_id: str = "",
        program: str = "main",
        path: str = "",
        line_number: str = "",
        file_name: str = "main.c",
        type: str = VariableTypes.DISTRIBUTION,
        cla: str = "",
        representation_types: list[str] | None = None,
        representation_sizes: list[int] | None = None,
        correlations: list[str] | None = None,
        analytic_ground_truth: str | None = None,
    ):
        """
        Initialise a BenchmarkingVariable.

        The last four arguments are this variable's own entries from
        signaloid.yaml. ``None`` means the variable declared none, so the
        run-wide value applies to it.

        Args:
            name: The name of the variable as traced in the application.
            description: Human-readable description of the variable.
            value_id: Unique identifier for the traced value.
            program: Name of the sub-program owning the variable.
            path: Filesystem path context for the variable.
            line_number: Source line number at which the variable is traced.
            file_name: Source file name in which the variable is declared.
            type: Output type. One of VariableTypes constants.
            cla: Command-line arguments used to isolate this variable.
            representation_types: Representation types to measure this
                variable on, in place of ``-u``.
            representation_sizes: Representation sizes to measure this
                variable on, in place of ``-s``.
            correlations: Correlation tracking modes to measure this
                variable on, in place of ``-c``.
            analytic_ground_truth: Absolute path of the script that writes
                this variable's exact reference, in place of
                ``--path-to-ground-truth-file``.
        """
        self.value_id = value_id
        self.name = name
        self.description = description
        self.program = program
        self.path = path
        self.file_name = file_name
        self.type = type
        self.line_number = line_number
        self.cla = cla
        self.representation_types = representation_types
        self.representation_sizes = representation_sizes
        self.correlations = correlations
        self.analytic_ground_truth = analytic_ground_truth
        # Configurations of this variable's matrix that cannot run, each
        # with its reason, keyed as configuration_string names them.
        # Decided by the Benchmark once the SDK is known, and empty until.
        self.skipped_configurations: dict[str, str] = {}
        self.command_line_arguments: str = ""
        self.timing_measurements: TimingMeasurements = TimingMeasurements()
        self.distribution_samples: DistributionSamples = DistributionSamples()
        self.emcc_results: EmccResults = EmccResults()
        self.asymptotic_distribution: AsymptoticDistribution = AsymptoticDistribution()
        # The description names this variable's result files. Every run of
        # characters other than letters, digits, underscores and dots becomes
        # one hyphen, so a description such as "Margin (qcrit/qlocal)" cannot
        # add a directory or a shell metacharacter to a path.
        self.formatted_description: str = re.sub(
            r"[^\w.]+", "-", self.description.lower()
        ).strip("-.")
        self.uxhw_distances: UxhwDistances = UxhwDistances()

    def representation_matrix(
        self,
        *,
        representation_types: list[str],
        representation_sizes: list[int],
        correlations: list[str],
    ) -> tuple[list[str], list[int], list[str]]:
        """
        Return the representations this variable is measured on.

        Args:
            representation_types: The run-wide types, from ``-u``.
            representation_sizes: The run-wide sizes, from ``-s``.
            correlations: The run-wide correlation modes, from ``-c``.

        Returns:
            The types, sizes and correlation modes, each taken from this
            variable's own signaloid.yaml entry when it declared one and
            from the run-wide argument otherwise.
        """
        return (
            (
                representation_types
                if self.representation_types is None
                else self.representation_types
            ),
            (
                representation_sizes
                if self.representation_sizes is None
                else self.representation_sizes
            ),
            correlations if self.correlations is None else self.correlations,
        )

    def benchmarked_configurations(
        self,
        *,
        representation_types: list[str],
        representation_sizes: list[int],
        correlations: list[str],
    ) -> list[str]:
        """
        Name every configuration this variable is measured on.

        Args:
            representation_types: The run-wide types, from ``-u``.
            representation_sizes: The run-wide sizes, from ``-s``.
            correlations: The run-wide correlation modes, from ``-c``.

        Returns:
            Each configuration of this variable's matrix that is not
            skipped, named as ``configuration_string`` names it.
        """
        types, sizes, modes = self.representation_matrix(
            representation_types=representation_types,
            representation_sizes=representation_sizes,
            correlations=correlations,
        )

        configurations = []
        for representation_type in types:
            for representation_size in sizes:
                for correlation in modes:
                    configuration = configuration_string(
                        representation_type=representation_type,
                        representation_size=representation_size,
                        correlation=correlation,
                    )
                    if configuration not in self.skipped_configurations:
                        configurations.append(configuration)

        return configurations

    def ground_truth_table(self, ground_truth_type: str) -> str:
        """
        Return the ground-truth table that holds this variable's reference.

        Args:
            ground_truth_type: The run-wide ground-truth type, from
                ``--ground-truth-type``.

        Returns:
            ``WeightedSamples`` for a variable with its own analytic
            reference, since an analytic script writes nothing else, and
            the run-wide type otherwise.
        """
        if self.analytic_ground_truth is not None:
            return RepresentationTypes.WEIGHTED_SAMPLES

        return ground_truth_type

    def clt_warnings(self) -> list[str]:
        """
        Warnings about the central limit theorem behind this variable's EMCC
        prediction. The thresholds are in ``CltAssumptions``.

        Returns:
            One message per failed check. Empty when every check passed or
            when no check has run yet.
        """
        asymptotic = self.asymptotic_distribution
        warnings: list[str] = []
        if self.type == VariableTypes.SCALAR:
            if asymptotic.is_normal is not None and not asymptotic.is_normal:
                warnings.append(
                    "The scaled scalar errors failed the Anderson-Darling "
                    "normality test, so the folded-normal prediction may be off."
                )
            return warnings

        if (
            asymptotic.tail_index is not None
            and asymptotic.tail_index <= CltAssumptions.TAIL_INDEX_LIMIT
        ):
            warnings.append(
                f"Tail index {asymptotic.tail_index:.2f} is at or below "
                f"{CltAssumptions.TAIL_INDEX_LIMIT}, so the variance is infinite "
                "and the Wasserstein-1 CLT condition of del Barrio, Giné and "
                "Matrán (1999) fails. The predicted EMCC is unreliable."
            )
        if (
            asymptotic.tail_share is not None
            and asymptotic.tail_share > CltAssumptions.TAIL_SHARE_LIMIT
        ):
            warnings.append(
                f"{asymptotic.tail_share:.0%} of the CLT integral comes from the "
                f"outer {CltAssumptions.TAIL_SHARE_MASS_FRACTION:.1%} of mass on "
                "each side, so the tails are heavy and the sqrt(N) regime is "
                "reached slowly. The predicted EMCC can be off by a factor."
            )
        if asymptotic.scaling_ratio is not None and (
            abs(asymptotic.scaling_ratio - 1.0) > CltAssumptions.SCALING_RATIO_TOLERANCE
        ):
            warnings.append(
                "sqrt(N) times the mean adversary distance changed by a factor "
                f"{asymptotic.scaling_ratio:.2f} between the smallest and the "
                "largest measured adversary sizes, so the sqrt(N) scaling the "
                "prediction assumes does not hold at these sizes."
            )

        return warnings

    def __str__(self) -> str:
        """
        Return a string representation of the BenchmarkingVariable.
        """
        return (
            f"BenchmarkingVariable("
            f"name='{self.name}', "
            f"description='{self.description}', "
            f"type='{self.type}', "
            f"file='{self.file_name}:{self.line_number}', "
            f"program='{self.program}')"
        )
