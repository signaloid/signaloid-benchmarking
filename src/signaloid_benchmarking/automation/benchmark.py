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

import subprocess
import yaml
import os
import re
import datetime
import functools
from signaloid_benchmarking.types import (
    BenchmarkingVariable,
)
from signaloid_benchmarking.automation.benchmarking_utils import (
    get_git_remote,
    expand_array_expressions,
)
from signaloid_benchmarking.automation.build import (
    compile_native,
    configuration_skip_reason,
    export_timing_env,
    run_timing_script,
)
from signaloid_benchmarking.automation.scce import (
    ScceRunner,
    scce_configuration_skip_reason,
)
from signaloid_benchmarking.automation.analysis import (
    load_emcc_data,
)
from signaloid_benchmarking.automation.arguments import (
    CORRELATION_CHOICES,
    REPRESENTATION_TYPE_CHOICES,
    validate_sweep_values,
)
from signaloid_benchmarking.config import (
    RepresentationTypes,
    Correlations,
    DistanceMetrics,
    ReportingMethods,
    SignaloidYaml,
    VariableTypes,
    TimingFormat,
    DEFAULT_UXHW_SDK_PATH,
    DEFAULT_CLI_HELPERS_PATH,
    DEFAULT_GROUND_TRUTH_SIZE,
    DEFAULT_ADVERSARY_MC_SIZE,
    DEFAULT_MAX_NUM_WEIGHTED_SAMPLES,
    DEFAULT_NUM_ADVERSARIES,
    DEFAULT_MAX_JUPITER_SIZE,
    DEFAULT_ADVERSARY_MAX_SIZE_SCALAR,
    command_line_arguments_hash,
    configuration_string,
)


class Benchmark:
    """
    Pipeline coordinator for the benchmarking automation tool.

    Holds the resolved configuration for a single application benchmark (paths
    to the application, the UxHw SDK and PIN, the representation types/sizes and
    correlation modes under test, ground-truth and adversary sample sizes, the
    distance metric, and reporting options) and threads it through the
    multi-stage pipeline. The per-stage work (compilation, database generation,
    sample generation, measurement loading, EMCC and UxHw-distance analysis, and
    report writing) lives in the focused ``build``, ``database_generator``,
    ``sample_generator``, ``measurement_loader``, ``analysis`` and
    ``report_writer`` modules. This class owns no per-stage logic itself and
    simply orchestrates those steps.
    """

    def __init__(
        self,
        path_to_application: str,
        path_to_uxhw_sdk: str = DEFAULT_UXHW_SDK_PATH,
        measure_dynamic_instructions: bool = False,
        has_analytic_ground_truth: bool = False,
        path_to_ground_truth_file: str = "",
        ground_truth_size: int = DEFAULT_GROUND_TRUTH_SIZE,
        adversary_mc_size: int = DEFAULT_ADVERSARY_MC_SIZE,
        adversary_max_size_scalar: int = DEFAULT_ADVERSARY_MAX_SIZE_SCALAR,
        representation_types: list[str] = [
            RepresentationTypes.ATHENS,
        ],
        representation_sizes: list[int] = [16, 32, 64, 128, 256, 512],
        max_jupiter_size: int = DEFAULT_MAX_JUPITER_SIZE,
        correlations: list[str] = [
            Correlations.DISABLED,
            Correlations.AUTOCORRELATION,
        ],
        ground_truth_type: str = RepresentationTypes.MONTE_CARLO,
        max_num_weighted_samples: int = DEFAULT_MAX_NUM_WEIGHTED_SAMPLES,
        num_parallel_workers: int | None = None,
        distance_type: str = DistanceMetrics.WASSERSTEIN_1,
        use_clt: bool = False,
        reporting_methods: list[str] = [
            ReportingMethods.MEAN,
            ReportingMethods.QUANTILE_95,
            ReportingMethods.QUANTILE_99,
        ],
        n_adversaries: int = DEFAULT_NUM_ADVERSARIES,
        use_binned_uxhw: bool = True,
        demo_cli_args: str = "",
        google_credentials: str | None = None,
        run_uxhw_scce: bool = False,
        path_to_cli_helpers: str = DEFAULT_CLI_HELPERS_PATH,
    ) -> None:
        """
        Store the resolved benchmark configuration.

        See the class docstring for the configuration groups. Each argument
        sets the correspondingly-named attribute.
        """
        self.path_to_application = os.path.expanduser(path_to_application)
        self.path_to_uxhw_sdk = os.path.expanduser(path_to_uxhw_sdk)
        self.measure_dynamic_instructions = measure_dynamic_instructions
        self.has_analytic_ground_truth = has_analytic_ground_truth
        if self.has_analytic_ground_truth:
            self.path_to_ground_truth_file = os.path.expanduser(
                path_to_ground_truth_file
            )
        self.ground_truth_size = ground_truth_size
        self.adversary_mc_size = adversary_mc_size
        self.adversary_max_size_scalar = adversary_max_size_scalar
        self.all_outputs_cla: str = ""
        self.representation_types = representation_types
        self.representation_sizes = representation_sizes
        self.correlations = correlations
        self.cwd = os.path.abspath(os.getcwd())
        self.results_dir = os.path.join(self.cwd, "results")
        self.logs_dir = os.path.join(self.cwd, "logs")
        self.plots_dir = os.path.join(self.results_dir, "plots")
        self.ground_truth_type = ground_truth_type
        self.max_num_weighted_samples = max_num_weighted_samples
        # None until get_machine_info resolves it to the detected core count
        # All readers run after that, by which point it is an int.
        self.num_parallel_workers: int | None = num_parallel_workers
        self.output_data_file = os.path.join(self.results_dir, "output_data.csv")
        self.asymptotic_dist_file = os.path.join(
            self.results_dir, "asymptotic_distances.csv"
        )
        self.uxhw_distance_file = os.path.join(self.results_dir, "uxhw_distances.csv")
        self.distance_type = distance_type
        self.use_clt = use_clt
        self.reporting_methods = reporting_methods
        self.variable_types: list[str] = []
        self.n_adversaries = n_adversaries
        self.use_binned_uxhw = use_binned_uxhw
        self.demo_cli_args = demo_cli_args
        self.google_credentials = google_credentials
        # With run_uxhw_scce the UxHw passes run on SCCE through the
        # signaloid-cli helper scripts instead of a local SDK build. The
        # runner is made once get_application_info has resolved what it
        # needs, and stays None otherwise.
        self.run_uxhw_scce = run_uxhw_scce
        self.path_to_cli_helpers = os.path.expanduser(path_to_cli_helpers)
        self.scce_runner: ScceRunner | None = None
        if self.run_uxhw_scce and self.measure_dynamic_instructions:
            print(
                "Warning! Intel PIN cannot instrument the UxHw runs on SCCE, so "
                "--measure-dynamic-instructions counts the native Monte Carlo "
                "binary only. The UxHw rows' Database Dyn. Inst. Count is the "
                "DynamicInstructions that SCCE reports, and their PIN count is "
                "left empty."
            )
        self.benchmarking_variables: list[BenchmarkingVariable] = []
        self.max_jupiter_size = max_jupiter_size
        # False until compile_native() sets it True on a successful native-MC
        # build (via Makefile or config.mk).
        self.has_native_mc = False
        # Populated by `load_measurement_dicts` (in `measurement_loader`)
        # via the orchestrator. Consumed by the Google-Sheets writer.
        self.uxhw_version: str = ""

    def get_machine_info(self) -> None:
        """
        Detect the machine model and CPU count, and resolve the parallel-worker
        count.
        """

        result = subprocess.check_output(["lscpu"], text=True)

        # Get the name of the computer model
        match = re.search(r"Model name:\s+(.+)", result)
        if match:
            self.machine_name = match.group(1).strip()
        else:
            self.machine_name = "Unknown"

        # Get the number of CPUs
        match = re.search(r"CPU\(s\)\s*:\s*(.+)", result)
        if match:
            self.n_processors = int(match.group(1).strip())
        else:
            self.n_processors = 1

        # Resolve an unset -j to the detected core count and clamp any request
        # above it.

        num_parallel_workers = (
            self.n_processors
            if self.num_parallel_workers is None
            else self.num_parallel_workers
        )

        if num_parallel_workers > self.n_processors:
            print(
                f"Warning! Requested {num_parallel_workers} parallel workers but only {self.n_processors} detected!"
            )
            num_parallel_workers = self.n_processors

        self.num_parallel_workers = num_parallel_workers

    def load_yaml(self) -> None:
        """
        Parse the application's signaloid.yaml into benchmarking variables.
        """

        self.benchmarking_variables = []
        program = "main"

        signaloid_yaml_path = self.path_to_application + "/signaloid.yaml"
        with open(signaloid_yaml_path, "r") as yaml_file:
            yaml_list = yaml.safe_load(yaml_file)

            # Get trace variable info for file path and line number
            trace_variables = yaml_list[SignaloidYaml.TRACE_VARIABLES]
            trace_variables = expand_array_expressions(trace_variables)

            if SignaloidYaml.BENCHMARKING_VARIABLES in yaml_list:
                if SignaloidYaml.ALL_OUTPUTS not in yaml_list:
                    raise ValueError(
                        f"signaloid.yaml at '{signaloid_yaml_path}' defines "
                        f"'{SignaloidYaml.BENCHMARKING_VARIABLES}' but is missing "
                        f"'{SignaloidYaml.ALL_OUTPUTS}'. An all-outputs invocation "
                        "is required to run combined-output measurements."
                    )
                all_outputs = yaml_list[SignaloidYaml.ALL_OUTPUTS]
                self.all_outputs_cla = all_outputs[0][
                    SignaloidYaml.COMMAND_LINE_ARGUMENTS
                ]
                full_cla = f"{self.demo_cli_args} {self.all_outputs_cla}".strip()
                self.value_id = command_line_arguments_hash(full_cla)

                # Get the list of BenchmarkingVariable entries
                benchmarking_variables = yaml_list[SignaloidYaml.BENCHMARKING_VARIABLES]

                for var_config in benchmarking_variables:
                    variable_name = var_config[SignaloidYaml.VARIABLE_NAME]
                    trace_variable = next(
                        (
                            elem
                            for elem in trace_variables
                            if elem.get(SignaloidYaml.EXPRESSION) == variable_name
                        ),
                        None,
                    )
                    if trace_variable is None:
                        raise ValueError(
                            f"Benchmarking variable '{variable_name}' has no matching traced variable!"
                        )
                    file_name = trace_variable[SignaloidYaml.FILE]
                    line_number = trace_variable[SignaloidYaml.LINE_NUMBER]
                    path = f"{self.path_to_application}/src/{file_name}"

                    variable_description = var_config[
                        SignaloidYaml.VARIABLE_DESCRIPTION
                    ]
                    variable_type = var_config[SignaloidYaml.OUTPUT_OBJECT]
                    command_line_args = var_config[SignaloidYaml.COMMAND_LINE_ARGUMENTS]

                    # Create BenchmarkingVariable object. The last four
                    # keys are optional, and an absent one stays None so
                    # the run-wide flag applies to this variable.
                    variable = BenchmarkingVariable(
                        value_id=self.value_id,
                        name=variable_name,
                        description=variable_description,
                        program=program,
                        path=path,
                        line_number=line_number,
                        file_name=file_name,
                        type=variable_type,
                        cla=command_line_args,
                        representation_types=self._read_sweep_override(
                            var_config,
                            SignaloidYaml.REPRESENTATION_TYPES,
                            allowed=REPRESENTATION_TYPE_CHOICES,
                        ),
                        representation_sizes=self._read_sweep_override(
                            var_config,
                            SignaloidYaml.REPRESENTATION_SIZES,
                            integers_only=True,
                        ),
                        correlations=self._read_sweep_override(
                            var_config,
                            SignaloidYaml.CORRELATION_TRACKING,
                            allowed=CORRELATION_CHOICES,
                        ),
                        analytic_ground_truth=self._read_analytic_ground_truth(
                            var_config
                        ),
                    )
                    # Every result file of a variable is named after its
                    # formatted description, so it must be non-empty and
                    # differ from every other variable's.
                    if not variable.formatted_description:
                        raise ValueError(
                            f"Benchmarking variable '{variable_name}' has the "
                            f"description '{variable_description}', which has "
                            "no letters or digits to name its result files."
                        )
                    for other in self.benchmarking_variables:
                        if other.formatted_description == (
                            variable.formatted_description
                        ):
                            raise ValueError(
                                f"Benchmarking variables '{other.name}' and "
                                f"'{variable_name}' would share the result "
                                f"files '{variable.formatted_description}'. "
                                f"Their descriptions '{other.description}' and "
                                f"'{variable_description}' differ only in "
                                "characters that file names drop."
                            )
                    self.benchmarking_variables.append(variable)
            else:
                print(
                    "Old Yaml format detected. Falling back to using trace variables as benchmarking variables."
                )
                self.all_outputs_cla = f"-S {len(trace_variables)}"
                full_cla = f"{self.demo_cli_args} {self.all_outputs_cla}".strip()
                self.value_id = command_line_arguments_hash(full_cla)
                for i, elem in enumerate(trace_variables):
                    file_name = elem[SignaloidYaml.FILE]
                    line_number = elem[SignaloidYaml.LINE_NUMBER]
                    path = f"{self.path_to_application}/src/{file_name}"
                    variable_name = elem[SignaloidYaml.EXPRESSION]
                    # Create BenchmarkingVariable object
                    variable = BenchmarkingVariable(
                        value_id=self.value_id,
                        name=variable_name,
                        description=f"Variable {i}",
                        program=program,
                        path=path,
                        line_number=line_number,
                        file_name=file_name,
                        type=VariableTypes.DISTRIBUTION,
                        cla=f"-S {i}",
                    )
                    self.benchmarking_variables.append(variable)

    def resolve_skipped_configurations(self) -> None:
        """
        Decide which configurations of each variable's matrix cannot run.

        A configuration is skipped when the SDK ships no runtime library
        for it, or on SCCE when there is no core for its representation
        type, or when it is a Jupiter size above ``max_jupiter_size``.
        Each distinct configuration is asked about once, and every skip is
        logged with its reason before anything is built.

        Raises:
            ValueError: If a variable has no configuration left to run,
                which would otherwise finish with nothing measured for it.
            RuntimeError: If the SDK cannot be asked about a
                configuration.
        """
        reasons: dict[tuple[str, int, str], str | None] = {}
        skipped_variables: dict[str, list[str]] = {}
        skip_reasons: dict[str, str] = {}
        for variable in self.benchmarking_variables:
            types, sizes, correlations = variable.representation_matrix(
                representation_types=self.representation_types,
                representation_sizes=self.representation_sizes,
                correlations=self.correlations,
            )
            variable.skipped_configurations = {}
            runnable = 0
            for representation_type in types:
                for representation_size in sizes:
                    for correlation in correlations:
                        key = (representation_type, representation_size, correlation)
                        if key not in reasons:
                            reasons[key] = self._configuration_skip_reason(
                                representation_type=representation_type,
                                representation_size=representation_size,
                                correlation=correlation,
                            )
                        reason = reasons[key]
                        if reason is None:
                            runnable += 1
                            continue

                        configuration = configuration_string(
                            representation_type=representation_type,
                            representation_size=representation_size,
                            correlation=correlation,
                        )
                        variable.skipped_configurations[configuration] = reason
                        skip_reasons[configuration] = reason
                        skipped_variables.setdefault(configuration, []).append(
                            variable.name
                        )

            if runnable == 0:
                raise ValueError(
                    f"Error! Benchmarking variable '{variable.name}' has no "
                    "configuration that can run. "
                    + " ".join(
                        f"{configuration} is skipped because {reason}."
                        for configuration, reason in (
                            variable.skipped_configurations.items()
                        )
                    )
                )

        for configuration, names in skipped_variables.items():
            print(
                f"Skipping {configuration} for {', '.join(names)}, because "
                f"{skip_reasons[configuration]}."
            )

    def _configuration_skip_reason(
        self,
        *,
        representation_type: str,
        representation_size: int,
        correlation: str,
    ) -> str | None:
        """
        Say why a configuration cannot run, if it cannot.

        On SCCE the answer depends on which cores it can build. Otherwise
        it depends on which runtime libraries the SDK ships.

        Args:
            representation_type: For example ``Europa``.
            representation_size: For example ``8``.
            correlation: ``Disabled`` or ``Autocorrelation``.

        Returns:
            The reason, or ``None`` when the configuration can run.
        """
        if self.run_uxhw_scce:
            return scce_configuration_skip_reason(
                representation_type=representation_type,
                representation_size=representation_size,
                max_jupiter_size=self.max_jupiter_size,
            )

        return configuration_skip_reason(
            path_to_application=self.path_to_application,
            path_to_uxhw_sdk=self.path_to_uxhw_sdk,
            representation_type=representation_type,
            representation_size=representation_size,
            correlation=correlation,
            max_jupiter_size=self.max_jupiter_size,
        )

    def has_configurations_left(self, variable: BenchmarkingVariable) -> bool:
        """
        Say whether a variable still has a configuration to measure.

        Args:
            variable: One of this benchmark's variables.

        Returns:
            ``False`` once every configuration of its matrix is skipped,
            for example because every UxHw run of it died.
        """
        return bool(
            variable.benchmarked_configurations(
                representation_types=self.representation_types,
                representation_sizes=self.representation_sizes,
                correlations=self.correlations,
            )
        )

    def require_a_configuration_left(self) -> None:
        """
        Stop the run when no variable has a configuration left.

        A UxHw run that dies drops only its own configuration, so the run
        goes on with the rest. Once nothing is left there is nothing to
        report, and the run stops here rather than at a later step.

        Raises:
            RuntimeError: If every configuration of every variable is
                skipped. The message gives each reason.
        """
        if any(
            self.has_configurations_left(variable)
            for variable in self.benchmarking_variables
        ):
            return

        raise RuntimeError(
            "Error! No benchmarking variable has a UxHw configuration left. "
            + " ".join(
                f"{configuration} of {variable.name} is skipped because {reason}."
                for variable in self.benchmarking_variables
                for configuration, reason in variable.skipped_configurations.items()
            )
        )

    @staticmethod
    def _read_sweep_override(
        var_config: dict,
        key: str,
        *,
        allowed: list[str] | None = None,
        integers_only: bool = False,
    ) -> list | None:
        """
        Read one optional sweep axis from a BenchmarkingVariables entry.

        A declared axis replaces the run-wide one for this variable. It is
        not intersected with it, so a variable gets exactly what its entry
        lists.

        Args:
            var_config: One entry of ``BenchmarkingVariables``.
            key: The signaloid.yaml key of the axis.
            allowed: The permitted values, or ``None`` for any value.
            integers_only: Whether every value must be an integer.

        Returns:
            The declared values, or ``None`` when the entry does not
            declare the key.

        Raises:
            ValueError: If the key is present with anything other than a
                non-empty list of permitted values. The rules and messages
                are the run-wide flags' own.
        """
        if key not in var_config:
            return None

        values = var_config[key]
        validate_sweep_values(
            label=(
                f"'{key}' of benchmarking variable "
                f"'{var_config[SignaloidYaml.VARIABLE_NAME]}'"
            ),
            values=values,
            allowed=allowed,
            integers_only=integers_only,
        )

        return list(values)

    def _read_analytic_ground_truth(self, var_config: dict) -> str | None:
        """
        Read the optional analytic reference of a BenchmarkingVariables entry.

        The value names a script relative to the application root, as the
        benchmarking template lays it out. It is checked here so a typo
        stops the run before any build rather than at the ground-truth
        stage.

        Args:
            var_config: One entry of ``BenchmarkingVariables``.

        Returns:
            The absolute path of the script, or ``None`` when the entry
            does not declare the key.

        Raises:
            ValueError: If the key is present but does not name an
                existing file.
        """
        key = SignaloidYaml.ANALYTIC_GROUND_TRUTH
        if key not in var_config:
            return None

        variable_name = var_config[SignaloidYaml.VARIABLE_NAME]
        script = var_config[key]
        if not isinstance(script, str) or not script.strip():
            raise ValueError(
                f"Error! '{key}' of benchmarking variable '{variable_name}' "
                f"must name a script, got {script!r}."
            )

        script_path = os.path.abspath(
            os.path.join(self.path_to_application, os.path.expanduser(script))
        )
        if not os.path.isfile(script_path):
            raise ValueError(
                f"Error! '{key}' of benchmarking variable '{variable_name}' "
                f"names {script!r}, but {script_path} is not a file."
            )

        return script_path

    @staticmethod
    def _resolve_application_version(path_to_application: str) -> str:
        """
        Resolve a non-empty version string for an application path.

        Uses the git short hash when the path is a git repository, and
        falls back to a date stamp otherwise (or when ``git rev-parse``
        fails). The fallback guarantees a non-empty value, which
        ``get-timings.sh`` requires via ``require_var APPLICATION_VERSION``.

        Args:
            path_to_application: Path to the application repository.

        Returns:
            The git short hash, or a date stamp when unavailable.
        """
        # A linked worktree or a submodule has a .git file, not a directory.
        if os.path.exists(os.path.join(path_to_application, ".git")):
            try:
                return subprocess.check_output(
                    [
                        "git",
                        "-C",
                        path_to_application,
                        "rev-parse",
                        "--short=7",
                        "HEAD",
                    ],
                    text=True,
                ).strip()
            except subprocess.CalledProcessError:
                print(
                    "Error retrieving git hash. "
                    "Falling back to date-based versioning."
                )
        else:
            print(
                "Warning! Application path is not a git repository. "
                "Falling back to date-based versioning."
            )
        return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    def get_application_info(self) -> None:
        """
        Resolve application identity and prepare the benchmark's inputs.

        Resolves the application name/version, loads signaloid.yaml, compiles
        the native-MC build, creates the output directories, and derives the
        database paths and shared timing environment.
        """
        self.application_version = self._resolve_application_version(
            self.path_to_application
        )
        application_name = [
            part for part in self.path_to_application.split("/") if part
        ][-1]
        self.application_name = application_name.replace("Signaloid-Demo-", "")

        # On SCCE the application is built from its commit, while the native
        # passes build the working tree, so the two must be the same code.
        # Checked before anything is compiled.
        if self.run_uxhw_scce:
            self.scce_runner = ScceRunner(
                path_to_application=self.path_to_application,
                path_to_cli_helpers=self.path_to_cli_helpers,
                application_name=self.application_name,
                application_version=self.application_version,
                demo_cli_args=self.demo_cli_args,
            )
            self.scce_runner.check_checkout_matches_remote()
            self.scce_runner.check_working_tree_is_committed()

        # Load the signaloid.yaml file
        self.load_yaml()
        self.resolve_skipped_configurations()

        # Attempt native compilation
        if self.num_parallel_workers is None:
            raise RuntimeError(
                "num_parallel_workers is unset; call get_machine_info() "
                "before get_application_info()."
            )
        num_parallel_workers = self.num_parallel_workers
        (
            self.native_executable_name,
            self.native_compilation_command,
            self.native_executable_dir,
            self.has_native_mc,
        ) = compile_native(
            path_to_application=self.path_to_application,
            num_parallel_workers=num_parallel_workers,
        )

        # Obtain git remote
        self.git_repo_remote = get_git_remote(self.path_to_application)

        ground_truth_prefix = adversary_prefix = self.application_name

        # Create output directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        # Define database paths
        self.tracing_db_path = os.path.join(
            self.results_dir,
            f"uxhwExecutionStatistics-{self.application_version}-{self.value_id}-tracing.db",
        )
        if self.has_native_mc:
            ground_truth_prefix += "-nativeMC"
            adversary_prefix += "-nativeMC"
        else:
            ground_truth_prefix += "-uxhwExecutionStatistics"
            adversary_prefix += "-uxhwExecutionStatistics"

        # signaloid.yaml has been read by now, so a variable that declares
        # its own analytic reference names the database too.
        if self.has_analytic_ground_truth or any(
            variable.analytic_ground_truth is not None
            for variable in self.benchmarking_variables
        ):
            ground_truth_prefix += "-analytic"

        self.ground_truth_db_path = os.path.join(
            self.results_dir,
            f"{ground_truth_prefix}-{self.application_version}-{self.value_id}-ground-{self.ground_truth_size}.db",
        )
        self.adversary_db_path = os.path.join(
            self.results_dir,
            f"{adversary_prefix}-{self.application_version}-{self.value_id}-adv-{self.adversary_mc_size}.db",
        )

        # Export common timing environment variables
        self.tracing_db_path = export_timing_env(
            path_to_uxhw_sdk=self.path_to_uxhw_sdk,
            measure_dynamic_instructions=self.measure_dynamic_instructions,
            path_to_application=self.path_to_application,
            application_name=self.application_name,
            application_version=self.application_version,
            results_dir=self.results_dir,
            logs_dir=self.logs_dir,
            tracing_db_path=self.tracing_db_path,
        )

    def intermediate_timings_path(self) -> str:
        """
        Path of the transient intermediate timings file.

        Returns:
            The absolute path of the intermediate file that bash writes
            line-by-line and Python parses into the canonical JSON artifact.
        """
        filename = (
            f"{self.application_name}-{self.application_version}"
            f"{TimingFormat.INTERMEDIATE_SUFFIX}"
        )
        return os.path.join(self.results_dir, filename)

    def json_timings_path(self) -> str:
        """
        Path of the canonical JSON timings artifact.

        Returns:
            The absolute path of the JSON artifact produced by the Python
            reader.
        """
        filename = (
            f"{self.application_name}-{self.application_version}"
            f"{TimingFormat.JSON_SUFFIX}"
        )
        return os.path.join(self.results_dir, filename)

    def bind_timing_script(self) -> "functools.partial[None]":
        """
        Bind the run_timing_script kwargs shared by every timing pass.

        Every caller (UxHw timing, native-MC timing, and the tracing database
        generator) passes the same seven kwargs sourced from this Benchmark.

        Returns:
            A partial that still needs the per-call arguments: the
            ``variable_index`` and exactly one mode flag (``timing`` /
            ``native_mc_timing`` / ``tracing``).
        """
        return functools.partial(
            run_timing_script,
            benchmarking_variables=self.benchmarking_variables,
            demo_cli_args=self.demo_cli_args,
            representation_types=self.representation_types,
            representation_sizes=self.representation_sizes,
            correlations=self.correlations,
            logs_dir=self.logs_dir,
            intermediate_timings_path=self.intermediate_timings_path(),
        )

    def generate_uxhw_timings(self) -> None:
        """
        Generate the UxHw timing data for every benchmarking variable.
        """
        print("Generating timing database.")
        bound_run_timing_script = self.bind_timing_script()
        for i, variable in enumerate(self.benchmarking_variables):
            # Every tracing run of this variable died, so there is nothing
            # to time.
            if not self.has_configurations_left(variable):
                continue
            bound_run_timing_script(variable_index=i, timing=True)

    def generate_native_mc_timings(self) -> None:
        """
        Generate the native Monte Carlo timing data.

        If EMCC data is not found on the Benchmark object, attempts to load it
        from the equivalent-MC output file and populate the benchmarking
        variables.
        """

        load_emcc_data(
            benchmarking_variables=self.benchmarking_variables,
            output_data_file=self.output_data_file,
        )

        print("Generating native Monte Carlo timing data")
        bound_run_timing_script = self.bind_timing_script()
        for i, variable in enumerate(self.benchmarking_variables):
            # The native build is timed at each equivalent Monte Carlo count.
            # A variable whose every tracing run died has none, the same
            # variables step 10 left out, so each timed variable has one
            # UxHw timing run and one native one for the loader to read.
            if not variable.emcc_results.equiv_mc_list:
                continue
            bound_run_timing_script(variable_index=i, native_mc_timing=True)
