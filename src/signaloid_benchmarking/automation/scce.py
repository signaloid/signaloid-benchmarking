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


import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, TextIO

from signaloid_benchmarking.automation import results_database
from signaloid_benchmarking.automation.benchmarking_utils import (
    get_git_branch,
    utc_timestamp,
)
from signaloid_benchmarking.automation.build import jupiter_size_skip_reason
from signaloid_benchmarking.config import (
    EquivMC,
    Scce,
    TimingFormat,
    command_line_arguments_hash,
    configuration_string,
    full_command_line_arguments,
)
from signaloid_benchmarking.types import BenchmarkingVariable
from signaloid.distributional.distributional import DistributionalValue

# The helper scripts, and the files they read and write. They read their
# credentials from files beside themselves and write their results to the
# root of the repository they run in, which here is the application.
BUILD_SCRIPT = "signaloid-cli-build.sh"
RUN_SCRIPT = "signaloid-cli-run.sh"
API_KEY_FILE = "SIGNALOID_SCCE_API_KEY"
LATEST_BUILD_ID_FILE = ".signaloid-cli-latest-build-id.txt"
LATEST_TASK_ID_FILE = ".signaloid-cli-latest-task-id.txt"

# Recorded as the UxHw SDK version of a run made on SCCE, where no SDK
# release file exists to read one from, and as its UxHw repetition target,
# since each configuration runs once, in the tracing step.
UXHW_SDK_VERSION = "SCCE"
UXHW_TARGET_REPETITIONS = "1"

# Every configuration is built on a core of this class. A core that has to
# be created gets this memory size.
CORE_CLASS = "C0Pro"
CORE_MEMORY_SIZE = 2_000_000_000

# Where the application's code lives, as SCCE and the native passes see it.
# The working tree is checked here before anything is compiled, so that both
# build the same code. The rest of the repository holds the benchmark's own
# outputs and is not compiled by either.
CHECKED_PATHS = ("src", "submodules", "inputs", "signaloid.yaml", "Makefile")

# The build resources the timing script copies into src, over a tracked copy
# when the application has one, so a modification to them is the tool's own.
BUILD_RESOURCES = ("src/Makefile", "src/Makefile.pro", "src/init.S", "src/startup.cpp")

# What the benchmark itself writes under the checked paths, as gitignore
# patterns. None of it is source, so it does not make the working tree differ
# from the commit. Given to git as an excludes file, so the patterns have
# git's own meaning.
GENERATED_ARTIFACTS = BUILD_RESOURCES + (
    # What the timing script builds and runs in src.
    "src/*.m",
    "src/main-*",
    "src/*.o",
    "src/opt.err",
    "src/*.stdout",
    "src/*.stderr",
    "src/sunflower-*.out",
    "src/demo-native-mc",
    "src/ground-truth.csv",
    # What the UxHw SDK's own build leaves in src: the program before the
    # timing script renames it, its intermediate code and its build logs.
    "src/main",
    "src/build/",
    "src/make.output",
    "src/ucc.output",
    "src/uld.output",
    "src/yoshiExecutionStatistics*",
    # What the native binary and the SDK runtime write where they run.
    f"src/{EquivMC.MC_OUTPUT_FILENAME}",
    "src/inscount.out",
    "src/signaloidUxHwExecutionStatistics*",
    f"inputs/{EquivMC.MC_OUTPUT_FILENAME}",
    "inputs/inscount.out",
    "inputs/signaloidUxHwExecutionStatistics*",
)

# Appended to every task's arguments. ``-j`` prints each output as a Ux
# string in JSON, keyed by variable name.
TASK_FLAGS = "-j"


@dataclass(frozen=True)
class _ScceTask:
    """
    One task that ran on SCCE.

    Attributes:
        task_id: The identifier SCCE gave the task.
        stdout: What the CLI printed for the task's stdout.
        stats: The ``Stats`` of the task record.
    """

    task_id: str
    stdout: str
    stats: dict[str, Any]


@dataclass(frozen=True)
class _ScceMeasurement:
    """
    The timing of one task, in the fields the measurement loader reads.

    Attributes:
        configuration: The configuration the task ran as, for example
            ``Athens-16``.
        time: The task's ``ProcessorTime``, in seconds. The application's
            own clock is not usable on SCCE, so this is what both In
            Application Time and Database Time record there.
        e2e_time: The task's ``ExecutionTimeInMilliseconds``, in seconds.
        dyn_inst_count: The task's ``DynamicInstructions``.
    """

    configuration: str
    time: float
    e2e_time: float | None
    dyn_inst_count: float | None


def scce_configuration_skip_reason(
    *,
    representation_type: str,
    representation_size: int,
    max_jupiter_size: int,
) -> str | None:
    """
    Say why a configuration cannot be benchmarked on SCCE, if it cannot.

    SCCE builds cores for Athens and Jupiter only. A Jupiter representation
    larger than ``--max-jupiter-size`` is refused as a memory guard, as it
    is for a local build.

    Args:
        representation_type: For example ``Europa``.
        representation_size: For example ``8``.
        max_jupiter_size: The largest Jupiter size to benchmark.

    Returns:
        The reason, or ``None`` when the configuration can run.
    """
    if representation_type not in Scce.MICROARCHITECTURES:
        return f"SCCE has no {representation_type} core"

    return jupiter_size_skip_reason(
        representation_type=representation_type,
        representation_size=representation_size,
        max_jupiter_size=max_jupiter_size,
    )


def _core_name(
    *,
    representation_type: str,
    representation_size: int,
    correlation: str,
) -> str:
    """
    Name the core a configuration is created as.

    Args:
        representation_type: For example ``Athens``.
        representation_size: For example ``16``.
        correlation: ``Disabled`` or ``Autocorrelation``.

    Returns:
        For example ``C0Pro-Athens-16-Autocorrelation``.
    """
    configuration = configuration_string(
        representation_type=representation_type,
        representation_size=representation_size,
        correlation=correlation,
    )

    return f"{CORE_CLASS}-{configuration}"


def _match_core(
    cores: list[dict[str, Any]],
    *,
    representation_type: str,
    representation_size: int,
    correlation: str,
) -> dict[str, Any] | None:
    """
    Find a core a configuration can run on.

    A core matches on its class, microarchitecture, precision and
    correlation tracking, whatever it is named, so a default core and a
    user-created one serve alike.

    Args:
        cores: Core records as ``signaloid-cli cores list`` returns them.
        representation_type: For example ``Athens``.
        representation_size: For example ``16``.
        correlation: ``Disabled`` or ``Autocorrelation``.

    Returns:
        The first matching core, or ``None``.
    """
    wanted = (
        CORE_CLASS,
        representation_type,
        int(representation_size),
        Scce.CORE_CORRELATION_TRACKING[correlation],
    )
    for core in cores:
        found = (
            core.get("Class"),
            core.get("Microarchitecture"),
            core.get("Precision"),
            core.get("CorrelationTracking"),
        )
        if found == wanted:
            return core

    return None


def _first_json_value(text: str) -> Any:
    """
    Decode the first JSON value in what the CLI printed.

    The CLI prefixes its output with progress text, so the value is read
    from the first brace, bracket or quote onwards.

    Args:
        text: What the CLI printed.

    Returns:
        The decoded value.

    Raises:
        ValueError: If the text holds no JSON value.
    """
    starts = [index for index in (text.find(mark) for mark in '{["') if index >= 0]
    if not starts:
        raise ValueError(f"No JSON was printed:\n{text.strip()}")
    try:
        value, _ = json.JSONDecoder().raw_decode(text, min(starts))
    except json.JSONDecodeError as error:
        raise ValueError(f"The JSON printed could not be parsed: {error}") from error

    return value


def _parse_task_stdout(text: str) -> dict[str, Any]:
    """
    Read the JSON the application printed under ``-j`` from what the CLI
    prints for a task's stdout.

    Some versions of the CLI print the stdout as one JSON string literal
    and others as it is. Either way the object is the first JSON value,
    decoded once more when it came as a string.

    Args:
        text: What ``signaloid-cli tasks output`` printed.

    Returns:
        The JSON object.

    Raises:
        ValueError: If the text holds no JSON object.
    """
    printed = _first_json_value(text)
    if isinstance(printed, str):
        try:
            printed = _first_json_value(printed)
        except ValueError as error:
            raise ValueError(
                "The task printed no JSON object. Was it run with -j?"
            ) from error
    if not isinstance(printed, dict):
        raise ValueError("The task printed no JSON object. Was it run with -j?")

    return printed


def _ux_strings_for(output: dict[str, Any], variable_name: str) -> list[str]:
    """
    The values the application printed for one variable under ``-j``.

    Args:
        output: The JSON object of the task's stdout.
        variable_name: For example ``outputVariables[0]``.

    Returns:
        The variable's ``values``, each a Ux string.

    Raises:
        ValueError: If the output holds no entry for the variable, or an
            empty one.
    """
    for plot in output.get("plots", []):
        if plot.get("variableID") == variable_name:
            values = [str(value) for value in plot.get("values", [])]
            if not values:
                raise ValueError(f"The task printed no value for {variable_name!r}.")
            return values

    raise ValueError(f"The task printed no output for {variable_name!r}.")


def _write_timing_run(
    handle: TextIO,
    *,
    application_name: str,
    application_version: str,
    command_line_arguments: str,
    measurements: list[_ScceMeasurement],
) -> None:
    """
    Write one run to the intermediate timing file, as get-timings.sh does.

    Args:
        handle: The open intermediate timing file.
        application_name: The application's name.
        application_version: The application's version.
        command_line_arguments: The variable's own arguments, which the
            measurement loader matches on. The tasks ran with the JSON flag
            appended, which is not recorded here, so an application's own
            ``-j`` cannot be mistaken for it.
        measurements: One measurement per configuration.
    """
    meta = (
        (TimingFormat.META_KEY_TIMESTAMP, utc_timestamp()),
        (TimingFormat.META_KEY_APPLICATION_NAME, application_name),
        (TimingFormat.META_KEY_APPLICATION_VERSION, application_version),
        (TimingFormat.META_KEY_UXHW_SDK_VERSION, UXHW_SDK_VERSION),
        (TimingFormat.META_KEY_COMMAND_LINE_ARGUMENTS, command_line_arguments),
        (
            TimingFormat.META_KEY_COMMAND_LINE_ARGUMENTS_HASH,
            command_line_arguments_hash(command_line_arguments),
        ),
        (TimingFormat.META_KEY_UXHW_TARGET_REPETITIONS, UXHW_TARGET_REPETITIONS),
    )
    for key, value in meta:
        handle.write(f"{TimingFormat.META_TAG} {key} {value}\n")

    for measurement in measurements:
        fields = [
            measurement.configuration,
            # In Application Time and Database Time, which the speedup
            # divides the native time by, are the one processor time SCCE
            # reports.
            str(measurement.time),
            str(measurement.time),
            _measurement_field(measurement.e2e_time),
            _measurement_field(measurement.dyn_inst_count),
            # No Intel PIN count is taken on SCCE.
            TimingFormat.MISSING_VALUE,
        ]
        handle.write(f"{TimingFormat.MEASUREMENT_TAG} {' '.join(fields)}\n")


def _measurement_field(value: float | None) -> str:
    """
    Write one numeric field of a measurement line.

    Args:
        value: The measured value, or ``None`` when it was not measured.

    Returns:
        The value as text, or the missing-value sentinel.
    """
    if value is None:
        return TimingFormat.MISSING_VALUE

    return str(value)


def _cli_output(arguments: list[str]) -> str:
    """
    Run ``signaloid-cli`` and return everything it printed.

    Args:
        arguments: The arguments after ``signaloid-cli``.

    Returns:
        Its stdout and stderr, in order.

    Raises:
        RuntimeError: If the command fails.
    """
    command = ["signaloid-cli", *arguments]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{' '.join(command)} failed (exit code {result.returncode}).\n"
            f"{result.stdout.strip()}"
        )

    return result.stdout


def _cli_json(arguments: list[str]) -> Any:
    """
    Run ``signaloid-cli`` and return the JSON it prints.

    Args:
        arguments: The arguments after ``signaloid-cli``.

    Returns:
        The decoded JSON.

    Raises:
        RuntimeError: If the command fails.
        ValueError: If it prints no JSON.
    """
    return _first_json_value(_cli_output(arguments))


def _read_latest(path_to_application: str, file_name: str) -> str:
    """
    Read one of the files the helper scripts leave in the application root.

    Args:
        path_to_application: The application's root.
        file_name: The file's name, for example ``LATEST_BUILD_ID_FILE``.

    Returns:
        The file's contents, stripped.

    Raises:
        RuntimeError: If the file is not there, which means the script did
            not get as far as writing it.
    """
    path = os.path.join(path_to_application, file_name)
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip()
    except FileNotFoundError as error:
        raise RuntimeError(
            f"The signaloid-cli helper script did not write {path}."
        ) from error


class ScceRunner:
    """
    Builds and runs an application on SCCE through the helper scripts.

    Cores and builds are found or made once per configuration and reused
    for every variable. The tasks' Ux strings go into the tracing database
    as they arrive, and their timing is kept for ``write_uxhw_timings``.
    """

    def __init__(
        self,
        *,
        path_to_application: str,
        path_to_cli_helpers: str,
        application_name: str,
        application_version: str,
        demo_cli_args: str,
    ) -> None:
        """
        Check the helper scripts are there and remember the application.

        Args:
            path_to_application: The application's root, which must be a
                clone of the repository SCCE builds.
            path_to_cli_helpers: Directory holding the helper scripts and
                their credential files.
            application_name: Recorded as the model of every run.
            application_version: Recorded in the timing file.
            demo_cli_args: Arguments given to every task before the
                variable's own.

        Raises:
            FileNotFoundError: If a helper script is missing.
        """
        self.path_to_application = path_to_application
        path_to_cli_helpers = os.path.abspath(os.path.expanduser(path_to_cli_helpers))
        self.path_to_cli_helpers = path_to_cli_helpers
        self.application_name = application_name
        self.application_version = application_version
        self.demo_cli_args = demo_cli_args
        for script in (BUILD_SCRIPT, RUN_SCRIPT):
            if not os.access(os.path.join(path_to_cli_helpers, script), os.X_OK):
                raise FileNotFoundError(
                    f"--run-uxhw-scce needs {script} in {path_to_cli_helpers}. "
                    "Point --path-to-cli-helpers at a checkout of "
                    "signaloid-cli-helpers."
                )
        # The cores SCCE offers, listed once on first need. Then the core
        # of each configuration and the build on each core, so each is asked
        # for once.
        self.cores: list[dict[str, Any]] | None = None
        self.cores_by_configuration: dict[str, dict[str, Any]] = {}
        self.build_ids: dict[str, str] = {}
        # Keyed by variable name, filled by trace_variable.
        self.measurements: dict[str, list[_ScceMeasurement]] = {}

    def login(self) -> None:
        """
        Authenticate the CLI with the API key beside the helper scripts.

        The scripts log in themselves, but the cores are listed before the
        first of them runs.

        Raises:
            RuntimeError: If the key file is missing or the login fails.
        """
        key_path = os.path.join(self.path_to_cli_helpers, API_KEY_FILE)
        try:
            with open(key_path, encoding="utf-8") as handle:
                api_key = handle.read().strip()
        except FileNotFoundError as error:
            raise RuntimeError(f"API key file not found: {key_path}") from error
        # The key is not echoed on failure.
        result = subprocess.run(
            ["signaloid-cli", "auth", "login", "--api-key", api_key, "--quiet"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "signaloid-cli auth login failed "
                f"(exit code {result.returncode}).\n{result.stderr.strip()}"
            )

    def _core(
        self,
        *,
        representation_type: str,
        representation_size: int,
        correlation: str,
    ) -> dict[str, Any]:
        """
        The core a configuration runs on, created when none matches.

        Args:
            representation_type: For example ``Athens``.
            representation_size: For example ``16``.
            correlation: ``Disabled`` or ``Autocorrelation``.

        Returns:
            The core's record, with at least ``CoreID`` and ``Name``.

        Raises:
            RuntimeError: If the CLI fails or a created core has no
                identifier.
        """
        configuration = configuration_string(
            representation_type=representation_type,
            representation_size=representation_size,
            correlation=correlation,
        )
        if configuration in self.cores_by_configuration:
            return self.cores_by_configuration[configuration]

        if self.cores is None:
            self.cores = []
            for arguments in (
                ["cores", "list", "--format", "json"],
                ["cores", "list", "--default", "--format", "json"],
            ):
                self.cores.extend(_cli_json(arguments).get("Cores", []))
        core = _match_core(
            self.cores,
            representation_type=representation_type,
            representation_size=representation_size,
            correlation=correlation,
        )
        if core is None:
            name = _core_name(
                representation_type=representation_type,
                representation_size=representation_size,
                correlation=correlation,
            )
            print(f"Creating SCCE core {name}")
            core = _cli_json(
                [
                    "cores",
                    "create",
                    "--name",
                    name,
                    "--class",
                    CORE_CLASS,
                    "--precision",
                    str(representation_size),
                    "--memory",
                    str(CORE_MEMORY_SIZE),
                    "--microarchitecture",
                    representation_type,
                    "--correlation-tracking",
                    Scce.CORE_CORRELATION_TRACKING[correlation],
                ]
            )
            self.cores.append(core)
        if not core.get("CoreID"):
            raise RuntimeError(f"The SCCE core record has no CoreID: {core}")
        print(
            f"Using SCCE core {core.get('Name')} ({core['CoreID']}) for {configuration}"
        )
        self.cores_by_configuration[configuration] = core

        return core

    def _build_id(self, core_id: str) -> str:
        """
        The build of the application for a core, made when there is none.

        Runs ``signaloid-cli-build.sh`` in the application, which pushes
        the current branch, connects the repository to SCCE if needed, and
        builds it on the core.

        Args:
            core_id: The core to build for.

        Returns:
            The build's identifier.

        Raises:
            RuntimeError: If the script fails.
        """
        if core_id in self.build_ids:
            return self.build_ids[core_id]

        script = os.path.join(self.path_to_cli_helpers, BUILD_SCRIPT)
        result = subprocess.run(
            [script, "--core-id", core_id],
            cwd=self.path_to_application,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{BUILD_SCRIPT} failed (exit code {result.returncode})."
            )
        build_id = _read_latest(self.path_to_application, LATEST_BUILD_ID_FILE)
        self.build_ids[core_id] = build_id

        return build_id

    def _run_task(self, *, build_id: str, arguments: str) -> _ScceTask:
        """
        Run one task of a build and fetch what it printed and what it cost.

        Runs ``signaloid-cli-run.sh`` in the application, then reads the
        task's record for its statistics and its stdout from the CLI.

        Args:
            build_id: The build to run.
            arguments: The task's command-line arguments.

        Returns:
            The task.

        Raises:
            RuntimeError: If the script or the CLI fails.
        """
        script = os.path.join(self.path_to_cli_helpers, RUN_SCRIPT)
        # Removed first, so that a task id found afterwards is this run's.
        task_id_path = os.path.join(self.path_to_application, LATEST_TASK_ID_FILE)
        if os.path.isfile(task_id_path):
            os.remove(task_id_path)
        result = subprocess.run(
            [script, "--build-id", build_id, "--args", arguments],
            cwd=self.path_to_application,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{RUN_SCRIPT} failed (exit code {result.returncode})."
                + self._task_outcome()
            )
        task_id = _read_latest(self.path_to_application, LATEST_TASK_ID_FILE)
        record = _cli_json(["tasks", "get", "--task-id", task_id, "--format", "json"])
        stdout = _cli_output(
            ["tasks", "output", "--task-id", task_id, "--stream", "stdout"]
        )

        return _ScceTask(task_id=task_id, stdout=stdout, stats=record.get("Stats", {}))

    def _task_outcome(self) -> str:
        """
        Say what became of the task the run script created, if it made one.

        The script gives up on a task that ends in any state but Completed,
        and its exit code alone does not say which state that was.

        Returns:
            A sentence naming the task and its status, to append to the
            error, or an empty string when no task was created.
        """
        try:
            task_id = _read_latest(self.path_to_application, LATEST_TASK_ID_FILE)
        except RuntimeError:
            return ""
        try:
            record = _cli_json(
                ["tasks", "get", "--task-id", task_id, "--format", "json"]
            )
        except (RuntimeError, ValueError):
            return f" Task {task_id} was created."

        return f" Task {task_id} ended with status {record.get('Status')}."

    def check_checkout_matches_remote(self) -> None:
        """
        Refuse to build unless the checkout is the commit origin has.

        SCCE builds what origin has for the branch, and the native passes
        build the checkout, so a benchmark of one commit needs the two to
        agree. The build script would push a checkout that is ahead and
        fail on one that is behind, neither of which is wanted mid-run.

        Raises:
            RuntimeError: If no branch is checked out, the fetch fails, or
                the two commits differ.
        """
        branch = get_git_branch(self.path_to_application)
        if branch is None:
            raise RuntimeError(
                f"{self.path_to_application} has no branch checked out, and "
                "SCCE builds a branch."
            )
        git = ["git", "-C", self.path_to_application]
        fetch = subprocess.run(
            [*git, "fetch", "--quiet", "origin", branch],
            capture_output=True,
            text=True,
            check=False,
        )
        if fetch.returncode != 0:
            raise RuntimeError(
                f"Could not fetch origin/{branch} of {self.path_to_application}."
                f"\n{fetch.stderr.strip()}"
            )
        local, remote = (
            subprocess.check_output([*git, "rev-parse", ref], text=True).strip()
            for ref in ("HEAD", "FETCH_HEAD")
        )
        if local != remote:
            raise RuntimeError(
                f"The checkout of {self.path_to_application} is at "
                f"{local[:7]}, but origin/{branch} is at {remote[:7]}. SCCE "
                "builds what origin has and the native passes build the "
                "checkout, so pull or push first."
            )

    def check_working_tree_is_committed(self) -> None:
        """
        Refuse to build unless the application's code is what its commit has.

        SCCE builds the commit, and the native passes build the working
        tree, so an edit to the code, a file not yet added, or a submodule
        that is not at its recorded commit would benchmark two different
        programs. Only the paths that hold code are checked, see
        ``CHECKED_PATHS``, and the files the benchmark itself writes there
        are allowed, see ``GENERATED_ARTIFACTS``.

        Raises:
            RuntimeError: If git cannot report the status, or anything
                else differs from the commit.
        """
        # The artifact patterns are handed to git as an excludes file, so
        # they are matched as a .gitignore would be.
        with tempfile.NamedTemporaryFile("w", suffix=".gitignore") as excludes:
            excludes.write("\n".join(GENERATED_ARTIFACTS) + "\n")
            excludes.flush()
            status = subprocess.run(
                [
                    "git",
                    "-C",
                    self.path_to_application,
                    "-c",
                    f"core.excludesFile={excludes.name}",
                    "status",
                    "--porcelain",
                    "--no-renames",
                    "--untracked-files=all",
                    "-z",
                    "--",
                    *CHECKED_PATHS,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        if status.returncode != 0:
            raise RuntimeError(
                f"Could not read the git status of {self.path_to_application}."
                f"\n{status.stderr.strip()}"
            )
        # An excludes file covers untracked files only. A tracked build
        # resource the timing script overwrote shows as modified, and is the
        # tool's own doing.
        differing = [
            entry
            for entry in status.stdout.split("\0")
            if entry and entry[3:] not in BUILD_RESOURCES
        ]
        if differing:
            shown = ", ".join(differing[:10])
            if len(differing) > 10:
                shown += f" and {len(differing) - 10} more"
            raise RuntimeError(
                f"The working tree of {self.path_to_application} differs from "
                f"its commit in: {shown}. SCCE builds the commit and the native "
                "passes build the working tree, so commit, stash or remove "
                "these first. A submodule listed here is not at its recorded "
                "commit, which `git submodule update` puts back."
            )

    def _command_line_arguments(self, variable: BenchmarkingVariable) -> str:
        """
        Build the arguments a variable is run with, before the JSON flag.

        This is what the timing file records and the measurement loader
        matches on.

        Args:
            variable: The variable.

        Returns:
            The run-wide arguments followed by the variable's own.
        """
        return full_command_line_arguments(self.demo_cli_args, variable.cla)

    def trace_variable(
        self,
        *,
        variable: BenchmarkingVariable,
        tracing_db_path: str,
        representation_types: list[str],
        representation_sizes: list[int],
        correlations: list[str],
    ) -> None:
        """
        Run one variable on every configuration of its matrix and store
        the results.

        Each task's Ux strings go into the tracing database, and its
        timing is kept for ``write_uxhw_timings``.

        Args:
            variable: The variable to trace.
            tracing_db_path: The tracing database to write to.
            representation_types: Run-wide types. The variable's own
                apply when it declares them.
            representation_sizes: Run-wide sizes, likewise.
            correlations: Run-wide correlation modes, likewise.

        Raises:
            RuntimeError: If a task record has no processor time.
            ValueError: If a task printed no Ux string for the variable, or
                more than one value for it.
        """
        types, sizes, modes = variable.representation_matrix(
            representation_types=representation_types,
            representation_sizes=representation_sizes,
            correlations=correlations,
        )
        task_arguments = f"{self.demo_cli_args} {variable.cla} {TASK_FLAGS}".strip()
        measurements: list[_ScceMeasurement] = []
        for representation_type in types:
            for representation_size in sizes:
                for correlation in modes:
                    configuration = configuration_string(
                        representation_type=representation_type,
                        representation_size=representation_size,
                        correlation=correlation,
                    )
                    if configuration in variable.skipped_configurations:
                        continue
                    core = self._core(
                        representation_type=representation_type,
                        representation_size=representation_size,
                        correlation=correlation,
                    )
                    build_id = self._build_id(core["CoreID"])
                    print(f"Running {variable.name} as {configuration} on SCCE")
                    task = self._run_task(build_id=build_id, arguments=task_arguments)
                    measurements.append(
                        self._record_task(
                            task,
                            variable=variable,
                            tracing_db_path=tracing_db_path,
                            configuration=configuration,
                            core_name=str(core.get("Name")),
                            task_arguments=task_arguments,
                            representation_type=representation_type,
                            representation_size=representation_size,
                            correlation=correlation,
                        )
                    )
        self.measurements[variable.name] = measurements

    def _record_task(
        self,
        task: _ScceTask,
        *,
        variable: BenchmarkingVariable,
        tracing_db_path: str,
        configuration: str,
        core_name: str,
        task_arguments: str,
        representation_type: str,
        representation_size: int,
        correlation: str,
    ) -> _ScceMeasurement:
        """
        Store a task's Ux strings in the tracing database and read its
        timing.

        Args:
            task: The task that ran.
            variable: The variable it ran for.
            tracing_db_path: The tracing database to write to.
            configuration: The configuration it ran as, for example
                ``Athens-16``.
            core_name: The core it ran on, recorded with the run.
            task_arguments: The arguments it was given, recorded with the
                run.
            representation_type: The configuration's representation type.
            representation_size: Its size.
            correlation: Its correlation tracking.

        Returns:
            The task's timing.

        Raises:
            RuntimeError: If the task record has no processor time.
            ValueError: If the task printed no Ux string for the variable,
                or more than one value for it, which a single-shot run does
                not.
        """
        output = _parse_task_stdout(task.stdout)
        values: list[float | DistributionalValue] = []
        for ux_string in _ux_strings_for(output, variable.name):
            value = DistributionalValue.parse(ux_string)
            if value is None:
                raise ValueError(
                    f"Task {task.task_id} printed {ux_string!r} for "
                    f"{variable.name}, which is not a Ux string."
                )
            values.append(value)
        if len(values) > 1:
            # A single-shot run prints one value per output. More means the
            # task was something else, such as a Monte Carlo run, whose
            # samples are not a UxHw result.
            raise ValueError(
                f"Task {task.task_id} printed {len(values)} values for "
                f"{variable.name}. A UxHw run prints one. Check the task's "
                f"arguments, {task_arguments!r}."
            )
        line_number = int(variable.line_number)
        results_database.record_inference(
            tracing_db_path,
            values,
            model=self.application_name,
            variant=core_name,
            device_path=task.task_id,
            inputs=task_arguments.split(),
            output_names=[variable.name] * len(values),
            declaration=(variable.program, variable.path, line_number, line_number),
            ur_type=representation_type,
            ur_order=representation_size,
            correlation_tracking=correlation,
        )
        stats = task.stats
        processor_time = stats.get("ProcessorTime")
        if processor_time is None:
            raise RuntimeError(
                f"The record of task {task.task_id} for {variable.name} has "
                "no ProcessorTime, so the run cannot be timed."
            )
        milliseconds = stats.get("ExecutionTimeInMilliseconds")

        return _ScceMeasurement(
            configuration=configuration,
            time=float(processor_time),
            e2e_time=None if milliseconds is None else milliseconds / 1000,
            dyn_inst_count=stats.get("DynamicInstructions"),
        )

    def write_uxhw_timings(
        self,
        *,
        intermediate_timings_path: str,
        benchmarking_variables: list[BenchmarkingVariable],
    ) -> None:
        """
        Write the tasks' timing as one run per variable, in the format
        the measurement loader reads.

        This is the first writer of the intermediate file in a run on
        SCCE, so it starts the file. The native timing pass that follows
        is told to append.

        Args:
            intermediate_timings_path: The intermediate timing file.
            benchmarking_variables: The variables ``trace_variable`` ran.
        """
        with open(intermediate_timings_path, "w", encoding="utf-8") as handle:
            for variable in benchmarking_variables:
                _write_timing_run(
                    handle,
                    application_name=self.application_name,
                    application_version=self.application_version,
                    command_line_arguments=self._command_line_arguments(variable),
                    measurements=self.measurements[variable.name],
                )
        os.environ[TimingFormat.APPEND_TO_OUTPUT_FILE_ENV_VAR] = "1"


def generate_uxhw_tracing_database_on_scce(
    *,
    runner: ScceRunner,
    benchmarking_variables: list[BenchmarkingVariable],
    tracing_db_path: str,
    representation_types: list[str],
    representation_sizes: list[int],
    correlations: list[str],
) -> None:
    """
    Generate the UxHw tracing database by running every variable on SCCE.

    Args:
        runner: The runner for the application.
        benchmarking_variables: Variables to trace.
        tracing_db_path: The tracing database. Removed first, so stale
            rows do not bleed into the new results.
        representation_types: Run-wide types, from ``-u``.
        representation_sizes: Run-wide sizes, from ``-s``.
        correlations: Run-wide correlation modes, from ``-c``.
    """
    print("Generating UxHw tracing database on SCCE.")
    if os.path.isfile(tracing_db_path):
        os.remove(tracing_db_path)
    runner.login()
    for variable in benchmarking_variables:
        runner.trace_variable(
            variable=variable,
            tracing_db_path=tracing_db_path,
            representation_types=representation_types,
            representation_sizes=representation_sizes,
            correlations=correlations,
        )
