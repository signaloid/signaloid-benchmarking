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
import shlex
import subprocess
import sys

from signaloid_benchmarking.types import BenchmarkingVariable
from signaloid_benchmarking.config import (
    EquivMC,
    RepresentationTypes,
    TimingFormat,
    configuration_string,
    get_resources_dir,
    get_timing_script,
)

# A make target added to the real build makefile. It prints the runtime
# library a configuration links, so the answer comes from the SDK and the
# application's config.mk exactly as the build sees them, including the SDK's
# own choice of library directory for this machine's CPU. It is read as a
# second makefile from standard input, because GNU Make 3.81, which macOS
# ships, has no --eval.
_RUNTIME_LIBRARY_TARGET = (
    "print-runtime-library: ; @echo $(UXHW_SDK_RUNTIME_LIB)/lib$(LIBUXHW_RUNTIME).a\n"
)


def runtime_library_path(
    *,
    path_to_application: str,
    path_to_uxhw_sdk: str,
    representation_type: str,
    representation_size: int,
    correlation: str,
) -> str:
    """
    Ask the build makefile which UxHw runtime library a configuration links.

    The SDK decides the library's directory and name, from the CPU it
    builds on and the precision it is set to. Asking it, rather than
    repeating its rules here, keeps the answer right across SDK versions.

    Args:
        path_to_application: Application root. The makefile runs in its
            ``src`` directory, as the build does, so it reads the same
            ``config.mk``.
        path_to_uxhw_sdk: Path to the UxHw SDK.
        representation_type: For example ``Europa``.
        representation_size: For example ``8``.
        correlation: ``Disabled`` or ``Autocorrelation``.

    Returns:
        The absolute path of the library. It need not exist.

    Raises:
        RuntimeError: If make cannot evaluate the build makefile.
    """
    # Drop inherited make flags, as get-timings.sh does before the real
    # build. An inherited -n, for example, would print the echo command
    # instead of the library.
    environment = {**os.environ}
    environment.pop("MAKEFLAGS", None)
    result = subprocess.run(
        [
            "make",
            "-s",
            "-f",
            str(get_resources_dir() / "C0Pro" / "Makefile.pro"),
            "-f",
            "-",
            "print-runtime-library",
            f"PATH_TO_UXHW_SDK={path_to_uxhw_sdk}",
            f"REPRESENTATION_TYPE={representation_type}",
            f"REPRESENTATION_SIZE={representation_size}",
            f"CORRELATION_TRACKING={correlation}",
        ],
        input=_RUNTIME_LIBRARY_TARGET,
        cwd=os.path.join(path_to_application, "src"),
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    # The makefile also reports on stdout whether it found config.mk, so
    # the library is the last line.
    lines = result.stdout.strip().splitlines()
    if result.returncode != 0 or not lines:
        configuration = configuration_string(
            representation_type=representation_type,
            representation_size=representation_size,
            correlation=correlation,
        )
        raise RuntimeError(
            "Could not ask the UxHw SDK which runtime library "
            f"{configuration} links. make exited with {result.returncode}. "
            f"{result.stderr.strip()}"
        )

    return lines[-1]


def jupiter_size_skip_reason(
    *,
    representation_type: str,
    representation_size: int,
    max_jupiter_size: int,
) -> str | None:
    """
    Say why a Jupiter configuration cannot be benchmarked, if it cannot.

    A Jupiter representation larger than ``--max-jupiter-size`` is refused
    as a memory guard, wherever it would run.

    Args:
        representation_type: For example ``Jupiter``.
        representation_size: For example ``64``.
        max_jupiter_size: The largest Jupiter size to benchmark.

    Returns:
        The reason, or ``None`` when the size is allowed.
    """
    if (
        representation_type == RepresentationTypes.JUPITER
        and representation_size > max_jupiter_size
    ):
        return f"it is above --max-jupiter-size {max_jupiter_size}, a memory guard"

    return None


def configuration_skip_reason(
    *,
    path_to_application: str,
    path_to_uxhw_sdk: str,
    representation_type: str,
    representation_size: int,
    correlation: str,
    max_jupiter_size: int,
) -> str | None:
    """
    Say why a configuration cannot be benchmarked, if it cannot.

    Two things stop one. The SDK may ship no runtime library for it, as SDK
    6.6.0 ships none for Europa or Jupiter with autocorrelation tracking.
    And a Jupiter representation larger than ``--max-jupiter-size`` is
    refused as a memory guard.

    Args:
        path_to_application: Application root.
        path_to_uxhw_sdk: Path to the UxHw SDK.
        representation_type: For example ``Europa``.
        representation_size: For example ``8``.
        correlation: ``Disabled`` or ``Autocorrelation``.
        max_jupiter_size: The largest Jupiter size to benchmark.

    Returns:
        The reason, or ``None`` when the configuration can run.

    Raises:
        RuntimeError: If make cannot evaluate the build makefile.
    """
    reason = jupiter_size_skip_reason(
        representation_type=representation_type,
        representation_size=representation_size,
        max_jupiter_size=max_jupiter_size,
    )
    if reason is not None:
        return reason

    library = runtime_library_path(
        path_to_application=path_to_application,
        path_to_uxhw_sdk=path_to_uxhw_sdk,
        representation_type=representation_type,
        representation_size=representation_size,
        correlation=correlation,
    )
    if not os.path.isfile(library):
        return f"the UxHw SDK has no {os.path.basename(library)}"

    return None


def export_timing_env(
    *,
    path_to_uxhw_sdk: str,
    measure_dynamic_instructions: bool,
    path_to_application: str,
    application_name: str,
    application_version: str,
    results_dir: str,
    logs_dir: str,
    tracing_db_path: str,
) -> str:
    """
    Export the environment variables needed by the timing bash scripts.

    Called after ``get_application_info()`` so all paths and config are
    available. ``TRACING_DB_ABS`` is preserved when already set. Otherwise
    ``tracing_db_path`` is used and exported.

    Args:
        path_to_uxhw_sdk: Path to the UxHw SDK.
        measure_dynamic_instructions: Whether to measure dynamic instruction
            counts with Intel PIN. Off by default. When True the kit is
            taken from ``PIN_ROOT``, which must already be set. When False
            ``PIN_ROOT`` is removed from the environment, so an inherited
            value cannot switch the measurement on behind the caller's back.
        path_to_application: Path to the application source tree.
        application_name: Application name (e.g. ``Finance-...``).
        application_version: Application version string.
        results_dir: Directory where results artifacts are written.
        logs_dir: Directory where logs are written.
        tracing_db_path: Default tracing-database path. Overridden when
            ``TRACING_DB_ABS`` is already set.

    Returns:
        The resolved tracing-database path (also written to
        ``TRACING_DB_ABS``). Callers should store it back to
        ``self.tracing_db_path``.
    """
    os.environ["BENCHMARKING_RESOURCES_DIR"] = str(get_resources_dir())
    os.environ["PATH_TO_UXHW_SDK"] = path_to_uxhw_sdk
    # The flag is the single switch for the dynamic instruction count.
    # Without it we drop PIN_ROOT, so a kit exported in the caller's shell
    # profile cannot silently turn the measurement on. get-timings.sh reads
    # the same variable and skips the count when it is absent.
    if measure_dynamic_instructions:
        pin_root = os.environ.get("PIN_ROOT", "")
        if not pin_root:
            raise ValueError(
                "--measure-dynamic-instructions needs an Intel PIN kit, but "
                "PIN_ROOT is not set. Export it to the kit directory, for "
                "example 'export PIN_ROOT=~/pin-external-4.2', or drop the "
                "flag to skip the dynamic instruction count."
            )
        os.environ["PIN_ROOT"] = os.path.expanduser(pin_root)
    else:
        os.environ.pop("PIN_ROOT", None)
    # The timing bash layer shells out to `python3 -m
    # signaloid_benchmarking...`. We pass our own interpreter so those
    # subprocesses use this venv (with the benchmarking dependencies) even when
    # the venv is not on PATH / not activated.
    os.environ["BENCHMARKING_PYTHON"] = sys.executable
    os.environ["APPLICATION_PATH"] = path_to_application
    os.environ["APPLICATION_NAME"] = application_name
    os.environ["APPLICATION_VERSION"] = application_version
    os.environ["PROGRAM"] = "main"
    os.environ["CLA_FOR_MULTIPLE_EXECUTIONS"] = "-M"
    os.environ[TimingFormat.APPEND_TO_OUTPUT_FILE_ENV_VAR] = "0"
    os.environ["RESULTS_DIR"] = results_dir
    os.environ["LOGS_DIR"] = logs_dir
    resolved_tracing_db_path = os.environ.get("TRACING_DB_ABS", tracing_db_path)
    os.environ["TRACING_DB_ABS"] = resolved_tracing_db_path
    return resolved_tracing_db_path


def run_timing_script(
    *,
    benchmarking_variables: list[BenchmarkingVariable],
    demo_cli_args: str,
    representation_types: list[str],
    representation_sizes: list[int],
    correlations: list[str],
    logs_dir: str,
    intermediate_timings_path: str,
    variable_index: int,
    timing: bool = False,
    tracing: bool = False,
    native_mc_timing: bool = False,
) -> None:
    """
    Run one per-variable pass of the timing bash script.

    Exactly one of ``timing``, ``tracing`` and ``native_mc_timing`` is set
    on each call, and the pass covers the variable at ``variable_index``.

    A UxHw run that dies in a tracing or timing pass does not fail it. The
    script records the configuration with the reason, and it is added to
    the variable's skipped configurations here, so later passes and the
    loader leave it out.

    Args:
        benchmarking_variables: Variables to run timing against.
        demo_cli_args: Demo-level command-line arguments string.
        representation_types: Run-wide representation types. A
            per-variable pass uses the variable's own when it has them.
        representation_sizes: Run-wide representation sizes, likewise.
        correlations: Run-wide correlation modes, likewise.
        logs_dir: Directory where the bash stderr log is written.
        intermediate_timings_path: Absolute path of the transient
            intermediate timings file the bash script writes.
        timing: When ``True``, run the per-variable UxHw-timing pass.
        tracing: When ``True``, run the per-variable UxHw-tracing
            pass.
        native_mc_timing: When ``True``, run the per-variable native-MC
            timing pass.
        variable_index: Index into ``benchmarking_variables`` of the
            variable this pass covers.

    Raises:
        RuntimeError: If the bash timing script exits with a non-zero
            status code. The captured stderr (from
            ``timing_script_stderr.log``) is included in the message.
    """
    variable = benchmarking_variables[variable_index]
    skip_uxhw = 0 if timing else 1
    skip_tracing = 0 if tracing else 1
    skip_native_mc = 0 if native_mc_timing else 1

    command_line_arguments = variable.cla
    if timing:
        command_line_arguments = "-T " + command_line_arguments

    # Set mode-specific scalar env vars
    if demo_cli_args:
        command_line_arguments = f"{demo_cli_args} {command_line_arguments}".strip()
    os.environ["CLA"] = command_line_arguments
    os.environ["SKIP_UXHW"] = str(skip_uxhw)
    os.environ["SKIP_UXHW_TRACING"] = str(skip_tracing)
    os.environ["SKIP_NATIVE_MC"] = str(skip_native_mc)

    # Wire the timing-intermediate tokens and file path through to
    # the bash script so its emitted tags match what the Python
    # reader expects (see TimingFormat in config.py).
    os.environ[TimingFormat.META_TAG_ENV_VAR] = TimingFormat.META_TAG
    os.environ[TimingFormat.MEASUREMENT_TAG_ENV_VAR] = TimingFormat.MEASUREMENT_TAG
    os.environ[TimingFormat.SAMPLE_TAG_ENV_VAR] = TimingFormat.SAMPLE_TAG
    os.environ[TimingFormat.INTERMEDIATE_FILE_ENV_VAR] = intermediate_timings_path

    # A record left by an earlier pass must not be read as this one's.
    failed_configurations_path = os.path.join(
        logs_dir, TimingFormat.FAILED_CONFIGURATIONS_FILENAME
    )
    if os.path.isfile(failed_configurations_path):
        os.remove(failed_configurations_path)
    os.environ[TimingFormat.FAILED_CONFIGURATIONS_FILE_ENV_VAR] = (
        failed_configurations_path
    )

    # A per-variable pass measures that variable on its own matrix, which
    # is the run-wide one unless its signaloid.yaml entry declares one.
    # Which configurations of it cannot run was decided in Python and is
    # handed over, so the timing layer and the loader agree on them.
    (
        representation_types,
        representation_sizes,
        correlations,
    ) = variable.representation_matrix(
        representation_types=representation_types,
        representation_sizes=representation_sizes,
        correlations=correlations,
    )
    skipped_configurations = variable.skipped_configurations

    # Build bash arrays (cannot be env vars) and source the script
    representations = " ".join(map(str, representation_types))
    rep_sizes = " ".join(map(str, representation_sizes))
    correlations_str = " ".join(map(str, correlations))
    skipped_str = " ".join(map(shlex.quote, skipped_configurations))

    traces_lines = (
        "    'addDistValueTrace "
        + f'{variable.name}   "{variable.file_name}:{variable.line_number}"'
        + "'\n"
    )
    # Before the EMCC is known there are no sizes, so the pass gets one.
    native_mc_sizes = " ".join(map(str, variable.emcc_results.equiv_mc_list)) or "50"

    quoted_timing_script = shlex.quote(str(get_timing_script()))

    bash_cmd = f"""
TRACES=(
{traces_lines})
REFERENCE_PRECISIONS=({native_mc_sizes})
REPRESENTATION_TYPES=({representations})
REPRESENTATION_SIZES=({rep_sizes})
CORRELATION_TRACKING_TYPES=({correlations_str})
SKIPPED_CONFIGURATIONS=({skipped_str})
. {quoted_timing_script}
"""
    stderr_log = os.path.join(logs_dir, "timing_script_stderr.log")
    # Use tee so stderr streams to terminal
    # (preserving interactive prompts) and is
    # also captured to a log file for debugging.
    quoted_stderr_log = shlex.quote(stderr_log)
    wrapped_cmd = f"{{ {bash_cmd.strip()} ; }} " f"2> >(tee {quoted_stderr_log} >&2)"
    result = subprocess.call(["bash", "-c", wrapped_cmd])
    # After the first timing script call, switch to append mode
    os.environ[TimingFormat.APPEND_TO_OUTPUT_FILE_ENV_VAR] = "1"
    if result != 0:
        stderr_content = ""
        if os.path.isfile(stderr_log):
            with open(stderr_log, "r") as f:
                stderr_content = f.read().strip()
        error_message = "Timing script failed " f"(exit code {result})."
        if stderr_content:
            error_message += "\nstderr:\n" + stderr_content
        error_message += (
            "\nSee also:\n" f"  - {logs_dir}/exec.stderr\n" f"  - {logs_dir}/opt.err"
        )
        raise RuntimeError(error_message)

    if tracing or timing:
        skip_failed_configurations(
            variable=benchmarking_variables[variable_index],
            failed_configurations_path=failed_configurations_path,
            representation_types=representation_types,
            representation_sizes=representation_sizes,
            correlations=correlations,
        )


def skip_failed_configurations(
    *,
    variable: BenchmarkingVariable,
    failed_configurations_path: str,
    representation_types: list[str],
    representation_sizes: list[int],
    correlations: list[str],
) -> None:
    """
    Skip the configurations whose UxHw run died in the last pass.

    get-timings.sh writes one ``<config>\\t<reason>`` line per run that
    died. Each is added to the variable's skipped configurations and
    printed like the skips decided before any build. The file is removed
    once read, so it never holds more than one pass.

    Args:
        variable: The variable the pass measured.
        failed_configurations_path: The file the script wrote. Absent
            when no run died.
        representation_types: Run-wide types, for a variable that
            declares none of its own.
        representation_sizes: Run-wide sizes, likewise.
        correlations: Run-wide correlation modes, likewise.
    """
    if not os.path.isfile(failed_configurations_path):
        return

    with open(failed_configurations_path, "r") as file:
        for line in file:
            configuration, _, reason = line.rstrip("\n").partition("\t")
            if not configuration:
                continue
            variable.skipped_configurations[configuration] = reason
            print(f"Skipping {configuration} for {variable.name}, because {reason}.")
    os.remove(failed_configurations_path)

    if not variable.benchmarked_configurations(
        representation_types=representation_types,
        representation_sizes=representation_sizes,
        correlations=correlations,
    ):
        print(
            f"Every configuration of {variable.name} is skipped, so nothing "
            "more is measured for it."
        )


def compile_native(
    *,
    path_to_application: str,
    num_parallel_workers: int,
) -> tuple[str, str, str, bool]:
    """
    Compile the application for native execution MC.

    Tries the following in order:

    1.  If a Makefile exists at the application root,
        run ``make local-build``.
    2.  Fall back to building a gcc command from
        config.mk.

    Args:
        path_to_application: Path to the application source tree.
        num_parallel_workers: Number of parallel ``make -j`` workers.

    Returns:
        Tuple ``(native_executable_name, native_compilation_command,
        native_executable_dir, has_native_mc)``. The caller should
        store these back to the corresponding ``Benchmark`` attributes.
        When the gcc-fallback path returns early (no ``SOURCES`` in
        ``config.mk``), the tuple is ``("", "", "", False)``.
    """
    makefile_path = os.path.join(path_to_application, "Makefile")

    use_makefile = os.path.isfile(makefile_path)
    if use_makefile:
        print(
            "Makefile detected. Attempting to compile native executable using 'make local-build'."
        )
        native_executable_name = "demo-native-mc"
        native_compilation_command = f"make -j{num_parallel_workers} local-build"
    else:
        print(
            "No Makefile detected. Attempting to compile native executable using gcc and config.mk."
        )
        config_path = f"{path_to_application}" "/src/config.mk"

        # Use make to expand variables (handles both plain lists
        # and Make functions like $(wildcard ...), $(filter-out ...), etc.)
        def _make_expand(var: str) -> str:
            result = subprocess.run(
                [
                    "make",
                    "-f",
                    config_path,
                    "-f",
                    "-",
                    f"print-{var}",
                ],
                capture_output=True,
                text=True,
                cwd=f"{path_to_application}/src",
                input=f"print-{var}:\n\t@echo $({var})\n",
            )
            # Empty stdout means "unset" only from a *successful* make (e.g. no
            # SOURCES, hence no native MC). A non-zero return (e.g. missing
            # config.mk) also yields empty stdout and must not be read as unset.
            if result.returncode != 0:
                raise RuntimeError(f"make failed: {result.stderr}")
            return result.stdout.strip()

        sources_str = _make_expand("SOURCES")
        if not sources_str:
            return ("", "", "", False)
        sources = sources_str.split()
        sources.append("uxhw.c")

        cflags_str = _make_expand("CFLAGS")
        cflags = cflags_str.split() if cflags_str else []

        # config.mk's -I/-L paths are written relative to src/ (like SOURCES),
        # but we compile from the application root, so rebase relative include
        # and library search paths onto src/ to match the src/-prefixed sources
        # below. Absolute paths (e.g. -I/opt/local/include) are left untouched.
        def _rebase_search_path(flag: str) -> str:
            for opt in ("-I", "-L"):
                if flag.startswith(opt):
                    path = flag[len(opt) :]
                    if path and not os.path.isabs(path):
                        return f"{opt}src/{path}"
            return flag

        cflags = [_rebase_search_path(f) for f in cflags]

        native_executable_name = "demo-native-mc"
        sources = [f"src/{s}" for s in sources]
        parts = (
            [
                "gcc",
                "-O3",
                "-o",
                native_executable_name,
                "-Isrc",
                "-I/opt/local/include",
            ]
            + cflags
            + sources
            + [
                "-L/opt/local/lib",
                "-lgsl",
                "-lgslcblas",
                "-lm",
            ]
        )
        native_compilation_command = " ".join(parts)

    # Symlink input files into the application root, skipping any that already
    # exist as non-symlinks (don't overwrite e.g. README.md). Also skip the
    # binary's own data.out: symlinking a stale one would let a run there write
    # back through it to the shared inputs/data.out (see
    # benchmarking_utils._symlink_application_inputs).
    subprocess.call(
        # ``inputs/*`` is relative to cwd (this call's ``cwd=path_to_application``).
        # Prefixing path_to_application would double it.
        f"for f in inputs/*; do "
        f'b="$(basename "$f")"; '
        f'if [ "$b" = "{EquivMC.MC_OUTPUT_FILENAME}" ]; then continue; fi; '
        f'if [ -e "$b" ] && [ ! -L "$b" ]; then '
        f'echo "WARNING: skipping symlink for $b (file already exists in application root)"; '
        f"continue; fi; "
        f'ln -sf "$f" .; done',
        shell=True,
        cwd=path_to_application,
        stdout=None,
        stderr=subprocess.DEVNULL,
    )

    # Always compile from the application root
    native_executable_dir = path_to_application

    # Clear any inherited jobserver flags to avoid
    # "warning: jobserver unavailable: using -j1" in sub-makes.
    clean_env = {**os.environ}
    clean_env.pop("MAKEFLAGS", None)

    print(
        f"Compiling native executable {native_executable_name} at {native_executable_dir}"
    )
    if use_makefile:
        clean_result = subprocess.run(
            f"make -j{num_parallel_workers} clean",
            shell=True,
            cwd=native_executable_dir,
            capture_output=True,
            text=True,
            env=clean_env,
        )
        if clean_result.returncode != 0:
            print(
                f"Warning: 'make clean' failed (exit code {clean_result.returncode}). "
                f"Continuing with build."
            )
    result = subprocess.run(
        native_compilation_command,
        shell=True,
        cwd=native_executable_dir,
        capture_output=True,
        text=True,
        env=clean_env,
    )
    if result.returncode != 0:
        print("Native compilation failed " f"(exit code {result.returncode}).")
        print(result.stdout)
        print(result.stderr)
        has_native_mc = False
    else:
        # Verify the executable was produced and resolve its location
        root_path = os.path.join(path_to_application, native_executable_name)
        src_path = os.path.join(path_to_application, "src", native_executable_name)
        if os.path.isfile(root_path):
            native_executable_dir = path_to_application
        elif os.path.isfile(src_path):
            native_executable_dir = os.path.join(path_to_application, "src")
        else:
            print(
                f"Warning: Compilation succeeded but '{native_executable_name}' "
                f"not found in {path_to_application} or {path_to_application}/src."
            )
            has_native_mc = False
            return (
                native_executable_name,
                native_compilation_command,
                native_executable_dir,
                has_native_mc,
            )
        has_native_mc = True
        print(
            f"Native compilation succeeded. "
            f"Executable: {native_executable_dir}/{native_executable_name}"
        )
    return (
        native_executable_name,
        native_compilation_command,
        native_executable_dir,
        has_native_mc,
    )
