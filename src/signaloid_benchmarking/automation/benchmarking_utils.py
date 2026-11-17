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

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Iterator
import difflib
import subprocess
import math
import os
import shlex
import pandas as pd
from signaloid_benchmarking.config import (
    BenchmarkingVariables,
    EquivMC,
    SignaloidYaml,
    TimingFormat,
    VariableTypes,
)
from signaloid.distributional.distributional import DistributionalValue
from signaloid_benchmarking.types import (
    BenchmarkingVariable,
    TaggedDistributionalValue,
)
import re
import numpy as np
import tempfile
import csv

_SAMPLES_KEY = "_samples"


def _symlink_application_inputs(inputs_dir: str, dest_dir: str) -> None:
    """
    Symlink an application's ``inputs/`` files into a run directory.

    The native binary writes its samples to a *relative* ``data.out`` in
    its working directory. A stale ``data.out`` left in ``inputs/`` (the
    bash timing layer runs the binary there) must never be symlinked in:
    the binary's ``fopen("data.out", "w")`` would follow the symlink and
    write *through* it to the single shared file, so concurrent workers
    (``-j > 1``) would clobber one another and different variables would
    end up with identical samples.

    Args:
        inputs_dir: The application's ``inputs/`` directory.
        dest_dir: The per-run working directory to populate with symlinks.
    """
    # Resolve to absolute paths so the symlink targets are valid even when
    # a relative application path was provided (os.symlink stores the
    # target verbatim, so a relative target would resolve against dest_dir).
    inputs_dir = os.path.abspath(inputs_dir)
    dest_dir = os.path.abspath(dest_dir)
    if not os.path.isdir(inputs_dir):
        return
    for entry in os.listdir(inputs_dir):
        if entry == EquivMC.MC_OUTPUT_FILENAME:
            # Never expose the binary's output file as an input.
            continue
        os.symlink(
            os.path.join(inputs_dir, entry),
            os.path.join(dest_dir, entry),
        )


def _read_pin_inst_count(path: str) -> float:
    """
    Read an inscount.out file produced by Intel PIN and return the count.

    Args:
        path: Path to the inscount.out file written by PIN's inscount0.so
            tool. Both absolute and relative paths are accepted. Relative
            paths are resolved against the current working directory. The
            file contains a single line of the form ``Count <integer>``.

    Returns:
        The instruction count as a float.

    Raises:
        ValueError: If the file content does not match the expected format.
        OSError: If the file cannot be opened.
    """
    with open(path) as f:
        content = f.read().strip()
    if not content.startswith("Count "):
        raise ValueError(
            f"{path} should start with 'Count ', but it reads {content!r}."
        )
    return float(content.removeprefix("Count "))


def _resolve_pin_inst_from_samples(
    samples: dict[str, list[str]],
) -> float | None:
    """
    Average per-iteration PIN instruction counts from collected sample files.

    Each ``samples`` value is a list of file paths whose counts are summed to
    give that iteration's total. The per-iteration totals are then averaged.

    Args:
        samples: Mapping from iteration index to a list of inscount.out
            paths whose counts are summed for that iteration.

    Returns:
        The averaged instruction count, or ``None`` if ``samples`` is empty.
    """
    if not samples:
        return None
    per_iteration_totals: list[float] = []
    for paths in samples.values():
        iteration_total = sum(_read_pin_inst_count(p) for p in paths)
        per_iteration_totals.append(iteration_total)
    return sum(per_iteration_totals) / len(per_iteration_totals)


def _resolve_time_from_samples(
    samples: dict[str, list[str]],
) -> float | None:
    """
    Average per-iteration float values from collected SAMPLE values.

    Like :func:`_resolve_pin_inst_from_samples` but for value-typed SAMPLE
    entries, where each value is a literal float string rather than a file path.
    Values are summed within an iteration, then averaged across iterations.
    Being field-agnostic, it serves the elapsedTime, databaseTime, and
    databaseDynInstCount fields uniformly.

    Args:
        samples: Mapping from iteration index to a list of literal
            float-string values emitted by the bash side as SAMPLE values for a
            value-typed field.

    Returns:
        The averaged value, or ``None`` if ``samples`` is empty or every value
        is non-numeric.
    """
    if not samples:
        return None
    per_iteration_totals: list[float] = []
    for tokens in samples.values():
        valid: list[float] = []
        for value_str in tokens:
            try:
                valid.append(float(value_str))
            except ValueError:
                continue
        if not valid:
            continue
        per_iteration_totals.append(sum(valid))
    if not per_iteration_totals:
        return None
    return sum(per_iteration_totals) / len(per_iteration_totals)


def _measurement_value(
    token: str,
    samples: dict[tuple[str, str], dict[str, list[str]]],
    sample_key: tuple[str, str],
    resolve: Callable[[dict[str, list[str]]], float | None],
) -> float | None:
    """
    Read one field of a MEASUREMENT line.

    A ``?`` sentinel is resolved from the SAMPLE lines of the same
    (config, field), which are then taken out of the run.

    Args:
        token: The field as written on the MEASUREMENT line.
        samples: The run's SAMPLE tokens, keyed by (config, field).
        sample_key: The (config, field) of this field.
        resolve: Turns the field's SAMPLE tokens into one value.

    Returns:
        The value, or ``None`` when neither the line nor the samples hold one.
    """
    if token != TimingFormat.MISSING_VALUE:
        return float(token)
    return resolve(samples.pop(sample_key, {}))


def parse_timing_intermediate_stream(lines: Iterable[str]) -> list[dict]:
    """
    Parse the tagged line stream written by get-timings.sh into a list
    of per-run dicts.

    A new run begins whenever a `META timestamp ...` line is seen. All
    subsequent META lines attach to that run, SAMPLE lines accumulate
    per-iteration tokens keyed by (config, field), and MEASUREMENT
    lines append to the run's `measurements` list. When a MEASUREMENT
    line arrives, any field whose value is the ``?`` sentinel is
    resolved from the accumulated SAMPLE data for the matching
    (config, field): path-typed samples (e.g. ``pinDynInstCount``) are
    resolved by reading and summing the referenced inscount.out files
    per iteration before averaging. Value-typed samples (e.g.
    ``elapsedTime``, ``databaseTime``, ``databaseDynInstCount``) are
    float-parsed in place and averaged the same way. Session-level
    META keys (see
    TimingFormat.SESSION_META_KEYS) are preserved on every run dict
    at this stage and are promoted to the top level by the caller
    that writes the canonical JSON document.

    Args:
        lines: Iterable of raw lines (with or without trailing
            newline). Accepts a file object, a list of strings, or any
            other iterable so the source can be swapped from a file to
            a live pipe without touching this logic.

    Returns:
        A list of run dicts, each carrying the META keys at the top
        level and a `measurements` list of per-config dicts.
    """
    runs: list[dict] = []
    current: dict | None = None
    for raw in lines:
        line = raw.rstrip("\n")
        if not line:
            continue
        tag, _, rest = line.partition(" ")
        if tag == TimingFormat.META_TAG:
            key, _, value = rest.partition(" ")
            if key == TimingFormat.META_KEY_TIMESTAMP and current is not None:
                current.pop(_SAMPLES_KEY, None)
                runs.append(current)
                current = None
            if current is None:
                # A timestamp starts a run. A META line before any
                # timestamp also opens one, so its key is not dropped.
                current = {
                    TimingFormat.JSON_KEY_MEASUREMENTS: [],
                    _SAMPLES_KEY: {},
                }
            current[key] = value
        elif tag == TimingFormat.SAMPLE_TAG:
            if current is None:
                continue
            tokens = rest.split(maxsplit=3)
            if len(tokens) != 4:
                continue
            s_config, s_field, s_iteration, s_token = tokens
            sample_key = (s_config, s_field)
            iteration_map: dict[str, list[str]] = current[_SAMPLES_KEY].setdefault(
                sample_key, {}
            )
            iteration_map.setdefault(s_iteration, []).append(s_token)
        elif tag == TimingFormat.MEASUREMENT_TAG:
            if current is None:
                continue
            tokens = rest.split()
            if len(tokens) != 6:
                continue
            config, t_val, db_t, e2e_t, db_i, pin_i = tokens
            samples = current[_SAMPLES_KEY]
            time_value = _measurement_value(
                t_val,
                samples,
                (config, TimingFormat.SAMPLE_FIELD_TIME),
                _resolve_time_from_samples,
            )
            db_time_value = _measurement_value(
                db_t,
                samples,
                (config, TimingFormat.SAMPLE_FIELD_DB_TIME),
                _resolve_time_from_samples,
            )
            db_inst_value = _measurement_value(
                db_i,
                samples,
                (config, TimingFormat.SAMPLE_FIELD_DB_DYN_INST_COUNT),
                _resolve_time_from_samples,
            )
            pin_inst_value = _measurement_value(
                pin_i,
                samples,
                (config, TimingFormat.SAMPLE_FIELD_PIN_INST),
                _resolve_pin_inst_from_samples,
            )
            e2e_value = None if e2e_t == TimingFormat.MISSING_VALUE else float(e2e_t)

            current[TimingFormat.JSON_KEY_MEASUREMENTS].append(
                {
                    TimingFormat.JSON_KEY_MEASUREMENT_CONFIG: config,
                    TimingFormat.JSON_KEY_MEASUREMENT_TIME: time_value,
                    TimingFormat.JSON_KEY_MEASUREMENT_DB_TIME: db_time_value,
                    TimingFormat.JSON_KEY_MEASUREMENT_E2E_TIME: e2e_value,
                    TimingFormat.JSON_KEY_MEASUREMENT_DB_DYN_INST_COUNT: (
                        db_inst_value
                    ),
                    TimingFormat.JSON_KEY_MEASUREMENT_PIN_DYN_INST_COUNT: (
                        pin_inst_value
                    ),
                }
            )
    if current is not None:
        current.pop(_SAMPLES_KEY, None)
        runs.append(current)
    return runs


def utc_timestamp() -> str:
    """
    The current UTC time, as every artifact of the benchmark writes it.

    Returns:
        The time as ``YYYY-MM-DDTHH:MM:SSZ``, the form ``date -u`` writes
        into the timing file.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_git_remote(directory: str) -> str | None:
    """
    Return the normalised ``origin`` remote URL of a Git repository.

    Strips the protocol, any embedded token, and the ``.git`` suffix.

    Args:
        directory: Path to the repository to inspect.

    Returns:
        The normalised remote URL, or ``None`` if ``directory`` is not a Git
        repository or has no ``origin`` remote.
    """

    if not os.path.isdir(os.path.join(directory, ".git")):
        print("Not a Git repository")
        return None

    try:
        # Run the 'git remote get-url origin' command
        remote_url = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            cwd=directory,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        # Handle SSH format: git@github.com:user/repo.git
        if "git@" in remote_url:
            remote_url = remote_url.split("git@")[1]
            remote_url = remote_url.replace(":", "/")
        # Handle HTTPS format: https://github.com/user/repo.git
        elif remote_url.startswith("https://"):
            # Remove the protocol
            remote_url = remote_url.replace("https://", "")
            # Remove token if present (TOKEN@github.com -> github.com)
            if "@" in remote_url:
                remote_url = remote_url.split("@", 1)[1]

        # Remove .git suffix if present
        if remote_url.endswith(".git"):
            remote_url = remote_url[:-4]

        print(f"Remote URL: {remote_url}")
        return remote_url

    except subprocess.CalledProcessError:
        print("No remote repository found")
        return None


def get_git_branch(directory: str) -> str | None:
    """
    Return the name of the branch currently checked out in a repository.

    Args:
        directory: Path to the repository to inspect.

    Returns:
        The branch name, or ``None`` if ``directory`` is not a Git
        repository, if Git is unavailable, or if ``HEAD`` is detached.
        A detached ``HEAD`` is common on CI checkouts, which check out a
        bare commit. There is no branch to name in that state, so the
        result is ``None`` rather than the literal string ``"HEAD"`` that
        ``git rev-parse --abbrev-ref HEAD`` would report.
    """

    # A worktree or a submodule has a .git file, not a .git directory.
    if not os.path.exists(os.path.join(directory, ".git")):
        print("Not a Git repository")
        return None

    try:
        # 'git symbolic-ref' exits non-zero on a detached HEAD instead of
        # reporting a placeholder branch name.
        branch = subprocess.check_output(
            ["git", "symbolic-ref", "--short", "HEAD"],
            cwd=directory,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        print(f"Branch: {branch}")
        return branch

    except subprocess.CalledProcessError:
        print("No branch checked out (detached HEAD)")
        return None

    except FileNotFoundError:
        print("Git is not available")
        return None


@contextmanager
def _native_mc_run(
    prefix: str,
    path: str,
    executable: str,
    executable_dir: str,
    cla: str,
    sample_size: int,
) -> Iterator[tuple["subprocess.CompletedProcess[str]", str]]:
    """
    Run the native MC binary in a throwaway working directory.

    Sets up an isolated temp dir, symlinks the application's ``inputs/``. Not
    the binary's own ``data.out`` (see _symlink_application_inputs). Then, runs
    ``<executable> <cla> -M <sample_size>`` there. The temp dir (and the yielded
    file) live only for the ``with`` body.

    Args:
        prefix: Prefix for the temporary directory name.
        path: Application root (its ``inputs/`` is symlinked in).
        executable: Native binary filename.
        executable_dir: Directory containing the native binary.
        cla: Command-line arguments passed to the binary.
        sample_size: Monte Carlo sample count (passed as ``-M``).

    Yields:
        A tuple of (completed process, path to the binary's ``data.out``).
    """
    with tempfile.TemporaryDirectory(prefix=prefix) as work_dir:
        _symlink_application_inputs(os.path.join(path, "inputs"), work_dir)

        argv = [
            os.path.join(executable_dir, executable),
            *shlex.split(cla),
            "-M",
            str(sample_size),
        ]
        result = subprocess.run(argv, cwd=work_dir, capture_output=True, text=True)
        yield result, os.path.join(work_dir, EquivMC.MC_OUTPUT_FILENAME)


def _run_scalar_native(
    path: str,
    executable: str,
    executable_dir: str,
    cla: str,
    sample_size: int,
    run_id: int,
) -> tuple[int, float | None]:
    """
    Run a native MC execution whose output is a scalar.

    Args:
        path: Application root (its ``inputs/`` is symlinked in).
        executable: Native binary filename.
        executable_dir: Directory containing the native binary.
        cla: Command-line arguments passed to the binary.
        sample_size: Monte Carlo sample count.
        run_id: Identifier for this run, used in warnings and the temp-dir name.

    Returns:
        A tuple of (sample_size, scalar value), where the value is ``None`` if
        the run failed or its output could not be parsed.
    """
    value: float | None = None
    with _native_mc_run(
        f"scalar_run_{run_id}_", path, executable, executable_dir, cla, sample_size
    ) as (result, data_file):
        if result.returncode != 0:
            print(
                f"Warning: Scalar run {run_id} (size={sample_size}) returned non-zero "
                f"exit code: {result.stderr}"
            )

        try:
            with open(data_file, "r") as f:
                for line_no, line in enumerate(f):
                    if line_no == 1:
                        value = float(line.strip())
                        break
        except Exception as e:
            print(
                f"Error reading {EquivMC.MC_OUTPUT_FILENAME} in "
                f"{os.path.dirname(data_file)}: {e}"
            )

    return sample_size, value


def clean_data_for_sheets(rows: list[list[Any]]) -> list[list[Any]]:
    """
    Clean row data for Google Sheets API compatibility.

    Renders distribution objects via ``repr``, blanks out NaN cells, and casts
    numpy scalars to plain floats.

    Args:
        rows: Rows of cell values to clean.

    Returns:
        The cleaned rows.
    """
    cleaned_rows: list[list[Any]] = []
    for row in rows:
        cleaned_row: list[Any] = []
        for cell in row:
            if isinstance(cell, (DistributionalValue, TaggedDistributionalValue)):
                cleaned_row.append(repr(cell))
            elif pd.isna(cell):
                cleaned_row.append("")
            elif isinstance(cell, (np.integer, np.floating)):
                cleaned_row.append(float(cell))
            else:
                cleaned_row.append(cell)
        cleaned_rows.append(cleaned_row)
    return cleaned_rows


def _missing_keys(entry: Any, keys: list[str], where: str, problems: list[str]) -> bool:
    """
    Record the keys a signaloid.yaml entry lacks.

    Args:
        entry: The entry as read from the YAML.
        keys: The keys it must have.
        where: How to name the entry in a problem.
        problems: The list each problem is added to.

    Returns:
        True when the entry is not a mapping or lacks a key.
    """
    if not isinstance(entry, dict):
        problems.append(f"{where} is not a set of 'Key: value' lines.")
        return True
    missing = [key for key in keys if key not in entry]
    if missing:
        problems.append(f"{where} has no {', '.join(missing)}.")
        return True

    return False


def check_signaloid_yaml(signaloid_yaml_path: str, signaloid_yaml: Any) -> None:
    """
    Check that signaloid.yaml has every key the run reads.

    A missing key would otherwise stop the run as a bare KeyError, and an
    unknown OutputObject would only fail after the tracing runs. Every
    problem is reported at once.

    Args:
        signaloid_yaml_path: The file, to name in the error.
        signaloid_yaml: Its contents as read by ``yaml.safe_load``.

    Raises:
        ValueError: A key is missing or has a value the run cannot use.
    """
    problems: list[str] = []

    if not isinstance(signaloid_yaml, dict):
        signaloid_yaml = {}
    trace_variables = signaloid_yaml.get(SignaloidYaml.TRACE_VARIABLES)
    if not isinstance(trace_variables, list) or not trace_variables:
        problems.append(f"There is no {SignaloidYaml.TRACE_VARIABLES} list.")
        trace_variables = []
    traced_entries = [
        entry
        for index, entry in enumerate(trace_variables)
        if not _missing_keys(
            entry,
            [SignaloidYaml.FILE, SignaloidYaml.LINE_NUMBER, SignaloidYaml.EXPRESSION],
            f"{SignaloidYaml.TRACE_VARIABLES} entry {index + 1}",
            problems,
        )
    ]

    # Only the newer format names its outputs. The older one benchmarks
    # every traced expression.
    benchmarking_variables = signaloid_yaml.get(SignaloidYaml.BENCHMARKING_VARIABLES)
    if benchmarking_variables is not None:
        all_outputs = signaloid_yaml.get(SignaloidYaml.ALL_OUTPUTS)
        if not isinstance(all_outputs, list) or not all_outputs:
            problems.append(
                f"There is a {SignaloidYaml.BENCHMARKING_VARIABLES} list but "
                f"no {SignaloidYaml.ALL_OUTPUTS} list. The tracing runs need "
                "the arguments that compute every output at once."
            )
        else:
            _missing_keys(
                all_outputs[0],
                [SignaloidYaml.COMMAND_LINE_ARGUMENTS],
                f"The first {SignaloidYaml.ALL_OUTPUTS} entry",
                problems,
            )
        if not isinstance(benchmarking_variables, list):
            problems.append(f"{SignaloidYaml.BENCHMARKING_VARIABLES} is not a list.")
            benchmarking_variables = []

        traced = [
            entry[SignaloidYaml.EXPRESSION]
            for entry in expand_array_expressions(traced_entries)
        ]
        output_objects = [VariableTypes.DISTRIBUTION, VariableTypes.SCALAR]
        for index, entry in enumerate(benchmarking_variables):
            where = f"{SignaloidYaml.BENCHMARKING_VARIABLES} entry {index + 1}"
            if _missing_keys(
                entry,
                [
                    SignaloidYaml.VARIABLE_NAME,
                    SignaloidYaml.VARIABLE_DESCRIPTION,
                    SignaloidYaml.OUTPUT_OBJECT,
                    SignaloidYaml.COMMAND_LINE_ARGUMENTS,
                ],
                where,
                problems,
            ):
                continue
            variable_name = entry[SignaloidYaml.VARIABLE_NAME]
            where = f"{where} ('{variable_name}')"
            output_object = entry[SignaloidYaml.OUTPUT_OBJECT]
            if output_object not in output_objects:
                problems.append(
                    f"{where} has {SignaloidYaml.OUTPUT_OBJECT} "
                    f"'{output_object}'. Use "
                    f"{' or '.join(repr(name) for name in output_objects)}."
                )
            if traced and variable_name not in traced:
                close = difflib.get_close_matches(str(variable_name), traced, n=1)
                hint = (
                    f"Did you mean '{close[0]}'?"
                    if close
                    else f"The traced expressions are {', '.join(traced)}."
                )
                problems.append(
                    f"{where} is not traced. Its {SignaloidYaml.VARIABLE_NAME} "
                    f"must match a {SignaloidYaml.TRACE_VARIABLES} expression. "
                    f"{hint}"
                )

    if problems:
        raise ValueError(
            f"{signaloid_yaml_path} cannot be used.\n  - " + "\n  - ".join(problems)
        )


# C keywords that start a statement which uses names rather than declaring
# them.
_STATEMENT_KEYWORDS = (
    "return|else|case|goto|sizeof|if|while|for|switch|do|break|continue"
)

# Comments, string and character literals, and preprocessor lines. They can
# hold a name, a ``;``, or a brace that is not code.
_NOT_CODE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'" r"|^[ \t]*#[^\n]*",
    re.DOTALL | re.MULTILINE,
)


def _code_lines(source: str) -> list[str]:
    """
    Blank out what is not code, keeping every line where it was.

    Args:
        source: A C source file.

    Returns:
        Its lines, with comments, literals, and preprocessor lines blanked.
    """

    def blank(match: re.Match[str]) -> str:
        return "\n" * match.group(0).count("\n")

    return _NOT_CODE.sub(blank, source).split("\n")


def _statements(code_lines: list[str]) -> list[tuple[int, int, str]]:
    """
    Split code into statements at ``;``, ``{``, and ``}``.

    A declaration can span lines, or name several variables, so it is
    judged as a whole statement rather than line by line.

    Args:
        code_lines: Lines from ``_code_lines``.

    Returns:
        Each statement's first line, last line, and text, with lines
        numbered from 1.
    """
    statements = []
    text: list[str] = []
    first_line = 0
    for line_number, line in enumerate(code_lines, start=1):
        for character in line:
            if not text and character.isspace():
                continue
            if not text:
                first_line = line_number
            text.append(character)
            if character in ";{}":
                statements.append((first_line, line_number, "".join(text)))
                text = []
        if text:
            text.append(" ")

    return statements


def _declaration_pattern(variable_name: str) -> re.Pattern[str]:
    """
    Match a C statement that declares ``variable_name``.

    The statement starts with one or more type words, such as ``uncertain
    double`` or ``char *const``. Other declarators may come first, as in
    ``double x, y[2]``. Then come the name and ``[``, ``=``, ``;``, ``,``,
    or ``)``. A use such as ``f(x)`` or ``x[0] = y`` has no type words
    before the name, so it does not match.

    Args:
        variable_name: The C identifier.

    Returns:
        The compiled pattern.
    """
    return re.compile(
        rf"^\s*(?!(?:{_STATEMENT_KEYWORDS})\b)(?:[A-Za-z_]\w*[\s*]+)+"
        r"(?:\**\s*[A-Za-z_]\w*[^;]*?,\s*)?\**\s*"
        + re.escape(variable_name)
        + r"\s*[\[=;,)]"
    )


def _is_use(statement: str, variable_name: str) -> bool:
    """
    Say whether a statement certainly uses names rather than declaring them.

    It starts with a keyword such as ``return``, or with the variable
    itself. Or it is a whole call such as ``f(x);``, or an assignment to
    another name. A macro that declares, such as ``DECLARE(x)``, has no
    ``;`` of its own, so it is not taken for a call.

    Args:
        statement: The statement text.
        variable_name: The traced C identifier.

    Returns:
        ``True`` when the statement is certainly not a declaration.
    """
    return bool(
        re.match(rf"\s*(?:{_STATEMENT_KEYWORDS})\b", statement)
        or re.match(rf"[\s*&(!~+-]*{re.escape(variable_name)}\b", statement)
        or re.fullmatch(r"\s*[A-Za-z_]\w*\s*\(.*\)\s*;\s*", statement, re.DOTALL)
        or re.match(
            r"\s*[A-Za-z_]\w*\s*(?:\[|\.|->|\+\+|--|[-+*/%&|^]?=(?!=))",
            statement,
        )
    )


def check_trace_variable_lines(
    path_to_application: str, trace_variables: list[dict]
) -> None:
    """
    Check that each traced expression's LineNumber declares its variable.

    The tracer resolves a trace against the line where the variable is
    declared, which is the line that holds its name. A line that only uses
    the variable gives an empty tracing database with no error, so this is
    checked before anything is built. A declaration may span lines or name
    several variables.

    A line that is certainly wrong stops the run. It does not hold the
    name, or it holds a use such as a call or an assignment. Every wrong
    entry is reported at once, each with the line or lines that do declare
    the variable. A line that holds the name in a layout this check does
    not recognize, such as a function parameter, only gives a warning.

    Args:
        path_to_application: The application root. Sources are in ``src``.
        trace_variables: The ``TraceVariables`` entries of signaloid.yaml.

    Raises:
        ValueError: An entry names a missing file, a line outside the
            file, or a line that does not declare its variable.
    """
    problems = []

    for entry in trace_variables:
        expression = entry[SignaloidYaml.EXPRESSION]
        file_name = entry[SignaloidYaml.FILE]
        line_number = entry[SignaloidYaml.LINE_NUMBER]
        where = f"'{expression}' in {file_name} line {line_number}"

        # The variable is the leading identifier, so 'outputVariables[0:3]'
        # and 'arguments.value' are declared as 'outputVariables' and
        # 'arguments'. An expression without one is left unchecked.
        name_match = re.match(r"\s*([A-Za-z_]\w*)", str(expression))
        if name_match is None:
            continue
        variable_name = name_match.group(1)

        source_path = os.path.join(path_to_application, "src", file_name)
        if not os.path.isfile(source_path):
            problems.append(f"{where} points to a missing file, {source_path}.")
            continue

        try:
            line = int(line_number)
        except (TypeError, ValueError):
            problems.append(f"{where} has a LineNumber that is not a whole number.")
            continue

        with open(source_path, "r", errors="replace") as source_file:
            source = source_file.read()
        lines = source.splitlines()
        code_lines = _code_lines(source)
        statements = _statements(code_lines)
        declares = _declaration_pattern(variable_name)
        name = re.compile(rf"\b{re.escape(variable_name)}\b")

        def statements_naming(index: int) -> list[str]:
            """The statements on line ``index`` when the line holds the name."""
            if not 1 <= index <= len(code_lines):
                return []
            if not name.search(code_lines[index - 1]):
                return []
            return [
                text
                for first, last, text in statements
                if first <= index <= last and name.search(text)
            ]

        candidates = statements_naming(line)
        if any(declares.search(text) for text in candidates):
            continue

        if 1 <= line <= len(lines) and lines[line - 1].strip():
            found = f"Line {line} reads '{lines[line - 1].strip()}'."
        elif 1 <= line <= len(lines):
            found = f"Line {line} is empty."
        else:
            found = f"{file_name} has {len(lines)} lines."

        if candidates and not all(_is_use(text, variable_name) for text in candidates):
            print(
                f"Warning: cannot tell whether {where} declares "
                f"'{variable_name}'. {found} If the tracing database comes "
                "out empty, set LineNumber to the line that declares it."
            )
            continue

        declared_on = [
            str(index)
            for index in range(1, len(code_lines) + 1)
            if any(declares.search(text) for text in statements_naming(index))
        ]
        if len(declared_on) == 1:
            hint = f"Did you mean LineNumber {declared_on[0]}?"
        elif declared_on:
            hint = (
                f"'{variable_name}' is declared on lines "
                f"{', '.join(declared_on)}. Pick the one in the scope "
                "where it is traced."
            )
        else:
            hint = f"No line in {file_name} declares '{variable_name}'."
        problems.append(f"{where} does not declare '{variable_name}'. {found} {hint}")

    if problems:
        raise ValueError(
            "signaloid.yaml TraceVariables do not match the source. "
            "LineNumber must be the line that declares the variable.\n  - "
            + "\n  - ".join(problems)
        )


def expand_array_expressions(data: list[dict]) -> list[dict]:
    """
    Expand array expressions like 'outputVariables[0:5]' into individual entries.

    Args:
        data: list of dictionaries with 'Expression' key containing array expressions

    Returns:
        Expanded list with individual entries for each array index
    """
    expanded = []

    for item in data:
        expression = item["Expression"]

        # Check if expression contains array slice notation [start:end]
        array_match = re.match(r"(\w+)\[(\d+):(\d+)\]", expression)

        if array_match:
            array_name = array_match.group(1)
            start_idx = int(array_match.group(2))
            end_idx = int(array_match.group(3))

            # Create individual entries for each index
            for i in range(start_idx, end_idx + 1):
                new_item = item.copy()
                new_item["Expression"] = f"{array_name}[{i}]"
                expanded.append(new_item)
        else:
            # Not an array expression, keep as-is
            expanded.append(item)

    return expanded


def _run_mc_simulation(
    work_item: tuple[str, int, str],
    path_to_application: str,
    native_executable_name: str,
    native_executable_dir: str,
) -> list[float]:
    """
    Run a single MC simulation in a separate process.

    Args:
        work_item: A ``(index, sample_size, cla)`` tuple describing the run.
        path_to_application: Application root (its ``inputs/`` is symlinked in).
        native_executable_name: Native binary filename.
        native_executable_dir: Directory containing the native binary.

    Returns:
        The scalar values parsed from the run's ``data.out``.

    Raises:
        RuntimeError: If the simulation exits with a non-zero status.
    """
    index, sub_size, cla = work_item

    values: list[float] = []
    with _native_mc_run(
        f"mc_sim_{index}_",
        path_to_application,
        native_executable_name,
        native_executable_dir,
        cla,
        sub_size,
    ) as (result, data_file):
        if result.returncode != 0:
            raise RuntimeError(
                f"Native Monte Carlo run {index} ({native_executable_name} "
                f"{cla}, sample size {sub_size}) exited with code "
                f"{result.returncode}.\n{result.stderr.strip()}"
            )

        # Read results
        if os.path.exists(data_file):
            with open(data_file, "r") as file:
                for line_no, line in enumerate(file):
                    if line_no > 0:  # Skip header
                        try:
                            values.append(float(line.strip()))
                        except ValueError:
                            continue

    return values


def compute_emcc_prediction(
    asymptotic_dist_quantity: float | None,
    distance: float,
) -> int | None:
    """
    Predict the equivalent Monte Carlo count from an asymptotic quantity.

    Computes ``(asymptotic_dist_quantity / distance) ** 2`` rounded up,
    the smallest sample count whose scaled statistic is at most ``distance``,
    clamped to a minimum of 1. Returns 1 if the inputs are invalid (e.g. a
    zero distance). Without an asymptotic statistic, which is the case for a
    metric with no CLT prediction, there is no prediction and ``None`` is
    returned rather than a count that could be read as the search floor.

    Args:
        asymptotic_dist_quantity: The asymptotic-distance statistic, or
            ``None`` when unavailable.
        distance: The UxHw-to-ground-truth distance.

    Returns:
        The predicted EMCC (at least 1), or ``None`` when there is no
        asymptotic statistic.
    """
    if asymptotic_dist_quantity is None:
        return None

    try:
        emcc_predicted = math.ceil((asymptotic_dist_quantity / distance) ** 2)
        # Check results make sense
        if emcc_predicted < 1:
            print("Warning: The predicted EMCC is below 1. Using 1.")
            emcc_predicted = 1
    except (TypeError, ZeroDivisionError, ValueError, OverflowError) as e:
        print(f"Warning: The EMCC prediction failed, so 1 is used. {e}")
        emcc_predicted = 1

    return emcc_predicted


def write_uxhw_distance_file(
    filename: str, benchmarking_variables: list[BenchmarkingVariable]
) -> None:
    """
    Write UxHw distance data to a CSV file.

    Args:
        filename: Path to the CSV file to write.
        benchmarking_variables: Benchmarking variables to write, each with a
            ``description`` and ``uxhw_distances``.
    """
    header = [
        BenchmarkingVariables.VARIABLE_DESCRIPTION,
        BenchmarkingVariables.UXHW_CONF,
        BenchmarkingVariables.UXHW_DISTANCE,
        BenchmarkingVariables.UXHW_BINNED_DISTANCE,
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        rows = [
            [
                variable.description,
                repr(record.uxhw_conf),
                record.uxhw_distance,
                (
                    ""
                    if record.uxhw_binned_distance is None
                    else record.uxhw_binned_distance
                ),
            ]
            for variable in benchmarking_variables
            for record in variable.uxhw_distances.records
        ]
        writer.writerows(rows)
