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

import csv
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from typing import Any, Callable
from signaloid_benchmarking.automation.sample_generator import (
    generate_database_native_mc,
)
from signaloid_benchmarking.types import BenchmarkingVariable
from signaloid_benchmarking.config import RepresentationTypes, VariableTypes

# The five expression-metadata columns that prefix every row in both the
# WeightedSamples and MonteCarlo data tables.
_SAMPLE_METADATA_COLUMNS = (
    "ValueId",
    "Expression_Name",
    "Expression_Subprogram",
    "Expression_DeclarationFileName",
    "Expression_DeclarationLineNumber",
)


def _variable_metadata(variable: BenchmarkingVariable) -> tuple[Any, ...]:
    return (
        variable.value_id,
        variable.name,
        variable.program,
        variable.path,
        variable.line_number,
    )


def _weighted_sample_rows(variable: BenchmarkingVariable) -> list[tuple[Any, ...]]:
    """
    Build ``(Position, Weight, MonteCarlo_Count)`` rows for one variable.

    Args:
        variable: The variable whose distribution samples are converted.

    Returns:
        One row per distribution sample or per scalar output value.
    """
    samples = variable.distribution_samples
    if variable.type == VariableTypes.DISTRIBUTION:
        return [
            (value, weight, 1) for value, weight in zip(samples.values, samples.weights)
        ]
    if variable.type == VariableTypes.SCALAR:
        return [
            (value, 1.0, mc_count)
            for mc_count, value_list in samples.scalar_output_dict.items()
            for value in value_list
        ]
    print(f"Warning! variable type {variable.type} not supported")
    return []


def _mc_sample_rows(variable: BenchmarkingVariable) -> list[tuple[Any, ...]]:
    """
    Build ``(Assignment_Index, Particle_Value, MonteCarlo_Count)`` rows for one
    variable.

    Args:
        variable: The variable whose distribution samples are converted.

    Returns:
        One row per distribution sample or per scalar output value.
    """
    samples = variable.distribution_samples
    if variable.type == VariableTypes.DISTRIBUTION:
        return [(1, value, 1) for value in samples.values]
    if variable.type == VariableTypes.SCALAR:
        return [
            (adversary_index, value, mc_count)
            for mc_count, value_list in samples.scalar_output_dict.items()
            for adversary_index, value in enumerate(value_list)
        ]
    print(f"Warning! variable type {variable.type} not supported")
    return []


def _write_samples_database(
    database_path: str,
    variables: list[BenchmarkingVariable],
    *,
    table_name: str,
    data_table_columns: str,
    value_columns: tuple[str, ...],
    row_builder: Callable[[BenchmarkingVariable], list[tuple[Any, ...]]],
) -> None:
    """
    Write a samples database with the shared two-table layout.

    Both ground-truth-style tables share the same structure: a data table
    (five expression-metadata columns + an autoincrement PK + three value
    columns) and a ``Printed_ValueIds`` table.

    Args:
        database_path: Path to the SQLite database to write.
        variables: Variables whose samples populate the data table.
        table_name: Name of the data table to create.
        data_table_columns: SQL for the PK + value columns.
        value_columns: Names of the value columns, for the INSERT.
        row_builder: Yields the value-column tuples for a variable's rows.

    Raises:
        ValueError: If ``row_builder`` yields no rows for a variable.
    """
    # Connect to SQLite database (creates a new file if it doesn't exist)
    with closing(sqlite3.connect(database_path)) as conn:
        cursor = conn.cursor()

        # Create two tables
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
        ValueId TEXT NOT NULL,
        Expression_Name TEXT NOT NULL,
        Expression_Subprogram TEXT NOT NULL,
        Expression_DeclarationFileName TEXT NOT NULL,
        Expression_DeclarationLineNumber TEXT NOT NULL,
        {data_table_columns}
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS Printed_ValueIds (
        ValueId TEXT NOT NULL,
        SampleType TEXT NOT NULL
        )
        """)

        table1_data = []
        for variable in variables:
            metadata = _variable_metadata(variable)
            value_rows = row_builder(variable)
            if not value_rows:
                raise ValueError(
                    f"No samples to write for benchmarking variable "
                    f"{variable.name!r} (type {variable.type!r}) into table "
                    f"{table_name} of {database_path!r}. Later stages would "
                    f"fail on it without naming it."
                )
            for value_row in value_rows:
                table1_data.append(metadata + value_row)

        columns = ", ".join(_SAMPLE_METADATA_COLUMNS + value_columns)
        placeholders = ", ".join(
            ["?"] * (len(_SAMPLE_METADATA_COLUMNS) + len(value_columns))
        )
        cursor.executemany(
            f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})",
            table1_data,
        )

        # Populate table2 with data
        table2_data = [
            (variables[0].value_id, table_name),
        ]

        cursor.executemany(
            "INSERT INTO Printed_ValueIds (ValueId, SampleType) VALUES (?, ?)",
            table2_data,
        )

        # Commit the changes. The connection is closed on with-block exit.
        conn.commit()


def generate_database_from_weighted_samples(
    database_path: str, variables: list[BenchmarkingVariable]
) -> None:
    """
    Write a ``WeightedSamples`` database for the given variables.

    Args:
        database_path: Path to the SQLite database to write.
        variables: Variables whose weighted samples are written.

    Raises:
        ValueError: If a variable has no samples to write. Raised before any
            row of this call is inserted, since a later stage would fail on
            the empty entry without naming the variable.
    """
    _write_samples_database(
        database_path,
        variables,
        table_name="WeightedSamples",
        data_table_columns=(
            "Id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
            "        Position FLOAT,\n"
            "        Weight FLOAT,\n"
            "        MonteCarlo_Count INTEGER"
        ),
        value_columns=("Position", "Weight", "MonteCarlo_Count"),
        row_builder=_weighted_sample_rows,
    )


def generate_database_from_mc_samples(
    database_path: str, variables: list[BenchmarkingVariable]
) -> None:
    """
    Write a ``MonteCarlo`` database for the given variables.

    Args:
        database_path: Path to the SQLite database to write.
        variables: Variables whose Monte Carlo samples are written.

    Raises:
        ValueError: If a variable has no samples to write. Raised before any
            row of this call is inserted, since a later stage would fail on
            the empty entry without naming the variable.
    """
    _write_samples_database(
        database_path,
        variables,
        table_name="MonteCarlo",
        data_table_columns=(
            "MC_Id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
            "        Assignment_Index INTEGER,\n"
            "        Particle_Value FLOAT,\n"
            "        MonteCarlo_Count INTEGER"
        ),
        value_columns=("Assignment_Index", "Particle_Value", "MonteCarlo_Count"),
        row_builder=_mc_sample_rows,
    )


def generate_analytic_ground_truth_database(
    *,
    benchmarking_variables: list[BenchmarkingVariable],
    path_to_application: str,
    path_to_ground_truth_file: str,
    ground_truth_size: int,
    ground_truth_db_path: str,
) -> None:
    """Generate the Ground Truth database using the analytic formula.

    Shells out to a ground-truth script, reads the resulting CSV, populates
    each variable's sample buffers, and writes the SQLite DB via
    generate_database_from_weighted_samples.

    Args:
        benchmarking_variables: List of variables to populate with ground
            truth samples.
        path_to_application: Root path of the application being benchmarked.
            The CSV is written to {path_to_application}/src/ground-truth.csv.
        path_to_ground_truth_file: Path to the Python script that generates
            the ground truth CSV.
        ground_truth_size: Number of ground truth samples to generate.
        ground_truth_db_path: Path at which to write the output SQLite DB.

    Raises:
        RuntimeError: If the ground truth script exits with a non-zero return
            code.
        ValueError: If the ground truth CSV is malformed, if it yields no
            rows for one of the benchmarking variables, or if it yields more
            than one row for a Scalar. Raised before anything is written.
    """
    print("Generating Ground Truth using analytic formula.")
    path_to_csv = f"{path_to_application}/src/ground-truth.csv"

    # Generate the CSV of values. Use argv-list invocation (shell=False) so
    # paths with spaces or shell metacharacters cannot break the call or inject.
    result = subprocess.run(
        [
            sys.executable,
            path_to_ground_truth_file,
            str(ground_truth_size),
            path_to_csv,
        ],
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        error_message = (
            "Ground truth generation failed " "(exit code " f"{result.returncode})."
        )
        if result.stderr and result.stderr.strip():
            error_message += "\nstderr:\n" + result.stderr
        raise RuntimeError(error_message)

    # Empty values
    for variable in benchmarking_variables:
        variable.distribution_samples.empty_values()

    # Lookup by name. Rows for unknown names are silently skipped.
    variables_by_name = {variable.name: variable for variable in benchmarking_variables}

    # Load csv data into BenchmarkingVariable objects.
    with open(path_to_csv, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)

        for line_number, row in enumerate(reader, start=1):
            if not row:
                continue  # skip blank lines (e.g. a trailing newline)
            row_variable = variables_by_name.get(row[0])
            if row_variable is None:
                continue  # rows for unknown names are skipped (see above)
            if len(row) < 3:
                raise ValueError(
                    f"Malformed ground-truth CSV {path_to_csv!r} at line "
                    f"{line_number}: expected 3 columns (name, value, weight), "
                    f"got {len(row)}."
                )
            try:
                value = float(row[1])
                weight = float(row[2])
            except ValueError as e:
                raise ValueError(
                    f"Malformed ground-truth CSV {path_to_csv!r} at line "
                    f"{line_number}: value and weight (columns 2-3) must be "
                    f"numeric, got {row[1]!r} and {row[2]!r}."
                ) from e
            row_variable.distribution_samples.values.append(value)
            row_variable.distribution_samples.weights.append(weight)

    # Checked before anything is written, so a mistake leaves no partial
    # database, and the message can name the script and the CSV, which is
    # where a name mismatch would be.
    #
    # ``_weighted_sample_rows`` reads a Scalar variable from
    # ``scalar_output_dict``, not from ``values``/``weights``. Key by
    # ``ground_truth_size`` so the row carries the same MonteCarlo_Count the
    # native ground-truth pass writes, which is what the max-MonteCarlo_Count
    # selection in the loader looks for.
    for variable in benchmarking_variables:
        samples = variable.distribution_samples
        if not samples.values:
            raise ValueError(
                f"Ground-truth script {path_to_ground_truth_file!r} wrote no "
                f"rows for {variable.name!r} to {path_to_csv!r}. The first "
                "column of each row must match a VariableName."
            )
        if variable.type == VariableTypes.SCALAR:
            # A Scalar's reference is one value. More rows would be stored at
            # equal weight and fail two steps later without naming the
            # variable, so they are refused here.
            if len(samples.values) != 1:
                raise ValueError(
                    f"Ground-truth script {path_to_ground_truth_file!r} wrote "
                    f"{len(samples.values)} rows for Scalar {variable.name!r} "
                    f"to {path_to_csv!r}. A Scalar takes exactly one row."
                )
            samples.scalar_output_dict = {ground_truth_size: samples.values}
            samples.values = []
            samples.weights = []

    # Convert to Database
    for variable in benchmarking_variables:
        generate_database_from_weighted_samples(ground_truth_db_path, [variable])


def generate_uxhw_tracing_database(
    *,
    benchmarking_variables: list[BenchmarkingVariable],
    tracing_db_path: str,
    run_timing_script: Callable[..., None],
) -> None:
    """Generate the UxHw tracing database.

    Runs tracing for each benchmarking variable individually using its
    own command-line arguments, then collects the UX strings from each
    run into a single tracing database.

    Args:
        benchmarking_variables: Variables to trace. One
            ``run_timing_script`` invocation is issued per variable.
        tracing_db_path: Filesystem path of the tracing database.
            Removed before the first tracing run so that stale data
            does not bleed into the new results.
        run_timing_script: Callable that executes the bash timing
            script for a single variable. Must accept at minimum the
            keyword arguments ``variable_index: int`` and
            ``tracing: bool``. All other required arguments are bound
            by the caller via ``functools.partial`` (or equivalent).
    """
    print("Generating UxHw tracing database.")
    if os.path.isfile(tracing_db_path):
        os.remove(tracing_db_path)
    for i in range(len(benchmarking_variables)):
        run_timing_script(variable_index=i, tracing=True)


def generate_adversary_database(
    *,
    has_native_mc: bool,
    adversary_mc_size: int,
    adversary_db_path: str,
    benchmarking_variables: list[BenchmarkingVariable],
    n_processors: int,
    n_adversaries: int,
    ground_truth_size: int,
    adversary_max_size_scalar: int,
    use_clt: bool,
    path_to_application: str,
    native_executable_name: str,
    native_executable_dir: str,
    demo_cli_args: str,
) -> None:
    """Generate the Adversarial MC database required for equivalent_mc.

    Generated via native execution. A native build is required.

    Args:
        has_native_mc: When ``True``, generate the database via native
            execution (otherwise raise).
        adversary_mc_size: Number of adversary Monte Carlo samples to
            generate.
        adversary_db_path: Filesystem path at which to write the
            adversary SQLite database.
        benchmarking_variables: Variables to populate with samples
            (native-MC path only).
        n_processors: Number of parallel worker processes
            (native-MC path only).
        n_adversaries: Number of independent adversary runs per
            scalar sample size (native-MC path only).
        ground_truth_size: Sample size used for the ground-truth
            scalar pass (native-MC path only; unused here but
            forwarded for signature parity).
        adversary_max_size_scalar: Upper bound used when generating
            the geometric series of scalar sample sizes
            (native-MC path only).
        use_clt: When ``True``, scalar sample sizes are
            taken from pre-computed EMCC predictions (native-MC path
            only).
        path_to_application: Root path of the application source tree
            (native-MC path only).
        native_executable_name: Filename of the compiled native
            binary (native-MC path only).
        native_executable_dir: Directory containing the native binary
            (native-MC path only).
        demo_cli_args: Per-application command-line argument prefix
            (native-MC path only).

    Raises:
        ValueError: If a variable ends up with no samples to write.
    """
    if has_native_mc:
        print("Generating Adversarial MC using native execution.")
        generate_database_native_mc(
            size=adversary_mc_size,
            database_path=adversary_db_path,
            benchmarking_variables=benchmarking_variables,
            n_processors=n_processors,
            n_adversaries=n_adversaries,
            ground_truth_size=ground_truth_size,
            adversary_max_size_scalar=adversary_max_size_scalar,
            use_clt=use_clt,
            path_to_application=path_to_application,
            native_executable_name=native_executable_name,
            native_executable_dir=native_executable_dir,
            demo_cli_args=demo_cli_args,
        )
    else:
        raise RuntimeError(
            "Adversary Monte Carlo generation requires a native build "
            "(a Makefile 'local-build' target or src/config.mk with SOURCES)."
        )


def generate_ground_truth_database(
    *,
    has_analytic_ground_truth: bool,
    has_native_mc: bool,
    benchmarking_variables: list[BenchmarkingVariable],
    path_to_application: str,
    path_to_ground_truth_file: str,
    ground_truth_size: int,
    ground_truth_db_path: str,
    ground_truth_type: str,
    max_num_weighted_samples: int,
    n_processors: int,
    n_adversaries: int,
    adversary_max_size_scalar: int,
    use_clt: bool,
    native_executable_name: str,
    native_executable_dir: str,
    demo_cli_args: str,
) -> None:
    """Generate the Ground Truth database required for equivalent_mc.

    A variable with an analytic reference gets it from its script, and
    every other variable gets a native Monte Carlo reference. A variable's
    script is its own ``AnalyticGroundTruth`` from signaloid.yaml, or the
    run-wide one when it declares none and ``--has-analytic-ground-truth``
    is set. Each distinct script runs once, for the variables that name
    it. Both kinds of reference go into the one database, the analytic
    ones as weighted samples and the rest as ``ground_truth_type``.

    Args:
        has_analytic_ground_truth: Whether a variable that declares no
            analytic reference of its own uses the run-wide script.
        has_native_mc: Whether the application has a native build, which
            every variable without an analytic reference needs.
        benchmarking_variables: Variables to populate with ground
            truth samples (analytic and native-MC paths).
        path_to_application: Root path of the application (analytic
            path: CSV written to
            ``{path_to_application}/src/ground-truth.csv``.
            native-MC path: working directory for the native binary).
        path_to_ground_truth_file: The run-wide analytic script, used by
            variables that declare none of their own.
        ground_truth_size: Number of ground truth samples to generate.
        ground_truth_db_path: Filesystem path at which to write the
            ground truth SQLite database.
        ground_truth_type: Representation type of the native Monte Carlo
            reference, which decides whether it is written as weighted
            samples.
        max_num_weighted_samples: Maximum number of weighted samples
            (native-MC path only).
        n_processors: Number of parallel worker processes
            (native-MC path only).
        n_adversaries: Number of independent adversary runs per
            scalar sample size (native-MC path only; unused here but
            forwarded for signature parity).
        adversary_max_size_scalar: Upper bound used when generating
            the geometric series of scalar sample sizes
            (native-MC path only; unused here but forwarded for
            signature parity).
        use_clt: When ``True``, scalar sample sizes are
            taken from pre-computed EMCC predictions (native-MC path
            only; unused for ground-truth scalars but forwarded for
            signature parity).
        native_executable_name: Filename of the compiled native
            binary (native-MC path only).
        native_executable_dir: Directory containing the native binary
            (native-MC path only).
        demo_cli_args: Per-application command-line argument prefix
            (native-MC path only).

    Raises:
        RuntimeError: If a variable needs a native Monte Carlo reference
            and the application has no native build. The message names
            those variables. Also if an analytic script exits with a
            non-zero return code.
        ValueError: If an analytic script's CSV is malformed, yields no
            rows for one of its variables, or yields more than one row for
            a Scalar.
    """
    variables_by_script: dict[str, list[BenchmarkingVariable]] = {}
    native_mc_variables: list[BenchmarkingVariable] = []
    for variable in benchmarking_variables:
        script = variable.analytic_ground_truth
        if script is None and has_analytic_ground_truth:
            # Absolute, so the run-wide script and a variable naming the
            # same file from signaloid.yaml run as one.
            script = os.path.abspath(path_to_ground_truth_file)
        if script is None:
            native_mc_variables.append(variable)
        else:
            variables_by_script.setdefault(script, []).append(variable)

    # Checked before any script runs, since a script can take a while.
    if native_mc_variables and not has_native_mc:
        raise RuntimeError(
            "Ground-truth generation requires an analytic ground-truth "
            "script (AnalyticGroundTruth in signaloid.yaml, or "
            "--has-analytic-ground-truth) or a native build (a Makefile "
            "'local-build' target or src/config.mk with SOURCES). No "
            "analytic reference covers "
            + ", ".join(variable.name for variable in native_mc_variables)
            + "."
        )

    for script, script_variables in variables_by_script.items():
        generate_analytic_ground_truth_database(
            benchmarking_variables=script_variables,
            path_to_application=path_to_application,
            path_to_ground_truth_file=script,
            ground_truth_size=ground_truth_size,
            ground_truth_db_path=ground_truth_db_path,
        )

    if not native_mc_variables:
        return

    print("Generating Ground Truth using native execution.")
    generate_database_native_mc(
        size=ground_truth_size,
        database_path=ground_truth_db_path,
        benchmarking_variables=native_mc_variables,
        n_processors=n_processors,
        n_adversaries=n_adversaries,
        ground_truth_size=ground_truth_size,
        adversary_max_size_scalar=adversary_max_size_scalar,
        use_clt=use_clt,
        path_to_application=path_to_application,
        native_executable_name=native_executable_name,
        native_executable_dir=native_executable_dir,
        demo_cli_args=demo_cli_args,
        weighted_samples=ground_truth_type == RepresentationTypes.WEIGHTED_SAMPLES,
        num_weighted_samples=max_num_weighted_samples,
        ground_truth=True,
    )
