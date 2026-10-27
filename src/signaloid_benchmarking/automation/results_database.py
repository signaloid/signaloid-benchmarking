#   Copyright (c) 2026, Signaloid.
#
#   Permission is hereby granted, free of charge, to any person obtaining a
#   copy of this software and associated documentation files (the "Software"),
#   to deal in the Software without restriction, including without limitation
#   the rights to use, copy, modify, merge, publish, distribute, sublicense,
#   and/or sell copies of the Software, and to permit persons to whom the
#   Software is furnished to do so, subject to the following conditions:
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

import copy
import re
import sqlite3
import uuid
from contextlib import closing
from typing import Any, Sequence

from signaloid_benchmarking.automation.benchmarking_utils import utc_timestamp
from signaloid_benchmarking.config import EquivMC, RepresentationTypes
from signaloid.distributional.distributional import DistributionalValue

# Every run recorded, whichever command produced it. This is the registry: to
# see what a database holds, read this table. It names the application
# (`Model`), what ran it (`DeviceVariant`, the SCCE core) and the run itself
# (`DevicePath`, the SCCE task), with the arguments it was given. A run on
# SCCE is timed by the benchmark's timing file, not here.
_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS Runs (
    RunId INTEGER PRIMARY KEY AUTOINCREMENT,
    Timestamp TEXT NOT NULL,
    Command TEXT NOT NULL,
    Model TEXT,
    DeviceVariant TEXT,
    DevicePath TEXT,
    Inputs TEXT,
    TableName TEXT,
    ExecutionId INTEGER
)
"""

# The outputs taken apart, one row per Dirac delta, so that SQL can work with
# the positions and masses directly.
#
# `Execution_Info_Table_ID` and `ValueId` together join a delta to its row in
# the tracing table. `RunId` is kept as well, so a delta reaches `Runs` and the
# model, core and task there without going through the tracing table.
_OUTPUT_DELTAS_TABLE = """
CREATE TABLE IF NOT EXISTS OutputDeltas (
    RunId INTEGER NOT NULL,
    Execution_Info_Table_ID INTEGER,
    ValueId TEXT,
    OutputIndex INTEGER NOT NULL,
    DeltaIndex INTEGER NOT NULL,
    Position REAL, -- SQLite stores bound NaN as NULL; Dist_Value preserves it.
    Mass REAL NOT NULL
)
"""

# What the core was configured as for a distributional run. One row per such
# run.
#
# `UR_Order_CoreLibrary` is not asked for: it is `UR_Order` plus
# `UR_ORDER_CORE_LIBRARY_OFFSET` for Athens, and `UR_Order` for the others.
#
# The column types are as specified. Note that SQLite gives a column declared
# `STRING` numeric affinity rather than text, since the name contains none of
# INT, CHAR, CLOB, TEXT, BLOB, REAL, FLOA or DOUB: a value that looks like a
# number will be stored as one. That is harmless for names like `TTR` or
# `Enabled` and worth knowing for anything else.
_EMULATOR_EXECUTION_INFO_TABLE = """
CREATE TABLE IF NOT EXISTS Emulator_Execution_Info (
    Execution_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    UR_Type STRING,
    UR_Order INTEGER,
    UR_Order_CoreLibrary INTEGER,
    CorrelationTracking_Status STRING
)
"""


# What `UR_Order_CoreLibrary` is above `UR_Order`, for Athens. Athens numbers
# its order differently from the core library's, and the other representations
# do not, so applying the offset to them would record an order the library
# never used.
UR_ORDER_CORE_LIBRARY_OFFSET = 3


def _ur_order_core_library(
    ur_order: int | None,
    ur_type: str | None = None,
) -> int | None:
    """
    Derive ``UR_Order_CoreLibrary`` from ``UR_Order``.

    Athens is offset by ``UR_ORDER_CORE_LIBRARY_OFFSET``. Every other
    representation records the order it was given. Both the execution info
    row and the tracing rows carry this, and both get it from here, so the
    two copies cannot disagree.

    Args:
        ur_order: The representation's order, or ``None``.
        ur_type: The representation type, for example ``Athens``.

    Returns:
        The core library's order, or ``None`` when ``ur_order`` is, since an
        offset from nothing is not a number.
    """

    if ur_order is None:
        return None

    if ur_type != RepresentationTypes.ATHENS:
        return int(ur_order)

    return int(ur_order) + UR_ORDER_CORE_LIBRARY_OFFSET


# The assignment index of the first observation of an output in a
# distributional inference. An output observed again gets the next index.
# The SDK's tracing numbers a variable assigned again the same way. Each
# index gives its row its own ``ValueId``, so the observations stay apart.
TRACING_ASSIGNMENT_INDEX = 1


# The namespace the Signaloid tooling generates value identifiers in, from
# https://github.com/signaloid/radar/issues/154.
VALUE_ID_NAMESPACE = uuid.UUID("c2e12447-3766-5dda-b3e8-e61df7fcda74")

# What a value identifier is spelled with, as
# `ValueID := "val_" + UUIDv5(ValueNamespace, ...)`.
VALUE_ID_PREFIX = "val_"


def _value_id(
    subprogram: str | None,
    expression_name: str | None,
    declaration_line: int | None,
    assignment_line: int | None,
    assignment_index: int,
) -> str | None:
    """
    Build the identifier of one traced value, as ``generateValueID`` in the
    Signaloid runtime does.

    It is ``VALUE_ID_PREFIX`` followed by a version 5 UUID over the
    subprogram, the expression name and the three numbers, concatenated with
    no separator, in ``VALUE_ID_NAMESPACE``, written as lowercase hexadecimal
    with the hyphens removed. Version 5 is the SHA-1 flavour, which is what
    ``uuid_generate_sha1`` produces.

    The identifier names an assignment of the value, not one row. An
    inference that observes an output again gives the observation the next
    assignment index, and so another identifier.

    Args:
        subprogram: The function the value is assigned in, for example
            ``main``.
        expression_name: The expression, for example ``outputVariables[0]``.
        declaration_line: The line the variable is declared on.
        assignment_line: The line it is assigned on.
        assignment_index: Which assignment of it this is.

    Returns:
        The prefix and 32 hexadecimal characters, or ``None`` when a part of
        the name is missing, since an identifier built over a gap would
        collide with every other one built over the same gap.
    """

    if (
        subprogram is None
        or expression_name is None
        or declaration_line is None
        or assignment_line is None
    ):
        return None

    name = (
        f"{subprogram}{expression_name}"
        f"{int(declaration_line)}{int(assignment_line)}{int(assignment_index)}"
    )

    return f"{VALUE_ID_PREFIX}{uuid.uuid5(VALUE_ID_NAMESPACE, name).hex}"


def _validate_table_name(name: str) -> str:
    """
    Check a caller-supplied table name.

    A table name cannot be a bound parameter, so it is interpolated into the
    statement and has to be checked rather than escaped.

    Args:
        name: The table name.

    Returns:
        The same name.

    Raises:
        ValueError: If the name is not a letter or underscore followed by
            letters, digits or underscores.
    """

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name or ""):
        raise ValueError(
            f"{name!r} is not a usable table name. Use a letter or an "
            f"underscore followed by letters, digits or underscores."
        )

    return name


# The tracing table, as (column, type), in the order the columns are declared.
#
# `Execution_Info_Table_ID` holds the `Emulator_Execution_Info.Execution_ID`
# of the run that produced the row, which is what the distance analysis joins
# on.
#
# `Particle_Value` is the number reported for the value and `Dist_Value` the
# whole distribution as a Ux String. A value that came back certain rather than
# distributional has a particle value and no `Dist_Value`.
_TRACING_COLUMNS = (
    ("ValueId", "TEXT"),
    ("Expression_Name", "TEXT"),
    ("Expression_Subprogram", "TEXT"),
    ("Expression_DeclarationFileName", "TEXT"),
    ("Expression_DeclarationLineNumber", "INTEGER"),
    ("Execution_Info_Table_ID", "INTEGER NOT NULL"),
    ("Particle_Value", "REAL"),
    ("Dist_Value", "TEXT"),
    ("Assignment_Index", "INTEGER NOT NULL"),
    # The same three values as the `Emulator_Execution_Info` row this points
    # at, repeated here as the UxHw SDK's own tracing table repeats them, so
    # that a traced value says what core produced it without a join. They
    # are filled from one place, so the two copies cannot disagree.
    ("UR_Type", "TEXT"),
    ("UR_Order", "INTEGER"),
    ("UR_Order_CoreLibrary", "INTEGER"),
)


def _tracing_table(name: str) -> str:
    """
    Build the statement creating a tracing table.

    The table holds one row per output element of a distributional
    inference, with the columns of ``_TRACING_COLUMNS``.

    Args:
        name: The table's name.

    Returns:
        The ``CREATE TABLE IF NOT EXISTS`` statement.
    """

    columns = ",\n    ".join(f"{column} {kind}" for column, kind in _TRACING_COLUMNS)

    return f"""
CREATE TABLE IF NOT EXISTS {_validate_table_name(name)} (
    {columns}
)
"""


_TABLES = (
    _RUNS_TABLE,
    _EMULATOR_EXECUTION_INFO_TABLE,
    _OUTPUT_DELTAS_TABLE,
)

_RUN_COLUMNS = (
    "Timestamp",
    "Command",
    "Model",
    "DeviceVariant",
    "DevicePath",
    "Inputs",
    # Which table this run's rows went into, so that a file holding several
    # can be read back without being told.
    "TableName",
    # The `Emulator_Execution_Info.Execution_ID` this run wrote, which is what
    # the tracing table's `Execution_Info_Table_ID` holds.
    "ExecutionId",
)


def _ux_string(value: float | DistributionalValue) -> str | None:
    """
    Export one returned value as a Ux String.

    ``DistributionalValue.export`` is the library's own implementation of
    the documented format, so the mapping from the Ux-Binary that arrived to
    the string stored here is the library's, not a second one written by
    hand that could drift from it.

    A single-precision value is widened to double first, which widens the
    position fields and nothing else. Every float32 is exactly representable
    as a double, so no value changes, and every stored string has the same
    shape whatever precision the value arrived in.

    Args:
        value: The value, as a float or a distributional value.

    Returns:
        The Ux String, or ``None`` when the value is not a distribution.

    Raises:
        TypeError: If the library exports bytes rather than text.
    """

    if not isinstance(value, DistributionalValue):
        return None

    if not value.double_precision:
        # A copy, because the caller goes on to use this value for the delta
        # rows and for printing, and the flag it carries is theirs.
        value = copy.copy(value)
        value.double_precision = True

    exported = value.export(to_str=True)
    if not isinstance(exported, str):
        raise TypeError("DistributionalValue.export(to_str=True) did not return text.")

    return exported


def _create_tables(
    cursor: sqlite3.Cursor,
    tracing_table: str | None = None,
) -> None:
    """
    Create the tables a run writes, where they do not exist yet.

    Args:
        cursor: A cursor on the database.
        tracing_table: The tracing table of an inference, when given.
    """
    for statement in _TABLES:
        cursor.execute(statement)

    if tracing_table is not None:
        cursor.execute(_tracing_table(tracing_table))


def _last_row_id(cursor: sqlite3.Cursor) -> int:
    """
    Read the id of the row the cursor just inserted.

    Args:
        cursor: The cursor the insert ran on.

    Returns:
        The row's id.

    Raises:
        RuntimeError: If SQLite reported none.
    """

    row_id = cursor.lastrowid
    if row_id is None:
        raise RuntimeError("SQLite reported no row id for the inserted row.")

    return row_id


def _insert_run(
    cursor: sqlite3.Cursor,
    fields: dict[str, Any],
) -> int:
    """
    Insert one row into ``Runs``.

    Args:
        cursor: A cursor on the database.
        fields: The row's values by column name. A column not in it is
            stored as null.

    Returns:
        The ``RunId`` the row was given.
    """

    row = tuple(fields.get(column) for column in _RUN_COLUMNS)
    columns = ", ".join(_RUN_COLUMNS)
    placeholders = ", ".join(["?"] * len(_RUN_COLUMNS))

    cursor.execute(
        f"INSERT INTO Runs ({columns}) VALUES ({placeholders})",
        row,
    )

    return _last_row_id(cursor)


def _insert_execution_info(
    cursor: sqlite3.Cursor,
    ur_type: str | None,
    ur_order: int | None,
    correlation_tracking: str | None,
) -> int:
    """
    Insert one row into ``Emulator_Execution_Info``.

    ``UR_Order_CoreLibrary`` is derived rather than given, see
    ``_ur_order_core_library``.

    Args:
        cursor: A cursor on the database.
        ur_type: The uncertainty representation the core was configured
            with.
        ur_order: Its order.
        correlation_tracking: Whether the core was tracking correlations.

    Returns:
        The ``Execution_ID`` the row was given.
    """

    cursor.execute(
        "INSERT INTO Emulator_Execution_Info "
        "(UR_Type, UR_Order, UR_Order_CoreLibrary, CorrelationTracking_Status) "
        "VALUES (?, ?, ?, ?)",
        (
            ur_type,
            ur_order,
            _ur_order_core_library(ur_order, ur_type),
            correlation_tracking,
        ),
    )

    return _last_row_id(cursor)


def record_inference(
    database_path: str,
    values: Sequence[float | DistributionalValue],
    *,
    model: str | None = None,
    variant: str | None = None,
    device_path: str | None = None,
    inputs: Sequence[str] | None = None,
    table_name: str = EquivMC.TRACING_TABLE,
    output_names: Sequence[str] | None = None,
    declaration: tuple[str | None, str | None, int | None, int | None] | None = None,
    ur_type: str | None = None,
    ur_order: int | None = None,
    correlation_tracking: str | None = None,
) -> int:
    """
    Store the outputs of one inference.

    Each output goes into the tracing table, whole, as a Ux String in
    ``Dist_Value``, and into ``OutputDeltas`` as one row per Dirac delta. A
    value that came back certain, as a plain float rather than a
    distribution, has a ``Particle_Value`` and no ``Dist_Value`` or deltas.

    The mean and variance are not stored. They are exactly recoverable from
    the delta rows, treating each delta as a weighted sample, with
    ``SELECT SUM(Position * Mass) / SUM(Mass) ... GROUP BY ValueId``.

    Args:
        database_path: The SQLite database, created when it does not exist.
        values: The outputs, each a float or a distributional value.
        model: The application, stored as ``Model``.
        variant: What ran it, stored as ``DeviceVariant``. On SCCE, the
            core.
        device_path: The run itself, stored as ``DevicePath``. On SCCE, the
            task.
        inputs: The arguments the run was given, stored as ``Inputs``.
        table_name: The table the outputs go into.
        output_names: The output variable names, one per output element.
            A name that appears more than once marks successive
            observations of one output. Each observation gets the next
            assignment index, and with it its own ``ValueId``, so the
            rows stay distinct. Which of them a reader uses is the
            reader's rule. An element beyond the names is an unnamed
            output of its own, not an observation of another.
        declaration: Where the output variable is declared and assigned, as
            (subprogram, file name, declaration line, assignment line).
        ur_type: The uncertainty representation the core was configured
            with.
        ur_order: Its order. ``UR_Order_CoreLibrary`` follows from this.
        correlation_tracking: Whether the core was tracking correlations.

    Returns:
        The ``RunId`` the run was given.

    Raises:
        ValueError: If ``table_name`` is not a usable table name.
    """

    _validate_table_name(table_name)

    names = list(output_names) if output_names is not None else []
    (
        subprogram,
        declaration_file,
        declaration_line,
        assignment_line,
    ) = declaration or (None, None, None, None)

    with closing(sqlite3.connect(database_path)) as connection:
        cursor = connection.cursor()
        _create_tables(cursor, tracing_table=table_name)

        # The execution info comes first: the tracing rows below carry its
        # `Execution_ID`, and `Runs` records it so that a row can be followed
        # back to the model, the core and the task as well.
        execution_id = _insert_execution_info(
            cursor,
            ur_type=ur_type,
            ur_order=ur_order,
            correlation_tracking=correlation_tracking,
        )

        run_id = _insert_run(
            cursor,
            {
                "Timestamp": utc_timestamp(),
                "Command": "inference",
                "TableName": table_name,
                "ExecutionId": execution_id,
                "Model": model,
                "DeviceVariant": variant,
                "DevicePath": device_path,
                "Inputs": " ".join(inputs) if inputs else None,
            },
        )

        output_rows: list[tuple[Any, ...]] = []
        delta_rows: list[tuple[Any, ...]] = []
        # How often each named output has been seen so far, which numbers
        # its assignments from TRACING_ASSIGNMENT_INDEX upwards. An unnamed
        # output is one of its own and keeps the first index.
        observations: dict[str, int] = {}

        for index, value in enumerate(values):
            name = names[index] if index < len(names) else None
            assignment_index = TRACING_ASSIGNMENT_INDEX
            if name is not None:
                assignment_index += observations.get(name, 0)
                observations[name] = assignment_index
            # One identifier per observation. It is a hash of the same fields
            # every time, so it is computed once and the deltas reuse it.
            identifier = _value_id(
                subprogram=subprogram,
                expression_name=name,
                declaration_line=declaration_line,
                assignment_line=assignment_line,
                assignment_index=assignment_index,
            )

            if isinstance(value, DistributionalValue):
                particle = (
                    float(value.particle_value)
                    if value.particle_value is not None
                    else None
                )
                dist_value = _ux_string(value)
                for delta_index, (position, mass) in enumerate(
                    zip(value.positions, value.masses)
                ):
                    delta_rows.append(
                        (
                            run_id,
                            execution_id,
                            identifier,
                            index,
                            delta_index,
                            float(position),
                            float(mass),
                        )
                    )
            else:
                particle = float(value)
                dist_value = None

            output_rows.append(
                (
                    identifier,
                    name,
                    subprogram,
                    declaration_file,
                    declaration_line,
                    execution_id,
                    particle,
                    dist_value,
                    assignment_index,
                    ur_type,
                    ur_order,
                    _ur_order_core_library(ur_order, ur_type),
                )
            )

        cursor.executemany(
            f"INSERT INTO {table_name} "
            f"({', '.join(column for column, _ in _TRACING_COLUMNS)}) "
            f"VALUES ({', '.join('?' * len(_TRACING_COLUMNS))})",
            output_rows,
        )
        cursor.executemany(
            "INSERT INTO OutputDeltas (RunId, Execution_Info_Table_ID, "
            "ValueId, OutputIndex, DeltaIndex, Position, Mass) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            delta_rows,
        )

        connection.commit()

    return run_id
