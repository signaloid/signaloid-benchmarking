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

import io
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from signaloid_benchmarking.automation.read_db_metrics import (
    main,
    read_db_metric,
)


def _make_db(path: str) -> None:
    """Populate a test SQLite database with known values.

    Creates the table queried by :mod:`read_db_metrics` and inserts two
    rows into ``runtimeStats``, so the metric must take the first.
    """
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE runtimeStats " "(Host_UserTimeElapsedWallClock REAL)"
        )
        conn.execute("INSERT INTO runtimeStats VALUES (1.5)")
        conn.execute("INSERT INTO runtimeStats VALUES (2.5)")
        conn.commit()


class TestReadDbMetrics(unittest.TestCase):
    """Coverage for read_db_metric helper and CLI entry point."""

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.tmp_path = Path(tmp_dir.name)

    def test_path_with_uri_reserved_chars_is_handled(self) -> None:
        """Paths containing ``?`` or ``#`` must be percent-encoded before
        being interpolated into the SQLite URI (regression: a raw
        ``f"file:{db_path}?mode=ro"`` would treat ``?`` as the query
        delimiter and fail to open the database)."""
        db = str(self.tmp_path / "weird?name#frag.db")
        _make_db(db)
        self.assertEqual(read_db_metric(db, "host-wallclock"), 1.5)

    def test_host_wallclock_returns_first_row(self) -> None:
        """``host-wallclock`` returns the first row of
        ``Host_UserTimeElapsedWallClock``."""
        db = str(self.tmp_path / "test.db")
        _make_db(db)
        result = read_db_metric(db, "host-wallclock")
        self.assertEqual(result, 1.5)

    def test_missing_db_file_raises_operational_error(self) -> None:
        """A path that does not exist raises ``sqlite3.OperationalError``."""
        missing = str(self.tmp_path / "nonexistent.db")
        with self.assertRaises(sqlite3.OperationalError):
            read_db_metric(missing, "host-wallclock")

    def test_schema_mismatch_raises_operational_error(self) -> None:
        """A DB that lacks the expected table raises
        ``sqlite3.OperationalError``."""
        db = str(self.tmp_path / "empty.db")
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE unrelated (x INTEGER)")
            conn.commit()
        with self.assertRaises(sqlite3.OperationalError):
            read_db_metric(db, "host-wallclock")

    def test_unknown_metric_raises_key_error(self) -> None:
        """Requesting an unrecognised metric raises ``KeyError``."""
        db = str(self.tmp_path / "test.db")
        _make_db(db)
        with self.assertRaisesRegex(KeyError, "unknown metric"):
            read_db_metric(db, "not-a-real-metric")

    def test_main_host_wallclock_prints_value(self) -> None:
        """``main`` with ``--metric host-wallclock`` prints the correct float."""
        db = str(self.tmp_path / "test.db")
        _make_db(db)

        buf: io.StringIO = io.StringIO()
        with patch.object(
            sys, "argv", ["read_db_metrics", db, "--metric", "host-wallclock"]
        ):
            with redirect_stdout(buf):
                main()

        self.assertEqual(float(buf.getvalue().strip()), 1.5)


if __name__ == "__main__":
    unittest.main()
