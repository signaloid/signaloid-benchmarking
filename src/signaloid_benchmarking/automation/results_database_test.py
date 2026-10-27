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

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from signaloid_benchmarking.automation import results_database


class TestRecordInferenceObservations(unittest.TestCase):
    """Repeated and unnamed outputs keep distinct identities."""

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.database_path = str(Path(tmp_dir.name) / "tracing.db")

    def _rows(self) -> list[tuple[str | None, int, str | None]]:
        with closing(sqlite3.connect(self.database_path)) as connection:
            return connection.execute(
                "SELECT Expression_Name, Assignment_Index, ValueId FROM "
                "TracingTable ORDER BY rowid"
            ).fetchall()

    def test_repeated_output_gets_successive_assignment_indices(self) -> None:
        results_database.record_inference(
            self.database_path,
            [1.0, 2.0],
            output_names=["y", "y"],
            declaration=("main", "main.c", 7, 7),
        )

        rows = self._rows()

        self.assertEqual([row[:2] for row in rows], [("y", 1), ("y", 2)])
        self.assertNotEqual(rows[0][2], rows[1][2])

    def test_unnamed_outputs_are_not_observations_of_one_another(self) -> None:
        results_database.record_inference(
            self.database_path,
            [1.0, 2.0, 3.0],
            output_names=["y"],
            declaration=("main", "main.c", 7, 7),
        )

        self.assertEqual(
            [row[:2] for row in self._rows()], [("y", 1), (None, 1), (None, 1)]
        )
