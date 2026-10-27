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

"""
Unit tests for ``generate_ground_truth_database``.

A variable takes its reference from its own ``AnalyticGroundTruth`` script,
else from the run-wide script when ``--has-analytic-ground-truth`` is set,
else from native Monte Carlo. Each distinct script runs once. The analytic
path runs real scripts here. Native Monte Carlo needs a built application,
so it is replaced with a mock that records which variables reached it.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from signaloid_benchmarking.automation.database_generator import (
    generate_ground_truth_database,
)
from signaloid_benchmarking.config import RepresentationTypes, VariableTypes
from signaloid_benchmarking.types import BenchmarkingVariable

NAMES = ("alpha", "beta", "gamma", "delta")


def _make_variable(name: str, script: Path | None = None) -> BenchmarkingVariable:
    """
    Create a Distribution variable, optionally with its own analytic script.

    Args:
        name: The variable name, matched against CSV rows.
        script: The variable's own ``AnalyticGroundTruth``, if any.

    Returns:
        The variable.
    """
    return BenchmarkingVariable(
        name=name,
        description=name,
        value_id=f"id_{name}",
        type=VariableTypes.DISTRIBUTION,
        analytic_ground_truth=None if script is None else str(script),
    )


def _write_script(script_path: Path, value: float, counter: Path) -> None:
    """
    Write a ground-truth script that writes one row for every name.

    Every row carries ``value``, so the database shows which script a
    variable's reference came from. Each run appends a line to
    ``counter``, so a test can count the runs.

    Args:
        script_path: Destination path for the script.
        value: The value of every row the script writes.
        counter: File the script appends one line to per run.
    """
    rows = [(name, str(value), "1.0") for name in NAMES]
    script_path.write_text(
        "import csv, sys\n"
        f"open({str(counter)!r}, 'a').write('ran\\n')\n"
        f"rows = {rows!r}\n"
        "with open(sys.argv[2], 'w', newline='') as f:\n"
        "    csv.writer(f).writerows(rows)\n",
        encoding="utf-8",
    )


class TestGenerateGroundTruthDatabase(unittest.TestCase):
    """Which reference each variable gets, and how often a script runs."""

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.tmp_path = Path(tmp_dir.name)
        (self.tmp_path / "app" / "src").mkdir(parents=True)
        self.db_path = self.tmp_path / "ground-truth.db"

        self.scripts: dict[str, Path] = {}
        self.counters: dict[str, Path] = {}
        for label, value in (("first", 1.0), ("second", 2.0), ("run_wide", 3.0)):
            self.scripts[label] = self.tmp_path / f"{label}.py"
            self.counters[label] = self.tmp_path / f"{label}.count"
            _write_script(self.scripts[label], value, self.counters[label])

        native_patch = patch(
            "signaloid_benchmarking.automation.database_generator."
            "generate_database_native_mc"
        )
        self.native_mc: MagicMock = native_patch.start()
        self.addCleanup(native_patch.stop)

    def _generate(
        self, variables: list[BenchmarkingVariable], **overrides: Any
    ) -> None:
        """
        Run the ground-truth stage with defaults that need no build.

        Args:
            variables: The benchmarking variables.
            **overrides: Arguments to change from the defaults.
        """
        arguments: dict[str, Any] = dict(
            has_analytic_ground_truth=False,
            has_native_mc=True,
            benchmarking_variables=variables,
            path_to_application=str(self.tmp_path / "app"),
            path_to_ground_truth_file="",
            ground_truth_size=1,
            ground_truth_db_path=str(self.db_path),
            ground_truth_type=RepresentationTypes.MONTE_CARLO,
            max_num_weighted_samples=1,
            n_processors=1,
            n_adversaries=1,
            adversary_max_size_scalar=1,
            use_clt=False,
            native_executable_name="demo-native-mc",
            native_executable_dir=str(self.tmp_path / "app"),
            demo_cli_args="",
        )
        arguments.update(overrides)
        generate_ground_truth_database(**arguments)

    def _runs(self, label: str) -> int:
        """
        Count how many times a script ran.

        Args:
            label: The script's label.

        Returns:
            The number of runs.
        """
        counter = self.counters[label]
        if not counter.exists():
            return 0

        return len(counter.read_text().splitlines())

    def _reference_values(self) -> dict[str, set[float]]:
        """
        Read back the analytic references written to the database.

        Returns:
            The set of values stored for each variable that has rows.
        """
        if not self.db_path.exists():
            return {}

        connection = sqlite3.connect(self.db_path)
        self.addCleanup(connection.close)
        values: dict[str, set[float]] = {}
        for value_id, position in connection.execute(
            "SELECT ValueId, Position FROM WeightedSamples"
        ):
            values.setdefault(value_id.removeprefix("id_"), set()).add(position)

        return values

    def _native_mc_names(self) -> list[str]:
        """
        Name the variables that reached native Monte Carlo.

        Returns:
            Their names, or an empty list when it was not called.
        """
        if not self.native_mc.called:
            return []

        variables = self.native_mc.call_args.kwargs["benchmarking_variables"]
        return [variable.name for variable in variables]

    def test_each_distinct_script_runs_once_for_the_variables_naming_it(
        self,
    ) -> None:
        """
        Two variables share a script and run it once. The script also
        writes rows for every other name, and those rows go nowhere.
        """
        variables = [
            _make_variable("alpha", self.scripts["first"]),
            _make_variable("beta", self.scripts["first"]),
            _make_variable("gamma", self.scripts["second"]),
            _make_variable("delta"),
        ]

        self._generate(variables)

        self.assertEqual(self._runs("first"), 1)
        self.assertEqual(self._runs("second"), 1)
        self.assertEqual(
            self._reference_values(),
            {"alpha": {1.0}, "beta": {1.0}, "gamma": {2.0}},
        )
        self.assertEqual(self._native_mc_names(), ["delta"])

    def test_run_wide_script_covers_variables_that_declare_none(self) -> None:
        """With the run-wide flag, only a declared script overrides it."""
        variables = [
            _make_variable("alpha", self.scripts["first"]),
            _make_variable("beta"),
        ]

        self._generate(
            variables,
            has_analytic_ground_truth=True,
            path_to_ground_truth_file=str(self.scripts["run_wide"]),
            ground_truth_type=RepresentationTypes.WEIGHTED_SAMPLES,
        )

        self.assertEqual(self._runs("first"), 1)
        self.assertEqual(self._runs("run_wide"), 1)
        self.assertEqual(self._reference_values(), {"alpha": {1.0}, "beta": {3.0}})
        self.assertFalse(self.native_mc.called)

    def test_a_declared_script_that_is_the_run_wide_one_runs_once(self) -> None:
        """The same file named both ways is still one script."""
        variables = [
            _make_variable("alpha", self.scripts["run_wide"]),
            _make_variable("beta"),
        ]

        self._generate(
            variables,
            has_analytic_ground_truth=True,
            path_to_ground_truth_file=str(self.scripts["run_wide"]),
            ground_truth_type=RepresentationTypes.WEIGHTED_SAMPLES,
        )

        self.assertEqual(self._runs("run_wide"), 1)

    def test_native_mc_receives_the_run_wide_ground_truth_type(self) -> None:
        """An analytic reference elsewhere does not change the native one."""
        variables = [
            _make_variable("alpha", self.scripts["first"]),
            _make_variable("beta"),
        ]

        self._generate(variables, ground_truth_type=RepresentationTypes.MONTE_CARLO)

        self.assertFalse(self.native_mc.call_args.kwargs["weighted_samples"])

    def test_an_analytic_only_run_needs_no_native_build(self) -> None:
        """Every variable is covered, so a missing native build is fine."""
        variables = [_make_variable("alpha", self.scripts["first"])]

        self._generate(variables, has_native_mc=False)

        self.assertEqual(self._reference_values(), {"alpha": {1.0}})
        self.assertFalse(self.native_mc.called)

    def test_an_uncovered_variable_without_a_native_build_is_named(self) -> None:
        """
        The error says which variables had no reference, and comes before
        any script runs, since a script can take a while.
        """
        variables = [
            _make_variable("alpha", self.scripts["first"]),
            _make_variable("beta"),
        ]

        with self.assertRaises(RuntimeError) as caught:
            self._generate(variables, has_native_mc=False)

        self.assertIn("beta", str(caught.exception))
        self.assertNotIn("alpha", str(caught.exception))
        self.assertEqual(self._runs("first"), 0)


if __name__ == "__main__":
    unittest.main()
