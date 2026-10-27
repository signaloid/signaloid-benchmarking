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
Unit tests for the representation matrix ``run_timing_script`` hands bash,
and for the skips it reads back.

The timing script is called once per variable. Each call must sweep that
variable's own matrix, which is the run-wide one unless its signaloid.yaml
entry declares one. A UxHw run that dies is recorded by the script and
read back as a skip of its configuration. Bash itself is not run here, the
command it would be given is captured instead.
"""

import contextlib
import io
import os
import tempfile
import unittest
from unittest.mock import patch

from signaloid_benchmarking.automation.build import run_timing_script
from signaloid_benchmarking.config import TimingFormat
from signaloid_benchmarking.types import BenchmarkingVariable


class TestRunTimingScriptMatrix(unittest.TestCase):
    """The bash arrays of a per-variable timing pass."""

    def setUp(self) -> None:
        # run_timing_script exports its settings through os.environ, so
        # snapshot and restore it to keep them out of other tests.
        snapshot = dict(os.environ)

        def restore_environ() -> None:
            os.environ.clear()
            os.environ.update(snapshot)

        self.addCleanup(restore_environ)

        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.logs_dir = tmp_dir.name

        call_patch = patch(
            "signaloid_benchmarking.automation.build.subprocess.call",
            return_value=0,
        )
        self.bash = call_patch.start()
        self.addCleanup(call_patch.stop)

        declaring = BenchmarkingVariable(
            name="outputVariables[0]", description="first", cla="-S 0"
        )
        declaring.representation_types = ["Athens"]
        declaring.representation_sizes = [8, 16]
        declaring.correlations = ["Autocorrelation"]
        declaring.skipped_configurations = {
            "Athens-16-Autocorrelation": "the UxHw SDK has no library for it"
        }
        inheriting = BenchmarkingVariable(
            name="outputVariables[1]", description="second", cla="-S 1"
        )
        self.variables = [declaring, inheriting]

    def _bash_command(self, variable_index: int) -> str:
        """
        Run a timing pass for one variable and capture the bash command.

        Args:
            variable_index: The variable to time.

        Returns:
            The command bash would have been given.
        """
        run_timing_script(
            benchmarking_variables=self.variables,
            demo_cli_args="",
            representation_types=["Jupiter"],
            representation_sizes=[32, 64],
            correlations=["Disabled"],
            logs_dir=self.logs_dir,
            intermediate_timings_path=os.path.join(self.logs_dir, "timings"),
            timing=True,
            variable_index=variable_index,
        )

        command: str = self.bash.call_args.args[0][2]

        return command

    def test_a_declaring_variable_sweeps_its_own_matrix(self) -> None:
        """Its signaloid.yaml values reach bash, not the run-wide ones."""
        command = self._bash_command(0)

        self.assertIn("REPRESENTATION_TYPES=(Athens)", command)
        self.assertIn("REPRESENTATION_SIZES=(8 16)", command)
        self.assertIn("CORRELATION_TRACKING_TYPES=(Autocorrelation)", command)

    def test_an_inheriting_variable_sweeps_the_run_wide_matrix(self) -> None:
        """A variable that declares nothing is timed as before."""
        command = self._bash_command(1)

        self.assertIn("REPRESENTATION_TYPES=(Jupiter)", command)
        self.assertIn("REPRESENTATION_SIZES=(32 64)", command)
        self.assertIn("CORRELATION_TRACKING_TYPES=(Disabled)", command)

    def test_bash_is_handed_the_variables_skipped_configurations(self) -> None:
        """The timing layer skips what Python decided, and nothing else."""
        self.assertIn(
            "SKIPPED_CONFIGURATIONS=(Athens-16-Autocorrelation)", self._bash_command(0)
        )

    def test_a_variable_with_no_skips_hands_bash_an_empty_list(self) -> None:
        """An empty list, so bash's set -u never sees the array unset."""
        self.assertIn("SKIPPED_CONFIGURATIONS=()", self._bash_command(1))


class TestRunTimingScriptFailedConfigurations(unittest.TestCase):
    """A UxHw run that dies becomes a skip of its configuration."""

    def setUp(self) -> None:
        snapshot = dict(os.environ)

        def restore_environ() -> None:
            os.environ.clear()
            os.environ.update(snapshot)

        self.addCleanup(restore_environ)

        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.logs_dir = tmp_dir.name
        self.record = os.path.join(
            self.logs_dir, TimingFormat.FAILED_CONFIGURATIONS_FILENAME
        )

        # Lines the stand-in for bash writes to the failure record.
        self.died: list[str] = []

        def bash(command: list[str]) -> int:
            path = os.environ[TimingFormat.FAILED_CONFIGURATIONS_FILE_ENV_VAR]
            if self.died:
                with open(path, "a") as file:
                    file.writelines(f"{line}\n" for line in self.died)

            return 0

        call_patch = patch(
            "signaloid_benchmarking.automation.build.subprocess.call",
            side_effect=bash,
        )
        call_patch.start()
        self.addCleanup(call_patch.stop)

        self.variable = BenchmarkingVariable(
            name="outputVariables[0]", description="first", cla="-S 0"
        )
        self.variable.skipped_configurations = {
            "Jupiter-64": "it is above --max-jupiter-size 32, a memory guard"
        }

    def _run(self, mode: str) -> str:
        """
        Run one pass of the variable over Jupiter 16 to 64.

        Args:
            mode: ``tracing``, ``timing`` or ``native_mc_timing``.

        Returns:
            What the pass printed.
        """
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            run_timing_script(
                benchmarking_variables=[self.variable],
                demo_cli_args="",
                representation_types=["Jupiter"],
                representation_sizes=[16, 32, 64],
                correlations=["Disabled"],
                logs_dir=self.logs_dir,
                intermediate_timings_path=os.path.join(self.logs_dir, "timings"),
                variable_index=0,
                **{mode: True},
            )

        return printed.getvalue()

    def test_a_run_that_died_is_skipped_with_its_reason(self) -> None:
        """It joins the skips decided before any build, and is logged."""
        reason = (
            "its tracing run was killed by signal 9 (SIGKILL), or exited with code 137"
        )
        self.died = [f"Jupiter-32\t{reason}"]

        printed = self._run("tracing")

        self.assertEqual(self.variable.skipped_configurations["Jupiter-32"], reason)
        self.assertIn(
            f"Skipping Jupiter-32 for outputVariables[0], because {reason}.",
            printed,
        )
        self.assertEqual(
            self.variable.benchmarked_configurations(
                representation_types=["Jupiter"],
                representation_sizes=[16, 32, 64],
                correlations=["Disabled"],
            ),
            ["Jupiter-16"],
        )
        self.assertFalse(os.path.exists(self.record), "the record is left behind")

    def test_a_record_left_by_an_earlier_pass_is_not_read(self) -> None:
        with open(self.record, "w") as file:
            file.write("Jupiter-16\tits timing run died in another pass\n")

        self._run("timing")

        self.assertNotIn("Jupiter-16", self.variable.skipped_configurations)

    def test_a_native_pass_records_nothing(self) -> None:
        """Only the UxHw passes run UxHw binaries."""
        self.variable.emcc_results.equiv_mc_list = [100]
        self.died = ["Jupiter-16\tits timing run died"]

        self._run("native_mc_timing")

        self.assertNotIn("Jupiter-16", self.variable.skipped_configurations)

    def test_a_variable_left_with_nothing_says_so(self) -> None:
        self.died = [
            "Jupiter-16\tits tracing run exited with code 1",
            "Jupiter-32\tits tracing run exited with code 1",
        ]

        printed = self._run("tracing")

        self.assertIn(
            "Every configuration of outputVariables[0] is skipped, so nothing "
            "more is measured for it.",
            printed,
        )


if __name__ == "__main__":
    unittest.main()
