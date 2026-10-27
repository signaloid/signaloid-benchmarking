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

"""Unit tests for ``Benchmark.bind_timing_script``.

``bind_timing_script`` binds the seven ``run_timing_script`` kwargs shared by
every timing pass into a ``functools.partial``. Callers then supply only the
per-call ``variable_index`` and one mode flag. It replaced four hand-maintained
copies of the same kwarg bundle, so these tests pin exactly which kwargs are
bound (and that the per-call args are deliberately left unbound).
"""

import contextlib
import functools
import io
import unittest
from unittest.mock import patch

from signaloid_benchmarking.automation.benchmark import Benchmark
from signaloid_benchmarking.automation.build import run_timing_script
from signaloid_benchmarking.types import BenchmarkingVariable


class TestBindTimingScript(unittest.TestCase):
    def _make_benchmark(self) -> Benchmark:
        benchmark = Benchmark(
            path_to_application="/tmp/app",
            representation_types=["Athens"],
            representation_sizes=[16, 32],
            demo_cli_args="--demo",
        )
        # Normally populated by get_application_info / get_machine_info.
        benchmark.all_outputs_cla = "-S 1"
        benchmark.benchmarking_variables = []
        benchmark.application_name = "app"
        benchmark.application_version = "v0"
        return benchmark

    def test_binds_shared_kwargs(self) -> None:
        benchmark = self._make_benchmark()

        bound = benchmark.bind_timing_script()

        self.assertIsInstance(bound, functools.partial)
        self.assertIs(bound.func, run_timing_script)
        self.assertEqual(
            bound.keywords,
            {
                "benchmarking_variables": [],
                "demo_cli_args": "--demo",
                "representation_types": benchmark.representation_types,
                "representation_sizes": benchmark.representation_sizes,
                "correlations": benchmark.correlations,
                "logs_dir": benchmark.logs_dir,
                "intermediate_timings_path": benchmark.intermediate_timings_path(),
            },
        )

    def test_per_call_args_not_prebound(self) -> None:
        bound = self._make_benchmark().bind_timing_script()
        for per_call_key in (
            "variable_index",
            "timing",
            "native_mc_timing",
            "tracing",
        ):
            self.assertNotIn(per_call_key, bound.keywords)


class TestResolveSkippedConfigurations(unittest.TestCase):
    """Each variable's configurations that cannot run, decided once."""

    # The SDK lacks the Europa autocorrelation library in this fixture.
    _NO_LIBRARY = "the UxHw SDK has no libuxhw_runtime_Europa_8_Autocorrelation.a"

    def setUp(self) -> None:
        self.asked: list[tuple[str, int, str]] = []

        def skip_reason(**kwargs: object) -> str | None:
            key = (
                str(kwargs["representation_type"]),
                int(str(kwargs["representation_size"])),
                str(kwargs["correlation"]),
            )
            self.asked.append(key)
            if key == ("Europa", 8, "Autocorrelation"):
                return self._NO_LIBRARY

            return None

        reason_patch = patch(
            "signaloid_benchmarking.automation.benchmark.configuration_skip_reason",
            side_effect=skip_reason,
        )
        reason_patch.start()
        self.addCleanup(reason_patch.stop)

        self.benchmark = Benchmark(
            path_to_application="/tmp/app",
            representation_types=["Athens", "Europa"],
            representation_sizes=[8],
            correlations=["Disabled", "Autocorrelation"],
        )

    def _resolve(self, *variables: BenchmarkingVariable) -> str:
        """
        Resolve the skips of the given variables.

        Args:
            *variables: The benchmarking variables.

        Returns:
            What was printed.
        """
        self.benchmark.benchmarking_variables = list(variables)
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            self.benchmark.resolve_skipped_configurations()

        return printed.getvalue()

    def test_a_skipped_configuration_is_recorded_with_its_reason(self) -> None:
        """Europa runs only without autocorrelation tracking."""
        variable = BenchmarkingVariable(name="a", description="a")

        self._resolve(variable)

        self.assertEqual(
            variable.skipped_configurations,
            {"Europa-8-Autocorrelation": self._NO_LIBRARY},
        )
        self.assertEqual(
            variable.benchmarked_configurations(
                representation_types=self.benchmark.representation_types,
                representation_sizes=self.benchmark.representation_sizes,
                correlations=self.benchmark.correlations,
            ),
            ["Athens-8", "Athens-8-Autocorrelation", "Europa-8"],
        )

    def test_each_skip_is_logged_once_with_the_variables_it_affects(
        self,
    ) -> None:
        """Two variables sharing a skip produce one line naming both."""
        printed = self._resolve(
            BenchmarkingVariable(name="a", description="a"),
            BenchmarkingVariable(name="b", description="b"),
        )

        self.assertEqual(
            printed.strip(),
            f"Skipping Europa-8-Autocorrelation for a, b, because {self._NO_LIBRARY}.",
        )

    def test_each_distinct_configuration_is_asked_about_once(self) -> None:
        """Variables sharing a matrix do not repeat the SDK question."""
        self._resolve(
            BenchmarkingVariable(name="a", description="a"),
            BenchmarkingVariable(name="b", description="b"),
        )

        self.assertEqual(len(self.asked), 4)
        self.assertEqual(len(set(self.asked)), 4)

    def test_a_variable_is_asked_about_its_own_matrix(self) -> None:
        """A variable declaring only Athens is never asked about Europa."""
        variable = BenchmarkingVariable(
            name="a", description="a", representation_types=["Athens"]
        )

        self._resolve(variable)

        self.assertEqual(variable.skipped_configurations, {})
        self.assertNotIn("Europa", {key[0] for key in self.asked})

    def test_a_variable_with_nothing_left_to_run_stops_the_run(self) -> None:
        """
        Europa with autocorrelation only would finish having measured
        nothing, so it stops before any build, naming the variable and why.
        """
        variable = BenchmarkingVariable(
            name="a",
            description="a",
            representation_types=["Europa"],
            correlations=["Autocorrelation"],
        )

        with self.assertRaises(ValueError) as caught:
            self._resolve(variable)

        message = str(caught.exception)
        self.assertIn("'a'", message)
        self.assertIn("Europa-8-Autocorrelation", message)
        self.assertIn(self._NO_LIBRARY, message)


class TestVariablesLeftAfterRunsDie(unittest.TestCase):
    """A variable whose every UxHw run died is left out, not fatal."""

    def setUp(self) -> None:
        self.benchmark = Benchmark(
            path_to_application="/tmp/app",
            representation_types=["Jupiter"],
            representation_sizes=[8, 16],
            correlations=["Disabled"],
        )
        self.all_died = BenchmarkingVariable(name="a", description="a")
        self.all_died.skipped_configurations = {
            "Jupiter-8": "its tracing run was killed by signal 9 (SIGKILL), or exited with code 137",
            "Jupiter-16": "its tracing run was killed by signal 9 (SIGKILL), or exited with code 137",
        }
        self.some_died = BenchmarkingVariable(name="b", description="b")
        self.some_died.skipped_configurations = {
            "Jupiter-16": "its timing run 1 of 4 exited with code 1",
        }
        self.some_died.emcc_results.equiv_mc_list = [100]

    def _timed_indices(self, method: str, mode: str) -> list[int]:
        """
        Run one timing step with the script stubbed out.

        Args:
            method: ``generate_uxhw_timings`` or
                ``generate_native_mc_timings``.
            mode: The pass flag that step sets.

        Returns:
            The index of each variable the step timed.
        """
        self.benchmark.benchmarking_variables = [self.all_died, self.some_died]
        with (
            patch.object(self.benchmark, "bind_timing_script") as bind,
            patch("signaloid_benchmarking.automation.benchmark.load_emcc_data"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            getattr(self.benchmark, method)()

        return [
            call.kwargs["variable_index"]
            for call in bind.return_value.call_args_list
            if call.kwargs.get(mode)
        ]

    def test_the_run_goes_on_while_one_variable_has_a_configuration(self) -> None:
        self.benchmark.benchmarking_variables = [self.all_died, self.some_died]

        self.benchmark.require_a_configuration_left()

    def test_the_run_stops_once_no_variable_has_one(self) -> None:
        """The message gives every reason, so the cause is plain."""
        self.some_died.skipped_configurations["Jupiter-8"] = (
            "its tracing run exited with code 2"
        )
        self.benchmark.benchmarking_variables = [self.all_died, self.some_died]

        with self.assertRaises(RuntimeError) as caught:
            self.benchmark.require_a_configuration_left()

        message = str(caught.exception)
        self.assertIn("Jupiter-8 of a is skipped because its tracing run", message)
        self.assertIn("Jupiter-8 of b is skipped because its tracing run", message)
        self.assertIn("exited with code 2", message)

    def test_step_10_times_only_variables_with_a_configuration_left(self) -> None:
        self.assertEqual(
            self._timed_indices("generate_uxhw_timings", "timing"),
            [1],
        )

    def test_step_11_times_only_variables_with_equivalent_counts(self) -> None:
        """
        A variable whose every tracing run died has no equivalent Monte
        Carlo count to run the native build at.
        """
        self.assertEqual(
            self._timed_indices("generate_native_mc_timings", "native_mc_timing"),
            [1],
        )


if __name__ == "__main__":
    unittest.main()


class TestScceWithDynamicInstructions(unittest.TestCase):
    """Intel PIN cannot reach a run on SCCE, and the user is told so."""

    def test_both_flags_warn_at_construction(self) -> None:
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            Benchmark(
                path_to_application="/app",
                run_uxhw_scce=True,
                measure_dynamic_instructions=True,
            )

        self.assertIn("Warning! Intel PIN cannot instrument", printed.getvalue())

    def test_scce_alone_does_not_warn(self) -> None:
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            Benchmark(path_to_application="/app", run_uxhw_scce=True)

        self.assertEqual(printed.getvalue(), "")
