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
import sqlite3
import subprocess
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from signaloid_benchmarking.automation.benchmarking_utils import (
    parse_timing_intermediate_stream,
)
from signaloid_benchmarking.automation.measurement_loader import (
    load_measurement_data,
)
from signaloid_benchmarking.automation.scce import (
    BUILD_SCRIPT,
    RUN_SCRIPT,
    ScceRunner,
    _ScceTask,
    _match_core,
    _parse_task_stdout,
    scce_configuration_skip_reason,
    _ux_strings_for,
)
from signaloid_benchmarking.config import (
    Correlations,
    EquivMC,
    RepresentationTypes,
    TimingFormat,
    VariableTypes,
)
from signaloid_benchmarking.equivalent_mc.load import _load_uxhw
from signaloid_benchmarking.types import BenchmarkingVariable

# What a task prints under ``-j``. The Ux string is one a task returned, a
# single Dirac delta at 9.407459.
_UX_STRING = (
    "9.407459Ux0400000000000000004022D09E7337B54A"
    "000000014022D09E7337B54A8000000000000000"
)
_TASK_STDOUT = (
    "{\n"
    '  "description": "A demo.",\n'
    '  "plots": [\n'
    "    {\n"
    '      "variableID": "outputVariables[0]",\n'
    '      "variableSymbol": "outputVariables[0]",\n'
    '      "variableDescription": "Call",\n'
    f'      "values": ["{_UX_STRING}"],\n'
    '      "particleValues": [9.407459]\n'
    "    }\n"
    "  ]\n"
    "}\n"
)

# What the CLI prints for that stdout: progress text, then the stdout as one
# JSON string literal.
_CLI_STDOUT = "- Downloading output...\n✔ " + json.dumps(_TASK_STDOUT) + "\n"

# What a task that was not single-shot prints, two values of the one
# output, the second a single Dirac delta at 10.0.
_SECOND_UX_STRING = (
    "10.000000Ux0400000000000000004024000000000000"
    "0000000140240000000000008000000000000000"
)
_TWO_VALUES_OUTPUT = json.loads(_TASK_STDOUT)
_TWO_VALUES_OUTPUT["plots"][0]["values"].append(_SECOND_UX_STRING)
_TWO_VALUES_OUTPUT["plots"][0]["particleValues"].append(10.0)
_TWO_VALUES_CLI_STDOUT = (
    "- Downloading output...\n✔ " + json.dumps(json.dumps(_TWO_VALUES_OUTPUT)) + "\n"
)

_CORE = {
    "Name": "C0Pro-XXS",
    "Class": "C0Pro",
    "Microarchitecture": "Athens",
    "Precision": 16,
    "CorrelationTracking": "Disable",
    "CoreID": "cor_xxs",
}
_CORES = [
    _CORE,
    {
        "Name": "C0-XS+",
        "Class": "C0",
        "Microarchitecture": "Athens",
        "Precision": 32,
        "CorrelationTracking": "Autocorrelation",
        "CoreID": "cor_c0",
    },
]


def _fake_helpers(tmp_path: Path) -> Path:
    """Two executable stand-ins for the helper scripts, in a new directory."""
    helpers = tmp_path / "helpers"
    helpers.mkdir()
    for script in (BUILD_SCRIPT, RUN_SCRIPT):
        path = helpers / script
        path.write_text("#!/bin/sh\n")
        path.chmod(0o755)

    return helpers


class TestParseTaskStdout(unittest.TestCase):
    def test_stdout_printed_as_a_string_literal_is_decoded(self) -> None:
        output = _parse_task_stdout(_CLI_STDOUT)

        self.assertEqual(output["plots"][0]["values"], [_UX_STRING])

    def test_stdout_printed_as_it_is_is_read(self) -> None:
        output = _parse_task_stdout("- Downloading output...\n" + _TASK_STDOUT)

        self.assertEqual(output["description"], "A demo.")

    def test_text_after_the_json_is_ignored(self) -> None:
        output = _parse_task_stdout(_TASK_STDOUT + "CPU time used: 1.0 seconds\n")

        self.assertEqual(output["description"], "A demo.")

    def test_no_json_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "No JSON"):
            _parse_task_stdout("CPU time used: 1.0 seconds\n")

    def test_a_json_value_that_is_not_an_object_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "no JSON object"):
            _parse_task_stdout('"just text"\n')


class TestUxStringsFor(unittest.TestCase):
    def test_values_of_the_named_variable(self) -> None:
        output = _parse_task_stdout(_TASK_STDOUT)

        self.assertEqual(_ux_strings_for(output, "outputVariables[0]"), [_UX_STRING])

    def test_missing_variable_raises(self) -> None:
        output = _parse_task_stdout(_TASK_STDOUT)

        with self.assertRaisesRegex(ValueError, "outputVariables\\[1\\]"):
            _ux_strings_for(output, "outputVariables[1]")


class TestMatchCore(unittest.TestCase):
    def test_matches_on_specification_not_name(self) -> None:
        core = _match_core(
            _CORES,
            representation_type=RepresentationTypes.ATHENS,
            representation_size=16,
            correlation=Correlations.DISABLED,
        )

        self.assertIsNotNone(core)
        assert core is not None
        self.assertEqual(core["CoreID"], "cor_xxs")

    def test_other_core_class_does_not_match(self) -> None:
        core = _match_core(
            _CORES,
            representation_type=RepresentationTypes.ATHENS,
            representation_size=32,
            correlation=Correlations.AUTOCORRELATION,
        )

        self.assertIsNone(core)


class TestSkipReason(unittest.TestCase):
    def test_unsupported_representation_type_is_skipped(self) -> None:
        reason = scce_configuration_skip_reason(
            representation_type=RepresentationTypes.ATLAS,
            representation_size=16,
            max_jupiter_size=32,
        )

        self.assertEqual(reason, "SCCE has no Atlas core")

    def test_jupiter_above_guard_is_skipped(self) -> None:
        reason = scce_configuration_skip_reason(
            representation_type=RepresentationTypes.JUPITER,
            representation_size=64,
            max_jupiter_size=32,
        )

        self.assertIsNotNone(reason)
        assert reason is not None
        self.assertIn("max-jupiter-size", reason)

    def test_athens_runs(self) -> None:
        reason = scce_configuration_skip_reason(
            representation_type=RepresentationTypes.ATHENS,
            representation_size=512,
            max_jupiter_size=32,
        )

        self.assertIsNone(reason)


class TestScceRunner(unittest.TestCase):
    """
    The runner with the cloud mocked out: the tracing database it writes
    is read back by the pipeline's own loader, and the timing it writes
    is read back by the measurement loader.
    """

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.tmp_path = Path(tmp_dir.name)
        self.tracing_db_path = str(self.tmp_path / "tracing.db")
        self.runner = ScceRunner(
            path_to_application=str(self.tmp_path),
            path_to_cli_helpers=str(_fake_helpers(self.tmp_path)),
            application_name="demo",
            application_version="abc1234",
            demo_cli_args="",
        )
        self.variable = BenchmarkingVariable(
            value_id="v",
            name="outputVariables[0]",
            description="Call",
            program="main",
            path="/app/src/main.c",
            line_number="44",
            file_name="main.c",
            type=VariableTypes.DISTRIBUTION,
            cla="-S 0",
        )
        self.task = _ScceTask(
            task_id="tsk_1",
            stdout=_CLI_STDOUT,
            stats={
                "ProcessorTime": 0.5,
                "ExecutionTimeInMilliseconds": 1500,
                "DynamicInstructions": 4321,
            },
        )

    def test_missing_helper_script_is_refused(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, BUILD_SCRIPT):
            ScceRunner(
                path_to_application=str(self.tmp_path),
                path_to_cli_helpers=str(self.tmp_path),
                application_name="demo",
                application_version="abc1234",
                demo_cli_args="",
            )

    def _trace(self, arguments: str = "-S 0 -j") -> None:
        with (
            patch.object(self.runner, "_core", return_value=_CORE),
            patch.object(self.runner, "_build_id", return_value="bld_1"),
            patch.object(self.runner, "_run_task", return_value=self.task) as run,
        ):
            self.runner.trace_variable(
                variable=self.variable,
                tracing_db_path=self.tracing_db_path,
                representation_types=[RepresentationTypes.ATHENS],
                representation_sizes=[16],
                correlations=[Correlations.DISABLED],
            )
        run.assert_called_once_with(build_id="bld_1", arguments=arguments)

    def test_traced_values_are_read_back_by_the_pipeline_loader(self) -> None:
        self._trace()

        values = _load_uxhw(
            db_path=self.tracing_db_path,
            table=EquivMC.TRACING_TABLE,
            target_expression=self.variable.name,
            ur_types=[RepresentationTypes.ATHENS],
            ur_sizes=[16],
        )

        self.assertEqual(len(values), 1)
        self.assertEqual(repr(values[0]), "Athens-16")
        self.assertEqual(values[0].correlation_tracking, Correlations.DISABLED)
        # The particle value in the string is the position, rounded.
        self.assertAlmostEqual(values[0].dv.positions[0], 9.407459, places=5)

    def test_tracing_database_records_the_run(self) -> None:
        self._trace()

        with closing(sqlite3.connect(self.tracing_db_path)) as connection:
            runs = connection.execute(
                "SELECT Model, DeviceVariant, DevicePath, Inputs FROM Runs"
            ).fetchall()
            rows = connection.execute(
                f"SELECT Expression_Name, Expression_Subprogram, "
                f"Expression_DeclarationFileName, "
                f"Expression_DeclarationLineNumber, UR_Type, UR_Order "
                f'FROM "{EquivMC.TRACING_TABLE}"'
            ).fetchall()

        self.assertEqual(runs, [("demo", "C0Pro-XXS", "tsk_1", "-S 0 -j")])
        self.assertEqual(
            rows,
            [("outputVariables[0]", "main", "/app/src/main.c", 44, "Athens", 16)],
        )

    def test_skipped_configuration_is_not_run(self) -> None:
        self.variable.skipped_configurations = {"Athens-16": "SCCE has no such core"}

        with (
            patch.object(self.runner, "_core") as core,
            patch.object(self.runner, "_run_task") as run_task,
        ):
            self.runner.trace_variable(
                variable=self.variable,
                tracing_db_path=self.tracing_db_path,
                representation_types=[RepresentationTypes.ATHENS],
                representation_sizes=[16],
                correlations=[Correlations.DISABLED],
            )

        core.assert_not_called()
        run_task.assert_not_called()
        self.assertFalse(os.path.exists(self.tracing_db_path))

    def test_task_printing_several_values_is_refused(self) -> None:
        # A single-shot run prints one value per output, so more means the
        # task was something else and its values are not a UxHw result.
        self.task = _ScceTask(
            task_id="tsk_1", stdout=_TWO_VALUES_CLI_STDOUT, stats=self.task.stats
        )

        with self.assertRaisesRegex(ValueError, "printed 2 values for outputVariables"):
            self._trace()

        self.assertFalse(os.path.exists(self.tracing_db_path))

    def test_task_without_processor_time_is_refused(self) -> None:
        self.task = _ScceTask(task_id="tsk_1", stdout=_CLI_STDOUT, stats={})

        with self.assertRaisesRegex(RuntimeError, "ProcessorTime"):
            self._trace()

    def test_cores_are_listed_once(self) -> None:
        listed: list[list[str]] = []

        def cli_json(arguments: list[str]) -> dict[str, list[dict[str, object]]]:
            listed.append(arguments)
            return {"Cores": _CORES if "--default" not in arguments else []}

        with patch("signaloid_benchmarking.automation.scce._cli_json", cli_json):
            for size in (16, 16, 32):
                try:
                    self.runner._core(
                        representation_type=RepresentationTypes.ATHENS,
                        representation_size=size,
                        correlation=Correlations.DISABLED,
                    )
                except (RuntimeError, KeyError):
                    # Size 32 matches nothing and the fake cannot create.
                    pass

        self.assertEqual(
            [arguments[:2] for arguments in listed if arguments[1] == "list"],
            [["cores", "list"], ["cores", "list"]],
        )

    def _load_timings(self) -> list[dict]:
        intermediate = str(self.tmp_path / "timings.intermediate")
        with patch.dict(os.environ, {TimingFormat.APPEND_TO_OUTPUT_FILE_ENV_VAR: "0"}):
            self.runner.write_uxhw_timings(
                intermediate_timings_path=intermediate,
                benchmarking_variables=[self.variable],
            )
            # The native pass that follows must append to this file.
            self.assertEqual(
                os.environ[TimingFormat.APPEND_TO_OUTPUT_FILE_ENV_VAR], "1"
            )
        with open(intermediate, encoding="utf-8") as handle:
            runs = parse_timing_intermediate_stream(handle)
        load_measurement_data(
            benchmarking_variables=[self.variable],
            runs=runs,
            demo_cli_args="",
            representation_sizes=[16],
            representation_types=[RepresentationTypes.ATHENS],
            correlations=[Correlations.DISABLED],
        )

        return runs

    def test_timing_is_read_back_by_the_measurement_loader(self) -> None:
        self._trace()

        runs = self._load_timings()

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0][TimingFormat.META_KEY_UXHW_SDK_VERSION], "SCCE")
        self.assertEqual(runs[0][TimingFormat.META_KEY_UXHW_TARGET_REPETITIONS], "1")
        self.assertEqual(runs[0][TimingFormat.META_KEY_COMMAND_LINE_ARGUMENTS], "-S 0")
        measurement = self.variable.timing_measurements.measurement_dict["Athens-16"]
        self.assertEqual(measurement["In Application Time"], 0.5)
        self.assertEqual(measurement["Database Time"], 0.5)
        self.assertEqual(measurement["End-to-End Time"], 1.5)
        self.assertEqual(measurement["Database Dyn. Inst. Count"], 4321.0)
        self.assertIsNone(measurement["PIN Dyn. Inst. Count"])

    def test_application_json_flag_survives_the_loader(self) -> None:
        # The application's own -j is part of the variable's recorded
        # arguments and must still match after the loader's normalisation.
        self.variable.cla = "-j 8 -S 0"
        self._trace(arguments="-j 8 -S 0 -j")

        self._load_timings()

        self.assertIn("Athens-16", self.variable.timing_measurements.measurement_dict)


def _git(*arguments: str, cwd: str) -> str:
    # The identity and the signing setting are given here, so the test does
    # not depend on the user's global git configuration, which may require a
    # GPG passphrase that no terminal is there to supply.
    return subprocess.check_output(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "commit.gpgsign=false",
            *arguments,
        ],
        cwd=cwd,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


class _ClonedApplication(unittest.TestCase):
    """A bare origin and a clone of it stand in for GitHub and the demo."""

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        tmp_path = Path(tmp_dir.name)
        origin = tmp_path / "origin.git"
        _git("init", "--bare", "--quiet", "-b", "main", str(origin), cwd=str(tmp_path))
        self.clone = str(tmp_path / "clone")
        _git("clone", "--quiet", str(origin), self.clone, cwd=str(tmp_path))
        (tmp_path / "clone" / "a").write_text("a\n")
        (tmp_path / "clone" / "src").mkdir()
        (tmp_path / "clone" / "src" / "main.c").write_text("int main(void) {}\n")
        _git("add", "a", "src/main.c", cwd=self.clone)
        _git("commit", "--quiet", "-m", "a", cwd=self.clone)
        _git("push", "--quiet", "origin", "main", cwd=self.clone)
        self.runner = ScceRunner(
            path_to_application=self.clone,
            path_to_cli_helpers=str(_fake_helpers(tmp_path)),
            application_name="demo",
            application_version="abc1234",
            demo_cli_args="",
        )


class TestCheckoutMatchesRemote(_ClonedApplication):
    def test_matching_checkout_passes(self) -> None:
        self.runner.check_checkout_matches_remote()

    def test_checkout_behind_origin_is_refused(self) -> None:
        # Someone else pushed, as happens to a demo checkout left alone.
        other = self.clone + "-other"
        _git("clone", "--quiet", self.clone, other, cwd=self.clone)
        _git(
            "remote",
            "set-url",
            "origin",
            _git("remote", "get-url", "origin", cwd=self.clone),
            cwd=other,
        )
        Path(other, "b").write_text("b\n")
        _git("add", "b", cwd=other)
        _git("commit", "--quiet", "-m", "b", cwd=other)
        _git("push", "--quiet", "origin", "main", cwd=other)

        with self.assertRaisesRegex(RuntimeError, "pull or push first"):
            self.runner.check_checkout_matches_remote()

    def test_checkout_ahead_of_origin_is_refused(self) -> None:
        Path(self.clone, "c").write_text("c\n")
        _git("add", "c", cwd=self.clone)
        _git("commit", "--quiet", "-m", "c", cwd=self.clone)

        with self.assertRaisesRegex(RuntimeError, "pull or push first"):
            self.runner.check_checkout_matches_remote()


class TestWorkingTreeIsCommitted(_ClonedApplication):
    """The code under the checked paths must be what the commit has."""

    def test_clean_tree_passes(self) -> None:
        self.runner.check_working_tree_is_committed()

    def test_edited_source_is_refused(self) -> None:
        Path(self.clone, "src", "main.c").write_text("int main(void) { return 1; }\n")

        with self.assertRaisesRegex(RuntimeError, " M src/main.c"):
            self.runner.check_working_tree_is_committed()

    def test_untracked_source_is_refused(self) -> None:
        Path(self.clone, "src", "extra.c").write_text("\n")
        Path(self.clone, "signaloid.yaml").write_text("\n")

        with self.assertRaisesRegex(
            RuntimeError, r"\?\? signaloid.yaml, \?\? src/extra.c"
        ):
            self.runner.check_working_tree_is_committed()

    def test_overwritten_build_resource_is_allowed(self) -> None:
        # The timing script copies its own Makefile.pro over a tracked one.
        Path(self.clone, "src", "Makefile.pro").write_text("tracked\n")
        _git("add", "src/Makefile.pro", cwd=self.clone)
        _git("commit", "--quiet", "-m", "makefile", cwd=self.clone)
        Path(self.clone, "src", "Makefile.pro").write_text("overwritten\n")

        self.runner.check_working_tree_is_committed()

    def test_generated_artifacts_and_outputs_are_allowed(self) -> None:
        for name in (
            "src/main-native",
            "src/main-Athens-16",
            "src/file.m",
            "src/Makefile.pro",
            "src/init.S",
            "src/data.out",
            "src/ground-truth.csv",
            "src/main",
            "src/build/nested/main.c.ll",
            "src/make.output",
            "inputs/data.out",
            # Outside the checked paths: the benchmark's own outputs.
            "demo-native-mc",
            "results/output_data.csv",
            ".signaloid-cli-latest-build-id.txt",
        ):
            path = Path(self.clone, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n")

        self.runner.check_working_tree_is_committed()
