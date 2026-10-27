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
Regression coverage for the build-resources fallback in get-timings.sh.

``get-timings.sh`` resolves the bundled coreClass build templates from
``BENCHMARKING_RESOURCES_DIR`` when the Python tool exports it, and otherwise
falls back to a path relative to its own location. The Python tool always
exports it, so the fallback only runs for standalone bash use and went
unnoticed while it pointed one directory level too high. These tests pin the
fallback to the same directory ``get_resources_dir()`` returns.

The tests after those run single functions of the script under its own shell
options, with the builds and the SQLite helpers stubbed. They cover the skip
list and what happens when a UxHw run dies. That run is recorded with its
reason, and the script goes on with the other configurations.
"""

import os
import re
import shlex
import subprocess
import sys
import tempfile
import unittest

from pathlib import Path

from signaloid_benchmarking.config import get_resources_dir

_SCRIPT = Path(__file__).resolve().parent / "get-timings.sh"

# Matches the default in `RESOURCES_DIR="${BENCHMARKING_RESOURCES_DIR:-...}"`.
_FALLBACK_PATTERN = re.compile(
    r'RESOURCES_DIR="\$\{BENCHMARKING_RESOURCES_DIR:-(?P<default>[^}]+)\}"'
)


def _fallback_resources_dir() -> Path:
    """Resolve the script's own default for the build-resources directory.

    Returns:
        The directory the fallback expands to, with ``$SCRIPT_DIR``
        substituted for the script's location and ``..`` segments collapsed.
    """
    match = _FALLBACK_PATTERN.search(_SCRIPT.read_text())
    assert match is not None, "could not find the RESOURCES_DIR fallback"
    default = match.group("default")
    assert "$SCRIPT_DIR" in default, f"fallback is not script-relative: {default}"
    return Path(default.replace("$SCRIPT_DIR", str(_SCRIPT.parent))).resolve()


class TestGetTimingsResourcesFallback(unittest.TestCase):
    """The standalone fallback must find the bundled build templates."""

    def test_fallback_directory_exists(self) -> None:
        fallback = _fallback_resources_dir()
        self.assertTrue(
            fallback.is_dir(),
            f"fallback build-resources directory does not exist: {fallback}",
        )

    def test_fallback_matches_python_resolver(self) -> None:
        """The bash fallback and get_resources_dir() must not diverge."""
        self.assertEqual(_fallback_resources_dir(), get_resources_dir().resolve())

    def test_fallback_contains_expected_templates(self) -> None:
        """Guard the files copy_build_resources() reads from that directory."""
        fallback = _fallback_resources_dir()
        for relative_path in (
            "C0/init.S",
            "common/startup.cpp",
            "C0Pro/Makefile.pro",
            "C0/Makefile",
        ):
            with self.subTest(relative_path=relative_path):
                self.assertTrue(
                    (fallback / relative_path).is_file(),
                    f"missing build template: {relative_path}",
                )


def _bash_function(name: str) -> str:
    """Return one function's definition from get-timings.sh.

    The script runs its whole pipeline when sourced, so a test takes just
    the functions it calls.

    Args:
        name: The function's name.

    Returns:
        The definition, from its opening line to its closing brace.
    """
    lines = _SCRIPT.read_text().splitlines()
    start = lines.index(f"{name}() {{")
    end = lines.index("}", start)
    return "\n".join(lines[start : end + 1])


def _skip_decisions(
    configurations: list[tuple[str, str, int]],
    skipped: list[str] | None,
) -> list[str]:
    """Run should_skip_representation under the script's own shell options.

    Args:
        configurations: (type, correlation, size) to ask about, in the
            argument order the script's loops use.
        skipped: The SKIPPED_CONFIGURATIONS list, or None to leave it unset.

    Returns:
        ``skip`` or ``run`` for each configuration.
    """
    script = [
        "set -euo pipefail",
        _bash_function("uxhw_config_string"),
        _bash_function("should_skip_representation"),
    ]
    if skipped is not None:
        script.append(
            "SKIPPED_CONFIGURATIONS=(" + " ".join(map(shlex.quote, skipped)) + ")"
        )
    for representation_type, correlation, size in configurations:
        script.append(
            f"if should_skip_representation {representation_type} {correlation} "
            f"{size}; then echo skip; else echo run; fi"
        )
    result = subprocess.run(
        ["bash", "-c", "\n".join(script)],
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout.split()


class TestShouldSkipRepresentation(unittest.TestCase):
    """The timing layer skips exactly what the Python side lists."""

    def test_a_listed_configuration_is_skipped(self) -> None:
        """Europa with autocorrelation is skipped, Europa without runs."""
        self.assertEqual(
            _skip_decisions(
                [("Europa", "Autocorrelation", 8), ("Europa", "Disabled", 8)],
                ["Europa-8-Autocorrelation"],
            ),
            ["skip", "run"],
        )

    def test_no_type_is_skipped_unless_listed(self) -> None:
        """
        Jupiter has no rule of its own any more. The Python side decides
        it from the SDK and the memory guard, like every other type.
        """
        self.assertEqual(
            _skip_decisions(
                [("Jupiter", "Autocorrelation", 8), ("Jupiter", "Disabled", 512)],
                [],
            ),
            ["run", "run"],
        )

    def test_an_unset_list_skips_nothing_under_set_u(self) -> None:
        """A standalone caller that defines no list does not abort."""
        self.assertEqual(_skip_decisions([("Athens", "Disabled", 8)], None), ["run"])


# A stand-in for a UxHw binary. It counts its own runs, and on the run its
# `.die` file names it is killed by SIGKILL, as the out-of-memory killer
# would, or exits with code 3. Otherwise it prints the line the timing
# layer parses.
_FAKE_PROGRAM = """#!/bin/sh
count=$(( $(cat "$0.count" 2>/dev/null || echo 0) + 1 ))
echo "$count" >"$0.count"
case "$(cat "$0.die" 2>/dev/null)" in
"kill $count") kill -9 $$ ;;
"exit $count") exit 3 ;;
esac
echo "CPU time used: 0.5 seconds"
"""

# A stand-in for Intel PIN. It runs the program and writes the count file
# PIN's inscount0 tool writes, unless the program died.
_FAKE_PIN = """#!/bin/sh
"$@" >/dev/null || exit $?
echo "Count 100" >inscount.out
"""

# The stand-in builds that run_uxhw_tracing compiles. A size listed in
# $DYING_SIZES is a binary that is killed on start. Any other writes its
# tracing DB, as the SDK does.
_FAKE_UXHW_MAKE = """uxhw_make() {
    local argument size="" database=""
    for argument in "$@"; do
        case "$argument" in
        clean) return 0 ;;
        REPRESENTATION_SIZE=*) size="${argument#*=}" ;;
        STATS_DB_FILENAME=*) database="${argument#*=}" ;;
        esac
    done
    if [[ " $DYING_SIZES " == *" $size "* ]]; then
        printf '#!/bin/sh\\nkill -9 $$\\n' >"$PROGRAM"
    else
        printf '#!/bin/sh\\ntouch "%s"\\necho traced\\n' "$database" >"$PROGRAM"
    fi
    chmod +x "$PROGRAM"
}"""

# A stand-in for the Python helpers run_uxhw_tracing calls. It logs each
# call, one per line, so a test can see which DBs reached the checks and
# the merge.
_FAKE_PYTHON = """#!/bin/sh
echo "$*" >>"$PYTHON_CALLS"
"""


def _write_executable(path: Path, text: str) -> None:
    """Write a script and make it executable.

    Args:
        path: Where to write it.
        text: Its contents.
    """
    path.write_text(text)
    path.chmod(0o755)


def _run_script(
    lines: list[str], environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run script lines under get-timings.sh's own shell options.

    Args:
        lines: Function definitions and calls, in order.
        environment: Variables added to this process's environment.

    Returns:
        The finished process, with stdout and stderr captured.
    """
    return subprocess.run(
        ["bash", "-c", "\n".join(["set -euo pipefail", *lines])],
        capture_output=True,
        text=True,
        env={**os.environ, **environment},
    )


class TestExitStatusReason(unittest.TestCase):
    """A run's exit status becomes a reason a reader can act on."""

    def _reason(self, status: int) -> str:
        """
        Describe an exit status as get-timings.sh does.

        Args:
            status: The status a run ended with.

        Returns:
            The reason.
        """
        completed = _run_script(
            [_bash_function("exit_status_reason"), f"exit_status_reason {status}"],
            {},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

        return completed.stdout.strip()

    def test_a_status_above_128_names_the_signal_and_the_exit_code(self) -> None:
        """
        137 is how the shell reports the out-of-memory killer's SIGKILL,
        and also what a program that calls exit(137) gives. The status
        cannot tell them apart, so the reason names both.
        """
        self.assertEqual(
            self._reason(137),
            "was killed by signal 9 (SIGKILL), or exited with code 137",
        )

    def test_any_other_status_is_an_exit_code(self) -> None:
        self.assertEqual(self._reason(3), "exited with code 3")

    def test_a_status_above_128_that_names_no_signal_is_an_exit_code(self) -> None:
        """A program may exit(250) itself. 122 is no signal."""
        self.assertEqual(self._reason(250), "exited with code 250")


class TimingLayerTestCase(unittest.TestCase):
    """A temporary application tree and the variables the script reads."""

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.root = Path(tmp_dir.name)
        for directory in ("src", "inputs", "logs", "results"):
            (self.root / directory).mkdir()
        self.failed_configurations = self.root / "logs" / "failed-configurations.tsv"
        self.intermediate = self.root / "results" / "timings.intermediate"
        self.environment = {
            "APPLICATION_PATH": str(self.root),
            "INPUT_DIR": "inputs",
            "LOGS_DIR": str(self.root / "logs"),
            "PROGRAM": "main",
            "CLA": "",
            "FAILED_CONFIGURATIONS_FILE": str(self.failed_configurations),
            "TIMING_INTERMEDIATE_FILE": str(self.intermediate),
            "TIMING_META_TAG": "META",
            "TIMING_MEASUREMENT_TAG": "MEASUREMENT",
            "TIMING_SAMPLE_TAG": "SAMPLE",
            "UXHW_DB_BASE": "signaloidUxHwExecutionStatistics",
            "M_CONFIG_FILE": "file.m",
            "EXEC_STDOUT": "exec.stdout",
            "EXEC_STDERR": "exec.stderr",
            "BENCHMARKING_PYTHON": sys.executable,
            "PACKAGE_IMPORT_ROOT": str(_SCRIPT.parents[3]),
            "TIMING_TARGET_TOTAL_TIME": "0.01",
            "TIMING_MIN_REPETITIONS": "2",
            "TIMING_MAX_REPETITIONS": "2",
        }

    def failures(self) -> list[str]:
        """
        Read the record of runs that died.

        Returns:
            The lines of the failure record, or none when it is absent.
        """
        if not self.failed_configurations.is_file():
            return []

        return self.failed_configurations.read_text().splitlines()


class TestTimingRunThatDies(TimingLayerTestCase):
    """run_uxhw_benchmarks goes on past a run that dies."""

    def _time(
        self, dying: dict[str, str], *, pin: bool = False, real_timer: bool = False
    ) -> None:
        """
        Time one configuration per entry, each with its own fake program.

        Args:
            dying: For each configuration, the run its program dies on,
                for example ``kill 1``, or an empty string to survive.
            pin: Whether to count instructions with the fake PIN.
            real_timer: Whether the end-to-end runs go through the real
                measure_process_time helper. Its import takes seconds, so
                other tests stand in a single run for it.
        """
        for configuration, die in dying.items():
            program = self.root / "src" / f"main-{configuration}"
            _write_executable(program, _FAKE_PROGRAM)
            program.with_name(f"{program.name}.die").write_text(die)
        _write_executable(self.root / "pin", _FAKE_PIN)
        sizes = " ".join(configuration.split("-")[1] for configuration in dying)
        # The script writes its header before any timing.
        if not self.intermediate.is_file():
            self.intermediate.write_text("META timestamp now\n")

        completed = _run_script(
            [
                *(
                    _bash_function(name)
                    for name in (
                        "warn",
                        "uxhw_config_string",
                        "should_skip_representation",
                        "cd_to_input_dir",
                        "compute_time_scaled_repetitions",
                        "emit_sample",
                        "emit_measurement",
                        "exit_status_reason",
                        "record_failed_configuration",
                        "abandon_uxhw_testcase",
                        "run_uxhw_benchmarks",
                    )
                ),
                "log_section() { :; }",
                "build_uxhw_testcase_binary() { :; }",
                "parse_cpu_time() { echo 0.5; }",
                "read_db_metric() { echo 0.25; }",
                (
                    _bash_function("measure_process_time")
                    if real_timer
                    else "measure_process_time() { $1 >/dev/null || return $?; echo 1; }"
                ),
                "CORRELATION_TRACKING_TYPES=(Disabled)",
                "REPRESENTATION_TYPES=(Athens)",
                f"REPRESENTATION_SIZES=({sizes})",
                f"PIN_ENABLED={int(pin)}",
                f"INSTRUCTION_COUNT_COMMAND={shlex.quote(str(self.root / 'pin'))}",
                "run_uxhw_benchmarks",
            ],
            self.environment,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_the_emulator_config_is_removed_from_src(self) -> None:
        """
        The timing runs leave the shell in the input directory, but the
        emulator config that main wrote is in src/, so it is removed from
        there. The application's own .m files are left alone.
        """
        config = self.root / "src" / "file.m"
        config.write_text("noDetach 1\n")
        application_file = self.root / "inputs" / "model.m"
        application_file.write_text("% the application's own\n")

        self._time({"Athens-4": ""})

        self.assertFalse(config.exists())
        self.assertTrue(application_file.exists())

    def test_each_run_that_dies_is_recorded_with_how_it_ended(self) -> None:
        """
        Two repetitions are timed, then three end-to-end runs. A death in
        any of them skips the configuration and timing goes on with the
        next. The end-to-end runs go through the real helper, so its exit
        status is what tells the signal apart.
        """
        self._time(
            {
                "Athens-4": "kill 1",
                "Athens-8": "kill 3",
                "Athens-16": "exit 3",
                "Athens-32": "kill 5",
            },
            real_timer=True,
        )

        self.assertEqual(
            self.failures(),
            [
                "Athens-4\tits timing warmup run was killed by signal 9 (SIGKILL), or exited with code 137",
                "Athens-8\tits timing run 2 of 2 was killed by signal 9 (SIGKILL), or exited with code 137",
                "Athens-16\tits timing run 2 of 2 exited with code 3",
                "Athens-32\tits end-to-end timing run was killed by signal 9 "
                "(SIGKILL), or exited with code 137",
            ],
        )

    def test_only_the_survivors_samples_and_measurement_remain(self) -> None:
        """
        A dead configuration's samples are cut by line count, so samples an
        earlier pass wrote under the same name are kept.
        """
        earlier_pass = [
            "META timestamp earlier",
            "SAMPLE Athens-8 elapsedTime 0 0.75",
            "MEASUREMENT Athens-8 ? ? 1.0 ? ?",
        ]
        self.intermediate.write_text("\n".join(earlier_pass) + "\n")

        self._time({"Athens-8": "kill 3", "Athens-64": ""})

        lines = self.intermediate.read_text().splitlines()
        self.assertEqual(lines[:3], earlier_pass)
        self.assertEqual(
            {line.split()[1] for line in lines[3:]},
            {"Athens-64"},
        )
        self.assertEqual(lines[-1].split()[:2], ["MEASUREMENT", "Athens-64"])

    def test_a_pin_run_that_dies_takes_its_count_files_with_it(self) -> None:
        """The second PIN run dies, so the first one's count file goes."""
        self._time({"Athens-8": "kill 5"}, pin=True)

        self.assertEqual(
            self.failures(),
            [
                "Athens-8\tits PIN run 2 of 2 was killed by signal 9 (SIGKILL), or exited with code 137"
            ],
        )
        self.assertEqual(list((self.root / "logs").glob("inscount-*")), [])


class TestAbandonUxhwTestcase(TimingLayerTestCase):
    """Abandoning a configuration removes its traces and nothing else."""

    def test_another_configurations_count_files_are_kept(self) -> None:
        """Athens-8 must not match Athens-8-Autocorrelation's files."""
        self.intermediate.write_text("kept\nkept\ncut\n")

        completed = _run_script(
            [
                _bash_function("warn"),
                _bash_function("record_failed_configuration"),
                _bash_function("abandon_uxhw_testcase"),
                'for name in Athens-8-0 Athens-8-Autocorrelation-0; do touch "$LOGS_DIR/inscount-$name-$$.out"; done',
                'abandon_uxhw_testcase Athens-8 "  2" "its timing run died"',
                'ls "$LOGS_DIR"',
            ],
            self.environment,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(self.intermediate.read_text(), "kept\nkept\n")
        self.assertEqual(self.failures(), ["Athens-8\tits timing run died"])
        remaining = completed.stdout.split()
        self.assertEqual(len(remaining), 2, remaining)
        self.assertIn("failed-configurations.tsv", remaining)
        self.assertTrue(
            any(
                name.startswith("inscount-Athens-8-Autocorrelation-0-")
                for name in remaining
            ),
            remaining,
        )


class TestTracingRunThatDies(TimingLayerTestCase):
    """run_uxhw_tracing goes on past a run that dies."""

    def _trace(self, dying_sizes: str) -> subprocess.CompletedProcess[str]:
        """
        Trace Athens 4 and 8 with fake builds and fake Python helpers.

        Args:
            dying_sizes: The sizes whose binaries are killed on start.

        Returns:
            The finished process.
        """
        _write_executable(self.root / "python", _FAKE_PYTHON)
        self.python_calls = self.root / "python-calls"
        self.tracing_db = self.root / "results" / "tracing.db"
        completed = _run_script(
            [
                *(
                    _bash_function(name)
                    for name in (
                        "warn",
                        "uxhw_config_string",
                        "should_skip_representation",
                        "exit_status_reason",
                        "record_failed_configuration",
                        "cd_to_input_dir",
                        "run_tracing_binary",
                        "run_uxhw_tracing",
                    )
                ),
                _FAKE_UXHW_MAKE,
                "log_section() { :; }",
                "write_emulator_config() { :; }",
                "CORRELATION_TRACKING_TYPES=(Disabled)",
                "REPRESENTATION_TYPES=(Athens)",
                "REPRESENTATION_SIZES=(4 8)",
                "run_uxhw_tracing",
            ],
            {
                **self.environment,
                "BENCHMARKING_PYTHON": str(self.root / "python"),
                "PYTHON_CALLS": str(self.python_calls),
                "DYING_SIZES": dying_sizes,
                "TRACING_DB_ABS": str(self.tracing_db),
                "TARGET_ARCH": "x86_64-unknown-linux-gnu",
                "ENABLE_UNCERTAIN_TYPE_MODIFIER": "OFF",
            },
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

        return completed

    def _calls(self, module: str) -> list[str]:
        """
        Read the logged calls of one Python helper.

        Args:
            module: The last part of a helper's module name.

        Returns:
            The arguments of each call of that helper.
        """
        if not self.python_calls.is_file():
            return []

        return [
            line
            for line in self.python_calls.read_text().splitlines()
            if f"automation.{module} " in line
        ]

    def test_a_dead_run_is_recorded_and_left_out_of_the_checks_and_merge(
        self,
    ) -> None:
        completed = self._trace("8")

        self.assertEqual(
            self.failures(),
            [
                "Athens-8\tits tracing run was killed by signal 9 (SIGKILL), or exited with code 137"
            ],
        )
        surviving_db = f"{self.tracing_db.with_suffix('')}-Athens-4.db"
        self.assertEqual(
            self._calls("merge_tracing_dbs"),
            [
                "-m signaloid_benchmarking.automation.merge_tracing_dbs "
                f"{self.tracing_db} {surviving_db}"
            ],
        )
        for module in ("check_traced_values_printed", "compare_tracing_ux_strings"):
            with self.subTest(module=module):
                calls = self._calls(module)
                self.assertEqual(len(calls), 1, calls)
                self.assertIn("Athens-4", calls[0])
        # Its -O2 run died too, which is no news once the config is skipped.
        self.assertNotIn("-O2 verification run failed", completed.stderr)

    def test_when_every_run_dies_nothing_is_merged_and_the_script_goes_on(
        self,
    ) -> None:
        completed = self._trace("4 8")

        self.assertEqual(len(self.failures()), 2)
        self.assertEqual(self._calls("merge_tracing_dbs"), [])
        self.assertIn("every tracing run died", completed.stderr)


if __name__ == "__main__":
    unittest.main()
