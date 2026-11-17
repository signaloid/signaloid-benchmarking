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
Unit tests for ``check_signaloid_yaml``, which checks that signaloid.yaml
has every key the run reads before the run starts.
"""

import copy
import unittest
from typing import Any

from signaloid_benchmarking.automation.benchmarking_utils import (
    check_signaloid_yaml,
)

PATH = "app/signaloid.yaml"

VALID: dict[str, Any] = {
    "TraceVariables": [
        {"File": "main.c", "LineNumber": 50, "Expression": "outputVariables[0:1]"}
    ],
    "BenchmarkingAllOutputs": [{"CommandLineArguments": "-S 2"}],
    "BenchmarkingVariables": [
        {
            "VariableName": "outputVariables[0]",
            "VariableDescription": "Price",
            "OutputObject": "Distribution",
            "CommandLineArguments": "-S 0",
        },
        {
            "VariableName": "outputVariables[1]",
            "VariableDescription": "Delta",
            "OutputObject": "Scalar",
            "CommandLineArguments": "-S 1",
        },
    ],
}


class TestCheckSignaloidYaml(unittest.TestCase):
    """Every key the run reads must be present and usable."""

    def _error(self, signaloid_yaml: Any) -> str:
        """
        Check a signaloid.yaml that must fail, and return the message.

        Args:
            signaloid_yaml: The parsed contents.

        Returns:
            The message of the ValueError raised.
        """
        with self.assertRaises(ValueError) as caught:
            check_signaloid_yaml(PATH, signaloid_yaml)
        message = str(caught.exception)
        self.assertIn(PATH, message)

        return message

    def test_valid_files_pass(self) -> None:
        """The newer format and the older one without outputs both pass."""
        check_signaloid_yaml(PATH, VALID)
        check_signaloid_yaml(PATH, {"TraceVariables": VALID["TraceVariables"]})

    def test_empty_file_and_missing_trace_list(self) -> None:
        """An empty file or one without TraceVariables is refused."""
        cases: list[Any] = [None, {}, {"TraceVariables": []}]
        for contents in cases:
            with self.subTest(contents):
                self.assertIn("There is no TraceVariables list.", self._error(contents))

    def test_missing_keys_are_named(self) -> None:
        """Each entry that lacks a key names the entry and the key."""
        signaloid_yaml = copy.deepcopy(VALID)
        del signaloid_yaml["TraceVariables"][0]["LineNumber"]
        del signaloid_yaml["BenchmarkingVariables"][1]["VariableDescription"]
        del signaloid_yaml["BenchmarkingVariables"][1]["OutputObject"]

        message = self._error(signaloid_yaml)

        self.assertIn("TraceVariables entry 1 has no LineNumber.", message)
        self.assertIn(
            "BenchmarkingVariables entry 2 has no VariableDescription, OutputObject.",
            message,
        )

    def test_missing_all_outputs(self) -> None:
        """BenchmarkingVariables without BenchmarkingAllOutputs is refused."""
        signaloid_yaml = copy.deepcopy(VALID)
        del signaloid_yaml["BenchmarkingAllOutputs"]

        self.assertIn("no BenchmarkingAllOutputs list", self._error(signaloid_yaml))

    def test_unknown_output_object(self) -> None:
        """A typo in OutputObject is caught and the two choices are given."""
        signaloid_yaml = copy.deepcopy(VALID)
        signaloid_yaml["BenchmarkingVariables"][0]["OutputObject"] = "distribution"

        message = self._error(signaloid_yaml)

        self.assertIn("('outputVariables[0]') has OutputObject 'distribution'", message)
        self.assertIn("Use 'Distribution' or 'Scalar'.", message)

    def test_untraced_name_suggests_the_closest(self) -> None:
        """A VariableName that is not traced suggests the nearest one."""
        signaloid_yaml = copy.deepcopy(VALID)
        signaloid_yaml["BenchmarkingVariables"][1]["VariableName"] = "outputVariable[1]"

        message = self._error(signaloid_yaml)

        self.assertIn("('outputVariable[1]') is not traced.", message)
        self.assertIn("Did you mean 'outputVariables[1]'?", message)

    def test_untraced_name_without_a_close_match_lists_all(self) -> None:
        """With no close match, every traced expression is listed."""
        signaloid_yaml = copy.deepcopy(VALID)
        signaloid_yaml["BenchmarkingVariables"][1]["VariableName"] = "zz"

        self.assertIn(
            "The traced expressions are outputVariables[0], outputVariables[1].",
            self._error(signaloid_yaml),
        )

    def test_every_problem_is_reported_at_once(self) -> None:
        """Problems in different parts of the file arrive in one error."""
        signaloid_yaml = copy.deepcopy(VALID)
        del signaloid_yaml["BenchmarkingAllOutputs"]
        signaloid_yaml["BenchmarkingVariables"][0]["OutputObject"] = "Scaler"

        message = self._error(signaloid_yaml)

        self.assertIn("BenchmarkingAllOutputs", message)
        self.assertIn("'Scaler'", message)


if __name__ == "__main__":
    unittest.main()
