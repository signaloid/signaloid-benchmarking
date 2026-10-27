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
Unit tests for the checks ``Benchmark.load_yaml`` makes on the file names that
each variable's description gives its result files.
"""

import tempfile
import unittest
from pathlib import Path

import yaml

from signaloid_benchmarking.automation.benchmark import Benchmark


def _load(directory: Path, descriptions: list[str]) -> Benchmark:
    """
    Write a minimal application with one output per description, and load it.

    Args:
        directory: The application root.
        descriptions: One ``VariableDescription`` per output.

    Returns:
        The Benchmark after ``load_yaml``.
    """
    signaloid_yaml = {
        "TraceVariables": [
            {
                "File": "main.c",
                "LineNumber": 50,
                "Expression": f"outputVariables[0:{len(descriptions) - 1}]",
            }
        ],
        "BenchmarkingVariables": [
            {
                "VariableName": f"outputVariables[{index}]",
                "VariableDescription": description,
                "OutputObject": "Distribution",
                "CommandLineArguments": f"-S {index}",
            }
            for index, description in enumerate(descriptions)
        ],
        "BenchmarkingAllOutputs": [{"CommandLineArguments": f"-S {len(descriptions)}"}],
    }
    (directory / "signaloid.yaml").write_text(
        yaml.safe_dump(signaloid_yaml, sort_keys=False),
        encoding="utf-8",
    )
    benchmark = Benchmark(path_to_application=str(directory))
    benchmark.load_yaml()

    return benchmark


class TestLoadYamlResultFileNames(unittest.TestCase):
    """Each variable's description must give it result files of its own."""

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.application = Path(tmp_dir.name)

    def test_description_with_a_slash_loads(self) -> None:
        """The description from the issue loads, with a flat file name."""
        benchmark = _load(
            self.application,
            ["DNBR Safety Margin (qcrit/qlocal)", "Second variable"],
        )

        variable = benchmark.benchmarking_variables[0]
        self.assertEqual(variable.description, "DNBR Safety Margin (qcrit/qlocal)")
        self.assertEqual(
            variable.formatted_description, "dnbr-safety-margin-qcrit-qlocal"
        )

    def test_descriptions_sharing_a_file_name_are_refused(self) -> None:
        """Two descriptions that differ only in dropped characters fail."""
        with self.assertRaises(ValueError) as context:
            _load(self.application, ["Margin (a/b)", "Margin a-b"])

        message = str(context.exception)
        self.assertIn("outputVariables[0]", message)
        self.assertIn("outputVariables[1]", message)
        self.assertIn("margin-a-b", message)

    def test_description_with_no_letters_or_digits_is_refused(self) -> None:
        """A description that leaves an empty file name fails."""
        with self.assertRaises(ValueError) as context:
            _load(self.application, ["First variable", "(/)"])

        self.assertIn("outputVariables[1]", str(context.exception))


if __name__ == "__main__":
    unittest.main()
