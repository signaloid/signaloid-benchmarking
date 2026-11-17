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
Unit tests for ``check_trace_variable_lines``, which checks each
``TraceVariables`` LineNumber against the source before a run starts.
"""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from signaloid_benchmarking.automation.benchmarking_utils import (
    check_trace_variable_lines,
)

# A main.c in the shape of the demos. Line 5 declares the traced array,
# line 9 only uses it, and line 7 is blank.
MAIN_C = """int
main(int argc, char *  argv[])
{
\tCommandLineArguments    arguments = (CommandLineArguments) { 0 };
\tuncertain double        outputVariables[kOutputVariableIndexMax] = { 0 };
\tconst char *            outputVariableNames[kOutputVariableIndexMax];

\t/* outputVariables holds every output */
\tcalculateOutput(&arguments, outputVariables);
\treturn outputVariables[0];
}
"""


class TestCheckTraceVariableLines(unittest.TestCase):
    """LineNumber must be the line that declares the traced variable."""

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.application = Path(tmp_dir.name)
        (self.application / "src").mkdir()
        (self.application / "src" / "main.c").write_text(MAIN_C)

    def _check(self, line_number: object, file_name: str = "main.c") -> None:
        """
        Check one entry tracing ``outputVariables[0:1]``.

        Args:
            line_number: The entry's LineNumber.
            file_name: The entry's File.
        """
        check_trace_variable_lines(
            str(self.application),
            [
                {
                    "File": file_name,
                    "LineNumber": line_number,
                    "Expression": "outputVariables[0:1]",
                }
            ],
        )

    def _error(self, line_number: object, file_name: str = "main.c") -> str:
        """
        Check an entry that must fail, and return the error message.

        Args:
            line_number: The entry's LineNumber.
            file_name: The entry's File.

        Returns:
            The message of the ValueError raised.
        """
        with self.assertRaises(ValueError) as caught:
            self._check(line_number, file_name)

        return str(caught.exception)

    def test_declaration_line_passes(self) -> None:
        """The declaring line passes, as an int or a quoted string."""
        self._check(5)
        self._check("5")

    def test_use_site_is_refused_with_the_declaration(self) -> None:
        """A line that only uses the variable suggests the declaration."""
        message = self._error(9)

        self.assertIn("main.c line 9", message)
        self.assertIn("calculateOutput(&arguments, outputVariables);", message)
        self.assertIn("Did you mean LineNumber 5?", message)

    def test_return_and_comment_are_not_declarations(self) -> None:
        """Neither ``return x[0];`` nor a comment naming x declares it."""
        for line_number in (8, 10):
            with self.subTest(line_number):
                self.assertIn("LineNumber 5?", self._error(line_number))

    def test_blank_and_out_of_range_lines(self) -> None:
        """A blank line and a line past the end are both refused."""
        self.assertIn("Line 7 is empty.", self._error(7))
        self.assertIn("main.c has 11 lines.", self._error(200))

    def test_shadowed_declarations_are_all_listed(self) -> None:
        """When the name is declared twice, both lines are suggested."""
        with (self.application / "src" / "main.c").open("a") as source:
            source.write("\nvoid\nhelper(void)\n{\n\tdouble  outputVariables[2];\n}\n")

        self.assertIn("declared on lines 5, 16.", self._error(9))

    def test_missing_file_and_bad_line_number(self) -> None:
        """A missing source file and a non-numeric LineNumber are refused."""
        self.assertIn("points to a missing file", self._error(5, file_name="mian.c"))
        self.assertIn("not a whole number", self._error("five"))

    def test_every_problem_is_reported_at_once(self) -> None:
        """Two wrong entries give one error naming both."""
        with self.assertRaises(ValueError) as caught:
            check_trace_variable_lines(
                str(self.application),
                [
                    {
                        "File": "main.c",
                        "LineNumber": 9,
                        "Expression": "outputVariables[0]",
                    },
                    {
                        "File": "main.c",
                        "LineNumber": 5,
                        "Expression": "outputVariableNames[0]",
                    },
                ],
            )
        message = str(caught.exception)

        self.assertIn("'outputVariables[0]' in main.c line 9", message)
        self.assertIn("'outputVariableNames[0]' in main.c line 5", message)
        self.assertIn("Did you mean LineNumber 6?", message)


# Declarations the tracer resolves that are not one variable on one line,
# and uses beside them. The line numbers are referred to below.
LAYOUTS_C = """int
calculate(double *  parameter, int count)
{
\tdouble\tx, severalOnOneLine[2];
\tdouble
\t\tsplitOverTwoLines[2];
\tdouble\ta = f(1, 2), afterACall[4] = { 0 };
\tconst char *\ts = "a ; in a string", afterAString[2];
\tDECLARE(double, fromAMacro)
\tcount = 0;
\tsplitOverTwoLines[0] = x;
\tcalculate(severalOnOneLine,
\t\t2);
\treturn 0;
}
"""


class TestCheckTraceVariableLinesLayouts(unittest.TestCase):
    """Declarations over several lines or of several variables are accepted."""

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.application = Path(tmp_dir.name)
        (self.application / "src").mkdir()
        (self.application / "src" / "main.c").write_text(LAYOUTS_C)

    def _check(self, expression: str, line_number: int) -> str:
        """
        Check one entry, and return what it printed.

        Args:
            expression: The traced expression.
            line_number: The entry's LineNumber.

        Returns:
            The standard output of the check.
        """
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            check_trace_variable_lines(
                str(self.application),
                [
                    {
                        "File": "main.c",
                        "LineNumber": line_number,
                        "Expression": expression,
                    }
                ],
            )

        return printed.getvalue()

    def test_several_declarators_and_split_declarations_pass(self) -> None:
        """The line that holds the name passes, with no warning."""
        for expression, line_number in (
            ("severalOnOneLine[0:1]", 4),
            ("splitOverTwoLines[0:1]", 6),
            ("afterACall[0]", 7),
            ("afterAString[0]", 8),
        ):
            with self.subTest(expression):
                self.assertEqual(self._check(expression, line_number), "")

    def test_the_type_line_of_a_split_declaration_is_refused(self) -> None:
        """The compiler records the name's line, so that is the one asked for."""
        with self.assertRaises(ValueError) as caught:
            self._check("splitOverTwoLines[0]", 5)

        self.assertIn("Did you mean LineNumber 6?", str(caught.exception))

    def test_uses_are_still_refused(self) -> None:
        """An assignment and a call over two lines both stop the run."""
        for expression, line_number in (
            ("splitOverTwoLines[0]", 11),
            ("severalOnOneLine[0]", 12),
        ):
            with self.subTest(expression):
                with self.assertRaises(ValueError):
                    self._check(expression, line_number)

    def test_an_unrecognized_declaration_only_warns(self) -> None:
        """A parameter, or a declaration made by a macro, gives a warning."""
        for expression, line_number in (
            ("parameter[0]", 2),
            ("fromAMacro", 9),
        ):
            with self.subTest(expression):
                self.assertIn(
                    "Warning: cannot tell whether", self._check(expression, line_number)
                )


if __name__ == "__main__":
    unittest.main()
