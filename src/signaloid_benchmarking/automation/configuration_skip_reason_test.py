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
Unit tests for ``configuration_skip_reason``.

A configuration is skipped when the SDK ships no runtime library for it,
and the answer comes from asking the real build makefile. These tests run
real ``make`` on the real ``Makefile.pro`` against a small fake SDK, whose
makefile picks a library directory and name the way the real SDK's does.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from signaloid_benchmarking.automation.build import configuration_skip_reason

# Chooses the library directory from PRECISION, as the real SDK chooses it
# from its precision setting, so a test can show that the application's
# config.mk is read before the SDK makefile, exactly as in a build.
_FAKE_SDK_MAKEFILE = """\
PRECISION ?= double-precision
UXHW_SDK_RUNTIME_LIB = $(PATH_TO_UXHW_SDK)/lib/$(PRECISION)
ifeq ($(CORRELATION_TRACKING),Autocorrelation)
FEATURE_SUFFIX := _Autocorrelation
endif
LIBUXHW_RUNTIME := uxhw_runtime_$(REPRESENTATION_TYPE)_$(REPRESENTATION_SIZE)$(FEATURE_SUFFIX)
"""


@unittest.skipUnless(shutil.which("make"), "make is not installed")
class TestConfigurationSkipReason(unittest.TestCase):
    """Which configurations the SDK cannot build, and why."""

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        root = Path(tmp_dir.name)

        self.sdk = root / "sdk"
        (self.sdk / "include").mkdir(parents=True)
        (self.sdk / "include" / "UxHw-SDK.mk").write_text(_FAKE_SDK_MAKEFILE)
        for precision, libraries in (
            (
                "double-precision",
                ("Europa_8", "Athens_8", "Athens_8_Autocorrelation"),
            ),
            ("single-precision", ("Europa_8_Autocorrelation",)),
        ):
            directory = self.sdk / "lib" / precision
            directory.mkdir(parents=True)
            for library in libraries:
                (directory / f"libuxhw_runtime_{library}.a").write_text("")

        self.application = root / "app"
        (self.application / "src").mkdir(parents=True)

    def _reason(
        self,
        representation_type: str,
        representation_size: int,
        correlation: str,
        *,
        max_jupiter_size: int = 32,
    ) -> str | None:
        """
        Ask about one configuration of the fake SDK.

        Args:
            representation_type: For example ``Europa``.
            representation_size: For example ``8``.
            correlation: ``Disabled`` or ``Autocorrelation``.
            max_jupiter_size: The largest Jupiter size to benchmark.

        Returns:
            Why it is skipped, or ``None`` when it can run.
        """
        return configuration_skip_reason(
            path_to_application=str(self.application),
            path_to_uxhw_sdk=str(self.sdk),
            representation_type=representation_type,
            representation_size=representation_size,
            correlation=correlation,
            max_jupiter_size=max_jupiter_size,
        )

    def test_a_configuration_without_a_runtime_library_is_skipped(self) -> None:
        """Europa ships no autocorrelation library here, as in SDK 6.6.0."""
        reason = self._reason("Europa", 8, "Autocorrelation")

        self.assertIsNotNone(reason)
        self.assertIn("libuxhw_runtime_Europa_8_Autocorrelation.a", reason or "")

    def test_a_configuration_with_its_runtime_library_runs(self) -> None:
        """The same type without autocorrelation tracking has a library."""
        self.assertIsNone(self._reason("Europa", 8, "Disabled"))
        self.assertIsNone(self._reason("Athens", 8, "Autocorrelation"))

    def test_the_applications_config_mk_is_read_first(self) -> None:
        """
        An application that sets single precision in its config.mk is
        asked about the single-precision libraries, as its build would be.
        """
        (self.application / "src" / "config.mk").write_text(
            "PRECISION = single-precision\n"
        )

        self.assertIsNone(self._reason("Europa", 8, "Autocorrelation"))
        self.assertIsNotNone(self._reason("Europa", 8, "Disabled"))

    def test_inherited_make_flags_do_not_change_the_answer(self) -> None:
        """
        A run started from a parent make inherits its MAKEFLAGS. With -n
        there, make would print the echo command instead of the library.
        """
        with mock.patch.dict(os.environ, {"MAKEFLAGS": "n"}):
            self.assertIsNone(self._reason("Europa", 8, "Disabled"))

    def test_jupiter_above_the_limit_is_skipped_as_a_memory_guard(self) -> None:
        """The memory guard applies before the SDK is asked anything."""
        reason = self._reason("Jupiter", 64, "Disabled", max_jupiter_size=32)

        self.assertIn("--max-jupiter-size 32", reason or "")

    def test_jupiter_within_the_limit_is_judged_by_its_library(self) -> None:
        """Jupiter has no special case beyond the memory guard."""
        reason = self._reason("Jupiter", 8, "Disabled", max_jupiter_size=32)

        self.assertIn("libuxhw_runtime_Jupiter_8.a", reason or "")

    def test_an_sdk_that_cannot_be_asked_is_an_error(self) -> None:
        """A missing SDK makefile fails loudly, naming the configuration."""
        shutil.rmtree(self.sdk / "include")

        with self.assertRaises(RuntimeError) as caught:
            self._reason("Europa", 8, "Autocorrelation")

        self.assertIn("Europa-8-Autocorrelation", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
