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
Unit tests for ``uses_binned_distance``, the rule every stage uses to pick
the UxHw distance an equivalence is measured on.
"""

import math
import unittest

from signaloid_benchmarking.config import uses_binned_distance


class TestUsesBinnedDistance(unittest.TestCase):
    """The binned distance is used only when asked for and usable."""

    def test_usable_binned_distance_is_used_when_asked_for(self) -> None:
        self.assertTrue(uses_binned_distance(0.25, True))

    def test_plain_distance_is_used_when_binned_is_not_asked_for(self) -> None:
        self.assertFalse(uses_binned_distance(0.25, False))

    def test_unusable_binned_distance_falls_back(self) -> None:
        """Missing, zero and non-finite binned distances are not usable."""
        for binned in (None, 0.0, math.inf, -math.inf, math.nan):
            with self.subTest(binned=binned):
                self.assertFalse(uses_binned_distance(binned, True))


if __name__ == "__main__":
    unittest.main()
