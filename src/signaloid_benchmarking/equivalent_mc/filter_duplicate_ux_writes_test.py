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
Unit tests for ``_filter_duplicate_ux_writes``, which keeps one row of a
value traced more than once.
"""

import unittest

import pandas as pd

from signaloid_benchmarking.equivalent_mc.load import _filter_duplicate_ux_writes


def _rows(ur_type: str, *ux_type_bytes: str) -> pd.DataFrame:
    """
    Rows of one traced value, one per Ux string.

    Args:
        ur_type: The configuration's representation, as the execution info
            records it.
        ux_type_bytes: The leading type byte of each row's Ux string, in hex.

    Returns:
        The rows, all in one duplicate group.
    """
    return pd.DataFrame(
        {
            "Execution_ID": 1,
            "Expression_Name": "x",
            "ValueId": "v",
            "Assignment_Index": 1,
            "UR_Type": ur_type,
            "Dist_Value": [f"Ux{byte}00" for byte in ux_type_bytes],
        }
    )


class TestFilterDuplicateUxWrites(unittest.TestCase):
    """The row whose Ux type matches the configuration is kept."""

    def test_each_type_keeps_its_own_row(self) -> None:
        for ur_type, byte in (("Athens", "04"), ("Jupiter", "06"), ("Europa", "08")):
            with self.subTest(ur_type=ur_type):
                kept = _filter_duplicate_ux_writes(_rows(ur_type, "07", byte))
                self.assertEqual(list(kept["Dist_Value"]), [f"Ux{byte}00"])

    def test_no_matching_row_keeps_the_first(self) -> None:
        kept = _filter_duplicate_ux_writes(_rows("Europa", "06", "07"))
        self.assertEqual(list(kept["Dist_Value"]), ["Ux0600"])


if __name__ == "__main__":
    unittest.main()
