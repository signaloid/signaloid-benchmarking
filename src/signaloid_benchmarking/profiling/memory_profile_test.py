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

import math
import unittest
from typing import Callable

from signaloid_benchmarking.config import (
    Correlations,
    RepresentationTypes,
    configuration_string,
)
from signaloid_benchmarking.profiling.memory_profile import (
    DEFAULT_POOL_SIZE,
    ProfileError,
    check_capacity_is_reachable,
    initial_bracket,
    parse_pool_capacity,
    search_peak_elements,
)

# A real exit line, copied from a run of a demo at Athens-4.
_REAL_EXIT_LINE = "Real Allocs: 60 Semantic Allocs: 205 Semantic Frees: 205."


def _pool_probe(peak: int, *, calls: list[int] | None = None) -> Callable[[int], int]:
    """
    Build a probe that behaves like the real uncertain memory pool.

    The pool starts at the size the binary was built with and doubles
    whenever the next element will not fit, so a run reports the first
    size in the doubling series that is at least the peak occupancy.

    Args:
        peak: The peak occupancy the simulated run needs.
        calls: Optional list, appended with every probed size, so a test
            can assert on how many probes a search took.

    Returns:
        A callable taking a starting size and returning the capacity.
    """

    def probe(start_size: int) -> int:
        if calls is not None:
            calls.append(start_size)
        capacity = start_size
        while capacity < peak:
            capacity *= 2

        return capacity

    return probe


class TestParsePoolCapacity(unittest.TestCase):
    """The capacity is read from the line the runtime prints at exit."""

    def test_reads_the_capacity_from_a_real_exit_line(self) -> None:
        """The format is taken from an actual run, not invented."""
        self.assertEqual(parse_pool_capacity(_REAL_EXIT_LINE), 60)

    def test_reads_the_capacity_from_surrounding_output(self) -> None:
        """A demo prints a great deal before the line that matters."""
        output = "\n".join(
            [
                "Laplace core library built for __AVX512F__",
                "Uncertain memory pool elements not provided. " "Setting to default 30",
                "Displacement at the horizon: -0.461572Ux04",
                _REAL_EXIT_LINE,
            ]
        )
        self.assertEqual(parse_pool_capacity(output), 60)

    def test_last_line_wins(self) -> None:
        """
        A demo running several phases prints one line per phase. The
        final one covers the whole run, so it is the one to believe.
        """
        output = "Real Allocs: 30 ...\nReal Allocs: 120 ..."
        self.assertEqual(parse_pool_capacity(output), 120)

    def test_missing_line_is_an_error_not_a_guess(self) -> None:
        """
        A run that printed no capacity has told us nothing. Returning a
        default here would silently publish a fabricated number.
        """
        with self.assertRaises(ProfileError) as caught:
            parse_pool_capacity("some output with no capacity in it")

        self.assertIn("Real Allocs", str(caught.exception))


class TestInitialBracket(unittest.TestCase):
    """One probe narrows the answer to a range the doubling implies."""

    def test_no_growth_bounds_from_above_only(self) -> None:
        """Reporting the starting size means it was never grown."""
        self.assertEqual(
            initial_bracket(probed_size=30, reported_capacity=30),
            (1, 30),
        )

    def test_one_doubling_excludes_the_starting_size(self) -> None:
        """
        Growing to 60 from 30 means 30 was not enough, so the answer is
        above 30 and at most 60.
        """
        self.assertEqual(
            initial_bracket(probed_size=30, reported_capacity=60),
            (31, 60),
        )

    def test_two_doublings_bracket_the_last_one(self) -> None:
        """Only the final doubling was needed, so it sets the floor."""
        self.assertEqual(
            initial_bracket(probed_size=30, reported_capacity=120),
            (61, 120),
        )

    def test_shrinking_pool_is_an_error(self) -> None:
        """
        A pool only grows. A capacity below the starting size means the
        build ignored the requested size, which would make every later
        probe meaningless.
        """
        with self.assertRaises(ProfileError) as caught:
            initial_bracket(probed_size=90, reported_capacity=30)

        self.assertIn("UNCERTAIN_MEMORY_POOL_SIZE", str(caught.exception))


class TestCapacityIsReachable(unittest.TestCase):
    """A capacity must be the probed size doubled some whole number of times."""

    def test_accepts_the_doubling_sequence(self) -> None:
        """A pool starting at 30 can only ever report 30, 60, 120, 240."""
        for capacity in (30, 60, 120, 240):
            with self.subTest(capacity=capacity):
                check_capacity_is_reachable(probed_size=30, reported_capacity=capacity)

    def test_rejects_a_capacity_off_the_sequence(self) -> None:
        """
        45 cannot double into 60. Seeing that means the build ignored the
        requested pool size, so the run describes a different binary.
        """
        with self.assertRaises(ProfileError) as caught:
            check_capacity_is_reachable(probed_size=45, reported_capacity=60)

        self.assertIn("UNCERTAIN_MEMORY_POOL_SIZE", str(caught.exception))

    def test_rejects_a_shrinking_pool(self) -> None:
        """A pool only grows, so a smaller capacity is impossible."""
        with self.assertRaises(ProfileError):
            check_capacity_is_reachable(probed_size=90, reported_capacity=60)


class TestBuildIgnoringThePoolSize(unittest.TestCase):
    """A build that drops the flag must fail loudly, not quietly."""

    def test_a_fixed_capacity_build_is_caught(self) -> None:
        """
        If the build ignores UNCERTAIN_MEMORY_POOL_SIZE every run reports
        the same default-derived capacity. Left unchecked the search reads
        that as "the candidate was too small" at every step and converges
        on 60, which is a plausible number and entirely wrong. It has to
        raise instead.
        """

        def ignores_the_flag(start_size: int) -> int:
            return 60

        with self.assertRaises(ProfileError) as caught:
            search_peak_elements(probe=ignores_the_flag, start_size=30)

        self.assertIn("not applying", str(caught.exception))


class TestSearchPeakElements(unittest.TestCase):
    """The search recovers the exact peak occupancy."""

    def test_finds_the_exact_peak_across_a_wide_range(self) -> None:
        """
        Exhaustive over every peak from 1 to 200. The search has to be
        exact, because its output is a hardware memory figure.
        """
        for peak in range(1, 201):
            with self.subTest(peak=peak):
                found, _ = search_peak_elements(probe=_pool_probe(peak))
                self.assertEqual(found, peak)

    def test_is_exact_regardless_of_the_starting_guess(self) -> None:
        """
        The starting size changes how many probes it costs, never the
        answer. A bad guess must not bias the result.
        """
        for start_size in (1, 7, 30, 64, 500):
            with self.subTest(start_size=start_size):
                found, _ = search_peak_elements(
                    probe=_pool_probe(97),
                    start_size=start_size,
                )
                self.assertEqual(found, 97)

    def test_probe_count_is_logarithmic(self) -> None:
        """
        Each probe costs a rebuild and a run, and a run can take twenty
        minutes. The cost has to stay logarithmic in the answer, not
        linear, or a sweep becomes unaffordable.
        """
        calls: list[int] = []
        peak = 1000
        search_peak_elements(
            probe=_pool_probe(peak, calls=calls),
            start_size=DEFAULT_POOL_SIZE,
        )
        # The bracket spans at most the final capacity, so bisection needs
        # about log2 of it, plus the one probe that established it.
        self.assertLessEqual(len(calls), math.ceil(math.log2(peak)) + 2)

    def test_every_probe_is_a_size_that_could_be_built(self) -> None:
        """A pool size has to be a positive whole number of elements."""
        calls: list[int] = []
        search_peak_elements(probe=_pool_probe(137, calls=calls))
        for probed in calls:
            self.assertIsInstance(probed, int)
            self.assertGreaterEqual(probed, 1)

    def test_peak_of_one_is_found(self) -> None:
        """A demo allocating a single distribution is the lower edge."""
        found, _ = search_peak_elements(probe=_pool_probe(1))
        self.assertEqual(found, 1)

    def test_non_positive_start_is_rejected(self) -> None:
        """A pool cannot start at zero, and the search would not end."""
        with self.assertRaises(ProfileError):
            search_peak_elements(probe=_pool_probe(10), start_size=0)


class TestConfigurationString(unittest.TestCase):
    """Configurations are named as the benchmarking tool names them."""

    def test_disabled_correlation_is_left_out(self) -> None:
        """Matches uxhw_config_string, which omits the default."""
        self.assertEqual(
            configuration_string(
                representation_type=RepresentationTypes.JUPITER,
                representation_size=8,
                correlation=Correlations.DISABLED,
            ),
            "Jupiter-8",
        )

    def test_autocorrelation_is_named(self) -> None:
        """
        The suffix is what lets a profile be lined up against a
        benchmarking result for the same configuration.
        """
        self.assertEqual(
            configuration_string(
                representation_type=RepresentationTypes.ATHENS,
                representation_size=16,
                correlation=Correlations.AUTOCORRELATION,
            ),
            "Athens-16-Autocorrelation",
        )


if __name__ == "__main__":
    unittest.main()
