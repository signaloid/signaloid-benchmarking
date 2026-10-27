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

import argparse
import re
import sys

# Capture the leading decimal float after ``CPU time used:`` and stop.
# Anything after the float is intentionally not anchored, because:
#   * The UxHw binary emits a Ux Data value right up against the float
#     (no separator), e.g. ``CPU time used: 0.000163Ux0400...000 seconds``.
#   * The native binary emits ``CPU time used: 0.0001 seconds``.
#   * Some outputs omit ``seconds`` entirely.
# Anchoring on the float alone matches all three.
_CPU_TIME_PATTERN = re.compile(r"CPU time used:\s*([0-9]+(?:\.[0-9]+)?)")


def _extract_cpu_time(content: str) -> float:
    """
    Extract the first ``CPU time used: N`` value from *content*.

    Single-binary stdouts (the per-iteration files from
    ``run_uxhw_benchmarks``) contain exactly one such line.

    Args:
        content: Raw text from a benchmark binary's stdout. Both
            ``CPU time used: 0.000198`` and the ``... seconds`` form are
            accepted.

    Returns:
        The CPU time as a float in seconds.

    Raises:
        ValueError: If no ``CPU time used: <value>`` line is present.
    """
    match = _CPU_TIME_PATTERN.search(content)
    if match is None:
        raise ValueError("no 'CPU time used: <value>' line found in stdout")
    return float(match.group(1))


def _read_stdout_text(path: str) -> str:
    """
    Read a stdout file as text, replacing any non-UTF-8 bytes.

    Treats the file as text even when it contains stray binary bytes. The
    ASCII ``CPU time used:`` line survives the replacement substitution.

    Args:
        path: Path to the stdout file to read.

    Returns:
        The file contents, with non-UTF-8 bytes replaced by U+FFFD.
    """
    # Pin the encoding so a non-UTF-8 system locale (e.g. C/POSIX on
    # minimal containers, or latin-1) doesn't silently change the
    # decoded input stream. `errors="replace"` then turns every
    # non-UTF-8 byte into U+FFFD regardless of locale.
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def read_cpu_time(path: str) -> float:
    """
    Read a benchmark stdout file and return its first CPU-time value.

    Args:
        path: Path to the stdout file written by the benchmark binary.

    Returns:
        The CPU time as a float in seconds.

    Raises:
        OSError: If the file cannot be opened.
        ValueError: If the file content does not contain a
            ``CPU time used: <value>`` line.
    """
    return _extract_cpu_time(_read_stdout_text(path))


def main() -> None:
    """
    Parse command-line arguments and emit the extracted CPU time.

    Intended for the bash layer via ``python3 -m
    signaloid_benchmarking.automation.parse_cpu_time``. Prints the seconds
    value from the file given.
    """
    parser = argparse.ArgumentParser(
        description="Extract 'CPU time used:' seconds from a benchmark stdout file.",
    )
    parser.add_argument(
        "path",
        help="Path to the stdout file produced by the benchmark binary.",
    )
    args = parser.parse_args()
    print(read_cpu_time(args.path))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        print(f"parse_cpu_time: {exc}", file=sys.stderr)
        sys.exit(1)
