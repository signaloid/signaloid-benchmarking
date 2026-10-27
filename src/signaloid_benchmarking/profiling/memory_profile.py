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
Find the exact number of uncertain memory pool elements a UxHw run needs,
so the memory a representation requires can be specified rather than
guessed.

External heap profilers cannot answer this. Valgrind cannot execute the
demo binaries at all, because the core library is built for AVX-512, and
no LD_PRELOAD tool can attach because the binaries are statically linked
at a fixed load address for the emulator.

The runtime answers it instead. The uncertain memory pool starts at a
size fixed at compile time and doubles whenever it runs short, and the
run reports the capacity it finished on. So one run of a binary built
with a starting size of S tells us whether S was enough, because the
reported capacity stays at S exactly when it was. That predicate is
monotone in S, so bisecting on it finds the peak occupancy exactly.

The cost is one rebuild and one run per probe, since the starting size is
an LLVM pass flag and cannot be set on a built binary.
"""

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

from signaloid_benchmarking.automation.arguments import (
    CORRELATION_CHOICES,
    REPRESENTATION_TYPE_CHOICES,
)
from signaloid_benchmarking.config import (
    Correlations,
    configuration_string,
    get_resources_dir,
)

# The line the UxHw runtime prints at exit. The first count is the pool
# capacity the run finished on, after any doublings.
_POOL_CAPACITY_PATTERN = re.compile(r"Real Allocs:\s*(\d+)")

# The pool size the runtime falls back to when the build does not set one.
# Only used as the first probe, so a wrong guess costs one extra run.
DEFAULT_POOL_SIZE = 30

# The demo build is driven through this makefile, matching get-timings.sh.
_UXHW_MAKEFILE = "Makefile.pro"

# Build variables get-timings.sh pins for every UxHw build. Kept identical
# so a profiled binary is built the same way as a benchmarked one.
_TARGET_ARCH = "x86_64-unknown-linux-gnu"
_M_CONFIG_FILE = "file.m"

# Build resources an application does not ship, staged into its src/ the
# same way copy_build_resources does in get-timings.sh. Each entry is a
# path relative to the bundled resources directory.
_BUILD_RESOURCES = (
    "C0/init.S",
    "common/startup.cpp",
    "C0Pro/Makefile.pro",
    "C0/Makefile",
)

# The emulator configuration the build consumes. Copied from
# write_emulator_config in get-timings.sh, minus the trace statements,
# which exist to record traced values and have no bearing on how many
# pool elements a run needs.
_EMULATOR_CONFIG = """noDetach 1

--
-- UR Type and Order do not affect this pass; no Monte Carlo seed is set.
--
newNode riscv UncertaintyRepresentationRQHR 64
cacheOff
ff
sizemem 50000000

ChangeGuestWorkingDir "../inputs"
setDbFilename "memory-profile-dummy.db" "MemoryProfileTable"
run
on
dumpInstructionDist
q
"""


class ProfileError(Exception):
    """A build, a run, or the search itself could not be completed."""


def parse_pool_capacity(output: str) -> int:
    """
    Read the pool capacity a run finished on.

    Args:
        output: The combined stdout and stderr of one run.

    Returns:
        The capacity in pool elements.

    Raises:
        ProfileError: If the run printed no capacity line. That means the
            binary is not a UxHw build, or it died before exiting
            cleanly, and in both cases the number would be a guess.
    """
    matches = _POOL_CAPACITY_PATTERN.findall(output)
    if not matches:
        raise ProfileError(
            "the run printed no 'Real Allocs:' line, so it reported no "
            "pool capacity. The binary either is not a UxHw build or did "
            "not reach a clean exit."
        )

    # The last line wins. A demo that runs several phases prints one per
    # phase, and the final one covers the whole run.
    return int(matches[-1])


def check_capacity_is_reachable(*, probed_size: int, reported_capacity: int) -> None:
    """
    Check a reported capacity could have come from the probed size.

    A pool starting at S only ever reports S * 2**n. Anything else means
    the build did not apply the requested starting size, and every later
    probe would be measuring a binary we did not ask for.

    Without this the search fails silently rather than loudly. If the
    build ignores ``UNCERTAIN_MEMORY_POOL_SIZE`` entirely, every run
    reports the default-derived capacity. A probe at 45 then reports 60,
    the search reads 60 as "45 was too small", and it converges on 60
    while the true answer is 37. That is a plausible wrong number, which
    is worse than an error.

    Args:
        probed_size: The starting pool size the binary was built with.
        reported_capacity: The capacity that run finished on.

    Raises:
        ProfileError: If the capacity is not the probed size doubled some
            whole number of times.
    """
    capacity = probed_size
    while capacity < reported_capacity:
        capacity *= 2

    if capacity != reported_capacity:
        raise ProfileError(
            f"a run built with a pool of {probed_size} reported a capacity "
            f"of {reported_capacity}, which is not {probed_size} doubled "
            "any number of times. The build is not applying "
            "UNCERTAIN_MEMORY_POOL_SIZE, so the search would converge on "
            "a number that describes a different binary."
        )


def initial_bracket(*, probed_size: int, reported_capacity: int) -> tuple[int, int]:
    """
    Turn one probe into the range the answer must lie in.

    The pool doubles, so a run that reports more than it started with has
    overflowed at least once. Reporting exactly the starting size means it
    never overflowed, which bounds the answer from above but not below.

    Args:
        probed_size: The starting pool size the binary was built with.
        reported_capacity: The capacity that run finished on.

    Returns:
        An inclusive ``(low, high)`` range containing the peak occupancy.

    Raises:
        ProfileError: If the capacity could not have come from the probed
            size by doubling, which means the build ignored it.
    """
    check_capacity_is_reachable(
        probed_size=probed_size,
        reported_capacity=reported_capacity,
    )

    if reported_capacity == probed_size:
        return 1, probed_size

    # It doubled at least once, so the last doubling was the one that was
    # needed. The answer is above half the final capacity.
    return reported_capacity // 2 + 1, reported_capacity


def search_peak_elements(
    *,
    probe: Callable[[int], int],
    start_size: int = DEFAULT_POOL_SIZE,
) -> tuple[int, int]:
    """
    Find the exact peak pool occupancy of one configuration.

    Bisects for the smallest starting size that the run never has to grow
    past. That size is the peak occupancy, because a pool grows exactly
    when the next element will not fit.

    Args:
        probe: Builds at a given starting pool size, runs, and returns the
            capacity the run finished on.
        start_size: The starting size of the first probe.

    Returns:
        A tuple of the peak occupancy in elements and the number of probes
        it took, so a caller can report the cost it paid.

    Raises:
        ProfileError: If ``start_size`` is not positive, or if any probe
            reports a capacity its starting size could not have produced.
    """
    if start_size < 1:
        raise ProfileError(f"start_size must be at least 1, got {start_size}")

    probes = 1
    low, high = initial_bracket(
        probed_size=start_size,
        reported_capacity=probe(start_size),
    )

    while low < high:
        candidate = (low + high) // 2
        probes += 1
        capacity = probe(candidate)
        check_capacity_is_reachable(
            probed_size=candidate,
            reported_capacity=capacity,
        )
        if capacity == candidate:
            # It never had to grow, so this size was enough.
            high = candidate
        else:
            low = candidate + 1

    return low, probes


def prepare_source_tree(*, source_directory: Path, path_to_uxhw_sdk: str) -> None:
    """
    Stage the build files an application does not ship.

    A demo checkout carries its own sources but not the build scaffolding,
    so a fresh clone has no ``Makefile.pro``, no startup code and no
    emulator configuration. ``get-timings.sh`` stages them on every run and
    this does the same, so the tool works on a clean checkout rather than
    only on a tree some earlier benchmarking run happened to leave behind.

    Args:
        source_directory: The demo's ``src`` directory.
        path_to_uxhw_sdk: Path to the UxHw SDK to build against.

    Raises:
        ProfileError: If a bundled resource is missing, which means the
            assets submodule was never initialised.
    """
    resources = get_resources_dir()
    for relative_path in _BUILD_RESOURCES:
        source = resources / relative_path
        if not source.exists():
            raise ProfileError(
                f"the bundled build resource {relative_path} is missing "
                f"from {resources}. The assets submodule is probably not "
                "initialised, so run 'git submodule update --init'."
            )
        shutil.copy(source, source_directory / Path(relative_path).name)

    (source_directory / _M_CONFIG_FILE).write_text(_EMULATOR_CONFIG, encoding="utf-8")

    # The makefile carries the SDK path as a literal, so it is rewritten
    # rather than passed in. Matches what get-timings.sh does with sed.
    makefile = source_directory / _UXHW_MAKEFILE
    rewritten = re.sub(
        r"^PATH_TO_UXHW_SDK\s*.*$",
        f"PATH_TO_UXHW_SDK={path_to_uxhw_sdk}",
        makefile.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    makefile.write_text(rewritten, encoding="utf-8")


def build_binary(
    *,
    source_directory: Path,
    pool_size: int,
    representation_type: str,
    representation_size: int,
    correlation: str,
    program_name: str,
    make_jobs: int,
) -> Path:
    """
    Build one UxHw binary with a given starting pool size.

    Mirrors ``build_uxhw_testcase_binary`` in ``get-timings.sh`` so the
    profiled binary matches the benchmarked one, with the pool size added.

    Args:
        source_directory: The demo's ``src`` directory.
        pool_size: Starting pool size, passed as
            ``UNCERTAIN_MEMORY_POOL_SIZE``.
        representation_type: For example ``Jupiter``.
        representation_size: For example ``8``.
        correlation: ``Disabled`` or ``Autocorrelation``.
        program_name: Output name for the built executable.
        make_jobs: Parallelism for the build.

    Returns:
        Path of the built executable.

    Raises:
        ProfileError: If either make invocation fails, or the build
            produces no executable at the expected path.
    """
    base = ["make", "-s", f"-j{make_jobs}", "-f", _UXHW_MAKEFILE]
    variables = [
        f"REPRESENTATION_TYPE={representation_type}",
        f"REPRESENTATION_SIZE={representation_size}",
        f"CORRELATION_TRACKING={correlation}",
        f"UNCERTAIN_MEMORY_POOL_SIZE={pool_size}",
        f"M_CONFIG_FILE={_M_CONFIG_FILE}",
        f"TARGET_ARCH={_TARGET_ARCH}",
        f"PROGRAM={program_name}",
    ]

    for command in (base + ["clean"], base + variables):
        completed = subprocess.run(
            command,
            cwd=source_directory,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise ProfileError(
                f"build failed at pool size {pool_size}: "
                f"{' '.join(shlex.quote(token) for token in command)}\n"
                f"{completed.stderr.strip()[-2000:]}"
            )

    executable = source_directory / program_name
    if not executable.exists():
        raise ProfileError(
            f"the build reported success at pool size {pool_size} but "
            f"produced no executable at {executable}"
        )

    return executable


def run_binary(*, executable: Path, demo_cli_args: str) -> int:
    """
    Run one built binary and read back the pool capacity it used.

    There is deliberately no timeout. Run time rises steeply with
    representation size, from seconds to tens of minutes on the same
    demo, so any limit chosen here would be wrong for some
    configuration. A caller that wants one can wrap the invocation in
    ``timeout``.

    Args:
        executable: The built UxHw executable.
        demo_cli_args: Extra arguments for the demo, as one string.

    Returns:
        The capacity in pool elements the run finished on.

    Raises:
        ProfileError: If the run exits non-zero.
    """
    command = [str(executable), *shlex.split(demo_cli_args)]
    completed = subprocess.run(
        command,
        cwd=executable.parent,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        raise ProfileError(
            f"{executable.name} exited {completed.returncode}\n"
            f"{completed.stderr.strip()[-2000:]}"
        )

    return parse_pool_capacity(completed.stdout + completed.stderr)


def profile_configuration(
    *,
    source_directory: Path,
    path_to_uxhw_sdk: str,
    representation_type: str,
    representation_size: int,
    correlation: str,
    demo_cli_args: str,
    make_jobs: int,
    start_size: int,
) -> dict:
    """
    Profile one configuration end to end.

    Args:
        source_directory: The demo's ``src`` directory.
        path_to_uxhw_sdk: Path to the UxHw SDK to build against.
        representation_type: For example ``Jupiter``.
        representation_size: For example ``8``.
        correlation: ``Disabled`` or ``Autocorrelation``.
        demo_cli_args: Extra arguments for the demo, as one string.
        make_jobs: Build parallelism.
        start_size: Starting pool size of the first probe.

    Returns:
        The result record, ready to serialise.
    """
    prepare_source_tree(
        source_directory=source_directory,
        path_to_uxhw_sdk=path_to_uxhw_sdk,
    )
    configuration = configuration_string(
        representation_type=representation_type,
        representation_size=representation_size,
        correlation=correlation,
    )
    # Named per configuration so a finished build is identifiable, not as
    # any concurrency guarantee. Every probe runs make clean in the shared
    # source directory, so two invocations against the same demo will
    # destroy each other's objects whatever the executables are called.
    # Sweep one demo serially, or give each invocation its own checkout.
    program_name = f"memory-profile-{configuration}"

    def probe(pool_size: int) -> int:
        executable = build_binary(
            source_directory=source_directory,
            pool_size=pool_size,
            representation_type=representation_type,
            representation_size=representation_size,
            correlation=correlation,
            program_name=program_name,
            make_jobs=make_jobs,
        )
        return run_binary(
            executable=executable,
            demo_cli_args=demo_cli_args,
        )

    peak_elements, probes = search_peak_elements(
        probe=probe,
        start_size=start_size,
    )

    return {
        "application": source_directory.parent.name,
        "uxhw_sdk_path": path_to_uxhw_sdk,
        "configuration": configuration,
        "representation_type": representation_type,
        "representation_size": representation_size,
        "correlation": correlation,
        "demo_command_line_arguments": demo_cli_args,
        "peak_uncertain_memory_pool_elements": peak_elements,
        "probes_taken": probes,
        "units": (
            "pool elements, not bytes. One element holds one "
            "distributional value of the representation named above, so "
            "the byte cost follows from the representation type and size."
        ),
    }


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser.

    Flag names follow ``automation/arguments.py`` wherever an equivalent
    exists, so the two tools read the same. The representation flags are
    singular here because this tool profiles one configuration and leaves
    sweeping to the caller.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        prog="signaloid-memory-profile",
        description=(
            "Find the exact number of uncertain memory pool elements a "
            "UxHw run needs, by bisecting on the pool size the binary is "
            "built with. One invocation profiles one configuration."
        ),
        epilog=(
            "Each probe stages build files into the demo's src directory "
            "and runs 'make clean' there, matching how get-timings.sh "
            "builds. Executables already built there are removed. They "
            "are artifacts rather than sources, so nothing is lost that "
            "cannot be rebuilt, but a tree prepared by hand comes back "
            "empty. For the same reason two invocations must not run "
            "against one demo checkout at the same time. Sweep a single "
            "demo serially, and parallelise across demos by giving each "
            "its own checkout."
        ),
    )
    parser.add_argument(
        "--path-to-application",
        dest="path_to_application",
        required=True,
        help="Path to the application/demo repository.",
    )
    parser.add_argument(
        "--path-to-uxhw-sdk",
        dest="path_to_uxhw_sdk",
        required=True,
        help=(
            "Path to the Signaloid UxHw SDK to build against. The pool "
            "element size depends on the SDK, so a profile is only "
            "meaningful against a named one."
        ),
    )
    parser.add_argument(
        "--representation-type",
        dest="representation_type",
        required=True,
        choices=REPRESENTATION_TYPE_CHOICES,
        help="Representation type to profile.",
    )
    parser.add_argument(
        "--representation-size",
        dest="representation_size",
        required=True,
        type=int,
        help="Representation size to profile.",
    )
    parser.add_argument(
        "--correlation",
        dest="correlation",
        default=Correlations.DISABLED,
        choices=CORRELATION_CHOICES,
        help=(
            "Correlation tracking mode. Tracking correlations needs state "
            "that disabling them does not, so this is its own axis."
        ),
    )
    parser.add_argument(
        "--demo-cli-args",
        dest="demo_cli_args",
        default="",
        help="Extra command-line arguments passed to the demo.",
    )
    parser.add_argument(
        "--start-pool-size",
        dest="start_pool_size",
        default=DEFAULT_POOL_SIZE,
        type=int,
        help=(
            "Starting pool size of the first probe. A closer guess saves "
            "probes but never changes the answer."
        ),
    )
    parser.add_argument(
        "--jobs",
        "-j",
        dest="make_jobs",
        default=1,
        type=int,
        help="Build parallelism for each probe.",
    )
    parser.add_argument(
        "--output",
        dest="output",
        help=(
            "Write the result as JSON here. One file per invocation, so "
            "concurrent invocations cannot corrupt each other. Prints to "
            "stdout when omitted."
        ),
    )

    return parser


def main() -> None:
    """
    Profile one configuration and write the result.

    Exits non-zero with a message on any build, run or search failure,
    rather than reporting a number it could not establish.
    """
    args = create_argument_parser().parse_args()
    source_directory = Path(args.path_to_application).expanduser() / "src"

    if not source_directory.is_dir():
        print(
            f"signaloid-memory-profile: {source_directory} is not a "
            "directory, so this is not a demo checkout",
            file=sys.stderr,
        )
        raise SystemExit(1)

    try:
        record = profile_configuration(
            source_directory=source_directory,
            path_to_uxhw_sdk=args.path_to_uxhw_sdk,
            representation_type=args.representation_type,
            representation_size=args.representation_size,
            correlation=args.correlation,
            demo_cli_args=args.demo_cli_args,
            make_jobs=args.make_jobs,
            start_size=args.start_pool_size,
        )
    except ProfileError as error:
        print(f"signaloid-memory-profile: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    text = json.dumps(record, indent=2) + "\n"
    if args.output:
        Path(args.output).expanduser().write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
