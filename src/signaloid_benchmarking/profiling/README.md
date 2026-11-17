# Profiling

Tools that measure the resource use of a UxHw® application. They are
separate from the benchmarking pipeline, but need the same setup, a Linux
x86-64 machine and the UxHw SDK. See
[Supported platforms](../../../README.md#supported-platforms).

## Memory profile

`signaloid-memory-profile` reports the exact number of uncertain memory
pool elements that one UxHw run needs. You can then specify the memory a
representation requires instead of guessing it.

```
signaloid-memory-profile \
    --path-to-application ~/my-application \
    --path-to-uxhw-sdk /opt/Signaloid-UxHw-SDK \
    --representation-type Athens \
    --representation-size 4 \
    --output athens-4.json
```

The output names the configuration and gives the peak pool size in
elements. Without `--output`, it prints to standard output.

```json
{
  "application": "my-application",
  "uxhw_sdk_path": "/opt/Signaloid-UxHw-SDK",
  "configuration": "Athens-4",
  "representation_type": "Athens",
  "representation_size": 4,
  "dependency_tracking": "None",
  "demo_command_line_arguments": "",
  "peak_uncertain_memory_pool_elements": 37,
  "probes_taken": 6,
  "units": "pool elements, not bytes. ..."
}
```

`-d` sets the dependency tracking. It defaults to `Advanced` for Jupiter,
Europa, and Mercury, which take nothing else, and to `None` for Athens and
Rome. Pass `-d Basic` to profile Athens or Rome with Basic dependency
tracking.

Run `signaloid-memory-profile --help` for the full list of options.

One invocation profiles one configuration. To sweep configurations, run
one invocation for each. A failed configuration then does not stop the
rest of the sweep.

Do not run two invocations against the same demo checkout at once. Every
probe runs `make clean` in the shared `src` directory, so concurrent
invocations destroy each other's objects no matter what the executables
are called. Sweep a single demo serially, and parallelize across demos,
each with its own checkout.

### Why heap profiling tools do not work

The usual heap profiling tools cannot measure these binaries.

`valgrind` cannot execute them on a CPU with AVX-512. There, the core
library is built for AVX-512, and `valgrind` does not support those
instructions, so it stops at the first EVEX-prefixed opcode.

`heaptrack`, `memusage`, and other tools that use `LD_PRELOAD` have nothing to
intercept. The demos are statically linked, by design, at a fixed load
address for the emulator. There is no dynamic symbol to interpose on.

### How it works instead

The uncertain memory pool starts at a size fixed at compile time and
doubles whenever it runs short. At exit the runtime prints the capacity
it finished on, as `Real Allocs: <n>`.

So one run answers "was the starting size enough", because the reported
capacity stays at the starting size exactly when the pool never grew. That
predicate is monotone in the starting size, so a bisection on it finds the
peak occupancy exactly.

For example, profile the DAE solver demo at Athens-4. A run that starts at
the default size of 30 reports 60. So the pool doubled once, and the answer
is between 31 and 60. The bisection finds 37. A manual check confirms it. A
build that starts at 36 grows to 72, and a build that starts at 37 stays at
37.

### Cost

The starting pool size is an LLVM pass flag, not a runtime option, so
each probe is a rebuild and a rerun. The search is logarithmic, with about
six probes, but the run time dominates. On the DAE solver demo, a single run
takes 4.6 seconds at Athens-4 and 1117 seconds at Athens-64. Expect a full
sweep to take a night.

There is no timeout option, because run times differ so much between
configurations that any fixed limit would be wrong for some of them. To set
a limit, wrap the invocation in `timeout`.

### Units

The result is a count of pool elements, not bytes. One element holds one
distributional value of the profiled representation. The byte cost follows
from the representation type and size, which the output records next to the
count.

### Side effect

Each probe runs `make clean` in the demo's `src` directory, matching how
`get-timings.sh` builds. The probe removes any executables already built
there. They are build artifacts, not sources, so you can rebuild anything
the probe removes. A directory that you prepared by hand loses its
executables.
