# Profiling

Tools that answer questions about a UxHw application's resource use,
kept apart from the benchmarking automation pipeline in `../automation/`.

## Memory profile

`signaloid-memory-profile` reports the exact number of uncertain memory
pool elements one UxHw run needs, so the memory a representation requires
can be specified rather than guessed.

```
signaloid-memory-profile \
    --path-to-application ~/my-application \
    --representation-type Athens \
    --representation-size 4 \
    --output athens-4.json
```

One invocation profiles one configuration. Sweeping is the caller's job,
which stops one failed configuration taking a whole sweep with it.

Do not run two invocations against the same demo checkout at once. Every
probe runs `make clean` in the shared `src` directory, so concurrent
invocations destroy each other's objects no matter what the executables
are called. Sweep a single demo serially and parallelise across demos,
each with its own checkout.

### Why not a heap profiler

Neither of the usual tools can measure these binaries.

Valgrind cannot execute them. The core library is built for AVX-512 and
valgrind does not support those instructions, so it aborts on the first
EVEX-prefixed opcode.

Heaptrack, `memusage` and any other `LD_PRELOAD` interposer attach to
nothing. The demos are statically linked, by design, at a fixed load
address for the emulator. There is no dynamic symbol to interpose on.

### How it works instead

The uncertain memory pool starts at a size fixed at compile time and
doubles whenever it runs short. At exit the runtime prints the capacity
it finished on, as `Real Allocs: <n>`.

So one run answers "was the starting size enough", because the reported
capacity stays at the starting size exactly when it was never grown. That
predicate is monotone in the starting size, so bisecting on it finds the
peak occupancy exactly.

Worked example on the DAE solver demo at Athens-4. Starting at the
default 30 reports 60, so one doubling happened and the answer lies
between 31 and 60. Bisecting lands on 37. Checked by hand, a build
starting at 36 grows to 72 and a build starting at 37 stays at 37.

### Cost

The starting pool size is an LLVM pass flag, not a runtime option, so
each probe is a rebuild and a rerun. The search is logarithmic, around
six probes, but the runs dominate. On the DAE solver demo a single run
takes 4.6 seconds at Athens-4 and 1117 seconds at Athens-64, so expect a
full sweep to be an overnight job.

There is no timeout option. Any limit would be wrong for some
configuration, given that spread. Wrap the invocation in `timeout` if you
want one.

### Units

The result is a count of pool elements, not bytes. One element holds one
distributional value of the profiled representation, so the byte cost
follows from the representation type and size, which are recorded
alongside the count.

### Side effect

Each probe runs `make clean` in the demo's `src` directory, matching how
`get-timings.sh` builds. Executables already built there are removed.
They are build artifacts rather than sources, so nothing is lost that
cannot be rebuilt, but a tree prepared by hand will come back empty.
