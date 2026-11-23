# Benchmarking Automation

`signaloid-benchmarking` benchmarks a UxHw® application against Monte Carlo.
It builds the application natively and with UxHw, and generates a ground
truth. It then measures how close each UxHw configuration gets to the ground
truth, computes its equivalent Monte Carlo count (EMCC), and times both
builds. The results go to CSV files and Markdown reports.

This guide covers a normal run. The [reference](reference.md) covers every
option, each pipeline step, and the output formats.

## Requirements

- Linux on an x86-64 CPU. A CPU with AVX-512 gives the fastest UxHw runs.
  See [Supported platforms](../../../README.md#supported-platforms).
- The Signaloid UxHw SDK. It comes installed on the Signaloid Computing
  Engine Amazon Machine Image (AMI). Pass its location with
  `--path-to-uxhw-sdk`, for example `/opt/Signaloid-UxHw-SDK`. Without a
  local SDK, you can run the UxHw passes on the Signaloid Cloud Compute
  Engine instead. See
  [Running on the Signaloid Cloud Compute Engine](#running-on-the-signaloid-cloud-compute-engine).
- The `signaloid-benchmarking` package. See
  [Installation](../../../README.md#installation).
- GCC (`gcc` and `g++`), GNU Make, Bash, and `lscpu`.
- The GNU Scientific Library (`libgsl-dev` on Ubuntu), if the application
  uses it. Link it with `-lgsl -lgslcblas -lm`.

## Preparing the application

To try the tool before you prepare your own application, use the public
[geometric Brownian motion demo](https://github.com/signaloid/Signaloid-Demo-Finance-OosterleeGrzelakBookGeometricBrownianMotionExample).
It is ready to benchmark.
```bash
git clone --recursive https://github.com/signaloid/Signaloid-Demo-Finance-OosterleeGrzelakBookGeometricBrownianMotionExample ~/gbm-demo
```

For your own application, start from the
[C project template](https://github.com/signaloid/Signaloid-Demo-General-C).
Applications built this way use the
[common utility routines](https://github.com/signaloid/Signaloid-Demo-CommonUtilityRoutines)
for the command-line options and the output file that the tool expects. See
[How the tool runs the program](#how-the-tool-runs-the-program).

The tool benchmarks a git repository with the following layout.

```
my-application/
├── signaloid.yaml      # Required. Declares the variables to benchmark.
└── src/
    ├── main.c          # Writes each output to an array, selects one with -S <n>.
    └── config.mk       # Optional. SOURCES and CFLAGS for the native build.
```

- **It must be a git repository.** The commit hash versions the output files.
- **It must have a `signaloid.yaml` at its root.** Without it, the run stops
  at the `Loading application info` step.
- **Its source must be in `src/`.**

The native Monte Carlo build comes from a `Makefile` with a `local-build`
target, or else from `src/config.mk` with `SOURCES`. The
[C project template](https://github.com/signaloid/Signaloid-Demo-General-C)
uses `config.mk`. With neither, the tool skips the native baseline, so it
reports no speedup. Check that the native build works before you start.
```bash
cd ~/my-application
make local-build
```

### `signaloid.yaml`

```yaml
TraceVariables:
  - Expression: "outputVariables[0:1]"
    File: "main.c"
    LineNumber: "42"

BenchmarkingAllOutputs:
  - CommandLineArguments: "-S 2"

BenchmarkingVariables:
  - VariableName: "outputVariables[0]"
    VariableDescription: "Option price"
    OutputObject: "Distribution"          # or "Scalar"
    CommandLineArguments: "-S 0"
  - VariableName: "outputVariables[1]"
    VariableDescription: "Option delta"
    OutputObject: "Scalar"
    CommandLineArguments: "-S 1"
```

- `TraceVariables` lists the C expressions to trace, with their file and
  line. A range such as `outputVariables[0:1]` expands to each element, and
  the end index is inclusive. See
  [`LineNumber` is where the variable is declared](#linenumber-is-where-the-variable-is-declared).
- `BenchmarkingAllOutputs` gives the arguments that make the program compute
  every output at once. The tracing runs use them.
- `BenchmarkingVariables` lists each output to benchmark, with the arguments
  that make the program compute only that output.

Each `BenchmarkingVariables` entry can also override the representation
types, sizes, and dependency tracking, or name its own analytic ground
truth. See [Per-variable fields](reference.md#per-variable-fields).

The tool checks `signaloid.yaml` before it builds anything. It stops the run
if a key is missing, if an `OutputObject` is not `Distribution` or `Scalar`,
or if a `VariableName` matches no traced expression. The error lists every
problem at once and suggests the closest traced name.

### `LineNumber` is where the variable is declared

Each `TraceVariables` entry must point at the line that declares the traced
variable, not a line that assigns or uses it. The tracer finds the variable
through its declaration. A line that only uses the variable gives an empty
tracing database, with no error.

```c
42	double	outputVariables[2];		/* LineNumber: "42" is right. */
...
87	outputVariables[0] = optionPrice;	/* LineNumber: "87" is wrong. */
```

For an array element or a struct member, such as `outputVariables[0:1]` or
`arguments.value`, use the line that declares the array or struct,
`outputVariables` or `arguments`. If the name is declared more than once,
use the declaration in the scope where you trace it. A declaration can name
more than one variable, as in `double x, outputVariables[2];`. If it spans
lines, use the line that holds the name.

The tool checks every entry before it builds anything. A line that does not
hold the name, or that uses it, stops the run with
`signaloid.yaml TraceVariables do not match the source`. The message
suggests the line that declares the variable. A line that holds the name in
a layout that the check does not recognize, such as a function parameter or
a declaration made by a macro, only gives a warning.

### How the tool runs the program

The tool runs the program with the arguments from `signaloid.yaml`, and adds
the following. The common utility routines implement all of it.

- **`-S <n>` selects one output.** The program computes and prints output
  `n` only. With `n` equal to the number of outputs, it computes them all.
- **`-M <count>` runs Monte Carlo.** The native build runs the computation
  `<count>` times with sampled inputs. It then writes `data.out` in its
  working directory. The first line is the CPU time in microseconds, and the
  tool skips it. Each later line holds one sample. A `Scalar` output writes
  one value. The tool runs each Monte Carlo run in a new temporary directory,
  with the files from the application's `inputs/` directory linked in.
- **Every run prints `CPU time used: <seconds> seconds`.** The timing steps
  read this line.

## Running a benchmark

### Check the setup with a quick run

A full run takes hours, so start with a small one. Its sizes are too small
for results that you report. This command runs it on the demo from
[Preparing the application](#preparing-the-application). For your own
application, change `--path-to-application`.

The tool writes its results and logs to the current directory, so start in
an empty one. A second run in the same directory overwrites the first.
```bash
mkdir ~/quick-run && cd ~/quick-run
signaloid-benchmarking \
    --path-to-application ~/gbm-demo \
    --path-to-uxhw-sdk /opt/Signaloid-UxHw-SDK \
    -u Athens \
    -s 16 64 \
    -d None \
    -r Mean \
    --ground-truth-type MonteCarlo \
    --ground-truth-size 10000 \
    --adversary-mc-size 10000 \
    --num-adversaries 10
```

Add `--print-config` to any command to see the resolved settings and the
benchmark matrix without running anything.

### Recommended settings

Use these settings for results that you report. The command benchmarks
Athens. For Rome, change `-u Athens` to `-u Rome`. For Europa or Mercury,
also change `-d None Basic` to `-d Advanced`, because they only take
Advanced dependency tracking.
```bash
signaloid-benchmarking \
    --path-to-application ~/my-application \
    --path-to-uxhw-sdk /opt/Signaloid-UxHw-SDK \
    -u Athens \
    -s 16 32 64 128 256 512 \
    -d None Basic \
    -r Mean Quantile-95 Quantile-99 \
    --ground-truth-type WeightedSamples \
    --ground-truth-size 2000000 \
    --max-num-weighted-samples 500000 \
    --adversary-mc-size 1000000 \
    --adversary-max-size-scalar 100000 \
    --num-adversaries 100 \
    --use-clt
```

Jupiter runs out of memory above size 32, so it uses smaller sizes.
```bash
signaloid-benchmarking \
    --path-to-application ~/my-application \
    --path-to-uxhw-sdk /opt/Signaloid-UxHw-SDK \
    -u Jupiter \
    -s 8 16 32 \
    --max-jupiter-size 32 \
    -d Advanced \
    -r Mean Quantile-95 Quantile-99 \
    --ground-truth-type WeightedSamples \
    --ground-truth-size 2000000 \
    --max-num-weighted-samples 500000 \
    --adversary-mc-size 1000000 \
    --adversary-max-size-scalar 100000 \
    --num-adversaries 100 \
    --use-clt
```

These commands match the `athens.yaml`, `europa.yaml`, and `jupiter.yaml`
presets. The settings do the following.

| Flag | Why |
|---|---|
| `--ground-truth-type WeightedSamples` | Collapses the ground truth into weighted samples, so every distance is faster to compute. |
| `--ground-truth-size 2000000` | The sampling error of the ground truth limits every distance. Keep it well above the largest EMCC that you want to resolve. |
| `--max-num-weighted-samples 500000` | More weighted samples keep the ground truth closer to the raw samples. Fewer make it faster to compare against. |
| `--adversary-mc-size 1000000` | The pool that the Monte Carlo adversaries are drawn from. |
| `--adversary-max-size-scalar 100000` | The largest scalar EMCC that the tool can resolve. |
| `--num-adversaries 100` | Fewer runs make `Quantile-99` close to the largest distance. Run time grows in proportion. |
| `--use-clt` | Predicts the EMCC from the asymptotic distance distribution. It is much faster than measuring it at these sizes. Without it, the tool only measures a distribution EMCC up to 50000. |

The commands do not set `-j`. By default, the tool uses every core.

### Configuration files

`--config <file>` reads the settings from a YAML file. The presets in
`src/signaloid_benchmarking/configs/` hold the recommended settings
(`athens.yaml`, `europa.yaml`, `jupiter.yaml`, and `full-sweep.yaml` for
Athens, Rome, Jupiter, and Europa) and the quick run (`quick.yaml`). In an installed package, print
their location with this command.
```bash
python -c "import pathlib, signaloid_benchmarking; print(pathlib.Path(signaloid_benchmarking.__file__).parent / 'configs')"
```

Command-line flags override the file, and the file overrides the defaults.
The keys are the `argparse` destination names, not the flag names. For
example, `--num-adversaries` is `n_adversaries`. The tool rejects unknown
keys. See [Configuration file keys](reference.md#configuration-file-keys).

### Main options

The tool needs `--path-to-application` and the four sweep options. Each sweep
option takes one or more values.

| Option | Values |
|---|---|
| `-u`, `--representation-types` | `Athens`, `Rome`, `Jupiter`, `Europa`, `Mercury` |
| `-s`, `--representation-sizes` | Integers, such as `16 32 64` |
| `-d`, `--dependency-tracking` | `None`, `Basic`, `Advanced` |
| `-r`, `--reporting-methods` | `Mean`, `Quantile-95`, `Quantile-99` |

Athens and Rome take `None` and `Basic`. Jupiter, Europa, and Mercury take
`Advanced` only. The tool skips any other pair, so `-u Athens Jupiter -d None
Advanced` runs `Athens` with `None` and `Jupiter` with `Advanced`. See
[Configurations that cannot run](reference.md#configurations-that-cannot-run)
for what each SDK supports.

The earlier names still work. These are `Atlas` for `Rome`, `-c` for `-d`,
`Autocorrelation` for `Basic`, and `Disabled` for `None`. For Jupiter,
Europa, and Mercury, `Disabled` means `Advanced`, which builds what
`Disabled` built before. So an old `-u Jupiter -c Disabled` still runs
Jupiter. This holds in configuration files and `signaloid.yaml` too.

The options that you are most likely to change are the following.

| Option | Default | Description |
|---|---|---|
| `--path-to-uxhw-sdk` | `~/project-uxhw-sdk` | Path to the UxHw SDK, for example `/opt/Signaloid-UxHw-SDK`. |
| `-j`, `--jobs` | Detected core count | Maximum number of parallel workers. UxHw runs always run one at a time. |
| `--demo-cli-args` | `""` | Extra arguments for every run of the application, for example `"--asc-file inputs/blink.asc"`. |
| `--has-analytic-ground-truth` | Off | Use an analytic ground truth. Needs `--ground-truth-type WeightedSamples` and `--path-to-ground-truth-file`. |
| `--print-config` | Off | Print the resolved settings and exit. |

Run `signaloid-benchmarking --help` or see the
[reference](reference.md#command-line-options) for every option.

## Optional features

### Counting dynamic instructions

Pass `--measure-dynamic-instructions` to also count the dynamic
instructions of each run with Intel Pin. This flag is the only switch. A run
without it never uses Intel Pin, even when `PIN_ROOT` is set.

1. Download Intel Pin 4.2 from the
   [Intel Pin downloads](https://www.intel.com/content/www/us/en/developer/articles/tool/pin-a-binary-instrumentation-tool-downloads.html)
   page, and extract it to a directory `<dir>`.
2. Build the instruction counter once. The kit does not include it.
   ```bash
   make -C <dir>/source/tools/ManualExamples obj-intel64/inscount0.so
   ```
3. Point `PIN_ROOT` at the kit, and add the flag to your command.
   ```bash
   export PIN_ROOT=<dir>
   ```

With the flag, a kit that is missing or not built is an error.

### Spreadsheet upload

Pass `--write-sheets` to also upload the results to a Google Sheets
spreadsheet. The upload needs the following. If any of them is missing, the
tool stops at startup, before it runs anything.

- The `sheets` extra. See [Installation](../../../README.md#installation).
- A Google Cloud service-account credentials file, given with
  `--google-credentials <path>` or `GOOGLE_APPLICATION_CREDENTIALS`.
- The `UXHW_SHEETS_DRIVE_FOLDER_ID` and `UXHW_SHEETS_TEMPLATE_ID`
  environment variables, for the target folder and the
  [spreadsheet template](https://docs.google.com/spreadsheets/d/1MKPkwY_B_m5WH-coqIy19Otpg3IE5yRSKaKGw6wxzL8).

### Running on the Signaloid Cloud Compute Engine

Pass `--run-uxhw-scce` to build and run the UxHw passes on the Signaloid
Cloud Compute Engine (SCCE) instead of with a local SDK. The native passes
still run locally. This mode needs the following.

- `signaloid-cli` and `jq` on your `PATH`.
- A checkout of the
  [`signaloid-cli` helper scripts](https://github.com/signaloid/signaloid-cli-helpers),
  with the `SIGNALOID_SCCE_API_KEY`, `GITHUB_TOKEN_FOR_SCCE_CONNECTION`, and
  `GITHUB_USERNAME` files next to them. Pass `--path-to-cli-helpers` if it is
  not at `~/signaloid-cli-helpers`.
- An application cloned from GitHub, at the commit that its origin has for
  the current branch, with no local edits. SCCE builds the pushed commit, and
  the native passes build your working tree, so the two must match.

SCCE has Athens and Jupiter cores only, so the tool skips other types. See
[Signaloid Cloud Compute Engine mode](reference.md#signaloid-cloud-compute-engine-mode)
for how the runs and timings work.

## Reading the results

The results are in `results/<description>.md`, one report for each output in
`signaloid.yaml`. `<description>` is the output's `VariableDescription` in
lowercase, with spaces and punctuation replaced by `-`.

Each row of the report is one configuration and one reporting method. The
**`Speedup`** column answers the main question. It is how many times faster
UxHw is than native Monte Carlo at the same accuracy. To find the best
configuration for a reporting method, take its row with the highest
`Speedup`.

| Column | Meaning |
|---|---|
| `UxHw Conf` | The configuration, such as `Athens-64-Basic` or `Jupiter-16-Advanced`. |
| `Reporting Method` | `Mean`, `Quantile-95`, or `Quantile-99`. |
| `UxHw Distance` | The distance from the UxHw output to the ground truth. |
| `Binned UxHw Distance` | The same, with the binned distance. The EMCC uses it by default. |
| `EMCC Predicted` | The EMCC predicted from the asymptotic distance distribution. |
| `EMCC` | The Monte Carlo sample count that matches UxHw's accuracy, and that the native baseline was timed at. With `--use-clt`, it is the predicted count. Without it, it is measured from the adversary runs. |
| `% MC beats UxHw` | The share of adversary runs at the EMCC that are closer to the ground truth than UxHw. It is a fraction from 0 to 1. |
| `In Application Time` | The CPU time that the UxHw run prints, in seconds. |
| `Database Time` | The host wall-clock time of the UxHw run, in seconds. The speedup divides by this. |
| `End-to-End Time` | The wall-clock time of the whole UxHw process, including start-up. |
| `Native In Application Time` | The CPU time that the native Monte Carlo build prints at the EMCC. The speedup divides this by `Database Time`. |
| `Native End-to-End Time` | The wall-clock time of the whole native process at the EMCC. |

Keep the following in mind when you read the numbers.

- **An `EMCC` of 1 is not a real count.** It means that no adversary sample
  count was as accurate as UxHw, so the true EMCC is above the largest one
  tested. Its speedup is meaningless.
- **`blow-up / excluded`** replaces the distance and EMCC of a configuration
  whose representation degraded. Do not use those rows.
- **A missing `Speedup`** means the application has no native build, or that
  configuration has no timing.
- **CLT warnings** under the table mean the assumptions behind the EMCC
  prediction may not hold for this output. Treat its predicted EMCC with
  care.

## Outputs

The tool writes to `results/` and `logs/` in the current directory.

| File | Contents |
|---|---|
| `results/<description>.md` | One report for each output. See [Reading the results](#reading-the-results). |
| `results/output_data.csv` | The distance and predicted EMCC of each configuration and reporting method. It has no timings or speedups. |
| `results/uxhw_distances.csv` | The distance from each UxHw configuration to the ground truth. |
| `results/asymptotic_distances.csv` | The asymptotic distance distribution of each variable. |
| `results/<app-name>-<hash>-timings.json` | All timing measurements. |
| `results/*.db` | SQLite databases for the ground truth, adversary, and tracing data. |
| `results/plots/` | Plots, when you turn them on. |
| `logs/` | Error, build, and timing logs. |

See [Output files](reference.md#output-files) for the formats.

## Troubleshooting

- **The run stopped with an error.** Read
  `logs/benchmarking_automation_error_<timestamp>.log` for the full stack
  trace. Build warnings are in `logs/uxhw-build.log` and
  `logs/native-mc-build.log`. UxHw build and run errors are in
  `logs/exec.stderr` and `logs/opt.err`.
- **`signaloid.yaml TraceVariables do not match the source`.** A
  `LineNumber` does not point at the line that declares its variable. Use
  the line that the message suggests. See
  [`LineNumber` is where the variable is declared](#linenumber-is-where-the-variable-is-declared).
- **The native build fails.** If the application uses GSL, check that
  `libgsl-dev`, or the equivalent package, is installed. Also check that
  `/opt/local/lib` and `/opt/local/include` exist on your system, or that the
  application's `Makefile` sets its own library paths. The build must put
  `demo-native-mc` in the application root.
- **A configuration is missing from the results.** The tool skips a
  combination that the SDK has no library for, a Jupiter size above
  `--max-jupiter-size`, and a run that fails, for example when it runs out
  of memory. Each variable's report lists its skipped combinations with the
  reason. See [Configurations that cannot run](reference.md#configurations-that-cannot-run).
- **`Cannot continue without EMCC data`.** Step 7 did not write
  `results/output_data.csv`. Find the first error in the output up to step
  7, fix it, and run again.
- **The run stopped part way.** The tool does not resume a run. The files
  that it wrote so far stay in `results/`, but they are incomplete. Run
  again from the start.
- **`--use-binned-uxhw` fails with `Wasserstein-2`.** The binned distance
  only works with `Wasserstein-1`. Add `--no-use-binned-uxhw`.
- **The application cannot find an input file.** The tool links the files in
  the application's `inputs/` directory flat into the working directory. So
  the application's default paths must not include the `inputs/` prefix.
- **Timing data from an earlier run looks mixed in.** Delete
  `results/<app-name>-<hash>-timings.*` and run again.
