# Benchmarking Automation

The signaloid_benchmarking.automation package enables the benchmarking
applications running with Signaloid UxHw technology for computing with
distributions, against Monte Carlo baselines. The tool compiles the subject
application from source code for native Monte Carlo execution, generates the
ground truth result dataset and databases used for the comparison to the Monte
Carlo baselines. It runs the UxHw variant of the application, computes
equivalent Monte Carlo (EMCC) metrics, and collects timing data. The tool writes
all results to CSV data files, Markdown reports, and (optionally) uploads to a
Google Sheet.

## Requirements and dependencies

The tool requires the following system dependencies, Python dependencies, and
third-party dependencies.

### System dependencies
- gcc or g++
- GNU Scientific Library (on Ubuntu: libgsl-dev).
- Python (3.11+, see root-level pyproject.toml)
- GNU Make (make)
- bash
- lscpu

C and C++ files are compiled separately and linked with `c++`. Applications that
link against GSL need `-lgsl -lgslcblas -lm`.

### Python
This tool is part of the `signaloid_benchmarking` package. Create and activate a
virtual environment, then install the package with pip:
```
python -m venv .venv
source .venv/bin/activate
pip install .
# Or, to also get the optional Google Sheets upload:
pip install ".[sheets]"
```
For development, you can install in editable mode with `pip install -e .`
instead.

The benchmarking pipeline's application dependencies (`pandas`, `tabulate`,
`tqdm`) are non-optional. Optional dependencies is group `sheets`, needed
for the Google Sheets upload (see [Google Sheets](#google-sheets-optional)):
install with `pip install ".[sheets]"`.

Commands in this guide assume you are using the virtual environment, so that the
`signaloid-benchmarking` entry point and `python -m
signaloid_benchmarking.automation` are available.

### Signaloid UxHw SDK

The benchmarking tool needs the Signaloid UxHw SDK to compile applications for
UxHw. Set the path to the UxHw SDK via command-line parameter
`--path-to-uxhw-sdk` (for example, `/opt/Signaloid-UxHw-SDK`).

Without a local SDK, pass `--run-uxhw-scce` to build and run the UxHw passes
on the Signaloid Cloud Compute Engine (SCCE) instead. See
[Running the UxHw passes on SCCE](#running-the-uxhw-passes-on-scce) for what
that needs.

### Application requirements

A benchmarkable application is a project which can connect to the [Signaloid
Cloud Developer Platform](https://signaloid.io), plus a small amount of
benchmarking metadata. Minimal layout:

```
your-app/
├── src/
│   └── main.c          # UxHw program: writes each output to an array, selects one via `-S <n>`
├── signaloid.yaml      # REQUIRED: declares the trace + benchmarking variables
└── src/config.mk       # OPTIONAL: SOURCES/CFLAGS for the native-MC build
```

The target application must:
1. **Be a Git repository** — the current commit hash versions the output files.
2. **Contain a `signaloid.yaml` at its root** — see [Application
   Configuration](#application-configuration). This is what makes a repo
   benchmarkable. Without it the run aborts at "Loading application info".
3. **Have its source in a `src/` directory.**

The build file is **optional**. Native compilation (for the native-MC baseline)
is resolved in this order: a `Makefile` at the root with a `local-build` target,
else `src/config.mk` with `SOURCES` (and optional `CFLAGS`). The `Makefile` is
**not** required and a `config.mk` alone is enough (as in the [C project
template](https://github.com/signaloid/Signaloid-Demo-General-C)). If neither is
present (or if `config.mk` defines no `SOURCES`), the native-MC baseline is
skipped: the UxHw benchmark and distance analysis still run but it does not
compute the native-vs-UxHw speedup.

See the [Signaloid documentation for more details about using GitHub repositories
with UxHw](https://docs.signaloid.io/docs/api/guides/builds/builds-repository/).


### Intel PIN (Optional)

Intel PIN adds a dynamic instruction count to a timing run. It is optional and
off by default. The tool turns it on only when you pass
`--measure-dynamic-instructions`. The kit location comes from the `PIN_ROOT`
environment variable.

That flag is the single switch. A run without it never uses PIN, even when
`PIN_ROOT` is set in your shell. So you do not need to unset anything to turn
the measurement off.

Passing the flag is an explicit opt-in, so an unset `PIN_ROOT`, or a kit that
is missing or not built, is a hard error. Without the flag a timing run still
completes. It logs that it skipped the count and reports `pinDynInstCount` as
missing.

The dynamic instruction count (`pinDynInstCount`) is from the `inscount0` tool.

Basic installation instructions:
1. Download a PIN kit for your platform from Intel's
   [Pin binary-instrumentation tool downloads](https://www.intel.com/content/www/us/en/developer/articles/tool/pin-a-binary-instrumentation-tool-downloads.html)
   (validated against PIN 4.2).
2. Extract it to a directory `<dir>` of your choice.
3. Build the `inscount0` counter once. The kit does not ship it prebuilt.
   ```
   make -C <dir>/source/tools/ManualExamples obj-intel64/inscount0.so
   ```
4. `export PIN_ROOT=<dir>`, then run with `--measure-dynamic-instructions`.


### Google Sheets (Optional)

The benchmarking tool can optionally upload the benchmarking results to a Google
Sheet. It only runs when `--write-sheets` is passed and is is disabled by
default.

To enable it you need:
1. The `sheets` extra installed:
   ```
   pip install ".[sheets]"
   ```
   A plain `pip install .` does **not** pull in the Sheets stack (`gspread`,
   `google-api-python-client`, `oauth2client`). Passing `--write-sheets`
   without it raises `RuntimeError` at startup (before the pipeline runs),
   naming the modules it could not find.
2. A Google Cloud service-account credentials JSON file, supplied via
   `--google-credentials <path>` or the `GOOGLE_APPLICATION_CREDENTIALS`
   environment variable (whichever resolves to an existing file). There is no
   built-in default — passing `--write-sheets` without a resolvable credentials
   file raises `RuntimeError` at startup (before the pipeline runs).
3. The target Google Drive folder and Sheets template, supplied via the
   `UXHW_SHEETS_DRIVE_FOLDER_ID` and `UXHW_SHEETS_TEMPLATE_ID` environment
   variables. `--write-sheets` without both set also raises `RuntimeError` at
   startup.

The Sheets template is public:
<https://docs.google.com/spreadsheets/d/1MKPkwY_B_m5WH-coqIy19Otpg3IE5yRSKaKGw6wxzL8>.
The tab-name and metadata-row constants in `config.py` (`ReportSheetTabs`,
`MetadataRowLabels`) must match this template exactly. When it changes, read the
full tab names from its **ODS** export, never the XLSX export (XLSX truncates
tab names to 31 characters).


## Source setup

1. Clone this repository and install:
   ```
   git clone --recurse-submodules <repo-url>
   cd signaloid-benchmarking
   python -m venv .venv
   source .venv/bin/activate
   pip install ".[sheets]"
   ```
   (`pip install .` gives a fully working benchmarking tool. The `[sheets]`
   extra additionally pulls in the optional Sheets stack for the Google Sheets
   upload.)

2. Verify the UxHw SDK is available at its expected path (or note the path for
   use with `--path-to-uxhw-sdk`).

3. Verify that the target application compiles natively:
   ```
   cd /path/to/application
   make local-build   # if using a Makefile
   ```
   The compiled executable (`demo-native-mc`) should be located at the
   application root.

## Usage

Run the tool as a module:
```
python -m signaloid_benchmarking.automation [OPTIONS]
```
or via the installed entry point:
```
signaloid-benchmarking [OPTIONS]
```


### Required Arguments

| Flag | Description |
|---|---|
| `-u`, `--representation-types` | Uncertain representation types. One or more of: `Athens`, `Atlas`, `Jupiter`, `Europa`. |
| `-s`, `--representation-sizes` | Uncertain representation sizes (integers). E.g., `16 32 64 128 256 512`. |
| `-c`, `--uncertainty-correlation_types` | Correlation tracking types. One or more of: `Disabled`, `Autocorrelation`. |
| `-r`, `--reporting-methods` | Reporting methods. One or more of: `Mean`, `Quantile-95`, `Quantile-99`. |
| `--path-to-application` | Path to the application repository to benchmark. |

### Optional Arguments

| Flag | Default | Description |
|---|---|---|
| `--path-to-uxhw-sdk` | `~/project-uxhw-sdk` | Path to the UxHw SDK. Not read with `--run-uxhw-scce`. |
| `--run-uxhw-scce` | `False` | Build and run the UxHw passes (steps 4 and 10) on SCCE through the signaloid-cli helper scripts instead of the local UxHw SDK. See [Running the UxHw passes on SCCE](#running-the-uxhw-passes-on-scce). |
| `--path-to-cli-helpers` | `~/signaloid-cli-helpers` | Directory holding `signaloid-cli-build.sh`, `signaloid-cli-run.sh` and their credential files. Used with `--run-uxhw-scce`. |
| `--measure-dynamic-instructions` | `False` | Also measure dynamic instruction counts with Intel PIN, using the kit that `PIN_ROOT` points at. This flag is the only way to switch the measurement on, so a run without it never invokes PIN even when `PIN_ROOT` is set. Passing it without a usable `PIN_ROOT` is an error. |
| `--demo-cli-args` | `""` | Extra command-line arguments passed to both native-MC and UxHw executions. Use this when the demo application requires additional flags (e.g., `--demo-cli-args "--asc-file inputs/blink.asc"`). |
| `--ground-truth-size` | `1` | Number of Monte Carlo samples for ground truth generation. |
| `--ground-truth-type` | `MonteCarlo` | Type of ground truth: `MonteCarlo` or `WeightedSamples`. |
| `--path-to-ground-truth-file` | — | Path to the Python script that generates analytic ground truth. |
| `--distance-type` | `Wasserstein-1` | Distance metric: `Wasserstein-1` or `Wasserstein-2`. An EMCC prediction only exists for `Wasserstein-1`. |
| `--use-binned-uxhw` / `--no-use-binned-uxhw` | `True` | Use binned Wasserstein-1 for UxHw distances (only valid with `Wasserstein-1`). Pass `--no-use-binned-uxhw` (or `use_binned_uxhw: false` in a config) to disable. |
| `--use-clt` | `False` | Predict the equivalent-MC count from the asymptotic (CLT / Brownian-bridge) distance distribution instead of measuring it via explicit adversary MC. `Wasserstein-1` only. |
| `--adversary-mc-size` | `1` | Number of Monte Carlo samples for the adversary database. |
| `--adversary-max-size-scalar` | `100000` | Maximum adversarial MC size for scalar output variables. |
| `--num-adversaries` | `100` | Number of adversary repetitions. |
| `--max-num-weighted-samples` | `1` | Maximum number of weighted samples for ground truth conversion. |
| `-j`, `--jobs` | `1` | Number of parallel workers for native MC generation. |
| `--config` | `None` | Path to a YAML config file whose keys populate defaults (see [Config Files](#config-files)). |
| `--print-config` | `False` | Resolve defaults + config + CLI, print the effective config and the swept benchmark matrix, then exit without running. |
| `--write-sheets` | `False` | Opt in to the Google Sheets upload step (pipeline step 15). Requires the `sheets` extra, a resolvable credentials file (`--google-credentials` or `GOOGLE_APPLICATION_CREDENTIALS`), and the `UXHW_SHEETS_DRIVE_FOLDER_ID` + `UXHW_SHEETS_TEMPLATE_ID` environment variables. Missing any of these raises `RuntimeError` at startup (before the pipeline runs). |

For an extensive list of command line arguments for his program please run with the `--help` command.

### Example

```
python -m signaloid_benchmarking.automation \
    --path-to-application ~/Signaloid-Demo-Example \
    --path-to-uxhw-sdk ~/project-uxhw-sdk \
    -u Athens Atlas \
    -s 16 32 64 128 256 512 \
    -c Disabled Autocorrelation \
    -r Mean Quantile-95 Quantile-99 \
    --ground-truth-size 1000000 \
    --adversary-mc-size 1000000 \
    --distance-type Wasserstein-1 \
    -j 4
```

## Running the UxHw passes on SCCE

With `--run-uxhw-scce`, the UxHw tracing (step 4) and UxHw timing (step 10)
passes do not build the application with a local SDK. They build and run it on
the Signaloid Cloud Compute Engine through the
[signaloid-cli helper scripts](https://github.com/signaloid/signaloid-cli-helpers).
Everything else, including the native Monte Carlo passes, runs locally as
before.

It needs:

- `signaloid-cli` and `jq` on `PATH`.
- A checkout of the helper scripts, with the three credential files beside
  them (`SIGNALOID_SCCE_API_KEY`, `GITHUB_TOKEN_FOR_SCCE_CONNECTION`,
  `GITHUB_USERNAME`). Point `--path-to-cli-helpers` at it when it is not at
  the default location.
- The application to be a clone of a GitHub repository in the SCCE format,
  checked out at the commit its origin has for the current branch, with no
  edits under `src/`, `submodules/` or `inputs/`, or to `signaloid.yaml`
  or the `Makefile`, beyond the files the tool itself writes there. SCCE
  builds the commit and the native passes build the working tree, so the
  run stops at step 2, before anything is compiled, when the two differ.
  Pull, push, commit, stash or remove the edits first. A submodule that is
  not at its recorded commit counts as an edit, and `git submodule update`
  puts it back.

What happens:

1. For each configuration, a core is chosen whose class is `C0Pro` and
   whose microarchitecture, precision and correlation tracking are the
   configuration's. A default core or a core of your own serves alike. When
   none matches, one is created, named like `C0Pro-Athens-16-Autocorrelation`,
   with 2 GB of memory. SCCE builds Athens and Jupiter cores only, so every
   other representation type is skipped with a logged reason, like a
   configuration the SDK ships no library for.
2. The application is built once per core with `signaloid-cli-build.sh`.
3. Each benchmarking variable is run once per configuration with
   `signaloid-cli-run.sh`, with its own command-line arguments plus `-j`,
   under which the application prints every output as a Ux string in JSON,
   keyed by variable name. The helper scripts leave their own records
   (`.signaloid-cli-latest-*.txt`, `builds/outputs/`, `results/outputs/`)
   in the application root. A task that ends in any state but `Completed`
   (for example `Stopped`, when the core runs out of memory) stops the run,
   naming the task and its state.
4. The Ux strings go into the tracing database through
   `results_database.record_inference`. It writes the `TracingTable` and
   `Emulator_Execution_Info` tables the distance analysis reads, and also a
   `Runs` registry (one row per task, naming the application, the core and
   the task) and the `OutputDeltas` table (one row per Dirac delta, for
   querying in SQL).
5. Step 10 does not run anything more. It records the tracing tasks' timing,
   one measurement per configuration, from the task record: In Application
   Time and Database Time (which the speedup divides the native time by)
   are both the task's `ProcessorTime`, End-to-End Time is its
   `ExecutionTimeInMilliseconds`, and Database Dyn. Inst. Count is its
   `DynamicInstructions`. The application's own `CPU time used` line is not
   used, because on SCCE it reads zero and some applications leave it out
   under `-j`. Intel PIN cannot instrument the UxHw runs on SCCE, so their
   PIN count is left empty. With `--measure-dynamic-instructions` only the
   native Monte Carlo binary is counted, and a warning says so at startup.

The `-O2` Ux-string verification of step 4 does not run on SCCE. The UxHw SDK
version is recorded as `SCCE` in the timings file and the report, since there
is no SDK release to read it from.

## Config Files

Due to the large number of command line arguments this tool contains, one can
create a configuration file with the `--config <file.yaml>` flag that stores the
information for data quality metrics like ground truth size etc. With the
presets in `configs/`, a standard run drops to:

```
signaloid-benchmarking --config configs/full-sweep.yaml \
    --path-to-application ~/Signaloid-Demo-Example
```

**Precedence** is `built-in defaults < config file < explicit CLI flags`. Config
keys are the argparse `dest` names (e.g. `representation_types`, `correlations`,
`use_binned_uxhw`), not the CLI flag spellings. Unknown keys (typos) are
rejected with a clear error rather than silently ignored.

Because the four sweep args (`representation_types`, `representation_sizes`,
`correlations`, `reporting_methods`) can be supplied via the file, they are no
longer `required` on the CLI — but at least one source must provide each, or the
run fails fast with a `ValueError`.

You can find these presets for the configuration YAML in `configs/`.

- `configs/quick.yaml`. A small, fast matrix for smoke tests. Its sizes are too
  small for results you report.
- `configs/athens.yaml`, `configs/jupiter.yaml` and `configs/europa.yaml`. The
  recommended settings for one representation type each. See
  [Recommended settings](#recommended-settings).
- `configs/full-sweep.yaml`. The same settings across all four representation
  types.

Sample schema (every key is optional and omitted keys fall back to defaults):

```yaml
# Benchmark matrix
representation_types: [Athens, Jupiter]
representation_sizes: [64, 128, 256, 512]
correlations: [Disabled, Autocorrelation]
reporting_methods: [Mean, Quantile-95]
distance_type: Wasserstein-1
use_binned_uxhw: true
use_clt: false
n_adversaries: 100  # dest name. The CLI flag is --num-adversaries
# plotting controls (all default off)
plot_distance_vs_asymptotic: false
plot_adversary_distances: false
plot_representative_mc: false
```

Use `--print-config` to resolve defaults + file + CLI, print the effective
configuration and the benchmark matrix that would be swept, then exit without
running:

```
python -m signaloid_benchmarking.automation --config configs/quick.yaml \
    --path-to-application ~/Signaloid-Demo-Example --print-config
```

### Recommended settings

For a high-quality benchmark, use the preset for the representation type you
benchmark, `configs/athens.yaml`, `configs/jupiter.yaml` or
`configs/europa.yaml`. Each sweeps sizes 16 to 512, or 8 to 32 for Jupiter,
with and without autocorrelation tracking, and reports Mean, Quantile-95 and
Quantile-99. They share these sizing settings.

| Key | Recommended | What it sets |
|---|---|---|
| `ground_truth_type` | `WeightedSamples` | How the reference is stored. Its Monte Carlo samples are collapsed into weighted samples, so every distance is measured against a smaller reference. |
| `ground_truth_size` | `2000000` | Monte Carlo samples drawn for the reference. Their sampling error sets a floor under every distance, so keep it well above the largest EMCC you want to resolve. Under `use_clt`, scalar adversary runs are also capped at this size. |
| `max_num_weighted_samples` | `500000` | Weighted samples the reference is collapsed into. More keeps it closer to the raw samples, and fewer makes it cheaper to compare against. |
| `adversary_mc_size` | `1000000` | Samples in the pool that distribution adversaries are drawn from. |
| `adversary_max_size_scalar` | `100000` | The largest scalar adversary size when EMCC is measured. A larger scalar EMCC cannot be resolved. |
| `n_adversaries` | `100` | Independent adversary runs per size. The Mean, Quantile-95 or Quantile-99 of their distances is compared with UxHw. With fewer than 100 runs, Quantile-99 is close to the largest distance. Run time grows in proportion. |
| `use_clt` | `true` | Predicts EMCC from the asymptotic distance distribution, instead of measuring it at every adversary size. It is much faster at these sizes. |
| `max_jupiter_size` | `32` | Jupiter only. Larger Jupiter sizes are not benchmarked, because they run out of memory. Keep it equal to the largest Jupiter size in `representation_sizes`. |

Keep these caveats in mind.

- These sizes are expensive. Check that a run works with `configs/quick.yaml`
  first.
- When `use_clt` is false, a distribution EMCC is only measured up to 50000,
  whatever `adversary_mc_size` is.
- The presets leave out paths, credentials and `-j`. Pass those on the command
  line.

### Plotting controls

Plotting is off by default, so a plain run no longer pays the plotting cost.
Three independent toggles enable the three plot calls in the equivalent-MC stage
and each is a `BooleanOptionalAction`, so it accepts both the positive and
`--no-...` forms on the CLI and the matching boolean key in a config file:

- `--plot-distance-vs-asymptotic` — empirical equivalent-MC distance
  distribution vs its asymptotic prediction (Brownian-bridge / half-normal).
- `--plot-adversary-distances` — adversary-distance plots (empty under
  `--use-clt`).
- `--plot-representative-mc` — a representative equivalent-MC run matching the
  UxHw–ground-truth distance (distribution outputs only).

## Pipeline Overview

The tool executes the following steps in order (shown as `[Step N/15]` in the
output):

1. **Machine Info** — Detects CPU model and core count via `lscpu`.
2. **Application Info** — Reads `signaloid.yaml`, resolves benchmarking
   variables, compiles the native MC executable, and sets up output paths.
3. **Ground Truth Database** — Generates the reference distribution database
   using (in priority order): analytic formula, then native MC execution. A
   native build (a `Makefile` `local-build` target or `src/config.mk` with
   `SOURCES`) or `--has-analytic-ground-truth` is required.
4. **UxHw Tracing Database** — Runs the application through the UxHw tracing
   pipeline to produce UxHw distributional outputs for each representation
   type/size/correlation combination. With `--run-uxhw-scce` the application
   is built and run on SCCE instead, see
   [Running the UxHw passes on SCCE](#running-the-uxhw-passes-on-scce), and
   the rest of this step does not apply. If a tracing database already exists, the
   script will prompt before overwriting. The tracing build uses `-O0`, so the
   `addDistValueTrace` `file:line` directives resolve against unoptimised debug
   info. Optimisation could change the traced values. To guard against that,
   each config is also built at `-O2` and its Ux strings are checked
   (byte-for-byte) against the `-O0` ones. That build is compiled *without*
   tracing, because the SDK forces `-O0` whenever tracing is on, so it has no
   tracing database. Both sides are therefore read from the runs' stdout, where
   printing an uncertain value emits its full Ux string. Any difference is
   reported as a warning and does not stop the run. A config in which neither
   build printed a Ux string is reported as a warning too, so a run that
   verified nothing cannot be mistaken for a passing one. If a config's `-O2` build or run fails, that config is skipped and
   reported as failing in an `-O2 ux-string verification FAILED` summary (the
   run still continues). Set `TRACING_VERIFY_OPTFLAGS` to compare against a
   different level.
5. **Asymptotic Distance Distributions** — Computes, per variable, the
   distribution of `sqrt(N) * distance(MC(N), ground truth)` in the limit of
   large `N`. For a distribution output under `Wasserstein-1` this is the
   Brownian-bridge limit of del Barrio, Giné and Matrán (1999), simulated on
   the knots of the ground truth's own CDF, with the mean taken from its
   exact formula. Its integrability condition cannot be checked on a finite
   sample, so two tail diagnostics of the ground truth are recorded and a
   warning is printed when they fail (see `CltAssumptions` in `config.py`).
   No limit exists for `Wasserstein-2` under realistic tails, so none is
   computed. For a scalar output the limit is a folded normal fitted to
   repeated native runs, with an Anderson-Darling normality check. Failed
   checks are repeated in the per-variable Markdown report and in the run
   summary.
6. **UxHw Distances** — Computes Wasserstein distances between each UxHw
   configuration and the ground truth.
7. **EMCC Predictions** — Predicts equivalent Monte Carlo sample counts from
   asymptotic distances.
8. **Adversary Database** — Generates adversarial Monte Carlo databases for EMCC
   validation.
9. **Equivalent Monte Carlo** — Computes the true EMCC by comparing adversary MC
   distances to UxHw distances.
10. **UxHw Timings**. Measures execution time for each UxHw configuration. It
    also measures dynamic instruction counts when Intel PIN is enabled. With
    `--run-uxhw-scce` it records the timing of the step 4 tasks instead.
11. **Native MC Timings** — Measures execution time for native MC at each EMCC
    sample size.
12. **Load Measurements** — Loads all measurement data from the timing file.
13. **Load Timing Data** — Loads equivalent MC timing data from native
    executions.
14. **Compute Results** — Merges all timing and EMCC data into final dataframes.
15. **Write Outputs** — Writes results to Markdown unconditionally, and to a
    Google Sheets spreadsheet when `--write-sheets` is passed.
    The Sheets upload
    is opt-in (default behaviour prints a skip message). When opted in: per
    variable, a "best-of triptych" of three plots — the adversary MC at the
    selected EMCC count, the UxHw distribution for the best UxHw configuration,
    and the ground truth — is uploaded to Google Drive and linked from the
    sheet. The "best" configuration is the one with the maximum speedup against
    the native MC baseline, and the selection is printed to the terminal during
    upload. All other generated plots remain in `results/plots/` and are not
    uploaded. Credentials come from `--google-credentials` or
    `GOOGLE_APPLICATION_CREDENTIALS`, and the target Drive folder + Sheets
    template from `UXHW_SHEETS_DRIVE_FOLDER_ID` + `UXHW_SHEETS_TEMPLATE_ID`.
    Passing `--write-sheets` without all of these resolving raises
    `RuntimeError` at startup, before step 1 runs.

## Application Configuration

The target application must contain a `signaloid.yaml` file at its root. The
file defines which variables to trace and benchmark.

### Required Fields

```yaml
TraceVariables:
  - Expression: "variableName"
    File: "main.c"
    LineNumber: "42"

BenchmarkingAllOutputs:
  - CommandLineArguments: "-S 2"

BenchmarkingVariables:
  - VariableName: "variableName"
    VariableDescription: "Description of the output"
    OutputObject: "Distribution"          # or "Scalar"
    CommandLineArguments: "-S 0"
```

- **`TraceVariables`**: Lists the C expressions to trace, with source file and
  line number.
- **`BenchmarkingAllOutputs`**: The command-line arguments that produce all
  outputs simultaneously (used for tracing runs).
- **`BenchmarkingVariables`**: Maps each traced variable to a human-readable
  description, its output type (`Distribution` or `Scalar`), and the
  command-line arguments to isolate that variable.

Array expressions such as `outputVariables[0:5]` in `TraceVariables` are
automatically expanded into individual entries (`outputVariables[0]`,
`outputVariables[1]`, ..., `outputVariables[5]`).

### Optional per-variable fields

Each `BenchmarkingVariables` entry may also declare four fields. Each one falls
back to the matching run-wide flag when absent, so an entry that declares none
behaves exactly as before.

```yaml
BenchmarkingVariables:
  - VariableName: "outputVariables[0]"
    VariableDescription: "First Variable"
    OutputObject: "Distribution"
    CommandLineArguments: "-S 0"
    RepresentationTypes: [Athens]            # instead of -u
    RepresentationSizes: [8, 16, 32]         # instead of -s
    CorrelationTracking: [Autocorrelation]   # instead of -c
    AnalyticGroundTruth: ground-truth.py     # instead of --path-to-ground-truth-file
```

- A declared field replaces the run-wide value for that variable. It is not
  intersected with it, so the variable is measured on exactly what its entry
  lists. The values are checked like the run-wide flags, and a bad one stops
  the run before any build, naming the variable.
- `AnalyticGroundTruth` names a script relative to the application root.
  Declaring it gives that variable an exact reference, stored as weighted
  samples whatever `--ground-truth-type` says. A variable that declares none
  uses the run-wide script when `--has-analytic-ground-truth` is set, and a
  native Monte Carlo reference otherwise. Each distinct script runs once, for
  the variables that name it, so one script may write rows for several
  outputs.
- Precedence is the per-variable field, then the command line or `--config`
  file, then the built-in default. So a per-variable field wins even over an
  explicit command-line flag, which is the reverse of the usual rule that
  explicit flags win. This is deliberate. The field records a fact about one
  output, and the flag describes the run.
- `--print-config` shows only the run-wide matrix, because `signaloid.yaml` is
  read once the pipeline starts.
- `MinimumCore` is not read by the benchmarking tool.

### Configurations that cannot run

A requested combination is skipped, and logged before anything is built, when
it cannot run. Two things decide that.

- The SDK ships no runtime library for it. The tool asks the SDK's own build
  makefile which `libuxhw_runtime_*.a` the combination would link, so the
  answer follows the SDK's layout and this machine's CPU. In SDK 6.6.0 neither
  Jupiter nor Europa ships an autocorrelation library, so
  `-u Athens Europa -c Disabled Autocorrelation` measures Europa with
  `Disabled` only.
- A Jupiter size above `--max-jupiter-size` is skipped as a memory guard.

The measurement check does not expect a skipped combination. A variable left
with nothing to run stops the run before any build, naming the variable.

A combination is also skipped when its UxHw run dies, for example when the
kernel's out-of-memory killer stops it, which is common for large Jupiter
sizes. The tracing and timing steps record each run that died with how it
ended, a signal or an exit code, and go on with the rest. The combination is
then skipped for that variable from the next step on, and logged with its
reason like the skips above. One that died while being timed keeps the
distance and equivalent Monte Carlo count its tracing run gave, with no
timing. A variable whose every run died is reported with no UxHw rows. The
run stops only when no variable has a combination left.

Each variable's markdown report lists its skipped combinations with the
reason for each.

### Source convention
The tool benchmarks one output at a time, so by convention `main.c`:
- writes each output into an array (e.g. `double outputVariables[N];`), and
- accepts a `-S <index>` command-line argument selecting which output to
  compute/emit.

`signaloid.yaml` ties these together: `TraceVariables` points
`File`/`LineNumber`/`Expression` at the array (`outputVariables[0:N]`), and each
`BenchmarkingVariables` entry pairs a `VariableName` (`outputVariables[i]`) with
the `CommandLineArguments` (`-S i`) that isolate it. The program should also
print a `CPU time used: <seconds> seconds` line to stdout — the timing layer
parses it for the reference timing.

## Bash Timing Scripts

The `src/signaloid_benchmarking/benchmark_timing/` directory contains the
underlying bash scripts used by the pipeline:

- **`get-timings.sh`**: The main timing and compilation driver. It is sourced
  (not executed) by the Python tool with pre-set environment variables. It
  handles UxHw compilation, native MC benchmarking, UxHw tracing, and timing
  collection (the UxHw cores are compiled and timed via UxHw. The dynamic
  instruction count comes from Intel PIN when PIN is enabled, and is left
  unset otherwise). Compilation warnings are redirected
  to log files (`uxhw-build.log` and `native-mc-build.log` in the `logs/`
  directory). Source files are discovered recursively (excluding `build/`
  directories), and C++ files (`.cc`, `.cpp`) are automatically included when
  present.
- **`get-timing-template.sh`**: A standalone template showing the required
  environment variables and how to source `get-timings.sh` directly from the
  command line.

The UxHw `.m` config files that drive the reference / Monte Carlo / tracing
passes are generated inline by the `write_emulator_config` shell function
(consumed by the UxHw `opt` transform via `--m-config-file`). There is no
separate template file.

## Output Files

Output files are organized into `results/` and `logs/` directories in the
current working directory:

```
results/          # Final outputs
├── timings.json
├── *.db          (ground truth, adversary, tracing)
├── *.csv         (output_data, uxhw_distances, asymptotic_distances)
├── *.md          (markdown reports)
└── plots/
```

| File | Description |
|---|---|
| `output_data.csv` | Full EMCC results with distances, timing data, and speedups for every variable and configuration. |
| `uxhw_distances.csv` | UxHw distance data (Wasserstein distances between each configuration and ground truth). |
| `asymptotic_distances.csv` | Asymptotic distance distribution parameters. Mean, quantiles and mean quantile per variable, then the normality verdict, scale and location of the fitted normal for scalars, and the tail index and tail share of the CLT condition for distributions. |
| `<description>.md` | Per-variable Markdown summary tables, then any CLT warnings, then the variable's skipped configurations with the reason for each. `<description>` is the variable's `VariableDescription` in lower case, with each run of characters other than letters, digits, `_` and `.` turned into `-`, then with leading and trailing `-` and `.` removed. The same name starts its plot and `-asymptotic.npy` file names, so two variables must not share it. |
| `<app-name>-<hash>-timings.json` | Canonical timing output as a single JSON document (in `results/`). Session-invariant fields (application identity, SDK versions, target UxHw repetition count) sit at the top level. A `runs` array holds per-run records, each with its own `timestamp`, `commandLineArguments`, `commandLineArgumentsHash`, and `measurements` array. The schema is defined by `TimingFormat` in `signaloid_benchmarking/config.py`. |
| `<app-name>-<hash>-timings.intermediate` | Transient flat file the bash timing script writes line-by-line. Python parses it into the JSON above and deletes it on success. It is kept on failure for debugging. |
| `*.db` | SQLite databases for ground truth, adversary, and tracing data (in `<application>/src/`). |

Example `*-timings.json` document:

```json
{
  "applicationName": "call-option",
  "applicationVersion": "a1b2c3d",
  "uxhwSdkVersion": "4.1.2",
  "uxhwTargetRepetitions": "20",
  "runs": [
    {
      "timestamp": "2026-04-15T14:22:10Z",
      "commandLineArguments": "-T 100 --strike 110",
      "commandLineArgumentsHash": "abc123",
      "measurements": [
        {
          "config": "Athens-16-Autocorrelation",
          "time": 1.23,
          "dbTime": 4.56,
          "e2eTime": 7.89,
          "dbDynInstCount": null,
          "pinDynInstCount": 2000.0
        }
      ]
    }
  ]
}
```

Notes on the schema:

- Missing numeric fields (e.g. `dbTime` on native runs) are serialised as
  `null`.
- `dbDynInstCount` is `null`. It held the emulated RISC-V instruction count,
  which only the removed emulator produced. On native builds the SDK's
  `EmulatedCPU_DynCnt` is not an instruction count, so it is not read, and the
  sheet's `Database Dyn. Inst. Count` column is left empty.
- `pinDynInstCount` is `null` when the run had no Intel PIN kit configured.
  Intel PIN is optional and off by default, so this is the common case.
- On a run made with `--run-uxhw-scce`, `commandLineArguments` is the
  variable's own arguments, as on every run; the `-j` JSON flag the tasks
  are given is not recorded, so an application argument that happens to be
  `-j` is matched correctly. The native pass of such a run states no
  `uxhwSdkVersion` or `uxhwTargetRepetitions`, and the UxHw runs' values
  (`SCCE` and `1`) are lifted to the top level.
- `uxhwTargetRepetitions` is the UxHw-loop target at session start. The actual
  rep count used per measurement can differ — native-MC rows use
  `NATIVE_MC_REPETITION` (dynamically computed per precision), and the UxHw
  `REPETITION` is rescaled per-testcase from a warmup run inside
  `run_uxhw_benchmarks`. Treat this field as the configured session target
  rather than a per-run ground truth.


### Error Logs

On failure, the pipeline writes a detailed traceback to a timestamped log file:
```
logs/benchmarking_automation_error_<timestamp>.log
```
This file contains the full Python traceback and the command-line arguments
used. Additionally:
- Bash timing script stderr is captured to `logs/timing_script_stderr.log`.
- UxHw compilation warnings are logged to `logs/uxhw-build.log`.
- Native MC compilation warnings are logged to `logs/native-mc-build.log`.
- Per-execution errors can be found in `logs/exec.stderr` and `logs/opt.err`.

## Common Issues

### Native Compilation Fails
- **Missing GSL**: Ensure `libgsl-dev` (or equivalent) is installed and that
  `/opt/local/lib` and `/opt/local/include` are valid paths on your system, or
  that the application's `Makefile` handles library paths.
- **No compilation method found**: The application needs one of: a root-level
  `Makefile` with a `local-build` target, or a `src/config.mk` with `SOURCES`
  defined.
- **Executable not found**: The native executable (`demo-native-mc`) should be
  placed at the application root by `make local-build` or built from
  `config.mk`.
- **Fallback**: If the timing script's ad-hoc compilation fails, it will look
  for a pre-built `demo-native-mc` in both `src/` and the application root
  (where `make local-build` places it).

### Timing Script Fails
- The error message will include captured stderr and point to `exec.stderr` and
  `opt.err` in the `logs/` directory. Check these files for UxHw compilation or
  runtime errors.
- If the tracing database already exists, the script will prompt `Do you want to
  continue execution? (y/n)`. Answer `y` to overwrite or `n` to abort.

### Timing Data Mismatch
- The intermediate timing file is automatically truncated at the start of each
  pipeline run, so stale data from previous runs should not cause issues. If
  problems persist, manually delete the `<app-name>-<hash>-timings.intermediate`
  and `<app-name>-<hash>-timings.json` files in the `results/` directory and
  re-run.

### EMCC Data Not Found
- If the pipeline is interrupted after tracing but before EMCC computation,
  re-running will attempt to load `results/output_data.csv`. If this file does
  not exist or is incomplete, delete it and re-run from the beginning.

### Analytic Ground Truth
- When using `--has-analytic-ground-truth`, you must also pass
  `--ground-truth-type WeightedSamples` and provide
  `--path-to-ground-truth-file`. The ground truth script must accept `<size>
  <output_csv_path>` as positional arguments.
- A per-variable `AnalyticGroundTruth` needs neither flag. Its script takes the
  same arguments and writes `name,value,weight` rows, where `name` is a
  `VariableName`. Rows for a variable that does not name that script are
  ignored.

### Parallelism
- The `-j` flag controls parallelism for native MC sample generation only. If
  `-j` exceeds the number of detected CPU cores, a warning is printed. UxHw
  executions are always sequential.

### Binned Distance Computation
- `--use-binned-uxhw` is only compatible with `--distance-type Wasserstein-1`.
  The tool will error if you try to combine it with `Wasserstein-2`.

### Input Files Not Found
- Input files from the application's `inputs/` directory are symlinked into the
  working directory. Files that already exist in the target directory (e.g.,
  `README.md`) are skipped with a warning to avoid overwriting repository files.
  If the demo application expects files at a relative path like
  `inputs/filename`, ensure the demo's default paths do not include the
  `inputs/` prefix, since the files are symlinked flat into the working
  directory.
