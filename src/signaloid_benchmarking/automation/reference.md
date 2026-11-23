# Benchmarking Automation Reference

This page covers the details of `signaloid-benchmarking`, the UxHw®
benchmarking tool. For a normal run, see the [guide](README.md).

## Command-line options

The tool needs `--path-to-application` and the four sweep options `-u`, `-s`,
`-d`, and `-r`. A configuration file can supply the sweep options instead.

| Option | Default | Description |
|---|---|---|
| `-u`, `--representation-types` | None | Representation types. One or more of `Athens`, `Rome`, `Jupiter`, `Europa`, and `Mercury`. `Atlas` is the earlier name of `Rome`. |
| `-s`, `--representation-sizes` | None | Representation sizes, as integers. |
| `-d`, `--dependency-tracking` | None | Dependency tracking. One or more of `None`, `Basic`, and `Advanced`. Athens and Rome take `None` and `Basic`. Jupiter, Europa, and Mercury take `Advanced` only. The earlier names `-c`, `--uncertainty-correlation_types`, `Disabled`, and `Autocorrelation` still work. `Disabled` means `None` for Athens and Rome, and `Advanced` for Jupiter, Europa, and Mercury. |
| `-r`, `--reporting-methods` | None | Reporting methods. One or more of `Mean`, `Quantile-95`, and `Quantile-99`. |
| `--path-to-application` | None | The application repository to benchmark. |
| `--path-to-uxhw-sdk` | `~/project-uxhw-sdk` | The UxHw SDK. Not read with `--run-uxhw-scce`. |
| `--run-uxhw-scce` | Off | Build and run the UxHw passes on SCCE. See [Signaloid Cloud Compute Engine mode](#signaloid-cloud-compute-engine-mode). |
| `--path-to-cli-helpers` | `~/signaloid-cli-helpers` | The `signaloid-cli` helper scripts and their credential files. Used with `--run-uxhw-scce`. |
| `--measure-dynamic-instructions` | Off | Count dynamic instructions with the Intel Pin kit at `PIN_ROOT`. Without a usable kit, this is an error. |
| `--demo-cli-args` | `""` | Extra arguments for both the native and the UxHw runs. |
| `--ground-truth-size` | `1000000` | Monte Carlo samples in the ground truth. |
| `--ground-truth-type` | `MonteCarlo` | `MonteCarlo` or `WeightedSamples`. |
| `--has-analytic-ground-truth` | Off | Use an analytic ground truth. Needs `--ground-truth-type WeightedSamples` and `--path-to-ground-truth-file`. |
| `--path-to-ground-truth-file` | None | The script that generates the analytic ground truth. See [Analytic ground truth](#analytic-ground-truth). |
| `--distance-type` | `Wasserstein-1` | `Wasserstein-1` or `Wasserstein-2`. An EMCC prediction only exists for `Wasserstein-1`. |
| `--use-binned-uxhw`, `--no-use-binned-uxhw` | On | Use the binned Wasserstein-1 distance for UxHw. `Wasserstein-1` only. |
| `--use-clt`, `--no-use-clt` | On | Predict the EMCC from the asymptotic distance distribution instead of measuring it with adversary runs. `Wasserstein-1` only. |
| `--adversary-mc-size` | `1000000` | Samples in the adversary pool. |
| `--adversary-max-size-scalar` | `100000` | Largest adversary size for scalar outputs. |
| `--num-adversaries` | `100` | Adversary runs at each size. |
| `--max-num-weighted-samples` | `1000000` | Weighted samples that the ground truth is collapsed into. |
| `--max-jupiter-size` | `32` | The tool skips larger Jupiter sizes. |
| `-j`, `--jobs`, `--num-parallel-workers` | Detected core count | Maximum parallel workers. See [Parallelism](#parallelism). |
| `--config` | None | A YAML configuration file. See [Configuration file keys](#configuration-file-keys). |
| `--print-config` | Off | Print the resolved settings and the benchmark matrix, then exit. |
| `--plot-distance-vs-asymptotic` | Off | Plot the measured EMCC distance distribution against its asymptotic prediction, a Brownian bridge or a half-normal. |
| `--plot-adversary-distances` | Off | Plot the adversary distances. Empty under `--use-clt`. |
| `--plot-representative-mc` | Off | Plot a Monte Carlo run whose distance matches the UxHw distance. Distribution outputs only. |
| `--google-credentials` | None | Service-account credentials for the Google Sheets upload. Falls back to `GOOGLE_APPLICATION_CREDENTIALS`. |
| `--write-sheets` | Off | Upload the results to Google Sheets at step 15. See [Spreadsheet upload](#spreadsheet-upload). |

Each plot option also has a `--no-...` form.

### Parallelism

`-j` sets the maximum number of workers for the native build, the Monte Carlo
sampling, and the EMCC stage. A value above the detected core count prints a
warning, and the tool uses the core count. UxHw runs always run one at a
time.

## Configuration file keys

Command-line flags override the configuration file, and the file overrides
the built-in defaults. The keys are the `argparse` destination names, so
`--num-adversaries` is `n_adversaries` and `-d` is `dependency_tracking`. The
earlier key `correlations` still works. The tool rejects unknown keys. Every
key is optional, but some source must supply each of the four sweep options,
or the run stops with a `ValueError`.

```yaml
representation_types: [Athens, Jupiter]
representation_sizes: [8, 16, 32]
dependency_tracking: [None, Basic, Advanced]
reporting_methods: [Mean, Quantile-95]
distance_type: Wasserstein-1
use_binned_uxhw: true
use_clt: false
n_adversaries: 100
plot_distance_vs_asymptotic: false
plot_adversary_distances: false
plot_representative_mc: false
```

The presets do not set paths, credentials, or `-j`. Pass those on the command
line.

## Per-variable fields

Each `BenchmarkingVariables` entry in `signaloid.yaml` can declare four more
fields. An absent field falls back to the run-wide setting.

```yaml
BenchmarkingVariables:
  - VariableName: "outputVariables[0]"
    VariableDescription: "First Variable"
    OutputObject: "Distribution"
    CommandLineArguments: "-S 0"
    RepresentationTypes: [Athens]            # instead of -u
    RepresentationSizes: [8, 16, 32]         # instead of -s
    DependencyTracking: [Basic]              # instead of -d
    AnalyticGroundTruth: ground-truth.py     # instead of --path-to-ground-truth-file
```

- `CorrelationTracking` is the earlier name of `DependencyTracking`, and
  still works.
- A declared field replaces the run-wide value for that variable. It is not
  intersected with it. A bad value stops the run before any build, and the
  error names the variable.
- A per-variable field wins over the command line and the configuration
  file. This is deliberate. The field records a fact about one output, and
  the flags describe the run.
- `AnalyticGroundTruth` names a script relative to the application root. The
  tool stores its output as weighted samples, whatever `--ground-truth-type`
  says. A variable without a script uses the run-wide script when
  `--has-analytic-ground-truth` is set, and a native Monte Carlo ground truth
  otherwise. Each distinct script runs once.
- `--print-config` shows only the run-wide matrix, because the tool reads
  `signaloid.yaml` when the pipeline starts.
- The tool does not read `MinimumCore`.

### Analytic ground truth

A ground-truth script takes `<size> <output_csv_path>` as positional
arguments. A per-variable script writes `name,value,weight` rows, where
`name` is a `VariableName`. The tool ignores rows for a variable that does
not name that script.

## Configurations that cannot run

The tool skips a combination that cannot run, logs it before it builds
anything, and lists it with its reason in the variable's report.

- **The type does not take the dependency tracking mode.** Athens and Rome
  take `None` and `Basic`. Jupiter, Europa, and Mercury take `Advanced`
  only.
- **The SDK has no runtime library for it.** The tool asks the build
  makefile of the SDK which `libuxhw_runtime_*.a` the combination would
  link, so the answer depends on the SDK and the CPU. SDK 8.5.0-rc.4 has
  Mercury libraries for AVX-512 machines only, at sizes 64 and 256.
- **It is a Jupiter size above `--max-jupiter-size`.** This guards against
  running out of memory.
- **Its UxHw run failed.** For example, the kernel's out-of-memory killer
  stopped it, which is common for large Jupiter sizes. The tool records the
  signal or exit code and skips the combination from the next step on. A
  combination whose timing run failed keeps the distance and EMCC from its
  tracing run.

### Earlier names in the SDK

The UxHw SDK still takes the earlier names, so the tool translates at the
SDK. Dependency tracking was called correlation tracking. The SDK builds
`None` and `Advanced` as `Disabled`, and `Basic` as `Autocorrelation`. Rome
was called Atlas. On an SDK without a Rome runtime library, the tool builds
Rome as Atlas. The results still say Rome, and the database readers read
`Atlas` as Rome.


If a variable has nothing left to run before the first build, the run stops
and names the variable. After a failure, the run stops only when no variable
has a combination left.

## Pipeline steps

The output shows each step as `[Step N/15]`. The tool does not resume a
failed run. Each run starts again at step 1.

1. **Machine info.** Reads the CPU model and core count with `lscpu`.
2. **Application info.** Reads `signaloid.yaml`, builds the native Monte
   Carlo executable, and sets up the output paths.
3. **Ground truth.** Generates the ground-truth database from the analytic
   script or from native Monte Carlo runs.
4. **UxHw tracing.** Deletes any old tracing database, then runs each UxHw
   configuration with tracing and records its outputs. See
   [Optimization check](#optimization-check).
5. **Asymptotic distances.** Computes the limit distribution of
   `sqrt(N) * distance(MC(N), ground truth)` for each variable. See
   [Asymptotic distance distribution](#asymptotic-distance-distribution).
6. **UxHw distances.** Computes the distance from each UxHw configuration to
   the ground truth.
7. **EMCC predictions.** Predicts the EMCC from the asymptotic distances.
8. **Adversaries.** Generates the adversary Monte Carlo databases.
9. **Equivalent Monte Carlo.** Measures the EMCC by comparing the adversary
   distances with the UxHw distances.
10. **UxHw timings.** Times each UxHw configuration, and counts its
    instructions with Intel Pin when that is on.
11. **Native timings.** Times the native build at each EMCC sample size.
12. **Load measurements.** Reads the timing file.
13. **Load timing data.** Reads the native timing data.
14. **Compute results.** Merges the timing and EMCC data.
15. **Write outputs.** Writes the Markdown reports, and uploads to Google
    Sheets with `--write-sheets`.

Step 7 writes `results/output_data.csv`. The later steps keep the results in
memory, and read the file back only when they find none there. If a step
prints `Cannot continue without EMCC data`, the file is missing, and the
first error is at or before step 7.

Each run writes the same file names, so a second run in the same directory
overwrites the first. Step 4 deletes the old tracing database, and the timing
step starts a new intermediate file.

### Optimization check

The tracing build uses `-O0`, so that the `file:line` trace points resolve
against debug information. Optimization could change the traced values. So
the tool also builds each configuration at `-O2`, without tracing, and
compares the Ux strings that both builds print, byte for byte.

A difference, or a configuration where neither build printed a Ux string,
gives a warning and the run continues. If the `-O2` build or run fails, the
tool lists the configuration in an `-O2 ux-string verification FAILED`
summary. Set `TRACING_VERIFY_OPTFLAGS` to compare against another
optimization level.

### Asymptotic distance distribution

For a distribution output under `Wasserstein-1`, the limit is the Brownian
bridge of del Barrio, Giné, and Matrán (1999). The tool simulates it on the
knots of the ground truth's CDF, and takes the mean from its exact formula.
A finite sample cannot show whether the limit's integrability condition
holds, so the tool checks two tail diagnostics of the ground truth and warns
when they fail. See `CltAssumptions` in `config.py`. `Wasserstein-2` has no
limit under realistic tails, so the tool computes none.

For a scalar output, the limit is a folded normal fitted to repeated native
runs, with an Anderson-Darling normality check. Each variable's report
repeats any failed check.

## Signaloid Cloud Compute Engine mode

With `--run-uxhw-scce`, steps 4 and 10 run on SCCE through the helper
scripts. Everything else runs locally.

Before step 2, the tool checks the application against its origin. The clone
must be at the commit that its origin has for the current branch. It must
have no edits under `src/`, `submodules/`, or `inputs/`, or to
`signaloid.yaml` or the `Makefile`, other than files the tool writes. A
submodule away from its recorded commit counts as an edit. Run
`git submodule update` to restore it.

1. For each configuration, the tool picks a `C0Pro` core whose
   microarchitecture, precision, and correlation tracking, its name for
   dependency tracking, match. If none matches, it creates one with 2 GB of
   memory, named like `C0Pro-Athens-16-Basic`. SCCE has Athens and Jupiter
   cores only.
2. It builds the application once for each core with
   `signaloid-cli-build.sh`.
3. It runs each variable once for each configuration with
   `signaloid-cli-run.sh`, adding `-j` so that the application prints its
   outputs as JSON Ux strings. A task that ends in any state other than
   `Completed`, such as `Stopped` after running out of memory, stops the run.
   The helper scripts leave `.signaloid-cli-latest-*.txt`, `builds/outputs/`,
   and `results/outputs/` in the application root.
4. It writes the Ux strings to the tracing database with
   `results_database.record_inference`. That writes the `TracingTable`,
   `Emulator_Execution_Info`, `Runs`, and `OutputDeltas` tables.
5. Step 10 runs nothing more. It takes one timing for each configuration from
   the tracing task. In Application Time and Database Time are the task's
   `ProcessorTime`. End-to-End Time is its `ExecutionTimeInMilliseconds`.
   Database Dyn. Inst. Count is its `DynamicInstructions`.

Intel Pin cannot instrument SCCE runs, so `--measure-dynamic-instructions`
only counts the native build. The `-O2` check does not run. The SDK version
is recorded as `SCCE`.

## Output files

| File | Contents |
|---|---|
| `output_data.csv` | The distance, binned distance, and predicted EMCC of each configuration and reporting method, written at step 7. It has no measured EMCC, timings, or speedups. Those are in `<description>.md`. |
| `uxhw_distances.csv` | The distance under `--distance-type` from each configuration to the ground truth. |
| `asymptotic_distances.csv` | For each variable, the mean and quantiles of the asymptotic distance distribution. For scalars, also the normality verdict and the fitted normal. For distributions, the tail index and tail share. |
| `<description>.md` | Each variable's summary tables, CLT warnings, and skipped configurations. |
| `<app-name>-<hash>-timings.json` | All timing measurements. See [Timings file](#timings-file). |
| `<app-name>-<hash>-timings.intermediate` | The raw file that the Bash timing script writes. Deleted on success and kept on failure. |
| `*.db` | SQLite databases for the ground truth, adversary, and tracing data. |

`<description>` is the variable's `VariableDescription` in lowercase, with
each run of characters other than letters, digits, `_`, and `.` replaced by
`-`, and leading and trailing `-` and `.` removed. The variable's plot files
and `-asymptotic.npy` file use the same name, so two variables must not
share it.

### Databases

| File | Contents | Table |
|---|---|---|
| `uxhwExecutionStatistics-<commit>-<id>-tracing.db` | The UxHw outputs of every configuration. | `TracingTable` |
| `<app>-nativeMC-<commit>-<id>-ground-<size>.db` | The ground truth. | `MonteCarlo` or `WeightedSamples`, after `--ground-truth-type` |
| `<app>-nativeMC-<commit>-<id>-adv-<size>.db` | The adversary pool. | |

`<app>` is the application's name, `<commit>` its commit hash, and `<id>` a
hash of the `BenchmarkingAllOutputs` arguments. Without a native build,
`-nativeMC` is `-uxhwExecutionStatistics` instead. With an analytic ground
truth, `-analytic` follows it. The
[differential accuracy tools](../differential_accuracy/README.md) read these
files.

The plots go to `results/plots/`. With `--write-sheets`, the tool uploads
three plots for each variable to Google Drive. These are the adversary run at
the selected EMCC, the UxHw distribution of the configuration with the
highest speedup, and the ground truth.

### Timings file

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
          "config": "Athens-16-Basic",
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

`TimingFormat` in `config.py` defines the schema.

- A missing number is `null`. For example, `dbTime` is `null` on native runs.
- `dbDynInstCount` is always `null`. Only the removed emulator produced it.
- `pinDynInstCount` is `null` without `--measure-dynamic-instructions`.
- `uxhwTargetRepetitions` is the target for the session. The tool rescales
  the UxHw repetitions of each test from a warmup run, and native runs use
  `NATIVE_MC_REPETITION`, so the actual counts differ.
- With `--run-uxhw-scce`, `commandLineArguments` does not include the added
  `-j`, and the top-level SDK fields are `SCCE` and `1`.

### Logs

| File | Contents |
|---|---|
| `logs/benchmarking_automation_error_<timestamp>.log` | The stack trace and command-line arguments of a failed run. |
| `logs/timing_script_stderr.log` | Standard error of the Bash timing script. |
| `logs/uxhw-build.log` | UxHw build warnings. |
| `logs/native-mc-build.log` | Native build warnings. |
| `logs/exec.stderr`, `logs/opt.err` | Errors from each UxHw build and run. |


## Bash timing scripts

`src/signaloid_benchmarking/benchmark_timing/` holds the Bash scripts.

- **`get-timings.sh`** builds and times the UxHw and native runs, and runs
  the tracing. The Python tool sources it with its environment already set.
  It finds sources recursively, skipping `build/` directories, and includes
  `.cc` and `.cpp` files. It compiles C and C++ separately and links with
  `c++`. If the native build fails, it looks for a prebuilt
  `demo-native-mc` in `src/`, then in the application root.
- **`get-timing-template.sh`** shows the environment variables needed to
  source `get-timings.sh` by hand.

The `write_emulator_config` function writes the `.m` files that the UxHw
`opt` transform reads through `--m-config-file`.

The build templates are in the assets submodule, at
`src/signaloid_benchmarking/assets/`.

## Spreadsheet upload

The upload uses the
[public template](https://docs.google.com/spreadsheets/d/1MKPkwY_B_m5WH-coqIy19Otpg3IE5yRSKaKGw6wxzL8).
`ReportSheetTabs` and `MetadataRowLabels` in `config.py` must match its tab
names and metadata rows exactly. When the template changes, read the tab
names from its ODS export. The XLSX export cuts them to 31 characters.

The upload needs the `gspread`, `google-api-python-client`, and
`oauth2client` packages from the `sheets` extra. If any requirement is
missing, the tool raises `RuntimeError` at startup and names it.
