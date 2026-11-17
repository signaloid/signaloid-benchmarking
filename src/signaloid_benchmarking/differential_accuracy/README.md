# Differential Accuracy

Tools for regression-testing a UxHw® SDK build. They compare how close two
UxHw runs of the same application get to the same ground truth. The baseline
run uses a known-good SDK, and the under-test run uses the new one. Both
commands come with the package and run on Linux and macOS. See
[Installation](../../../README.md#installation). Run either with `--help`
for every option.

## Regression-testing an SDK build

1. Run `signaloid-benchmarking` on the same application with the same
   settings twice, once with each SDK. Use a separate directory for each
   run, because a run overwrites the results in its directory.
   ```bash
   mkdir ~/run-baseline && cd ~/run-baseline
   signaloid-benchmarking --path-to-uxhw-sdk /opt/sdk-baseline <settings>
   mkdir ~/run-under-test && cd ~/run-under-test
   signaloid-benchmarking --path-to-uxhw-sdk /opt/sdk-under-test <settings>
   ```
2. Compare the two tracing databases against the baseline's ground truth
   with `distance_uxhw_vs_mc`. See [Example](#example).
3. Optionally, drop the configurations that neither run gets close enough
   to the ground truth to compare, with `ks_convergence_gate`. See
   [Running the gate](#running-the-gate).

The [reference](../automation/reference.md#databases) lists the database
files that a run writes, and their tables.

## `distance_uxhw_vs_mc`

`distance_uxhw_vs_mc` answers one question. **Did the under-test
configuration get closer to the ground truth than the baseline did, or
further away?** It never judges a configuration on its own. Every number in
the report compares a *baseline* run and an *under-test* run, both measured
against the same ground truth. Despite its name, it compares two UxHw runs
with each other, not UxHw with Monte Carlo.

Source is [distance_uxhw_vs_mc.py](distance_uxhw_vs_mc.py).

### How it works

The tool reads the variables to measure from a `signaloid.yaml`. For every
variable, and for every UxHw configuration (representation type, size and
dependency tracking) found in the databases, the tool does four things.

1. It loads the ground truth from the ground truth database.
2. It loads the matching baseline and under-test UxHw distributions.
3. It computes the distance from each of them to the ground truth.
4. It reports the difference between those two distances, in absolute and
   relative terms, with a traffic-light score.

The distance depends on the variable's `OutputObject`.

| `OutputObject` | Distance | Binned columns |
| --- | --- | --- |
| `Distribution` | Wasserstein-1, plus a binned Wasserstein-1 | Separate binned figure |
| `Scalar` | Relative error | Copied from the non-binned figure |

For `Scalar` variables the binned columns repeat the non-binned ones. They
only keep the report table the same shape.

### Example

This compares the Athens results of the two runs from
[Regression-testing an SDK build](#regression-testing-an-sdk-build). The runs
used the recommended settings, so the ground truth table is
`WeightedSamples`. With `--ground-truth-type MonteCarlo`, it is `MonteCarlo`.

```bash
BASELINE=~/run-baseline/results
UNDER_TEST=~/run-under-test/results
distance_uxhw_vs_mc \
    --ground-truth-database $BASELINE/*-ground-*.db \
    --ground-truth-table WeightedSamples \
    --uxhw-baseline-database $BASELINE/uxhwExecutionStatistics-*-tracing.db \
    --uxhw-under-test-database $UNDER_TEST/uxhwExecutionStatistics-*-tracing.db \
    --uxhw-tables TracingTable \
    --baseline-uncertainty-representation-type Athens \
    --under-test-uncertainty-representation-type Athens \
    --uncertainty-representation-sizes 16 32 64 128 256 512 \
    --yaml-signaloid ~/my-application/signaloid.yaml \
    --benchmark my-application \
    -o report.md
```

You choose which side is the baseline. That choice sets the sign of every
relative figure. The report reads as *"the under-test configuration is N%
better or worse than the baseline"*.

### Options

#### Ground truth

| Option | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--ground-truth-database` | yes | none | Database holding the ground truth. |
| `--ground-truth-table` | no | `MonteCarlo` | Table in that database to read. |

The table name also picks how the table is read. Only a table named exactly
`MonteCarlo` is read as Monte Carlo samples. Any other name is read as
weighted samples. A table of raw samples read under the wrong name gives
meaningless distances, not an error.

#### UxHw data

| Option | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--uxhw-baseline-database` | yes | none | Database holding the baseline UxHw results. |
| `--uxhw-under-test-database` | yes | none | Database holding the under-test UxHw results. |
| `--uxhw-tables` | yes | none | One or more tables to read from *both* databases. |

#### Configuration filters

| Option | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--baseline-uncertainty-representation-type` | yes | none | Representation types to match in the baseline data. |
| `--under-test-uncertainty-representation-type` | yes | none | Representation types to match in the under-test data. |
| `-s`, `--uncertainty-representation-sizes` | yes | none | Representation sizes to match, as integers. |

Valid representation types are `Athens`, `Rome`, `Jupiter`, `Europa`, and
`Mercury`. `Atlas`, the earlier name of Rome, also works. A `Rome` filter
matches the databases of SDK releases from before and after the rename. All
three options take one or more space-separated values. The size filter
applies to both databases.

The two type filters are separate on purpose. They let you compare *across*
representations, for example an `Athens` baseline against a `Rome`
candidate.

#### Variables and output

| Option | Required | Default | Meaning |
| --- | --- | --- | --- |
| `-y`, `--yaml-signaloid` | yes | none | The `signaloid.yaml` that declares the variables to measure. |
| `--benchmark` | no | none | Label written to the report's `Benchmark` column. |
| `-o`, `--output` | no | `output.md` | Where to write the markdown report. |

### The `signaloid.yaml` contract

The variables to measure come from the `BenchmarkingVariables` list. This
tool only reads `VariableName` and `OutputObject`. The rest of the
benchmarking pipeline uses the other keys.

```yaml
BenchmarkingVariables:
    - VariableName: outputVariables[0]
      VariableDescription: SmallTree top-event probability
      OutputObject: Distribution
      CommandLineArguments: -S 0
    - VariableName: outputVariables[1]
      VariableDescription: TwoTrain top-event probability
      OutputObject: Distribution
      CommandLineArguments: -S 1
```

#### Older `TraceVariables` files

If the YAML file has no `BenchmarkingVariables` key, the tool measures every
`TraceVariables` entry instead. It expands range expressions, so
`outputVariables[0:3]` becomes `outputVariables[0]` to `outputVariables[3]`.
The end index is **inclusive**. Every variable found this way is treated as
a `Distribution`. New YAML files should declare `BenchmarkingVariables`.

### Database expectations

Both UxHw databases must contain the tables named by `--uxhw-tables`. They
must also describe their configurations in an `Emulator_Execution_Info`
table.

The tool only understands the representation type names that SDK 5.0 and
later write, such as `Athens`. Databases from older SDK releases use other
names, such as `TTR`. The tool does not convert them. They fail with
`ValueError: No results from Execution info query.` The SDK records
dependency tracking under its earlier name, correlation tracking, and the
tool reads it as dependency tracking.

### Reading the report

The report is a GitHub-flavored Markdown table. It has one row per
variable and configuration.

| Benchmark | Expression | Baseline Conf | Under-test Conf | WD Baseline | WD Under-test | WD Diff | Relative Diff Percentage | Scoring |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cli-e2e | `outputDistribution` | Athens-16 | Rome-16 | 0.233545 | 0.0600875 | -0.173458 | -74.2716 | `🟢` |

The real table has five more columns. These are the `Binned WD` versions and
`Binned Scoring`. They are left out here to save width.

| Column | Meaning |
| --- | --- |
| `Benchmark` | Whatever was passed to `--benchmark`. |
| `Expression` | The `VariableName` being measured. |
| `Baseline Conf` and `Under-test Conf` | The paired configurations. The format is below. |
| `WD Baseline` | Distance from the baseline to the ground truth. |
| `WD Under-test` | Distance from the under-test run to the ground truth. |
| `WD Diff` | `WD Under-test` minus `WD Baseline`. Negative is an improvement. |
| `Relative Diff Percentage` | That difference as a percentage of `WD Baseline`. |
| `Scoring` | Traffic light for the relative difference. See below. |

**Lower is better throughout, and negative differences are good.** A
negative `WD Diff` means the under-test run is closer to the ground
truth than the baseline is.

The configuration strings have the form `Representation-Size`, followed by
the dependency tracking unless it is `None`. So `Athens-64` has no
dependency tracking, `Athens-64-Basic` has Basic, and `Jupiter-64-Advanced`
has Advanced. This string
matches the `UXHW_CONF` key in the EMCC CSV byte for byte. That lets you join
timing data against these accuracy results.

The report keeps every row, including rows where nothing changed. If the
YAML file declares no variables, the report holds the line
`No configurations compared for <benchmark>.` instead of a table.

#### Scoring legend

| Score | Condition |
| --- | --- |
| `🟢` | Improvement. The relative difference is negative, or a NaN or infinite baseline became finite. |
| `⚪` | No change. The relative difference is zero, within `1e-14`. |
| `🟡` | Regression of at most `1e-6`%. |
| `🟠` | Regression of at most `1e-1`%. |
| `🔴` | Regression larger than `1e-1`%, or an under-test distance that is NaN. |

Note the following edge cases.

- **A baseline distance of exactly zero** means the baseline matched the
  ground truth exactly. There is no scale to measure a relative change
  against. If the under-test distance is also zero, the score is `⚪`. If it
  is non-zero, the relative difference is infinite and the score is `🔴`. If
  it is NaN, the score is `🔴`.
- **A NaN or infinite baseline that becomes finite** always scores `🟢`.
  Going from unusable to measurable is an improvement at any size.
- **Both distances NaN, or both infinite**, scores `🔴`, even when the two
  runs are identical. This can happen for `Scalar` variables whose ground
  truth is zero.

### Pairing configurations

The tool sorts the baseline and under-test configurations by size, then by
dependency tracking, then by configuration string to break ties. It then
pairs them in order. The order the databases return rows in does not
matter. Each pair must have the same size and dependency tracking. The
representation type may differ, since that is what the two type filters
are for. For example, a baseline of `Athens-16` and `Rome-32` pairs with
an under-test of `Rome-32` and `Rome-16` as `Athens-16` with `Rome-16` and
`Rome-32` with `Rome-32`. Athens and Rome never pair with Jupiter, Europa,
or Mercury, because those only have `Advanced` dependency tracking.

### Troubleshooting

The tool exits with status 255 and writes no report in both of these cases.

#### `UxHw baseline and UxHw under-test differ in configurations. Exiting.`

The two databases gave different numbers of matching configurations, so
they cannot be paired. Usually one side is missing a run. Check that both
databases contain the same sizes and dependency tracking for the tables
you named.

#### `Cannot pair UxHw baseline Athens-64 with UxHw under-test Athens-32. Exiting.`

Both databases gave the same number of configurations, but they do not
match. In this example the baseline has an Athens-64 run and the under-test
database has an Athens-32 run instead. Check which sizes each database
holds.

#### `ValueError` from the loaders

The loaders raise `ValueError` when a variable is missing from a table, or
when no configuration matches the filters. Check the variable names in the
YAML file and the filters on the command line. Databases from SDK releases
older than 5.0 also fail this way, as described above.

## `ks_convergence_gate`

`ks_convergence_gate` answers a question that comes before the distance
comparison. **Does either run get close enough to the ground truth for the
comparison to mean anything?** When neither run converges, a change in the
distance is noise. The gate finds those configurations and drops their rows
from a `distance_uxhw_vs_mc` report.

Source is [ks_convergence_gate.py](ks_convergence_gate.py).

### How the gate works

For every `Distribution` variable, the gate runs a bootstrapped
Kolmogorov-Smirnov test on each database. It tests each dependency tracking
mode once, at the largest size the database holds. Whether a configuration
converges depends on its largest size, so the smaller sizes are skipped.

- Each bootstrap draws N/2 values from a representation of size N.
- There are 10 bootstraps by default. Fisher's method combines their
  p-values.
- A side passes when the combined p-value is at least the significance
  level, 0.05 by default.
- The random generator is seeded, so the verdicts repeat from run to run.

The level is fixed, because a level that shrinks with size, such as
1/log2(N), rejects even a perfect representation at small sizes.

The gate computes the statistic itself instead of calling SciPy. At an atom
in the ground truth, such as a payoff that is zero most of the time, the
SciPy test fails even an exact representation. The gate compares against the
left limit of the ground-truth CDF there. On a continuous ground truth, the
two statistics agree.

The decision for a configuration combines both sides.

| Decision | When |
| --- | --- |
| `✅ measured` | Both sides pass. |
| `⚠️ measured` | One side passes. The note says which. The row stays, since that difference is itself a regression or an improvement. |
| `⛔ not measured` | Neither side passes, or the ground truth cannot be tested. The rows of every size of the configuration are dropped. |

An expression traced at more than one place gives more than one value with
the same configuration. Each occurrence is decided and filtered on its own.

Some configurations are never measured, whatever the test says.

- A ground truth with zero spread is a point mass. Every representation of
  it is either exact or wrong by the same amount, so the distances mean
  nothing.
- A ground truth with mass at NaN or infinity has no CDF on the real line,
  and no finite distance to anything.

Zero-mass entries at NaN or infinity are parser placeholders, and both sides
ignore them. Mass there is never dropped, so a representation that misses it
cannot pass.

`Scalar` variables have a single ground-truth value, so there is no
distribution to test. They are never gated and always listed as measured.

### Running the gate

The gate takes the same database, filter, and YAML options as
`distance_uxhw_vs_mc`, so both tools can take the same argument list. This
runs the comparison and then gates it.

```bash
BASELINE=~/run-baseline/results
UNDER_TEST=~/run-under-test/results
ARGUMENTS=(
    --ground-truth-database $BASELINE/*-ground-*.db
    --ground-truth-table WeightedSamples
    --uxhw-baseline-database $BASELINE/uxhwExecutionStatistics-*-tracing.db
    --uxhw-under-test-database $UNDER_TEST/uxhwExecutionStatistics-*-tracing.db
    --uxhw-tables TracingTable
    --baseline-uncertainty-representation-type Athens
    --under-test-uncertainty-representation-type Athens
    --uncertainty-representation-sizes 16 32 64 128 256 512
    --yaml-signaloid ~/my-application/signaloid.yaml
    --benchmark my-application
)
distance_uxhw_vs_mc "${ARGUMENTS[@]}" -o report.md
ks_convergence_gate "${ARGUMENTS[@]}" \
    --comparison-report report.md \
    --output measured.md \
    --gate-output gate.md
```

`measured.md` is `report.md` without the rows that are not measured.
`gate.md` lists every configuration with its verdicts.

### Gate options

Besides the shared options, the gate takes these.

| Option | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--gate-output` | yes | none | Where to write the gate table. |
| `--comparison-report` | no | none | A `distance_uxhw_vs_mc` report to filter. |
| `-o`, `--output` | no | none | Where to write the filtered report. Give it together with `--comparison-report`. |
| `--significance-level` | no | `0.05` | Combined p-value below which a side fails. |
| `--number-of-bootstraps` | no | `10` | Bootstraps per test. |
| `--seed` | no | `0` | Seed of the random generator. |

### Reading the gate table

| Column | Meaning |
| --- | --- |
| `Configuration` | Representation and dependency tracking, such as `Athens-None` or `Jupiter-Advanced`. |
| `Occurrence` | Which value this is when one expression name is traced at more than one place. It counts from 0, in the order `distance_uxhw_vs_mc` pairs them. |
| `Size` | The largest size tested on each side. |
| `Baseline` and `Under-test` | The verdict and combined p-value, or `n/a` when that side has no rows. |
| `Decision` and `Note` | See the table above. |

If every row of a report table is dropped, the filtered report holds a note
in place of the table. Like `distance_uxhw_vs_mc`, the gate only understands
the keywords that SDK 5.0 and later write.
