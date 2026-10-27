# Differential accuracy

Tools that compare how close two UxHw runs land to the same ground truth.

## `distance_uxhw_vs_mc`

`distance_uxhw_vs_mc` answers one question. **Did the under-test
configuration get closer to the ground truth than the baseline did, or
further away?** It never judges a configuration on its own. Every number in
the report compares a *baseline* run and an *under-test* run, both measured
against the same ground truth.

Source is [distance_uxhw_vs_mc.py](distance_uxhw_vs_mc.py).

### How it works

The tool reads the variables to measure from a `signaloid.yaml`. For every
variable, and for every UxHw configuration (representation type, size and
correlation tracking) found in the databases, the tool does four things.

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

### Running it

Installing the package puts `distance_uxhw_vs_mc` on the path. From a
checkout, run it through Poetry.

```bash
poetry run distance_uxhw_vs_mc --help
```

It also runs as a module.

```bash
python -m signaloid_benchmarking.differential_accuracy.distance_uxhw_vs_mc --help
```

### Example

This compares the Atlas-16 results of two builds of the same application.
The old build is the baseline and the new build is under test.

```bash
distance_uxhw_vs_mc \
    --ground-truth-database ground-truth.db \
    --ground-truth-table MonteCarlo \
    --uxhw-baseline-database old-build.db \
    --uxhw-under-test-database new-build.db \
    --uxhw-tables Evaluation \
    --baseline-uncertainty-representation-type Atlas \
    --under-test-uncertainty-representation-type Atlas \
    --uncertainty-representation-sizes 16 \
    --yaml-signaloid signaloid.yaml \
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

Valid representation types are `Athens`, `Jupiter`, `Atlas` and `Europa`.
All three options take several space-separated values. The size filter
applies to both databases.

The two type filters are separate on purpose. They let you compare *across*
representations, for example an `Athens` baseline against an `Atlas`
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

If the yaml has no `BenchmarkingVariables` key, the tool measures every
`TraceVariables` entry instead. It expands range expressions, so
`outputVariables[0:3]` becomes `outputVariables[0]` to `outputVariables[3]`.
The end index is **inclusive**. Every variable found this way is treated as
a `Distribution`. New yaml files should declare `BenchmarkingVariables`.

### Database expectations

Both UxHw databases must contain the tables named by `--uxhw-tables`. They
must also describe their configurations in an `Emulator_Execution_Info`
table.

The tool only understands the keywords that 5.0 and later SDKs write, such
as `Athens` and `Autocorrelation`. Databases from older SDKs use other
names, such as `TTR` and `OFF`. The tool does not convert them. They fail
with `ValueError: No results from Execution info query.`

### Reading the report

The report is a GitHub-flavoured markdown table. It has one row per
variable and configuration.

| Benchmark | Expression | Baseline Conf | Under-test Conf | WD Baseline | WD Under-test | WD Diff | Relative Diff Percentage | Scoring |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cli-e2e | outputDistribution | Athens-16 | Atlas-16 | 0.233545 | 0.0600875 | -0.173458 | -74.2716 | 🟢 |

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
negative `WD Diff` means the under-test run landed closer to the ground
truth than the baseline did.

The configuration strings have the form `Representation-Size`. Enabled
correlation tracking adds `-Autocorrelation`, as in `Athens-16` or
`Atlas-64-Autocorrelation`. Disabled correlation tracking is the default and
adds nothing. So `Atlas-64` means correlation tracking was off. This string
matches the `UXHW_CONF` key in the EMCC CSV byte for byte. That lets you join
timing data against these accuracy results.

The report keeps every row, including rows where nothing changed. If the
yaml declares no variables, the report holds the line
`No configurations compared for <benchmark>.` instead of a table.

#### Scoring legend

| Score | Condition |
| --- | --- |
| 🟢 | Improvement. The relative difference is negative, or a NaN or infinite baseline became finite. |
| ⚪ | No change. The relative difference is zero, within `1e-14`. |
| 🟡 | Regression of at most `1e-6`%. |
| 🟠 | Regression of at most `1e-1`%. |
| 🔴 | Regression larger than `1e-1`%, or an under-test distance that is NaN. |

Some edge cases are worth knowing.

- **A baseline distance of exactly zero** means the baseline matched the
  ground truth exactly. There is no scale to measure a relative change
  against. If the under-test distance is also zero, the score is ⚪. If it
  is non-zero, the relative difference is infinite and the score is 🔴. If
  it is NaN, the score is 🔴.
- **A NaN or infinite baseline that becomes finite** always scores 🟢.
  Going from unusable to measurable is an improvement at any size.
- **Both distances NaN, or both infinite**, scores 🔴, even when the two
  runs are identical. This can happen for `Scalar` variables whose ground
  truth is zero.

### Pairing configurations

The tool sorts the baseline and under-test configurations by size, then by
correlation tracking, then by configuration string to break ties. It then
pairs them in order. The order the databases return rows in does not
matter. Each pair must have the same size and correlation tracking. The
representation type may differ, since that is what the two type filters
are for. For example, a baseline of `Athens-16` and `Atlas-32` pairs with
an under-test of `Atlas-32` and `Jupiter-16` as `Athens-16` with
`Jupiter-16` and `Atlas-32` with `Atlas-32`.

### Troubleshooting

The tool exits with status 255 and writes no report in both of these cases.

#### `UxHw baseline and UxHw under-test differ in configurations. Exiting.`

The two databases gave different numbers of matching configurations, so
they cannot be paired. Usually one side is missing a run. Check that both
databases contain the same sizes and correlation settings for the tables
you named.

#### `Cannot pair UxHw baseline Athens-64 with UxHw under-test Athens-32. Exiting.`

Both databases gave the same number of configurations, but they do not
match. In this example the baseline has an Athens-64 run and the under-test
database has an Athens-32 run instead. Check which sizes each database
holds.

#### `ValueError` from the loaders

The loaders raise `ValueError` when a variable is missing from a table, or
when no configuration matches the filters. Check the variable names in the
yaml and the filters on the command line. Databases from SDKs older than
5.0 also fail this way, as described above.

## `ks_convergence_gate`

`ks_convergence_gate` answers a question that comes before the distance
comparison. **Does either run get close enough to the ground truth for the
comparison to mean anything?** When neither run converges, a change in the
distance is noise. The gate finds those configurations and drops their rows
from a `distance_uxhw_vs_mc` report.

Source is [ks_convergence_gate.py](ks_convergence_gate.py).

### How the gate works

For every `Distribution` variable, the gate runs a bootstrapped
Kolmogorov-Smirnov test on each database. It tests each correlation-tracking
mode once, at the largest size the database holds. Whether a configuration
converges depends on its largest size, so the smaller sizes are skipped.

- Each bootstrap draws N/2 values from a representation of size N.
- There are 10 bootstraps by default. Fisher's method combines their
  p-values.
- A side passes when the combined p-value is at least the significance
  level, 0.05 by default.
- The random generator is seeded, so the verdicts repeat from run to run.

The level is fixed. A level that shrinks with size, such as 1/log2(N),
rejects a perfect representation at size 2 every time and at size 32 one
run in four.

The gate computes the statistic itself, where the library's bootstrapped
test calls scipy. scipy's one-sample test compares the empirical CDF just
before each sample point with the ground-truth CDF at the point. That is
right for a continuous ground truth. On a ground truth with an atom, such as
a payoff that is zero most of the time, it counts the atom's mass as a gap
and fails even an exact representation. The gate compares against the
ground-truth CDF's left limit there instead. On a continuous ground truth
the two statistics agree.

The decision for a configuration combines both sides.

| Decision | When |
| --- | --- |
| ✅ measured | Both sides pass. |
| ⚠️ measured | One side passes. The note says which. The row stays, since that difference is itself a regression or an improvement. |
| ⛔ not measured | Neither side passes, or the ground truth cannot be tested. The rows of every size of the configuration are dropped. |

An expression traced at several places gives several values with the same
configuration. Each occurrence is decided and filtered on its own.

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

Installing the package puts `ks_convergence_gate` on the path. It also runs
as a module.

```bash
python -m signaloid_benchmarking.differential_accuracy.ks_convergence_gate --help
```

The gate takes the same database, filter and yaml options as
`distance_uxhw_vs_mc`, so both tools can take the same argument list. This
runs the comparison and then gates it.

```bash
ARGUMENTS=(
    --ground-truth-database ground-truth.db
    --uxhw-baseline-database old-build.db
    --uxhw-under-test-database new-build.db
    --uxhw-tables Evaluation
    --baseline-uncertainty-representation-type Atlas
    --under-test-uncertainty-representation-type Atlas
    --uncertainty-representation-sizes 16 32 64
    --yaml-signaloid signaloid.yaml
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
| `Configuration` | Representation and correlation mode, such as `Athens-Disabled`. |
| `Occurrence` | Which value this is when one expression name is traced at several places. It counts from 0, in the order `distance_uxhw_vs_mc` pairs them. |
| `Size` | The largest size tested on each side. |
| `Baseline` and `Under-test` | The verdict and combined p-value, or `n/a` when that side has no rows. |
| `Decision` and `Note` | See the table above. |

If every row of a report table is dropped, the filtered report holds a note
in place of the table. Like `distance_uxhw_vs_mc`, the gate only understands
the keywords that 5.0 and later SDKs write.
