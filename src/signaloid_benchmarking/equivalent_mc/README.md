# Equivalent Monte Carlo

This Python library computes the equivalent Monte Carlo count for each UxHw®
configuration. It compares the UxHw result against a ground-truth Monte Carlo
simulation and a series of adversary Monte Carlo simulations.

## Terms

### Equivalent Monte Carlo count

The library computes one equivalent Monte Carlo count (EMCC) for each
variable and each UxHw configuration. The EMCC is the smallest number of
Monte Carlo samples that gives a result at least as accurate as the UxHw
configuration, for the chosen distance metric and reporting method.

The library computes it in the following way.

1. For each adversary sample count, the library computes the distance from
   each adversary run to the ground truth.
2. The reporting method reduces those distances to one statistic. The
   statistic is the mean, the 95th percentile, or the 99th percentile.
3. The EMCC is the smallest adversary sample count whose statistic is less
   than the distance from the UxHw configuration to the ground truth.

For example, with the Wasserstein-1 metric and the `Mean` reporting method,
the EMCC is the smallest sample count at which the mean Wasserstein-1
distance of the adversary runs is less than the UxHw distance. If no
adversary sample count is good enough, the library prints a warning and sets
the EMCC to 1.

With `use_clt`, the library does not search the adversary runs. It uses the
count predicted from the asymptotic distance distribution instead.

### Other terms

- **Ground truth.** Samples that act as the reference for the correct result
  of the application under uncertainty. The samples can come from a large
  Monte Carlo simulation, or be weighted samples from an analytic formula or
  an empirical method.
- **Adversary.** A Monte Carlo run of a given sample count that the library
  compares with UxHw. The library draws the adversary runs from a pool of
  Monte Carlo samples.

## Usage

Most users get the EMCC from `signaloid-benchmarking`, which calls this
library at step 9. Call it yourself only to recompute the EMCC from the
databases of an earlier run.

### `load_data_and_compute_equivalent_mc`

`load_data_and_compute_equivalent_mc(args)` loads the ground-truth, UxHw, and
adversary distributions from their databases. It then computes the equivalent
Monte Carlo count for each benchmarking variable, prints the results, and
optionally writes plots. It takes a single `LoadDataComputeEquivalentMCArgs`
`TypedDict`.

The function only measures the configurations already listed in each
variable's `emcc_results.emcc_data`. That list needs one entry for each
configuration and reporting method, with its distances and predicted EMCC.
For a variable whose list is empty, the function measures nothing.
`compute_emcc_predictions` in `signaloid_benchmarking.automation.analysis`
fills the list from the `uxhw_distances.csv` and `asymptotic_distances.csv`
files of a run. [`equiv_mc_test.py`](./equiv_mc_test.py) shows the whole
sequence.

The following example sets a representative subset of the keys. The
database names are those of a pipeline run. See
[Databases](../automation/reference.md#databases).

```python
from signaloid_benchmarking import (
    LoadDataComputeEquivalentMCArgs,
    load_data_and_compute_equivalent_mc,
)

args: LoadDataComputeEquivalentMCArgs = {
    "benchmarking_variables": benchmarking_variables,   # list[BenchmarkingVariable]
    "ground_truth_database_path": "results/my-application-nativeMC-a1b2c3d-9f8e7d-ground-10000.db",
    "ground_truth_table_name": "MonteCarlo",
    "ground_truth_type": "MonteCarlo",
    "adversary_database_path": "results/my-application-nativeMC-a1b2c3d-9f8e7d-adv-10000.db",
    "adversary_table_name": "MonteCarlo",
    "adversary_size_step": 1,
    "adversary_size_min": 1,
    "adversary_size_max": None,
    "n_adversaries": 100,
    "uxhw_database_path": "results/uxhwExecutionStatistics-a1b2c3d-9f8e7d-tracing.db",
    "uxhw_table_names": ["TracingTable"],
    "uxhw_ur_types": ["Athens"],
    "uxhw_ur_sizes": [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    "dependency_tracking_modes": ["None"],
    "distance_type": "Wasserstein-1",
    "n_processes": 1,
    "use_clt": False,
    "use_adaptive_steps": False,
    "use_binned_uxhw": False,
    "plot_distributions": False,
    "plot_comparison_distributions": False,
    "plot_adversary_distances": False,
    "plots_dir": "",
}

load_data_and_compute_equivalent_mc(args)
```

See the `LoadDataComputeEquivalentMCArgs` definition in
[`equivalent_mc_main.py`](./equivalent_mc_main.py) for the full set of keys
and their types. The function ignores the `output_file`, `auto_prefix`, and
`reporting_methods` keys. They stay optional so that callers that still set
them pass type checking.

## Outputs

The function prints the following information.

- Statistics on the ground truth and adversary samples.
- Statistics on the UxHw distributions.
- A table of equivalent Monte Carlo results, with the Monte Carlo counts,
  distances, and analytic predictions.

The function stores the results on each variable, in
`variable.emcc_results`. It writes no data files. It writes PNG plots to
`plots_dir`, or to the current directory when `plots_dir` is empty, only
for the plot keys you enable.

| Key | Plots |
| --- | --- |
| `plot_distributions` | The ground truth and each UxHw configuration. For distribution outputs, also a representative Monte Carlo run at each equivalent Monte Carlo count. |
| `plot_comparison_distributions` | The scaled adversary distances at each equivalent Monte Carlo count against their asymptotic prediction. |
| `plot_adversary_distances` | The adversary distances at each sample count. |

## Auditing the quality of Monte Carlo data

When you compare against Monte Carlo, you need to know how many samples are
enough for the ground truth and for the adversary pool. There is no
universal answer. Use the following guidelines.

1. Enable adversary-distance plotting. If the data quality is good, the
   measured adversary distances stay close to the predicted line. Large
   deviations usually come from a poor ground truth or an adversary pool
   that is too small. The tool also compares `sqrt(N)` times the mean
   adversary distance at the smallest and the largest measured sizes. It
   prints a warning when the ratio is far from 1. This check is the same
   test in numbers.
2. Use analytic formulas. If you know an analytic formula for the target
   distribution, use it. A ground truth of 1 million weighted samples from
   the exact probability density function is almost always better than 10
   to 100 million random samples. The reason is that the error of Monte
   Carlo only falls with the inverse square root of the sample count.
3. Focus on smaller representation sizes. A ground truth of 1 million Monte
   Carlo samples is often not good enough for Athens-256 and Athens-512. It
   is good enough for smaller representation sizes, and you can use the
   analytic estimates to extrapolate to larger sizes.
