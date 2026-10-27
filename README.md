# Signaloid Benchmarking

The Signaloid Benchmarking package provides tools for benchmarking
applications that use Signaloid's UxHw® technology for distributional
arithmetic against equivalent Monte Carlo methods. The benchmarking tool
builds and runs an application natively and with UxHw, computes the
equivalent Monte Carlo count of each UxHw configuration, collects timing
data, and writes reports.

The package provides the `signaloid_benchmarking` Python package. It builds
on the [Signaloid Data Toolkit](https://github.com/signaloid/signaloid-data-toolkit)
(the `signaloid` package on PyPI), which provides the Ux Data parsing, distance
metrics, and plotting the benchmarking tools use.

## Contents

| Package | Description |
|---|---|
| [`signaloid_benchmarking.automation`](src/signaloid_benchmarking/automation/README.md) | The `signaloid-benchmarking` command-line tool that runs the whole benchmarking pipeline. |
| [`signaloid_benchmarking.equivalent_mc`](src/signaloid_benchmarking/equivalent_mc/README.md) | Computes equivalent Monte Carlo counts from ground truth, adversary, and UxHw databases. |
| [`signaloid_benchmarking.profiling`](src/signaloid_benchmarking/profiling/README.md) | The `signaloid-memory-profile` command-line tool that measures the uncertain memory pool an application needs. |
| [`signaloid_benchmarking.differential_accuracy`](src/signaloid_benchmarking/differential_accuracy/README.md) | The `distance_uxhw_vs_mc` command-line tool that compares how close a baseline and an under-test UxHw run land to the same ground truth, and the `ks_convergence_gate` tool that drops the configurations that converge on neither run. |
| `signaloid_benchmarking.benchmark_timing` | The bash timing and compilation scripts the pipeline sources. |
| `signaloid_benchmarking.distribution_helpers` | Helpers for collapsing, binning, and checking the health of distributions. |
| `signaloid_benchmarking.configs` | Ready-made YAML sweep configurations for the `--config` argument. |
| `signaloid_benchmarking.assets` | The build-template assets, a git submodule. |

## Requirements

The Signaloid Benchmarking package requires Python 3.11 or later. See
`pyproject.toml` for the full list of Python dependencies. The benchmarking
tool also needs system tools and the Signaloid UxHw SDK. See the automation
[README.md](src/signaloid_benchmarking/automation/README.md) for the full
list.

The build-template assets are a git submodule at
`src/signaloid_benchmarking/assets`. Clone with `--recurse-submodules`, or run
`git submodule update --init` after cloning.

## Installation

Install the latest version from the GitHub repository.
```bash
python -m pip install git+https://github.com/signaloid/signaloid-benchmarking
```

Alternatively, clone this repository and install from source.
```bash
git clone --recurse-submodules https://github.com/signaloid/signaloid-benchmarking
cd signaloid-benchmarking
python -m pip install .
```

Add the `sheets` extra to also install the optional Google Sheets upload.
```bash
python -m pip install ".[sheets]"
```

## Usage

### Benchmarking UxHw applications

Use the `signaloid-benchmarking` command-line tool to benchmark an application
running with UxHw against a Monte Carlo baseline. The following example
benchmarks for UxHw Core microarchitectures Athens and Jupiter for precisions 8,
16, and 32, for both types of correlation tracking.

```bash
signaloid-benchmarking \
    --path-to-application ./my-uxhw-app \
    --path-to-uxhw-sdk ~/project-uxhw-sdk \
    -u Athens Jupiter \
    -s 8 16 32 \
    -c Disabled Autocorrelation \
    -r Mean
```

The tool needs access to the Signaloid UxHw SDK to build the applications for
UxHw, or `--run-uxhw-scce` to build and run them on the Signaloid Cloud
Compute Engine through the signaloid-cli helper scripts instead. The Intel Pin
tool is optional and off by default. Export `PIN_ROOT` and
pass `--measure-dynamic-instructions` to also measure the dynamic instruction
count. Without that flag the run never uses Pin, even when `PIN_ROOT` is set,
and reports the count as missing. Arguments
`-u/--representation-types`, `-s/--representation-sizes`,
`-c/--uncertainty-correlation_types`, `-r/--reporting-methods` can also be
supplied using a YAML file with `--config <file>`. The `configs/` directory
of the package holds ready-made sweeps.

For details, see the package [README.md](src/signaloid_benchmarking/automation/README.md).

### Profiling memory use

Use the `signaloid-memory-profile` command-line tool to find the exact number
of uncertain memory pool elements one UxHw run needs.

```bash
signaloid-memory-profile \
    --path-to-application ./my-uxhw-app \
    --representation-type Athens \
    --representation-size 4 \
    --output athens-4.json
```

For details, see the package [README.md](src/signaloid_benchmarking/profiling/README.md).

### Comparing accuracy of two UxHw runs

Use the `distance_uxhw_vs_mc` command-line tool to check whether an
under-test UxHw run landed closer to the ground truth than a baseline run.
It writes a markdown table with the distance of each run to the ground truth
and a traffic-light score per configuration.

```bash
distance_uxhw_vs_mc \
    --ground-truth-database ground-truth.db \
    --uxhw-baseline-database old-build.db \
    --uxhw-under-test-database new-build.db \
    --uxhw-tables Evaluation \
    --baseline-uncertainty-representation-type Atlas \
    --under-test-uncertainty-representation-type Atlas \
    --uncertainty-representation-sizes 16 \
    --yaml-signaloid signaloid.yaml \
    -o report.md
```

For details, see the package [README.md](src/signaloid_benchmarking/differential_accuracy/README.md).

### Computing equivalent Monte Carlo counts

Import `load_data_and_compute_equivalent_mc` from `signaloid_benchmarking` to
compute equivalent Monte Carlo counts from existing ground truth, adversary,
and UxHw databases without running the whole pipeline. For details, see the
package [README.md](src/signaloid_benchmarking/equivalent_mc/README.md).

## Development

Install the package and its development dependencies with Poetry, then run
the checks that CI runs.

```bash
poetry install --all-extras
poetry run flake8 src/
poetry run black --check src/
poetry run mypy -p signaloid_benchmarking
poetry run pytest
```

The unit tests live beside the code they test, as `*_test.py` files. The
end-to-end smoke test in [`tests/e2e/`](tests/e2e/README.md) runs the whole
pipeline against a pinned public demo. It needs a UxHw SDK and network
access, and is skipped otherwise.

