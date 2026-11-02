# Nvidia Comms Perf Suite

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Proprietary-red)](LICENSE)

A comprehensive toolkit for GPU Communications Libraries performance testing and data analysis.

## Overview

Nvidia Comms Perf Suite is a collection of utilities designed to streamline the process of testing and analyzing performance data from NVIDIA GPU Communications Libraries. The toolkit provides:

1. **Test Script Generation**: Generate test scripts for performance benchmarking across different systems and configurations.
2. **Result Processing**: Process test results using one of three approaches:
   - **Convert**: Save results to local files in various formats (e.g., CSV) for analysis or sharing
   - **Upload**: Store results directly in databases (PostgreSQL, MySQL, SQLite) and/or cloud storage (S3)
   - **Report**: Generate comprehensive PDF performance analysis reports with customizable charts and visualizations

## Installation

### Virtual Environment (Recommended)

We recommend using a virtual environment to install Nvidia Comms Perf Suite. This helps isolate dependencies and avoid conflicts with other Python packages:

```bash
# Create a virtual environment
python -m venv nvcomms-perf-env

# Activate the virtual environment
source nvcomms-perf-env/bin/activate
```

### Build the package with uv

We use uv to build distributable artifacts (sdist/wheel):

```bash
# From the repo root
uv build

# Artifacts will be created under dist/
ls -l dist/
```

#### PostgreSQL URL Format

Nvidia Comms Perf Suite uses psycopg3 by default. Use this URL format:

- **PostgreSQL**: `postgresql+psycopg://user:password@host:port/database`

## Quick Start

### CLI usage
```bash
> nvcomms-perf --help

Usage: nvcomms-perf [OPTIONS] COMMAND [ARGS]...

  Performance Tools for NVIDIA GPU Communications Libraries

Options:
  -v, --verbose         Increase verbosity (-v, -vv)
  --version             Show version and exit
  --install-completion  Install completion for the current shell.
  --show-completion     Show completion for the current shell, to copy it or
                        customize the installation.
  --help                Show this message and exit.

Commands:
  generate-job-script  Generate job scripts for GPU Comm Libraries...
  parse-and-upload     (Deprecated) Use [bold red]convert[/bold red] instead
  convert              Convert test results to a local file in the...
  upload               Upload test results to a database (PostgreSQL,...
  report               Generate performance analysis reports from test results
```

### Workflow

The typical workflow for using nvcomms-perf is:

1. **Generate test scripts** using `generate-job-script`
2. **Run the tests** on your cluster (produces test output files)
3. **Process results** using one of these options:
   - `convert`: Save results to a local file (CSV, etc.)
   - `upload`: Upload results directly to a database and/or S3
   - `report`: Create PDF report from test tarball or CSV

### Step 1: Test Generation

Generate job scripts for performance testing:

```bash
# Generate a job script and print to stdout
nvcomms-perf generate-job-script --system CLUSTER_1 --systems-toml systems.toml --user-toml user.toml --testset-toml testset.toml

# Generate a job script and save to file
nvcomms-perf generate-job-script --system CLUSTER_1 --systems-toml systems.toml --user-toml user.toml --testset-toml testset.toml --output job.sbatch

# Generate with test report
nvcomms-perf generate-job-script --system CLUSTER_1 --systems-toml systems.toml --user-toml user.toml --testset-toml testset.toml --generate-test-report junit --output job.sbatch
```

### Step 2: Run Tests

After generating the job script, submit it to your cluster:

```bash
# Submit the job
sbatch job.sbatch

# The job will produce test output files (logs, tarballs, etc.)
```

### Step 3: Process Results

Choose one of the following options to process your test results:

#### Option A: Convert to Local File

Use `convert` to save results to a local file for analysis, sharing, or archival:

```bash
# Convert a single NCCL test output to a local CSV file
nvcomms-perf convert -i nccl-tests:/path/to/nccl-tests.log -o results.csv

# Convert test results from multiple tarballs to a single local CSV file
nvcomms-perf convert -i tarball:/path/to/archive1.tar.gz -i tarball:/path/to/archive2.tar.gz -o combined_results.csv

# Convert with explicit output format (CSV) and save to local file
nvcomms-perf convert -i nccl-tests:/path/to/test.log -o csv:/tmp/results.csv

# Convert tarball to local CSV file without progress bar
nvcomms-perf convert -i tarball:/path/to/archive.tar.gz -o results.csv --hide-progress

# Convert and stop on first error (instead of continuing)
nvcomms-perf convert -i tarball:/path/to/archive.tar.gz -o results.csv --stop-on-error

# Convert using custom number of parallel workers for faster processing
nvcomms-perf convert -i tarball:/path/to/archive.tar.gz -o results.csv --max-workers 8
```

#### Option B: Upload to Database/S3

Use `upload` to directly store results in a database and/or S3:

```bash
# Upload to PostgreSQL database
nvcomms-perf upload -i tarball:/path/to/archive.tar.gz --db-url postgresql+psycopg://user:pass@host:port/database

# Upload to S3
nvcomms-perf upload -i tarball:/path/to/archive.tar.gz --s3-bucket my-bucket --s3-region us-east-1

# Upload to both PostgreSQL and S3
nvcomms-perf upload -i tarball:/path/to/archive.tar.gz --db-url postgresql+psycopg://user:pass@host:port/database --s3-bucket my-bucket

# Upload multiple files
nvcomms-perf upload -i tarball:/path/to/archive1.tar.gz -i tarball:/path/to/archive2.tar.gz --db-url postgresql+psycopg://user:pass@host:port/database

# Use environment variables for credentials
export NVCOMMS_PERF_DB_URL="postgresql+psycopg://user:pass@host:port/database"
export NVCOMMS_PERF_S3_BUCKET="my-bucket"
export NVCOMMS_PERF_S3_REGION="us-east-1"
nvcomms-perf upload -i tarball:/path/to/archive.tar.gz
```

#### Option C: Generate Performance Reports

Use `report` to create comprehensive PDF performance analysis reports with various chart types:

**Available PDF Report Types:**
- **Basic Report** (`basic_report`):
    Basic report for NCCL includes: 
    - **Default Configuration Comparison**: Curves run with default NCCL algorithm and protocol. One NCCL_TESTS_SPLIT_MASK and scale (num nodes and num processes) per chart.
    - **Scale Comparison**: Curves run with default NCCL algorithm and protocol. One NCCL_TESTS_SPLIT_MASK per chart and all different scales in one chart.

- **Comprehensive Report** (`comprehensive_report`):
    Comprehensive report for NCCL includes the charts mentioned in "Basic report", additionally it includes:
    - **Algorithm Comparison**: Performance across different NCCL algorithms and protocols. One NCCL_TESTS_SPLIT_MASK and scale (num nodes and num processes) per chart.
    - **P2P Advanced**: Comparison of P2P-specific variables: NCCL_NCHANNELS_PER_NET_PEER, NCCL_P2P_NET_CHUNKSIZE and NCCL_IB_QPS_PER_CONNECTION.

The PDF report adds an empty page for Notes by the graphs, Data Tables and test commands for each of the legends in the graphs.

```bash
# Basic report with default settings
nvcomms-perf report -i results.csv -o report.pdf

# Comprehensive report with more graphs (if applicable to the data)
nvcomms-perf report -i results.csv -o report.pdf --pdf-type comprehensive_report

# Report from tarball with custom configuration
nvcomms-perf report -i tarball:/path/to/results.tar.gz -o report.pdf --plot-config custom_plot.toml --style-config custom_style.toml

# Report without data tables, test commands and empty pages for quicker generation times.
nvcomms-perf report -i results.csv -o report.pdf --no-data-tables --no-test-commands --no-empty-pages
```

### Configuration Examples

#### generate-job-script configuration examples

##### systems.toml
```toml
[common]
run_params = { "mpi_launcher" = "srun" }

[CLUSTER_1]
run_params = { "partition" = "gpu", "account" = "gpu" }
```

##### user.toml
```toml
CUDA_HOME = "/path/to/cuda"
NCCL_HOME = "/path/to/nccl"
NCCL_TEST_PATH = "/path/to/nccl_tests"

# SBATCH directives for job submission
sbatch = [
    "--gres=gpu:8",
    "--ntasks-per-node=8"
]
```

##### testset.toml
```toml
# NCCL Test Configuration
description = "allreduce_algo_proto_sweep"
library_name = "nccl"

# Optional: Custom commands for setup/cleanup
[common.sweep_custom_commands]
pre_command = "export NCCL_MNNVL_CLIQUE_ID=$(($SLURM_PROCID / $CLIQUE_SIZE))"
post_command = "unset NCCL_MNNVL_CLIQUE_ID"

[[test]]
name = "all_reduce"
binary = "all_reduce_perf"
envs_merge_strategy = "override"
args_merge_strategy = "override"
[test.sweep_envs]
NCCL_ALGO = ["RING", "TREE", "NVLSTREE"]
NCCL_PROTO = ["SIMPLE", "LL128"]
```

**Custom Command Options:**
- `pre_command`: Command executed before the test runs (limited to maximum 1 value)
- `post_command`: Command executed after the test completes (limited to maximum 1 value)
- Commands can be defined in `[common.sweep_custom_commands]` for all tests or in individual `[test.sweep_custom_commands]` sections

#### report configuation examples

##### plot_config.toml
```toml
[settings.x_axis]
scale = "semilogx"
ticks = [1024, 10240, 102400, 1048576, 10485760, 104857600, 1073741824]
labels = ["1KB", "10KB", "100KB", "1MB", "10MB", "100MB", "1GB"]

[settings.y_axis.limits]
auto_detect_margin = 0.1
auto_detect_round_to = 50
default_min = 0

[settings.data_processing]
default_performance_column = "max_bus_bw"

# Define multiple performance column types
[settings.data_processing.performance_columns.max_bus_bw]
sources = ["test_result.out_of_place.bus_bw", "test_result.in_place.bus_bw"]
operation = "max"
description = "Maximum bandwidth"
default = true

# Chart Type Definitions
[[chart]]
id = "default_comparison"
name = "Default Configuration Comparison"
description = "Compare performance with default algorithm and protocol settings"
performance_columns = ["max_bus_bw"]

[chart.fixed_params]
NCCL_ALGO = "DEFAULT"
NCCL_PROTO = "DEFAULT"
NCCL_MIN_CTAS = "DEFAULT"

[chart.chart_split_params]
num_nodes = "all"
num_processes = "all"
NCCL_TESTS_SPLIT_MASK = "all"

[chart.chart_combine_params]
local_register = "all"

# Analysis Profile Definitions
[[analysis_profiles]]
id = "collective_basic"
name = "Basic Collective Operations Analysis"
description = "Basic analysis for collective operations"
test_binaries = [
    "all_reduce_perf",
    "all_gather_perf",
    "reduce_scatter_perf"
]
chart_types = [
    "default_comparison"
]

# PDF Report Configurations
[[pdf_configs]]
id = "basic_report"
name = "Basic Performance Analysis"
description = "Basic analysis focusing on default configurations"
analysis_profiles = [
    "collective_basic"
]
```

### Legacy Command: parse-and-upload (Deprecated)

The `parse-and-upload` command is deprecated and will be removed in a future version. It combined the functionality of both `convert` and `upload` commands. Please use the separate commands for better control and flexibility.

```bash
# Old command (deprecated) - parsed and uploaded to CSV in one step
nvcomms-perf parse-and-upload /path/to/tarballs.tar.gz --upload-to csv

# New equivalent - just use convert
nvcomms-perf convert -i tarball:/path/to/tarballs.tar.gz -o results.csv

# Old command (deprecated) - parsed and uploaded to database
nvcomms-perf parse-and-upload /path/to/tarballs.tar.gz

# New equivalent - just use upload
nvcomms-perf upload -i tarball:/path/to/tarballs.tar.gz --db-url postgresql+psycopg://user:pass@host:port/database

# Old command with Splunk (deprecated)
nvcomms-perf parse-and-upload /path/to/tarballs.tar.gz --upload-to splunk

# New approach - convert to CSV first, then use separate tools for Splunk
nvcomms-perf convert -i tarball:/path/to/tarballs.tar.gz -o results.csv
# Then use Splunk forwarders or APIs to upload the CSV
```

## Plugin System

nvcomms-perf supports a plugin system similar to pytest, allowing you to extend the tool with custom commands. Plugins are automatically discovered and loaded via Python entry points.

### Using Plugins

Once a plugin is installed, its commands are automatically available in the CLI:

```bash
# View all commands including plugins
nvcomms-perf --help

# Use a plugin command
nvcomms-perf my-plugin-command
```

### Creating Plugins

To create a plugin:

1. Create a Python package that inherits from `nvcomms_perf.plugins.NvCommsPerfPlugin`
2. Define your plugin's entry point in setup.py or pyproject.toml
3. Install your plugin package

Example plugin structure:

```python
from nvcomms_perf.plugins import NvCommsPerfPlugin
import typer

class MyPlugin(NvCommsPerfPlugin):
    @property
    def name(self) -> str:
        return "my-command"

    @property
    def help(self) -> str:
        return "My custom plugin"

    def get_command(self) -> typer.Typer:
        app = typer.Typer()

        @app.command()
        def hello(name: str = "World"):
            print(f"Hello {name}!")

        return app
```

Register the plugin via entry points:

```toml
# pyproject.toml
[project.entry-points.nvcomms_perf]
my-plugin = "my_plugin_module:MyPlugin"
```

See the [minimal plugin](examples/minimal_plugin.py) and [plugin development guide](docs/plugin_development.rst) for examples.

## Documentation

For detailed documentation on each component:

- [Test Generation Documentation](docs/testing_automation/README.rst)
- [Convert Command Documentation](docs/convert/README.rst)
- [Upload Command Documentation](docs/upload/README.rst)
- [Report Command Documentation](docs/report/README.rst)
- [Legacy Parse and Upload Documentation](docs/parse_and_upload/README.rst) (Deprecated)
- [API Reference](docs/api/index.rst)
- [Contributing Guide](docs/contributing.rst)
- [Changelog](docs/changelog.rst)

## License

Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
