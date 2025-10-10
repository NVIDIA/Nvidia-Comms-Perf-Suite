..
   Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
   See LICENSE.txt for license information

Report Generation
=================

Overview
--------
The ``report`` command generates performance analysis reports from test results. It supports CSV files and tarballs as input, and produces PDF reports with various analysis types and visualizations.

Quick Start
-----------

.. code-block:: bash

    # Basic report with default settings
    nvcomms-perf report -i results.csv -o report.pdf

    # Comprehensive report with custom configuration
    nvcomms-perf report -i results.csv -o report.pdf --pdf-type comprehensive_report --plot-config custom_plot.toml --style-config custom_style.toml

Input Types
-----------
- **CSV Files**: Direct test result data
- **Tarballs**: Archives containing test results

Report Types
------------
The report generator offers two PDF report types:

Basic Report (``basic_report``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Default report type
- Includes:
  - Basic collective operations analysis
  - Basic P2P operations analysis
- Uses default algorithm configurations
- Suitable for routine performance testing

Comprehensive Report (``comprehensive_report``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Advanced report type
- Includes:
  - Advanced collective operations analysis
  - Advanced P2P operations analysis
  - Algorithm comparisons
  - Detailed optimizations
- Suitable for performance tuning

Chart Types
-----------

Default Configuration Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purpose:
    Compare performance with default algorithm and protocol settings

Fixed Parameters:
    - NCCL_ALGO = "DEFAULT"
    - NCCL_PROTO = "DEFAULT"
    - NCCL_MIN_CTAS = "DEFAULT"

Split Parameters:
    - num_nodes = "all"
    - num_processes = "all"
    - NCCL_TESTS_SPLIT_MASK = "all"

Combine Parameters:
    - local_register = "all"


Algorithm Comparison
~~~~~~~~~~~~~~~~~~~~
Purpose:
    Compare performance of different NCCL algorithms

Fixed Parameters:
    - NCCL_MIN_CTAS = "DEFAULT"

Split Parameters:
    - num_nodes = "all"
    - num_processes = "all"
    - NCCL_TESTS_SPLIT_MASK = "all"

Combine Parameters:
    - NCCL_ALGO = "all"
    - NCCL_PROTO = "all"
    - local_register = "all"


Scale Comparison
~~~~~~~~~~~~~~~~
Purpose:
    Compare performance across different node and process counts

Fixed Parameters:
    - NCCL_ALGO = "DEFAULT"
    - NCCL_PROTO = "DEFAULT"
    - NCCL_MIN_CTAS = "DEFAULT"

Split Parameters:
    - NCCL_TESTS_SPLIT_MASK = "all"

Combine Parameters:
    - num_nodes = "all"
    - num_processes = "all"
    - local_register = "all"

P2P Basic Comparison
~~~~~~~~~~~~~~~~~~~~
Purpose:
    Compare basic P2P performance with default configurations

Fixed Parameters:
    - NCCL_ALGO = "DEFAULT"
    - NCCL_PROTO = "DEFAULT"
    - NCCL_MIN_CTAS = "DEFAULT"

Split Parameters:
    - num_nodes = "all"
    - num_processes = "all"
    - NCCL_TESTS_SPLIT_MASK = "all"

Combine Parameters:
    - local_register = "all"

P2P Advanced Comparison
~~~~~~~~~~~~~~~~~~~~~~~
Purpose:
    Compare P2P performance with advanced settings

Fixed Parameters:
    - None

Split Parameters:
    - num_nodes = "all"
    - num_processes = "all"
    - NCCL_TESTS_SPLIT_MASK = "all"

Combine Parameters:
    - local_register = "all"
    - NCCL_NCHANNELS_PER_NET_PEER = "all"
    - NCCL_P2P_NET_CHUNKSIZE = "all"
    - NCCL_IB_QPS_PER_CONNECTION = "all"

Performance Metrics
-------------------
The report generator supports multiple performance metrics:

max_bus_bw
~~~~~~~~~~
- Default performance metric
- Maximum bandwidth from:
  - out_of_place.bus_bw
  - in_place.bus_bw

min_latency
~~~~~~~~~~~
- Minimum time from:
  - out_of_place.time
  - in_place.time

min_bus_bw
~~~~~~~~~~
- Minimum bandwidth from:
  - out_of_place.bus_bw
  - in_place.bus_bw

Report Components
-----------------

The report generation system is composed of several layers that work together to create the final PDF report. This layered approach provides flexibility and makes it easier to create custom reports by allowing you to mix and match different components:

**Flexibility Benefits:**
- **Modular Design**: Each layer can be configured independently
- **Custom Combinations**: Mix different chart types with different test binaries
- **Reusable Components**: Chart types and analysis profiles can be reused across different reports
- **Scalable Architecture**: Easy to add new chart types, test binaries, or analysis profiles
- **Targeted Analysis**: Create reports focused on specific aspects (e.g., only P2P operations, only algorithm comparisons)

Analysis Profiles
~~~~~~~~~~~~~~~~~
Combines chart types with specific test binaries:

1. **Collective Basic** (``collective_basic``)
   - Test binaries: all_reduce_perf, all_gather_perf, reduce_scatter_perf
   - Charts: default_comparison, scale_comparison
   - Used in: basic_report

2. **Collective Advanced** (``collective_advanced``)
   - Test binaries: all_reduce_perf, all_gather_perf, reduce_scatter_perf
   - Charts: default_comparison, scale_comparison, algo_comparison
   - Used in: comprehensive_report

3. **P2P Basic** (``p2p_basic``)
   - Test binaries: sendrecv_perf, alltoall_perf
   - Charts: default_comparison, scale_comparison
   - Used in: basic_report

4. **P2P Advanced** (``p2p_advanced``)
   - Test binaries: sendrecv_perf, alltoall_perf
   - Charts: default_comparison, p2p_advanced
   - Used in: comprehensive_report

PDF Report Types
~~~~~~~~~~~~~~~~
Combines analysis profiles into complete reports:

1. **Basic Report** (``basic_report``)
   - Includes: collective_basic, p2p_basic
   - Focus: Default configurations and basic scaling
   - Best for: Regular performance testing

2. **Comprehensive Report** (``comprehensive_report``)
   - Includes: collective_advanced, p2p_advanced
   - Focus: Algorithm comparison and detailed analysis
   - Best for: Performance optimization

Report Generation Flow
~~~~~~~~~~~~~~~~~~~~~~
1. **Data Input**
   - CSV files or tarballs are loaded
   - Data is preprocessed according to configuration

2. **Chart Generation**
   For each analysis profile in the PDF type:
   - Load specified test binary data
   - Generate charts defined in the profile
   - Apply styling from style configuration

3. **Report Assembly**
   - Charts are organized by analysis profile
   - Data tables are added
   - Test commands are included
   - Empty pages are inserted

Example Flow
~~~~~~~~~~~~
For a basic report with collective operations:

.. code-block:: text

    Data (CSV/Tarball)
    ├── collective_basic profile
    │   ├── all_reduce_perf data
    │   │   ├── default_comparison chart
    │   │   └── scale_comparison chart
    │   ├── all_gather_perf data
    │   │   ├── default_comparison chart
    │   │   └── scale_comparison chart
    │   └── reduce_scatter_perf data
    │       ├── default_comparison chart
    │       └── scale_comparison chart
    └── p2p_basic profile
        ├── sendrecv_perf data
        │   ├── default_comparison chart
        │   └── scale_comparison chart
        └── alltoall_perf data
            ├── default_comparison chart
            └── scale_comparison chart

Configuration Impact
~~~~~~~~~~~~~~~~~~~~
- **Plot Configuration**: Defines chart types, analysis profiles, and PDF types
- **Style Configuration**: Controls visual appearance of all charts
- **Runtime Options**: Control additional content (data tables, test commands)

Command Line Options
--------------------

Required Arguments
~~~~~~~~~~~~~~~~~~
- ``-i, --input``: Input file(s) to process (multiple allowed)
- ``-o, --output``: Output report path

Report Configuration
~~~~~~~~~~~~~~~~~~~~
- ``--pdf-type``: Report type (default: basic_report)
  - ``basic_report``: Default configurations and basic analysis
  - ``comprehensive_report``: Detailed analysis with algorithm comparisons
- ``--plot-config``: Custom plot configuration file
- ``--style-config``: Custom style configuration file

Display Options
~~~~~~~~~~~~~~~
- ``--show-progress/--hide-progress``: Control progress bar
- ``--include-empty-pages/--no-empty-pages``: Add note pages after charts
- ``--include-data-tables/--no-data-tables``: Include raw data tables
- ``--include-test-commands/--no-test-commands``: Show test commands

Error Handling
~~~~~~~~~~~~~~
- ``--continue-on-error/--fail-on-error``: Control error behavior

Configuration Files
-------------------

Style Configuration (style_config.toml)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Controls report appearance, formatting and color palette.

.. code-block:: toml

    [pdf_settings]
    page_size = "letter"
    orientation = "landscape"
    dpi = 300

    [chart_style]
    font_size = 12
    line_width = 2.0
    grid = true

Writing Plot Configuration
--------------------------

The plot configuration file defines chart types, analysis profiles, and PDF report types. Here's how to write your own configuration:

.. code-block:: toml

    # =============================================================================
    # NCCL Performance Analysis Chart Configuration
    # =============================================================================

    [settings.x_axis]
    scale = "semilogx"
    ticks = [1024, 10240, 102400, 1048576, 10485760, 104857600, 1073741824, 10737418240, 107374182400]
    labels = ["1KB", "10KB", "100KB", "1MB", "10MB", "100MB", "1GB", "10GB", "100GB"]

    [settings.y_axis.limits]
    auto_detect_margin = 0.1
    auto_detect_round_to = 50
    default_min = 0
    default_max = 50

    [settings.data_processing]
    default_performance_column = "max_bus_bw"
    size_column = "test_result.size"

    # Define multiple performance column types
    [settings.data_processing.performance_columns.max_bus_bw]
    sources = ["test_result.out_of_place.bus_bw", "test_result.in_place.bus_bw"]
    operation = "max"
    description = "Maximum bandwidth"
    default = true

    [settings.data_processing.performance_columns.min_latency]
    sources = ["test_result.out_of_place.time", "test_result.in_place.time"]
    operation = "min"
    description = "Minimum latency"

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

    [[chart]]
    id = "algo_comparison"
    name = "Algorithm Comparison"
    description = "Compare performance of different NCCL algorithms"
    performance_columns = ["max_bus_bw"]

    [chart.fixed_params]
    NCCL_MIN_CTAS = "DEFAULT"

    [chart.chart_split_params]
    num_nodes = "all"
    num_processes = "all"
    NCCL_TESTS_SPLIT_MASK = "all"

    [chart.chart_combine_params]
    NCCL_ALGO = "all"
    NCCL_PROTO = "all"
    local_register = "all"

    [[analysis_profiles]]
    id = "collective_advanced"
    name = "Advanced Collective Operations Analysis"
    description = "Detailed analysis including algorithm comparisons"
    test_binaries = [
        "all_reduce_perf",
        "all_gather_perf",
        "reduce_scatter_perf"
    ]
    chart_types = [
        "default_comparison",
        "algo_comparison"
    ]

    # PDF Report Configurations
    [[pdf_configs]]
    id = "comprehensive_report"
    name = "Comprehensive Performance Analysis"
    description = "Detailed analysis including algorithm comparisons"
    analysis_profiles = [
        "collective_advanced"
    ]

Key Components
~~~~~~~~~~~~~~

1. **Settings Section**:
   - ``x_axis``: X-axis configuration (scale, ticks, labels)
   - ``y_axis.limits``: Y-axis limit configuration
   - ``data_processing``: Performance column definitions and data processing settings

2. **Chart Definitions**:
   - Each chart is defined under ``[[chart]]``
   - ``id``: Unique chart type identifier
   - ``name``: Human-readable chart name
   - ``description``: Chart description
   - ``performance_columns``: Which performance metrics to plot
   - ``fixed_params``: Parameters that are fixed for this chart type
   - ``chart_split_params``: Parameters that create separate charts
   - ``chart_combine_params``: Parameters that are combined within charts

3. **Analysis Profiles**:
   - Each profile is defined under ``[[analysis_profiles]]``
   - ``id``: Unique profile identifier
   - ``name``: Human-readable profile name
   - ``description``: Profile description
   - ``test_binaries``: List of test binaries to analyze
   - ``chart_types``: List of chart types to generate

4. **PDF Configurations**:
   - Each PDF type is defined under ``[[pdf_configs]]``
   - ``id``: Unique PDF type identifier
   - ``name``: Human-readable PDF type name
   - ``description``: PDF type description
   - ``analysis_profiles``: List of analysis profiles to include

Chart Type Parameter Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chart parameters control how data is filtered and grouped:

- **Fixed Parameters**: Values that must match exactly (e.g., NCCL_ALGO="DEFAULT")
- **Split Parameters**: Each unique combination creates a separate chart
- **Combine Parameters**: All combinations appear within the same chart

Parameter Syntax and Behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``

**Fixed Parameters** (``[chart.fixed_params]``):
    Parameters with exact values that filter the data. Only data matching these exact values will be included in the chart.

    Syntax: ``parameter_name = "exact_value"``
    
    Example:
    - ``NCCL_ALGO = "DEFAULT"``: Only include data where NCCL_ALGO equals exactly "DEFAULT"
    - ``NCCL_PROTO = "DEFAULT"``: Only include data where NCCL_PROTO equals exactly "DEFAULT"
    - ``NCCL_MIN_CTAS = "DEFAULT"``: Only include data where NCCL_MIN_CTAS equals exactly "DEFAULT"

**Split Parameters** (``[chart.chart_split_params]``):
    Parameters that create separate charts. Each unique combination of these parameters generates its own chart.

    Syntax: ``parameter_name = "all"`` or ``parameter_name = ["value1", "value2", ...]``
    
    Examples:
    - ``num_nodes = "all"``: Create a separate chart for each unique number of nodes
    - ``num_processes = "all"``: Create a separate chart for each unique number of processes
    - ``NCCL_TESTS_SPLIT_MASK = "all"``: Create a separate chart for each unique split mask value

    Result: If you have data with 2 nodes, 3 process counts, and 2 split mask values, you'll get 2 × 3 × 2 = 12 separate charts.

**Combine Parameters** (``[chart.chart_combine_params]``):
    Parameters that are combined within the same chart. All combinations of these parameters appear as different lines/curves in the same chart.

    Syntax: ``parameter_name = "all"`` or ``parameter_name = ["value1", "value2", ...]``
    
    Examples:
    - ``local_register = "all"``: Show all register types as different lines in the same chart
    - ``NCCL_ALGO = "all"``: Show all algorithms as different lines in the same chart
    - ``NCCL_PROTO = "all"``: Show all protocols as different lines in the same chart

    Result: If you have data with 3 algorithms and 2 protocols, you'll get 3 × 2 = 6 different lines in the same chart.

Chart Type Parameter Interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The three parameter types work together to control chart generation:

1. **Fixed parameters** filter the data first (only matching data is considered)
2. **Split parameters** determine how many charts are created
3. **Combine parameters** determine how many lines appear in each chart

**Important**: If no data matches the specified parameters, the parameter is ignored and the process continues. This ensures that reports can still be generated even if some expected data is missing.

Example scenario with the following configuration:
- Fixed: ``NCCL_ALGO = "DEFAULT"``
- Split: ``num_nodes = "all"``, ``num_processes = "all"``
- Combine: ``local_register = "all"``

If your data contains:
- 2 unique node counts (1, 2)
- 3 unique process counts (4, 8, 16)
- 2 unique register types (disabled, enabled)

Result:
- 2 × 3 = 6 charts will be created (one for each node/process combination)
- Each chart will have 2 lines (one for each register type)
- Only data with NCCL_ALGO="DEFAULT" will be included

**No Match Handling**:
- If no data has NCCL_ALGO="DEFAULT", the fixed parameter is ignored and all data is used
- If no data has a specific node count, that chart is simply not created
- If no data has a specific register type, that line is not shown in the chart

Examples
--------

Basic Report Generation
~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: bash

    # Generate report from CSV
    nvcomms-perf report -i results.csv -o report.pdf

    # Generate report from tarball
    nvcomms-perf report -i tarball:/path/to/results.tar.gz -o report.pdf

Tarball Analysis
~~~~~~~~~~~~~~~~
.. code-block:: bash

    # Process multiple tarballs with data tables and test commands
    nvcomms-perf report -i tarball:/path/to/nccl_tests_run1.tar.gz -i tarball:/path/to/nccl_tests_run2.tar.gz -o nccl_analysis.pdf --pdf-type comprehensive_report --include-data-tables --include-test-commands --continue-on-error

Advanced Analysis
~~~~~~~~~~~~~~~~~
.. code-block:: bash

    # Comprehensive report with custom configuration
    nvcomms-perf report -i results.csv -o report.pdf --pdf-type comprehensive_report --plot-config custom_plot.toml --style-config custom_style.toml --include-data-tables --include-test-commands

Multiple Data Sources
~~~~~~~~~~~~~~~~~~~~~
.. code-block:: bash

    # Combine multiple test results
    nvcomms-perf report -i results1.csv -i results2.csv -i tarball:/path/to/archive.tar.gz -o combined_report.pdf --pdf-type comprehensive_report
