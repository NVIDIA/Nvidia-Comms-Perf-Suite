..
   Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
   See LICENSE.txt for license information

Changelog
=========

All notable changes to nvcomms-perf will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[0.7.1] - 2025-08-22
--------------------

Added
~~~~~
- [generate-job-script] Added sweep_custom_commands support in testset configuration to enable custom shell commands before and after test execution

[0.7.0] - 2025-07-31
--------------------

Added
~~~~~
- New ``report`` command for generating PDF performance analysis reports from test results
- [report] Support for CSV files and tarballs as input sources
- [report] Two PDF report types: basic_report and comprehensive_report
- [report] Five chart types: Default Configuration Comparison, Algorithm Comparison, Scale Comparison, P2P Basic, P2P Advanced
- [report] Analysis profiles system for combining chart types with specific test binaries
- [report] Custom plot and style configuration support via TOML files
- [report] Chart type parameter configuration system with fixed, split, and combine parameters
- [report] Performance metrics customizations: max_bus_bw, min_latency, min_bus_bw
- [report] Progress tracking and error handling with ``--continue-on-error`` option
- [report] Multiple input file processing to create a single report
- [report] Optional data tables, test commands, and empty pages for notes in reports
- [report] Added APIs to directly make calls to the report generation pipeline
- [report] Documentation for the report command and pdf report parameters

Changed
~~~~~~~
- Updated dependencies to latest versions
- Improved speed for csv output of "convert" with migration to polars from pandas

Fixed
~~~~~
- Improved handling of xNCCL_*=default in convert
- Fix start_time reporting in test_metadata

[0.6.2] - 2025-07-21
--------------------

Added
~~~~~
- Added column_property fields to PerfRecord for easier dashboard filtering and improved query performance
- Enhanced error handling with improved progress reporting and completion indicators
- Added cluster name capture from SLURM_CLUSTER_NAME environment variable in sweep metadata
- Made SQLDBOptions immutable (frozen=True) for better data integrity and hashability

Changed
~~~~~~~
- Improved error handling during file parsing to continue processing even when some records fail
- Updated system and network metadata fields to use unified naming convention for environment variables
- Enhanced progress reporting with clearer status updates and completion information

Fixed
~~~~~
- Fixed variable naming conflicts between `collect_sweep_metadata` and environment variables from `/etc/os-release`
- Skipped incomplete metadata sets to prevent parsing errors
- Fixed environment setup to consistently prefix system label variables

[0.6.1] - 2025-07-07
--------------------

Added
~~~~~
- Advanced parsing and representation of NCCL test command-line options for more flexible configuration extraction
- New model for handling NCCL test command options supporting both dictionary and command-line string inputs
- Enhanced test coverage for command-line parsing, configuration precedence, and performance record creation

Changed
~~~~~~~
- Moved `psycopg[binary]` and `boto3` from optional to required dependencies for streamlined installation
- Updated all documentation to use new PostgreSQL URL format (`postgresql+psycopg://`) for consistency
- Renamed several configuration fields for consistency and clarity
- Updated configuration logic to prioritize command-line options and streamline merging from multiple sources
- Improved CodeRabbit configuration formatting and consistency

Fixed
~~~~~
- Fixed undefined behavior (UB) bug in NCCL test argument parsing
- Improved handling of command-line arguments for robust parsing and error resilience
- Fixed installation issues with `psycopg2-binary` on Python 3.9 on Mac

Removed
~~~~~~~
- Removed `psycopg2-binary` dependency due to installation issues on Mac with Python 3.9
- Removed specialized chat personas from CodeRabbit configuration

[0.6.0] - 2025-06-25
--------------------

Added
~~~~~
- New ``convert`` command for converting test results (from tarballs or individual test outputs) to a local file in the specified format (currently supports CSV output)
- New ``upload`` command for uploading test results to databases (PostgreSQL, MySQL, SQLite) and/or S3 storage
- Support for processing multiple test result sources in parallel with ``convert`` and ``upload`` commands
- Progress bars for better visibility during file processing
- Ability to process individual NCCL test output files directly (not just tarballs)
- Automatic deduplication in ``upload`` command to avoid duplicate uploads
- Support for both database and S3 storage backends simultaneously
- Environment variable support for all upload configuration options
- Schema support for PostgreSQL databases in SQLDBClient
- Plugin system implementation for nvcomms-perf extensions
- Partner scripts for enhanced functionality
- New metadata parser for improved data handling
- NCCL-tests parser enhancements
- Description field added to PerfRecord model
- CODEOWNERS file for better code review management
- CodeRabbit integration for automated code review
- Symmetric memory testcase configuration (bewilliams user toml)

Changed
~~~~~~~
- The ``--verbose`` flag now supports multiple levels (``-v`` for INFO, ``-vv`` for DEBUG)
- Improved logging configuration with better formatting and level control
- Added database drivers (`psycopg[binary]`) and S3 support (`boto3`) as core dependencies
- Replaced ``parse-and-upload`` with two alternative commands: ``convert`` (for local files) and ``upload`` (for databases/S3)
- CI pipeline now automatically adds MR reviewers
- Internal container images now include configuration files
- Removed runtime connect from default systems toml configuration

Deprecated
~~~~~~~~~~
- The ``parse-and-upload`` command is deprecated in favor of the new ``convert`` and ``upload`` commands

[v0.5.0] - 2025-05-22
---------------------
Added
~~~~~
- Added support for binary name suffixes (e.g. _mpi) to make test configurations reusable across different container images

Changed
~~~~~~~
- Documentation and test configuration updates.

[v0.4.0] - 2025-05-12
---------------------

Added
~~~~~
- Generate test report after tests are done (JUNIT support via --generate-test-report)
- Add flag --append-config-contents to include TOML file contents in generated job scripts
- Add workspace support for container runs
- Support for parsing both avg and max output from nccl-tests
- System and user configuration for AWS-IAD cluster
- New configs for algo/proto and platform perf sweeps

Changed
~~~~~~~
- Quotes added to env values in generated scripts
- Redirect stdout/stderr/dbg to each rank's individual directory
- Fix sweep dir for archival, make default case insensitive, ignore_DBG dirs

Deprecated
~~~~~~~~~~
- None

Removed
~~~~~~~
- None

Fixed
~~~~~
- Fix dependency issue caused by typer and click
- Fix environment variables for different testing configs

Security
~~~~~~~~
- None

[v0.3.0] - 2025-05-01
---------------------
Added
~~~~~
- Add --ignore-errors flag to continue parsing and upload for erroneous logs
- Update configurations to reduce total runtime
- Set NCCL_IB_SL=1 for cluster
- Add in GB200 IB HCA list to resolve NCCL not knowing which NICs are the IB...

Changed
~~~~~~~
- Minor changes on CI and docs

Fixed
~~~~~
- Fix nvl domain data collection with slurm

[v0.2.0] - 2025-04-24
---------------------
Added
~~~~~
- Add support to read xNCCL env variables
- Add NVL domain support
- Add parser to accept tarball files
- Add single entry for both generate-job-script and parser
- Artifact test results upon completion
- Add a new forwarder: csv
- Add testing config files for NCCL 2.27 testing

Changed
~~~~~~~
- Update GB200 input configurations
- Fix mpirun launcher use with mpi_type
- Fix parse_and_upload to fit the new metadata format
- Update docs and README
- Make --output of generate-job-script optional, and suppress info logs by default
- Bugfix: run_type_params is not defined in template

[v0.1.0] - 2025-04-15
---------------------
Added
~~~~~
- Initial release of nvcomms-perf
- Testing Automation framework for generating and managing test scripts
- Performance Data Processing tools for parsing and forwarding data
- Support for NCCL performance data parsing
- CSV and Splunk data forwarding capabilities
