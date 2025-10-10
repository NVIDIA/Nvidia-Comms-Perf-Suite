..
   Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
   See LICENSE.txt for license information

Convert Command
===============

Overview
--------

The ``convert`` command is used to convert test results (from tarballs or individual test outputs) to a local file in the specified format (such as CSV).

Quick Start
-----------

Convert test results to a local file:

.. code-block:: bash

   # Convert a single NCCL test output to a local CSV file
   nvcomms-perf convert -i nccl-tests:/path/to/nccl-tests.log -o results.csv

   # Convert test results from tarball to a local CSV file
   nvcomms-perf convert -i tarball:/path/to/archive.tar.gz -o results.csv

   # Convert multiple test result sources to a single local CSV file
   nvcomms-perf convert -i tarball:/path/to/archive1.tar.gz -i tarball:/path/to/archive2.tar.gz -o combined_results.csv

   # Specify output format explicitly
   nvcomms-perf convert -i nccl-tests:/path/to/test.log -o csv:/tmp/results.csv

   # Hide progress bar
   nvcomms-perf convert -i tarball:/path/to/archive.tar.gz -o results.csv --hide-progress

   # Stop on first error
   nvcomms-perf convert -i tarball:/path/to/archive.tar.gz -o results.csv --stop-on-error

   # Use custom number of parallel workers
   nvcomms-perf convert -i tarball:/path/to/archive.tar.gz -o results.csv --max-workers 8

Command-line Interface
----------------------

.. code-block:: bash

   Usage: nvcomms-perf convert [OPTIONS]

     Convert test results to a local file in the specified format.

   Options:
     -i, --input INPUTFILE           Input file(s) to convert.

                                     Supported input types: nccl-tests, tarball
                                     Format: <type>:<path> Example: nccl-tests:/path/to/nccl-tests.log
                                     Multiple input files can be specified by using --input multiple times. Each input file will be processed and combined into the final output.  [required]
     -o, --output OUTPUTFILE         Output file path to write the converted results.

                                     Supported output types: csv
                                     Format: <type>:<path> OR <path> Example: csv:/tmp/results.csv , /tmp/results.csv
                                     If type is not specified, it will be inferred from the file extension.  [required]
     --max-workers INTEGER           Maximum number of parallel workers for parsing. Default: MIN(number of CPU cores, number of input files)
     -p, --show-progress / -P, --hide-progress
                                     Show progress bar during parsing  [default: show-progress]
     -c, --continue-on-error / -C, --stop-on-error
                                     Continue processing other files if one fails [default: stop-on-error]
     --help                          Show this message and exit.

Migration from parse-and-upload
-------------------------------

The ``convert`` command replaces the deprecated ``parse-and-upload`` command. Use ``convert`` to save test results to a local file instead of uploading to a database.

.. code-block:: bash

   # Old command (deprecated)
   nvcomms-perf parse-and-upload /path/to/tarballs.tar.gz

   # New command
   nvcomms-perf convert -i tarball:/path/to/tarballs.tar.gz -o results.csv

Key Differences
~~~~~~~~~~~~~~~

1. **Purpose**: ``convert`` is for converting test results to a local file, not for uploading to a database.
2. **Input specification**: Explicit type required (e.g., ``tarball:`` or ``nccl-tests:``)
3. **Output specification**: Output must be specified with ``-o``, always creating a local file
4. **Multiple inputs**: Can process multiple test result sources in one command using multiple ``-i`` options
5. **Progress control**: Use ``--show-progress`` / ``--hide-progress``
6. **Error handling**: Use ``--continue-on-error`` / ``--stop-on-error``

Supported Formats
-----------------

- Input: NCCL test output, tarball
- Output: CSV (more formats may be added in the future)
