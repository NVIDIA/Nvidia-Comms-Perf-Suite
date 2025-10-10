..
   Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
   See LICENSE.txt for license information

Getting Started with Nvidia Comms Perf Suite
=================================

This guide will help you get started with Nvidia Comms Perf Suite quickly.

Installation
------------

Virtual Environment (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend using a virtual environment to install Nvidia Comms Perf Suite. This helps isolate dependencies and avoid conflicts with other Python packages:

.. code-block:: bash

   # Create a virtual environment
   python -m venv nvcomms-perf-env

   # Activate the virtual environment
   source nvcomms-perf-env/bin/activate

Install Nvidia Comms Perf Suite
~~~~~~~~~~~~~~~~~~~~

Install Nvidia Comms Perf Suite using pip:

.. code-block:: bash

   pip install nvcomms-perf --index-url <insert url>

Quick Start
-----------

Command-line Help
~~~~~~~~~~~~~~~~~

To see all available commands and options:

.. code-block:: bash

   # Show all available commands
   nvcomms-perf --help

   # Show options for a specific command
   nvcomms-perf <command> --help
   # For example:
   nvcomms-perf generate-job-script --help
   nvcomms-perf convert --help
   nvcomms-perf upload --help

Workflow Overview
~~~~~~~~~~~~~~~~~

The typical workflow for using Nvidia Comms Perf Suite is:

1. **Generate** test scripts using ``generate-job-script``
2. **Run** the tests on your cluster
3. **Process** results using one of these options:

   - ``convert``: Save results to a local file (CSV, etc.)
   - ``upload``: Upload results directly to a database and/or S3

Step 1: Test Generation
~~~~~~~~~~~~~~~~~~~~~~~

Generate a job script for performance testing:

.. code-block:: bash

   # Generate a job script and print to stdout
   nvcomms-perf generate-job-script --system CLUSTER_1 --systems-toml systems.toml --user-toml user.toml --testset-toml testset.toml

   # Generate a job script and save to file
   nvcomms-perf generate-job-script --system CLUSTER_1 --systems-toml systems.toml --user-toml user.toml --testset-toml testset.toml --output job.sbatch

Step 2: Run Tests
~~~~~~~~~~~~~~~~~

Submit the generated job script to your cluster:

.. code-block:: bash

   # Submit the job
   sbatch job.sbatch

   # The job will produce test output files

Step 3: Process Results
~~~~~~~~~~~~~~~~~~~~~~~

Choose one of the following options to process your test results:

**Option A: Convert to Local File**

Use ``convert`` to save results to a local file:

.. code-block:: bash

   # Convert a single NCCL test output to a local CSV file
   nvcomms-perf convert -i nccl-tests:/path/to/nccl-tests.log -o results.csv

   # Convert test results from tarball to a local CSV file
   nvcomms-perf convert -i tarball:/path/to/archive.tar.gz -o results.csv

   # Convert multiple test result sources to a single local CSV file
   nvcomms-perf convert -i tarball:/path/to/archive1.tar.gz -i tarball:/path/to/archive2.tar.gz -o combined_results.csv

**Option B: Upload to Database/S3**

Use ``upload`` to directly store results:

.. code-block:: bash

   # Upload to PostgreSQL database
   nvcomms-perf upload -i tarball:/path/to/archive.tar.gz --db-url postgresql+psycopg://user:pass@host:port/database

   # Upload to S3
   nvcomms-perf upload -i tarball:/path/to/archive.tar.gz --s3-bucket my-bucket --s3-region us-east-1

   # Upload to both PostgreSQL and S3
   nvcomms-perf upload -i tarball:/path/to/archive.tar.gz \
      --db-url postgresql+psycopg://user:pass@host:port/database \
      --s3-bucket my-bucket --s3-region us-east-1

Next Steps
----------

1. Read the detailed documentation for each component:
   - `Test Generation <testing_automation/README.html>`_
   - `Convert Command <convert/README.html>`_
   - `Upload Command <upload/README.html>`_
   - `Legacy Parse and Upload <parse_and_upload/README.html>`_ (Deprecated)

2. Check out the API reference for more advanced usage:
   - `Testing Automation API <api/testing_automation/api.html>`_
   - `Parse and Upload API <api/parse_and_upload/api.html>`_

3. If you want to contribute, read the `Contributing Guide <contributing.html>`_.
