..
   Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
   See LICENSE.txt for license information

Upload Command
==============

Overview
--------

The ``upload`` command uploads test results to databases (PostgreSQL, MySQL, SQLite) and/or S3 cloud storage. It provides a flexible way to store and manage performance test data for analysis and long-term storage.

Key Features
~~~~~~~~~~~~

- Upload to multiple storage backends simultaneously
- Support for PostgreSQL, MySQL, SQLite databases
- S3-compatible object storage support
- Automatic deduplication to avoid duplicate uploads
- Progress tracking with detailed status updates
- Parallel processing for improved performance
- Environment variable support for credentials

Quick Start
-----------

Upload test results to various storage backends:

.. code-block:: bash

   # Upload to PostgreSQL database
   nvcomms-perf upload -i tarball:/path/to/test-results.tar.gz --db-url postgresql+psycopg://user:pass@host:port/database

   # Upload to S3
   nvcomms-perf upload -i tarball:/path/to/test-results.tar.gz --s3-bucket my-bucket --s3-region us-east-1

   # Upload to both PostgreSQL and S3
   nvcomms-perf upload -i tarball:/path/to/test-results.tar.gz \
       --db-url postgresql+psycopg://user:pass@host:port/database \
       --s3-bucket my-bucket --s3-region us-east-1

   # Upload multiple files
   nvcomms-perf upload -i tarball:/path/to/archive1.tar.gz -i tarball:/path/to/archive2.tar.gz \
       --db-url postgresql+psycopg://user:pass@host:port/database

Command-line Interface
----------------------

Options
~~~~~~~

**Input Options:**

- ``-i, --input``: Input file(s) to upload. Supported types: nccl-tests, tarball [required]

**Database Options:**

- ``--db-url``: Database URL (e.g., postgresql+psycopg://user:pass@host:port/db) [env: NVCOMMS_PERF_DB_URL]
- ``--db-schema``: Database schema to use (PostgreSQL only) [env: NVCOMMS_PERF_DB_SCHEMA]

**S3 Options:**

- ``--s3-bucket``: S3 bucket name [env: NVCOMMS_PERF_S3_BUCKET]
- ``--s3-region``: S3 region [env: NVCOMMS_PERF_S3_REGION]
- ``--s3-endpoint``: S3 endpoint URL (for S3-compatible services) [env: NVCOMMS_PERF_S3_ENDPOINT]
- ``--s3-access-key``: S3 access key ID [env: NVCOMMS_PERF_S3_ACCESS_KEY or AWS_ACCESS_KEY_ID]
- ``--s3-secret-key``: S3 secret access key [env: NVCOMMS_PERF_S3_SECRET_KEY or AWS_SECRET_ACCESS_KEY]

**Processing Options:**

- ``--max-workers``: Maximum number of parallel workers for parsing
- ``-p, --show-progress / -P, --hide-progress``: Show/hide progress bar [default: show]
- ``-c, --continue-on-error / -C, --stop-on-error``: Continue processing if a file fails [default: continue]

Database Configuration
----------------------

PostgreSQL
~~~~~~~~~~

.. code-block:: bash

   # Basic PostgreSQL connection
   nvcomms-perf upload -i tarball:/path/to/results.tar.gz \
       --db-url postgresql+psycopg://user:password@localhost:5432/perfdb

   # With specific schema
   nvcomms-perf upload -i tarball:/path/to/results.tar.gz \
       --db-url postgresql+psycopg://user:password@localhost:5432/perfdb \
       --db-schema performance_data

MySQL
~~~~~

.. code-block:: bash

   # MySQL connection
   nvcomms-perf upload -i tarball:/path/to/results.tar.gz \
       --db-url mysql://user:password@localhost:3306/perfdb

SQLite
~~~~~~

.. code-block:: bash

   # SQLite local file
   nvcomms-perf upload -i tarball:/path/to/results.tar.gz \
       --db-url sqlite:///path/to/database.db

   # SQLite in-memory (for testing)
   nvcomms-perf upload -i tarball:/path/to/results.tar.gz \
       --db-url sqlite:///:memory:

S3 Configuration
----------------

AWS S3
~~~~~~

.. code-block:: bash

   # Using command-line options
   nvcomms-perf upload -i tarball:/path/to/results.tar.gz \
       --s3-bucket my-perf-data \
       --s3-region us-east-1 \
       --s3-access-key AKIAIOSFODNN7EXAMPLE \
       --s3-secret-key wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

   # Using environment variables (recommended)
   export NVCOMMS_PERF_S3_BUCKET="my-perf-data"
   export NVCOMMS_PERF_S3_REGION="us-east-1"
   export AWS_ACCESS_KEY_ID="AKIAIOSFODNN7EXAMPLE"
   export AWS_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
   nvcomms-perf upload -i tarball:/path/to/results.tar.gz

S3-Compatible Services
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # MinIO example
   nvcomms-perf upload -i tarball:/path/to/results.tar.gz \
       --s3-bucket perf-data \
       --s3-endpoint http://minio.example.com:9000 \
       --s3-access-key minioadmin \
       --s3-secret-key minioadmin

   # Ceph Object Gateway example
   nvcomms-perf upload -i tarball:/path/to/results.tar.gz \
       --s3-bucket perf-data \
       --s3-endpoint http://ceph-gateway.example.com:8080 \
       --s3-access-key your-access-key \
       --s3-secret-key your-secret-key

Advanced Usage
--------------

Deduplication
~~~~~~~~~~~~~

The upload command automatically checks for duplicate files:

1. **File-level deduplication**: Identical files are detected and skipped
2. **Database deduplication**: Files already in the database are skipped
3. **Hash-based detection**: Uses file content hash for accurate detection

.. code-block:: bash

   # Upload will skip files that already exist in the database
   nvcomms-perf upload -i tarball:/path/to/results.tar.gz \
       --db-url postgresql+psycopg://user:pass@host/db

   # Output will show:
   # ⚠ file.tar.gz already exists in the database

Parallel Processing
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Use custom number of workers for large uploads
   nvcomms-perf upload -i tarball:/path/to/large-dataset.tar.gz \
       --db-url postgresql+psycopg://user:pass@host/db \
       --max-workers 16

   # Limit workers for resource-constrained environments
   nvcomms-perf upload -i tarball:/path/to/results.tar.gz \
       --db-url postgresql+psycopg://user:pass@host/db \
       --max-workers 2

Error Handling
~~~~~~~~~~~~~~

.. code-block:: bash

   # Stop on first error (default: continue)
   nvcomms-perf upload -i tarball:/path/to/results.tar.gz \
       --db-url postgresql+psycopg://user:pass@host/db \
       --stop-on-error

   # Continue processing other files if one fails (default)
   nvcomms-perf upload -i tarball:/path/to/results.tar.gz \
       --db-url postgresql+psycopg://user:pass@host/db \
       --continue-on-error

Combined Storage
~~~~~~~~~~~~~~~~

Upload to both database and S3 for redundancy:

.. code-block:: bash

   # Upload to both PostgreSQL and S3
   nvcomms-perf upload -i tarball:/path/to/results.tar.gz \
       --db-url postgresql+psycopg://user:pass@host:5432/perfdb \
       --db-schema test_results \
       --s3-bucket perf-data-backup \
       --s3-region us-west-2

   # The database will store S3 paths for reference
   # raw_data_path: s3://perf-data-backup/uuid/filename.tar.gz

Environment Variables
---------------------

The upload command supports environment variables for all options:

.. code-block:: bash

   # Database configuration
   export NVCOMMS_PERF_DB_URL="postgresql+psycopg://user:pass@host:5432/perfdb"
   export NVCOMMS_PERF_DB_SCHEMA="performance"

   # S3 configuration
   export NVCOMMS_PERF_S3_BUCKET="my-perf-data"
   export NVCOMMS_PERF_S3_REGION="us-east-1"
   export NVCOMMS_PERF_S3_ENDPOINT="https://s3.amazonaws.com"
   export NVCOMMS_PERF_S3_ACCESS_KEY="your-access-key"
   export NVCOMMS_PERF_S3_SECRET_KEY="your-secret-key"

   # Or use standard AWS environment variables
   export AWS_ACCESS_KEY_ID="your-access-key"
   export AWS_SECRET_ACCESS_KEY="your-secret-key"

   # Now upload without specifying credentials
   nvcomms-perf upload -i tarball:/path/to/results.tar.gz

Upload Summary
--------------

After upload completes, you'll see a detailed summary:

.. code-block:: text

   ==================================================
   UPLOAD SUMMARY
   ==================================================
   Database: postgresql+psycopg://user:****@localhost:5432/perfdb
   Total files processed: 5
   Total files uploaded to S3: 5
   Total records to upload to database: 150

   Processed Files:

     File: test-results-001.tar.gz
     Sweep ID: 123e4567-e89b-12d3-a456-426614174000
     Records: 30
     S3 Path: s3://my-bucket/123e4567-e89b-12d3-a456-426614174000/test-results-001.tar.gz

     File: test-results-002.tar.gz
     Sweep ID: 123e4567-e89b-12d3-a456-426614174001
     Records: 30
     S3 Path: s3://my-bucket/123e4567-e89b-12d3-a456-426614174001/test-results-002.tar.gz

Best Practices
--------------

1. **Use environment variables** for credentials instead of command-line options
2. **Enable deduplication** by always using the same database for uploads
3. **Use S3 for raw data** and database for structured/searchable data
4. **Set appropriate worker counts** based on your system resources
5. **Monitor progress** with ``--show-progress`` for large uploads
6. **Use schemas** in PostgreSQL to organize data by project or team
