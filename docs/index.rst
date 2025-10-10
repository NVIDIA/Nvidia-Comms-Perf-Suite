..
   Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
   See LICENSE.txt for license information

Welcome to Nvidia Comms Perf Suite's documentation!
===================================================

Nvidia Comms Perf Suite is a comprehensive toolkit for GPU performance testing and analysis. It follows a modular workflow:

1. **Test Generation**: Generate test scripts for running benchmarks across different systems and configurations
2. **Result Processing**: Process test results using one of these options:

   - **Convert**: Save results to structured local files (CSV, etc.) for analysis
   - **Upload**: Store results directly in databases or cloud storage (PostgreSQL, MySQL, SQLite, S3)

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started

.. toctree::
   :maxdepth: 2
   :caption: User Guides

   testing_automation/README
   convert/README
   upload/README
   report/README
   parse_and_upload/README
   plugin_development

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/testing_automation/api
   api/report/api
   api/parse_and_upload/api
   api/plugins

.. toctree::
   :maxdepth: 2
   :caption: Contributing

   contributing

.. toctree::
   :maxdepth: 2
   :caption: Changelog

   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
