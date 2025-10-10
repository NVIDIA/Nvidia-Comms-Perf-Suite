Contributing to nvcomms-perf
============================

Thank you for your interest in contributing to nvcomms-perf! This document provides guidelines and instructions for contributing to the project.

Development Environment Setup
-----------------------------

We recommend using `uv` for Python package management as it's faster and more reliable than traditional tools.

1. **Install uv**:

   .. code-block:: bash

      # On macOS/Linux
      curl -LsSf https://astral.sh/uv/install.sh | sh

      # On Windows (PowerShell)
      iwr https://astral.sh/uv/install.ps1 -useb | iex

2. **Clone the repository**:

   .. code-block:: bash

      git clone <insert url>
      cd nvcomms-perf

3. **Create a virtual environment and install dependencies**:

   .. code-block:: bash

      # Create a virtual environment
      uv venv

      # Install dependencies including development tools
      uv pip install -e ".[dev]"

Code Style and Linting
----------------------

We use `ruff` for linting and code formatting. To check your code:

.. code-block:: bash

   # Check code style
   uv run ruff check src/nvcomms_perf/

   # Format code
   uv run ruff format src/nvcomms_perf/

Running Tests
-------------

We use `pytest` for testing. To run the tests:

.. code-block:: bash

   # Run all tests
   uv run pytest tests/

   # Run tests with coverage
   uv run pytest --cov=src/nvcomms_perf tests/

   # Run tests for a specific module
   uv run pytest tests/testing_automation/

Building the Package
--------------------

To build the package locally:

.. code-block:: bash

   # Build the package
   uv build

   # This will create distribution files in the dist/ directory

Building Documentation
----------------------

We use Sphinx for documentation. To build the docs:

.. code-block:: bash

   # Install documentation dependencies
   uv pip install -e ".[docs]"

   # Build the documentation
   cd docs
   make html

   # The documentation will be available in docs/_build/html/

Submitting Changes
------------------

1. **Create a branch** for your changes:

   .. code-block:: bash

      git checkout -b your-name/your-feature-name

2. **Make your changes** and commit them:

   .. code-block:: bash

      git add .
      git commit -m "Description of your changes"

3. **Push your branch** and create a merge request:

   .. code-block:: bash

      git push origin your-name/your-feature-name

4. **Create a merge request** on Github and wait for review.

Code Review Process
-------------------

1. All code changes must go through a code review process.
2. At least one reviewer must approve your changes.
3. All tests must pass before merging.
4. Documentation should be updated to reflect any changes.

Release Process
---------------

1. Update the changelog in `docs/changelog.rst`.
2. Create a tag for the release, e.g. `v0.1.0`.
3. The CI/CD pipeline will automatically build and publish the package.

Getting Help
------------

If you have questions or need help, please:

1. Check the existing documentation.
2. Open an issue on Github.
3. Contact the maintainers directly.

Thank you for contributing to the Nvidia Comms Perf Suite! 
