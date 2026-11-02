..
   Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
   See LICENSE.txt for license information

==================
Plugin Development
==================

This guide explains how to develop plugins for nvcomms-perf, which uses a plugin system similar to pytest.

Overview
========

nvcomms-perf plugins allow you to:

* Add new commands to the CLI
* Extend functionality without modifying the core codebase
* Distribute custom tools as separate packages
* Share functionality across teams

Quick Start
===========

1. Create a Simple Plugin
-------------------------

.. code-block:: python

    # my_plugin.py
    from nvcomms_perf.plugins import NvCommsPerfPlugin
    import typer

    class MyPlugin(NvCommsPerfPlugin):
        @property
        def name(self) -> str:
            return "my-command"  # The command name in the CLI

        @property
        def help(self) -> str:
            return "My custom plugin"

        def get_command(self):
            def my_command(name: str = "World"):
                """My custom command."""
                print(f"Hello {name}!")
            return my_command

2. Package Your Plugin
----------------------

Create a ``setup.py`` or ``pyproject.toml``:

The packaging examples below assume the framework is already installed from a
trusted checkout. Install that checkout explicitly before installing your plugin;
these examples do not fetch the framework by package name.

.. code-block:: python

    # setup.py
    from setuptools import setup

    setup(
        name="nvcomms-perf-my-plugin",
        version="0.1.0",
        py_modules=["my_plugin"],
        install_requires=["typer"],
        entry_points={
            "nvcomms_perf": [
                "my-plugin = my_plugin:MyPlugin",
            ],
        },
    )

Or with ``pyproject.toml``:

.. code-block:: toml

    [build-system]
    requires = ["setuptools", "wheel"]
    build-backend = "setuptools.build_meta"

    [project]
    name = "nvcomms-perf-my-plugin"
    version = "0.1.0"
    dependencies = ["typer"]

    [project.entry-points.nvcomms_perf]
    my-plugin = "my_plugin:MyPlugin"

3. Install and Use
------------------

.. code-block:: bash

    # Install the framework from your local checkout
    python -m pip install -e /path/to/framework-checkout

    # From your plugin directory, install the plugin
    python -m pip install -e .

    # Use the plugin
    nvcomms-perf my-command
    nvcomms-perf my-command "NCP User"

Plugin Architecture
===================

Base Class
----------

All plugins must inherit from ``NvCommsPerfPlugin``:

.. code-block:: python

    from abc import ABC, abstractmethod
    from typing import Union, Callable, Optional
    import typer

    class NvCommsPerfPlugin(ABC):
        @property
        @abstractmethod
        def name(self) -> str:
            """Plugin name used for the CLI command."""
            pass

        @property
        def help(self) -> Optional[str]:
            """Optional help text for the plugin command."""
            return None

        @abstractmethod
        def get_command(self) -> Union[typer.Typer, Callable]:
            """Return a Typer app or command function."""
            pass

        def initialize(self) -> None:
            """Optional initialization method."""
            pass

Plugin Types
------------

1. Single Command Plugin
~~~~~~~~~~~~~~~~~~~~~~~~

For simple plugins with one command:

.. code-block:: python

    class SimplePlugin(NvCommsPerfPlugin):
        @property
        def name(self) -> str:
            return "simple"

        def get_command(self):
            def simple_cmd(message: str):
                print(f"Message: {message}")
            return simple_cmd

2. Multi-Command Plugin
~~~~~~~~~~~~~~~~~~~~~~~

For plugins with multiple subcommands:

.. code-block:: python

    class ComplexPlugin(NvCommsPerfPlugin):
        @property
        def name(self) -> str:
            return "complex"

        def get_command(self) -> typer.Typer:
            app = typer.Typer(help="Complex plugin with subcommands")

            @app.command()
            def create(name: str):
                """Create something."""
                print(f"Creating {name}")

            @app.command()
            def delete(name: str):
                """Delete something."""
                print(f"Deleting {name}")

            return app

Usage:

.. code-block:: bash

    nvcomms-perf complex create myfile
    nvcomms-perf complex delete myfile

Advanced Features
=================

1. Using Typer Features
-----------------------

Leverage all Typer features in your plugins:

.. code-block:: python

    from typing import Annotated, Optional
    import typer

    class AdvancedPlugin(NvCommsPerfPlugin):
        @property
        def name(self) -> str:
            return "advanced"

        def get_command(self):
            def advanced_cmd(
                input_file: Annotated[typer.FileText, typer.Argument(help="Input file")],
                output: Annotated[Optional[typer.FileTextWrite], typer.Option("-o", "--output")] = None,
                verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
                count: Annotated[int, typer.Option("--count", "-c", min=1, max=100)] = 1
            ):
                """Advanced command with various options."""
                content = input_file.read()

                if verbose:
                    print(f"Processing {input_file.name}...")

                for i in range(count):
                    if output:
                        output.write(f"[{i+1}] {content}")
                    else:
                        print(f"[{i+1}] {content}")

            return advanced_cmd

2. Plugin Initialization
------------------------

Use the ``initialize`` method for setup tasks:

.. code-block:: python

    import logging

    class InitializedPlugin(NvCommsPerfPlugin):
        def initialize(self):
            """Setup logging, load config, etc."""
            self.logger = logging.getLogger(f"nvcomms_perf.plugins.{self.name}")
            self.logger.info("Plugin initialized")

            # Load configuration
            self.config = self.load_config()

        def load_config(self):
            # Load plugin-specific configuration
            return {}

3. Error Handling
-----------------

Implement robust error handling:

.. code-block:: python

    class RobustPlugin(NvCommsPerfPlugin):
        @property
        def name(self) -> str:
            return "robust"

        def get_command(self):
            def robust_cmd(path: str):
                """Command with error handling."""
                try:
                    # Your logic here
                    process_path(path)
                except FileNotFoundError:
                    print(f"Error: File '{path}' not found")
                    raise typer.Exit(1)
                except PermissionError:
                    print(f"Error: Permission denied for '{path}'")
                    raise typer.Exit(2)
                except Exception as e:
                    print(f"Unexpected error: {e}")
                    raise typer.Exit(3)

            return robust_cmd

Testing Plugins
===============

Unit Tests
----------

.. code-block:: python

    # test_my_plugin.py
    import pytest
    from unittest.mock import Mock, patch
    from my_plugin import MyPlugin

    def test_plugin_properties():
        plugin = MyPlugin()
        assert plugin.name == "my-command"
        assert plugin.help == "My custom plugin"

    def test_plugin_command():
        plugin = MyPlugin()
        command = plugin.get_command()

        # Test the command function
        with patch('builtins.print') as mock_print:
            command("Test")
            mock_print.assert_called_with("Hello Test!")

Integration Tests
-----------------

Test your plugin with the actual CLI:

.. code-block:: python

    # test_integration.py
    import subprocess

    def test_plugin_cli():
        result = subprocess.run(
            ["nvcomms-perf", "my-command", "Test"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "Hello Test!" in result.stdout

Best Practices
==============

1. Naming Conventions
---------------------

* Use descriptive command names
* Avoid conflicts with existing commands
* Follow kebab-case for command names

2. Documentation
----------------

Always provide:

* Clear docstrings for commands
* Help text for all options
* Usage examples in your README

3. Dependencies
---------------

* Minimize external dependencies
* Specify version constraints
* Test with different nvcomms-perf versions

4. Logging
----------

Use the standard logging module:

.. code-block:: python

    import logging

    class LoggingPlugin(NvCommsPerfPlugin):
        def get_command(self):
            def cmd():
                logger = logging.getLogger(__name__)
                logger.info("Command started")
                # Your logic here
                logger.info("Command completed")
            return cmd

5. Configuration
----------------

Support configuration files:

.. code-block:: python

    from pathlib import Path
    import yaml

    class ConfigurablePlugin(NvCommsPerfPlugin):
        def get_command(self):
            def cmd(config: Path = None):
                if config:
                    with open(config) as f:
                        settings = yaml.safe_load(f)
                else:
                    settings = self.default_settings()

                # Use settings
            return cmd

Distribution
============

PyPI
----

Publish to PyPI for public plugins:

.. code-block:: bash

    # Build
    python -m build

    # Upload
    python -m twine upload dist/*

Private Registry
----------------

For internal plugins:

.. code-block:: bash

    # Upload to private registry
    twine upload --repository-url https://your-registry.com dist/*

    # Install from private registry
    pip install nvcomms-perf-my-plugin --index-url https://your-registry.com

Debugging
=========

Enable Verbose Mode
-------------------

.. code-block:: bash

    nvcomms-perf --verbose my-command

Plugin Loading Issues
---------------------

If your plugin isn't loading:

1. Check entry points are correct
2. Verify the plugin is installed
3. Check for import errors
4. Enable verbose mode to see loading errors

Common Issues
-------------

1. **Import errors**: Ensure all dependencies are installed
2. **Name conflicts**: Check no other plugin uses the same name
3. **Type errors**: Verify your plugin implements all required methods
4. **Entry point errors**: Double-check the entry point syntax

Examples
========

See the examples directory in the nvcomms-perf repository for complete working examples:

* ``examples/minimal_plugin.py``: Simplest possible plugin

The separately packaged example plugin has been removed. Use the packaging
instructions above to register your own plugin.

Support
=======

For help with plugin development:

* Check the :doc:`API documentation <api/plugins>`
* Review existing plugins for examples
* Open an issue for questions or bugs
