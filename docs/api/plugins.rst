..
   Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
   See LICENSE.txt for license information

Plugin System API Reference
===========================

.. module:: nvcomms_perf.plugins

This module provides the plugin system for nvcomms-perf.

Base Classes
------------

.. autoclass:: nvcomms_perf.plugins.NvCommsPerfPlugin
   :members:
   :undoc-members:
   :show-inheritance:

Plugin Loader
-------------

.. module:: nvcomms_perf.plugins.loader

.. autoclass:: nvcomms_perf.plugins.loader.PluginLoader
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: nvcomms_perf.plugins.loader.get_plugin_loader

.. autofunction:: nvcomms_perf.plugins.loader.load_and_register_plugins

Entry Points
------------

nvcomms-perf uses Python entry points for plugin discovery. Plugins should be registered under the ``nvcomms_perf`` entry point group.

Example entry point configuration in ``setup.py``:

.. code-block:: python

    setup(
        # ... other setup parameters ...
        entry_points={
            "nvcomms_perf": [
                "my-plugin = my_plugin_module:MyPluginClass",
            ],
        },
    )

Example entry point configuration in ``pyproject.toml``:

.. code-block:: toml

    [project.entry-points.nvcomms_perf]
    my-plugin = "my_plugin_module:MyPluginClass"
