# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

"""Plugin loader for Nvidia Comms Perf Suite."""

import logging
from typing import List, Dict, Optional, Tuple
from importlib.metadata import entry_points
import typer

from . import NvCommsPerfPlugin

logger = logging.getLogger(__name__)

# Entry point group name for nvcomms-perf plugins
ENTRY_POINT_GROUP = "nvcomms_perf"


class PluginLoader:
    """Discovers and loads plugins for Nvidia Comms Perf Suite."""

    def __init__(self):
        self.plugins: Dict[str, NvCommsPerfPlugin] = {}
        self.failed_plugins: List[Tuple[str, Exception]] = []

    def discover_plugins(self) -> List[str]:
        """Discover available plugins through entry points.

        Returns:
            List of discovered plugin names.
        """
        discovered = []

        # Get all entry points in the nvcomms_perf group
        eps = entry_points()
        if hasattr(eps, "select"):
            # Python 3.10+
            plugin_eps = eps.select(group=ENTRY_POINT_GROUP)  # type: ignore[attr-defined]
        else:
            # Python 3.9
            plugin_eps = eps.get(ENTRY_POINT_GROUP, [])

        for ep in plugin_eps:
            discovered.append(ep.name)
            logger.debug(f"Discovered plugin: {ep.name} from {ep.value}")

        return discovered

    def load_plugin(self, plugin_name: str) -> Optional[NvCommsPerfPlugin]:
        """Load a specific plugin by name.

        Args:
            plugin_name: Name of the plugin to load.

        Returns:
            The loaded plugin instance, or None if loading failed.
        """
        try:
            # Get the entry point
            eps = entry_points()
            if hasattr(eps, "select"):
                plugin_eps = eps.select(group=ENTRY_POINT_GROUP)  # type: ignore[attr-defined]
            else:
                plugin_eps = eps.get(ENTRY_POINT_GROUP, [])

            for ep in plugin_eps:
                if ep.name == plugin_name:
                    # Load the plugin class
                    plugin_class = ep.load()

                    # Ensure it's a subclass of NvCommsPerfPlugin
                    if not issubclass(plugin_class, NvCommsPerfPlugin):
                        raise TypeError(f"Plugin {plugin_name} must inherit from NvCommsPerfPlugin")

                    # Instantiate the plugin
                    plugin = plugin_class()

                    # Initialize the plugin
                    plugin.initialize()

                    logger.info(f"Successfully loaded plugin: {plugin_name}")
                    return plugin

            logger.warning(f"Plugin {plugin_name} not found in entry points")
            return None

        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")
            self.failed_plugins.append((plugin_name, e))
            return None

    def load_all_plugins(self) -> Dict[str, NvCommsPerfPlugin]:
        """Load all discovered plugins.

        Returns:
            Dictionary mapping plugin names to plugin instances.
        """
        plugin_names = self.discover_plugins()

        for name in plugin_names:
            plugin = self.load_plugin(name)
            if plugin:
                self.plugins[plugin.name] = plugin

        if self.failed_plugins:
            logger.warning(
                f"Failed to load {len(self.failed_plugins)} plugin(s): "
                f"{', '.join(name for name, _ in self.failed_plugins)}"
            )

        return self.plugins

    def register_plugin_commands(self, app: typer.Typer) -> None:
        """Register all loaded plugin commands with the main Typer app.

        Args:
            app: The main Typer application instance.
        """
        for plugin_name, plugin in self.plugins.items():
            try:
                command = plugin.get_command()

                # Register the command with the app
                if isinstance(command, typer.Typer):
                    # It's a Typer app, add it as a subcommand
                    app.add_typer(command, name=plugin.name, help=plugin.help)
                else:
                    # It's a function, add it as a command
                    app.command(name=plugin.name, help=plugin.help)(command)

                logger.debug(f"Registered command for plugin: {plugin_name}")

            except Exception as e:
                logger.error(f"Failed to register command for plugin {plugin_name}: {e}")


# Global plugin loader instance
_loader: Optional[PluginLoader] = None


def get_plugin_loader() -> PluginLoader:
    """Get the global plugin loader instance."""
    global _loader
    if _loader is None:
        _loader = PluginLoader()
    return _loader


def load_and_register_plugins(app: typer.Typer) -> None:
    """Convenience function to load all plugins and register their commands.

    Args:
        app: The main Typer application instance.
    """
    loader = get_plugin_loader()
    loader.load_all_plugins()
    loader.register_plugin_commands(app)
