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

"""Tests for the nvcomms-perf plugin system."""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Type
import typer

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from nvcomms_perf.plugins import NvCommsPerfPlugin
from nvcomms_perf.plugins.loader import PluginLoader, ENTRY_POINT_GROUP


# Test plugin implementations
class TestPlugin(NvCommsPerfPlugin):
    """Test plugin for unit tests."""

    @property
    def name(self) -> str:
        return "test-plugin"

    @property
    def help(self) -> str:
        return "Test plugin"

    def get_command(self) -> typer.Typer:
        app = typer.Typer()

        @app.command()
        def test_cmd():
            print("Test command executed")

        return app


class TestSimplePlugin(NvCommsPerfPlugin):
    """Test plugin with single command."""

    @property
    def name(self) -> str:
        return "test-simple"

    def get_command(self):
        def simple_cmd(message: str):
            print(f"Simple: {message}")
        return simple_cmd


class InvalidPlugin:
    """Invalid plugin that doesn't inherit from NvCommsPerfPlugin."""
    pass


# Fixtures
@pytest.fixture
def plugin_loader():
    """Create a fresh plugin loader instance."""
    return PluginLoader()


@pytest.fixture
def test_plugin():
    """Create a test plugin instance."""
    return TestPlugin()


@pytest.fixture
def simple_plugin():
    """Create a simple plugin instance."""
    return TestSimplePlugin()


@pytest.fixture
def mock_entry_point(request):
    """Create a mock entry point with configurable properties."""
    ep = Mock()
    ep.name = getattr(request, 'param', {}).get('name', 'test')
    ep.load.return_value = getattr(request, 'param', {}).get('plugin_class', TestPlugin)
    ep.value = f"{ep.load.return_value.__module__}:{ep.load.return_value.__name__}"
    return ep


@pytest.fixture
def mock_entry_points():
    """Mock the entry_points function."""
    with patch('nvcomms_perf.plugins.loader.entry_points') as mock_eps:
        yield mock_eps


# Plugin Loader Tests
def test_plugin_loader_initialization(plugin_loader):
    """Test that plugin loader initializes with empty state."""
    assert plugin_loader.plugins == {}
    assert plugin_loader.failed_plugins == []


def test_load_plugin_success(plugin_loader, mock_entry_points):
    """Test successful plugin loading."""
    # Create a mock entry point
    mock_ep = Mock()
    mock_ep.name = "test"
    mock_ep.load.return_value = TestPlugin

    mock_entry_points.return_value.select.return_value = [mock_ep]

    plugin = plugin_loader.load_plugin("test")

    assert plugin is not None
    assert isinstance(plugin, TestPlugin)
    assert plugin.name == "test-plugin"


def test_load_plugin_not_found(plugin_loader, mock_entry_points):
    """Test loading a plugin that doesn't exist."""
    mock_entry_points.return_value.select.return_value = []

    plugin = plugin_loader.load_plugin("nonexistent")

    assert plugin is None


def test_load_plugin_invalid_class(plugin_loader, mock_entry_points):
    """Test loading a plugin that doesn't inherit from NvCommsPerfPlugin."""
    mock_ep = Mock()
    mock_ep.name = "invalid"
    mock_ep.load.return_value = InvalidPlugin

    mock_entry_points.return_value.select.return_value = [mock_ep]

    plugin = plugin_loader.load_plugin("invalid")

    assert plugin is None
    assert len(plugin_loader.failed_plugins) == 1
    assert plugin_loader.failed_plugins[0][0] == "invalid"
    assert isinstance(plugin_loader.failed_plugins[0][1], TypeError)


def test_load_plugin_import_error(plugin_loader, mock_entry_points):
    """Test handling of import errors during plugin loading."""
    mock_ep = Mock()
    mock_ep.name = "broken"
    mock_ep.load.side_effect = ImportError("Module not found")

    mock_entry_points.return_value.select.return_value = [mock_ep]

    plugin = plugin_loader.load_plugin("broken")

    assert plugin is None
    assert len(plugin_loader.failed_plugins) == 1
    assert plugin_loader.failed_plugins[0][0] == "broken"
    assert isinstance(plugin_loader.failed_plugins[0][1], ImportError)


@pytest.mark.parametrize("plugin_names", [
    [],
    ["plugin1"],
    ["plugin1", "plugin2", "plugin3"],
])
def test_discover_plugins(plugin_loader, mock_entry_points, plugin_names):
    """Test plugin discovery with various numbers of plugins."""
    # Create mock entry points with proper name attributes
    mock_eps = []
    for name in plugin_names:
        ep = Mock()
        ep.name = name  # This sets the actual .name attribute
        mock_eps.append(ep)

    mock_entry_points.return_value.select.return_value = mock_eps

    discovered = plugin_loader.discover_plugins()

    assert len(discovered) == len(plugin_names)
    assert discovered == plugin_names


def test_discover_plugins_python39_compat(plugin_loader, mock_entry_points):
    """Test plugin discovery using Python 3.9 API."""
    # Simulate Python 3.9 where entry_points() returns a dict-like object
    mock_eps_dict = Mock()
    del mock_eps_dict.select  # Remove select method to simulate Python 3.9

    plugin_names = ["plugin1", "plugin2"]
    mock_eps = []
    for name in plugin_names:
        ep = Mock()
        ep.name = name
        mock_eps.append(ep)

    mock_eps_dict.get.return_value = mock_eps

    mock_entry_points.return_value = mock_eps_dict

    discovered = plugin_loader.discover_plugins()

    assert len(discovered) == 2
    assert discovered == plugin_names
    mock_eps_dict.get.assert_called_once_with(ENTRY_POINT_GROUP, [])


@pytest.mark.parametrize("plugin_class,expected_method", [
    (TestPlugin, "add_typer"),
    (TestSimplePlugin, "command"),
])
def test_register_plugin_commands(plugin_loader, plugin_class, expected_method):
    """Test registering different types of plugin commands."""
    plugin = plugin_class()
    plugin_loader.plugins[plugin.name] = plugin

    mock_app = MagicMock(spec=typer.Typer)

    plugin_loader.register_plugin_commands(mock_app)

    # Check that the appropriate registration method was called
    if expected_method == "add_typer":
        mock_app.add_typer.assert_called_once()
        call_kwargs = mock_app.add_typer.call_args[1]
        assert call_kwargs['name'] == plugin.name
        assert call_kwargs['help'] == plugin.help
    else:
        mock_app.command.assert_called_once_with(
            name=plugin.name,
            help=plugin.help
        )


def test_register_plugin_commands_error_handling(plugin_loader, caplog):
    """Test error handling during command registration."""
    # Create a plugin that raises an error when getting command
    mock_plugin = Mock(spec=NvCommsPerfPlugin)
    mock_plugin.name = "broken"
    mock_plugin.get_command.side_effect = RuntimeError("Command creation failed")

    plugin_loader.plugins["broken"] = mock_plugin

    mock_app = Mock(spec=typer.Typer)

    # Should not raise, but log the error
    plugin_loader.register_plugin_commands(mock_app)

    assert "Failed to register command for plugin broken" in caplog.text


def test_load_all_plugins(plugin_loader, mock_entry_points):
    """Test loading all discovered plugins."""
    # Create mock entry points for different plugin types
    mock_ep1 = Mock()
    mock_ep1.name = "test1"
    mock_ep1.load.return_value = TestPlugin

    mock_ep2 = Mock()
    mock_ep2.name = "test2"
    mock_ep2.load.return_value = TestSimplePlugin

    mock_entry_points.return_value.select.return_value = [mock_ep1, mock_ep2]

    plugins = plugin_loader.load_all_plugins()

    assert len(plugins) == 2
    assert "test-plugin" in plugins  # Note: keyed by plugin.name, not entry point name
    assert "test-simple" in plugins
    assert isinstance(plugins["test-plugin"], TestPlugin)
    assert isinstance(plugins["test-simple"], TestSimplePlugin)


def test_load_all_plugins_with_failures(plugin_loader, mock_entry_points, caplog):
    """Test loading plugins when some fail to load."""
    # Create proper mock entry points
    mock_ep_good = Mock()
    mock_ep_good.name = "good"
    mock_ep_good.load.return_value = TestPlugin

    mock_ep_bad = Mock()
    mock_ep_bad.name = "bad"
    mock_ep_bad.load.side_effect = ImportError("Failed")

    mock_ep_invalid = Mock()
    mock_ep_invalid.name = "invalid"
    mock_ep_invalid.load.return_value = InvalidPlugin

    mock_entry_points.return_value.select.return_value = [mock_ep_good, mock_ep_bad, mock_ep_invalid]

    plugins = plugin_loader.load_all_plugins()

    # Only the good plugin should be loaded
    assert len(plugins) == 1
    assert "test-plugin" in plugins

    # Check that failures were logged
    assert len(plugin_loader.failed_plugins) == 2
    assert "Failed to load 2 plugin(s)" in caplog.text


# Plugin Base Class Tests
def test_plugin_abc_cannot_instantiate():
    """Test that NvCommsPerfPlugin cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        NvCommsPerfPlugin()


def test_plugin_abc_must_implement_name():
    """Test that plugins must implement the name property."""
    class IncompletePlugin(NvCommsPerfPlugin):
        def get_command(self):
            return lambda: None

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompletePlugin()


def test_plugin_abc_must_implement_get_command():
    """Test that plugins must implement the get_command method."""
    class IncompletePlugin(NvCommsPerfPlugin):
        @property
        def name(self):
            return "incomplete"

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompletePlugin()


def test_plugin_help_default(test_plugin):
    """Test that help property has correct default behavior."""
    # TestPlugin implements help, so it should return the value
    assert test_plugin.help == "Test plugin"

    # Create a plugin without help
    class MinimalPlugin(NvCommsPerfPlugin):
        @property
        def name(self):
            return "minimal"

        def get_command(self):
            return lambda: None

    minimal = MinimalPlugin()
    assert minimal.help is None  # Default value


def test_plugin_initialize(test_plugin, simple_plugin):
    """Test that initialize method can be called on plugins."""
    # Should not raise any exceptions
    test_plugin.initialize()
    simple_plugin.initialize()


# Integration Tests
@pytest.fixture
def populated_loader(plugin_loader, test_plugin, simple_plugin):
    """Create a loader with pre-loaded plugins."""
    plugin_loader.plugins = {
        test_plugin.name: test_plugin,
        simple_plugin.name: simple_plugin,
    }
    return plugin_loader


def test_full_plugin_lifecycle(mock_entry_points):
    """Test the complete plugin discovery, loading, and registration lifecycle."""
    # Setup mock entry points
    mock_ep1 = Mock()
    mock_ep1.name = "ep1"
    mock_ep1.load.return_value = TestPlugin

    mock_ep2 = Mock()
    mock_ep2.name = "ep2"
    mock_ep2.load.return_value = TestSimplePlugin

    mock_entry_points.return_value.select.return_value = [mock_ep1, mock_ep2]

    # Create loader and app
    loader = PluginLoader()
    app = MagicMock(spec=typer.Typer)

    # Load all plugins
    plugins = loader.load_all_plugins()
    assert len(plugins) == 2

    # Register commands
    loader.register_plugin_commands(app)

    # Verify both registration methods were called
    assert app.add_typer.call_count == 1  # TestPlugin returns Typer app
    assert app.command.call_count == 1     # TestSimplePlugin returns function


# Global Functions Tests
def test_get_plugin_loader_singleton():
    """Test that get_plugin_loader returns the same instance."""
    from nvcomms_perf.plugins.loader import get_plugin_loader, _loader

    # Reset global state
    import nvcomms_perf.plugins.loader
    nvcomms_perf.plugins.loader._loader = None

    loader1 = get_plugin_loader()
    loader2 = get_plugin_loader()

    assert loader1 is loader2
    assert nvcomms_perf.plugins.loader._loader is loader1


def test_load_and_register_plugins(mock_entry_points):
    """Test the convenience function for loading and registering plugins."""
    from nvcomms_perf.plugins.loader import load_and_register_plugins

    # Reset global state
    import nvcomms_perf.plugins.loader
    nvcomms_perf.plugins.loader._loader = None

    # Setup mocks
    mock_ep = Mock()
    mock_ep.name = "test"
    mock_ep.load.return_value = TestPlugin

    mock_entry_points.return_value.select.return_value = [mock_ep]

    app = MagicMock(spec=typer.Typer)

    # Call the convenience function
    load_and_register_plugins(app)

    # Verify plugin was loaded and registered
    assert app.add_typer.call_count == 1
