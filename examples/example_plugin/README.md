# Nvidia Comms Perf Suite Plugin Example

This directory contains an example plugin for Nvidia Comms Perf Suite, demonstrating how to extend the tool with custom commands.

## Plugin System Overview

Nvidia Comms Perf Suite supports a plugin system similar to pytest, allowing you to:
- Create custom commands that integrate seamlessly with the main CLI
- Distribute plugins as separate packages
- Automatically discover and load plugins via entry points

## Creating a Plugin

### 1. Plugin Structure

Your plugin should:
- Inherit from `nvcomms_perf.plugins.NvCommsPerfPlugin`
- Implement required properties and methods
- Be packaged with proper entry points

### 2. Basic Plugin Example

```python
from nvcomms_perf.plugins import NvCommsPerfPlugin
import typer

class MyPlugin(NvCommsPerfPlugin):
    @property
    def name(self) -> str:
        return "my-command"  # This becomes the CLI command name

    @property
    def help(self) -> str:
        return "My custom plugin"

    def get_command(self) -> typer.Typer:
        # Return either a Typer app (for subcommands) or a function
        app = typer.Typer()

        @app.command()
        def hello(name: str = "World"):
            print(f"Hello {name}!")

        return app
```

### 3. Plugin Setup

Create a `setup.py` or `pyproject.toml` with entry points:

```python
# setup.py
setup(
    name="nvcomms-perf-my-plugin",
    entry_points={
        "nvcomms_perf": [
            "my-plugin = my_plugin_module:MyPlugin",
        ],
    },
)
```

Or with `pyproject.toml`:

```toml
[project.entry-points.nvcomms_perf]
my-plugin = "my_plugin_module:MyPlugin"
```

## Installing the Example Plugin

From this directory:

```bash
# Install in development mode
pip install -e .

# Or install normally
pip install .
```

## Using the Plugin

Once installed, the plugin commands are automatically available:

```bash
# View all commands including plugins
nvcomms-perf --help

# Use the example plugin
nvcomms-perf example hello "NCP User"
nvcomms-perf example hello "NCP User" --excited

# Process a file
nvcomms-perf example process input.txt
nvcomms-perf example process input.txt -o output.txt --uppercase
```

## Plugin Types

### 1. Multi-Command Plugin (Recommended)

Create a plugin that provides multiple subcommands:

```python
def get_command(self) -> typer.Typer:
    app = typer.Typer()
    app.command()(self.cmd1)
    app.command()(self.cmd2)
    return app
```

### 2. Single Command Plugin

Create a plugin that provides a single command:

```python
def get_command(self) -> Callable:
    def my_command(arg: str):
        print(f"Processing {arg}")
    return my_command
```

## Best Practices

1. **Naming**: Use descriptive command names that don't conflict with existing commands
2. **Documentation**: Provide clear help text for all commands and options
3. **Error Handling**: Handle errors gracefully and provide helpful error messages
4. **Logging**: Use the standard logging module for debug output
5. **Dependencies**: Declare all dependencies in your plugin's setup configuration

## Debugging Plugins

Enable verbose mode to see plugin loading information:

```bash
nvcomms-perf --verbose example hello
```

This will show:
- Which plugins were discovered
- Which plugins were successfully loaded
- Any errors during plugin loading
