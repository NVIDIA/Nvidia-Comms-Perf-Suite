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

"""Example plugin for nvcomms-perf."""

import logging
from typing import Annotated, Optional, Union, Callable
import typer

# Import the base plugin class from nvcomms-perf
# When developing a real plugin, you would import this from the installed package
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src"))
from nvcomms_perf.plugins import NvCommsPerfPlugin

logger = logging.getLogger(__name__)


class ExamplePlugin(NvCommsPerfPlugin):
    """An example plugin demonstrating how to extend nvcomms-perf."""

    @property
    def name(self) -> str:
        """Plugin command name."""
        return "example"

    @property
    def help(self) -> str:
        """Plugin help text."""
        return "Run example plugin commands to demonstrate the plugin system"

    def initialize(self) -> None:
        """Initialize the plugin."""
        logger.info("Example plugin initialized")

    def get_command(self) -> typer.Typer:
        """Return the Typer app for this plugin."""
        # Create a sub-application for this plugin
        plugin_app = typer.Typer(
            help=self.help,
            no_args_is_help=True
        )

        # Add commands to the plugin app
        plugin_app.command()(self.hello_command)
        plugin_app.command(name="process")(self.process_command)

        return plugin_app

    def hello_command(
        self,
        name: Annotated[str, typer.Argument(help="Name to greet")] = "World",
        excited: Annotated[bool, typer.Option("--excited", "-e", help="Add excitement")] = False
    ) -> None:
        """Say hello from the example plugin."""
        greeting = f"Hello, {name}!"
        if excited:
            greeting += " 🎉"

        logger.info(f"Greeting generated: {greeting}")
        print(greeting)

    def process_command(
        self,
        input_file: Annotated[typer.FileText, typer.Argument(help="Input file to process")],
        output_file: Annotated[Optional[typer.FileTextWrite], typer.Option("-o", "--output", help="Output file")] = None,
        uppercase: Annotated[bool, typer.Option("--uppercase", "-u", help="Convert to uppercase")] = False
    ) -> None:
        """Process a file using the example plugin."""
        content = input_file.read()

        if uppercase:
            content = content.upper()

        if output_file:
            output_file.write(content)
            print(f"Processed content written to {output_file.name}")
        else:
            print(content)


# Alternative: Simple command plugin example
class SimpleExamplePlugin(NvCommsPerfPlugin):
    """A simpler plugin that provides just a single command."""

    @property
    def name(self) -> str:
        return "simple-example"

    @property
    def help(self) -> str:
        return "A simple single-command plugin"

    def get_command(self) -> Union[typer.Typer, Callable]:
        """Return a simple command function instead of a Typer app."""
        def simple_command(
            message: Annotated[str, typer.Argument(help="Message to display")],
            repeat: Annotated[int, typer.Option("--repeat", "-r", help="Number of times to repeat")] = 1
        ) -> None:
            """Simple command that prints a message."""
            for i in range(repeat):
                print(f"[{i+1}] {message}")

        return simple_command


__all__ = ["ExamplePlugin", "SimpleExamplePlugin"]
