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

"""CLI entry point for Nvidia Comms Perf Suite."""

import os
import logging
from typing import Annotated

import typer
from rich.logging import RichHandler
from importlib.metadata import version as get_version

from .testing_automation.cli import generate_job_script_cmd
from .plugins.loader import load_and_register_plugins
from .analysis.cli import convert_test_results, upload_test_results
from .report.cli import generate_report

disable_rich = os.environ.get("DISABLE_RICH", "").lower() in ("1", "true", "yes")

# Create the main Typer app
app = typer.Typer(
    help="Nvidia Comms Perf Suite - Performance Tools for NVIDIA GPU Communications Libraries",
    no_args_is_help=True,
    rich_markup_mode="rich" if not disable_rich else None,
)


def setup_logging(verbose: int = 0) -> None:
    """Configure global logging settings.

    Args:
        verbose: If True, enables more detailed logging with timestamps and module names.
    """
    # Remove any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = (
        RichHandler(
            rich_tracebacks=True,
            markup=False,
            show_level=True,
            show_path=False,
            show_time=False,
        )
        if not disable_rich
        else logging.StreamHandler()
    )

    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG

    # Configure logging format and level based on verbose flag
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[handler],
    )


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose", "-v", help="Increase verbosity (-v, -vv)", count=True, show_default=False, metavar=""
        ),
    ] = 0,
    show_version: Annotated[bool, typer.Option("--version", is_eager=True, help="Show version and exit")] = False,
) -> None:
    """Nvidia Comms Perf Suite - Performance Tools for NVIDIA GPU Communications Libraries."""
    if show_version:
        print(f"nvcomms-perf version {get_version('nvcomms-perf')}")
        raise typer.Exit()

    # Only allow verbose if a command is provided
    if verbose and not ctx.invoked_subcommand:
        raise typer.BadParameter("--verbose requires a command")

    setup_logging(verbose)


# Add commands from submodules
app.command(name="generate-job-script")(generate_job_script_cmd)
app.command(name="convert")(convert_test_results)
app.command(name="upload")(upload_test_results)
app.command(name="report")(generate_report)

# Load and register plugins
try:
    load_and_register_plugins(app)
except Exception as e:
    # Log plugin loading errors but don't fail the entire application
    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to load plugins: {e}")

if __name__ == "__main__":
    app()
