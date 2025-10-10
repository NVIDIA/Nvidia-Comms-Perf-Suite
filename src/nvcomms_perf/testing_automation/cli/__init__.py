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

"""CLI interface for testing automation."""

import logging
from pathlib import Path
from typing import Annotated, Optional, List

import typer

from ..script_generator import generate_job_script, RunType, ScriptType, TestReportType

logger = logging.getLogger(__name__)

def generate_job_script_cmd(
    system: Annotated[str, typer.Option(help="System name (e.g., CLUSTER_1, CLUSTER_2)", show_default=False)],
    systems_toml: Annotated[Path, typer.Option(..., help="Path to systems TOML file", show_default=False, exists=True, file_okay=True, dir_okay=False, readable=True)],
    user_toml: Annotated[Path, typer.Option(..., help="Path to user TOML file", show_default=False, exists=True, file_okay=True, dir_okay=False, readable=True)],
    testset_toml: Annotated[Path, typer.Option(..., help="Path to testset TOML file", show_default=False, exists=True, file_okay=True, dir_okay=False, readable=True)],
    run_type: Annotated[RunType, typer.Option(help="Run type", show_default=True, show_choices=True)] = RunType.SRUN,
    script_type: Annotated[ScriptType, typer.Option(help="Type of job script to generate", show_default=True, show_choices=True)] = ScriptType.SBATCH,
    generate_test_report: Annotated[Optional[List[TestReportType]], typer.Option("--generate-test-report", help="Generate test report after all tests are completed", show_default=True, show_choices=True)] = None,
    append_config_contents: Annotated[bool, typer.Option("--append-config-contents", help="Append the contents of the TOML files to the generated job script")] = False,
    # Note: if a file named "-" exists in current directory, and not writable, it will prompt an error, because of validation "exists=True".
    #       This is a rare case and can be resolved easily, so won't handle it.
    output: Annotated[Path, typer.Option("-o", "--output", help="File to write job script to. Default: stdout", show_default=False, file_okay=True, dir_okay=False, writable=True)] = Path("-")
) -> None:
    """Generate job scripts for GPU Comm Libraries performance tests."""

    # Read and store contents of the TOML files
    try:
        systems_toml_content = systems_toml.read_text()
        user_toml_content = user_toml.read_text()
        testset_toml_content = testset_toml.read_text()
    except Exception as e:
        logger.error(f"Error reading TOML files: {e}")
        raise typer.Exit(1)

    script = generate_job_script(systems_toml_content, user_toml_content, testset_toml_content, system, run_type, script_type, append_config_contents, generate_test_report or [])

    if output == Path("-"):
        print(script)
    else:
        output.write_text(script)
        logger.info(f"Job script written to: {output}")
