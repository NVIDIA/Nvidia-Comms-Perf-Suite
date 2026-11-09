#!/usr/bin/env python3

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

from typing import List
import tomli
import logging
from pathlib import Path
from enum import Enum
from .system_config import SystemConfig
from .user_config import UserConfig
from .testset_config import TestsetConfig
from .sbatch import resolve_sbatch_directives
from jinja2 import Environment, FileSystemLoader

default_template_dir = Path(__file__).resolve().parent / "templates"

logger = logging.getLogger(__name__)


# TODO: Add OpenMPI
class RunType(str, Enum):
    """Enum for different run types."""

    SRUN = "srun"
    CONTAINER = "container"


# TODO: Add interactive and bash
class ScriptType(str, Enum):
    """Enum for different script types."""

    SBATCH = "sbatch"


class TestReportType(str, Enum):
    """Enum for different test report types."""

    JUNIT = "junit"


def generate_job_script(
    systems_toml: str,
    user_toml: str,
    testset_toml: str,
    system_name: str,
    run_type: RunType,
    script_type: ScriptType,
    append_config_contents: bool = False,
    generate_test_report: List[TestReportType] = [],
    testset_name: str = "",
) -> str:
    try:
        testset_config = tomli.loads(testset_toml)
    except tomli.TOMLDecodeError as e:
        raise ValueError(f"Error parsing testset TOML file {testset_toml}: {e}")
    except Exception as e:
        raise ValueError(f"Error loading testset TOML file {testset_toml}: {e}")

    library_name = testset_config.get("library_name", "")
    if library_name == "":
        raise ValueError("'library_name' not found in testset TOML file")

    if run_type == RunType.SRUN or run_type == RunType.CONTAINER:
        launcher = "srun"
        launcher_bin = "srun"
    elif run_type == RunType.OPENMPI or run_type == RunType.MPICH:
        launcher = run_type.value
        launcher_bin = "mpirun"
    else:
        raise ValueError(f"Invalid run type: {run_type}")

    running_env = "container" if run_type == "container" else "baremetal"

    try:
        systems_config = SystemConfig(systems_toml, system_name)
        user_config = UserConfig(user_toml, system_name, running_env, library_name)
        testset_config = TestsetConfig(testset_toml)
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")

    sbatch_directives = []
    if script_type == ScriptType.SBATCH:
        sbatch_directives = resolve_sbatch_directives(
            systems_config.sbatch_params,
            user_config.get_sbatch_directives,
            testset_config.sbatch_directives,
            job_name=testset_name,
        )
        template = (
            f"{library_name}.sbatch.j2"
            if (default_template_dir / f"{library_name}.sbatch.j2").exists()
            else "base.sbatch.j2"
        )
        system = {
            "labels": systems_config.labels,
            "system_name": systems_config.system_name,
            "launcher_params": systems_config.get_launcher_params(launcher),
            "library_envs": systems_config.get_library_params(library_name),
        }
        user = {
            "env_vars": user_config.get_env_vars,
            "pyxis_params": user_config.get_pyxis_params,
            "prologue_commands": user_config.get_prologue_commands,
            "epilogue_commands": user_config.get_epilogue_commands,
        }
        testset = {
            "description": testset_config.description,
            "library_name": testset_config.library_name,
            "testset_iterations": testset_config.testset_iterations,
            "test_iterations": testset_config.test_iterations,
            "use_default_params": testset_config.use_default_params,
            "tests": testset_config.get_tests(),
        }
    elif script_type == "interactive":
        template = "interactive.j2"
        system = {}
        user = {}
        testset = {}
        if launcher != "mpirun":
            logger.warning("Interactive mode is only supported for mpirun, using mpirun as launcher")
            launcher = "mpirun"
    elif script_type == "bash":
        template = "bash.j2"
        system = {}
        user = {}
        testset = {}
        if launcher != "mpirun":
            logger.warning("Bash mode is only supported for mpirun, using mpirun as launcher")
            launcher = "mpirun"
    else:
        raise ValueError(f"Invalid script type: {script_type}")

    generate_test_report = list(set(generate_test_report))
    generate_junit_report = TestReportType.JUNIT in generate_test_report

    jinja_env = Environment(loader=FileSystemLoader([default_template_dir]), trim_blocks=True, lstrip_blocks=True)
    template = jinja_env.get_template(template)
    script = template.render(
        launcher=launcher_bin,
        running_env=running_env,
        system=system,
        user=user,
        testset=testset,
        sbatch_directives=sbatch_directives,
        generate_junit_report=generate_junit_report,
        append_config_contents=append_config_contents,
        arg_system=system_name,
        arg_run_type=run_type,
        arg_script_type=script_type,
        systems_toml_content=systems_toml,
        user_toml_content=user_toml,
        testset_toml_content=testset_toml,
    )

    # Some templates may accidentally emit a leading blank line (e.g., from Jinja
    # whitespace around comments/blocks). Strip only leading newlines so scripts
    # start immediately at the shebang / first directive.
    script = script.lstrip("\n")

    logger.info("Job script generation completed")

    return script
