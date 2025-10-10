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

import logging
import yaml
from typing import Optional, Union, IO
from pydantic import BaseModel, ConfigDict, computed_field
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class TestMetadata(BaseModel):
    test_name: str
    log_name: str
    timestamp: datetime
    hostname: str
    container: Optional[str] = None
    num_nodes: int
    num_processes: int
    num_process_per_node: int
    unique_nvl_domains: Optional[int] = None
    num_gpus_per_nvl_domain: Optional[str] = None
    slurm_id: Optional[int] = None
    slurm_job_name: Optional[str] = None
    node_list: Optional[str] = None
    test_binary: str
    default_config: bool
    iteration: int
    cmd: str  # TODO: parse cmd to env, mpirun args, srun args, etc., make it a dict, consider computed_field

    @computed_field
    @property
    def env(self) -> Optional[dict[str, str]]:
        if match := re.search(r"env (.+?) bash -c ", self.cmd):
            env_part = match.group(1)
            return {k: v for k, v in (item.strip().split("=", 1) for item in env_part.split("env"))}

        return None

    model_config = ConfigDict(
        extra="allow",
        frozen=True,
    )

    @classmethod
    def read_yaml(cls, obj: Union[str, bytes, IO]) -> "TestMetadata":
        """
        Parse YAML content into a TestMetadata instance.

        Accepts:
        - str: YAML content as a string (not a file path)
        - bytes: YAML content as bytes (passed directly to yaml.safe_load)
        - file-like object: Any object with a .read() method

        This mimics yaml.safe_load behavior: both str and bytes are treated as YAML content, not as file paths.
        """
        if isinstance(obj, (str, bytes)):
            # Treat as YAML content (not a file path)
            return cls.model_validate(yaml.safe_load(obj))
        elif hasattr(obj, "read"):
            # File-like object
            return cls.model_validate(yaml.safe_load(obj))
        else:
            raise TypeError("Unsupported input type for read_yaml; must be str, bytes, or file-like object.")
