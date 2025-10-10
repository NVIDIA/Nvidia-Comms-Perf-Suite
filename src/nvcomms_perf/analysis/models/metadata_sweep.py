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
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class SystemMetadata(BaseModel):
    system_name: str
    gpu_arch_type: Optional[str] = None
    cpu_model_name: Optional[str] = None
    cpu_arch_type: Optional[str] = None
    cuda_driver_version: Optional[str] = None
    mpi_type: Optional[str] = None
    mpi_version: Optional[str] = None
    hpcx_version: Optional[str] = None
    os_type: Optional[str] = None
    os_version: Optional[str] = None
    linux_kernel_version: Optional[str] = None
    adaptive_routing: Optional[bool] = None

    model_config = ConfigDict(
        extra="allow",
        frozen=True,
    )


class NetworkMetadata(BaseModel):
    nics: Optional[str] = None
    switch_name: Optional[str] = None
    switch_type: Optional[str] = None
    switch_version: Optional[str] = None
    network_name: Optional[str] = None
    network_version: Optional[str] = None
    network_plugin_name: Optional[str] = None
    network_plugin_version: Optional[str] = None
    mofed_version: Optional[str] = None
    libfabric_version: Optional[str] = None

    model_config = ConfigDict(
        extra="allow",
        frozen=True,
    )


class SweepMetadata(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    raw_data_path: Optional[str] = None
    data_source: Optional[str] = None
    data_reviewer: Optional[str] = None

    cluster_name: Optional[str] = None
    system_metadata: SystemMetadata
    network_metadata: NetworkMetadata

    model_config = ConfigDict(
        extra="allow",
        frozen=True,
    )

    @classmethod
    def read_yaml(cls, obj: Union[str, bytes, IO]) -> "SweepMetadata":
        """
        Parse YAML content into a SweepMetadata instance.

        Accepts:
        - str: YAML content as a string (not a file path)
        - bytes: YAML content as bytes (passed directly to yaml.safe_load)
        - file-like object: Any object with a .read() method

        This mimics yaml.safe_load behavior: both str and bytes are treated as YAML content, not as file paths.
        """
        if isinstance(obj, (str, bytes)):
            return cls.model_validate(yaml.safe_load(obj))
        elif hasattr(obj, "read"):
            return cls.model_validate(yaml.safe_load(obj))
        else:
            raise TypeError("Unsupported input type for read_yaml; must be str, bytes, or file-like object.")
