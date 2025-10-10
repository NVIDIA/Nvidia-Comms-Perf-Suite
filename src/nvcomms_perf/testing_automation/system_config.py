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

import tomli
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

@dataclass
class SystemConfig:
    """Class to manage system configurations from TOML files."""
    config_content: str
    config: Dict[str, Any] = field(default_factory=dict, init=False)
    system_name: str

    def __post_init__(self) -> None:
        """Initialize after dataclass initialization."""
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load the systems configuration from a TOML file."""
        try:
            config = tomli.loads(self.config_content)
            if not config:
                raise ValueError("No systems configuration found")
            if self.system_name not in config:
                raise ValueError("System '{self.system_name}' not found")

            common_config = config.get("common", {})
            system_config = config.get(self.system_name, {})

            merged_config = {
                "labels": {},
                "run_params": self._merge_run_params(common_config.get("run_params", {}), system_config.get("run_params", {}))
            }
            merged_config["labels"].update(common_config.get("labels", {}))
            merged_config["labels"].update(system_config.get("labels", {}))

            logger.info("Loaded systems configuration successfully")
            return merged_config
        except tomli.TOMLDecodeError as e:
            raise ValueError(f"Error parsing systems configuration: {e}")
        except Exception as e:
            raise ValueError(f"Error loading systems configuration : {e}")

    def _merge_run_params(self, common_params: Dict[str, List[str]], system_params: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Merge common run parameters with system-specific run parameters.

        Args:
            common_params: Dictionary of common run parameters (values are lists)
            system_params: Dictionary of system-specific run parameters (values are lists)

        Returns:
            Dictionary with merged run parameters, preserving order and keeping last occurrence of duplicates
        """
        merged_params = {}
        # Get all unique keys from both dictionaries
        all_keys = set(common_params.keys())
        all_keys.update(system_params.keys())
        for param in all_keys:
            # Combine lists and remove duplicates while preserving order (keeping last occurrence)
            values = common_params.get(param, []) + system_params.get(param, [])
            seen = set()
            merged_params[param] = [x for x in reversed(values) if not (x in seen or seen.add(x))][::-1]
        return merged_params

    @property
    def labels(self) -> Dict[str, str]:
        """Get labels for a specific system."""
        return self.config.get("labels", {})

    @property
    def sbatch_params(self) -> List[str]:
        """Get sbatch parameters for a specific system."""
        return self.config.get("run_params", {}).get("sbatch", [])

    @property
    def srun_params(self) -> List[str]:
        """Get srun parameters for a specific system."""
        return self.config.get("run_params", {}).get("srun", [])

    @property
    def openmpi_params(self) -> List[str]:
        """Get openmpi parameters for a specific system."""
        return self.config.get("run_params", {}).get("openmpi", [])

    @property
    def mpich_params(self) -> List[str]:
        """Get mpich parameters for a specific system."""
        return self.config.get("run_params", {}).get("mpich", [])

    @property
    def nccl_params(self) -> List[str]:
        """Get nccl parameters for a specific system."""
        return self.config.get("run_params", {}).get("nccl", [])

    def get_library_params(self, library_name: str) -> List[str]:
        """Get library parameters for a specific system."""
        if library_name == "nccl":
            return self.nccl_params
        elif library_name == "nvshmem":
            return []
        else:
            raise ValueError(f"Invalid library_name: {library_name}")

    def get_launcher_params(self, launcher: str) -> List[str]:
        """Get run type parameters for a specific system."""
        if launcher == "srun":
            return self.srun_params
        elif launcher == "openmpi":
            return self.openmpi_params
        elif launcher == "mpich":
            return self.mpich_params
        else:
            raise ValueError(f"Invalid launcher: {launcher}")
