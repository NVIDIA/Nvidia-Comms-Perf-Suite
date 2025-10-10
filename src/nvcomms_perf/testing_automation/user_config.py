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
from typing import Dict, Any

logger = logging.getLogger(__name__)

@dataclass
class UserConfig:
    """Class to manage user configurations from TOML files.
    
    Supports four levels of configuration precedence:
    1. Library-specific (highest)
    2. User-specific
    3. System-specific
    4. Global (lowest)
    """
    config_content: str
    config: Dict[str, Any] = field(default_factory=dict, init=False)
    system_name: str
    running_env: str
    library_name: str

    def __post_init__(self) -> None:
        """Initialize after dataclass initialization."""
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load the user configuration from a TOML file."""
        try:
            config = tomli.loads(self.config_content)
            if not config:
                raise ValueError("No user configuration found")

            # Merge configurations in order of precedence, handling nested dictionaries
            merged_config = self._merge_configs(config)

            logger.info("Loaded user configuration successfully")
            return merged_config
        except tomli.TOMLDecodeError as e:
            raise ValueError(f"Error parsing user configuration: {e}")
        except Exception as e:
            raise ValueError(f"Error loading user configuration: {e}")

    def _merge_configs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configurations in order of precedence, handling nested dictionaries.
        
        Args:
            config: The TOML configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        # Get configurations at different levels
        system_config = config.get(self.system_name, {})
        running_env_config = system_config.get(self.running_env, {})
        library_config = running_env_config.get(self.library_name, {})

        # Merge configurations in order of precedence
        merged_config = {}
        env_vars = {}
        sbatch_directives = []
        pyxis_params = []
        prologue_commands = ""
        epilogue_commands = ""

        # Merge configurations at all levels
        configs = [config, system_config, running_env_config, library_config]
        for current_config in configs:
            for key, value in current_config.items():
                if key == 'sbatch':
                    sbatch_directives.extend(value)
                elif key == 'pyxis':
                    pyxis_params.extend(value)
                elif key == 'prologue_commands':
                    prologue_commands = value
                elif key == 'epilogue_commands':
                    epilogue_commands = value
                elif isinstance(value, (str, int, float, bool)):
                    env_vars[key] = value
        
        # Only add non-empty collections to merged_config
        if env_vars:
            merged_config['env_vars'] = env_vars
        if sbatch_directives:
            merged_config['sbatch'] = sbatch_directives
        if pyxis_params:
            merged_config['pyxis'] = pyxis_params
        if prologue_commands:
            merged_config['prologue_commands'] = prologue_commands
        if epilogue_commands:
            merged_config['epilogue_commands'] = epilogue_commands

        return merged_config
    
    @property
    def get_env_vars(self) -> Dict[str, str]:
        """Get all environment variables from the configuration.
        
        Returns:
            Dictionary of environment variables with their values
        """
        return self.config.get('env_vars', {})

    @property
    def get_sbatch_directives(self) -> list:
        """Get SBATCH directives from the configuration."""
        return self.config.get('sbatch', []) 
    
    @property
    def get_pyxis_params(self) -> list:
        """Get Pyxis parameters from the configuration."""
        return self.config.get('pyxis', [])
    
    @property
    def get_prologue_commands(self) -> str:
        """Get prologue commands from the configuration."""
        return self.config.get('prologue_commands', "")

    @property
    def get_epilogue_commands(self) -> str:
        """Get epilogue commands from the configuration."""
        return self.config.get('epilogue_commands', "")
