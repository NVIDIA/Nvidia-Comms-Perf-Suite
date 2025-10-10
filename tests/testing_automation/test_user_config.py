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

import unittest
import sys
import os
from pathlib import Path
import tomli

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from nvcomms_perf.testing_automation.user_config import UserConfig

class TestUserConfig(unittest.TestCase):
    """Test suite for UserConfig class."""

    def setUp(self):
        """Set up test environment."""
        # Create test configuration content
        self.config_content = """# Default values at top level
CUDA_HOME = "/default/cuda"
sbatch = ["#SBATCH --job-name=test"]
"""

        self.config = UserConfig(
            config_content=self.config_content,
            system_name="h",
            running_env="baremetal",
            library_name="NCCL"
        )

    def test_merge_configs_basic(self):
        """Test basic merging of configurations."""
        config = {
            'CUDA_HOME': '/default/cuda',
            'sbatch': ['#SBATCH --job-name=test'],
            'h': {
                'CUDA_HOME': '/h/cuda',
                'sbatch': ['#SBATCH --partition=h']
            }
        }
        result = self.config._merge_configs(config)

        expected = {
            'env_vars': {
                'CUDA_HOME': '/h/cuda'
            },
            'sbatch': ['#SBATCH --job-name=test', '#SBATCH --partition=h']
        }
        self.assertDictEqual(result, expected)

    def test_merge_configs_nested(self):
        """Test merging of nested configurations."""
        config = {
            'CUDA_HOME': '/default/cuda',
            'sbatch': ['#SBATCH --job-name=test'],
            'h': {
                'CUDA_HOME': '/h/cuda',
                'sbatch': ['#SBATCH --partition=h'],
                'baremetal': {
                    'MPI_HOME': '/h/baremetal/mpi',
                    'sbatch': ['#SBATCH --constraint=baremetal'],
                    'NCCL': {
                        'NCCL_HOME': '/h/baremetal/NCCL/nccl',
                        'sbatch': ['#SBATCH --gpus-per-node=8']
                    }
                }
            }
        }
        result = self.config._merge_configs(config)

        expected = {
            'env_vars': {
                'CUDA_HOME': '/h/cuda',
                'MPI_HOME': '/h/baremetal/mpi',
                'NCCL_HOME': '/h/baremetal/NCCL/nccl'
            },
            'sbatch': [
                '#SBATCH --job-name=test',
                '#SBATCH --partition=h',
                '#SBATCH --constraint=baremetal',
                '#SBATCH --gpus-per-node=8'
            ]
        }
        self.assertDictEqual(result, expected)

    def test_merge_configs_missing_levels(self):
        """Test merging when some configuration levels are missing."""
        config = {
            'CUDA_HOME': '/default/cuda',
            'sbatch': ['#SBATCH --job-name=test'],
            'h': {
                'CUDA_HOME': '/h/cuda',
                'sbatch': ['#SBATCH --partition=h']
            }
        }
        result = self.config._merge_configs(config)

        expected = {
            'env_vars': {
                'CUDA_HOME': '/h/cuda'
            },
            'sbatch': ['#SBATCH --job-name=test', '#SBATCH --partition=h']
        }
        self.assertDictEqual(result, expected)

    def test_merge_configs_empty(self):
        """Test merging with empty configuration."""
        config = {}
        result = self.config._merge_configs(config)
        expected = {}  # Empty config should result in empty dict
        self.assertDictEqual(result, expected)

    def test_merge_configs_no_sbatch(self):
        """Test merging when no sbatch directives are present."""
        config = {
            'CUDA_HOME': '/default/cuda',
            'h': {
                'CUDA_HOME': '/h/cuda'
            }
        }
        result = self.config._merge_configs(config)

        expected = {
            'env_vars': {
                'CUDA_HOME': '/h/cuda'
            }
        }
        self.assertDictEqual(result, expected)

    def test_merge_configs_sbatch_only(self):
        """Test merging when only sbatch directives are present."""
        config = {
            'sbatch': ['#SBATCH --job-name=test'],
            'h': {
                'sbatch': ['#SBATCH --partition=h']
            }
        }
        result = self.config._merge_configs(config)

        expected = {
            'sbatch': ['#SBATCH --job-name=test', '#SBATCH --partition=h']
        }
        self.assertDictEqual(result, expected)

    def test_merge_configs_different_system(self):
        """Test merging with a different system name."""
        self.config.system_name = "CW-DFW"
        config = {
            'CUDA_HOME': '/default/cuda',
            'sbatch': ['#SBATCH --job-name=test'],
            'h': {
                'CUDA_HOME': '/h/cuda',
                'sbatch': ['#SBATCH --partition=h']
            }
        }
        result = self.config._merge_configs(config)

        expected = {
            'env_vars': {
                'CUDA_HOME': '/default/cuda'
            },
            'sbatch': ['#SBATCH --job-name=test']
        }
        self.assertDictEqual(result, expected)

    def test_merge_configs_pyxis_container(self):
        """Test merging of pyxis parameters in container mode."""
        self.config.running_env = "container"
        config = {
            'pyxis': ['--container-image=base:latest'],
            'h': {
                'pyxis': ['--container-image=h:latest'],
                'container': {
                    'pyxis': ['--container-mounts=/data:/data'],
                    'NCCL': {
                        'pyxis': ['--container-workdir=/workspace']
                    }
                }
            }
        }
        result = self.config._merge_configs(config)

        expected = {
            'pyxis': [
                '--container-image=base:latest',
                '--container-image=h:latest',
                '--container-mounts=/data:/data',
                '--container-workdir=/workspace'
            ]
        }
        self.assertDictEqual(result, expected)

    def test_merge_configs_pyxis_baremetal(self):
        """Test collection of pyxis parameters in baremetal mode."""
        self.config.running_env = "baremetal"
        config = {
            'pyxis': ['--container-image=base:latest'],
            'h': {
                'pyxis': ['--container-image=h:latest'],
                'baremetal': {
                    'pyxis': ['--container-mounts=/data:/data'],
                    'NCCL': {
                        'pyxis': ['--container-workdir=/workspace']
                    }
                }
            }
        }
        result = self.config._merge_configs(config)
        expected = {
            'pyxis': [
                '--container-image=base:latest',
                '--container-image=h:latest',
                '--container-mounts=/data:/data',
                '--container-workdir=/workspace'
            ]
        }
        self.assertDictEqual(result, expected)

    def test_merge_configs_pyxis_empty(self):
        """Test handling of empty pyxis parameters."""
        self.config.running_env = "container"
        config = {
            'h': {
                'container': {
                    'NCCL': {}
                }
            }
        }
        result = self.config._merge_configs(config)
        expected = {}  # Empty config should result in empty dict
        self.assertDictEqual(result, expected)

    def test_merge_configs_prologue_commands(self):
        """Test merging of prologue commands with normal precedence rules."""
        config = {
            'prologue_commands': 'module load cuda/12.0',
            'h': {
                'prologue_commands': 'module load openmpi/4.1.0',
                'baremetal': {
                    'prologue_commands': 'module load hpcx/2.14.0',
                    'NCCL': {
                        'prologue_commands': 'export NCCL_HOME=${NCCL_HOME}'
                    }
                }
            }
        }
        result = self.config._merge_configs(config)

        expected = {
            'prologue_commands': 'export NCCL_HOME=${NCCL_HOME}'
        }
        self.assertDictEqual(result, expected)

    def test_merge_configs_epilogue_commands(self):
        """Test merging of epilogue commands with normal precedence rules."""
        config = {
            'epilogue_commands': 'module unload cuda/12.0',
            'h': {
                'epilogue_commands': 'module unload openmpi/4.1.0',
                'baremetal': {
                    'epilogue_commands': 'module unload hpcx/2.14.0',
                    'NCCL': {
                        'epilogue_commands': 'echo "NCCL tests completed"'
                    }
                }
            }
        }
        result = self.config._merge_configs(config)

        expected = {
            'epilogue_commands': 'echo "NCCL tests completed"'
        }
        self.assertDictEqual(result, expected)

    def test_merge_configs_commands_empty(self):
        """Test handling of empty prologue and epilogue commands."""
        config = {
            'h': {
                'baremetal': {
                    'NCCL': {}
                }
            }
        }
        result = self.config._merge_configs(config)
        expected = {}  # Empty config should result in empty dict
        self.assertDictEqual(result, expected)

    def test_merge_configs_commands_with_other_params(self):
        """Test merging of prologue and epilogue commands with other parameters."""
        config = {
            'prologue_commands': 'module load cuda/12.0',
            'epilogue_commands': 'module unload cuda/12.0',
            'sbatch': ['#SBATCH --job-name=test'],
            'h': {
                'prologue_commands': 'module load openmpi/4.1.0',
                'epilogue_commands': 'module unload openmpi/4.1.0',
                'sbatch': ['#SBATCH --partition=h'],
                'baremetal': {
                    'prologue_commands': 'module load hpcx/2.14.0',
                    'epilogue_commands': 'module unload hpcx/2.14.0',
                    'NCCL': {
                        'prologue_commands': 'export NCCL_HOME=${NCCL_HOME}',
                        'epilogue_commands': 'echo "NCCL tests completed"',
                        'sbatch': ['#SBATCH --gpus-per-node=8']
                    }
                }
            }
        }
        result = self.config._merge_configs(config)

        expected = {
            'prologue_commands': 'export NCCL_HOME=${NCCL_HOME}',
            'epilogue_commands': 'echo "NCCL tests completed"',
            'sbatch': [
                '#SBATCH --job-name=test',
                '#SBATCH --partition=h',
                '#SBATCH --gpus-per-node=8'
            ]
        }
        self.assertDictEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
