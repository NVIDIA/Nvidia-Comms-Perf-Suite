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

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from nvcomms_perf.testing_automation.system_config import SystemConfig

class TestSystemConfig(unittest.TestCase):
    def setUp(self):
        # Create test configuration content
        self.config_content = '''
[common]
run_params = { sbatch = ["--exclusive"] }

[test_system]
labels = { key1 = "value1", key2 = "value2" }
run_params = { sbatch = ["--time=01:00:00"] }
'''

        self.config = SystemConfig(
            config_content=self.config_content,
            system_name='test_system'
        )

    def test_merge_run_params_basic(self):
        """Test basic merging of run parameters with no duplicates."""
        common_params = {
            "sbatch": ["--time=01:00:00", "--nodes=1"],
            "openmpi": ["--mca btl ^openib"]
        }
        system_params = {
            "sbatch": ["--partition=debug"],
            "nccl": ["NCCL_DEBUG=INFO"]
        }
        expected = {
            "sbatch": ["--time=01:00:00", "--nodes=1", "--partition=debug"],
            "openmpi": ["--mca btl ^openib"],
            "nccl": ["NCCL_DEBUG=INFO"]
        }
        result = self.config._merge_run_params(common_params, system_params)
        self.assertEqual(result, expected)

    def test_merge_run_params_duplicates(self):
        """Test merging run parameters with duplicates - last occurrence should be kept."""
        common_params = {
            "sbatch": ["--time=01:00:00", "--nodes=1", "--partition=debug"],
            "openmpi": ["--mca btl ^openib"]
        }
        system_params = {
            "sbatch": ["--partition=debug", "--nodes=2"],
            "openmpi": ["--mca btl ^openib", "--mca btl_base_verbose 1"]
        }
        expected = {
            "sbatch": ["--time=01:00:00", "--nodes=1", "--partition=debug", "--nodes=2"],
            "openmpi": ["--mca btl ^openib", "--mca btl_base_verbose 1"]
        }
        result = self.config._merge_run_params(common_params, system_params)
        self.assertEqual(result, expected)

    def test_merge_run_params_multiple_duplicates(self):
        """Test merging run parameters with multiple duplicates - verify last occurrence is kept."""
        common_params = {
            "sbatch": ["--time=01:00:00", "--nodes=1", "--partition=debug"],
            "openmpi": ["--mca btl ^openib", "--mca btl_base_verbose 1"]
        }
        system_params = {
            "sbatch": ["--partition=debug", "--nodes=1", "--time=02:00:00"],
            "openmpi": ["--mca btl ^openib", "--mca btl_base_verbose 2"]
        }
        expected = {
            "sbatch": ["--time=01:00:00", "--partition=debug", "--nodes=1", "--time=02:00:00"],
            "openmpi": ["--mca btl_base_verbose 1", "--mca btl ^openib", "--mca btl_base_verbose 2"]
        }
        result = self.config._merge_run_params(common_params, system_params)
        self.assertEqual(result, expected)

    def test_merge_run_params_empty(self):
        """Test merging run parameters with empty dictionaries."""
        common_params = {}
        system_params = {}
        expected = {}
        result = self.config._merge_run_params(common_params, system_params)
        self.assertEqual(result, expected)

    def test_merge_run_params_one_empty(self):
        """Test merging run parameters with one empty dictionary."""
        common_params = {"sbatch": ["--time=01:00:00"]}
        system_params = {}
        expected = {"sbatch": ["--time=01:00:00"]}
        result = self.config._merge_run_params(common_params, system_params)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
