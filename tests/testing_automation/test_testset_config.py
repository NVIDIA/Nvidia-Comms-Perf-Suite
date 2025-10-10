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

# pytest tries to collect all class starting with 'Test'
# so we need to rename the class to avoid being collected
from nvcomms_perf.testing_automation.testset_config import TestsetConfig as xTestsetConfig

class TestTestsetConfig(unittest.TestCase):
    """Test suite for TestsetConfig class, which handles NCCL test configurations from TOML files."""

    def setUp(self):
        """Set up test environment by creating a temporary TOML configuration file.

        Creates a test configuration with:
        - Common section: NCCL_DEBUG=INFO and -n=2
        - Test section: NCCL_ALGO=RING and -g=1
        """

        # Create a test configuration with basic NCCL parameters
        config_content = '''
[common]
sweep_envs = { NCCL_DEBUG = "INFO" }
sweep_args = { "-n" = "2" }

[tests.test1]
sweep_envs = { NCCL_ALGO = "RING" }
sweep_args = { "-g" = "1" }
'''

        self.config = xTestsetConfig(config_content)

    def test_ensure_list_values_mixed(self):
        """Test _ensure_list_values with mixed single values and lists."""
        input_dict = {
            "NCCL_DEBUG": "INFO",
            "NCCL_ALGO": ["RING", "TREE"],
            "NCCL_PROTO": "LL",
            "NCCL_NTHREADS": [1, 2, 4]
        }
        expected = {
            "NCCL_DEBUG": ["INFO"],
            "NCCL_ALGO": ["RING", "TREE"],
            "NCCL_PROTO": ["LL"],
            "NCCL_NTHREADS": [1, 2, 4]
        }
        result = self.config._ensure_list_values(input_dict)
        self.assertEqual(result, expected)

    def test_ensure_list_values_empty(self):
        """Test _ensure_list_values with empty dictionary."""
        input_dict = {}
        expected = {}
        result = self.config._ensure_list_values(input_dict)
        self.assertEqual(result, expected)

    def test_merge_dictionaries_override(self):
        """Test _merge_dictionaries with override strategy."""
        dict1 = {
            "NCCL_DEBUG": ["INFO"],
            "NCCL_ALGO": ["RING"],
            "NCCL_PROTO": ["LL"]
        }
        dict2 = {
            "NCCL_DEBUG": ["WARN"],
            "NCCL_ALGO": ["TREE"],
            "NCCL_NTHREADS": ["4"]
        }
        expected = {
            "NCCL_DEBUG": ["WARN"],
            "NCCL_ALGO": ["TREE"],
            "NCCL_PROTO": ["LL"],
            "NCCL_NTHREADS": ["4"]
        }
        result = self.config._merge_dictionaries(dict1, dict2, "override")
        self.assertEqual(result, expected)

    def test_merge_dictionaries_merge(self):
        """Test _merge_dictionaries with merge strategy."""
        dict1 = {
            "NCCL_DEBUG": ["INFO"],
            "NCCL_ALGO": ["RING"],
            "NCCL_PROTO": ["LL"]
        }
        dict2 = {
            "NCCL_DEBUG": ["WARN"],
            "NCCL_ALGO": ["TREE"],
            "NCCL_NTHREADS": ["4"]
        }
        expected = {
            "NCCL_DEBUG": ["INFO", "WARN"],
            "NCCL_ALGO": ["RING", "TREE"],
            "NCCL_PROTO": ["LL"],
            "NCCL_NTHREADS": ["4"]
        }
        result = self.config._merge_dictionaries(dict1, dict2, "merge")
        # Sort lists for comparison since set operation doesn't preserve order
        result = {k: sorted(v) for k, v in result.items()}
        expected = {k: sorted(v) for k, v in expected.items()}
        self.assertEqual(result, expected)

    def test_merge_dictionaries_invalid_strategy(self):
        """Test _merge_dictionaries with invalid strategy."""
        dict1 = {
            "NCCL_DEBUG": ["INFO"],
            "NCCL_ALGO": ["RING"]
        }
        dict2 = {
            "NCCL_DEBUG": ["WARN"],
            "NCCL_ALGO": ["TREE"]
        }
        with self.assertRaises(ValueError) as context:
            self.config._merge_dictionaries(dict1, dict2, "invalid")
        self.assertIn("Invalid merge strategy", str(context.exception))

    def test_process_exceptions_basic(self):
        """Test _process_exceptions with basic exception processing."""
        exceptions = [{
            "sweep_envs": {
                "NCCL_DEBUG": "INFO",
                "NCCL_ALGO": ["RING", "TREE"]
            },
            "sweep_args": {
                "-n": ["2"],
                "-g": "1"
            }
        }]
        expected = [{
            "sweep_envs": {
                "NCCL_DEBUG": ["INFO"],
                "NCCL_ALGO": ["RING", "TREE"]
            },
            "sweep_args": {
                "-n": ["2"],
                "-g": ["1"]
            }
        }]
        result = self.config._process_exceptions(exceptions)
        self.assertEqual(result, expected)

    def test_process_exceptions_multiple(self):
        """Test _process_exceptions with multiple exceptions."""
        exceptions = [
            {
                "sweep_envs": {"NCCL_DEBUG": "INFO"},
                "sweep_args": {"-n": ["2", "4"]}
            },
            {
                "sweep_envs": {"NCCL_ALGO": ["RING", "TREE"]},
                "sweep_args": {"-g": "1"}
            }
        ]
        expected = [
            {
                "sweep_envs": {"NCCL_DEBUG": ["INFO"]},
                "sweep_args": {"-n": ["2", "4"]}
            },
            {
                "sweep_envs": {"NCCL_ALGO": ["RING", "TREE"]},
                "sweep_args": {"-g": ["1"]}
            }
        ]
        result = self.config._process_exceptions(exceptions)
        # Sort lists for comparison since sorting is applied in the function
        for r, e in zip(result, expected):
            for key in ["sweep_envs", "sweep_args"]:
                r[key] = {k: sorted(v) for k, v in r[key].items()}
                e[key] = {k: sorted(v) for k, v in e[key].items()}
        self.assertEqual(result, expected)

    def test_process_exceptions_empty(self):
        """Test _process_exceptions with empty exceptions list."""
        exceptions = []
        expected = []
        result = self.config._process_exceptions(exceptions)
        self.assertEqual(result, expected)

    def test_filter_combinations_no_exceptions(self):
        """Test _filter_combinations_by_exceptions with no exceptions."""
        self.maxDiff = None
        combinations = [
            {"env": {"NCCL_DEBUG": "INFO", "NCCL_ALGO": "RING", "NCCL_MIN_CTAS": 3}, "arg": {"-n": "2", "-g": "1"}},
            {"env": {"NCCL_DEBUG": "INFO", "NCCL_ALGO": "TREE", "NCCL_MIN_CTAS": 3}, "arg": {"-n": "2", "-g": "1"}},
            {"env": {"NCCL_DEBUG": "WARN", "NCCL_ALGO": "RING", "NCCL_MIN_CTAS": 3}, "arg": {"-n": "2", "-g": "1"}},
            {"env": {"NCCL_DEBUG": "WARN", "NCCL_ALGO": "TREE", "NCCL_MIN_CTAS": 3}, "arg": {"-n": "2", "-g": "1"}},
            {"env": {"NCCL_DEBUG": "INFO", "NCCL_ALGO": "RING", "NCCL_MIN_CTAS": 3}, "arg": {"-n": "4", "-g": "1"}},
            {"env": {"NCCL_DEBUG": "INFO", "NCCL_ALGO": "TREE", "NCCL_MIN_CTAS": 5}, "arg": {"-n": "4", "-g": "1"}},
            {"env": {"NCCL_DEBUG": "WARN", "NCCL_ALGO": "RING"}, "arg": {"-n": "4", "-g": "1"}},
            {"env": {"NCCL_DEBUG": "WARN", "NCCL_ALGO": "TREE", "NCCL_MIN_CTAS": 5}, "arg": {"-n": "4", "-g": "1"}},
            {"env": {"NCCL_DEBUG": "INFO", "NCCL_ALGO": "RING"}, "arg": {"-n": "2", "-g": "1"}},
        ]
        exceptions = []
        expected = combinations
        result = self.config._filter_combinations_by_exceptions(combinations, exceptions)
        self.assertEqual(self._sort_combinations(result), self._sort_combinations(expected))

    def test_filter_combinations_env_exceptions(self):
        """Test _filter_combinations_by_exceptions with environment variable exceptions."""
        self.maxDiff = None
        combinations = [
            {"env": {"NCCL_DEBUG": "INFO", "NCCL_ALGO": "RING", "NCCL_MIN_CTAS": 3}, "arg": {"-n": "2", "-g": "1"}},
            {"env": {"NCCL_DEBUG": "INFO", "NCCL_ALGO": "TREE", "NCCL_MIN_CTAS": 3}, "arg": {"-n": "2", "-g": "1"}},
            {"env": {"NCCL_DEBUG": "WARN", "NCCL_ALGO": "RING", "NCCL_MIN_CTAS": 3}, "arg": {"-n": "2", "-g": "1"}},
            {"env": {"NCCL_DEBUG": "WARN", "NCCL_ALGO": "TREE", "NCCL_MIN_CTAS": 3}, "arg": {"-n": "2", "-g": "1"}},
            {"env": {"NCCL_DEBUG": "INFO", "NCCL_ALGO": "RING", "NCCL_MIN_CTAS": 3}, "arg": {"-n": "4", "-g": "1"}},
            {"env": {"NCCL_DEBUG": "INFO", "NCCL_ALGO": "TREE", "NCCL_MIN_CTAS": 5}, "arg": {"-n": "4", "-g": "1"}},
            {"env": {"NCCL_DEBUG": "WARN", "NCCL_ALGO": "RING"}, "arg": {"-n": "4", "-g": "1"}},
            {"env": {"NCCL_DEBUG": "WARN", "NCCL_ALGO": "TREE", "NCCL_MIN_CTAS": 5}, "arg": {"-n": "4", "-g": "1"}},
            {"env": {"NCCL_DEBUG": "INFO", "NCCL_ALGO": "RING"}, "arg": {"-n": "2", "-g": "1"}},
        ]
        exceptions = [
            {"sweep_envs": {"NCCL_DEBUG": ["INFO"]}, "sweep_args": {}}
        ]
        expected = [
            {"env": {"NCCL_DEBUG": "WARN", "NCCL_ALGO": "RING", "NCCL_MIN_CTAS": 3}, "arg": {"-n": "2", "-g": "1"}},
            {"env": {"NCCL_DEBUG": "WARN", "NCCL_ALGO": "TREE", "NCCL_MIN_CTAS": 3}, "arg": {"-n": "2", "-g": "1"}},
            {"env": {"NCCL_DEBUG": "WARN", "NCCL_ALGO": "RING"}, "arg": {"-n": "4", "-g": "1"}},
            {"env": {"NCCL_DEBUG": "WARN", "NCCL_ALGO": "TREE", "NCCL_MIN_CTAS": 5}, "arg": {"-n": "4", "-g": "1"}}
        ]
        result = self.config._filter_combinations_by_exceptions(combinations, exceptions)
        self.assertEqual(self._sort_combinations(result), self._sort_combinations(expected))

    def _sort_combinations(self, combinations):
        """Helper function to sort combinations for consistent comparison.

        Args:
            combinations: List of dictionaries containing env and arg combinations

        Returns:
            Sorted list of combinations based on env and arg contents
        """
        return sorted(combinations, key=lambda x: (
            str(sorted(x["env"].items())),
            str(sorted(x["arg"].items()))
        ))

    def test_create_sweep_combinations_basic(self):
        """Test basic sweep combination creation."""
        sweep_dict = {
            "NCCL_DEBUG": ["INFO", "WARN"],
            "NCCL_ALGO": ["RING", "TREE"]
        }
        expected = [
            {"NCCL_DEBUG": "INFO", "NCCL_ALGO": "RING"},
            {"NCCL_DEBUG": "INFO", "NCCL_ALGO": "TREE"},
            {"NCCL_DEBUG": "WARN", "NCCL_ALGO": "RING"},
            {"NCCL_DEBUG": "WARN", "NCCL_ALGO": "TREE"}
        ]
        self.assertEqual(self.config._create_sweep_combinations(sweep_dict), expected)

    def test_create_sweep_combinations_empty(self):
        """Test sweep combination creation with empty input."""
        self.assertEqual(self.config._create_sweep_combinations({}), [{}])

    def test_create_sweep_combinations_single_param(self):
        """Test sweep combination creation with single parameter."""
        sweep_dict = {"NCCL_DEBUG": ["INFO", "WARN", "ERROR"]}
        expected = [
            {"NCCL_DEBUG": "INFO"},
            {"NCCL_DEBUG": "WARN"},
            {"NCCL_DEBUG": "ERROR"}
        ]
        self.assertEqual(self.config._create_sweep_combinations(sweep_dict), expected)

    def test_create_sweep_combinations_mixed_types(self):
        """Test sweep combination creation with mixed value types."""
        sweep_dict = {
            "NCCL_DEBUG": ["INFO", "WARN"],
            "NCCL_NTHREADS": [1, 2],
            "NCCL_USE_CPU": [True, False]
        }
        expected = [
            {"NCCL_DEBUG": "INFO", "NCCL_NTHREADS": 1, "NCCL_USE_CPU": True},
            {"NCCL_DEBUG": "INFO", "NCCL_NTHREADS": 1, "NCCL_USE_CPU": False},
            {"NCCL_DEBUG": "INFO", "NCCL_NTHREADS": 2, "NCCL_USE_CPU": True},
            {"NCCL_DEBUG": "INFO", "NCCL_NTHREADS": 2, "NCCL_USE_CPU": False},
            {"NCCL_DEBUG": "WARN", "NCCL_NTHREADS": 1, "NCCL_USE_CPU": True},
            {"NCCL_DEBUG": "WARN", "NCCL_NTHREADS": 1, "NCCL_USE_CPU": False},
            {"NCCL_DEBUG": "WARN", "NCCL_NTHREADS": 2, "NCCL_USE_CPU": True},
            {"NCCL_DEBUG": "WARN", "NCCL_NTHREADS": 2, "NCCL_USE_CPU": False}
        ]
        self.assertEqual(self.config._create_sweep_combinations(sweep_dict), expected)

    def test_create_sweep_combinations_with_default(self):
        """Test sweep combination creation with default values."""
        sweep_dict = {
            "NCCL_DEBUG": ["INFO", "default"],
            "NCCL_ALGO": ["RING", "TREE"]
        }
        expected = [
            {"NCCL_DEBUG": "INFO", "NCCL_ALGO": "RING"},
            {"NCCL_DEBUG": "INFO", "NCCL_ALGO": "TREE"},
            {"xNCCL_DEBUG": "default", "NCCL_ALGO": "RING"},
            {"xNCCL_DEBUG": "default", "NCCL_ALGO": "TREE"}
        ]
        self.assertEqual(self.config._create_sweep_combinations(sweep_dict, deal_with_default=True), expected)

    def test_create_sweep_combinations_with_default_no_transform(self):
        """Test sweep combination creation with default values but no transformation."""
        sweep_dict = {
            "NCCL_DEBUG": ["INFO", "default"],
            "NCCL_ALGO": ["RING", "TREE"]
        }
        expected = [
            {"NCCL_DEBUG": "INFO", "NCCL_ALGO": "RING"},
            {"NCCL_DEBUG": "INFO", "NCCL_ALGO": "TREE"},
            {"NCCL_DEBUG": "default", "NCCL_ALGO": "RING"},
            {"NCCL_DEBUG": "default", "NCCL_ALGO": "TREE"}
        ]
        self.assertEqual(self.config._create_sweep_combinations(sweep_dict, deal_with_default=False), expected)

    def test_create_sweep_combinations_multiple_defaults(self):
        """Test sweep combination creation with multiple default values."""
        sweep_dict = {
            "NCCL_DEBUG": ["INFO", "default"],
            "NCCL_ALGO": ["RING", "default"]
        }
        expected = [
            {"NCCL_DEBUG": "INFO", "NCCL_ALGO": "RING"},
            {"NCCL_DEBUG": "INFO", "xNCCL_ALGO": "default"},
            {"xNCCL_DEBUG": "default", "NCCL_ALGO": "RING"},
            {"xNCCL_DEBUG": "default", "xNCCL_ALGO": "default"}
        ]
        self.assertEqual(self.config._create_sweep_combinations(sweep_dict, deal_with_default=True), expected)

    def test_create_sweep_combinations_custom_commands_basic(self):
        """Test sweep combination creation with custom commands."""
        sweep_dict = {
            "pre_command": ["export VAR1=1"],
            "post_command": ["unset VAR1"]
        }
        expected = [
            {"pre_command": "export VAR1=1", "post_command": "unset VAR1"}
        ]
        self.assertEqual(self.config._create_sweep_combinations(sweep_dict), expected)

    def test_create_sweep_combinations_custom_commands_single(self):
        """Test sweep combination creation with single custom commands."""
        sweep_dict = {
            "pre_command": ["export TEST_VAR=1"],
            "post_command": ["unset TEST_VAR"]
        }
        expected = [
            {"pre_command": "export TEST_VAR=1", "post_command": "unset TEST_VAR"}
        ]
        self.assertEqual(self.config._create_sweep_combinations(sweep_dict), expected)

    def test_create_sweep_combinations_custom_commands_partial(self):
        """Test sweep combination creation with only pre_command or post_command."""
        # Only pre_command - should NOT have post_command
        sweep_dict_pre = {"pre_command": ["export TEST_VAR=1"]}
        result_pre = self.config._create_sweep_combinations(sweep_dict_pre)
        self.assertEqual(len(result_pre), 1)
        self.assertEqual(result_pre[0]["pre_command"], "export TEST_VAR=1")
        self.assertNotIn("post_command", result_pre[0])  # Explicitly check NO post_command
        
        # Only post_command - should NOT have pre_command
        sweep_dict_post = {"post_command": ["unset TEST_VAR"]}
        result_post = self.config._create_sweep_combinations(sweep_dict_post)
        self.assertEqual(len(result_post), 1)
        self.assertEqual(result_post[0]["post_command"], "unset TEST_VAR")
        self.assertNotIn("pre_command", result_post[0])  # Explicitly check NO pre_command

    def test_create_sweep_combinations_multiple_pre_commands_error(self):
        """Test that multiple pre_command values raise an error."""
        sweep_dict = {
            "pre_command": ["export VAR1=1", "export VAR2=2"]
        }
        with self.assertRaises(ValueError) as context:
            self.config._create_sweep_combinations(sweep_dict)
        
        error_msg = str(context.exception)
        self.assertIn("pre_command can have a maximum of 1 value", error_msg)
        self.assertIn("2 values were provided", error_msg)
        self.assertIn("Failed to generate script", error_msg)

    def test_create_sweep_combinations_multiple_post_commands_error(self):
        """Test that multiple post_command values raise an error."""
        sweep_dict = {
            "post_command": ["unset VAR1", "unset VAR2", "unset VAR3"]
        }
        with self.assertRaises(ValueError) as context:
            self.config._create_sweep_combinations(sweep_dict)
        
        error_msg = str(context.exception)
        self.assertIn("post_command can have a maximum of 1 value", error_msg)
        self.assertIn("3 values were provided", error_msg)
        self.assertIn("Failed to generate script", error_msg)

    def test_create_sweep_combinations_both_multiple_commands_error(self):
        """Test that multiple pre_command and post_command values raise an error."""
        sweep_dict = {
            "pre_command": ["export VAR1=1", "export VAR2=2"],
            "post_command": ["unset VAR1"]
        }
        with self.assertRaises(ValueError) as context:
            self.config._create_sweep_combinations(sweep_dict)
        
        # Should fail on pre_command first
        error_msg = str(context.exception)
        self.assertIn("pre_command can have a maximum of 1 value", error_msg)

if __name__ == '__main__':
    unittest.main()
