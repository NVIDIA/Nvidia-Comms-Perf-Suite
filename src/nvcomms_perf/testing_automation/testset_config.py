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
import itertools
from typing import Dict, List, Any
import os

logger = logging.getLogger(__name__)

@dataclass
class TestsetConfig:
    """Class to parse testset configurations from TOML files."""
    config_content: str
    config: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Initialize after dataclass initialization."""
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load the testset configuration from a TOML file."""
        try:
            config = tomli.loads(self.config_content)
            if not config:
                raise ValueError("No testset configuration found")

            common_config = config.get("common", {})
            tests_config = config.get("test", [])

            merged_config = {
                "description": config.get("description", ""),
                "library_name": config.get("library_name", ""),
                "testset_iterations": config.get("testset_iterations", 1),
                "test_iterations": config.get("test_iterations", 1),
                "use_default_params": config.get("use_default_params", 0),
                "allowed_envs_to_sweep_on_default": config.get("allowed_envs_to_sweep_on_default", []),
                "allowed_args_to_sweep_on_default": config.get("allowed_args_to_sweep_on_default", []),
                "tests": []
            }

            for test_config in tests_config:
                envs_merge_strategy = test_config.get("envs_merge_strategy", "override")
                args_merge_strategy = test_config.get("args_merge_strategy", "override")

                for sweep_type in ["sweep_envs", "sweep_args", "sweep_custom_commands"]:
                    if sweep_type not in test_config:
                        test_config[sweep_type] = self._ensure_list_values(
                            common_config.get(sweep_type, {})
                        )
                    else:
                        test_config[sweep_type] = self._merge_dictionaries(
                            common_config.get(sweep_type, {}),
                            test_config[sweep_type],
                            envs_merge_strategy if sweep_type == "sweep_envs" else args_merge_strategy
                        )

                # Handle exceptions
                test_config["exceptions"] = self._process_exceptions(
                    common_config.get("exceptions", []) + test_config.get("exceptions", [])
                )

                merged_config["tests"].append({
                    "name": test_config.get("name", ""),
                    "binary": test_config.get("binary", "") or common_config.get("binary", ""),
                    "path": test_config.get("path", "") or common_config.get("path", ""),
                    "sweep_envs": test_config.get("sweep_envs", {}),
                    "sweep_args": test_config.get("sweep_args", {}),
                    "sweep_custom_commands": test_config.get("sweep_custom_commands", {}),
                    "exceptions": test_config.get("exceptions", [])
                })

            self._validate_default_config(merged_config)

            logger.info("Loaded testset configuration successfully")
            return merged_config
        except tomli.TOMLDecodeError as e:
            raise ValueError(f"Error parsing testset configuration: {e}")
        except Exception as e:
            raise ValueError(f"Error loading testset configuration: {e}")

    @property
    def description(self) -> str:
        """Get the test description."""
        return self.config.get("description", "")

    @property
    def library_name(self) -> str:
        """Get the library name."""
        return self.config.get("library_name", "")

    @property
    def testset_iterations(self) -> int:
        """Get the testset iterations."""
        return self.config.get("testset_iterations", 1)

    @property
    def test_iterations(self) -> int:
        """Get the test iterations."""
        return self.config.get("test_iterations", 1)

    @property
    def use_default_params(self) -> bool:
        """Get the use_default_params flag."""
        return self.config.get("use_default_params", 0)

    def get_tests(self) -> List[Dict[str, Any]]:
        """Get the tests from the configuration.

        Returns a list of test configurations. For example:

        .. code-block:: python

            [
                {
                    "name": "allreduce",
                    "path": "/path/to/",
                    "binary": "all_reduce_perf",
                    "combinations": [
                        {
                            "env": {"NCCL_DEBUG": "INFO", "NCCL_ALGO": "RING"},
                            "arg": {"-n": "2", "-g": "1"}
                        },
                        {
                            "env": {"NCCL_DEBUG": "INFO", "NCCL_ALGO": "TREE"},
                            "arg": {"-n": "4", "-g": "2"}
                        }
                    ]
                }
            ]

        """
        tests = []

        for test_config in self.config.get("tests", []):
            sweep_envs_combinations = self._create_sweep_combinations(test_config["sweep_envs"], deal_with_default=True)
            sweep_args_combinations = self._create_sweep_combinations(test_config["sweep_args"])
            sweep_custom_commands_combinations = self._create_sweep_combinations(test_config["sweep_custom_commands"])

            # Create all combinations of environment variables, arguments, and custom commands
            test_combinations = []
            for env_combo in sweep_envs_combinations:
                for arg_combo in sweep_args_combinations:
                    for custom_commands_combo in sweep_custom_commands_combinations:
                        combo = {
                            "env": env_combo.copy(),
                            "arg": arg_combo,
                            "custom_commands": custom_commands_combo,
                            "env_combo_index": sweep_envs_combinations.index(env_combo) + 1,
                            "arg_combo_index": sweep_args_combinations.index(arg_combo) + 1,
                            "custom_commands_combo_index": sweep_custom_commands_combinations.index(custom_commands_combo) + 1
                        }
                        test_combinations.append(combo)

            # Filter out combinations that match exceptions
            filtered_combinations = self._filter_combinations_by_exceptions(
                test_combinations,
                test_config["exceptions"]
            )

            tests.append({
                "name": test_config["name"],
                "binary": os.path.join(test_config["path"], test_config["binary"]),
                "combinations": filtered_combinations
            })

        return tests

    def _validate_default_config(self, config: Dict[str, Any]) -> None:
        """Validate that a default configuration meets requirements.

        A default configuration should:
        1. Only have envs in allowed_envs_to_sweep_on_default can have more than one value in sweep_envs
        2. Only have args in allowed_args_to_sweep_on_default can have more than one value in sweep_args
        3. All other sweep parameters must be single values (string or list with one element)

        If these conditions are not met, raise error and exit.
        """
        # Skip validation if not marked as default
        if not config.get("use_default_params", 0):
            return

        allowed_envs_to_sweep_on_default = config.get("allowed_envs_to_sweep_on_default", [])
        allowed_args_to_sweep_on_default = config.get("allowed_args_to_sweep_on_default", [])

        # Check test-specific sweep parameters
        for test in config.get("tests", []):
            test_sweep_envs = test.get("sweep_envs", {})
            for key, value in test_sweep_envs.items():
                # All sweep parameters in test sections must be single values
                if isinstance(value, list) and len(value) > 1 and key not in allowed_envs_to_sweep_on_default:
                    error_msg = f"environment variable {key} can't have multiple values in default configuration, found in test: {test.get('name', 'unknown')}"
                    raise ValueError(error_msg)

            # Check for test-specific sweep_args with multiple values
            test_sweep_args = test.get("sweep_args", {})
            for key, value in test_sweep_args.items():
                if isinstance(value, list) and len(value) > 1 and key not in allowed_args_to_sweep_on_default:
                    error_msg = f"argument {key} can't have multiple values in default configuration, found in test: {test.get('name', 'unknown')}"
                    raise ValueError(error_msg)

    def _create_sweep_combinations(self, sweep_dict: Dict[str, List[Any]], deal_with_default: bool = False) -> List[Dict[str, Any]]:
        """Create all possible combinations from a dictionary of sweep parameters.

        Args:
            sweep_dict: Dictionary where keys are parameter names and values are lists of possible values
            deal_with_default: Whether to handle 'default' values specially

        Returns:
            List of dictionaries, each containing one possible combination of parameters
        """
        if not sweep_dict:
            return [{}]

        # Ensure all values are lists
        sweep_dict = self._ensure_list_values(sweep_dict)

        # Validate pre_command and post_command have at most 1 value each
        for key in ['pre_command', 'post_command']:
            if key in sweep_dict and len(sweep_dict[key]) > 1:
                error_msg = f"Error: {key} can have a maximum of 1 value, but {len(sweep_dict[key])} values were provided. Failed to generate script."
                raise ValueError(error_msg)

        keys = sweep_dict.keys()
        values = sweep_dict.values()
        combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

        if deal_with_default:
            new_combinations = []
            for combo in combinations:
                new_combo = {}
                for key, value in combo.items():
                    if value.lower() == "default":
                        new_combo["x" + key] = value.lower()
                    else:
                        new_combo[key] = value
                new_combinations.append(new_combo)
            combinations = new_combinations

        return combinations

    def _filter_combinations_by_exceptions(self, combinations, exceptions) -> List[Dict[str, Any]]:
        """Filter combinations based on exceptions.

        Args:
            combinations: List of dictionaries, each containing environment and argument combinations
            exceptions: List of exception dictionaries with sweep_envs and sweep_args

        Returns:
            Filtered list of combinations that don't match any exception
        """
        filtered_combinations = []

        for combo in combinations:
            env_combo = combo.get("env", {})
            arg_combo = combo.get("arg", {})

            # Check if this combination matches any exception
            matches_exception = False

            for exception in exceptions:
                exception_envs = exception.get("sweep_envs", {})
                exception_args = exception.get("sweep_args", {})

                if not exception_envs and not exception_args:
                    continue

                # Check if environment variables match the exception
                env_vars_contain_exception = all(
                    k in env_combo and (env_combo[k] in v or v == [])
                    for k, v in exception_envs.items()
                )

                # Check if arguments match the exception
                args_contain_exception = all(
                    k in arg_combo and (arg_combo[k] in v or v == [])
                    for k, v in exception_args.items()
                )

                # If both environment and arguments match, this combination is excluded
                if env_vars_contain_exception and args_contain_exception:
                    matches_exception = True
                    break

            # If no exception matches, keep this combination
            if not matches_exception:
                filtered_combinations.append(combo)

        return filtered_combinations

    def _ensure_list_values(self, dictionary: Dict[str, Any]) -> Dict[str, Any]:
        """Convert single values to lists in a dictionary.

        Args:
            dictionary: Dictionary with potential single values

        Returns:
            Dictionary with all values as lists
        """
        return {k: v if isinstance(v, list) else [v] for k, v in dictionary.items()}

    def _merge_dictionaries(self, dict1: Dict[str, Any], dict2: Dict[str, Any], merge_strategy: str = "override") -> Dict[str, Any]:
        """Merge two dictionaries with list values based on the specified strategy.

        Args:
            dict1: First dictionary (typically common config)
            dict2: Second dictionary (typically test config)
            merge_strategy: Strategy for merging ("override" or "merge")

        Returns:
            Merged dictionary
        """
        if merge_strategy == "override":
            # Use dict2 values, falling back to dict1 for missing keys
            result = dict1.copy()  # Start with a copy of dict1
            result.update(dict2)   # Update with dict2 values
            return {k: v if isinstance(v, list) else [v] for k, v in result.items()}
        elif merge_strategy == "merge":
            # Merge values from both dictionaries
            result = {}
            # Get all unique keys from both dictionaries
            all_keys = set(dict1.keys())
            all_keys.update(dict2.keys())
            for key in all_keys:
                values1 = dict1.get(key, [])
                if not isinstance(values1, list):
                    values1 = [values1]
                values2 = dict2.get(key, [])
                if not isinstance(values2, list):
                    values2 = [values2]
                result[key] = list(set(values1 + values2))
            return result
        else:
            raise ValueError(f"Invalid merge strategy: {merge_strategy}. Must be 'override' or 'merge'.")

    def _process_exceptions(self, exceptions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process exceptions to ensure sweep environments and arguments are properly formatted.

        Args:
            exceptions: List of exception dictionaries

        Returns:
            Processed list of exceptions
        """
        for exception in exceptions:
            if "sweep_envs" in exception:
                exception["sweep_envs"] = {
                    k: v if isinstance(v, list) else [v]
                    for k, v in exception["sweep_envs"].items()
                }
            if "sweep_args" in exception:
                exception["sweep_args"] = {
                    k: v if isinstance(v, list) else [v]
                    for k, v in exception["sweep_args"].items()
                }
        return exceptions
