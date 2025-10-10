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

"""
Tests for plot configuration validation, specifically parameter conflict detection.
"""

import pytest
import tempfile
import os
from pydantic import ValidationError

from nvcomms_perf.report.models.plot_config import ChartType, ChartConfigLoader


class TestParameterConflictValidation:
    """Test parameter conflict validation in ChartType configuration."""

    def test_valid_chart_type_no_conflicts(self):
        """Test that a valid chart type with no parameter conflicts passes validation."""
        chart_config = {
            "id": "test_chart",
            "name": "test_chart",
            "description": "Test chart",
            "fixed_params": {"param1": "value1"},
            "chart_split_params": {"param2": "all"},
            "chart_combine_params": {"param3": "all"}
        }

        chart = ChartType(**chart_config)  # type: ignore
        assert chart.name == "test_chart"
        assert chart.fixed_params == {"param1": "value1"}
        assert chart.chart_split_params == {"param2": "all"}
        assert chart.chart_combine_params == {"param3": "all"}

    def test_fixed_and_split_param_conflict(self):
        """Test that a conflict between fixed_params and chart_split_params raises an error."""
        chart_config = {
            "id": "conflict_chart",
            "name": "conflict_chart",
            "description": "Chart with parameter conflict",
            "fixed_params": {"param1": "value1"},
            "chart_split_params": {"param1": "all"},  # Same parameter as fixed_params
            "chart_combine_params": {"param2": "all"}
        }

        with pytest.raises(ValidationError) as exc_info:
            ChartType(**chart_config)

        error_message = str(exc_info.value)
        assert "parameter conflicts" in error_message
        assert "param1" in error_message
        assert "fixed_params and chart_split_params" in error_message

    def test_fixed_and_combine_param_conflict(self):
        """Test that a conflict between fixed_params and chart_combine_params raises an error."""
        chart_config = {
            "id": "conflict_chart",
            "name": "conflict_chart",
            "description": "Chart with parameter conflict",
            "fixed_params": {"param1": "value1"},
            "chart_split_params": {"param2": "all"},
            "chart_combine_params": {"param1": "all"}  # Same parameter as fixed_params
        }

        with pytest.raises(ValidationError) as exc_info:
            ChartType(**chart_config)

        error_message = str(exc_info.value)
        assert "parameter conflicts" in error_message
        assert "param1" in error_message
        assert "fixed_params and chart_combine_params" in error_message

    def test_split_and_combine_param_conflict(self):
        """Test that a conflict between chart_split_params and chart_combine_params raises an error."""
        chart_config = {
            "id": "conflict_chart",
            "name": "conflict_chart",
            "description": "Chart with parameter conflict",
            "fixed_params": {"param1": "value1"},
            "chart_split_params": {"param2": "all"},
            "chart_combine_params": {"param2": "all"}  # Same parameter as chart_split_params
        }

        with pytest.raises(ValidationError) as exc_info:
            ChartType(**chart_config)

        error_message = str(exc_info.value)
        assert "parameter conflicts" in error_message
        assert "param2" in error_message
        assert "chart_split_params and chart_combine_params" in error_message

    def test_multiple_conflicts(self):
        """Test that multiple parameter conflicts are detected."""
        chart_config = {
            "id": "multiple_conflicts",
            "name": "multiple_conflicts",
            "description": "Chart with multiple parameter conflicts",
            "fixed_params": {"param1": "value1", "param2": "value2"},
            "chart_split_params": {"param1": "all", "param3": "all"},  # param1 conflict
            "chart_combine_params": {"param2": "all", "param3": "all"}  # param2 and param3 conflicts
        }

        with pytest.raises(ValidationError) as exc_info:
            ChartType(**chart_config)

        error_message = str(exc_info.value)
        # Should detect at least one of the conflicts
        assert "parameter conflicts" in error_message

    def test_empty_parameter_sections(self):
        """Test that empty parameter sections don't cause conflicts."""
        chart_config = {
            "id": "empty_params",
            "name": "empty_params",
            "description": "Chart with empty parameter sections",
            "fixed_params": {},
            "chart_split_params": {},
            "chart_combine_params": {}
        }

        chart = ChartType(**chart_config)  # type: ignore
        assert chart.fixed_params == {}
        assert chart.chart_split_params == {}
        assert chart.chart_combine_params == {}

    def test_missing_parameter_sections(self):
        """Test that missing parameter sections don't cause conflicts."""
        chart_config = {
            "id": "missing_params",
            "name": "missing_params",
            "description": "Chart with missing parameter sections"
        }

        chart = ChartType(**chart_config)  # type: ignore
        assert chart.fixed_params == {}
        assert chart.chart_split_params == {}
        assert chart.chart_combine_params == {}

    def test_real_world_example_conflict(self):
        """Test with a real-world example that would cause a conflict."""
        chart_config = {
            "id": "algo_comparison",
            "name": "algo_comparison",
            "description": "Algorithm comparison chart",
            "performance_columns": ["max_bus_bw"],
            "fixed_params": {"NCCL_ALGO": "DEFAULT"},
            "default_values": {},
            "chart_split_params": {"num_nodes": "all"},
            "chart_combine_params": {"NCCL_ALGO": "all"}  # Conflict with fixed_params
        }

        with pytest.raises(ValidationError) as exc_info:
            ChartType(**chart_config)

        error_message = str(exc_info.value)
        assert "NCCL_ALGO" in error_message
        assert "fixed_params and chart_combine_params" in error_message

    def test_fixed_and_default_values_conflict(self):
        """Test that a conflict between fixed_params and default_values raises an error."""
        chart_config = {
            "id": "conflict_chart",
            "name": "conflict_chart",
            "description": "Chart with parameter conflict",
            "performance_columns": ["max_bus_bw"],
            "fixed_params": {"param1": "value1"},
            "default_values": {"param1": "default_value"},  # Same parameter as fixed_params
            "chart_split_params": {"param2": "all"},
            "chart_combine_params": {"param3": "all"}
        }

        with pytest.raises(ValidationError) as exc_info:
            ChartType(**chart_config)

        error_message = str(exc_info.value)
        assert "parameter conflicts" in error_message
        assert "param1" in error_message
        assert "fixed_params and default_values" in error_message

    def test_combine_and_default_values_conflict(self):
        """Test that a conflict between chart_combine_params and default_values raises an error."""
        chart_config = {
            "id": "conflict_chart",
            "name": "conflict_chart",
            "description": "Chart with parameter conflict",
            "performance_columns": ["max_bus_bw"],
            "fixed_params": {"param1": "value1"},
            "default_values": {"param2": "default_value"},
            "chart_split_params": {"param3": "all"},
            "chart_combine_params": {"param2": "all"}  # Same parameter as default_values
        }

        with pytest.raises(ValidationError) as exc_info:
            ChartType(**chart_config)

        error_message = str(exc_info.value)
        assert "parameter conflicts" in error_message
        assert "param2" in error_message
        assert "chart_combine_params and default_values" in error_message

    def test_toml_file_with_conflicts(self):
        """Test that validation catches conflicts in TOML file loading."""
        # Create a temporary TOML file with parameter conflicts
        problematic_toml = """
[[chart]]
id = "conflict_test"
name = "Conflict Test Chart"
description = "Chart with parameter conflicts"
performance_columns = ["max_bus_bw"]

[chart.fixed_params]
NCCL_ALGO = "DEFAULT"

[chart.chart_split_params]
NCCL_ALGO = "all"  # Conflict: same parameter in fixed_params and chart_split_params

[chart.chart_combine_params]
local_register = "all"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(problematic_toml)
            temp_file = f.name

        try:
            with pytest.raises(ValidationError) as exc_info:
                ChartConfigLoader.load_from_toml(temp_file)

            error_message = str(exc_info.value)
            assert "parameter conflicts" in error_message
            assert "NCCL_ALGO" in error_message
            assert "fixed_params and chart_split_params" in error_message

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.unlink(temp_file)
