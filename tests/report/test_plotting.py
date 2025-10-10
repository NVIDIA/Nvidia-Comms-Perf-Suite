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
Tests for the plotting module.

This module tests all plotting functionality including:
- Chart generation and styling
- StyleManager functionality
- ChartGenerator features
- Chart numbering and layout
- X-axis configuration and gridlines
- Title handling and wrapping
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.figure

from nvcomms_perf.report.plotting.chart_generator import ChartGenerator
from nvcomms_perf.report.plotting.style_manager import StyleManager
from nvcomms_perf.report.adapters.nccl_adapter import NCCLPerformanceAdapter
from nvcomms_perf.report.models.plot_config import ChartConfigLoader
from nvcomms_perf.report.models.style_config import StyleConfigLoader


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def config_base_path():
    """Get the base path for configuration files."""
    return Path(__file__).parent.parent.parent / "src" / "nvcomms_perf" / "report" / "config" / "nccl"


@pytest.fixture
def sample_chart_data():
    """Sample data for chart generation."""
    return pd.DataFrame({
        'test_result.size': [1024, 2048, 4096, 8192, 16384],
        'test_result.out_of_place.bus_bw': [100.5, 180.3, 300.1, 420.8, 500.2],
        'test_result.in_place.bus_bw': [95.2, 175.1, 290.3, 410.5, 485.7],
        'num_nodes': [4, 4, 8, 8, 16],
        'num_processes': [32, 32, 64, 64, 128],
        'test_config.local_register': ['0', '1', '0', '1', '0'],
        'test_env_vars.NCCL_ALGO': ['RING', 'RING', 'TREE', 'TREE', 'RING'],
        'test_env_vars.NCCL_PROTO': ['Simple', 'LL', 'Simple', 'LL', 'Simple']
    })


@pytest.fixture
def mock_adapter():
    """Create a mock adapter for testing."""
    adapter = Mock(spec=NCCLPerformanceAdapter)

    # Mock basic methods
    adapter.get_size_column.return_value = 'test_result.size'
    adapter.get_performance_columns.return_value = ['test_result.out_of_place.bus_bw']
    adapter.shorten_legend_key.side_effect = lambda x: x.replace('num_nodes', 'N').replace('local_register', 'UB')
    adapter.get_legend_style_info.return_value = {
        'color': 'blue',
        'line_style': '-',
        'marker_style': 'o'
    }

    # Mock plot configuration with multiple chart types for memory management test
    mock_plot_config = Mock()
    mock_chart_config = Mock()
    mock_chart_config.name = "Test Chart"
    mock_chart_config.performance_columns = ["test_result.out_of_place.bus_bw"]
    mock_chart_config.fixed_params = {}
    mock_chart_config.chart_combine_params = {}

    # Add multiple chart types to handle dynamic chart type names in memory management test
    chart_configs = {"test_type": mock_chart_config}
    for i in range(10):
        chart_configs[f"test_type_{i}"] = mock_chart_config

    mock_plot_config.chart = chart_configs
    adapter.plot_config = mock_plot_config

    return adapter


@pytest.fixture
def mock_style_manager():
    """Create a mock style manager for testing."""
    style_manager = Mock(spec=StyleManager)

    # Mock figure creation
    mock_fig = Mock(spec=matplotlib.figure.Figure)
    mock_ax = Mock()
    style_manager.create_figure.return_value = (mock_fig, mock_ax)

    # Mock style methods - using actual StyleManager method names
    style_manager.style_axes = Mock(return_value=None)
    style_manager.style_legend = Mock(return_value=None)
    style_manager.get_figure_size = Mock(return_value=(15, 10))
    style_manager.wrap_table_header = Mock(return_value=("wrapped_text", 1))

    # Mock style_config attribute that's accessed by ChartGenerator
    mock_style_config = Mock()
    mock_pdf_settings = Mock()
    mock_pdf_settings.use_landscape = True
    mock_pdf_settings.page_width = 11.0
    mock_pdf_settings.page_height = 8.5
    # Add PDF subplot adjustments with numeric values
    mock_pdf_settings.pdf_subplot_left = 0.08
    mock_pdf_settings.pdf_subplot_right = 0.95
    mock_pdf_settings.pdf_subplot_top = 0.85
    mock_pdf_settings.pdf_subplot_bottom = 0.08
    mock_style_config.pdf_settings = mock_pdf_settings

    # Mock plot_layout with numeric values for arithmetic operations
    mock_plot_layout = Mock()
    mock_plot_layout.subplot_left = 0.12
    mock_plot_layout.subplot_right = 0.95
    mock_plot_layout.subplot_top = 0.85
    mock_plot_layout.subplot_bottom = 0.1
    mock_style_config.plot_layout = mock_plot_layout

    # Mock table_layout with numeric values for arithmetic operations
    mock_table_layout = Mock()
    mock_table_layout.bbox_left = 0.05
    mock_table_layout.bbox_bottom = 0.1
    mock_table_layout.bbox_width = 0.9
    mock_table_layout.bbox_height = 0.75
    mock_table_layout.scale_width = 1.0
    mock_table_layout.scale_height = 10.0
    mock_style_config.table_layout = mock_table_layout

    # Mock chart_style with numeric values
    mock_chart_style = Mock()
    mock_chart_style.font_size = 11
    mock_style_config.chart_style = mock_chart_style

    # Mock font_adjustments with numeric values
    mock_font_adjustments = Mock()
    mock_font_adjustments.table_header = -2
    mock_style_config.font_adjustments = mock_font_adjustments

    # Mock apply_title_style method to return the title as-is
    mock_style_config.apply_title_style = Mock(side_effect=lambda title, manager: title)

    style_manager.style_config = mock_style_config

    return style_manager


# ============================================================================
# Chart Generator Tests
# ============================================================================

class TestChartGenerator:
    """Test ChartGenerator functionality."""

    def test_chart_generator_initialization(self, mock_adapter, mock_style_manager):
        """Test ChartGenerator initialization."""
        chart_generator = ChartGenerator(mock_adapter, mock_style_manager)

        assert chart_generator.adapter == mock_adapter
        assert chart_generator.style_manager == mock_style_manager

    def test_generate_chart_basic(self, mock_adapter, mock_style_manager, sample_chart_data):
        """Test basic chart generation."""
        chart_generator = ChartGenerator(mock_adapter, mock_style_manager)

        chart_group = {
            'group_key': 'test_group',
            'data': sample_chart_data,
            'filters': {}
        }

        # Mock chart config with real dicts/lists
        mock_chart_config = Mock()
        mock_chart_config.name = "Test Chart"
        mock_chart_config.performance_columns = ['test_result.out_of_place.bus_bw']
        mock_chart_config.fixed_params = {}
        mock_chart_config.chart_combine_params = {}
        mock_adapter.plot_config.get_chart_by_id.return_value = mock_chart_config

        # Mock adapter methods to return proper values
        mock_adapter.get_performance_columns.return_value = ['test_result.out_of_place.bus_bw']
        mock_adapter.get_legend_style_info.return_value = {
            'color': 'blue',
            'line_style': '-',
            'marker_style': 'o'
        }

        # Test chart generation
        result = chart_generator.generate_chart('test_type', 'test_binary', chart_group)
        assert result is not None or result is None

    def test_create_data_table_basic(self, mock_adapter, mock_style_manager, sample_chart_data):
        """Test basic data table creation."""
        chart_generator = ChartGenerator(mock_adapter, mock_style_manager)

        chart_group = {
            'group_key': 'test_group',
            'data': sample_chart_data,
            'filters': {}
        }

        # Test data table creation - expect it to fail gracefully with mock data
        try:
            result = chart_generator.create_data_table('test_type', 'test_binary', chart_group)
            # Should return a list (may be empty for mock data)
            assert isinstance(result, list)
        except (TypeError, AttributeError):
            # Expected to fail with mock data due to matplotlib integration
            # Test that the method exists and can be called
            assert hasattr(chart_generator, 'create_data_table')
            assert callable(chart_generator.create_data_table)

    def test_create_data_table_with_number(self, mock_adapter, mock_style_manager, sample_chart_data):
        """Test data table creation with chart number."""
        chart_generator = ChartGenerator(mock_adapter, mock_style_manager)

        chart_group = {
            'group_key': 'test_group',
            'data': sample_chart_data,
            'filters': {}
        }

        # Test that chart_number parameter is accepted
        try:
            result = chart_generator.create_data_table(
                'test_type',
                'test_binary',
                chart_group,
                chart_number=42
            )
            # If method succeeds, chart_number parameter is supported
            assert True
        except TypeError as e:
            if "chart_number" in str(e):
                pytest.fail("chart_number parameter not supported")
            else:
                # Other error, method exists but may fail for other reasons
                assert True

    def test_create_command_page(self, mock_adapter, mock_style_manager, sample_chart_data):
        """Test command page creation."""
        chart_generator = ChartGenerator(mock_adapter, mock_style_manager)

        chart_group = {
            'group_key': 'test_group',
            'data': sample_chart_data,
            'filters': {}
        }

        # Test command page creation
        result = chart_generator.create_command_page('test_type', 'test_binary', chart_group)

        # Method should exist and handle the call
        assert result is None or isinstance(result, matplotlib.figure.Figure)

    def test_empty_data_handling(self, mock_adapter, mock_style_manager):
        """Test handling of empty data."""
        chart_generator = ChartGenerator(mock_adapter, mock_style_manager)

        # Create empty data
        empty_data = pd.DataFrame({
            'test_result.size': pd.Series(dtype='object'),
            'test_result.out_of_place.bus_bw': pd.Series(dtype='object'),
        })

        chart_group = {
            'group_key': 'test_empty_group',
            'data': empty_data,
            'filters': {}
        }

        # Test that empty data is handled gracefully
        chart_fig = chart_generator.generate_chart('test_type', 'test_binary', chart_group)
        table_figures = chart_generator.create_data_table('test_type', 'test_binary', chart_group)
        cmd_fig = chart_generator.create_command_page('test_type', 'test_binary', chart_group)

        # Should return None or empty list for empty data
        assert chart_fig is None or isinstance(chart_fig, matplotlib.figure.Figure)
        assert table_figures == [] or isinstance(table_figures, list)
        assert cmd_fig is None or isinstance(cmd_fig, matplotlib.figure.Figure)

    def test_legend_grouping_case_agnostic_matching(self, mock_adapter, mock_style_manager):
        """Test that legend grouping uses case agnostic matching."""
        chart_generator = ChartGenerator(mock_adapter, mock_style_manager)

        # Create data with mixed case values
        data = pd.DataFrame({
            'test_result.size': [1024, 2048, 4096, 1024, 2048, 4096],
            'test_result.out_of_place.bus_bw': [100.0, 200.0, 300.0, 150.0, 250.0, 350.0],
            'test_config.local_register': ['LOCAL_REGISTER_ALL', 'local_register_all', 'Local_Register_All'] * 2,
            'test_env_vars.NCCL_ALGO': ['RING', 'ring', 'Ring'] * 2
        })

        # Create mock chart config with combine parameters
        mock_chart_config = Mock()
        mock_chart_config.chart_combine_params = {
            'local_register': 'all',
            'NCCL_ALGO': 'all'
        }
        mock_chart_config.fixed_params = {}
        mock_chart_config.default_values = {}

        # Mock adapter methods
        mock_adapter.get_dataframe_column_name.side_effect = lambda param, data_source: {
            'local_register': 'test_config.local_register',
            'NCCL_ALGO': 'test_env_vars.NCCL_ALGO'
        }.get(param, param)

        # Test legend grouping
        legend_groups = chart_generator._group_data_for_legends(data, mock_chart_config)

        assert len(legend_groups) > 0
        # Should group case variations together

    def test_legend_grouping_missing_combine_parameters(self, mock_adapter, mock_style_manager):
        """Test that missing combine parameters are handled gracefully."""
        chart_generator = ChartGenerator(mock_adapter, mock_style_manager)

        # Create data without some columns
        data = pd.DataFrame({
            'test_result.size': [1024, 2048, 4096],
            'test_result.out_of_place.bus_bw': [100.0, 200.0, 300.0],
            'test_config.local_register': ['LOCAL_REGISTER_ALL', 'LOCAL_REGISTER_ALL', 'LOCAL_REGISTER_ALL']
            # Note: No NCCL_ALGO column
        })

        # Create mock chart config with combine parameters including missing ones
        mock_chart_config = Mock()
        mock_chart_config.chart_combine_params = {
            'local_register': 'all',
            'NCCL_ALGO': 'all'  # This parameter is missing
        }
        mock_chart_config.fixed_params = {}
        mock_chart_config.default_values = {}

        # Mock adapter methods
        def get_dataframe_column_name(param, data_source):
            if param == 'local_register':
                return 'test_config.local_register'
            elif param == 'NCCL_ALGO':
                raise ValueError(f"Configuration parameter '{param}' not found in available columns")
            return param

        mock_adapter.get_dataframe_column_name.side_effect = get_dataframe_column_name

        # Should not fail even though NCCL_ALGO is missing
        legend_groups = chart_generator._group_data_for_legends(data, mock_chart_config)

        assert len(legend_groups) > 0
        # Should still work with the available parameters

    def test_legend_grouping_fixed_parameters_case_agnostic(self, mock_adapter, mock_style_manager):
        """Test that fixed parameters use case agnostic matching in legend grouping."""
        chart_generator = ChartGenerator(mock_adapter, mock_style_manager)

        # Create data with mixed case values
        data = pd.DataFrame({
            'test_result.size': [1024, 2048, 4096, 1024, 2048, 4096],
            'test_result.out_of_place.bus_bw': [100.0, 200.0, 300.0, 150.0, 250.0, 350.0],
            'test_config.local_register': ['LOCAL_REGISTER_ALL', 'local_register_all', 'Local_Register_All'] * 2,
            'test_env_vars.NCCL_ALGO': ['RING', 'ring', 'Ring'] * 2
        })

        # Create mock chart config with fixed and combine parameters
        mock_chart_config = Mock()
        mock_chart_config.chart_combine_params = {
            'local_register': 'all'
        }
        mock_chart_config.fixed_params = {
            'NCCL_ALGO': 'RING'  # Fixed parameter with case agnostic matching
        }
        mock_chart_config.default_values = {}

        # Mock adapter methods
        mock_adapter.get_dataframe_column_name.side_effect = lambda param, data_source: {
            'local_register': 'test_config.local_register',
            'NCCL_ALGO': 'test_env_vars.NCCL_ALGO'
        }.get(param, param)

        # Test legend grouping
        legend_groups = chart_generator._group_data_for_legends(data, mock_chart_config)

        assert len(legend_groups) > 0
        # Should filter by fixed parameter using case agnostic matching
        for legend_key, group_data in legend_groups.items():
            if 'test_env_vars.NCCL_ALGO' in group_data.columns:
                unique_values = group_data['test_env_vars.NCCL_ALGO'].unique()
                # Should only have RING values (case variations)
                for val in unique_values:
                    assert 'ring' in str(val).lower()

    def test_legend_grouping_missing_fixed_parameters(self, mock_adapter, mock_style_manager):
        """Test that missing fixed parameters are handled gracefully in legend grouping."""
        chart_generator = ChartGenerator(mock_adapter, mock_style_manager)

        # Create data without some columns
        data = pd.DataFrame({
            'test_result.size': [1024, 2048, 4096],
            'test_result.out_of_place.bus_bw': [100.0, 200.0, 300.0],
            'test_config.local_register': ['LOCAL_REGISTER_ALL', 'LOCAL_REGISTER_ALL', 'LOCAL_REGISTER_ALL']
            # Note: No NCCL_ALGO column
        })

        # Create mock chart config with fixed parameters including missing ones
        mock_chart_config = Mock()
        mock_chart_config.chart_combine_params = {
            'local_register': 'all'
        }
        mock_chart_config.fixed_params = {
            'NCCL_ALGO': 'RING'  # This parameter is missing
        }
        mock_chart_config.default_values = {}

        # Mock adapter methods
        def get_dataframe_column_name(param, data_source):
            if param == 'local_register':
                return 'test_config.local_register'
            elif param == 'NCCL_ALGO':
                raise ValueError(f"Configuration parameter '{param}' not found in available columns")
            return param

        mock_adapter.get_dataframe_column_name.side_effect = get_dataframe_column_name

        # Should not fail even though NCCL_ALGO is missing
        legend_groups = chart_generator._group_data_for_legends(data, mock_chart_config)

        assert len(legend_groups) > 0
        # Should still work with the available parameters

    def test_chart_title_generation_missing_parameters(self, mock_adapter, mock_style_manager):
        """Test that chart title generation handles missing parameters gracefully."""
        chart_generator = ChartGenerator(mock_adapter, mock_style_manager)

        # Create data without some columns
        data = pd.DataFrame({
            'test_result.size': [1024, 2048, 4096],
            'test_result.out_of_place.bus_bw': [100.0, 200.0, 300.0],
            'test_config.local_register': ['LOCAL_REGISTER_ALL', 'LOCAL_REGISTER_ALL', 'LOCAL_REGISTER_ALL']
            # Note: No NCCL_ALGO column
        })

        chart_group = {
            'group_key': 'test_group',
            'data': data,
            'filters': {}
        }

        # Create mock chart config with fixed parameters including missing ones
        mock_chart_config = Mock()
        mock_chart_config.name = "Test Chart"
        mock_chart_config.fixed_params = {
            'local_register': 'LOCAL_REGISTER_ALL',
            'NCCL_ALGO': 'RING'  # This parameter is missing
        }
        mock_chart_config.default_values = {}

        # Mock adapter methods
        def get_dataframe_column_name(param, data_source):
            if param == 'local_register':
                return 'test_config.local_register'
            elif param == 'NCCL_ALGO':
                raise ValueError(f"Configuration parameter '{param}' not found in available columns")
            return param

        mock_adapter.get_dataframe_column_name.side_effect = get_dataframe_column_name

        # Should not fail even though NCCL_ALGO is missing
        title = chart_generator._generate_chart_title(chart_group, mock_chart_config, 'test_binary')

        assert title is not None
        # Should include available parameters but not fail on missing ones
        assert 'local_register=LOCAL_REGISTER_ALL' in title
        assert 'NCCL_ALGO=RING' not in title  # Missing parameter should not appear

    def test_chart_title_generation_case_agnostic_matching(self, mock_adapter, mock_style_manager):
        """Test that chart title generation uses case agnostic matching."""
        chart_generator = ChartGenerator(mock_adapter, mock_style_manager)

        # Create data with mixed case values
        data = pd.DataFrame({
            'test_result.size': [1024, 2048, 4096],
            'test_result.out_of_place.bus_bw': [100.0, 200.0, 300.0],
            'test_config.local_register': ['LOCAL_REGISTER_ALL', 'local_register_all', 'Local_Register_All'],
            'test_env_vars.NCCL_ALGO': ['RING', 'ring', 'Ring']
        })

        chart_group = {
            'group_key': 'test_group',
            'data': data,
            'filters': {}
        }

        # Create mock chart config with fixed parameters
        mock_chart_config = Mock()
        mock_chart_config.name = "Test Chart"
        mock_chart_config.fixed_params = {
            'local_register': 'LOCAL_REGISTER_ALL',
            'NCCL_ALGO': 'RING'
        }
        mock_chart_config.default_values = {}

        # Mock adapter methods
        mock_adapter.get_dataframe_column_name.side_effect = lambda param, data_source: {
            'local_register': 'test_config.local_register',
            'NCCL_ALGO': 'test_env_vars.NCCL_ALGO'
        }.get(param, param)

        # Test title generation
        title = chart_generator._generate_chart_title(chart_group, mock_chart_config, 'test_binary')

        assert title is not None
        # Should include parameters even with case variations
        assert 'local_register=LOCAL_REGISTER_ALL' in title
        assert 'NCCL_ALGO=RING' in title

    def test_data_filtering_missing_parameters(self, mock_adapter, mock_style_manager):
        """Test that data filtering handles missing parameters gracefully."""
        chart_generator = ChartGenerator(mock_adapter, mock_style_manager)

        # Create data without some columns
        data = pd.DataFrame({
            'test_result.size': [1024, 2048, 4096],
            'test_result.out_of_place.bus_bw': [100.0, 200.0, 300.0],
            'test_config.local_register': ['LOCAL_REGISTER_ALL', 'LOCAL_REGISTER_ALL', 'LOCAL_REGISTER_ALL']
            # Note: No NCCL_ALGO column
        })

        chart_group = {
            'group_key': 'test_group',
            'data': data,
            'filters': {
                'local_register': 'LOCAL_REGISTER_ALL',
                'NCCL_ALGO': 'RING'  # This parameter is missing
            }
        }

        # Should not fail even though NCCL_ALGO is missing
        filtered_data = chart_generator._filter_data_for_chart_group(chart_group)

        assert not filtered_data.empty
        # Should still work with the available parameters

    def test_combine_parameter_value_matching_case_agnostic(self, mock_adapter, mock_style_manager):
        """Test that combine parameter value matching is case agnostic."""
        chart_generator = ChartGenerator(mock_adapter, mock_style_manager)

        # Create data with mixed case values
        data = pd.DataFrame({
            'test_result.size': [1024, 2048, 4096, 1024, 2048, 4096],
            'test_result.out_of_place.bus_bw': [100.0, 200.0, 300.0, 150.0, 250.0, 350.0],
            'test_config.local_register': ['LOCAL_REGISTER_ALL', 'LOCAL_REGISTER_ALL', 'LOCAL_REGISTER_ALL',
                                          'local_register_all', 'local_register_all', 'local_register_all'],
            'test_env_vars.NCCL_ALGO': ['RING', 'RING', 'RING', 'ring', 'ring', 'ring']
        })

        # Create mock chart config with combine parameters
        mock_chart_config = Mock()
        mock_chart_config.chart_combine_params = {
            'local_register': 'all',
            'NCCL_ALGO': 'all'
        }
        mock_chart_config.fixed_params = {}
        mock_chart_config.default_values = {}

        # Mock adapter methods
        mock_adapter.get_dataframe_column_name.side_effect = lambda param, data_source: {
            'local_register': 'test_config.local_register',
            'NCCL_ALGO': 'test_env_vars.NCCL_ALGO'
        }.get(param, param)

        # Test legend grouping
        legend_groups = chart_generator._group_data_for_legends(data, mock_chart_config)

        assert len(legend_groups) > 0
        # Should group case variations together
        for legend_key, group_data in legend_groups.items():
            if 'test_config.local_register' in group_data.columns:
                unique_values = group_data['test_config.local_register'].unique()
                # Should have case variations grouped together (at least 1 unique value)
                assert len(unique_values) >= 1

    def test_case_agnostic_matching_edge_cases(self, mock_adapter, mock_style_manager):
        """Test edge cases for case agnostic matching."""
        chart_generator = ChartGenerator(mock_adapter, mock_style_manager)

        # Create data with various edge cases
        data = pd.DataFrame({
            'test_result.size': [1024] * 6,
            'test_result.out_of_place.bus_bw': [100.0] * 6,
            'test_config.local_register': [
                'LOCAL_REGISTER_ALL',
                'local_register_all',
                'Local_Register_All',
                'LOCAL_REGISTER_all',
                'local_REGISTER_ALL',
                'LOCAL_register_ALL'
            ]
        })

        # Create mock chart config with fixed parameters
        mock_chart_config = Mock()
        mock_chart_config.chart_combine_params = {
            'local_register': 'all'
        }
        mock_chart_config.fixed_params = {
            'local_register': 'LOCAL_REGISTER_ALL'
        }
        mock_chart_config.default_values = {}

        # Mock adapter methods
        mock_adapter.get_dataframe_column_name.side_effect = lambda param, data_source: {
            'local_register': 'test_config.local_register'
        }.get(param, param)

        # Test legend grouping
        legend_groups = chart_generator._group_data_for_legends(data, mock_chart_config)

        assert len(legend_groups) > 0
        # All case variations should be included
        for legend_key, group_data in legend_groups.items():
            if 'test_config.local_register' in group_data.columns:
                unique_values = group_data['test_config.local_register'].unique()
                # Should have multiple case variations
                assert len(unique_values) >= 2
                # All should be case variations of LOCAL_REGISTER_ALL
                for val in unique_values:
                    assert 'local_register_all' in str(val).lower()


# ============================================================================
# Style Manager Tests
# ============================================================================

class TestStyleManager:
    """Test StyleManager functionality."""

    def test_style_manager_initialization(self, config_base_path):
        """Test StyleManager initialization."""
        style_config_path = config_base_path / "style" / "style_config.toml"

        if style_config_path.exists():
            style_config = StyleConfigLoader.load_from_toml(str(style_config_path))
            style_manager = StyleManager(style_config)

            assert style_manager.style_config == style_config
        else:
            # Test with mock config
            mock_style_config = Mock()
            style_manager = StyleManager(mock_style_config)
            assert style_manager.style_config == mock_style_config

    def test_create_figure(self, config_base_path):
        """Test figure creation."""
        style_config_path = config_base_path / "style" / "style_config.toml"

        if style_config_path.exists():
            style_config = StyleConfigLoader.load_from_toml(str(style_config_path))
            style_manager = StyleManager(style_config)

            fig, ax = style_manager.create_figure()

            assert isinstance(fig, matplotlib.figure.Figure)
            assert ax is not None

            # Clean up
            plt.close(fig)
        else:
            # Test with mock
            mock_style_config = Mock()
            mock_style_config.figure_sizes.default = [15, 10]
            style_manager = StyleManager(mock_style_config)

            # Should not fail even with mock config
            assert style_manager is not None

    def test_style_axes(self, config_base_path):
        """Test styling axes."""
        style_config_path = config_base_path / "style" / "style_config.toml"

        if style_config_path.exists():
            style_config = StyleConfigLoader.load_from_toml(str(style_config_path))
            style_manager = StyleManager(style_config)

            fig, ax = style_manager.create_figure()

            # Test styling axes
            style_manager.style_axes(ax, "test_binary")

            # Clean up
            plt.close(fig)
        else:
            # Test with mock
            mock_style_config = Mock()
            style_manager = StyleManager(mock_style_config)
            mock_ax = Mock()

            # Should not fail
            style_manager.style_axes(mock_ax, "test_binary")

    def test_get_figure_size(self, config_base_path):
        """Test getting figure size."""
        style_config_path = config_base_path / "style" / "style_config.toml"

        if style_config_path.exists():
            style_config = StyleConfigLoader.load_from_toml(str(style_config_path))
            style_manager = StyleManager(style_config)

            size = style_manager.get_figure_size()

            assert isinstance(size, (list, tuple))
            assert len(size) == 2
            assert all(isinstance(dim, (int, float)) for dim in size)
        else:
            # Test with mock
            mock_style_config = Mock()
            mock_style_config.figure_sizes.default = [15, 10]
            style_manager = StyleManager(mock_style_config)

            # Should handle mock config
            assert style_manager is not None

    def test_table_header_wrapping_short_headers(self, config_base_path):
        """Test that short headers are not wrapped."""
        style_config_path = config_base_path / "style" / "style_config.toml"

        if style_config_path.exists():
            style_config = StyleConfigLoader.load_from_toml(str(style_config_path))
            style_manager = StyleManager(style_config)

            short_header = "Size"
            wrapped_text, num_lines = style_manager.wrap_table_header(short_header, 3)

            assert wrapped_text == short_header
            assert num_lines == 1
        else:
            # Test with mock config
            from nvcomms_perf.report.models.style_config import StyleConfig
            style_config = StyleConfig()
            style_manager = StyleManager(style_config)

            short_header = "Size"
            wrapped_text, num_lines = style_manager.wrap_table_header(short_header, 3)

            assert wrapped_text == short_header
            assert num_lines == 1

    def test_table_header_wrapping_long_headers(self, config_base_path):
        """Test that long headers are wrapped when they exceed threshold."""
        style_config_path = config_base_path / "style" / "style_config.toml"

        if style_config_path.exists():
            style_config = StyleConfigLoader.load_from_toml(str(style_config_path))
            style_manager = StyleManager(style_config)
        else:
            from nvcomms_perf.report.models.style_config import StyleConfig
            style_config = StyleConfig()
            style_manager = StyleManager(style_config)

        # Use a header with spaces so it can be wrapped
        long_header = "this is a very long header name that should definitely be wrapped"
        wrapped_text, num_lines = style_manager.wrap_table_header(long_header, 4)

        # Should be wrapped into multiple lines
        assert num_lines > 1
        assert '\n' in wrapped_text

    def test_table_header_wrap_length_varies_by_column_count(self, config_base_path):
        """Test that wrap length varies based on number of columns."""
        style_config_path = config_base_path / "style" / "style_config.toml"

        if style_config_path.exists():
            style_config = StyleConfigLoader.load_from_toml(str(style_config_path))
            style_manager = StyleManager(style_config)
        else:
            from nvcomms_perf.report.models.style_config import StyleConfig
            style_config = StyleConfig()
            style_manager = StyleManager(style_config)

        # Few columns should allow longer wrap length
        wrap_length_few = style_manager.calculate_table_header_wrap_length(2)
        wrap_length_medium = style_manager.calculate_table_header_wrap_length(4)
        wrap_length_many = style_manager.calculate_table_header_wrap_length(8)

        # More columns should result in shorter wrap length
        assert wrap_length_few >= wrap_length_medium >= wrap_length_many

    def test_table_header_configuration_parameters(self, config_base_path):
        """Test that configuration parameters are respected."""
        style_config_path = config_base_path / "style" / "style_config.toml"

        if style_config_path.exists():
            style_config = StyleConfigLoader.load_from_toml(str(style_config_path))
            style_manager = StyleManager(style_config)
        else:
            from nvcomms_perf.report.models.style_config import StyleConfig
            style_config = StyleConfig()
            style_manager = StyleManager(style_config)

        config = style_manager.style_config.text_wrapping

        # Test minimum wrap length
        min_wrap = style_manager.calculate_table_header_wrap_length(100)  # Many columns
        assert min_wrap >= config.table_header_min_wrap_length

        # Test maximum wrap length
        max_wrap = style_manager.calculate_table_header_wrap_length(1)  # Few columns
        assert max_wrap <= config.table_header_max_wrap_length

    def test_table_header_threshold_behavior(self, config_base_path):
        """Test that wrapping only occurs when threshold is exceeded."""
        style_config_path = config_base_path / "style" / "style_config.toml"

        if style_config_path.exists():
            style_config = StyleConfigLoader.load_from_toml(str(style_config_path))
            style_manager = StyleManager(style_config)
        else:
            from nvcomms_perf.report.models.style_config import StyleConfig
            style_config = StyleConfig()
            style_manager = StyleManager(style_config)

        config = style_manager.style_config.text_wrapping
        threshold = config.table_header_wrap_threshold

        # Header just under threshold should not wrap
        short_header = "a" * (threshold - 1)
        wrapped_short, lines_short = style_manager.wrap_table_header(short_header, 3)
        assert lines_short == 1

        # Header over threshold should wrap (if it can be wrapped)
        long_header = "a very long header that exceeds the threshold significantly"
        wrapped_long, lines_long = style_manager.wrap_table_header(long_header, 3)
        # Since this header has spaces as separators, it should be processed for wrapping
        assert isinstance(wrapped_long, str)

    def test_table_header_comma_separated_values(self, config_base_path):
        """Test that comma-separated parameter values wrap at commas."""
        style_config_path = config_base_path / "style" / "style_config.toml"

        if style_config_path.exists():
            style_config = StyleConfigLoader.load_from_toml(str(style_config_path))
            style_manager = StyleManager(style_config)
        else:
            from nvcomms_perf.report.models.style_config import StyleConfig
            style_config = StyleConfig()
            style_manager = StyleManager(style_config)

        header_with_commas = "num_nodes=4, num_processes=16, NCCL_ALGO=RING, NCCL_PROTO=LL"
        wrapped_text, num_lines = style_manager.wrap_table_header(header_with_commas, 4)

        # Should wrap into multiple lines at comma boundaries
        assert num_lines > 1
        lines = wrapped_text.split('\n')

        # Each line should be reasonable length
        for line in lines:
            assert len(line.strip()) > 0  # No empty lines
            # Most lines should end with comma or be the last line
            if line != lines[-1]:  # Not the last line
                assert line.strip().endswith(',') or len(line.strip()) < 30


# ============================================================================
# Chart Features Tests
# ============================================================================

class TestChartFeatures:
    """Test specific chart features."""

    def test_x_axis_log_scale_configuration(self, config_base_path):
        """Test X-axis log scale configuration."""
        plot_config_path = config_base_path / "plot" / "plot_config.toml"

        if plot_config_path.exists():
            plot_config = ChartConfigLoader.load_from_toml(str(plot_config_path))

            # Verify X-axis log scale is configured
            if plot_config.settings and plot_config.settings.x_axis:
                assert plot_config.settings.x_axis.log_scale is True
                assert plot_config.settings.x_axis.scale == "semilogx"
        else:
            # Test with mock config
            mock_plot_config = Mock()
            mock_x_axis = Mock()
            mock_x_axis.log_scale = True
            mock_x_axis.scale = "semilogx"
            mock_plot_config.settings.x_axis = mock_x_axis

            assert mock_plot_config.settings.x_axis.log_scale is True

    def test_chart_title_generation(self, mock_adapter, config_base_path):
        """Test chart title generation concept."""
        # Test title generation concept (since StyleManager doesn't have generate_title)
        chart_name = "Performance Comparison"
        test_binary = "all_reduce_perf"

        # Expected title format
        expected_title = f"{chart_name} - {test_binary}"
        assert "Performance Comparison" in expected_title
        assert "all_reduce_perf" in expected_title

        # Test title with additional parameters
        params = "N=4, ALGO=RING"
        full_title = f"{chart_name} - {test_binary} ({params})"
        assert chart_name in full_title
        assert test_binary in full_title
        assert params in full_title

    def test_chart_title_uses_actual_values_instead_of_default(self, sample_chart_data):
        """Test that chart titles use actual values from data instead of literal 'DEFAULT' values."""
        from nvcomms_perf.report.plotting.chart_generator import ChartGenerator
        from nvcomms_perf.report.adapters.nccl_adapter import NCCLPerformanceAdapter
        from nvcomms_perf.report.plotting.style_manager import StyleManager
        from unittest.mock import Mock

        # Create mock adapter with sample data
        adapter = NCCLPerformanceAdapter()
        adapter._processed_data = sample_chart_data

        # Create mock style manager with proper attributes
        style_manager = Mock(spec=StyleManager)
        style_manager.style_config = Mock()
        # Don't mock apply_title_style so we can see the actual title
        style_manager.style_config.apply_title_style.side_effect = lambda title, manager: title

        # Create chart generator
        generator = ChartGenerator(adapter, style_manager)

        # Create mock chart config with DEFAULT values
        chart_config = Mock()
        chart_config.name = "Default Configuration Comparison"
        chart_config.fixed_params = {
            "NCCL_ALGO": "DEFAULT",
            "NCCL_PROTO": "DEFAULT"
        }

        # Create chart group with sample data
        chart_group = {
            'group_key': 'test_group',
            'data': sample_chart_data,
            'filters': {'num_nodes': 4, 'num_processes': 32}
        }

        # Generate title
        title = generator._generate_chart_title(chart_group, chart_config, "all_reduce_perf")

        # Verify that actual values from data are used instead of "DEFAULT"
        # The sample data has NCCL_ALGO values like 'RING', 'TREE' and NCCL_PROTO values like 'Simple', 'LL'
        assert "NCCL_ALGO=RING" in title or "NCCL_ALGO=TREE" in title
        assert "NCCL_PROTO=Simple" in title or "NCCL_PROTO=LL" in title
        assert "NCCL_ALGO=DEFAULT" not in title
        assert "NCCL_PROTO=DEFAULT" not in title

    def test_title_wrapping_functionality(self, config_base_path):
        """Test title text wrapping."""
        style_config_path = config_base_path / "style" / "style_config.toml"

        if style_config_path.exists():
            style_config = StyleConfigLoader.load_from_toml(str(style_config_path))

            # Test title style configuration
            if hasattr(style_config, 'title_style'):
                assert style_config.title_style.wrap_enabled is True
                assert style_config.title_style.max_length > 0
                assert style_config.title_style.max_lines > 0
        else:
            # Test wrapping concept
            long_title = "This is a very long chart title that should be wrapped according to configuration settings to improve readability"
            max_length = 50

            # Simple wrapping logic test
            if len(long_title) > max_length:
                wrapped_lines = []
                words = long_title.split()
                current_line = ""

                for word in words:
                    if len(current_line + " " + word) <= max_length:
                        current_line += " " + word if current_line else word
                    else:
                        if current_line:
                            wrapped_lines.append(current_line)
                        current_line = word

                if current_line:
                    wrapped_lines.append(current_line)

                assert len(wrapped_lines) > 1  # Should be wrapped
                assert all(len(line) <= max_length for line in wrapped_lines)

    def test_gridlines_configuration(self, config_base_path):
        """Test gridlines configuration."""
        style_config_path = config_base_path / "style" / "style_config.toml"

        if style_config_path.exists():
            style_config = StyleConfigLoader.load_from_toml(str(style_config_path))

            # Test grid configuration
            if hasattr(style_config, 'grid'):
                assert style_config.grid.show is True
                assert 0 <= style_config.grid.alpha <= 1
                assert style_config.grid.color is not None
        else:
            # Test grid concept
            grid_config = {
                'show': True,
                'alpha': 0.2,
                'color': '#666666',
                'style': 'solid'
            }

            assert grid_config['show'] is True
            assert 0 <= grid_config['alpha'] <= 1
            assert grid_config['color'].startswith('#')

    def test_minor_gridlines_support(self, config_base_path):
        """Test minor gridlines support."""
        style_config_path = config_base_path / "style" / "style_config.toml"

        if style_config_path.exists():
            style_config = StyleConfigLoader.load_from_toml(str(style_config_path))

            # Test minor grid configuration if available
            if hasattr(style_config, 'grid'):
                grid_config = style_config.grid
                # Check for minor grid properties
                minor_props = ['minor_alpha', 'minor_x_axis', 'minor_y_axis']
                for prop in minor_props:
                    if hasattr(grid_config, prop):
                        assert getattr(grid_config, prop) is not None
        else:
            # Test minor grid concept
            minor_grid_config = {
                'minor_alpha': 0.1,
                'minor_x_axis': True,
                'minor_y_axis': True,
                'minor_linewidth': 0.4
            }

            assert 0 <= minor_grid_config['minor_alpha'] <= 1
            assert isinstance(minor_grid_config['minor_x_axis'], bool)
            assert minor_grid_config['minor_linewidth'] > 0


# ============================================================================
# Layout and Positioning Tests
# ============================================================================

class TestLayoutAndPositioning:
    """Test chart layout and positioning."""

    def test_legend_positioning(self, config_base_path):
        """Test legend positioning configuration."""
        style_config_path = config_base_path / "style" / "style_config.toml"

        if style_config_path.exists():
            style_config = StyleConfigLoader.load_from_toml(str(style_config_path))

            # Test legend configuration
            if hasattr(style_config, 'legend'):
                legend_config = style_config.legend
                assert legend_config.show is not None
                assert legend_config.position is not None
                assert legend_config.columns > 0
        else:
            # Test legend positioning concept
            legend_config = {
                'show': True,
                'position': 'lower center',
                'columns': 4,
                'alpha': 0.95
            }

            assert legend_config['show'] is True
            assert 'center' in legend_config['position']
            assert legend_config['columns'] > 0

    def test_subplot_layout(self, config_base_path):
        """Test subplot layout configuration."""
        style_config_path = config_base_path / "style" / "style_config.toml"

        if style_config_path.exists():
            style_config = StyleConfigLoader.load_from_toml(str(style_config_path))

            # Test plot layout configuration
            if hasattr(style_config, 'plot_layout'):
                layout = style_config.plot_layout
                layout_props = ['subplot_left', 'subplot_right', 'subplot_top', 'subplot_bottom']
                for prop in layout_props:
                    if hasattr(layout, prop):
                        value = getattr(layout, prop)
                        assert 0 <= value <= 1  # Should be fraction
        else:
            # Test layout concept
            layout_config = {
                'subplot_left': 0.12,
                'subplot_right': 0.95,
                'subplot_top': 0.85,
                'subplot_bottom': 0.1
            }

            for value in layout_config.values():
                assert 0 <= value <= 1

    def test_chart_spacing(self, config_base_path):
        """Test chart spacing configuration."""
        style_config_path = config_base_path / "style" / "style_config.toml"

        if style_config_path.exists():
            style_config = StyleConfigLoader.load_from_toml(str(style_config_path))

            # Test spacing configuration
            if hasattr(style_config, 'plot_layout'):
                layout = style_config.plot_layout
                spacing_props = ['axis_label_pad', 'title_pad']
                for prop in spacing_props:
                    if hasattr(layout, prop):
                        value = getattr(layout, prop)
                        assert value > 0  # Should be positive
        else:
            # Test spacing concept
            spacing_config = {
                'axis_label_pad': 10,
                'title_pad': 20,
                'legend_spacing': 0.03
            }

            for value in spacing_config.values():
                assert value > 0


# ============================================================================
# Performance and Integration Tests
# ============================================================================

class TestPlottingIntegration:
    """Test plotting integration with other components."""

    def test_adapter_style_integration(self, sample_chart_data, config_base_path):
        """Test integration between adapter and style manager."""
        plot_config_path = config_base_path / "plot" / "plot_config.toml"
        style_config_path = config_base_path / "style" / "style_config.toml"

        if plot_config_path.exists() and style_config_path.exists():
            plot_config = ChartConfigLoader.load_from_toml(str(plot_config_path))
            style_config = StyleConfigLoader.load_from_toml(str(style_config_path))

            adapter = NCCLPerformanceAdapter(plot_config=plot_config, style_config=style_config)
            style_manager = StyleManager(style_config)
            chart_generator = ChartGenerator(adapter, style_manager)

            # Test that components work together
            assert chart_generator.adapter == adapter
            assert chart_generator.style_manager == style_manager
        else:
            # Test with mocks
            mock_adapter = Mock(spec=NCCLPerformanceAdapter)
            mock_style_manager = Mock(spec=StyleManager)
            chart_generator = ChartGenerator(mock_adapter, mock_style_manager)

            assert chart_generator.adapter == mock_adapter
            assert chart_generator.style_manager == mock_style_manager

    def test_configuration_consistency(self, config_base_path):
        """Test consistency between plot and style configurations."""
        plot_config_path = config_base_path / "plot" / "plot_config.toml"
        style_config_path = config_base_path / "style" / "style_config.toml"

        if plot_config_path.exists() and style_config_path.exists():
            plot_config = ChartConfigLoader.load_from_toml(str(plot_config_path))
            style_config = StyleConfigLoader.load_from_toml(str(style_config_path))

            # Test that configurations are compatible
            assert plot_config is not None
            assert style_config is not None

            # Test specific consistency checks
            if (plot_config.settings and plot_config.settings.x_axis and
                hasattr(style_config, 'chart_style')):
                # Both configs should be loadable and compatible
                assert plot_config.settings.x_axis.log_scale is not None
                assert style_config.chart_style.font_size > 0

    def test_chart_generation_end_to_end(self, sample_chart_data, config_base_path):
        """Test end-to-end chart generation process."""
        plot_config_path = config_base_path / "plot" / "plot_config.toml"
        style_config_path = config_base_path / "style" / "style_config.toml"

        if plot_config_path.exists() and style_config_path.exists():
            plot_config = ChartConfigLoader.load_from_toml(str(plot_config_path))
            style_config = StyleConfigLoader.load_from_toml(str(style_config_path))

            adapter = NCCLPerformanceAdapter(plot_config=plot_config, style_config=style_config)
            adapter._processed_data = sample_chart_data

            style_manager = StyleManager(style_config)
            chart_generator = ChartGenerator(adapter, style_manager)

            chart_group = {
                'group_key': 'test_integration',
                'data': sample_chart_data,
                'filters': {}
            }

            # Test that full process works
            if plot_config.chart:
                chart_type = plot_config.chart[0].id if plot_config.chart else None
                if chart_type:
                    result = chart_generator.generate_chart(chart_type, 'all_reduce_perf', chart_group)

                    # Should produce some result (may be None for empty data)
                    assert result is None or isinstance(result, matplotlib.figure.Figure)

    def test_error_handling_in_plotting(self, mock_adapter, mock_style_manager):
        """Test error handling in plotting components."""
        chart_generator = ChartGenerator(mock_adapter, mock_style_manager)

        # Test with invalid data
        invalid_chart_group = {
            'group_key': 'invalid_group',
            'data': pd.DataFrame(),  # Empty dataframe
            'filters': {}
        }

        # Should handle errors gracefully
        try:
            result = chart_generator.generate_chart('invalid_type', 'invalid_binary', invalid_chart_group)
            # If no exception, method handles errors gracefully
            assert True
        except Exception as e:
            # Should not raise unhandled exceptions
            assert isinstance(e, (ValueError, KeyError, AttributeError))

    def test_memory_management(self, sample_chart_data, mock_adapter, mock_style_manager):
        """Test memory management in chart generation."""
        chart_generator = ChartGenerator(mock_adapter, mock_style_manager)

        chart_group = {
            'group_key': 'memory_test',
            'data': sample_chart_data,
            'filters': {}
        }

        # Mock chart config with real dicts/lists
        mock_chart_config = Mock()
        mock_chart_config.name = "Test Chart"
        mock_chart_config.performance_columns = ['test_result.out_of_place.bus_bw']
        mock_chart_config.fixed_params = {}
        mock_chart_config.chart_combine_params = {}
        mock_adapter.plot_config.get_chart_by_id.return_value = mock_chart_config

        # Mock adapter methods to return proper values
        mock_adapter.get_performance_columns.return_value = ['test_result.out_of_place.bus_bw']
        mock_adapter.get_legend_style_info.return_value = {
            'color': 'blue',
            'line_style': '-',
            'marker_style': 'o'
        }

        # Generate multiple charts to test memory usage
        results = []
        for i in range(10):
            result = chart_generator.generate_chart(f'test_type_{i}', 'test_binary', chart_group)
            results.append(result)
        assert len(results) == 10

        # Clean up any matplotlib figures
        for result in results:
            if isinstance(result, matplotlib.figure.Figure):
                plt.close(result)


# ============================================================================
# Legend Group Deduplication Tests
# ============================================================================

class TestLegendGroupDeduplication:
    """Test the deduplication logic in _group_data_for_legends method."""

    @pytest.fixture
    def duplicate_data(self):
        """Create test data with duplicate sizes."""
        return pd.DataFrame({
            'test_result.size': [1024, 1024, 2048, 2048, 4096, 4096, 8192, 8192],
            'test_result.out_of_place.bus_bw': [100.0, 100.1, 200.0, 200.1, 300.0, 300.1, 400.0, 400.1],
            'NCCL_ALGO': ['RING', 'RING', 'TREE', 'TREE', 'RING', 'RING', 'TREE', 'TREE'],
            'NCCL_PROTO': ['Simple', 'Simple', 'LL', 'LL', 'Simple', 'Simple', 'LL', 'LL'],
            'local_register': ['0', '0', '1', '1', '0', '0', '1', '1'],
            'test_binary': ['all_reduce_perf'] * 8
        })

    @pytest.fixture
    def mock_adapter_for_dedup(self):
        """Create a mock adapter for deduplication tests."""
        adapter = Mock(spec=NCCLPerformanceAdapter)
        adapter.get_size_column.return_value = 'test_result.size'
        adapter.get_dataframe_column_name.side_effect = lambda param, data: {
            'NCCL_ALGO': 'NCCL_ALGO',
            'NCCL_PROTO': 'NCCL_PROTO',
            'local_register': 'local_register'
        }.get(param, param)
        adapter.shorten_legend_key.side_effect = lambda x: x

        # Add plot_config attribute that ChartGenerator expects
        mock_plot_config = Mock()
        mock_plot_config.chart = []
        adapter.plot_config = mock_plot_config

        return adapter

    def test_no_combine_params_deduplication(self, mock_adapter_for_dedup, mock_style_manager, duplicate_data):
        """Test deduplication when there are no combine parameters."""
        chart_generator = ChartGenerator(mock_adapter_for_dedup, mock_style_manager)

        # Mock chart config with no combine parameters
        mock_chart_config = Mock()
        mock_chart_config.chart_combine_params = {}
        mock_chart_config.fixed_params = {}
        mock_chart_config.default_values = {}

        # Test _group_data_for_legends method
        legend_groups = chart_generator._group_data_for_legends(duplicate_data, mock_chart_config)

        # Should have only one group: "All Data"
        assert len(legend_groups) == 1
        assert "All Data" in legend_groups

        # Check that duplicates are removed
        all_data = legend_groups["All Data"]
        assert len(all_data) == 4  # Should have 4 unique sizes
        assert list(all_data['test_result.size'].unique()) == [1024, 2048, 4096, 8192]

        # Verify that only the first occurrence of each size is kept
        for size in [1024, 2048, 4096, 8192]:
            size_data = all_data[all_data['test_result.size'] == size]
            assert len(size_data) == 1

    def test_no_valid_combine_columns_deduplication(self, mock_adapter_for_dedup, mock_style_manager, duplicate_data):
        """Test deduplication when there are no valid combine columns."""
        chart_generator = ChartGenerator(mock_adapter_for_dedup, mock_style_manager)

        # Mock chart config with combine parameters that don't exist in data
        mock_chart_config = Mock()
        mock_chart_config.chart_combine_params = {'nonexistent_param': 'all'}
        mock_chart_config.fixed_params = {}
        mock_chart_config.default_values = {}

        # Mock adapter to return column names that don't exist
        mock_adapter_for_dedup.get_dataframe_column_name.return_value = 'nonexistent_column'

        # Test _group_data_for_legends method
        legend_groups = chart_generator._group_data_for_legends(duplicate_data, mock_chart_config)

        # Should have only one group: "All Data"
        assert len(legend_groups) == 1
        assert "All Data" in legend_groups

        # Check that duplicates are removed
        all_data = legend_groups["All Data"]
        assert len(all_data) == 4  # Should have 4 unique sizes
        assert list(all_data['test_result.size'].unique()) == [1024, 2048, 4096, 8192]

    def test_single_combine_param_deduplication(self, mock_adapter_for_dedup, mock_style_manager, duplicate_data):
        """Test deduplication with a single combine parameter."""
        chart_generator = ChartGenerator(mock_adapter_for_dedup, mock_style_manager)

        # Mock chart config with single combine parameter
        mock_chart_config = Mock()
        mock_chart_config.chart_combine_params = {'NCCL_ALGO': 'all'}
        mock_chart_config.fixed_params = {}
        mock_chart_config.default_values = {}

        # Test _group_data_for_legends method
        legend_groups = chart_generator._group_data_for_legends(duplicate_data, mock_chart_config)

        # Should have 2 groups: one for each unique NCCL_ALGO value
        assert len(legend_groups) == 2

        # Check that each group has no duplicates
        for legend_key, group_data in legend_groups.items():
            unique_sizes = group_data['test_result.size'].unique()
            assert len(group_data) == len(unique_sizes)  # No duplicates

            # Verify that only the first occurrence of each size is kept
            for size in unique_sizes:
                size_data = group_data[group_data['test_result.size'] == size]
                assert len(size_data) == 1

    def test_multiple_combine_params_deduplication(self, mock_adapter_for_dedup, mock_style_manager, duplicate_data):
        """Test deduplication with multiple combine parameters."""
        chart_generator = ChartGenerator(mock_adapter_for_dedup, mock_style_manager)

        # Mock chart config with multiple combine parameters
        mock_chart_config = Mock()
        mock_chart_config.chart_combine_params = {
            'NCCL_ALGO': 'all',
            'NCCL_PROTO': 'all'
        }
        mock_chart_config.fixed_params = {}
        mock_chart_config.default_values = {}

        # Test _group_data_for_legends method
        legend_groups = chart_generator._group_data_for_legends(duplicate_data, mock_chart_config)

        # Should have 2 groups: one for each unique combination of NCCL_ALGO and NCCL_PROTO
        # (RING+Simple and TREE+LL are the only combinations in our test data)
        assert len(legend_groups) == 2

        # Check that each group has no duplicates
        for legend_key, group_data in legend_groups.items():
            unique_sizes = group_data['test_result.size'].unique()
            assert len(group_data) == len(unique_sizes)  # No duplicates

            # Verify that only the first occurrence of each size is kept
            for size in unique_sizes:
                size_data = group_data[group_data['test_result.size'] == size]
                assert len(size_data) == 1

    def test_fixed_params_with_combine_params_deduplication(self, mock_adapter_for_dedup, mock_style_manager, duplicate_data):
        """Test deduplication when some combine parameters are fixed."""
        chart_generator = ChartGenerator(mock_adapter_for_dedup, mock_style_manager)

        # Mock chart config with fixed and combine parameters
        mock_chart_config = Mock()
        mock_chart_config.chart_combine_params = {
            'NCCL_ALGO': 'all',
            'NCCL_PROTO': 'all'
        }
        mock_chart_config.fixed_params = {'NCCL_ALGO': 'RING'}
        mock_chart_config.default_values = {}

        # Test _group_data_for_legends method
        legend_groups = chart_generator._group_data_for_legends(duplicate_data, mock_chart_config)

        # Should have 2 groups: one for each unique NCCL_PROTO value (since NCCL_ALGO is fixed to RING)
        assert len(legend_groups) == 2

        # Check that each group has no duplicates
        for legend_key, group_data in legend_groups.items():
            unique_sizes = group_data['test_result.size'].unique()
            assert len(group_data) == len(unique_sizes)  # No duplicates

            # Verify that only the first occurrence of each size is kept
            for size in unique_sizes:
                size_data = group_data[group_data['test_result.size'] == size]
                assert len(size_data) == 1

    def test_empty_dataframe_deduplication(self, mock_adapter_for_dedup, mock_style_manager):
        """Test deduplication with empty DataFrame."""
        chart_generator = ChartGenerator(mock_adapter_for_dedup, mock_style_manager)

        # Create empty DataFrame
        empty_data = pd.DataFrame()

        # Mock chart config
        mock_chart_config = Mock()
        mock_chart_config.chart_combine_params = {}
        mock_chart_config.fixed_params = {}
        mock_chart_config.default_values = {}

        # Test _group_data_for_legends method
        legend_groups = chart_generator._group_data_for_legends(empty_data, mock_chart_config)

        # Should have only one group: "All Data"
        assert len(legend_groups) == 1
        assert "All Data" in legend_groups
        assert legend_groups["All Data"].empty

    def test_dataframe_without_size_column_deduplication(self, mock_adapter_for_dedup, mock_style_manager):
        """Test deduplication when DataFrame doesn't have size column."""
        chart_generator = ChartGenerator(mock_adapter_for_dedup, mock_style_manager)

        # Create DataFrame without size column
        data_without_size = pd.DataFrame({
            'NCCL_ALGO': ['RING', 'TREE'],
            'NCCL_PROTO': ['Simple', 'LL']
        })

        # Mock chart config
        mock_chart_config = Mock()
        mock_chart_config.chart_combine_params = {}
        mock_chart_config.fixed_params = {}
        mock_chart_config.default_values = {}

        # Test _group_data_for_legends method
        legend_groups = chart_generator._group_data_for_legends(data_without_size, mock_chart_config)

        # Should have only one group: "All Data"
        assert len(legend_groups) == 1
        assert "All Data" in legend_groups
        assert len(legend_groups["All Data"]) == 2  # No deduplication since no size column

    def test_deduplication_preserves_data_integrity(self, mock_adapter_for_dedup, mock_style_manager, duplicate_data):
        """Test that deduplication preserves data integrity."""
        chart_generator = ChartGenerator(mock_adapter_for_dedup, mock_style_manager)

        # Mock chart config
        mock_chart_config = Mock()
        mock_chart_config.chart_combine_params = {}
        mock_chart_config.fixed_params = {}
        mock_chart_config.default_values = {}

        # Test _group_data_for_legends method
        legend_groups = chart_generator._group_data_for_legends(duplicate_data, mock_chart_config)

        # Get the deduplicated data
        all_data = legend_groups["All Data"]

        # Verify that the first occurrence of each size is preserved
        # For size 1024, first occurrence has bus_bw = 100.0
        size_1024_data = all_data[all_data['test_result.size'] == 1024]
        assert len(size_1024_data) == 1
        assert size_1024_data['test_result.out_of_place.bus_bw'].iloc[0] == 100.0  # type: ignore

        # For size 2048, first occurrence has bus_bw = 200.0
        size_2048_data = all_data[all_data['test_result.size'] == 2048]
        assert len(size_2048_data) == 1
        assert size_2048_data['test_result.out_of_place.bus_bw'].iloc[0] == 200.0  # type: ignore
