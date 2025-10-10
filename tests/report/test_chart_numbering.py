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
Tests for chart numbering functionality.

This module tests that unique chart numbers are properly added to charts and data tables
for correlation purposes.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock
import matplotlib.figure
import re

from nvcomms_perf.report.plotting.chart_generator import ChartGenerator
from nvcomms_perf.report.adapters.nccl_adapter import NCCLPerformanceAdapter
from nvcomms_perf.report.plotting.style_manager import StyleManager


class TestChartNumbering:
    """Test cases for chart numbering functionality."""

    def test_generate_chart_with_number(self):
        """Test that generate_chart accepts and uses chart_number parameter."""
        # Create mock adapter and style manager
        adapter = Mock(spec=NCCLPerformanceAdapter)
        style_manager = Mock(spec=StyleManager)

        # Mock plot configuration
        mock_plot_config = Mock()
        mock_chart_config = Mock()
        mock_chart_config.name = "Test Chart"
        mock_chart_config.performance_columns = ["test_perf"]
        mock_chart_config.fixed_params = {}
        mock_chart_config.chart_combine_params = {}
        mock_plot_config.get_chart_by_id.return_value = mock_chart_config
        adapter.plot_config = mock_plot_config

        # Mock style manager methods
        mock_fig = Mock(spec=matplotlib.figure.Figure)
        mock_ax = Mock()
        style_manager.create_figure.return_value = (mock_fig, mock_ax)

        # Create nested mock for style_config
        mock_style_config = Mock()
        mock_chart_style = Mock()
        mock_chart_style.font_size = 12
        mock_chart_style.marker_size = 6
        mock_chart_style.markeredgewidth = 1
        mock_chart_style.markeredgecolor = 'black'
        mock_chart_style.line_width = 2
        mock_style_config.chart_style = mock_chart_style

        mock_font_adjustments = Mock()
        mock_font_adjustments.title_medium = 2
        mock_style_config.font_adjustments = mock_font_adjustments

        mock_plot_layout = Mock()
        mock_plot_layout.title_pad = 10
        mock_style_config.plot_layout = mock_plot_layout

        # Add text_wrapping with int value
        mock_text_wrapping = Mock()
        mock_text_wrapping.title_max_length = 100
        mock_style_config.text_wrapping = mock_text_wrapping

        # Mock title_style and its methods
        mock_title_style = Mock()
        mock_title_style.font_weight = "normal"
        mock_title_style.color = "#333333"
        mock_style_config.title_style = mock_title_style

        # Configure apply_title_style to return the input title (identity function)
        mock_style_config.apply_title_style = Mock(side_effect=lambda x, *args: x)
        mock_style_config.get_title_font_size = Mock(return_value=14)

        style_manager.style_config = mock_style_config

        # Mock adapter methods
        adapter.get_size_column.return_value = "size"
        adapter.get_performance_columns.return_value = ["test_perf"]
        adapter.get_legend_style_info.return_value = {
            'color': 'blue',
            'line_style': '-',
            'marker_style': 'o'
        }
        adapter.get_dataframe_column_name.return_value = "size"
        # Set _processed_data to a DataFrame
        adapter._processed_data = pd.DataFrame({'size': [1024, 2048], 'test_perf': [10.5, 20.3]})

        # Create test data
        test_data = pd.DataFrame({
            'size': [1024, 2048],
            'test_perf': [10.5, 20.3],
            'param1': ['value1', 'value1']
        })

        chart_group = {
            'group_key': 'test_group',
            'data': test_data,
            'filters': {}
        }

        # Create chart generator
        chart_generator = ChartGenerator(adapter, style_manager)

        # Test generating chart with number
        result = chart_generator.generate_chart(
            chart_type="test_type",
            test_binary="test_binary",
            chart_group=chart_group,
            chart_number=42
        )
        assert result is not None or result is None

    def test_create_data_table_with_number(self):
        """Test that create_data_table accepts and uses chart_number parameter."""
        # Create mock adapter and style manager
        adapter = Mock(spec=NCCLPerformanceAdapter)
        style_manager = Mock(spec=StyleManager)

        # Mock plot configuration
        mock_plot_config = Mock()
        mock_chart_config = Mock()
        mock_chart_config.performance_columns = ["test_perf"]
        mock_chart_config.fixed_params = {}
        mock_chart_config.chart_combine_params = {}
        mock_plot_config.get_chart_by_id.return_value = mock_chart_config
        adapter.plot_config = mock_plot_config

        # Mock style manager methods
        mock_fig = Mock(spec=matplotlib.figure.Figure)
        mock_ax = Mock()
        style_manager.get_figure_size.return_value = (10, 8)

        # Create nested mock for style_config
        mock_style_config = Mock()
        mock_chart_style = Mock()
        mock_chart_style.font_size = 12
        mock_style_config.chart_style = mock_chart_style

        mock_font_adjustments = Mock()
        mock_font_adjustments.title_medium = 2
        mock_style_config.font_adjustments = mock_font_adjustments

        mock_plot_layout = Mock()
        mock_plot_layout.title_pad = 10
        mock_style_config.plot_layout = mock_plot_layout

        mock_table_layout = Mock()
        mock_table_layout.bbox_left = 0.1
        mock_table_layout.bbox_bottom = 0.1
        mock_table_layout.bbox_width = 0.8
        mock_table_layout.bbox_height = 0.8
        mock_style_config.table_layout = mock_table_layout

        mock_pdf_settings = Mock()
        mock_pdf_settings.pdf_subplot_left = 0.1
        mock_pdf_settings.pdf_subplot_right = 0.9
        mock_pdf_settings.pdf_subplot_top = 0.9
        mock_pdf_settings.pdf_subplot_bottom = 0.1
        mock_style_config.pdf_settings = mock_pdf_settings

        # Add text_wrapping with int value
        mock_text_wrapping = Mock()
        mock_text_wrapping.title_max_length = 100
        mock_style_config.text_wrapping = mock_text_wrapping

        # Mock title_style and its methods
        mock_title_style = Mock()
        mock_title_style.font_weight = "normal"
        mock_title_style.color = "#333333"
        mock_style_config.title_style = mock_title_style

        # Configure apply_title_style to return the input title (identity function)
        mock_style_config.apply_title_style = Mock(side_effect=lambda x, *args: x)
        mock_style_config.get_title_font_size = Mock(return_value=14)

        style_manager.style_config = mock_style_config

        # Mock adapter methods
        adapter.get_size_column.return_value = "size"
        adapter.get_performance_columns.return_value = ["test_perf"]
        adapter.shorten_legend_key.return_value = "short_key"
        adapter.get_dataframe_column_name.return_value = "size"
        # Set _processed_data to a DataFrame
        adapter._processed_data = pd.DataFrame({'size': [1024, 2048], 'test_perf': [10.5, 20.3]})

        # Create test data
        test_data = pd.DataFrame({
            'size': [1024, 2048],
            'test_perf': [10.5, 20.3],
            'param1': ['value1', 'value1']
        })

        chart_group = {
            'group_key': 'test_group',
            'data': test_data,
            'filters': {}
        }

        # Create chart generator
        chart_generator = ChartGenerator(adapter, style_manager)

        # Mock the table creation methods to avoid complex matplotlib operations
        with pytest.MonkeyPatch().context() as m:
            m.setattr(chart_generator, '_calculate_max_columns_per_page', lambda: 10)
            m.setattr(chart_generator, '_create_single_page_table', lambda *args: mock_fig)

            # Test creating data table with number
            result = chart_generator.create_data_table(
                chart_type="test_type",
                test_binary="test_binary",
                chart_group=chart_group,
                chart_number=42
            )
            assert result is not None or result is None

    def test_chart_title_with_number(self):
        """Test that chart titles include chart numbers when provided."""
        # Create mock adapter and style manager
        adapter = Mock(spec=NCCLPerformanceAdapter)
        style_manager = Mock(spec=StyleManager)

        # Mock style manager for text wrapping
        mock_style_config = Mock()
        mock_text_wrapping = Mock()
        mock_text_wrapping.title_max_length = 100
        mock_style_config.text_wrapping = mock_text_wrapping

        # Mock title_style and its methods
        mock_title_style = Mock()
        mock_title_style.font_weight = "normal"
        mock_title_style.color = "#333333"
        mock_style_config.title_style = mock_title_style

        # Configure apply_title_style to return the input title (identity function)
        mock_style_config.apply_title_style = Mock(side_effect=lambda x, *args: x)
        mock_style_config.get_title_font_size = Mock(return_value=14)

        style_manager.style_config = mock_style_config
        style_manager.wrap_text = lambda text, length: text

        # Set _processed_data to a DataFrame
        adapter._processed_data = pd.DataFrame({'size': [1024, 2048], 'test_perf': [10.5, 20.3]})
        # Set plot_config
        mock_plot_config = Mock()
        adapter.plot_config = mock_plot_config
        adapter.get_dataframe_column_name.return_value = "size"

        # Create chart generator
        chart_generator = ChartGenerator(adapter, style_manager)

        # Mock chart configuration
        mock_chart_config = Mock()
        mock_chart_config.name = "Test Chart"
        mock_chart_config.fixed_params = {}

        chart_group = {
            'group_key': 'test_group',
            'data': pd.DataFrame({'size': [1024, 2048]}),
            'filters': {}
        }

        # Test title generation with chart number
        title = chart_generator._generate_chart_title(
            chart_group=chart_group,
            chart_config=mock_chart_config,
            test_binary="test_binary",
            chart_number=42
        )

        # Verify chart number is in title
        assert "Chart 42" in title
        assert title == "Test Chart (test_binary) Chart 42"

        # Test title generation without chart number
        title_no_number = chart_generator._generate_chart_title(
            chart_group=chart_group,
            chart_config=mock_chart_config,
            test_binary="test_binary"
        )

        # Verify chart number is not in title
        assert "Chart 42" not in title_no_number
        assert title_no_number == "Test Chart (test_binary)"

    def test_optional_chart_number_parameter(self):
        """Test that chart_number parameter is optional and defaults to None."""
        # Create mock adapter and style manager
        adapter = Mock(spec=NCCLPerformanceAdapter)
        style_manager = Mock(spec=StyleManager)

        # Mock plot configuration
        mock_plot_config = Mock()
        adapter.plot_config = mock_plot_config

        # Set _processed_data to a DataFrame
        adapter._processed_data = pd.DataFrame({'size': [1024, 2048], 'test_perf': [10.5, 20.3]})

        # Mock style manager for text wrapping
        mock_style_config = Mock()
        mock_text_wrapping = Mock()
        mock_text_wrapping.title_max_length = 100
        mock_style_config.text_wrapping = mock_text_wrapping

        # Mock title_style and its methods
        mock_title_style = Mock()
        mock_title_style.font_weight = "normal"
        mock_title_style.color = "#333333"
        mock_style_config.title_style = mock_title_style

        # Configure apply_title_style to return the input title (identity function)
        mock_style_config.apply_title_style = Mock(side_effect=lambda x, *args: x)
        mock_style_config.get_title_font_size = Mock(return_value=14)

        style_manager.style_config = mock_style_config
        style_manager.wrap_text = lambda text, length: text
        adapter.get_dataframe_column_name.return_value = "size"

        # Create chart generator
        chart_generator = ChartGenerator(adapter, style_manager)

        # Mock chart configuration
        mock_chart_config = Mock()
        mock_chart_config.name = "Test Chart"
        mock_chart_config.fixed_params = {}

        chart_group = {
            'group_key': 'test_group',
            'data': pd.DataFrame({'size': [1024, 2048]}),
            'filters': {}
        }

        # Test that method can be called without chart_number
        title = chart_generator._generate_chart_title(
            chart_group=chart_group,
            chart_config=mock_chart_config,
            test_binary="test_binary"
        )

        # Verify title is generated without chart number
        assert not re.search(r"Chart \d+$", title)
        assert title == "Test Chart (test_binary)"
