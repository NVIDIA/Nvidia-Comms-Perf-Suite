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
Tests for the report API module.

This module tests the direct API functionality for generating reports
from dataframe strings, plot config strings, and style config strings.
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from nvcomms_perf.report.api import (
    ReportAPI,
    generate_pdf_report_from_dataframe_string,
    generate_pdf_report_from_dataframe_object,
    validate_dataframe_string,
    validate_config_string
)


def create_mock_style_config():
    """Create a properly configured mock style config with all required attributes."""
    mock_style_config = MagicMock()

    # PDF settings
    mock_style_config.pdf_settings = MagicMock()
    mock_style_config.pdf_settings.page_width = 8.5
    mock_style_config.pdf_settings.page_height = 11.0
    mock_style_config.pdf_settings.use_landscape = False
    mock_style_config.pdf_settings.pdf_dpi = 300

    # Margin attributes
    mock_style_config.pdf_settings.margin_left = 0.5
    mock_style_config.pdf_settings.margin_right = 0.5
    mock_style_config.pdf_settings.margin_top = 1.0
    mock_style_config.pdf_settings.margin_bottom = 1.0
    # Subplot parameters (set after MagicMock assignment)
    mock_style_config.pdf_settings.pdf_subplot_left = 0.1
    mock_style_config.pdf_settings.pdf_subplot_right = 0.9
    mock_style_config.pdf_settings.pdf_subplot_top = 0.9
    mock_style_config.pdf_settings.pdf_subplot_bottom = 0.1
    # Color attributes
    mock_style_config.pdf_settings.pdf_facecolor = 'white'
    mock_style_config.pdf_settings.pdf_edgecolor = 'black'

    # Grid attributes
    mock_style_config.grid.color = '#cccccc'
    mock_style_config.grid.linestyle = '-'
    mock_style_config.grid.linewidth = 0.5
    mock_style_config.grid.alpha = 0.8
    mock_style_config.grid.style = '-'
    mock_style_config.grid.show = True
    mock_style_config.grid.major_color = '#cccccc'
    mock_style_config.grid.major_style = '-'
    mock_style_config.grid.major_linewidth = 0.5
    mock_style_config.grid.minor_x_axis = False
    mock_style_config.grid.minor_y_axis = False
    mock_style_config.grid.minor_alpha = 0.3
    mock_style_config.grid.minor_color = '#dddddd'
    mock_style_config.grid.minor_style = ':'
    mock_style_config.grid.minor_linewidth = 0.3

    # Chart style attributes
    mock_style_config.chart_style.axes_facecolor = 'white'
    mock_style_config.chart_style.figure_facecolor = 'white'
    mock_style_config.chart_style.text_color = 'black'
    mock_style_config.chart_style.axes_edgecolor = 'black'
    mock_style_config.chart_style.axes_linewidth = 1.0
    mock_style_config.chart_style.xtick_color = 'black'
    mock_style_config.chart_style.ytick_color = 'black'
    mock_style_config.chart_style.spine_color = 'black'
    mock_style_config.chart_style.axes_labelcolor = 'black'
    mock_style_config.chart_style.axes_titlecolor = 'black'
    mock_style_config.chart_style.background_color = 'white'
    # Spine settings
    mock_style_config.chart_style.axes_spines_top = True
    mock_style_config.chart_style.axes_spines_bottom = True
    mock_style_config.chart_style.axes_spines_left = True
    mock_style_config.chart_style.axes_spines_right = True
    # Additional axes properties
    mock_style_config.chart_style.axes_grid = True
    mock_style_config.chart_style.axes_axisbelow = True
    mock_style_config.chart_style.axes_unicode_minus = False
    mock_style_config.chart_style.axes_prop_cycle = None
    mock_style_config.chart_style.axes_titlesize = 12
    mock_style_config.chart_style.axes_labelsize = 10
    mock_style_config.chart_style.axes_linewidth = 0.8
    mock_style_config.chart_style.axes_labelpad = 4.0
    mock_style_config.chart_style.axes_formatter_useoffset = True
    mock_style_config.chart_style.axes_formatter_use_mathtext = False
    mock_style_config.chart_style.axes_formatter_min_exponent = 0
    mock_style_config.chart_style.axes_formatter_limit = -7
    mock_style_config.chart_style.axes_formatter_use_locale = False
    mock_style_config.chart_style.axes_autolayout = False
    mock_style_config.chart_style.axes_xmargin = 0.05
    mock_style_config.chart_style.axes_ymargin = 0.05
    mock_style_config.chart_style.axes_zmargin = 0.05
    # Font and line properties
    mock_style_config.chart_style.font_size = 12
    mock_style_config.chart_style.line_width = 2.0
    mock_style_config.chart_style.marker_size = 6
    mock_style_config.chart_style.markeredgewidth = 1.0
    mock_style_config.chart_style.markeredgecolor = 'black'
    # Background effects
    mock_style_config.chart_style.background_gradient = False
    mock_style_config.chart_style.background_gradient_alpha = 0.1
    mock_style_config.chart_style.background_pattern = "none"

    # Font adjustments
    mock_style_config.font_adjustments.label_large = 2
    mock_style_config.font_adjustments.label_small = -1
    mock_style_config.font_adjustments.title_large = 4
    mock_style_config.font_adjustments.title_medium = 2
    mock_style_config.font_adjustments.table_header = 1

    # Legend settings
    mock_style_config.legend.fontsize = 10

    # Ticks settings
    mock_style_config.ticks.minor_length = 3
    mock_style_config.ticks.major_length = 6
    mock_style_config.ticks.minor_width = 0.5
    mock_style_config.ticks.major_width = 1.0
    mock_style_config.ticks.tick_direction = 'out'
    mock_style_config.ticks.major_color = 'black'
    mock_style_config.ticks.minor_color = 'black'

    # Page layout
    mock_style_config.page_layout.note_grid_start = 10
    mock_style_config.page_layout.note_grid_end = 90
    mock_style_config.page_layout.note_grid_step = 5
    mock_style_config.page_layout.command_start_y = 0.9

    # Grid lines
    mock_style_config.grid_lines.note_page_linewidth = 0.5

    # Text positioning
    mock_style_config.text_positioning.title_x = 0.5
    mock_style_config.text_positioning.title_y = 0.95

    # Title style
    mock_style_config.title_style.font_weight = 'bold'
    mock_style_config.title_style.color = 'black'
    mock_style_config.title_style.alignment = 'center'
    mock_style_config.title_style.vertical_alignment = 'top'

    # Plot layout
    mock_style_config.plot_layout.title_pad = 20

    # Table layout
    mock_style_config.table_layout.scale_width = 1.0
    mock_style_config.table_layout.scale_height = 1.0

    # Method mocks
    mock_style_config.get_figure_size = MagicMock(return_value=(8.5, 11.0))
    mock_style_config.get_title_font_size = MagicMock(return_value=16)
    mock_style_config.apply_title_style = MagicMock(side_effect=lambda title, style_manager: title)

    return mock_style_config


class TestReportAPI:
    """Test cases for the ReportAPI class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.api = ReportAPI()

        # Sample CSV data
        self.sample_csv_data = """tool_binary,test_name,bandwidth_gbps,latency_us
nccl-tests,all_reduce,100.5,50.2
nccl-tests,all_gather,95.3,45.1
nccl-tests,broadcast,98.7,48.9"""

        # Sample plot config TOML
        self.sample_plot_config = """[[pdf_configs]]
id = \"basic_report\"
name = \"Basic Report\"
description = \"A basic PDF report\"
analysis_profiles = [\"default_profile\"]

[[analysis_profiles]]
id = \"default_profile\"
name = \"Default Analysis Profile\"
description = \"Default analysis profile for NCCL tests\"
test_binaries = [\"nccl-tests\"]
chart_types = [\"bandwidth\", \"latency\"]
"""

        # Sample style config TOML
        self.sample_style_config = """[pdf_settings]
page_width = 8.5
page_height = 11.0
use_landscape = false
pdf_dpi = 300
pdf_facecolor = "white"
pdf_edgecolor = "black"
"""

    def test_parse_dataframe_from_input_valid(self):
        """Test parsing valid DataFrame from string."""
        df = self.api._parse_dataframe_from_input(self.sample_csv_data)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 4)
        assert list(df.columns) == ['tool_binary', 'test_name', 'bandwidth_gbps', 'latency_us']
        assert df.iloc[0]['tool_binary'] == 'nccl-tests'
        assert df.iloc[0]['bandwidth_gbps'] == 100.5

    def test_parse_dataframe_from_input_invalid(self):
        """Test parsing invalid DataFrame from string."""
        invalid_csv = "This is not CSV data at all"
        df = self.api._parse_dataframe_from_input(invalid_csv)
        assert df.shape == (0, 1)
        assert list(df.columns) == [invalid_csv]

    def test_parse_config_from_input_plot(self):
        """Test parsing plot config from string."""
        config = self.api._parse_config_from_input(self.sample_plot_config, 'plot')
        assert config is not None
        assert hasattr(config, 'analysis_profiles')

    def test_parse_config_from_input_style(self):
        """Test parsing style config from string."""
        config = self.api._parse_config_from_input(self.sample_style_config, 'style')
        assert config is not None
        assert hasattr(config, 'pdf_settings')

    def test_parse_config_from_input_invalid_type(self):
        """Test parsing config with invalid type."""
        with pytest.raises(ValueError, match="Unknown config type: invalid"):
            self.api._parse_config_from_input("some config", 'invalid')

    def test_parse_config_from_input_invalid_format(self):
        """Test parsing invalid config format."""
        invalid_config = "invalid toml format ["
        with pytest.raises(Exception):
            self.api._parse_config_from_input(invalid_config, 'plot')

    def test_load_default_configs(self):
        """Test loading default configurations."""
        with patch('nvcomms_perf.report.cli.report._get_default_config_path') as mock_get_path, \
             patch('nvcomms_perf.report.models.ChartConfigLoader.load_from_toml') as mock_plot_load, \
             patch('nvcomms_perf.report.models.StyleConfigLoader.load_from_toml') as mock_style_load:
            mock_plot_config = MagicMock()
            mock_style_config = MagicMock()
            mock_plot_load.return_value = mock_plot_config
            mock_style_load.return_value = mock_style_config
            mock_get_path.side_effect = ['/path/to/plot.toml', '/path/to/style.toml']
            self.api._load_default_configs()
            assert self.api._default_plot_config == mock_plot_config
            assert self.api._default_style_config == mock_style_config
            mock_plot_load.assert_called_once_with('/path/to/plot.toml')
            mock_style_load.assert_called_once_with('/path/to/style.toml')

    def test_generate_pdf_report_from_dataframe(self):
        """Test generating PDF report from DataFrame string."""
        with patch.object(ReportAPI, '_parse_dataframe_from_input') as mock_parse_df, \
             patch.object(ReportAPI, '_parse_config_from_input') as mock_parse_config, \
             patch('nvcomms_perf.report.models.ChartConfigLoader.load_from_toml') as mock_plot_loader, \
             patch('nvcomms_perf.report.models.StyleConfigLoader.load_from_toml') as mock_style_loader, \
             patch('nvcomms_perf.report.api.NCCLPerformanceAdapter') as mock_adapter_class, \
             patch('nvcomms_perf.report.api.generate_pdf_report') as mock_generate, \
             patch('nvcomms_perf.report.plotting.style_manager.StyleManager.apply_global_style') as mock_apply_style, \
             patch('nvcomms_perf.report.plotting.style_manager.StyleManager.get_figure_size', return_value=(8.5, 11.0)) as mock_get_figure_size:
            mock_df = pd.DataFrame({'test': [1, 2, 3]})
            mock_plot_config = MagicMock()
            mock_style_config = create_mock_style_config()
            mock_adapter = MagicMock()
            # Configure the adapter's style config with real numbers for subplot parameters
            mock_adapter.style_config = mock_style_config
            # Configure adapter methods that might be called
            mock_adapter.get_processed_data = MagicMock(return_value=mock_df)
            mock_adapter.get_chart_data = MagicMock(return_value=[])
            mock_parse_df.return_value = mock_df
            mock_parse_config.side_effect = [mock_plot_config, mock_style_config]
            mock_plot_loader.return_value = mock_plot_config
            mock_style_loader.return_value = mock_style_config
            mock_adapter_class.return_value = mock_adapter
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                output_path = temp_file.name
            try:
                self.api.generate_pdf_report_from_input(
                    dataframe_input=self.sample_csv_data,
                    output_path=output_path,
                    plot_config_input=self.sample_plot_config,
                    style_config_input=self.sample_style_config
                )
                mock_parse_df.assert_called_once_with(self.sample_csv_data)
                assert mock_parse_config.call_count == 2
                mock_adapter_class.assert_called_once_with(
                    plot_config=mock_plot_config,
                    style_config=mock_style_config
                )
                mock_generate.assert_called_once()
            finally:
                Path(output_path).unlink(missing_ok=True)

    def test_generate_pdf_report_from_dataframe_object(self):
        """Test generating PDF report from DataFrame object."""
        with patch.object(ReportAPI, '_parse_config_from_input') as mock_parse_config, \
             patch('nvcomms_perf.report.models.StyleConfigLoader.load_from_toml') as mock_style_loader, \
             patch('nvcomms_perf.report.api.NCCLPerformanceAdapter') as mock_adapter_class, \
             patch('nvcomms_perf.report.api.generate_pdf_report') as mock_generate, \
             patch('nvcomms_perf.report.plotting.style_manager.StyleManager.apply_global_style') as mock_apply_style, \
             patch('nvcomms_perf.report.plotting.style_manager.StyleManager.get_figure_size', return_value=(8.5, 11.0)) as mock_get_figure_size:
            mock_df = pd.DataFrame({'test': [1, 2, 3]})
            mock_plot_config = MagicMock()
            mock_style_config = create_mock_style_config()
            mock_adapter = MagicMock()
            # Configure the adapter's style config with real numbers for subplot parameters
            mock_adapter.style_config = mock_style_config
            # Configure adapter methods that might be called
            mock_adapter.get_processed_data = MagicMock(return_value=mock_df)
            mock_adapter.get_chart_data = MagicMock(return_value=[])
            mock_parse_config.side_effect = [mock_plot_config, mock_style_config]
            mock_style_loader.return_value = mock_style_config
            mock_adapter_class.return_value = mock_adapter
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                output_path = temp_file.name
            try:
                from nvcomms_perf.report.api import generate_pdf_report_from_dataframe_object
                generate_pdf_report_from_dataframe_object(
                    dataframe=mock_df,
                    output_path=output_path,
                    plot_config_str=self.sample_plot_config,
                    style_config_str=self.sample_style_config
                )
                assert mock_parse_config.call_count == 2
                mock_adapter_class.assert_called_once_with(
                    plot_config=mock_plot_config,
                    style_config=mock_style_config
                )
                mock_generate.assert_called_once()
            finally:
                Path(output_path).unlink(missing_ok=True)

    def test_validate_dataframe_string_valid(self):
        """Test validating valid DataFrame string."""
        result = self.api.validate_dataframe_input(self.sample_csv_data)
        assert result['valid'] is True
        assert result['shape'] == (3, 4)
        assert 'columns' in result
        assert 'memory_usage_mb' in result
        assert 'dtypes' in result

    def test_validate_dataframe_string_invalid(self):
        """Test validating invalid DataFrame string."""
        invalid_csv = "This is not CSV data at all"
        result = self.api.validate_dataframe_input(invalid_csv)
        assert 'valid' in result
        assert result['valid'] is True
        assert result['shape'] == (0, 1)
        assert result['columns'] == [invalid_csv]

    def test_validate_config_string_plot(self):
        """Test validating plot config string."""
        result = self.api.validate_config_input(self.sample_plot_config, 'plot')
        assert result['valid'] is True
        assert result['config_type'] == 'plot'
        assert 'analysis_profiles' in result
        assert 'pdf_configs' in result

    def test_validate_config_string_style(self):
        """Test validating style config string."""
        result = self.api.validate_config_input(self.sample_style_config, 'style')
        assert result['valid'] is True
        assert result['config_type'] == 'style'
        assert 'pdf_settings' in result
        assert result['pdf_settings']['page_width'] == 8.5
        assert result['pdf_settings']['page_height'] == 11.0

    def test_validate_config_string_style_with_code_display(self):
        valid_style_config = """[pdf_settings]
page_width = 8.5
page_height = 11.0

[code_display]
show_test_commands = false
command_format = "raw"
command_max_lines = 15
command_truncate_length = 120
show_configurations = true
config_format = "formatted"
config_max_lines = 25
config_truncate_length = 90
syntax_highlighting = false
highlight_theme = "monokai"
code_font_family = "Source Code Pro"
code_font_size = 12
code_block_width = 0.85
code_block_margin = 0.06
code_block_padding = 0.03
include_environment_vars = false
include_arguments = true
group_commands_by_test = false
sort_commands_by_execution_order = false

[code_display.block_style]
background_color = "#1a1a1a"
border_color = "#333333"
border_width = 3
border_style = "double"
corner_radius = 10
shadow_enabled = false
shadow_offset_x = 3
shadow_offset_y = 3
shadow_blur = 6
shadow_color = "#000000"
shadow_alpha = 0.15

[code_display.text_style]
font_family = "Fira Code"
font_size = 11
line_height = 1.5
color = "#ffffff"
selection_color = "#0056b3"
selection_background = "#e6f3ff"

[code_display.command_style]
prompt_color = "#888888"
command_color = "#ffffff"
argument_color = "#cccccc"
option_color = "#00ff00"
variable_color = "#ffff00"
comment_color = "#888888"

[code_display.config_style]
section_header_color = "#ffffff"
section_header_background = "#333333"
key_color = "#00ff00"
value_color = "#ffffff"
comment_color = "#888888"
separator_color = "#666666"
"""
        result = self.api.validate_config_input(valid_style_config, 'style')
        assert result['valid'] is True
        assert result['config_type'] == 'style'
        assert 'pdf_settings' in result

    def test_validate_config_string_style_invalid_code_display(self):
        invalid_style_config = """[pdf_settings]
page_width = 8.5
page_height = 11.0

[code_display]
command_format = "invalid_format"  # Invalid format
command_max_lines = -1  # Invalid value
code_block_width = 2.0  # Invalid value (should be <= 1)
"""
        result = self.api.validate_config_input(invalid_style_config, 'style')
        assert result['valid'] is False
        assert result['config_type'] == 'style'
        assert 'error' in result

    def test_validate_config_string_invalid(self):
        invalid_config = "invalid toml format ["
        result = self.api.validate_config_input(invalid_config, 'plot')
        assert result['valid'] is False
        assert result['config_type'] == 'plot'
        assert 'error' in result


class TestConvenienceFunctions:
    """Test cases for the convenience functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_csv_data = """tool_binary,test_name,bandwidth_gbps
nccl-tests,all_reduce,100.5
nccl-tests,all_gather,95.3"""

        self.sample_plot_config = """[[pdf_configs]]
id = \"basic_report\"
name = \"Basic Report\"
description = \"A basic PDF report\"
analysis_profiles = [\"default_profile\"]

[[analysis_profiles]]
id = \"default_profile\"
name = \"Default Analysis Profile\"
description = \"Default analysis profile for NCCL tests\"
test_binaries = [\"nccl-tests\"]
chart_types = [\"bandwidth\", \"latency\"]
"""

        self.sample_style_config = """[pdf_settings]
page_width = 8.5
page_height = 11.0
use_landscape = false
pdf_dpi = 300
pdf_facecolor = "white"
pdf_edgecolor = "black"
"""

    def test_generate_pdf_report_from_dataframe_string(self):
        """Test convenience function for DataFrame string."""
        with patch('nvcomms_perf.report.api.ReportAPI.generate_pdf_report_from_input') as mock_func:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                output_path = temp_file.name
            try:
                from nvcomms_perf.report.api import generate_pdf_report_from_dataframe_string
                generate_pdf_report_from_dataframe_string(
                    dataframe_str=self.sample_csv_data,
                    output_path=output_path,
                    plot_config_str=self.sample_plot_config,
                    style_config_str=self.sample_style_config
                )
                mock_func.assert_called_once_with(
                    dataframe_input=self.sample_csv_data,
                    output_path=output_path,
                    pdf_type='basic_report',
                    plot_config_input=self.sample_plot_config,
                    style_config_input=self.sample_style_config
                )
            finally:
                Path(output_path).unlink(missing_ok=True)

    def test_generate_pdf_report_from_dataframe_object(self):
        """Test convenience function for DataFrame object."""
        with patch.object(ReportAPI, '_parse_config_from_input') as mock_parse_config, \
             patch('nvcomms_perf.report.models.ChartConfigLoader.load_from_toml') as mock_plot_loader, \
             patch('nvcomms_perf.report.models.StyleConfigLoader.load_from_toml') as mock_style_loader, \
             patch('nvcomms_perf.report.api.NCCLPerformanceAdapter') as mock_adapter_class, \
             patch('nvcomms_perf.report.api.generate_pdf_report') as mock_generate:
            mock_df = pd.DataFrame({'test': [1, 2, 3]})
            mock_plot_config = MagicMock()
            mock_style_config = create_mock_style_config()
            mock_adapter = MagicMock()
            mock_adapter.style_config = mock_style_config
            mock_adapter.get_processed_data = MagicMock(return_value=mock_df)
            mock_adapter.get_chart_data = MagicMock(return_value=[])
            mock_parse_config.side_effect = [mock_plot_config, mock_style_config]
            mock_plot_loader.return_value = mock_plot_config
            mock_style_loader.return_value = mock_style_config
            mock_adapter_class.return_value = mock_adapter
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                output_path = temp_file.name
            try:
                from nvcomms_perf.report.api import generate_pdf_report_from_dataframe_object
                generate_pdf_report_from_dataframe_object(
                    dataframe=mock_df,
                    output_path=output_path,
                    plot_config_str=self.sample_plot_config,
                    style_config_str=self.sample_style_config
                )
                assert mock_parse_config.call_count == 2
                mock_adapter_class.assert_called_once_with(
                    plot_config=mock_plot_config,
                    style_config=mock_style_config
                )
                mock_generate.assert_called_once()
            finally:
                Path(output_path).unlink(missing_ok=True)

    def test_validate_dataframe_string(self):
        """Test convenience function for DataFrame validation."""
        from nvcomms_perf.report.api import validate_dataframe_string
        result = validate_dataframe_string(self.sample_csv_data)
        assert result['valid'] is True
        assert result['shape'] == (2, 3)
        assert 'columns' in result

    def test_validate_config_string(self):
        """Test convenience function for config validation."""
        from nvcomms_perf.report.api import validate_config_string
        result = validate_config_string(self.sample_plot_config, 'plot')
        assert result['valid'] is True
        assert result['config_type'] == 'plot'
        assert 'analysis_profiles' in result


class TestIntegration:
    """Integration tests for the API functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_csv_data = """tool_binary,test_name,bandwidth_gbps,latency_us
nccl-tests,all_reduce,100.5,50.2
nccl-tests,all_gather,95.3,45.1
nccl-tests,broadcast,98.7,48.9"""

        self.sample_plot_config = """[[pdf_configs]]
id = "basic_report"
name = "Basic Report"
description = "A basic PDF report"
analysis_profiles = ["default_profile"]

[[analysis_profiles]]
id = "default_profile"
name = "Default Analysis Profile"
description = "Default analysis profile for NCCL tests"
test_binaries = ["nccl-tests"]
chart_types = ["bandwidth", "latency"]
"""

        self.sample_style_config = """[pdf_settings]
page_width = 8.5
page_height = 11.0
use_landscape = false
pdf_dpi = 300
pdf_facecolor = "white"
pdf_edgecolor = "black"
"""

    def test_api_imports_correctly(self):
        """Test that the API can be imported correctly."""
        from nvcomms_perf.report import (
            ReportAPI,
            generate_pdf_report_from_dataframe_string,
            generate_pdf_report_from_dataframe_object,
            validate_dataframe_string,
            validate_config_string
        )

        assert ReportAPI is not None
        assert generate_pdf_report_from_dataframe_string is not None
        assert generate_pdf_report_from_dataframe_object is not None
        assert validate_dataframe_string is not None
        assert validate_config_string is not None

    def test_api_class_instantiation(self):
        """Test that the API class can be instantiated."""
        api = ReportAPI()
        assert api is not None
        assert hasattr(api, '_default_plot_config')
        assert hasattr(api, '_default_style_config')

    def test_read_csv_class_method(self):
        """Test the read_csv class method."""
        df = ReportAPI.read_csv(self.sample_csv_data)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 4)
        assert list(df.columns) == ['tool_binary', 'test_name', 'bandwidth_gbps', 'latency_us']

    def test_read_plot_config_class_method(self):
        """Test the read_plot_config class method."""
        config = ReportAPI.read_plot_config(self.sample_plot_config)
        assert config is not None
        assert hasattr(config, 'analysis_profiles')

    def test_read_style_config_class_method(self):
        """Test the read_style_config class method."""
        config = ReportAPI.read_style_config(self.sample_style_config)
        assert config is not None
        assert hasattr(config, 'pdf_settings')
