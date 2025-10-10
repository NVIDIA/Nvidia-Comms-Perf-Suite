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
Tests for the models module.

This module tests all model functionality including:
- Plot configuration models and validation
- Style configuration models and validation
- Configuration loading from TOML files
- Model field validation and constraints
"""

import pytest
import tomli
from pathlib import Path
from unittest.mock import patch, mock_open

from nvcomms_perf.report.models.plot_config import (
    ChartConfig,
    Settings,
    XAxisConfig,
    YAxisConfig,
    YAxisLimits,
    YAxisLimitsTestBinary,
    ChartOrdering,
    DataProcessing,
    StyleMappings,
    ChartType,
    AnalysisProfile,
    PDFConfig,
    ChartConfigLoader
)
from nvcomms_perf.report.models.style_config import (
    StyleConfig,
    TitleStyleConfig,
    StyleConfigLoader
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def config_base_path():
    """Get the base path for configuration files."""
    return Path(__file__).parent.parent.parent / "src" / "nvcomms_perf" / "report" / "config" / "nccl"


@pytest.fixture
def sample_toml_plot_config():
    """Sample TOML content for plot configuration."""
    return """
[settings]
[settings.x_axis]
scale = "semilogx"
log_scale = true

[settings.data_processing]
default_performance_column = "max_bus_bw"
size_column = "test_result.size"

[chart.test_chart]
name = "Test Chart"
description = "A test chart configuration"
performance_columns = ["max_bus_bw"]

[analysis_profiles.basic]
name = "Basic Analysis"
description = "Basic performance analysis"
test_binaries = ["all_reduce_perf"]
chart_types = ["test_chart"]

[pdf_configs.basic_report]
name = "Basic Report"
description = "Basic performance report"
analysis_profiles = ["basic"]
"""


@pytest.fixture
def sample_toml_style_config():
    """Sample TOML content for style configuration."""
    return """
[figure_sizes]
default = [15, 10]
dual_plot = [18, 8]
subplot = [18, 12]

[chart_style]
font_size = 11
line_width = 2.2
marker_size = 5

[legend]
show = true
position = "lower center"
columns = 4
fontsize = 9

[grid]
show = true
alpha = 0.2
color = "#666666"

[title_style]
font_weight = "normal"
font_size_adjustment = 3
max_length = 100
max_lines = 5
wrap_enabled = true
"""


# ============================================================================
# Configuration Loading Tests
# ============================================================================

class TestConfigurationLoading:
    """Test configuration loading from TOML files."""

    @patch('builtins.open', new_callable=mock_open)
    @patch('tomli.load')
    def test_chart_config_loader_success(self, mock_tomli_load, mock_file, sample_toml_plot_config):
        """Test successful chart configuration loading."""
        # Mock TOML data with new array structure
        mock_toml_data = {
            "settings": {
                "x_axis": {"scale": "semilogx", "log_scale": True},
                "data_processing": {
                    "default_performance_column": "max_bus_bw",
                    "size_column": "test_result.size"
                }
            },
            "chart": [
                {
                    "id": "test_chart",
                    "name": "Test Chart",
                    "description": "A test chart configuration",
                    "performance_columns": ["max_bus_bw"]
                }
            ],
            "analysis_profiles": [
                {
                    "id": "basic",
                    "name": "Basic Analysis",
                    "description": "Basic performance analysis",
                    "test_binaries": ["all_reduce_perf"],
                    "chart_types": ["test_chart"]
                }
            ],
            "pdf_configs": [
                {
                    "id": "basic_report",
                    "name": "Basic Report",
                    "description": "Basic performance report",
                    "analysis_profiles": ["basic"]
                }
            ]
        }

        mock_tomli_load.return_value = mock_toml_data

        config = ChartConfigLoader.load_from_toml("test_config.toml")

        assert isinstance(config, ChartConfig)
        assert config.settings is not None
        assert config.settings.x_axis is not None
        assert config.settings.x_axis.log_scale is True
        assert config.chart is not None
        assert len(config.chart) == 1
        assert config.chart[0].id == "test_chart"
        assert config.analysis_profiles is not None
        assert len(config.analysis_profiles) == 1
        assert config.analysis_profiles[0].id == "basic"
        assert config.pdf_configs is not None
        assert len(config.pdf_configs) == 1
        assert config.pdf_configs[0].id == "basic_report"

    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_chart_config_loader_file_not_found(self, mock_file):
        """Test chart configuration loading with missing file."""
        with pytest.raises(FileNotFoundError):
            ChartConfigLoader.load_from_toml("nonexistent.toml")

    @patch('builtins.open', new_callable=mock_open)
    @patch('tomli.load', side_effect=tomli.TOMLDecodeError(msg="Invalid TOML", doc="", pos=0))
    def test_chart_config_loader_invalid_toml(self, mock_tomli_load, mock_file):
        """Test chart configuration loading with invalid TOML."""
        with pytest.raises(tomli.TOMLDecodeError):
            ChartConfigLoader.load_from_toml("invalid.toml")

    @patch('builtins.open', new_callable=mock_open)
    @patch('tomli.load')
    def test_style_config_loader_success(self, mock_tomli_load, mock_file):
        """Test successful style configuration loading."""
        # Mock TOML data
        mock_toml_data = {
            "figure_sizes": {
                "default": [15, 10],
                "dual_plot": [18, 8],
                "subplot": [18, 12]
            },
            "chart_style": {
                "font_size": 11,
                "line_width": 2.2,
                "marker_size": 5
            },
            "legend": {
                "show": True,
                "position": "lower center",
                "columns": 4,
                "fontsize": 9
            },
            "title_style": {
                "font_weight": "normal",
                "font_size_adjustment": 3,
                "max_length": 100,
                "max_lines": 5,
                "wrap_enabled": True
            }
        }

        mock_tomli_load.return_value = mock_toml_data

        config = StyleConfigLoader.load_from_toml("test_style.toml")

        assert isinstance(config, StyleConfig)
        assert config.figure_sizes.default == [15, 10]
        assert config.chart_style.font_size == 11
        assert config.legend.show is True
        assert config.title_style.font_weight == "normal"

    def test_load_actual_toml_file(self, config_base_path):
        """Test loading actual TOML configuration files."""
        plot_config_path = config_base_path / "plot" / "plot_config.toml"
        style_config_path = config_base_path / "style" / "style_config.toml"

        if plot_config_path.exists():
            config = ChartConfigLoader.load_from_toml(str(plot_config_path))
            assert isinstance(config, ChartConfig)

        if style_config_path.exists():
            config = StyleConfigLoader.load_from_toml(str(style_config_path))
            assert isinstance(config, StyleConfig)
            assert config.figure_sizes is not None
            assert config.chart_style is not None

    def test_configuration_helper_methods(self):
        """Test configuration helper methods."""
        # Test that configurations can be serialized/validated
        plot_config = ChartConfig()
        style_config = StyleConfig()

        # These should not raise exceptions
        assert isinstance(plot_config, ChartConfig)
        assert isinstance(style_config, StyleConfig)

        # Test model serialization
        plot_dict = plot_config.model_dump()
        style_dict = style_config.model_dump()

        assert isinstance(plot_dict, dict)
        assert isinstance(style_dict, dict)


# ============================================================================
# Integration and Validation Tests
# ============================================================================

class TestModelIntegration:
    """Test model integration and cross-validation."""

    def test_chart_config_validation_errors(self):
        """Test ChartConfig validation errors."""
        # Test chart type name uniqueness validation
        chart_types = [
            ChartType(id="chart1", name="Same Name", description="First chart", performance_columns=["col1"]),
            ChartType(id="chart2", name="Same Name", description="Second chart", performance_columns=["col2"])
        ]

        with pytest.raises(ValueError, match="Chart type names must be unique"):
            ChartConfig(chart=chart_types)

    def test_model_field_constraints(self):
        """Test that model field constraints are properly enforced."""
        # Test that models can be instantiated with valid parameters
        chart_type = ChartType(id="test", name="Test", description="Test", performance_columns=["col1"])
        assert chart_type.name == "Test"

        profile = AnalysisProfile(
            id="test",
            name="Test",
            description="Test",
            test_binaries=["test"],
            chart_types=["chart1"]
        )
        assert profile.name == "Test"

        pdf_config = PDFConfig(
            id="test",
            name="Test",
            description="Test",
            analysis_profiles=["profile1"]
        )
        assert pdf_config.name == "Test"

    def test_model_defaults_consistency(self):
        """Test that model defaults are consistent across related models."""
        # Test that default values are sensible
        x_axis = XAxisConfig(scale="semilogx", log_scale=True)
        assert x_axis.log_scale is True
        assert x_axis.scale == "semilogx"

        y_axis_limits = YAxisLimits(
            auto_detect_margin=0.1,
            auto_detect_round_to=50,
            default_min=0,
            default_max=None,
            p2p_min=0,
            p2p_max=None
        )
        assert y_axis_limits.default_min == 0
        assert y_axis_limits.auto_detect_margin == 0.1

        title_style = TitleStyleConfig()
        assert title_style.wrap_enabled is True
        assert title_style.max_lines > 0
        assert title_style.max_length > 0

    def test_configuration_reference_validation(self):
        """Test validation of cross-references between configuration sections."""
        # Test that analysis profiles can reference valid chart types
        chart = [ChartType(
            id="valid_chart",
            name="Valid Chart",
            description="Valid chart description",
            performance_columns=["bandwidth"]
        )]

        profile = AnalysisProfile(
            id="test_profile",
            name="Test Profile",
            description="Test profile description",
            test_binaries=["test_binary"],
            chart_types=["valid_chart"]
        )

        pdf_config = PDFConfig(
            id="test_pdf",
            name="Test PDF",
            description="Test PDF description",
            analysis_profiles=["test_profile"]
        )

        # This should work without validation errors
        config = ChartConfig(
            chart=chart,
            analysis_profiles=[profile],
            pdf_configs=[pdf_config]
        )

        assert config.chart is not None and len(config.chart) > 0
        assert config.chart[0].id == "valid_chart"
        assert config.analysis_profiles is not None and len(config.analysis_profiles) > 0
        assert config.analysis_profiles[0].id == "test_profile"
        assert config.pdf_configs is not None and len(config.pdf_configs) > 0
        assert config.pdf_configs[0].id == "test_pdf"

    def test_model_serialization_roundtrip(self):
        """Test that models can be serialized and deserialized correctly."""
        # Create a complex configuration
        x_axis = XAxisConfig(log_scale=True, scale="semilogx")
        settings = Settings(x_axis=x_axis)
        chart_type = ChartType(id="test", name="Test", description="Test chart", performance_columns=["test_col"])

        # Test serialization
        chart_dict = chart_type.model_dump()
        assert "id" in chart_dict
        assert "name" in chart_dict
        assert "description" in chart_dict

        # Test deserialization
        restored_chart = ChartType(**chart_dict)
        assert restored_chart.id == "test"
        assert restored_chart.name == "Test"
        assert restored_chart.description == "Test chart"

        # Test full configuration
        config = ChartConfig(
            settings=settings,
            chart=[chart_type]
        )

        config_dict = config.model_dump()
        restored_config = ChartConfig(**config_dict)

        assert restored_config.settings is not None
        assert restored_config.settings.x_axis is not None
        assert restored_config.settings.x_axis.log_scale is True
        assert restored_config.chart is not None and len(restored_config.chart) == 1
        assert restored_config.chart[0].id == "test"
