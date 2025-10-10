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
Tests for plot configuration models and functionality.

This module tests the plot configuration models and related functionality.
"""

import pytest
from typing import Dict, Any
from unittest.mock import Mock, patch

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


class TestXAxisConfig:
    """Test the XAxisConfig model."""

    def test_x_axis_config_default_values(self):
        """Test that XAxisConfig has correct default values."""
        config = XAxisConfig()  # type: ignore
        assert config.scale == "semilogx"
        assert config.log_scale is True

    def test_x_axis_config_custom_values(self):
        """Test that XAxisConfig accepts custom values."""
        config = XAxisConfig(scale="linear", log_scale=False)
        assert config.scale == "linear"
        assert config.log_scale is False


class TestYAxisLimits:
    """Test the YAxisLimits model."""

    def test_y_axis_limits_default_values(self):
        """Test that YAxisLimits has correct default values."""
        limits = YAxisLimits()  # type: ignore

        # Test Y-axis defaults
        assert limits.auto_detect_margin == 0.1
        assert limits.auto_detect_round_to == 50
        assert limits.default_min == 0
        assert limits.default_max is None
        assert limits.p2p_min == 0
        assert limits.p2p_max is None
        assert limits.test_binaries == {}

    def test_y_axis_limits_custom_values(self):
        """Test that YAxisLimits accepts custom values."""
        limits = YAxisLimits(  # type: ignore
            auto_detect_margin=0.2,
            auto_detect_round_to=100,
            default_min=10,
            default_max=1000,
            p2p_min=5,
            p2p_max=500,
            test_binaries={
                "test_binary": YAxisLimitsTestBinary(min=1, max=100)
            }
        )

        assert limits.auto_detect_margin == 0.2
        assert limits.auto_detect_round_to == 100
        assert limits.default_min == 10
        assert limits.default_max == 1000
        assert limits.p2p_min == 5
        assert limits.p2p_max == 500
        assert limits.test_binaries is not None
        assert "test_binary" in limits.test_binaries
        assert limits.test_binaries["test_binary"].min == 1
        assert limits.test_binaries["test_binary"].max == 100


class TestChartConfig:
    """Test the ChartConfig model."""

    def test_chart_config_with_axis_configs(self):
        """Test that ChartConfig works with the new axis configuration structure."""
        config = ChartConfig(
            settings=Settings(
                x_axis=XAxisConfig(scale="linear", log_scale=False),
                y_axis=YAxisConfig(limits=YAxisLimits(default_min=0, default_max=100))  # type: ignore
            )
        )

        assert config.settings is not None
        assert config.settings.x_axis is not None
        assert config.settings.x_axis.scale == "linear"
        assert config.settings.x_axis.log_scale is False
        assert config.settings.y_axis is not None
        assert config.settings.y_axis.limits is not None
        assert config.settings.y_axis.limits.default_min == 0
        assert config.settings.y_axis.limits.default_max == 100


class TestChartConfigLoader:
    """Test the ChartConfigLoader utility class."""

    @patch('tomli.load')
    @patch('builtins.open')
    def test_load_from_toml_with_x_axis_config(self, mock_open, mock_tomli_load):
        """Test loading TOML configuration with X-axis configuration."""
        # Mock TOML data with X-axis configuration
        mock_toml_data = {
            "settings": {
                "x_axis": {
                    "scale": "linear",
                    "log_scale": False
                },
                "y_axis": {
                    "limits": {
                        "default_min": 0,
                        "default_max": 1000
                    }
                }
            }
        }

        mock_tomli_load.return_value = mock_toml_data
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file

        # Load configuration
        config = ChartConfigLoader.load_from_toml("test_config.toml")

        # Verify the configuration was loaded correctly
        assert config.settings is not None
        assert config.settings.x_axis is not None
        assert config.settings.x_axis.scale == "linear"
        assert config.settings.x_axis.log_scale is False
        assert config.settings.y_axis is not None
        assert config.settings.y_axis.limits is not None
        assert config.settings.y_axis.limits.default_min == 0
        assert config.settings.y_axis.limits.default_max == 1000

        # Verify file was opened and TOML was loaded
        mock_open.assert_called_once_with("test_config.toml", 'rb')
        mock_tomli_load.assert_called_once_with(mock_file)
