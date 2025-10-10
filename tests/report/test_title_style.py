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
Tests for title style configuration and formatting.

This module tests the new title style configuration system that ensures
consistent title formatting across all report components.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import matplotlib.pyplot as plt

from nvcomms_perf.report.models.style_config import (
    StyleConfig, TitleStyleConfig, StyleConfigLoader
)
from nvcomms_perf.report.plotting.style_manager import StyleManager
from nvcomms_perf.report.plotting.chart_generator import ChartGenerator
from nvcomms_perf.report.adapters.nccl_adapter import NCCLPerformanceAdapter


class TestTitleStyleConfig:
    """Test the TitleStyleConfig class and its validation."""

    def test_title_style_config_defaults(self):
        """Test that TitleStyleConfig has correct default values."""
        config = TitleStyleConfig()

        assert config.font_weight == "normal"
        assert config.font_size_adjustment == 3
        assert config.max_length == 100
        assert config.max_lines == 5
        assert config.wrap_enabled is True
        assert config.alignment == "center"
        assert config.vertical_alignment == "top"
        assert config.color == "#333333"
        assert config.line_spacing == 1.2

    def test_title_style_config_custom_values(self):
        """Test that TitleStyleConfig accepts custom values."""
        config = TitleStyleConfig(
            font_weight="bold",
            font_size_adjustment=5,
            max_length=80,
            max_lines=3,
            wrap_enabled=False,
            alignment="left",
            vertical_alignment="bottom",
            color="#000000",
            line_spacing=1.5
        )

        assert config.font_weight == "bold"
        assert config.font_size_adjustment == 5
        assert config.max_length == 80
        assert config.max_lines == 3
        assert config.wrap_enabled is False
        assert config.alignment == "left"
        assert config.vertical_alignment == "bottom"
        assert config.color == "#000000"
        assert config.line_spacing == 1.5

    def test_title_style_config_font_weight_validation(self):
        """Test that font weight validation works correctly."""
        # Valid values
        valid_weights = ["normal", "bold", "light", "ultralight", "heavy", "ultrabold"]
        for weight in valid_weights:
            config = TitleStyleConfig(font_weight=weight)
            assert config.font_weight == weight

        # Invalid value should raise error
        with pytest.raises(ValueError, match="Font weight must be one of"):
            TitleStyleConfig(font_weight="invalid")

    def test_title_style_config_alignment_validation(self):
        """Test that alignment validation works correctly."""
        # Valid values
        valid_alignments = ["left", "center", "right"]
        for alignment in valid_alignments:
            config = TitleStyleConfig(alignment=alignment)
            assert config.alignment == alignment

        # Invalid value should raise error
        with pytest.raises(ValueError, match="Alignment must be one of"):
            TitleStyleConfig(alignment="invalid")

    def test_title_style_config_vertical_alignment_validation(self):
        """Test that vertical alignment validation works correctly."""
        # Valid values
        valid_alignments = ["top", "center", "bottom"]
        for alignment in valid_alignments:
            config = TitleStyleConfig(vertical_alignment=alignment)
            assert config.vertical_alignment == alignment

        # Invalid value should raise error
        with pytest.raises(ValueError, match="Vertical alignment must be one of"):
            TitleStyleConfig(vertical_alignment="invalid")


class TestStyleConfigTitleMethods:
    """Test the title-related methods in StyleConfig."""

    def test_apply_title_style_no_wrapping(self):
        """Test apply_title_style when wrapping is disabled."""
        config = StyleConfig()
        config.title_style.wrap_enabled = False

        title = "This is a test title"
        result = config.apply_title_style(title)

        assert result == title  # Should return unchanged title

    def test_apply_title_style_with_wrapping(self):
        """Test apply_title_style when wrapping is enabled."""
        config = StyleConfig()
        config.title_style.wrap_enabled = True
        config.title_style.max_length = 20

        # Create a long title that should be wrapped
        long_title = "This is a very long title that should be wrapped"

        # Mock the StyleManager import at the module level where it's used
        with patch('nvcomms_perf.report.plotting.style_manager.StyleManager') as mock_style_manager_class:
            mock_manager_instance = Mock()
            mock_manager_instance.wrap_text.return_value = "This is a very long\ntitle that should be\nwrapped"
            mock_style_manager_class.return_value = mock_manager_instance

            result = config.apply_title_style(long_title)

            # Should call wrap_text with the title and max_length
            mock_manager_instance.wrap_text.assert_called_once_with(long_title, 20)
            assert result == "This is a very long\ntitle that should be\nwrapped"

    def test_apply_title_style_short_title(self):
        """Test apply_title_style with a short title that doesn't need wrapping."""
        config = StyleConfig()
        config.title_style.wrap_enabled = True
        config.title_style.max_length = 100

        short_title = "Short title"
        result = config.apply_title_style(short_title)

        assert result == short_title  # Should return unchanged title

    def test_get_title_font_size(self):
        """Test get_title_font_size calculation."""
        config = StyleConfig()
        config.chart_style.font_size = 12
        config.title_style.font_size_adjustment = 3

        font_size = config.get_title_font_size()

        assert font_size == 15  # 12 + 3

    def test_get_title_font_size_with_different_base(self):
        """Test get_title_font_size with different base font size."""
        config = StyleConfig()
        config.chart_style.font_size = 10
        config.title_style.font_size_adjustment = 5

        font_size = config.get_title_font_size()

        assert font_size == 15  # 10 + 5


class TestTitleStyleIntegration:
    """Test integration of title styling with chart generation."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a minimal style config
        self.style_config = StyleConfig()
        self.style_config.title_style.font_weight = "normal"
        self.style_config.title_style.font_size_adjustment = 3
        self.style_config.title_style.max_length = 50
        self.style_config.title_style.wrap_enabled = True
        self.style_config.title_style.alignment = "center"
        self.style_config.title_style.vertical_alignment = "top"
        self.style_config.title_style.color = "#333333"

        # Create mock adapter and style manager
        self.mock_adapter = Mock(spec=NCCLPerformanceAdapter)
        self.mock_adapter.plot_config = None
        self.style_manager = StyleManager(self.style_config)

        # Create chart generator
        self.chart_generator = ChartGenerator(self.mock_adapter, self.style_manager)

    def test_chart_title_generation_with_wrapping(self):
        """Test that chart titles are properly wrapped."""
        self.style_config.title_style.max_length = 10  # Force wrapping for the test
        # Mock the style manager's wrap_text method
        with patch.object(self.style_manager, 'wrap_text') as mock_wrap:
            mock_wrap.return_value = "Long Chart Title\nThat Needs Wrapping"

            # Create a mock chart group and config
            chart_group = {'group_key': 'test_group', 'data': pd.DataFrame()}
            chart_config = Mock()
            chart_config.name = "This is a very long chart title that should be wrapped according to the configuration"
            chart_config.fixed_params = {}  # Set to empty dict to avoid iteration error

            # Generate title
            title = self.chart_generator._generate_chart_title(
                chart_group, chart_config, "test_binary", 1
            )

            # Should call apply_title_style which should call wrap_text
            mock_wrap.assert_called_once()
            assert title == "Long Chart Title\nThat Needs Wrapping"

    def test_chart_title_generation_without_wrapping(self):
        """Test that short chart titles are not wrapped."""
        # Mock the style manager's wrap_text method
        with patch.object(self.style_manager, 'wrap_text') as mock_wrap:
            # Create a mock chart group and config with short title
            chart_group = {'group_key': 'test_group', 'data': pd.DataFrame()}
            chart_config = Mock()
            chart_config.name = "Short Title"
            chart_config.fixed_params = {}  # Set to empty dict to avoid iteration error

            # Generate title
            title = self.chart_generator._generate_chart_title(
                chart_group, chart_config, "test_binary", 1
            )

            # Should not call wrap_text for short titles
            mock_wrap.assert_not_called()
            assert "Short Title" in title

    def test_command_page_title_styling(self):
        """Test that command page titles use consistent styling."""
        # Mock the style manager's wrap_text method
        with patch.object(self.style_manager, 'wrap_text') as mock_wrap:
            mock_wrap.return_value = "Test Commands: Long Title\nThat Needs Wrapping"

            # Create test data
            chart_title = "This is a very long chart title that should be wrapped"
            legend_commands = {"legend1": ["command1"], "legend2": ["command2"]}

            # Generate command pages
            figures = self.chart_generator._create_multi_legend_command_pages(
                chart_title, legend_commands, 1
            )

            # Should call wrap_text for long titles
            mock_wrap.assert_called()

            # Should create figures
            assert len(figures) > 0

    def test_comprehensive_command_page_title_styling(self):
        """Test that comprehensive command page titles use consistent styling."""
        # Mock the style manager's wrap_text method
        with patch.object(self.style_manager, 'wrap_text') as mock_wrap:
            mock_wrap.return_value = "All Test Commands\n(Page 1 of 2)"

            # Create test data
            all_commands = {
                "legend1": {"command1": [1, 2]},
                "legend2": {"command2": [3, 4]}
            }

            # Generate comprehensive command pages
            figures = self.chart_generator.create_comprehensive_command_page(all_commands)

            # Should call wrap_text for long titles
            mock_wrap.assert_called()

            # Should create figures
            assert len(figures) > 0


class TestTitleStyleConfigurationLoading:
    """Test loading title style configuration from TOML files."""

    def test_load_title_style_from_toml(self):
        """Test that title style configuration loads correctly from TOML."""
        # Create a minimal TOML-like structure
        toml_data = {
            "title_style": {
                "font_weight": "bold",
                "font_size_adjustment": 5,
                "max_length": 80,
                "max_lines": 3,
                "wrap_enabled": False,
                "alignment": "left",
                "vertical_alignment": "bottom",
                "color": "#000000",
                "line_spacing": 1.5
            }
        }

        # Create config from TOML data
        config = StyleConfig.from_toml(toml_data)

        # Verify title style settings
        assert config.title_style.font_weight == "bold"
        assert config.title_style.font_size_adjustment == 5
        assert config.title_style.max_length == 80
        assert config.title_style.max_lines == 3
        assert config.title_style.wrap_enabled is False
        assert config.title_style.alignment == "left"
        assert config.title_style.vertical_alignment == "bottom"
        assert config.title_style.color == "#000000"
        assert config.title_style.line_spacing == 1.5

    def test_title_style_defaults_when_not_in_toml(self):
        """Test that title style uses defaults when not specified in TOML."""
        # Create TOML data without title_style section
        toml_data = {
            "chart_style": {
                "font_size": 12
            }
        }

        # Create config from TOML data
        config = StyleConfig.from_toml(toml_data)

        # Should use default values
        assert config.title_style.font_weight == "normal"
        assert config.title_style.font_size_adjustment == 3
        assert config.title_style.max_length == 100
        assert config.title_style.wrap_enabled is True


class TestTitleStyleConsistency:
    """Test that title styling is consistent across different components."""

    def test_title_font_size_consistency(self):
        """Test that all titles use the same font size calculation."""
        config = StyleConfig()
        config.chart_style.font_size = 10
        config.title_style.font_size_adjustment = 4

        expected_font_size = 14  # 10 + 4

        # All title font size calculations should return the same value
        assert config.get_title_font_size() == expected_font_size

        # Test with different base font size
        config.chart_style.font_size = 12
        expected_font_size = 16  # 12 + 4
        assert config.get_title_font_size() == expected_font_size

    def test_title_wrapping_consistency(self):
        """Test that title wrapping behavior is consistent."""
        config = StyleConfig()
        config.title_style.wrap_enabled = True
        config.title_style.max_length = 30

        long_title = "This is a very long title that should be wrapped consistently"

        # Mock the StyleManager import at the module level where it's used
        with patch('nvcomms_perf.report.plotting.style_manager.StyleManager') as mock_style_manager_class:
            mock_manager_instance = Mock()
            mock_manager_instance.wrap_text.return_value = "This is a very long\ntitle that should be\nwrapped consistently"
            mock_style_manager_class.return_value = mock_manager_instance

            # Multiple calls should produce the same result
            result1 = config.apply_title_style(long_title)
            result2 = config.apply_title_style(long_title)

            assert result1 == result2
            assert mock_manager_instance.wrap_text.call_count == 2
