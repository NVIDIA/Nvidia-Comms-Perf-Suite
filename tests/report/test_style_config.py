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
Tests for style configuration models.
"""

import pytest
import tomli
from typing import Dict
from pathlib import Path

from nvcomms_perf.report.models.style_config import (
    StyleConfig,
    FigureSizesConfig,
    ChartStyleConfig,
    LegendConfig,
    GridConfig,
    AnnotationsConfig,
    ColorSchemeConfig,
    FontAdjustmentsConfig,
    PlotLayoutConfig,
    AnnotationPositioningConfig,
    GridLinesConfig,
    TicksConfig,
    LegendPositioningConfig,
    BackgroundEffectsConfig,
    PageLayoutConfig,
    TextWrappingConfig,
    TableLayoutConfig,
    TextPositioningConfig,
    ColorPaletteConfig,
    PDFSettingsConfig,
    StyleConfigLoader,
    # New code display config classes
    CodeDisplayConfig,
    CodeBlockStyleConfig,
    CodeTextStyleConfig,
    CommandStyleConfig,
    ConfigStyleConfig
)


class TestStyleConfig:
    """Test StyleConfig model and its sub-models."""

    def test_figure_sizes_config_validation(self):
        """Test FigureSizesConfig validation."""
        # Valid configuration
        config = FigureSizesConfig(default=[15, 10], dual_plot=[18, 8], subplot=[18, 12])
        assert config.default == [15, 10]
        assert config.dual_plot == [18, 8]
        assert config.subplot == [18, 12]

        # Invalid: wrong number of dimensions
        with pytest.raises(ValueError, match="Figure size must have exactly 2 values"):
            FigureSizesConfig(default=[15])

        # Invalid: negative dimensions
        with pytest.raises(ValueError, match="Figure dimensions must be positive"):
            FigureSizesConfig(default=[15, -10])

    def test_chart_style_config_validation(self):
        """Test ChartStyleConfig validation with enhanced background options."""
        # Valid configuration with enhanced background options
        config = ChartStyleConfig(
            font_size=11,
            line_width=2.2,
            marker_size=5,
            background_color="#ffffff",
            # Test enhanced background options
            background_gradient=True,
            background_gradient_alpha=0.05,
            background_pattern="dots"
        )
        assert config.font_size == 11
        assert config.line_width == 2.2
        assert config.marker_size == 5
        assert config.background_color == "#ffffff"
        assert config.background_gradient is True
        assert config.background_gradient_alpha == 0.05
        assert config.background_pattern == "dots"

        # Test default values for enhanced options
        config_defaults = ChartStyleConfig()
        assert config_defaults.background_gradient is False  # Default from TOML
        assert config_defaults.background_gradient_alpha == 0.05  # Default from TOML
        assert config_defaults.background_pattern == "none"  # Default from TOML

    def test_legend_config_validation(self):
        """Test LegendConfig validation."""
        # Valid configuration
        config = LegendConfig(
            show=True,
            position="lower center",
            columns=4,
            alpha=0.95
        )
        assert config.show is True
        assert config.position == "lower center"
        assert config.columns == 4
        assert config.alpha == 0.95

        # Invalid: alpha out of range
        with pytest.raises(ValueError):
            LegendConfig(alpha=1.5)

        # Invalid: negative columns
        with pytest.raises(ValueError):
            LegendConfig(columns=0)

    def test_grid_config_validation(self):
        """Test GridConfig validation."""
        # Valid configuration
        config = GridConfig(
            show=True,
            alpha=0.2,
            color="#666666",
            style="solid",
            minor_alpha=0.1
        )
        assert config.show is True
        assert config.alpha == 0.2
        assert config.color == "#666666"
        assert config.style == "solid"
        assert config.minor_alpha == 0.1

        # Test new minor gridline configuration options
        config_with_minor = GridConfig(
            show=True,
            alpha=0.2,
            color="#666666",
            style="solid",
            minor_alpha=0.1,
            minor_x_axis=True,
            minor_y_axis=False,
            # Test enhanced grid options
            major_linewidth=0.8,
            minor_linewidth=0.4,
            major_style="solid",
            minor_style="dashed",
            major_color="#666666",
            minor_color="#999999"
        )
        assert config_with_minor.minor_x_axis is True
        assert config_with_minor.minor_y_axis is False
        assert config_with_minor.major_linewidth == 0.8
        assert config_with_minor.minor_linewidth == 0.4
        assert config_with_minor.major_style == "solid"
        assert config_with_minor.minor_style == "dashed"
        assert config_with_minor.major_color == "#666666"
        assert config_with_minor.minor_color == "#999999"

        # Test default values for new options
        config_defaults = GridConfig()
        assert config_defaults.minor_x_axis is True  # Default from TOML
        assert config_defaults.minor_y_axis is True  # Default from TOML

        # Invalid: alpha out of range
        with pytest.raises(ValueError):
            GridConfig(alpha=-0.1)

    def test_annotations_config_validation(self):
        """Test AnnotationsConfig validation."""
        # Valid configuration
        config = AnnotationsConfig(
            show_final_bw=True,
            fontsize=9,
            fontweight="bold",
            zorder=5,
            bbox_alpha=0.92
        )
        assert config.show_final_bw is True
        assert config.fontsize == 9
        assert config.fontweight == "bold"
        assert config.zorder == 5
        assert config.bbox_alpha == 0.92

        # Invalid: fontsize too small
        with pytest.raises(ValueError):
            AnnotationsConfig(fontsize=0)

    def test_ticks_config_validation(self):
        """Test TicksConfig validation with enhanced options."""
        # Valid configuration with enhanced options
        config = TicksConfig(
            minor_length=3,
            minor_width=0.5,
            major_length=5,
            major_width=0.8,
            # Test enhanced tick options
            minor_color="#999999",
            major_color="#333333",
            tick_direction="out",
            minor_alpha=0.7,
            major_alpha=1.0
        )
        assert config.minor_length == 3
        assert config.minor_width == 0.5
        assert config.major_length == 5
        assert config.major_width == 0.8
        assert config.minor_color == "#999999"
        assert config.major_color == "#333333"
        assert config.tick_direction == "out"
        assert config.minor_alpha == 0.7
        assert config.major_alpha == 1.0

        # Test default values for enhanced options
        config_defaults = TicksConfig()
        assert config_defaults.minor_color == "#999999"  # Default from TOML
        assert config_defaults.major_color == "#333333"  # Default from TOML
        assert config_defaults.tick_direction == "out"  # Default from TOML
        assert config_defaults.minor_alpha == 0.7  # Default from TOML
        assert config_defaults.major_alpha == 1.0  # Default from TOML

    def test_pdf_settings_config_validation(self):
        """Test PDFSettingsConfig validation."""
        # Valid configuration
        config = PDFSettingsConfig(
            use_landscape=True,
            page_width=11.0,
            page_height=14.0,
            margin_left=0.5,
            pdf_dpi=300
        )
        assert config.use_landscape is True
        assert config.page_width == 11.0
        assert config.page_height == 14.0
        assert config.margin_left == 0.5
        assert config.pdf_dpi == 300

        # Invalid: negative page dimensions
        with pytest.raises(ValueError):
            PDFSettingsConfig(page_width=-11.0)

        # Invalid: dpi too low
        with pytest.raises(ValueError):
            PDFSettingsConfig(pdf_dpi=0)

    def test_style_config_defaults(self):
        """Test StyleConfig with default values."""
        config = StyleConfig()

        # Check that all sub-configs are initialized with defaults
        assert isinstance(config.figure_sizes, FigureSizesConfig)
        assert isinstance(config.chart_style, ChartStyleConfig)
        assert isinstance(config.legend, LegendConfig)
        assert isinstance(config.grid, GridConfig)
        assert isinstance(config.annotations, AnnotationsConfig)

        assert isinstance(config.color_scheme, ColorSchemeConfig)
        assert isinstance(config.font_adjustments, FontAdjustmentsConfig)
        assert isinstance(config.plot_layout, PlotLayoutConfig)
        assert isinstance(config.annotation_positioning, AnnotationPositioningConfig)
        assert isinstance(config.grid_lines, GridLinesConfig)
        assert isinstance(config.ticks, TicksConfig)
        assert isinstance(config.legend_positioning, LegendPositioningConfig)
        assert isinstance(config.background_effects, BackgroundEffectsConfig)
        assert isinstance(config.page_layout, PageLayoutConfig)
        assert isinstance(config.text_wrapping, TextWrappingConfig)
        assert isinstance(config.table_layout, TableLayoutConfig)
        assert isinstance(config.text_positioning, TextPositioningConfig)
        assert isinstance(config.color_palette, ColorPaletteConfig)
        assert isinstance(config.pdf_settings, PDFSettingsConfig)

        # Check default dictionaries
        assert config.split_mask_y_limits == {"0x0": 400, "0x7": 400}
        assert config.operation_y_limits == {"alltoall_sendrecv": 400}

    def test_style_config_from_toml(self):
        """Test loading StyleConfig from TOML dictionary."""
        toml_dict = {
            "figure_sizes": {
                "default": [15, 10],
                "dual_plot": [18, 8],
                "subplot": [18, 12]
            },
            "chart_style": {
                "font_size": 11,
                "line_width": 2.2,
                "thick_line_width": 3.0,
                "marker_size": 5,
                "markeredgewidth": 1.2,
                "markeredgecolor": "white",
                "background_color": "#ffffff",
                "text_color": "#333333"
            },
            "legend": {
                "show": True,
                "position": "lower center",
                "columns": 4,
                "alpha": 0.95
            },
            "grid": {
                "show": True,
                "alpha": 0.2,
                "color": "#666666",
                "style": "solid",
                "minor_alpha": 0.1
            },
            "split_mask_y_limits": {
                "0x0": 400,
                "0x7": 400
            },
            "operation_y_limits": {
                "alltoall_sendrecv": 400
            }
        }

        config = StyleConfig.from_toml(toml_dict)

        assert config.figure_sizes.default == [15, 10]
        assert config.chart_style.font_size == 11
        assert config.legend.show is True
        assert config.grid.alpha == 0.2
        assert config.split_mask_y_limits["0x0"] == 400
        assert config.operation_y_limits["alltoall_sendrecv"] == 400

    def test_style_config_helper_methods(self):
        """Test StyleConfig helper methods."""
        config = StyleConfig()

        # Test get_figure_size
        assert config.get_figure_size('default') == [15, 10]
        assert config.get_figure_size('dual_plot') == [18, 8]
        assert config.get_figure_size('subplot') == [18, 12]
        assert config.get_figure_size('unknown') == [15, 10]  # Falls back to default

        # Test get_font_size
        assert config.get_font_size() == 11  # Base size
        assert config.get_font_size('title_large') == 16  # 11 + 5
        assert config.get_font_size('title_medium') == 14  # 11 + 3
        assert config.get_font_size('label_small') == 10  # 11 - 1
        assert config.get_font_size('table_header') == 9  # 11 - 2

        # Test get_y_limit
        # Test with split mask
        assert config.get_y_limit('allreduce', '0x0') == 400
        assert config.get_y_limit('allreduce', '0x7') == 400

        # Test with operation-specific limits
        assert config.get_y_limit('alltoall_sendrecv') == 400

        # Test default limits
        assert config.get_y_limit('allreduce') == 400  # collective ops
        assert config.get_y_limit('sendrecv') == 50  # p2p ops
        assert config.get_y_limit('send') == 50  # p2p ops
        assert config.get_y_limit('recv') == 50  # p2p ops

    def test_load_actual_toml_file(self):
        """Test loading the actual style_config.toml file."""
        # Get the path to the actual TOML file
        toml_path = Path(__file__).parent.parent / "src" / "nvcomms_perf" / "report" / "config" / "nccl" / "style" / "style_config.toml"

        if toml_path.exists():
            # Load the TOML file
            with open(toml_path, 'rb') as f:
                toml_dict = tomli.load(f)

            # Create StyleConfig from the loaded TOML
            config = StyleConfig.from_toml(toml_dict)

            # Verify some key values from the actual TOML
            assert config.figure_sizes.default == [15, 10]
            assert config.chart_style.font_size == 11
            assert config.chart_style.line_width == 2.2
            assert config.legend.position == "lower center"
            assert config.legend.columns == 4
            assert config.legend_positioning.adaptive_columns is False
            assert config.grid.alpha == 0.2
            assert config.pdf_settings.use_landscape is True
            assert config.pdf_settings.page_width == 11.0
            assert config.pdf_settings.page_height == 14.0

            # Verify dictionaries
            assert config.split_mask_y_limits["0x0"] == 400
            assert config.split_mask_y_limits["0x7"] == 400
            assert config.operation_y_limits["alltoall_sendrecv"] == 400

            # Verify helper methods work with loaded config
            assert config.get_figure_size('dual_plot') == [18, 8]
            assert config.get_font_size('title_large') == 16
            assert config.get_y_limit('allreduce', '0x0') == 400

    def test_validation_errors(self):
        """Test that validation errors are properly raised."""
        # Test invalid TOML dict
        invalid_toml = {
            "figure_sizes": {
                "default": [15]  # Invalid: only one dimension
            }
        }

        with pytest.raises(ValueError):
            StyleConfig.from_toml(invalid_toml)

        # Test invalid type
        invalid_toml = {
            "chart_style": {
                "font_size": "eleven"  # Invalid: should be int
            }
        }

        with pytest.raises(ValueError):
            StyleConfig.from_toml(invalid_toml)


class TestCodeDisplayConfig:
    """Test the new code display configuration classes."""

    def test_code_block_style_config_validation(self):
        """Test CodeBlockStyleConfig validation."""
        # Valid configuration
        config = CodeBlockStyleConfig(
            background_color="#f8f9fa",
            border_color="#e9ecef",
            border_width=2,
            border_style="dashed",
            corner_radius=6,
            shadow_enabled=True,
            shadow_offset_x=3,
            shadow_offset_y=3,
            shadow_blur=6,
            shadow_color="#000000",
            shadow_alpha=0.15
        )
        assert config.background_color == "#f8f9fa"
        assert config.border_color == "#e9ecef"
        assert config.border_width == 2
        assert config.border_style == "dashed"
        assert config.corner_radius == 6
        assert config.shadow_enabled is True
        assert config.shadow_offset_x == 3
        assert config.shadow_offset_y == 3
        assert config.shadow_blur == 6
        assert config.shadow_color == "#000000"
        assert config.shadow_alpha == 0.15

        # Test default values
        config_defaults = CodeBlockStyleConfig()
        assert config_defaults.background_color == "#f8f9fa"
        assert config_defaults.border_color == "#e9ecef"
        assert config_defaults.border_width == 1
        assert config_defaults.border_style == "solid"
        assert config_defaults.corner_radius == 4
        assert config_defaults.shadow_enabled is True
        assert config_defaults.shadow_offset_x == 2
        assert config_defaults.shadow_offset_y == 2
        assert config_defaults.shadow_blur == 4
        assert config_defaults.shadow_color == "#000000"
        assert config_defaults.shadow_alpha == 0.1

        # Invalid: negative border width
        with pytest.raises(ValueError, match="Input should be greater than or equal to 0"):
            CodeBlockStyleConfig(border_width=-1)

        # Invalid: negative corner radius
        with pytest.raises(ValueError, match="Input should be greater than or equal to 0"):
            CodeBlockStyleConfig(corner_radius=-1)

        # Invalid: negative shadow blur
        with pytest.raises(ValueError, match="Input should be greater than or equal to 0"):
            CodeBlockStyleConfig(shadow_blur=-1)

        # Invalid: shadow alpha out of range
        with pytest.raises(ValueError, match="Input should be less than or equal to 1"):
            CodeBlockStyleConfig(shadow_alpha=1.5)

    def test_code_text_style_config_validation(self):
        """Test CodeTextStyleConfig validation."""
        # Valid configuration
        config = CodeTextStyleConfig(
            font_family="Consolas",
            font_size=10,
            line_height=1.4,
            color="#1a1a1a",
            selection_color="#0056b3",
            selection_background="#e6f3ff"
        )
        assert config.font_family == "Consolas"
        assert config.font_size == 10
        assert config.line_height == 1.4
        assert config.color == "#1a1a1a"
        assert config.selection_color == "#0056b3"
        assert config.selection_background == "#e6f3ff"

        # Test default values
        config_defaults = CodeTextStyleConfig()
        assert config_defaults.font_family == "Courier New"
        assert config_defaults.font_size == 9
        assert config_defaults.line_height == 1.2
        assert config_defaults.color == "#212529"
        assert config_defaults.selection_color == "#007bff"
        assert config_defaults.selection_background == "#e3f2fd"

        # Invalid: font size too small
        with pytest.raises(ValueError, match="Input should be greater than or equal to 1"):
            CodeTextStyleConfig(font_size=0)

        # Invalid: line height too small
        with pytest.raises(ValueError, match="Input should be greater than 0"):
            CodeTextStyleConfig(line_height=0)

    def test_command_style_config_validation(self):
        """Test CommandStyleConfig validation."""
        # Valid configuration
        config = CommandStyleConfig(
            prompt_color="#495057",
            command_color="#1a1a1a",
            argument_color="#6c757d",
            option_color="#0056b3",
            variable_color="#28a745",
            comment_color="#6c757d"
        )
        assert config.prompt_color == "#495057"
        assert config.command_color == "#1a1a1a"
        assert config.argument_color == "#6c757d"
        assert config.option_color == "#0056b3"
        assert config.variable_color == "#28a745"
        assert config.comment_color == "#6c757d"

        # Test default values
        config_defaults = CommandStyleConfig()
        assert config_defaults.prompt_color == "#6c757d"
        assert config_defaults.command_color == "#212529"
        assert config_defaults.argument_color == "#495057"
        assert config_defaults.option_color == "#007bff"
        assert config_defaults.variable_color == "#28a745"
        assert config_defaults.comment_color == "#6c757d"

    def test_config_style_config_validation(self):
        """Test ConfigStyleConfig validation."""
        # Valid configuration
        config = ConfigStyleConfig(
            section_header_color="#343a40",
            section_header_background="#f8f9fa",
            key_color="#0056b3",
            value_color="#1a1a1a",
            comment_color="#6c757d",
            separator_color="#dee2e6"
        )
        assert config.section_header_color == "#343a40"
        assert config.section_header_background == "#f8f9fa"
        assert config.key_color == "#0056b3"
        assert config.value_color == "#1a1a1a"
        assert config.comment_color == "#6c757d"
        assert config.separator_color == "#dee2e6"

        # Test default values
        config_defaults = ConfigStyleConfig()
        assert config_defaults.section_header_color == "#495057"
        assert config_defaults.section_header_background == "#e9ecef"
        assert config_defaults.key_color == "#007bff"
        assert config_defaults.value_color == "#212529"
        assert config_defaults.comment_color == "#6c757d"
        assert config_defaults.separator_color == "#dee2e6"

    def test_code_display_config_validation(self):
        """Test CodeDisplayConfig validation."""
        # Valid configuration
        config = CodeDisplayConfig(
            show_test_commands=True,
            command_format="highlighted",
            command_max_lines=15,
            command_truncate_length=120,
            show_configurations=True,
            config_format="formatted",
            config_max_lines=25,
            config_truncate_length=90,
            syntax_highlighting=True,
            highlight_theme="github",
            code_font_family="Fira Code",
            code_font_size=10,
            code_block_width=0.85,
            code_block_margin=0.06,
            code_block_padding=0.03,
            include_environment_vars=False,
            include_arguments=True,
            group_commands_by_test=False,
            sort_commands_by_execution_order=False
        )
        assert config.show_test_commands is True
        assert config.command_format == "highlighted"
        assert config.command_max_lines == 15
        assert config.command_truncate_length == 120
        assert config.show_configurations is True
        assert config.config_format == "formatted"
        assert config.config_max_lines == 25
        assert config.config_truncate_length == 90
        assert config.syntax_highlighting is True
        assert config.highlight_theme == "github"
        assert config.code_font_family == "Fira Code"
        assert config.code_font_size == 10
        assert config.code_block_width == 0.85
        assert config.code_block_margin == 0.06
        assert config.code_block_padding == 0.03
        assert config.include_environment_vars is False
        assert config.include_arguments is True
        assert config.group_commands_by_test is False
        assert config.sort_commands_by_execution_order is False

        # Test default values
        config_defaults = CodeDisplayConfig()
        assert config_defaults.show_test_commands is True
        assert config_defaults.command_format == "formatted"
        assert config_defaults.command_max_lines == 10
        assert config_defaults.command_truncate_length == 100
        assert config_defaults.show_configurations is True
        assert config_defaults.config_format == "formatted"
        assert config_defaults.config_max_lines == 20
        assert config_defaults.config_truncate_length == 80
        assert config_defaults.syntax_highlighting is True
        assert config_defaults.highlight_theme == "default"
        assert config_defaults.code_font_family == "monospace"
        assert config_defaults.code_font_size == 9
        assert config_defaults.code_block_width == 0.9
        assert config_defaults.code_block_margin == 0.05
        assert config_defaults.code_block_padding == 0.02
        assert config_defaults.include_environment_vars is True
        assert config_defaults.include_arguments is True
        assert config_defaults.group_commands_by_test is True
        assert config_defaults.sort_commands_by_execution_order is True

        # Test sub-configurations are created
        assert isinstance(config_defaults.block_style, CodeBlockStyleConfig)
        assert isinstance(config_defaults.text_style, CodeTextStyleConfig)
        assert isinstance(config_defaults.command_style, CommandStyleConfig)
        assert isinstance(config_defaults.config_style, ConfigStyleConfig)

        # Invalid: command format
        with pytest.raises(ValueError, match="Input should be 'raw', 'formatted' or 'highlighted'"):
            CodeDisplayConfig(command_format="invalid")  # type: ignore

        # Invalid: config format
        with pytest.raises(ValueError, match="Input should be 'raw', 'formatted' or 'highlighted'"):
            CodeDisplayConfig(config_format="invalid")  # type: ignore

        # Invalid: command max lines too small
        with pytest.raises(ValueError, match="Input should be greater than or equal to 1"):
            CodeDisplayConfig(command_max_lines=0)

        # Invalid: command truncate length too small
        with pytest.raises(ValueError, match="Input should be greater than or equal to 1"):
            CodeDisplayConfig(command_truncate_length=0)

        # Invalid: config max lines too small
        with pytest.raises(ValueError, match="Input should be greater than or equal to 1"):
            CodeDisplayConfig(config_max_lines=0)

        # Invalid: config truncate length too small
        with pytest.raises(ValueError, match="Input should be greater than or equal to 1"):
            CodeDisplayConfig(config_truncate_length=0)

        # Invalid: code font size too small
        with pytest.raises(ValueError, match="Input should be greater than or equal to 1"):
            CodeDisplayConfig(code_font_size=0)

        # Invalid: code block width out of range
        with pytest.raises(ValueError, match="Input should be less than or equal to 1"):
            CodeDisplayConfig(code_block_width=1.5)

        # Invalid: code block width too small
        with pytest.raises(ValueError, match="Input should be greater than 0"):
            CodeDisplayConfig(code_block_width=0)

        # Invalid: code block margin out of range
        with pytest.raises(ValueError, match="Input should be less than or equal to 1"):
            CodeDisplayConfig(code_block_margin=1.5)

        # Invalid: code block margin negative
        with pytest.raises(ValueError, match="Input should be greater than or equal to 0"):
            CodeDisplayConfig(code_block_margin=-0.1)

        # Invalid: code block padding out of range
        with pytest.raises(ValueError, match="Input should be less than or equal to 1"):
            CodeDisplayConfig(code_block_padding=1.5)

        # Invalid: code block padding negative
        with pytest.raises(ValueError, match="Input should be greater than or equal to 0"):
            CodeDisplayConfig(code_block_padding=-0.1)

    def test_code_display_config_custom_sub_configs(self):
        """Test CodeDisplayConfig with custom sub-configurations."""
        # Create custom sub-configurations
        custom_block_style = CodeBlockStyleConfig(
            background_color="#2d3748",
            border_color="#4a5568",
            border_width=2,
            corner_radius=8
        )

        custom_text_style = CodeTextStyleConfig(
            font_family="JetBrains Mono",
            font_size=11,
            color="#e2e8f0"
        )

        custom_command_style = CommandStyleConfig(
            prompt_color="#a0aec0",
            command_color="#f7fafc",
            option_color="#63b3ed"
        )

        custom_config_style = ConfigStyleConfig(
            section_header_color="#2d3748",
            key_color="#63b3ed",
            value_color="#f7fafc"
        )

        # Create CodeDisplayConfig with custom sub-configs
        config = CodeDisplayConfig(
            block_style=custom_block_style,
            text_style=custom_text_style,
            command_style=custom_command_style,
            config_style=custom_config_style
        )

        # Verify custom sub-configs are used
        assert config.block_style.background_color == "#2d3748"
        assert config.block_style.border_color == "#4a5568"
        assert config.block_style.border_width == 2
        assert config.block_style.corner_radius == 8

        assert config.text_style.font_family == "JetBrains Mono"
        assert config.text_style.font_size == 11
        assert config.text_style.color == "#e2e8f0"

        assert config.command_style.prompt_color == "#a0aec0"
        assert config.command_style.command_color == "#f7fafc"
        assert config.command_style.option_color == "#63b3ed"

        assert config.config_style.section_header_color == "#2d3748"
        assert config.config_style.key_color == "#63b3ed"
        assert config.config_style.value_color == "#f7fafc"

    def test_style_config_with_code_display_defaults(self):
        """Test that StyleConfig includes code_display with default values."""
        config = StyleConfig()

        # Verify code_display is included
        assert hasattr(config, 'code_display')
        assert isinstance(config.code_display, CodeDisplayConfig)

        # Verify default values are set
        assert config.code_display.show_test_commands is True
        assert config.code_display.command_format == "formatted"
        assert config.code_display.show_configurations is True
        assert config.code_display.config_format == "formatted"
        assert config.code_display.syntax_highlighting is True

        # Verify sub-configurations are created
        assert isinstance(config.code_display.block_style, CodeBlockStyleConfig)
        assert isinstance(config.code_display.text_style, CodeTextStyleConfig)
        assert isinstance(config.code_display.command_style, CommandStyleConfig)
        assert isinstance(config.code_display.config_style, ConfigStyleConfig)

    def test_style_config_from_toml_with_code_display(self):
        """Test loading StyleConfig from TOML with code display configuration."""
        # Mock TOML data with code display configuration
        toml_data = {
            "code_display": {
                "show_test_commands": False,
                "command_format": "highlighted",
                "command_max_lines": 20,
                "command_truncate_length": 150,
                "show_configurations": False,
                "config_format": "raw",
                "config_max_lines": 30,
                "config_truncate_length": 120,
                "syntax_highlighting": False,
                "highlight_theme": "monokai",
                "code_font_family": "Source Code Pro",
                "code_font_size": 12,
                "code_block_width": 0.8,
                "code_block_margin": 0.08,
                "code_block_padding": 0.04,
                "include_environment_vars": False,
                "include_arguments": False,
                "group_commands_by_test": False,
                "sort_commands_by_execution_order": False,
                "block_style": {
                    "background_color": "#1a1a1a",
                    "border_color": "#333333",
                    "border_width": 3,
                    "border_style": "double",
                    "corner_radius": 10,
                    "shadow_enabled": False
                },
                "text_style": {
                    "font_family": "Fira Code",
                    "font_size": 11,
                    "line_height": 1.5,
                    "color": "#ffffff"
                },
                "command_style": {
                    "prompt_color": "#888888",
                    "command_color": "#ffffff",
                    "argument_color": "#cccccc",
                    "option_color": "#00ff00",
                    "variable_color": "#ffff00",
                    "comment_color": "#888888"
                },
                "config_style": {
                    "section_header_color": "#ffffff",
                    "section_header_background": "#333333",
                    "key_color": "#00ff00",
                    "value_color": "#ffffff",
                    "comment_color": "#888888",
                    "separator_color": "#666666"
                }
            }
        }

        # Load configuration
        config = StyleConfig.from_toml(toml_data)

        # Verify code display configuration was loaded correctly
        assert config.code_display.show_test_commands is False
        assert config.code_display.command_format == "highlighted"
        assert config.code_display.command_max_lines == 20
        assert config.code_display.command_truncate_length == 150
        assert config.code_display.show_configurations is False
        assert config.code_display.config_format == "raw"
        assert config.code_display.config_max_lines == 30
        assert config.code_display.config_truncate_length == 120
        assert config.code_display.syntax_highlighting is False
        assert config.code_display.highlight_theme == "monokai"
        assert config.code_display.code_font_family == "Source Code Pro"
        assert config.code_display.code_font_size == 12
        assert config.code_display.code_block_width == 0.8
        assert config.code_display.code_block_margin == 0.08
        assert config.code_display.code_block_padding == 0.04
        assert config.code_display.include_environment_vars is False
        assert config.code_display.include_arguments is False
        assert config.code_display.group_commands_by_test is False
        assert config.code_display.sort_commands_by_execution_order is False

        # Verify sub-configurations were loaded correctly
        assert config.code_display.block_style.background_color == "#1a1a1a"
        assert config.code_display.block_style.border_color == "#333333"
        assert config.code_display.block_style.border_width == 3
        assert config.code_display.block_style.border_style == "double"
        assert config.code_display.block_style.corner_radius == 10
        assert config.code_display.block_style.shadow_enabled is False

        assert config.code_display.text_style.font_family == "Fira Code"
        assert config.code_display.text_style.font_size == 11
        assert config.code_display.text_style.line_height == 1.5
        assert config.code_display.text_style.color == "#ffffff"

        assert config.code_display.command_style.prompt_color == "#888888"
        assert config.code_display.command_style.command_color == "#ffffff"
        assert config.code_display.command_style.argument_color == "#cccccc"
        assert config.code_display.command_style.option_color == "#00ff00"
        assert config.code_display.command_style.variable_color == "#ffff00"
        assert config.code_display.command_style.comment_color == "#888888"

        assert config.code_display.config_style.section_header_color == "#ffffff"
        assert config.code_display.config_style.section_header_background == "#333333"
        assert config.code_display.config_style.key_color == "#00ff00"
        assert config.code_display.config_style.value_color == "#ffffff"
        assert config.code_display.config_style.comment_color == "#888888"
        assert config.code_display.config_style.separator_color == "#666666"

    def test_style_config_from_toml_partial_code_display(self):
        """Test loading StyleConfig from TOML with partial code display configuration."""
        # Mock TOML data with only some code display settings
        toml_data = {
            "code_display": {
                "show_test_commands": False,
                "command_format": "raw",
                "syntax_highlighting": False
            }
        }

        # Load configuration
        config = StyleConfig.from_toml(toml_data)

        # Verify specified values were loaded
        assert config.code_display.show_test_commands is False
        assert config.code_display.command_format == "raw"
        assert config.code_display.syntax_highlighting is False

        # Verify other values use defaults
        assert config.code_display.show_configurations is True  # default
        assert config.code_display.config_format == "formatted"  # default
        assert config.code_display.command_max_lines == 10  # default
        assert config.code_display.highlight_theme == "default"  # default

        # Verify sub-configurations use defaults
        assert config.code_display.block_style.background_color == "#f8f9fa"  # default
        assert config.code_display.text_style.font_family == "Courier New"  # default
        assert config.code_display.command_style.prompt_color == "#6c757d"  # default
        assert config.code_display.config_style.key_color == "#007bff"  # default

    def test_style_config_from_toml_invalid_code_display(self):
        """Test loading StyleConfig from TOML with invalid code display configuration."""
        # Mock TOML data with invalid code display settings
        toml_data = {
            "code_display": {
                "command_format": "invalid_format",  # Invalid format
                "command_max_lines": -1,  # Invalid value
                "code_block_width": 2.0  # Invalid value (should be <= 1)
            }
        }

        # Should raise validation error
        with pytest.raises(ValueError):
            StyleConfig.from_toml(toml_data)

    def test_style_config_loader_with_code_display(self):
        """Test StyleConfigLoader with code display configuration."""
        # Create a temporary TOML file with code display config
        import tempfile
        import os

        toml_content = """
[code_display]
show_test_commands = false
command_format = "highlighted"
command_max_lines = 25
syntax_highlighting = true
highlight_theme = "dracula"
code_font_family = "JetBrains Mono"

[code_display.block_style]
background_color = "#282a36"
border_color = "#44475a"
border_width = 2
corner_radius = 8

[code_display.text_style]
font_family = "JetBrains Mono"
font_size = 10
color = "#f8f8f2"

[code_display.command_style]
prompt_color = "#6272a4"
command_color = "#f8f8f2"
option_color = "#50fa7b"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(toml_content)
            temp_file = f.name

        try:
            # Load configuration using StyleConfigLoader
            config = StyleConfigLoader.load_from_toml(temp_file)

            # Verify code display configuration was loaded correctly
            assert config.code_display.show_test_commands is False
            assert config.code_display.command_format == "highlighted"
            assert config.code_display.command_max_lines == 25
            assert config.code_display.syntax_highlighting is True
            assert config.code_display.highlight_theme == "dracula"
            assert config.code_display.code_font_family == "JetBrains Mono"

            # Verify sub-configurations were loaded correctly
            assert config.code_display.block_style.background_color == "#282a36"
            assert config.code_display.block_style.border_color == "#44475a"
            assert config.code_display.block_style.border_width == 2
            assert config.code_display.block_style.corner_radius == 8

            assert config.code_display.text_style.font_family == "JetBrains Mono"
            assert config.code_display.text_style.font_size == 10
            assert config.code_display.text_style.color == "#f8f8f2"

            assert config.code_display.command_style.prompt_color == "#6272a4"
            assert config.code_display.command_style.command_color == "#f8f8f2"
            assert config.code_display.command_style.option_color == "#50fa7b"

        finally:
            # Clean up temporary file
            os.unlink(temp_file)
