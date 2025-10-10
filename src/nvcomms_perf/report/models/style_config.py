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
Style configuration models for report generation.

This module provides Pydantic models for validating and managing style configurations
loaded from TOML files. It ensures type safety and provides defaults for all styling options.
"""

from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class FigureSizesConfig(BaseModel):
    """Configuration for figure sizes in inches."""

    default: List[float] = Field(default=[15, 10], description="Default figure size (width, height)")
    dual_plot: List[float] = Field(default=[18, 8], description="Dual plot figure size")
    subplot: List[float] = Field(default=[18, 12], description="Subplot figure size")

    @field_validator('default', 'dual_plot', 'subplot')
    @classmethod
    def validate_figure_size(cls, v: List[float]) -> List[float]:
        """Validate that figure sizes have exactly 2 dimensions."""
        if len(v) != 2:
            raise ValueError("Figure size must have exactly 2 values (width, height)")
        if any(val <= 0 for val in v):
            raise ValueError("Figure dimensions must be positive")
        return v


class ChartStyleConfig(BaseModel):
    """Configuration for chart styling properties."""

    font_size: int = Field(default=11, ge=1)
    line_width: float = Field(default=2.2, gt=0)
    thick_line_width: float = Field(default=3.0, gt=0)
    marker_size: int = Field(default=5, ge=1)
    markeredgewidth: float = Field(default=1.2, gt=0)
    markeredgecolor: str = Field(default="white")
    background_color: str = Field(default="#ffffff")
    text_color: str = Field(default="#333333")
    axes_facecolor: str = Field(default="#ffffff")
    axes_edgecolor: str = Field(default="#333333")
    axes_labelcolor: str = Field(default="#333333")
    axes_spines_top: bool = Field(default=False)
    axes_spines_right: bool = Field(default=False)
    xtick_color: str = Field(default="#333333")
    ytick_color: str = Field(default="#333333")
    # Enhanced background options
    background_gradient: bool = Field(default=False)
    background_gradient_alpha: float = Field(default=0.05, ge=0, le=1)
    background_pattern: str = Field(default="none")  # none, dots, lines


class LegendConfig(BaseModel):
    """Configuration for legend properties."""

    show: bool = Field(default=True)
    position: str = Field(default="lower center")
    columns: int = Field(default=4, ge=1)
    alpha: float = Field(default=0.95, ge=0, le=1)
    shadow: bool = Field(default=True)
    framealpha: float = Field(default=0.95, ge=0, le=1)
    edgecolor: str = Field(default="#e0e0e0")
    fancybox: bool = Field(default=True)
    fontsize: int = Field(default=9, ge=1)


class GridConfig(BaseModel):
    """Configuration for grid properties."""

    show: bool = Field(default=True)
    alpha: float = Field(default=0.2, ge=0, le=1)
    color: str = Field(default="#666666")
    style: str = Field(default="solid")
    minor_alpha: float = Field(default=0.1, ge=0, le=1)
    minor_x_axis: bool = Field(default=True, description="Show minor gridlines for X-axis")
    minor_y_axis: bool = Field(default=True, description="Show minor gridlines for Y-axis")
    # Enhanced grid options
    major_linewidth: float = Field(default=0.8, gt=0)
    minor_linewidth: float = Field(default=0.4, gt=0)
    major_style: str = Field(default="solid")
    minor_style: str = Field(default="dashed")
    major_color: str = Field(default="#666666")
    minor_color: str = Field(default="#999999")


class AnnotationsConfig(BaseModel):
    """Configuration for annotation properties."""

    show_final_bw: bool = Field(default=True)
    fontsize: int = Field(default=9, ge=1)
    fontweight: str = Field(default="bold")
    zorder: int = Field(default=5)
    bbox_boxstyle: str = Field(default="round,pad=0.4")
    bbox_facecolor: str = Field(default="white")
    bbox_alpha: float = Field(default=0.92, ge=0, le=1)
    bbox_edgecolor: str = Field(default="#e0e0e0")
    bbox_linewidth: int = Field(default=1, ge=0)


class CodeBlockStyleConfig(BaseModel):
    """Configuration for code block styling."""

    background_color: str = Field(default="#f8f9fa")
    border_color: str = Field(default="#e9ecef")
    border_width: int = Field(default=1, ge=0)
    border_style: str = Field(default="solid")
    corner_radius: int = Field(default=4, ge=0)
    shadow_enabled: bool = Field(default=True)
    shadow_offset_x: int = Field(default=2)
    shadow_offset_y: int = Field(default=2)
    shadow_blur: int = Field(default=4, ge=0)
    shadow_color: str = Field(default="#000000")
    shadow_alpha: float = Field(default=0.1, ge=0, le=1)


class CodeTextStyleConfig(BaseModel):
    """Configuration for code text styling."""

    font_family: str = Field(default="Courier New")
    font_size: int = Field(default=9, ge=1)
    line_height: float = Field(default=1.2, gt=0)
    color: str = Field(default="#212529")
    selection_color: str = Field(default="#007bff")
    selection_background: str = Field(default="#e3f2fd")


class CommandStyleConfig(BaseModel):
    """Configuration for command line styling."""

    prompt_color: str = Field(default="#6c757d")
    command_color: str = Field(default="#212529")
    argument_color: str = Field(default="#495057")
    option_color: str = Field(default="#007bff")
    variable_color: str = Field(default="#28a745")
    comment_color: str = Field(default="#6c757d")


class ConfigStyleConfig(BaseModel):
    """Configuration for configuration display styling."""

    section_header_color: str = Field(default="#495057")
    section_header_background: str = Field(default="#e9ecef")
    key_color: str = Field(default="#007bff")
    value_color: str = Field(default="#212529")
    comment_color: str = Field(default="#6c757d")
    separator_color: str = Field(default="#dee2e6")


class CodeDisplayConfig(BaseModel):
    """Configuration for code display settings."""

    # Test command display settings
    show_test_commands: bool = Field(default=True, description="Whether to include test commands in the report")
    command_format: Literal["raw", "formatted", "highlighted"] = Field(default="formatted", description="Format for displaying test commands")
    command_max_lines: int = Field(default=10, ge=1, description="Maximum number of lines to show per command")
    command_truncate_length: int = Field(default=100, ge=1, description="Maximum characters per command line before truncation")

    # Configuration display settings
    show_configurations: bool = Field(default=True, description="Whether to include configuration details in the report")
    config_format: Literal["raw", "formatted", "highlighted"] = Field(default="formatted", description="Format for displaying configurations")
    config_max_lines: int = Field(default=20, ge=1, description="Maximum number of lines to show per configuration")
    config_truncate_length: int = Field(default=80, ge=1, description="Maximum characters per configuration line before truncation")

    # Code highlighting settings
    syntax_highlighting: bool = Field(default=True, description="Enable syntax highlighting for code blocks")
    highlight_theme: str = Field(default="default", description="Syntax highlighting theme to use")
    code_font_family: str = Field(default="monospace", description="Font family for code blocks")
    code_font_size: int = Field(default=9, ge=1, description="Font size for code blocks")

    # Layout settings
    code_block_width: float = Field(default=0.9, gt=0, le=1, description="Width of code blocks as fraction of page width")
    code_block_margin: float = Field(default=0.05, ge=0, le=1, description="Margin around code blocks as fraction of page width")
    code_block_padding: float = Field(default=0.02, ge=0, le=1, description="Internal padding for code blocks")

    # Filtering and organization
    include_environment_vars: bool = Field(default=True, description="Include environment variables in command display")
    include_arguments: bool = Field(default=True, description="Include command line arguments in display")
    group_commands_by_test: bool = Field(default=True, description="Group commands by test type")
    sort_commands_by_execution_order: bool = Field(default=True, description="Sort commands by their execution order")

    # Sub-configurations
    block_style: CodeBlockStyleConfig = Field(default_factory=CodeBlockStyleConfig)
    text_style: CodeTextStyleConfig = Field(default_factory=CodeTextStyleConfig)
    command_style: CommandStyleConfig = Field(default_factory=CommandStyleConfig)
    config_style: ConfigStyleConfig = Field(default_factory=ConfigStyleConfig)


class ColorSchemeConfig(BaseModel):
    """Configuration for color schemes."""

    baseline: str = Field(default="black")
    fallback: str = Field(default="black")
    comparison: str = Field(default="blue")
    palette: str = Field(default="tab20")


class FontAdjustmentsConfig(BaseModel):
    """Configuration for font size adjustments relative to base font size."""

    title_large: int = Field(default=5)
    title_medium: int = Field(default=3)
    label_large: int = Field(default=1)
    label_small: int = Field(default=-1)
    table_header: int = Field(default=-2)


class PlotLayoutConfig(BaseModel):
    """Configuration for plot layout and spacing."""

    subplot_left: float = Field(default=0.12, ge=0, le=1)
    subplot_right: float = Field(default=0.95, ge=0, le=1)
    subplot_top: float = Field(default=0.85, ge=0, le=1)
    subplot_bottom: float = Field(default=0.1, ge=0, le=1)
    axis_label_pad: int = Field(default=10, ge=0)
    title_pad: int = Field(default=20, ge=0)


class AnnotationPositioningConfig(BaseModel):
    """Configuration for annotation positioning."""

    text_offset_x: int = Field(default=7)
    text_offset_y: int = Field(default=7)
    shadow_offset_x: int = Field(default=1)
    shadow_offset_y: int = Field(default=-1)
    shadow_alpha: float = Field(default=0.2, ge=0, le=1)


class GridLinesConfig(BaseModel):
    """Configuration for grid line properties."""

    major_linewidth: float = Field(default=0.8, gt=0)
    minor_linewidth: float = Field(default=0.4, gt=0)
    note_page_linewidth: float = Field(default=0.3, gt=0)


class TicksConfig(BaseModel):
    """Configuration for tick properties."""

    minor_length: int = Field(default=3, ge=0)
    minor_width: float = Field(default=0.5, gt=0)
    major_length: int = Field(default=5, ge=0)
    major_width: float = Field(default=0.8, gt=0)
    # Enhanced tick options
    minor_color: str = Field(default="#999999")
    major_color: str = Field(default="#333333")
    tick_direction: str = Field(default="out")
    minor_alpha: float = Field(default=0.7, ge=0, le=1)
    major_alpha: float = Field(default=1.0, ge=0, le=1)


class LegendPositioningConfig(BaseModel):
    """Configuration for legend positioning and spacing."""

    anchor_x: float = Field(default=0.5, ge=0, le=1)
    anchor_y_offset: float = Field(default=-0.15)
    height_per_row: float = Field(default=0.03, gt=0)
    bottom_margin_base: float = Field(default=0.2, ge=0)
    max_legend_height: float = Field(default=0.25, gt=0, le=1)
    adaptive_columns: bool = Field(default=False)
    adaptive_font_size: bool = Field(default=True)


class BackgroundEffectsConfig(BaseModel):
    """Configuration for background effects."""

    gradient_alpha: float = Field(default=0.05, ge=0, le=1)
    intro_text_alpha: float = Field(default=0.8, ge=0, le=1)
    intro_text_padding: int = Field(default=1, ge=0)


class PageLayoutConfig(BaseModel):
    """Configuration for page layout."""

    command_start_y: float = Field(default=0.85, ge=0, le=1)
    command_step_y: float = Field(default=0.055, gt=0)
    command_line_spacing: float = Field(default=0.027, gt=0)
    note_grid_start: int = Field(default=5, ge=0)
    note_grid_end: int = Field(default=95, ge=0)
    note_grid_step: int = Field(default=5, gt=0)


class TextWrappingConfig(BaseModel):
    """Configuration for text wrapping limits."""

    command_max_length: int = Field(default=130, gt=0)
    config_name_max_length: int = Field(default=10, gt=0)
    command_break_threshold: int = Field(default=20, gt=0)
    title_max_length: int = Field(default=100, gt=0)
    title_max_lines: int = Field(default=5, gt=0)
    # Table header wrapping configuration
    table_header_min_wrap_length: int = Field(default=12, gt=0)
    table_header_max_wrap_length: int = Field(default=35, gt=0)
    table_header_wrap_threshold: int = Field(default=20, gt=0)


class TitleStyleConfig(BaseModel):
    """Configuration for consistent title formatting across all titles."""

    font_weight: str = Field(default="normal", description="Font weight: normal, bold, light")
    font_size_adjustment: int = Field(default=3, description="Font size adjustment relative to base font_size")
    max_length: int = Field(default=100, gt=0, description="Maximum characters per line before wrapping")
    max_lines: int = Field(default=5, gt=0, description="Maximum number of lines for wrapped titles")
    wrap_enabled: bool = Field(default=True, description="Whether to enable title wrapping")
    alignment: str = Field(default="center", description="Text alignment: left, center, right")
    vertical_alignment: str = Field(default="top", description="Vertical alignment: top, center, bottom")
    color: str = Field(default="#333333", description="Title color")
    line_spacing: float = Field(default=1.2, gt=0, description="Line spacing for multi-line titles")

    @field_validator('font_weight')
    @classmethod
    def validate_font_weight(cls, v: str) -> str:
        """Validate font weight values."""
        valid_weights = ['normal', 'bold', 'light', 'ultralight', 'heavy', 'ultrabold']
        if v not in valid_weights:
            raise ValueError(f"Font weight must be one of: {valid_weights}")
        return v

    @field_validator('alignment')
    @classmethod
    def validate_alignment(cls, v: str) -> str:
        """Validate text alignment values."""
        valid_alignments = ['left', 'center', 'right']
        if v not in valid_alignments:
            raise ValueError(f"Alignment must be one of: {valid_alignments}")
        return v

    @field_validator('vertical_alignment')
    @classmethod
    def validate_vertical_alignment(cls, v: str) -> str:
        """Validate vertical alignment values."""
        valid_alignments = ['top', 'center', 'bottom']
        if v not in valid_alignments:
            raise ValueError(f"Vertical alignment must be one of: {valid_alignments}")
        return v


class TableLayoutConfig(BaseModel):
    """Configuration for table layout."""

    bbox_left: float = Field(default=0.05, ge=0, le=1)
    bbox_bottom: float = Field(default=0.1, ge=0, le=1)
    bbox_width: float = Field(default=0.9, gt=0, le=1)
    bbox_height: float = Field(default=0.75, gt=0, le=1)
    scale_width: float = Field(default=1, gt=0)
    scale_height: float = Field(default=10.0, gt=0)


class TextPositioningConfig(BaseModel):
    """Configuration for text positioning as fractions of axes."""

    title_x: float = Field(default=0.5, ge=0, le=1)
    title_y: float = Field(default=0.95, ge=0, le=1)
    intro_text_x: float = Field(default=0.05, ge=0, le=1)
    intro_text_y: float = Field(default=0.95, ge=0, le=1)
    note_bottom_x: float = Field(default=0.5, ge=0, le=1)
    note_bottom_y: float = Field(default=0.05, ge=0, le=1)


class ColorPaletteConfig(BaseModel):
    """Configuration for color palette."""

    num_colors: int = Field(default=20, gt=0)


class PDFSettingsConfig(BaseModel):
    """Configuration for PDF page size and layout."""

    use_landscape: bool = Field(default=True)
    page_width: float = Field(default=11.0, gt=0)
    page_height: float = Field(default=14.0, gt=0)
    margin_left: float = Field(default=0.5, ge=0)
    margin_right: float = Field(default=0.5, ge=0)
    margin_top: float = Field(default=1.0, ge=0)
    margin_bottom: float = Field(default=1.0, ge=0)
    pdf_subplot_left: float = Field(default=0.08, ge=0, le=1)
    pdf_subplot_right: float = Field(default=0.95, ge=0, le=1)
    pdf_subplot_top: float = Field(default=0.85, ge=0, le=1)
    pdf_subplot_bottom: float = Field(default=0.08, ge=0, le=1)
    pdf_dpi: int = Field(default=300, gt=0)
    pdf_facecolor: str = Field(default="white")
    pdf_edgecolor: str = Field(default="none")


class StyleConfig(BaseModel):
    """Main style configuration model containing all sub-configurations."""

    figure_sizes: FigureSizesConfig = Field(default_factory=FigureSizesConfig)
    chart_style: ChartStyleConfig = Field(default_factory=ChartStyleConfig)
    legend: LegendConfig = Field(default_factory=LegendConfig)
    grid: GridConfig = Field(default_factory=GridConfig)
    annotations: AnnotationsConfig = Field(default_factory=AnnotationsConfig)

    split_mask_y_limits: Dict[str, float] = Field(default_factory=lambda: {"0x0": 400, "0x7": 400})
    operation_y_limits: Dict[str, float] = Field(default_factory=lambda: {"alltoall_sendrecv": 400})
    color_scheme: ColorSchemeConfig = Field(default_factory=ColorSchemeConfig)
    font_adjustments: FontAdjustmentsConfig = Field(default_factory=FontAdjustmentsConfig)
    plot_layout: PlotLayoutConfig = Field(default_factory=PlotLayoutConfig)
    annotation_positioning: AnnotationPositioningConfig = Field(default_factory=AnnotationPositioningConfig)
    grid_lines: GridLinesConfig = Field(default_factory=GridLinesConfig)
    ticks: TicksConfig = Field(default_factory=TicksConfig)
    legend_positioning: LegendPositioningConfig = Field(default_factory=LegendPositioningConfig)
    background_effects: BackgroundEffectsConfig = Field(default_factory=BackgroundEffectsConfig)
    page_layout: PageLayoutConfig = Field(default_factory=PageLayoutConfig)
    text_wrapping: TextWrappingConfig = Field(default_factory=TextWrappingConfig)
    title_style: TitleStyleConfig = Field(default_factory=TitleStyleConfig)
    table_layout: TableLayoutConfig = Field(default_factory=TableLayoutConfig)
    text_positioning: TextPositioningConfig = Field(default_factory=TextPositioningConfig)
    color_palette: ColorPaletteConfig = Field(default_factory=ColorPaletteConfig)
    pdf_settings: PDFSettingsConfig = Field(default_factory=PDFSettingsConfig)
    code_display: CodeDisplayConfig = Field(default_factory=CodeDisplayConfig)

    @classmethod
    def from_toml(cls, toml_dict: Dict) -> 'StyleConfig':
        """
        Create a StyleConfig instance from a TOML dictionary.

        Args:
            toml_dict: Dictionary loaded from TOML file

        Returns:
            StyleConfig instance with validated values
        """
        try:
            return cls(**toml_dict)
        except Exception as e:
            logger.error(f"Error parsing style configuration: {e}")
            raise

    def get_figure_size(self, size_type: str = 'default') -> List[float]:
        """
        Get figure size by type.

        Args:
            size_type: Type of figure size ('default', 'dual_plot', 'subplot')

        Returns:
            List of [width, height] in inches
        """
        size_map = {
            'default': self.figure_sizes.default,
            'dual_plot': self.figure_sizes.dual_plot,
            'subplot': self.figure_sizes.subplot
        }
        return size_map.get(size_type, self.figure_sizes.default)

    def get_font_size(self, adjustment_type: Optional[str] = None) -> int:
        """
        Get font size with optional adjustment.

        Args:
            adjustment_type: Type of font adjustment from font_adjustments

        Returns:
            Adjusted font size
        """
        base_size = self.chart_style.font_size
        if adjustment_type:
            adjustment = getattr(self.font_adjustments, adjustment_type, 0)
            return base_size + adjustment
        return base_size

    def get_y_limit(self, operation_type: str, split_mask: Optional[str] = None) -> float:
        """
        Get Y-axis limit for a given operation type and optional split mask.

        Args:
            operation_type: Type of operation (e.g., 'allreduce', 'alltoall')
            split_mask: Optional split mask value (e.g., '0x0', '0x7')

        Returns:
            Y-axis limit value
        """
        # Check split mask limits first
        if split_mask and split_mask in self.split_mask_y_limits:
            return self.split_mask_y_limits[split_mask]

        # Check operation-specific limits
        if operation_type in self.operation_y_limits:
            return self.operation_y_limits[operation_type]

        # Default based on operation category (hardcoded since axis config moved to plot_config.toml)
        if operation_type in ['sendrecv', 'send', 'recv']:
            return 50.0  # Default for P2P operations
        else:
            return 400.0  # Default for collective operations

    def apply_title_style(self, title: str, style_manager=None) -> str:
        """
        Apply consistent title styling and wrapping based on title_style configuration.

        Args:
            title: Original title text
            style_manager: Optional StyleManager instance to use for wrapping

        Returns:
            Styled and wrapped title text
        """
        if not self.title_style.wrap_enabled:
            return title

        # Apply wrapping if title exceeds max length
        if len(title) > self.title_style.max_length:
            if style_manager is not None:
                # Use provided style manager
                title = style_manager.wrap_text(title, self.title_style.max_length)
            else:
                # Fallback: create new instance (for backward compatibility)
                from ..plotting.style_manager import StyleManager
                temp_style_manager = StyleManager(self)
                title = temp_style_manager.wrap_text(title, self.title_style.max_length)

        return title

    def get_title_font_size(self) -> int:
        """
        Get the font size for titles based on title_style configuration.

        Returns:
            Font size for titles
        """
        return self.chart_style.font_size + self.title_style.font_size_adjustment


@dataclass
class StyleConfigLoader:
    """Utility class for loading and validating style configuration."""

    @staticmethod
    def load_from_toml(toml_path: str) -> StyleConfig:
        """
        Load and validate style configuration from TOML file.

        Args:
            toml_path: Path to the TOML configuration file

        Returns:
            Validated StyleConfig instance

        Raises:
            ValidationError: If configuration is invalid
            FileNotFoundError: If TOML file not found
            TOMLDecodeError: If TOML syntax is invalid
        """
        import tomli

        try:
            with open(toml_path, 'rb') as f:
                toml_data = tomli.load(f)

            logger.debug(f"Successfully loaded TOML from {toml_path}")

            # Validate and create StyleConfig instance
            config = StyleConfig.from_toml(toml_data)

            logger.info(f"Successfully validated style configuration from {toml_path}")
            return config

        except FileNotFoundError:
            logger.error(f"Configuration file not found: {toml_path}")
            raise
        except tomli.TOMLDecodeError as e:
            logger.error(f"Invalid TOML syntax in {toml_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading style configuration: {e}")
            raise
