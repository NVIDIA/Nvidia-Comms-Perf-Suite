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
Pydantic models for NCCL Performance Analysis Chart Configuration validation.

This module defines the data models for validating the plot_config.toml file
used in NCCL performance analysis and reporting.
"""

from typing import Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from dataclasses import dataclass


class XAxisConfig(BaseModel):
    """X-axis configuration."""

    scale: Optional[str] = Field(
        "semilogx",
        description="X-axis scale type (e.g., 'semilogx', 'linear', 'log')"
    )
    log_scale: Optional[bool] = Field(
        True,
        description="Use logarithmic scale for X-axis (default: True)"
    )
    ticks: Optional[List[float]] = Field(
        default=None,
        description="X-axis tick positions"
    )
    labels: Optional[List[str]] = Field(
        default=None,
        description="X-axis tick labels"
    )


class YAxisLimitsTestBinary(BaseModel):
    """Y-axis limits for specific test binaries."""

    min: Optional[float] = Field(None, description="Minimum Y-axis value")
    max: Optional[float] = Field(None, description="Maximum Y-axis value")


class YAxisLimits(BaseModel):
    """Y-axis limits configuration."""

    # Y-axis settings
    auto_detect_margin: Optional[float] = Field(
        0.1,
        description="Margin to add above maximum value (default: 0.1 = 10%)"
    )
    auto_detect_round_to: Optional[int] = Field(
        50,
        description="Round up to nearest multiple (default: 50)"
    )
    default_min: Optional[float] = Field(0, description="Default minimum value")
    default_max: Optional[float] = Field(None, description="Default maximum value override")
    p2p_min: Optional[float] = Field(0, description="P2P operation minimum value")
    p2p_max: Optional[float] = Field(None, description="P2P operation maximum value override")
    test_binaries: Optional[Dict[str, YAxisLimitsTestBinary]] = Field(
        default_factory=dict,
        description="Test binary specific limits"
    )


class YAxisConfig(BaseModel):
    """Y-axis configuration."""

    limits: Optional[YAxisLimits] = Field(
        default=None,
        description="Y-axis limits configuration"
    )


class ChartOrderingSortOrder(BaseModel):
    """Sort order configuration for chart ordering."""

    num_nodes: Optional[bool] = Field(True, description="Sort by number of nodes")
    num_processes: Optional[bool] = Field(True, description="Sort by number of processes")
    local_register: Optional[bool] = Field(True, description="Sort by local register setting")
    NCCL_TESTS_SPLIT_MASK: Optional[bool] = Field(True, description="Sort by NCCL_TESTS_SPLIT_MASK")


class ParameterMapping(BaseModel):
    """Parameter mapping configuration."""

    column: str = Field(..., description="Column name to map to")
    type: Literal["numeric", "string", "hex"] = Field(..., description="Data type of the parameter")
    aliases: Optional[List[str]] = Field(default_factory=list, description="Alternative column names")
    extract_from_config_id: Optional[bool] = Field(False, description="Extract value from config ID")


class ChartOrdering(BaseModel):
    """Chart ordering configuration."""

    enable_sorting: Optional[bool] = Field(True, description="Enable chart sorting")
    sort_order: Optional[ChartOrderingSortOrder] = Field(
        default=None,
        description="Sort order configuration"
    )
    parameter_mappings: Optional[Dict[str, ParameterMapping]] = Field(
        default_factory=dict,
        description="Parameter mapping configuration"
    )


class PerformanceColumn(BaseModel):
    """Performance column configuration."""

    sources: List[str] = Field(..., description="Source column names")
    operation: Literal["max", "min", "sum", "mean", "median", "std", "var", "none"] = Field(..., description="Operation to apply")
    description: str = Field(..., description="Column description")
    default: Optional[bool] = Field(False, description="Mark as default performance column")


class DataProcessing(BaseModel):
    """Data processing configuration."""

    default_performance_column: Optional[str] = Field(
        "max_bus_bw",
        description="Default performance column to use"
    )
    size_column: Optional[str] = Field(
        "test_result.size",
        description="Column containing size data"
    )
    performance_columns: Optional[Dict[str, PerformanceColumn]] = Field(
        default_factory=dict,
        description="Performance column definitions"
    )
    dtypes: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Column data types for optimization"
    )


class StyleMappings(BaseModel):
    """Style mappings configuration."""

    style_column: Optional[str] = Field(
        "local_register",
        description="Column that determines line and marker styles"
    )
    line_styles: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Line style mappings"
    )
    marker_styles: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Marker style mappings"
    )
    default_colors: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Color mappings for special configurations"
    )


class CodeDisplayConfig(BaseModel):
    """Code display configuration for test commands and configurations."""

    # Test command display settings
    show_test_commands: Optional[bool] = Field(
        True,
        description="Whether to include test commands in the report"
    )
    command_format: Optional[Literal["raw", "formatted", "highlighted"]] = Field(
        "formatted",
        description="Format for displaying test commands"
    )
    command_max_lines: Optional[int] = Field(
        10,
        description="Maximum number of lines to show per command"
    )
    command_truncate_length: Optional[int] = Field(
        100,
        description="Maximum characters per command line before truncation"
    )

    # Configuration display settings
    show_configurations: Optional[bool] = Field(
        True,
        description="Whether to include configuration details in the report"
    )
    config_format: Optional[Literal["raw", "formatted", "highlighted"]] = Field(
        "formatted",
        description="Format for displaying configurations"
    )
    config_max_lines: Optional[int] = Field(
        20,
        description="Maximum number of lines to show per configuration"
    )
    config_truncate_length: Optional[int] = Field(
        80,
        description="Maximum characters per configuration line before truncation"
    )

    # Code highlighting settings
    syntax_highlighting: Optional[bool] = Field(
        True,
        description="Enable syntax highlighting for code blocks"
    )
    highlight_theme: Optional[str] = Field(
        "default",
        description="Syntax highlighting theme to use"
    )
    code_font_family: Optional[str] = Field(
        "monospace",
        description="Font family for code blocks"
    )
    code_font_size: Optional[int] = Field(
        9,
        description="Font size for code blocks"
    )

    # Layout settings
    code_block_width: Optional[float] = Field(
        0.9,
        description="Width of code blocks as fraction of page width"
    )
    code_block_margin: Optional[float] = Field(
        0.05,
        description="Margin around code blocks as fraction of page width"
    )
    code_block_padding: Optional[float] = Field(
        0.02,
        description="Internal padding for code blocks"
    )

    # Filtering and organization
    include_environment_vars: Optional[bool] = Field(
        True,
        description="Include environment variables in command display"
    )
    include_arguments: Optional[bool] = Field(
        True,
        description="Include command line arguments in display"
    )
    group_commands_by_test: Optional[bool] = Field(
        True,
        description="Group commands by test type"
    )
    sort_commands_by_execution_order: Optional[bool] = Field(
        True,
        description="Sort commands by their execution order"
    )


class Settings(BaseModel):
    """Main settings configuration."""

    x_axis: Optional[XAxisConfig] = Field(
        default=None,
        description="X-axis configuration"
    )
    y_axis: Optional[YAxisConfig] = Field(
        default=None,
        description="Y-axis configuration"
    )
    chart_ordering: Optional[ChartOrdering] = Field(
        default=None,
        description="Chart ordering configuration"
    )
    data_processing: Optional[DataProcessing] = Field(
        default=None,
        description="Data processing configuration"
    )
    style_mappings: Optional[StyleMappings] = Field(
        default=None,
        description="Style mappings configuration"
    )
    code_display: Optional[CodeDisplayConfig] = Field(
        default=None,
        description="Code display configuration for test commands and configurations"
    )


class ChartType(BaseModel):
    """Chart type configuration."""

    id: str = Field(..., description="Unique chart type identifier")
    name: str = Field(..., description="Chart type name")
    description: str = Field(..., description="Chart type description")
    performance_columns: Optional[List[str]] = Field(
        None,
        description="Performance columns to plot"
    )
    fixed_params: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Fixed parameter values"
    )
    default_values: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Default parameter values"
    )
    chart_split_params: Optional[Dict[str, Union[str, List[str]]]] = Field(
        default_factory=dict,
        description="Parameters to split charts by"
    )
    chart_combine_params: Optional[Dict[str, Union[str, List[str]]]] = Field(
        default_factory=dict,
        description="Parameters to combine in charts"
    )

    @model_validator(mode='after')
    def validate_parameter_conflicts(self):
        """Validate that parameters are not repeated between fixed_params, default_values, chart_split_params, and chart_combine_params."""
        fixed_params = set(self.fixed_params.keys()) if self.fixed_params else set()
        default_values = set(self.default_values.keys()) if self.default_values else set()
        split_params = set(self.chart_split_params.keys()) if self.chart_split_params else set()
        combine_params = set(self.chart_combine_params.keys()) if self.chart_combine_params else set()

        # Check for conflicts between fixed_params and split_params
        fixed_split_conflict = fixed_params & split_params
        if fixed_split_conflict:
            raise ValueError(
                f"Chart type '{self.name}' has parameter conflicts: "
                f"parameters {fixed_split_conflict} appear in both fixed_params and chart_split_params. "
                f"A parameter cannot be both fixed and used for splitting charts."
            )

        # Check for conflicts between fixed_params and combine_params
        fixed_combine_conflict = fixed_params & combine_params
        if fixed_combine_conflict:
            raise ValueError(
                f"Chart type '{self.name}' has parameter conflicts: "
                f"parameters {fixed_combine_conflict} appear in both fixed_params and chart_combine_params. "
                f"A parameter cannot be both fixed and used for combining in charts."
            )

        # Check for conflicts between fixed_params and default_values
        fixed_default_conflict = fixed_params & default_values
        if fixed_default_conflict:
            raise ValueError(
                f"Chart type '{self.name}' has parameter conflicts: "
                f"parameters {fixed_default_conflict} appear in both fixed_params and default_values. "
                f"A parameter cannot be both fixed and have a default value."
            )

        # Check for conflicts between split_params and combine_params
        split_combine_conflict = split_params & combine_params
        if split_combine_conflict:
            raise ValueError(
                f"Chart type '{self.name}' has parameter conflicts: "
                f"parameters {split_combine_conflict} appear in both chart_split_params and chart_combine_params. "
                f"A parameter cannot be used for both splitting charts and combining in charts."
            )

        # Check for conflicts between split_params and default_values
        split_default_conflict = split_params & default_values
        if split_default_conflict:
            raise ValueError(
                f"Chart type '{self.name}' has parameter conflicts: "
                f"parameters {split_default_conflict} appear in both chart_split_params and default_values. "
                f"A parameter cannot be used for splitting charts and have a default value."
            )

        # Check for conflicts between combine_params and default_values
        combine_default_conflict = combine_params & default_values
        if combine_default_conflict:
            raise ValueError(
                f"Chart type '{self.name}' has parameter conflicts: "
                f"parameters {combine_default_conflict} appear in both chart_combine_params and default_values. "
                f"A parameter cannot be used for combining in charts and have a default value."
            )

        return self


class AnalysisProfile(BaseModel):
    """Analysis profile configuration."""

    id: str = Field(..., description="Unique analysis profile identifier")
    name: str = Field(..., description="Profile name")
    description: str = Field(..., description="Profile description")
    test_binaries: List[str] = Field(..., description="Test binaries to include")
    chart_types: List[str] = Field(..., description="Chart types to generate")


class PDFConfig(BaseModel):
    """PDF report configuration."""

    id: str = Field(..., description="Unique PDF config identifier")
    name: str = Field(..., description="Report name")
    description: str = Field(..., description="Report description")
    analysis_profiles: List[str] = Field(..., description="Analysis profiles to include")


class ChartConfig(BaseModel):
    """Root configuration model for plot_config.toml."""

    settings: Optional[Settings] = Field(
        default=None,
        description="Global settings"
    )
    chart: Optional[List[ChartType]] = Field(
        default_factory=list,
        description="Chart type definitions"
    )
    analysis_profiles: Optional[List[AnalysisProfile]] = Field(
        default_factory=list,
        description="Analysis profile definitions"
    )
    pdf_configs: Optional[List[PDFConfig]] = Field(
        default_factory=list,
        description="PDF report configurations"
    )

    @field_validator('chart')
    @classmethod
    def validate_chart_types(cls, v: List[ChartType]) -> List[ChartType]:
        """Validate that chart types have unique names and IDs."""
        if not v:
            return v

        names = [chart.name for chart in v]
        ids = [chart.id for chart in v]

        if len(names) != len(set(names)):
            raise ValueError("Chart type names must be unique")

        if len(ids) != len(set(ids)):
            raise ValueError("Chart type IDs must be unique")

        return v

    @model_validator(mode='after')
    def validate_references(self):
        """Validate that analysis profiles and PDF configs reference existing items."""
        # Create lookup dictionaries for easier validation
        chart_lookup = {chart.id: chart for chart in (self.chart or [])}
        profile_lookup = {profile.id: profile for profile in (self.analysis_profiles or [])}

        # Validate analysis profiles reference existing chart types
        if self.analysis_profiles and self.chart:
            available_chart_types = set(chart_lookup.keys())
            for profile in self.analysis_profiles:
                invalid_types = set(profile.chart_types) - available_chart_types
                if invalid_types:
                    raise ValueError(
                        f"Analysis profile '{profile.id}' references invalid chart types: {invalid_types}"
                    )

        # Validate PDF configs reference existing analysis profiles
        if self.pdf_configs and self.analysis_profiles:
            available_profiles = set(profile_lookup.keys())
            for pdf_config in self.pdf_configs:
                invalid_profiles = set(pdf_config.analysis_profiles) - available_profiles
                if invalid_profiles:
                    raise ValueError(
                        f"PDF config '{pdf_config.id}' references invalid analysis profiles: {invalid_profiles}"
                    )

        return self

    def get_chart_by_id(self, chart_id: str) -> Optional[ChartType]:
        """Get a chart type by its ID."""
        if not self.chart:
            return None
        for chart in self.chart:
            if chart.id == chart_id:
                return chart
        return None

    def get_analysis_profile_by_id(self, profile_id: str) -> Optional[AnalysisProfile]:
        """Get an analysis profile by its ID."""
        if not self.analysis_profiles:
            return None
        for profile in self.analysis_profiles:
            if profile.id == profile_id:
                return profile
        return None

    def get_pdf_config_by_id(self, config_id: str) -> Optional[PDFConfig]:
        """Get a PDF config by its ID."""
        if not self.pdf_configs:
            return None
        for config in self.pdf_configs:
            if config.id == config_id:
                return config
        return None


@dataclass
class ChartConfigLoader:
    """Utility class for loading and validating chart configuration."""

    @staticmethod
    def load_from_toml(toml_path: str) -> ChartConfig:
        """
        Load and validate chart configuration from TOML file.

        Args:
            toml_path: Path to the TOML configuration file

        Returns:
            Validated ChartConfig instance

        Raises:
            ValidationError: If configuration is invalid
            FileNotFoundError: If TOML file not found
            TOMLDecodeError: If TOML syntax is invalid
        """
        import tomli
        import logging

        logger = logging.getLogger(__name__)

        try:
            with open(toml_path, 'rb') as f:
                toml_data = tomli.load(f)

            logger.debug(f"Successfully loaded TOML from {toml_path}")

            # Validate and create ChartConfig instance
            config = ChartConfig(**toml_data)

            logger.info(f"Successfully validated chart configuration from {toml_path}")
            return config

        except FileNotFoundError:
            logger.error(f"Configuration file not found: {toml_path}")
            raise
        except tomli.TOMLDecodeError as e:
            logger.error(f"Invalid TOML syntax in {toml_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading chart configuration: {e}")
            raise
