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
Chart generator for creating matplotlib charts from NCCL performance data.

This module handles the generation of individual performance charts based on
the configuration and data provided by the NCCL adapter.
"""

import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib.transforms import Bbox
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from .style_manager import StyleManager

from ..adapters.nccl_adapter import NCCLPerformanceAdapter

logger = logging.getLogger(__name__)


class ChartGenerator:
    """Generates matplotlib charts from NCCL performance data."""

    def __init__(self, adapter: NCCLPerformanceAdapter, style_manager: StyleManager):
        """Initialize chart generator.

        Args:
            adapter: NCCL performance adapter with processed data
            style_manager: Style manager for consistent styling
        """
        self.adapter = adapter
        self.style_manager = style_manager
        self.plot_config = adapter.plot_config

        # Update style manager with plot config for Y-axis limits
        if hasattr(self.style_manager, 'plot_config'):
            self.style_manager.plot_config = self.plot_config

    def cleanup(self) -> None:
        """Clean up resources and close all figures to free memory."""
        try:
            self.style_manager.close_all_figures()
            logger.info("ChartGenerator cleanup completed")
        except Exception as e:
            logger.warning(f"Error during ChartGenerator cleanup: {e}")

    def generate_chart(self, chart_type: str, test_binary: str,
                      chart_group: Dict[str, Any], chart_number: Optional[int] = None) -> Optional[matplotlib.figure.Figure]:
        """Generate a single chart for a chart type, test binary, and group.

        Args:
            chart_type: Type of chart to generate
            test_binary: Test binary name
            chart_group: Chart group information from adapter
            chart_number: Optional unique chart number for correlation with data tables

        Returns:
            Matplotlib figure containing the chart, or None if no data is available
        """
        logger.info(f"Generating chart: {chart_type} for {test_binary}")

        # Get chart configuration
        if not self.plot_config or not self.plot_config.chart:
            raise ValueError("Plot configuration not available")
        chart_config = self.plot_config.get_chart_by_id(chart_type)
        if not chart_config:
            raise ValueError(f"Chart type '{chart_type}' not found in configuration")

        # Create figure and axes
        fig, ax = self.style_manager.create_figure()

        # Filter data for this chart group
        filtered_data = self._filter_data_for_chart_group(chart_group)
        if filtered_data.empty:
            logger.warning(f"No data found for chart group: {chart_group['group_key']}")
            return None

        # Get size column
        size_col = self.adapter.get_size_column()
        if size_col not in filtered_data.columns:
            logger.error(f"Size column '{size_col}' not found in data")
            return None

        # Get performance columns
        performance_columns = chart_config.performance_columns
        if not performance_columns:
            performance_columns = [self.adapter.get_performance_columns()[0]] if self.adapter.get_performance_columns() else []

        if not performance_columns:
            logger.error("No performance columns available for plotting")
            return None

        # Group data by combine parameters
        legend_groups = self._group_data_for_legends(filtered_data, chart_config)



        # Plot each legend group
        legend_labels = []
        for legend_key, group_data in legend_groups.items():
            if group_data.empty:
                continue

            # Get style information for this legend
            style_info = self.adapter.get_legend_style_info(legend_key, chart_type)

            # Plot only the first performance column to avoid duplicate lines
            # If multiple performance columns are specified, only use the first one
            perf_col = performance_columns[0] if performance_columns else None
            if not perf_col or perf_col not in group_data.columns:
                continue

            # Sort by size for proper line plotting
            plot_data = group_data.sort_values(size_col)

            # Create the plot
            ax.plot(
                plot_data[size_col],
                plot_data[perf_col],
                color=style_info['color'],
                linestyle=style_info['line_style'],
                marker=style_info['marker_style'],
                markersize=self.style_manager.style_config.chart_style.marker_size,
                markeredgewidth=self.style_manager.style_config.chart_style.markeredgewidth,
                markeredgecolor=self.style_manager.style_config.chart_style.markeredgecolor,
                linewidth=self.style_manager.style_config.chart_style.line_width,
                label=legend_key
            )

            # Add annotations if enabled
            if plot_data[perf_col].notna().any():  # type: ignore
                self.style_manager.add_annotations(
                    ax,
                    np.array(plot_data[size_col].values),
                    np.array(plot_data[perf_col].values),
                    legend_key
                )

            legend_labels.append(legend_key)

        # Apply styling to axes
        # Collect all performance data for Y-axis limit calculation
        all_performance_data = []
        for legend_key, group_data in legend_groups.items():
            if not group_data.empty:  # type: ignore
                for perf_col in performance_columns:
                    if perf_col in group_data.columns:
                        valid_data = group_data[perf_col].dropna()
                        if not valid_data.empty:
                            all_performance_data.extend(valid_data.values)

        # Convert to numpy array for Y-axis limit calculation
        performance_array = np.array(all_performance_data) if all_performance_data else None

        self.style_manager.style_axes(ax, test_binary, performance_array)

        # Add title with fixed and split parameters, including chart number if provided
        title = self._generate_chart_title(chart_group, chart_config, test_binary, chart_number)

        # Get title styling properties
        title_style = self.style_manager.style_config.title_style
        title_font_size = self.style_manager.style_config.get_title_font_size()

        ax.set_title(title, fontsize=title_font_size,
                    fontweight=title_style.font_weight,
                    color=title_style.color,
                    pad=self.style_manager.style_config.plot_layout.title_pad)

        # Add legend
        if legend_labels:
            self.style_manager.style_legend(ax, legend_labels)

        # The style manager handles subplot adjustments for legend positioning
        # No additional subplot adjustments needed here

        return fig

    def _filter_data_for_chart_group(self, chart_group: Dict[str, Any]) -> pd.DataFrame:
        """Filter processed data for a specific chart group.

        Args:
            chart_group: Chart group information containing 'data' key

        Returns:
            Filtered DataFrame
        """
        # The chart_group already contains the filtered data
        if 'data' in chart_group:
            return chart_group['data']

        # Fallback to old method if needed
        if self.adapter._processed_data is None:
            return pd.DataFrame()

        filtered_data = self.adapter._processed_data.copy()

        # Apply filter conditions if using old format
        filter_conditions = chart_group.get('filter_conditions', {})
        for column, value in filter_conditions.items():
            if column in filtered_data.columns:
                filtered_data = filtered_data[filtered_data[column] == value]

        return filtered_data  # type: ignore

    def _group_data_for_legends(self, data: pd.DataFrame,
                               chart_config: Any) -> Dict[str, pd.DataFrame]:
        """Group data by combine parameters to create legend groups.

        Args:
            data: Filtered data for the chart
            chart_config: Chart configuration

        Returns:
            Dictionary mapping legend keys to DataFrames
        """
        combine_params = chart_config.chart_combine_params or {}
        fixed_params = chart_config.fixed_params or {}

        if not combine_params:
            # If no combine parameters, return single group with deduplication
            size_col = self.adapter.get_size_column()
            if size_col in data.columns and not data.empty:
                # Group by size and take the first occurrence of each size
                data = data.groupby(size_col, as_index=False).first()
            return {"All Data": data}

        # Get actual column names for combine parameters
        combine_columns = []
        for param, values in combine_params.items():
            try:
                actual_column = self.adapter.get_dataframe_column_name(param, data)
                if actual_column in data.columns:
                    combine_columns.append((param, actual_column))
                else:
                    logger.warning(f"Combine parameter column '{actual_column}' not found for parameter '{param}', skipping")
            except ValueError as e:
                logger.warning(f"Combine parameter '{param}' not found in data: {e}")

        if not combine_columns:
            # No valid combine columns, return single group with deduplication
            size_col = self.adapter.get_size_column()
            if size_col in data.columns and not data.empty:
                # Group by size and take the first occurrence of each size
                data = data.groupby(size_col, as_index=False).first()
            return {"All Data": data}

        # Group by combine parameters
        legend_groups = {}

        # Check if any combine parameters are also fixed parameters
        # If so, we should only group by the fixed values, not all unique values
        fixed_combine_params = {}
        for param, values in combine_params.items():
            if param in fixed_params:
                fixed_combine_params[param] = fixed_params[param]

        # Check for default values that should be treated as fixed
        # This handles cases where a parameter has a default value but is still in combine_params
        if hasattr(chart_config, 'default_values') and chart_config.default_values:
            for param, default_value in chart_config.default_values.items():
                if param in combine_params and param not in fixed_params:
                    # If this parameter has a default value and is in combine_params but not fixed_params,
                    # treat it as fixed for grouping purposes
                    fixed_combine_params[param] = default_value

        # Filter out combine parameters that are fixed from the combine_columns list
        # This prevents creating unnecessary legend groups for fixed parameters
        filtered_combine_columns = []
        for param_name, column_name in combine_columns:
            if param_name not in fixed_combine_params:
                filtered_combine_columns.append((param_name, column_name))

        # Use filtered combine columns instead of original
        combine_columns = filtered_combine_columns

        # If ALL combine parameters are fixed, return single group
        if not combine_columns:
            legend_parts = []
            for param_name, fixed_value in fixed_combine_params.items():
                legend_parts.append(f"{param_name}={fixed_value}")

            legend_key = ", ".join(legend_parts)
            shortened_key = self.adapter.shorten_legend_key(legend_key)
            legend_groups[shortened_key] = data
            return legend_groups

        # Get unique combinations of combine parameter values
        if len(combine_columns) == 1:
            param_name, column_name = combine_columns[0]

            # If this parameter is fixed or has a default value, only use that value
            if param_name in fixed_combine_params:
                fixed_value = fixed_combine_params[param_name]
                # Make value matching case agnostic
                group_data = data[data[column_name].astype(str).str.lower() == str(fixed_value).lower()]
                legend_key = f"{param_name}={fixed_value}"
                # Shorten the legend key
                shortened_key = self.adapter.shorten_legend_key(legend_key)
                legend_groups[shortened_key] = group_data
            else:
                # Use all unique values as before
                for value in data[column_name].unique():
                    group_data = data[data[column_name] == value]
                    legend_key = f"{param_name}={value}"
                    # Shorten the legend key
                    shortened_key = self.adapter.shorten_legend_key(legend_key)
                    legend_groups[shortened_key] = group_data
        else:
            # Multiple combine parameters

            # If any parameters are fixed, filter the data to only include those combinations
            if fixed_combine_params:
                # Create filter condition for fixed parameters
                condition = pd.Series([True] * len(data), index=data.index)
                for param_name, fixed_value in fixed_combine_params.items():
                    try:
                        column_name = self.adapter.get_dataframe_column_name(param_name, data)
                        if column_name in data.columns:
                            # Make value matching case agnostic
                            condition &= (data[column_name].astype(str).str.lower() == str(fixed_value).lower())
                        else:
                            logger.warning(f"Fixed parameter column '{column_name}' not found for parameter '{param_name}', skipping")
                    except ValueError as e:
                        logger.warning(f"Fixed parameter '{param_name}' not found in data: {e}")

                # Apply the filter to get only data that matches fixed parameters
                filtered_data = data[condition]

                # Get unique combinations from filtered data
                unique_combinations = filtered_data[list(dict(combine_columns).values())].drop_duplicates()  # type: ignore
            else:
                # No fixed parameters, use all data
                unique_combinations = data[list(dict(combine_columns).values())].drop_duplicates()  # type: ignore

            for _, row in unique_combinations.iterrows():
                # Create filter condition for this combination
                condition = pd.Series([True] * len(data), index=data.index)
                legend_parts = []

                for param_name, column_name in combine_columns:
                    value = row[column_name]
                    # Make value matching case agnostic
                    condition &= (data[column_name].astype(str).str.lower() == str(value).lower())
                    legend_parts.append(f"{param_name}={value}")

                legend_key = ", ".join(legend_parts)
                # Shorten the legend key
                shortened_key = self.adapter.shorten_legend_key(legend_key)
                group_data = data[condition]
                legend_groups[shortened_key] = group_data

        # Sort legend groups for consistent ordering
        sorted_groups = {}
        for key in sorted(legend_groups.keys()):
            group_data = legend_groups[key]

            # Ensure each legend group has only one data point per size
            # This prevents multiple data points for the same size within a legend group
            size_col = self.adapter.get_size_column()
            if size_col in group_data.columns and not group_data.empty:
                # Group by size and take the first occurrence of each size
                # This ensures uniqueness while preserving the data structure
                group_data = group_data.groupby(size_col, as_index=False).first()

            sorted_groups[key] = group_data

        return sorted_groups

    def _generate_chart_title(self, chart_group: Dict[str, Any],
                             chart_config: Any, test_binary: str = "", chart_number: Optional[int] = None) -> str:
        """Generate chart title including fixed and split parameters.

        Args:
            chart_group: Chart group information
            chart_config: Chart configuration
            test_binary: Test binary name to include in title
            chart_number: Optional unique chart number for correlation with data tables

        Returns:
            Formatted chart title
        """
        title_parts = [chart_config.name]

        # Add test binary name if provided
        if test_binary:
            title_parts.append(f"({test_binary})")

        # Get filtered data to check which parameters are present
        filtered_data = self._filter_data_for_chart_group(chart_group)

        # Add fixed parameters to title (only if present in data)
        fixed_params = chart_config.fixed_params or {}
        if fixed_params:
            fixed_parts = []
            for param, config_value in fixed_params.items():
                # Check if parameter column exists in data
                try:
                    actual_column = self.adapter.get_dataframe_column_name(param, filtered_data)
                    if actual_column in filtered_data.columns:
                        # If the config value is "DEFAULT", look up the actual value from the data
                        if config_value == "DEFAULT":
                            # Get the actual value from the data
                            actual_values = filtered_data[actual_column].dropna().unique()
                            if len(actual_values) > 0:
                                # Use the first unique value found in the data
                                actual_value = actual_values[0]
                                fixed_parts.append(f"{param}={actual_value}")
                            else:
                                # No actual values found, use the config value
                                fixed_parts.append(f"{param}={config_value}")
                        else:
                            # Use the config value as-is
                            fixed_parts.append(f"{param}={config_value}")
                except ValueError:
                    # Parameter not found in data, skip it
                    continue
            if fixed_parts:
                title_parts.append(f"[{', '.join(fixed_parts)}]")

        # Add split parameters from chart group (only if present in data)
        # Use filters information if available
        filters = chart_group.get('filters', {})
        if filters:
            filter_parts = []
            for param, value in filters.items():
                # Check if parameter column exists in data
                try:
                    actual_column = self.adapter.get_dataframe_column_name(param, filtered_data)
                    if actual_column in filtered_data.columns:
                        filter_parts.append(f"{param}={value}")
                except ValueError:
                    # Parameter not found in data, skip it
                    continue
            if filter_parts:
                title_parts.append(f"({', '.join(filter_parts)})")
        else:
            # Fallback to old title method
            group_title = chart_group.get('title', '')
            if ' | ' in group_title:
                # Extract split parameters from group title
                parts = group_title.split(' | ')
                if len(parts) > 1:
                    split_info = ' | '.join(parts[1:])  # Skip the first part (chart name)
                    title_parts.append(f"({split_info})")

        # Add chart number if provided
        if chart_number is not None:
            title_parts.append(f"Chart {chart_number}")

        # Join and wrap title if needed
        title = ' '.join(title_parts)

        # Apply consistent title styling using the new title_style configuration
        title = self.style_manager.style_config.apply_title_style(title, self.style_manager)

        return title

    def create_data_table(self, chart_type: str, test_binary: str,
                         chart_group: Dict[str, Any], chart_number: Optional[int] = None) -> List[matplotlib.figure.Figure]:
        """Create data table(s) for the chart, splitting large tables across multiple pages.

        Args:
            chart_type: Type of chart
            test_binary: Test binary name
            chart_group: Chart group information
            chart_number: Optional unique chart number for correlation with charts

        Returns:
            List of matplotlib figures containing the data table(s), or empty list if no data is available
        """
        logger.info(f"Creating data table for: {chart_type} - {test_binary}")

        # Get chart configuration
        if not self.plot_config or not self.plot_config.chart:
            raise ValueError("Plot configuration not available")
        chart_config = self.plot_config.get_chart_by_id(chart_type)
        if not chart_config:
            raise ValueError(f"Chart type '{chart_type}' not found in configuration")

        # Filter data for this chart group
        filtered_data = self._filter_data_for_chart_group(chart_group)
        if filtered_data.empty:
            logger.warning(f"No data found for table: {chart_group['group_key']}")
            return []

        # Get size column
        size_col = self.adapter.get_size_column()
        if size_col not in filtered_data.columns:
            logger.error(f"Size column '{size_col}' not found in data")
            return []

        # Get performance columns
        performance_columns = chart_config.performance_columns
        if not performance_columns:
            performance_columns = [self.adapter.get_performance_columns()[0]] if self.adapter.get_performance_columns() else []

        # Group data by combine parameters
        legend_groups = self._group_data_for_legends(filtered_data, chart_config)

        # Create table data
        table_data = []
        unique_sizes = sorted(filtered_data[size_col].unique())

        # Header row
        header = ['Size'] + list(legend_groups.keys())
        table_data.append(header)

        # Data rows
        for size in unique_sizes:
            row = [self._format_size_for_table(size)]

            for legend_key, group_data in legend_groups.items():
                size_data = group_data[group_data[size_col] == size]
                if not size_data.empty and performance_columns:
                    # Use first performance column for table
                    perf_col = performance_columns[0]
                    if perf_col in size_data.columns:
                        value = size_data[perf_col].iloc[0]  # type: ignore
                        row.append(f"{value:.2f}")
                    else:
                        row.append("N/A")
                else:
                    row.append("N/A")

            table_data.append(row)

        # Calculate how many columns can fit on a page
        max_columns_per_page = self._calculate_max_columns_per_page()

        # Split table into multiple pages if needed
        figures = []
        if len(header) <= max_columns_per_page:
            # Single page table
            fig = self._create_single_page_table(table_data, chart_group, chart_config, test_binary, chart_number)
            if fig is not None:
                figures.append(fig)
        else:
            # Multi-page table
            figures = self._create_multi_page_table(table_data, chart_group, chart_config, test_binary, max_columns_per_page, chart_number)

        logger.info(f"Created {len(figures)} data table page(s) for {chart_group['group_key']}")
        return figures

    def _calculate_max_columns_per_page(self) -> int:
        """Calculate the maximum number of columns that can fit on a page.

        Returns:
            Maximum number of columns per page
        """
        # Get page dimensions from style configuration
        pdf_settings = self.style_manager.style_config.pdf_settings
        table_layout = self.style_manager.style_config.table_layout

        # Calculate available width for table
        page_width = pdf_settings.page_width
        available_width = page_width * (table_layout.bbox_width - table_layout.bbox_left)

        # Estimate column width (including padding and borders)
        # This is a conservative estimate - adjust based on testing
        estimated_column_width = 1.2  # inches per column

        max_columns = int(available_width / estimated_column_width)

        # Ensure at least 2 columns (Size + 1 data column)
        return max(2, max_columns)

    def _create_single_page_table(self, table_data: List[List[str]], chart_group: Dict[str, Any],
                                 chart_config: Any, test_binary: str, chart_number: Optional[int] = None) -> Optional[matplotlib.figure.Figure]:
        """Create a single page data table.

        Args:
            table_data: Complete table data
            chart_group: Chart group information
            chart_config: Chart configuration
            test_binary: Test binary name

        Returns:
            Matplotlib figure containing the data table, or None if creation fails
        """
        # Create figure with consistent sizing
        self.style_manager.apply_global_style()
        fig = plt.figure(figsize=self.style_manager.get_figure_size())
        ax = fig.add_subplot(111)

        # Apply subplot adjustments from PDF settings
        pdf_settings = self.style_manager.style_config.pdf_settings
        fig.subplots_adjust(
            left=pdf_settings.pdf_subplot_left,
            right=pdf_settings.pdf_subplot_right,
            top=pdf_settings.pdf_subplot_top,
            bottom=pdf_settings.pdf_subplot_bottom
        )

        # Wrap column headers using style manager method
        header = table_data[0]
        num_columns = len(header)

        wrapped_header = []
        wrapped_header_lines = []
        for i, col_label in enumerate(header):
            if i == 0 or col_label == 'Size':  # Don't wrap the Size column
                wrapped_header.append(col_label)
                wrapped_header_lines.append(1)
            else:
                wrapped_text, num_lines = self.style_manager.wrap_table_header(col_label, num_columns)
                wrapped_header.append(wrapped_text)
                wrapped_header_lines.append(num_lines)

        # Calculate header row height based on max number of lines, add 10% margin
        max_lines = max(wrapped_header_lines)
        base_height = 0.15
        header_height = base_height * max_lines * 1.1  # 10% extra

        # Create table with wrapped headers
        table = ax.table(
            cellText=table_data[1:],  # Exclude header from cellText
            colLabels=wrapped_header,   # Use wrapped header as column labels
            cellLoc='center',
            loc='center',
            bbox=Bbox(np.array([
                [self.style_manager.style_config.table_layout.bbox_left,
                 self.style_manager.style_config.table_layout.bbox_bottom],
                [self.style_manager.style_config.table_layout.bbox_left +
                 self.style_manager.style_config.table_layout.bbox_width,
                 self.style_manager.style_config.table_layout.bbox_bottom +
                 self.style_manager.style_config.table_layout.bbox_height]
            ]))
        )

        # Style the table with adaptive header height
        self._style_table(table, header_height=header_height)

        # Remove axes
        ax.axis('off')

        # Add title
        title = f"Data Table: {self._generate_chart_title(chart_group, chart_config, test_binary, chart_number)}"

        # Get title styling properties
        title_style = self.style_manager.style_config.title_style
        title_font_size = self.style_manager.style_config.get_title_font_size()

        ax.set_title(title, fontsize=title_font_size,
                    fontweight=title_style.font_weight,
                    color=title_style.color,
                    pad=self.style_manager.style_config.plot_layout.title_pad)

        return fig

    def _create_multi_page_table(self, table_data: List[List[str]], chart_group: Dict[str, Any],
                                chart_config: Any, test_binary: str, max_columns_per_page: int, chart_number: Optional[int] = None) -> List[matplotlib.figure.Figure]:
        """Create multiple page data table by splitting columns.

        Note: This method creates all page figures at once and returns them as a list.
        This can trigger matplotlib's "too many open figures" warning if there are many
        columns that need to be split across multiple pages. The PDF generator handles
        this by temporarily increasing the warning threshold and ensuring all figures
        are properly closed after being saved.

        Args:
            table_data: Complete table data
            chart_group: Chart group information
            chart_config: Chart configuration
            test_binary: Test binary name
            max_columns_per_page: Maximum columns per page

        Returns:
            List of matplotlib figures containing the data table pages
        """
        figures = []
        header = table_data[0]
        data_rows = table_data[1:]

        # Split columns into pages
        num_pages = (len(header) - 1 + max_columns_per_page - 1) // (max_columns_per_page - 1)  # -1 for Size column

        for page_num in range(num_pages):
            start_col = page_num * (max_columns_per_page - 1)
            end_col = min((page_num + 1) * (max_columns_per_page - 1), len(header) - 1)

            # Create page header (Size + selected columns)
            wrapped_header = [header[0]]  # Size column

            # Wrap header text for selected columns
            num_columns = end_col - start_col + 1  # Number of data columns plus Size column
            for col_idx in range(start_col, end_col):
                wrapped_text, num_lines = self.style_manager.wrap_table_header(header[col_idx + 1], num_columns)
                wrapped_header.append(wrapped_text)

            # Calculate header height based on wrapped text
            wrapped_header_lines = []
            for h in wrapped_header:
                if '\n' in h:
                    wrapped_header_lines.append(len(h.split('\n')))
                else:
                    wrapped_header_lines.append(1)

            max_lines = max(wrapped_header_lines)
            base_height = 0.15
            header_height = base_height * max_lines * 1.1  # 10% extra

            page_data = [wrapped_header]

            for row in data_rows:
                page_row = [row[0]] + row[start_col + 1:end_col + 1]  # Size + data columns
                page_data.append(page_row)

            # Create figure for this page
            self.style_manager.apply_global_style()
            fig = plt.figure(figsize=self.style_manager.get_figure_size())
            ax = fig.add_subplot(111)

            # Apply subplot adjustments from PDF settings
            pdf_settings = self.style_manager.style_config.pdf_settings
            fig.subplots_adjust(
                left=pdf_settings.pdf_subplot_left,
                right=pdf_settings.pdf_subplot_right,
                top=pdf_settings.pdf_subplot_top,
                bottom=pdf_settings.pdf_subplot_bottom
            )

            # Create table
            table = ax.table(
                cellText=page_data[1:],  # Exclude header from cellText
                colLabels=page_data[0],   # Use wrapped header as column labels
                cellLoc='center',
                loc='center',
                bbox=Bbox(np.array([
                    [self.style_manager.style_config.table_layout.bbox_left,
                     self.style_manager.style_config.table_layout.bbox_bottom],
                    [self.style_manager.style_config.table_layout.bbox_left +
                     self.style_manager.style_config.table_layout.bbox_width,
                     self.style_manager.style_config.table_layout.bbox_bottom +
                     self.style_manager.style_config.table_layout.bbox_height]
                ]))
            )

            # Style the table with adaptive header height
            self._style_table(table, header_height=header_height)

            # Remove axes
            ax.axis('off')

            # Add title with page information
            base_title = f"Data Table: {self._generate_chart_title(chart_group, chart_config, test_binary, chart_number)}"
            title = f"{base_title} (Page {page_num + 1} of {num_pages})"

            # Get title styling properties
            title_style = self.style_manager.style_config.title_style
            title_font_size = self.style_manager.style_config.get_title_font_size()

            ax.set_title(title, fontsize=title_font_size,
                        fontweight=title_style.font_weight,
                        color=title_style.color,
                        pad=self.style_manager.style_config.plot_layout.title_pad)

            figures.append(fig)

        return figures

    def _style_table(self, table, header_height=0.15) -> None:
        """Apply basic styling to a table. header_height is adaptive for wrapped headers."""
        table.auto_set_font_size(False)
        table.set_fontsize(self.style_manager.style_config.chart_style.font_size +
                          self.style_manager.style_config.font_adjustments.table_header)
        table.scale(self.style_manager.style_config.table_layout.scale_width,
                   self.style_manager.style_config.table_layout.scale_height)

        # Style the header row with increased height and background color
        header_color = "#76B900"

        # Get table dimensions from the cells dictionary
        if hasattr(table, '_cells') and table._cells:
            # Find the maximum row and column indices
            max_row = max(pos[0] for pos in table._cells.keys())
            max_col = max(pos[1] for pos in table._cells.keys())

            # Style the header row (row 0)
            for i in range(max_col + 1):
                cell = table._cells.get((0, i))  # Header cells are in row 0
                if cell:
                    # Set background color for header row
                    cell.set_facecolor(header_color)
                    # Set adaptive height for header row
                    cell.set_height(header_height)
                    # Center the text vertically and horizontally
                    cell.set_text_props(ha='center', va='center')

            # Adjust row heights for data rows to maintain proportion
            for row in range(1, max_row + 1):
                for col in range(max_col + 1):
                    cell = table._cells.get((row, col))
                    if cell:
                        cell.set_height(0.08)  # Standard height for data rows

    def create_command_page(self, chart_type: str, test_binary: str,
                           chart_group: Dict[str, Any]) -> Optional[matplotlib.figure.Figure]:
        """Create a page showing test commands for the chart.

        Args:
            chart_type: Type of chart
            test_binary: Test binary name
            chart_group: Chart group information

        Returns:
            Matplotlib figure containing the command information, or None if no data is available
        """
        logger.info(f"Creating command page for: {chart_type} - {test_binary}")

        # Create figure with consistent sizing
        self.style_manager.apply_global_style()
        fig = plt.figure(figsize=self.style_manager.get_figure_size())
        ax = fig.add_subplot(111)

        # Apply subplot adjustments from PDF settings
        pdf_settings = self.style_manager.style_config.pdf_settings
        fig.subplots_adjust(
            left=pdf_settings.pdf_subplot_left,
            right=pdf_settings.pdf_subplot_right,
            top=pdf_settings.pdf_subplot_top,
            bottom=pdf_settings.pdf_subplot_bottom
        )

        # Filter data for this chart group
        filtered_data = self._filter_data_for_chart_group(chart_group)
        if filtered_data.empty:
            logger.warning(f"No data found for commands: {chart_group['group_key']}")
            return None

        # Get test command column
        test_cmd_col = self.adapter.get_dataframe_column_name('test_cmd', filtered_data)
        if test_cmd_col not in filtered_data.columns:
            logger.warning(f"Test command column '{test_cmd_col}' not found in data")
            return None

        # Get unique commands
        unique_commands = filtered_data[test_cmd_col].dropna().unique()

        # Set up text positioning
        page_layout = self.style_manager.style_config.page_layout
        y_pos = page_layout.command_start_y

        # Add title
        group_key = chart_group.get('group_key', 'Unknown')
        title = f"Test Commands: {group_key}"
        ax.text(0.5, 0.95, title, transform=ax.transAxes,
               fontsize=self.style_manager.style_config.chart_style.font_size +
                        self.style_manager.style_config.font_adjustments.title_medium,
               fontweight='bold', ha='center', va='top')

        # Add commands
        max_line_length = self.style_manager.estimate_max_command_line_length(ax)
        for i, command in enumerate(unique_commands):
            if y_pos < 0.1:  # Stop if we run out of space
                break

            # Wrap command text
            wrapped_command = self.style_manager.wrap_text(
                command,
                max_line_length
            )

            # Add command number and text
            command_text = f"{i+1}. {wrapped_command}"
            ax.text(0.05, y_pos, command_text, transform=ax.transAxes,
                   fontsize=self.style_manager.style_config.chart_style.font_size,
                   ha='left', va='top', fontfamily='monospace')

            # Calculate space needed for this command
            num_lines = len(wrapped_command.split('\n'))
            y_pos -= page_layout.command_step_y + (num_lines - 1) * page_layout.command_line_spacing

        # Remove axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        return fig

    def collect_commands_for_chart(self, chart_type: str, test_binary: str,
                                  chart_group: Dict[str, Any], chart_number: int) -> Dict[str, List[str]]:
        """Collect unique test commands for a chart group, mapped to their legend entries.

        Args:
            chart_type: Type of chart
            test_binary: Test binary name
            chart_group: Chart group information
            chart_number: Chart number for reference

        Returns:
            Dictionary mapping legend keys to lists of command strings found in this chart group
        """
        logger.debug(f"Collecting commands for chart {chart_number}: {chart_type} - {test_binary}")

        # Filter data for this chart group
        filtered_data = self._filter_data_for_chart_group(chart_group)
        if filtered_data.empty:
            logger.warning(f"No data found for commands: {chart_group['group_key']}")
            return {}

        # Get chart configuration for legend grouping
        if not self.plot_config or not self.plot_config.chart:
            logger.warning("Plot configuration not available for legend grouping")
            return {}

        chart_config = self.plot_config.get_chart_by_id(chart_type)
        if not chart_config:
            logger.warning(f"Chart type '{chart_type}' not found in configuration")
            return {}

        # Group data by combine parameters to get legend groups
        legend_groups = self._group_data_for_legends(filtered_data, chart_config)

        # Get test command column
        test_cmd_col = self.adapter.get_dataframe_column_name('test_cmd', filtered_data)
        if test_cmd_col not in filtered_data.columns:
            logger.warning(f"Test command column '{test_cmd_col}' not found in data")
            return {}

        # Collect commands for each legend group
        legend_commands = {}
        for legend_key, group_data in legend_groups.items():
            # Get unique commands for this legend group
            unique_commands = group_data[test_cmd_col].dropna().unique().tolist()
            if unique_commands:
                legend_commands[legend_key] = unique_commands

        logger.debug(f"Found commands for {len(legend_commands)} legend groups in chart {chart_number}")
        return legend_commands

    def create_command_pages_for_chart(self, chart_type: str, test_binary: str,
                                      chart_group: Dict[str, Any], chart_number: int) -> List[matplotlib.figure.Figure]:
        """Create command pages for a specific chart, with multiple legend groups per page.

        Args:
            chart_type: Type of chart
            test_binary: Test binary name
            chart_group: Chart group information
            chart_number: Chart number for reference

        Returns:
            List of matplotlib figures containing the command information for this chart
        """
        logger.info(f"Creating command pages for chart {chart_number}: {chart_type} - {test_binary}")

        # Get chart configuration for legend grouping
        if not self.plot_config or not self.plot_config.chart:
            logger.warning("Plot configuration not available for legend grouping")
            return []

        chart_config = self.plot_config.get_chart_by_id(chart_type)
        if not chart_config:
            logger.warning(f"Chart type '{chart_type}' not found in configuration")
            return []

        # Filter data for this chart group
        filtered_data = self._filter_data_for_chart_group(chart_group)
        if filtered_data.empty:
            logger.warning(f"No data found for commands: {chart_group['group_key']}")
            return []

        # Group data by combine parameters to get legend groups
        legend_groups = self._group_data_for_legends(filtered_data, chart_config)

        # Get test command column
        test_cmd_col = self.adapter.get_dataframe_column_name('test_cmd', filtered_data)
        if test_cmd_col not in filtered_data.columns:
            logger.warning(f"Test command column '{test_cmd_col}' not found in data")
            return []

        # Generate chart title for command pages
        chart_title = self._generate_chart_title(chart_group, chart_config, test_binary, chart_number)

        # Collect all commands with their legend keys for this chart
        all_legend_commands = {}
        for legend_key, group_data in legend_groups.items():
            # Get unique commands for this legend group
            unique_commands = group_data[test_cmd_col].dropna().unique().tolist()
            if unique_commands:
                all_legend_commands[legend_key] = unique_commands

        if not all_legend_commands:
            logger.debug(f"No commands found for chart {chart_number}")
            return []

        # Create pages that can contain multiple legend groups
        figures = self._create_multi_legend_command_pages(
            chart_title, all_legend_commands, chart_number
        )

        logger.info(f"Created {len(figures)} command pages for chart {chart_number}")
        return figures

    def _create_multi_legend_command_pages(self, chart_title: str,
                                          legend_commands: Dict[str, List[str]],
                                          chart_number: int) -> List[matplotlib.figure.Figure]:
        """Create command pages that can contain multiple legend groups from the same chart.

        Args:
            chart_title: Title of the chart
            legend_commands: Dictionary mapping legend keys to lists of commands
            chart_number: Chart number for reference

        Returns:
            List of matplotlib figures containing the command information
        """
        if not legend_commands:
            return []

        # Calculate how many command entries we can fit per page
        entries_per_page = self._calculate_command_entries_per_page()

        # Flatten all commands with their legend keys for pagination
        command_entries = []
        for legend_key, commands in legend_commands.items():
            for command in commands:
                command_entries.append({
                    'legend_key': legend_key,
                    'command': command
                })

        # Sort by legend key first, then by command for consistent ordering
        command_entries.sort(key=lambda x: (x['legend_key'], x['command']))

        figures = []

        # Split command entries into pages
        for page_num in range(0, len(command_entries), entries_per_page):
            page_entries = command_entries[page_num:page_num + entries_per_page]

            # Create figure for this page
            self.style_manager.apply_global_style()
            fig = plt.figure(figsize=self.style_manager.get_figure_size())
            ax = fig.add_subplot(111)

            # Apply subplot adjustments from PDF settings
            pdf_settings = self.style_manager.style_config.pdf_settings
            fig.subplots_adjust(
                left=pdf_settings.pdf_subplot_left,
                right=pdf_settings.pdf_subplot_right,
                top=pdf_settings.pdf_subplot_top,
                bottom=pdf_settings.pdf_subplot_bottom
            )

            # Set up text positioning
            page_layout = self.style_manager.style_config.page_layout
            y_pos = page_layout.command_start_y
            max_line_length = self.style_manager.estimate_max_command_line_length(ax)

            # Calculate total pages for this chart
            total_pages = (len(command_entries) + entries_per_page - 1) // entries_per_page

            # Add chart title with page information
            base_title = f"Test Commands: {chart_title}"
            title = f"{base_title} (Page {page_num // entries_per_page + 1} of {total_pages})"

            # Apply consistent title styling using the new title_style configuration
            title = self.style_manager.style_config.apply_title_style(title, self.style_manager)

            # Get title styling properties
            title_style = self.style_manager.style_config.title_style
            title_font_size = self.style_manager.style_config.get_title_font_size()

            ax.text(0.5, 0.95, title, transform=ax.transAxes,
                   fontsize=title_font_size,
                   fontweight=title_style.font_weight,
                   ha=title_style.alignment,
                   va=title_style.vertical_alignment,
                   color=title_style.color)

            # Add command entries with legend labels
            for i, entry in enumerate(page_entries):
                if y_pos < 0.1:  # Stop if we run out of space
                    break

                legend_key = entry['legend_key']
                command = entry['command']

                # Wrap command text
                wrapped_command = self.style_manager.wrap_text(
                    command,
                    max_line_length
                )

                # Add legend key as label
                label_text = f"[{legend_key}]"
                ax.text(0.05, y_pos, label_text, transform=ax.transAxes,
                       fontsize=self.style_manager.style_config.chart_style.font_size +
                                self.style_manager.style_config.font_adjustments.label_small,
                       ha='left', va='top', fontweight='bold', color='blue')

                # Add command text (indented)
                command_text = f"  {wrapped_command}"
                ax.text(0.05, y_pos - 0.02, command_text, transform=ax.transAxes,
                       fontsize=self.style_manager.style_config.chart_style.font_size,
                       ha='left', va='top', fontfamily='monospace')

                # Calculate space needed for this command entry
                num_lines = len(wrapped_command.split('\n'))
                y_pos -= page_layout.command_step_y + (num_lines - 1) * page_layout.command_line_spacing + 0.02

            # Remove axes
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

            figures.append(fig)

        return figures

    def create_comprehensive_command_page(self, all_commands: Dict[str, Dict[str, List[int]]]) -> List[matplotlib.figure.Figure]:
        """Create comprehensive command pages showing all unique commands with their legend labels and chart references.

        Args:
            all_commands: Dictionary mapping legend keys to dictionaries of command strings to chart numbers where they appear

        Returns:
            List of matplotlib figures containing the command information
        """
        logger.info(f"Creating comprehensive command pages for {len(all_commands)} legend groups")

        if not all_commands:
            logger.warning("No commands to display")
            return []

        # Flatten the structure to get all unique commands with their legend labels and chart references
        command_entries = []
        for legend_key, commands_dict in all_commands.items():
            for command, chart_numbers in commands_dict.items():
                command_entries.append({
                    'legend_key': legend_key,
                    'command': command,
                    'chart_numbers': chart_numbers
                })

        # Sort by legend key first, then by command for consistent ordering
        command_entries.sort(key=lambda x: (x['legend_key'], x['command']))

        # Calculate how many command entries we can fit per page
        page_layout = self.style_manager.style_config.page_layout
        entries_per_page = self._calculate_command_entries_per_page()

        figures = []

        # Split command entries into pages
        for page_num in range(0, len(command_entries), entries_per_page):
            page_entries = command_entries[page_num:page_num + entries_per_page]

            # Create figure for this page
            self.style_manager.apply_global_style()
            fig = plt.figure(figsize=self.style_manager.get_figure_size())
            ax = fig.add_subplot(111)

            # Apply subplot adjustments from PDF settings
            pdf_settings = self.style_manager.style_config.pdf_settings
            fig.subplots_adjust(
                left=pdf_settings.pdf_subplot_left,
                right=pdf_settings.pdf_subplot_right,
                top=pdf_settings.pdf_subplot_top,
                bottom=pdf_settings.pdf_subplot_bottom
            )

            # Set up text positioning
            y_pos = page_layout.command_start_y
            max_line_length = self.style_manager.estimate_max_command_line_length(ax)

            # Add title
            total_pages = (len(command_entries) + entries_per_page - 1) // entries_per_page
            title = f"All Test Commands (Page {page_num // entries_per_page + 1} of {total_pages})"

            # Apply consistent title styling using the new title_style configuration
            title = self.style_manager.style_config.apply_title_style(title, self.style_manager)

            # Get title styling properties
            title_style = self.style_manager.style_config.title_style
            title_font_size = self.style_manager.style_config.get_title_font_size()

            ax.text(0.5, 0.95, title, transform=ax.transAxes,
                   fontsize=title_font_size,
                   fontweight=title_style.font_weight,
                   ha=title_style.alignment,
                   va=title_style.vertical_alignment,
                   color=title_style.color)

            # Add command entries with legend labels and chart references
            for i, entry in enumerate(page_entries):
                if y_pos < 0.1:  # Stop if we run out of space
                    break

                legend_key = entry['legend_key']
                command = entry['command']
                chart_numbers = entry['chart_numbers']

                # Wrap command text
                wrapped_command = self.style_manager.wrap_text(
                    command,
                    max_line_length
                )

                # Format chart numbers
                chart_refs = ", ".join(map(str, sorted(chart_numbers)))

                # Add legend key as label
                label_text = f"[{legend_key}]"
                ax.text(0.05, y_pos, label_text, transform=ax.transAxes,
                       fontsize=self.style_manager.style_config.chart_style.font_size +
                                self.style_manager.style_config.font_adjustments.label_small,
                       ha='left', va='top', fontweight='bold', color='blue')

                # Add command text (indented)
                command_text = f"  {wrapped_command}"
                ax.text(0.05, y_pos - 0.02, command_text, transform=ax.transAxes,
                       fontsize=self.style_manager.style_config.chart_style.font_size,
                       ha='left', va='top', fontfamily='monospace')

                # Add chart references on the right
                ref_text = f"Charts: {chart_refs}"
                ax.text(0.95, y_pos, ref_text, transform=ax.transAxes,
                       fontsize=self.style_manager.style_config.chart_style.font_size +
                                self.style_manager.style_config.font_adjustments.label_small,
                       ha='right', va='top', alpha=0.7)

                # Calculate space needed for this command entry
                num_lines = len(wrapped_command.split('\n'))
                y_pos -= page_layout.command_step_y + (num_lines - 1) * page_layout.command_line_spacing + 0.02

            # Remove axes
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

            figures.append(fig)

        logger.info(f"Created {len(figures)} comprehensive command pages")
        return figures

    def _calculate_commands_per_page(self) -> int:
        """Calculate how many commands can fit on a single page.

        Returns:
            Number of commands that can fit per page
        """
        # Estimate based on page layout settings
        page_layout = self.style_manager.style_config.page_layout
        available_height = page_layout.command_start_y - 0.1  # Leave 10% margin at bottom
        command_height = page_layout.command_step_y

        # Account for title and subtitle space (approximately 0.15 total)
        title_space = 0.15
        usable_height = available_height - title_space

        # Calculate commands per page (less conservative)
        commands_per_page = int(usable_height / command_height)

        # Ensure reasonable bounds: at least 3 commands, at most 15
        commands_per_page = max(3, min(15, commands_per_page))

        logger.debug(f"Command pagination: available_height={available_height:.3f}, "
                    f"command_height={command_height:.3f}, commands_per_page={commands_per_page}")

        return commands_per_page

    def _calculate_command_entries_per_page(self) -> int:
        """Calculate how many command entries (with legend labels) can fit on a single page.

        Returns:
            Number of command entries that can fit per page
        """
        # Estimate based on page layout settings, accounting for legend labels
        page_layout = self.style_manager.style_config.page_layout
        available_height = page_layout.command_start_y - 0.1  # Leave 10% margin at bottom
        command_height = page_layout.command_step_y + 0.02  # Add extra space for legend label

        # Account for title and subtitle space (approximately 0.15 total)
        title_space = 0.15
        usable_height = available_height - title_space

        # Calculate entries per page (less conservative)
        entries_per_page = int(usable_height / command_height)

        # Ensure reasonable bounds: at least 2 entries, at most 12
        entries_per_page = max(2, min(12, entries_per_page))

        logger.debug(f"Command entry pagination: available_height={available_height:.3f}, "
                    f"command_height={command_height:.3f}, entries_per_page={entries_per_page}")

        return entries_per_page

    def _format_size_for_table(self, size: float) -> str:
        """Format size value for display in table.

        Args:
            size: Size value in bytes

        Returns:
            Formatted size string
        """
        if size < 1024:
            return f"{int(size)}B"
        elif size < 1024 * 1024:
            return f"{size/1024:.0f}KB"
        elif size < 1024 * 1024 * 1024:
            return f"{size/(1024*1024):.0f}MB"
        else:
            return f"{size/(1024*1024*1024):.1f}GB"
