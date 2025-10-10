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
Style manager for matplotlib plots.

This module handles the application of style configurations to matplotlib plots,
ensuring consistent styling across all charts in the report.
"""

import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib.axes import Axes
from matplotlib.ticker import LogLocator
import numpy as np
from typing import Optional, Tuple, List, Any
import logging
from contextlib import contextmanager

from ..models import StyleConfig
from ..models.plot_config import ChartConfig

logger = logging.getLogger(__name__)


class StyleManager:
    """Manages matplotlib styling based on configuration."""

    def __init__(self, style_config: StyleConfig, plot_config: Optional[ChartConfig] = None):
        """Initialize style manager with configuration.

        Args:
            style_config: Style configuration object
            plot_config: Optional plot configuration object for Y-axis limits
        """
        self.style_config = style_config
        self.plot_config = plot_config
        self._applied_global_style = False

    def apply_global_style(self) -> None:
        """Apply global matplotlib style settings."""
        if self._applied_global_style:
            return

        # Set up matplotlib with custom style
        plt.style.use('default')  # Start with default style

        plt.rcParams.update({
            'axes.facecolor': 'white',
            'axes.edgecolor': 'black',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'grid.linestyle': self.style_config.grid.style,
            'grid.alpha': self.style_config.grid.alpha,
            'grid.color': self.style_config.grid.color,
        })

        # Apply global chart style
        chart_style = self.style_config.chart_style

        # Set global font properties
        plt.rcParams.update({
            'font.size': chart_style.font_size,
            'axes.labelsize': chart_style.font_size + self.style_config.font_adjustments.label_large,
            'axes.titlesize': chart_style.font_size + self.style_config.font_adjustments.title_medium,
            'xtick.labelsize': chart_style.font_size + self.style_config.font_adjustments.label_small,
            'ytick.labelsize': chart_style.font_size + self.style_config.font_adjustments.label_small,
            'legend.fontsize': self.style_config.legend.fontsize,
            'figure.titlesize': chart_style.font_size + self.style_config.font_adjustments.title_large,

            # Colors
            'axes.facecolor': chart_style.axes_facecolor,
            'axes.edgecolor': chart_style.axes_edgecolor,
            'axes.labelcolor': chart_style.axes_labelcolor,
            'text.color': chart_style.text_color,
            'xtick.color': chart_style.xtick_color,
            'ytick.color': chart_style.ytick_color,
            'figure.facecolor': chart_style.background_color,
            'savefig.facecolor': self.style_config.pdf_settings.pdf_facecolor,
            'savefig.edgecolor': self.style_config.pdf_settings.pdf_edgecolor,
            'savefig.dpi': self.style_config.pdf_settings.pdf_dpi,

            # Spines
            'axes.spines.top': chart_style.axes_spines_top,
            'axes.spines.right': chart_style.axes_spines_right,

            # Lines and markers
            'lines.linewidth': chart_style.line_width,
            'lines.markersize': chart_style.marker_size,
            'lines.markeredgewidth': chart_style.markeredgewidth,
            'lines.markeredgecolor': chart_style.markeredgecolor,

            # Grid - Enhanced with new options
            'axes.grid': self.style_config.grid.show,
            'grid.alpha': self.style_config.grid.alpha,
            'grid.color': self.style_config.grid.major_color,
            'grid.linestyle': self.style_config.grid.major_style,
            'grid.linewidth': self.style_config.grid.major_linewidth,

            # Ticks - Enhanced with new options
            'xtick.minor.size': self.style_config.ticks.minor_length,
            'ytick.minor.size': self.style_config.ticks.minor_length,
            'xtick.major.size': self.style_config.ticks.major_length,
            'ytick.major.size': self.style_config.ticks.major_length,
            'xtick.minor.width': self.style_config.ticks.minor_width,
            'ytick.minor.width': self.style_config.ticks.minor_width,
            'xtick.major.width': self.style_config.ticks.major_width,
            'ytick.major.width': self.style_config.ticks.major_width,
            'xtick.direction': self.style_config.ticks.tick_direction,
            'ytick.direction': self.style_config.ticks.tick_direction,
        })

        self._applied_global_style = True
        logger.info("Applied global matplotlib style configuration")

    def get_figure_size(self, orientation: str = 'landscape') -> Tuple[float, float]:
        """Get figure size based on PDF settings and orientation.

        Args:
            orientation: 'landscape' or 'portrait'

        Returns:
            Tuple of (width, height) in inches
        """
        pdf_settings = self.style_config.pdf_settings

        if orientation == 'landscape' and pdf_settings.use_landscape:
            # For landscape, use height as width and width as height
            available_width = pdf_settings.page_height - pdf_settings.margin_left - pdf_settings.margin_right
            available_height = pdf_settings.page_width - pdf_settings.margin_top - pdf_settings.margin_bottom
        else:
            # For portrait
            available_width = pdf_settings.page_width - pdf_settings.margin_left - pdf_settings.margin_right
            available_height = pdf_settings.page_height - pdf_settings.margin_top - pdf_settings.margin_bottom

        return (available_width, available_height)

    def create_figure(self, orientation: str = 'landscape') -> Tuple[matplotlib.figure.Figure, Axes]:
        """Create a new figure with proper styling.

        Args:
            orientation: 'landscape' or 'portrait'

        Returns:
            Tuple of (figure, axes)
        """
        self.apply_global_style()

        figsize = self.get_figure_size(orientation)
        fig, ax = plt.subplots(figsize=figsize)

        # Apply subplot adjustments for PDF
        pdf_settings = self.style_config.pdf_settings
        fig.subplots_adjust(
            left=pdf_settings.pdf_subplot_left,
            right=pdf_settings.pdf_subplot_right,
            top=pdf_settings.pdf_subplot_top,
            bottom=pdf_settings.pdf_subplot_bottom
        )

        return fig, ax

    def close_figure(self, fig: matplotlib.figure.Figure) -> None:
        """Close a matplotlib figure to free memory.

        Args:
            fig: Matplotlib figure to close
        """
        try:
            plt.close(fig)
            logger.debug("Closed matplotlib figure")
        except Exception as e:
            logger.warning(f"Failed to close figure: {e}")

    def close_all_figures(self) -> None:
        """Close all open matplotlib figures to free memory."""
        try:
            plt.close('all')
            logger.info("Closed all matplotlib figures")
        except Exception as e:
            logger.warning(f"Failed to close all figures: {e}")

    @contextmanager
    def figure_context(self, orientation: str = 'landscape'):
        """Context manager for creating and automatically closing figures.

        Args:
            orientation: 'landscape' or 'portrait'

        Yields:
            Tuple of (figure, axes)
        """
        fig, ax = self.create_figure(orientation)
        try:
            yield fig, ax
        finally:
            self.close_figure(fig)

    def style_axes(self, ax: Axes, test_binary: str = "", data: Optional[np.ndarray] = None) -> None:
        """Apply styling to axes.

        Args:
            ax: Matplotlib axes object
            test_binary: Test binary name for context-specific styling
            data: Optional data array for calculating Y-axis limits
        """
        # Set x-axis properties
        # Check scale type from plot config
        scale_type_default = "semilogx"  # Default to semilogx
        x_axis_config = None
        scale_type = scale_type_default
        if self.plot_config and self.plot_config.settings and self.plot_config.settings.x_axis:
            x_axis_config = self.plot_config.settings.x_axis
            scale_type = x_axis_config.scale or scale_type_default

        # Determine if we should use log scale
        use_log_scale = scale_type in ["semilogx", "log", "loglog"]

        if use_log_scale:
            ax.set_xscale('log')

            # Use configured ticks and labels from plot config for log scale
            if x_axis_config and x_axis_config.ticks and x_axis_config.labels:
                # Set the major tick positions to match the configured ticks
                ax.set_xticks(x_axis_config.ticks)
                # Apply the configured labels
                ax.set_xticklabels(x_axis_config.labels)
                logger.debug(f"Applied configured X-axis ticks: {x_axis_config.ticks}")
                logger.debug(f"Applied configured X-axis labels: {x_axis_config.labels}")

                # Set minor locator to ensure minor ticks are generated between major ticks
                # Use a comprehensive set of minor tick positions
                subs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
                ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=subs))

                # Force minor ticks to be visible
                ax.minorticks_on()
                logger.debug("Applied configured X-axis with minor ticks for log scale")
            else:
                # Fallback to auto-determination if no configuration provided
                ax.xaxis.set_major_locator(LogLocator(base=10.0))
                ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=list(np.arange(2, 10) * 0.1)))
                ax.minorticks_on()
                logger.debug("Using auto-determined X-axis ticks for log scale")
        else:
            # For linear scale, use configured tick positions and labels from plot config
            if x_axis_config and x_axis_config.ticks and x_axis_config.labels:
                ax.set_xticks(x_axis_config.ticks)
                ax.set_xticklabels(x_axis_config.labels)
                logger.debug(f"Applied configured X-axis ticks for linear scale: {x_axis_config.ticks}")
                logger.debug(f"Applied configured X-axis labels for linear scale: {x_axis_config.labels}")

        ax.set_xlabel('Message Size', fontsize=self.style_config.chart_style.font_size +
                      self.style_config.font_adjustments.label_large)

        # Set y-axis properties
        ax.set_ylabel('Bandwidth (GB/s)', fontsize=self.style_config.chart_style.font_size +
                      self.style_config.font_adjustments.label_large)

        # Set y-axis limits based on test binary and data
        y_limits = self._get_y_limits_for_test_binary(test_binary, data)
        if y_limits:
            ax.set_ylim(y_limits[0], y_limits[1])

        # Apply grid styling with proper order for log scale
        if self.style_config.grid.show:
            # For log scale, apply grid in specific order to ensure minor gridlines work
            if use_log_scale:
                # First, apply major gridlines
                ax.grid(True, alpha=self.style_config.grid.alpha,
                       color=self.style_config.grid.major_color,
                       linestyle=self.style_config.grid.major_style,
                       linewidth=self.style_config.grid.major_linewidth)

                # Then apply minor gridlines for X-axis if enabled
                if self.style_config.grid.minor_x_axis:
                    ax.grid(True, which='minor', alpha=self.style_config.grid.minor_alpha,
                           color=self.style_config.grid.minor_color,
                           linestyle=self.style_config.grid.minor_style,
                           linewidth=self.style_config.grid.minor_linewidth)
                    logger.debug("Applied minor gridlines for X-axis log scale")

                # Apply minor gridlines for Y-axis if enabled
                if self.style_config.grid.minor_y_axis:
                    ax.grid(True, which='minor', alpha=self.style_config.grid.minor_alpha,
                           color=self.style_config.grid.minor_color,
                           linestyle=self.style_config.grid.minor_style,
                           linewidth=self.style_config.grid.minor_linewidth)
                    logger.debug("Applied minor gridlines for Y-axis")
            else:
                # For linear scale, apply standard grid
                ax.grid(True, alpha=self.style_config.grid.alpha,
                       color=self.style_config.grid.major_color,
                       linestyle=self.style_config.grid.major_style,
                       linewidth=self.style_config.grid.major_linewidth)

                # Apply minor gridlines if enabled
                if self.style_config.grid.minor_x_axis or self.style_config.grid.minor_y_axis:
                    ax.grid(True, which='minor', alpha=self.style_config.grid.minor_alpha,
                           color=self.style_config.grid.minor_color,
                           linestyle=self.style_config.grid.minor_style,
                           linewidth=self.style_config.grid.minor_linewidth)

        # Set tick parameters for both major and minor ticks with enhanced styling
        ax.tick_params(axis='both', which='major',
                      length=self.style_config.ticks.major_length,
                      width=self.style_config.ticks.major_width,
                      color=self.style_config.ticks.major_color,
                      direction=self.style_config.ticks.tick_direction)
        ax.tick_params(axis='both', which='minor',
                      length=self.style_config.ticks.minor_length,
                      width=self.style_config.ticks.minor_width,
                      color=self.style_config.ticks.minor_color,
                      direction=self.style_config.ticks.tick_direction)

        # Apply background effects if enabled
        self.apply_background_effects(ax)

        # Set X-axis limits to ensure all configured ticks are visible
        self._set_x_axis_limits(ax, use_log_scale, x_axis_config)

    def _set_x_axis_limits(self, ax: Axes, use_log_scale: bool, x_axis_config: Optional[Any]) -> None:
        """Set X-axis limits to ensure all configured ticks are visible.

        Args:
            ax: Matplotlib axes object
            use_log_scale: Whether using log scale
            x_axis_config: X-axis configuration from plot config
        """
        if not x_axis_config or not x_axis_config.ticks:
            return

        ticks = x_axis_config.ticks
        if not ticks:
            return

        # For log scale, ensure limits include all configured ticks
        if use_log_scale:
            min_tick = min(ticks)
            max_tick = max(ticks)

            # Add some padding for log scale
            min_limit = min_tick * 0.8
            max_limit = max_tick * 1.2

            # Ensure limits are positive for log scale
            min_limit = max(min_limit, 1)

            ax.set_xlim(min_limit, max_limit)
            logger.debug(f"Set X-axis limits for log scale: [{min_limit}, {max_limit}]")
        else:
            # For linear scale, use standard padding
            min_tick = min(ticks)
            max_tick = max(ticks)

            # Add 10% padding
            range_size = max_tick - min_tick
            padding = range_size * 0.1

            ax.set_xlim(min_tick - padding, max_tick + padding)
            logger.debug(f"Set X-axis limits for linear scale: [{min_tick - padding}, {max_tick + padding}]")

        # Force minor gridlines to be visible
        self._ensure_minor_gridlines_visible(ax, use_log_scale)

    def _ensure_minor_gridlines_visible(self, ax: Axes, use_log_scale: bool) -> None:
        """Ensure minor gridlines are properly visible.

        Args:
            ax: Matplotlib axes object
            use_log_scale: Whether using log scale
        """
        if not self.style_config.grid.show:
            return

        # For log scale, ensure minor gridlines are applied
        if use_log_scale and self.style_config.grid.minor_x_axis:
            # Force minor ticks to be visible
            ax.minorticks_on()

            # Apply minor gridlines again to ensure they're visible
            ax.grid(True, which='minor', alpha=self.style_config.grid.minor_alpha,
                   color=self.style_config.grid.minor_color,
                   linestyle=self.style_config.grid.minor_style,
                   linewidth=self.style_config.grid.minor_linewidth)

            # Set minor tick parameters
            ax.tick_params(axis='x', which='minor',
                         length=self.style_config.ticks.minor_length,
                         width=self.style_config.ticks.minor_width,
                         color=self.style_config.ticks.minor_color,
                         direction=self.style_config.ticks.tick_direction)

            logger.debug("Forced minor gridlines visibility for X-axis log scale")

        # For Y-axis minor gridlines
        if self.style_config.grid.minor_y_axis:
            ax.tick_params(axis='y', which='minor',
                         length=self.style_config.ticks.minor_length,
                         width=self.style_config.ticks.minor_width,
                         color=self.style_config.ticks.minor_color,
                         direction=self.style_config.ticks.tick_direction)

    def _get_y_limits_for_test_binary(self, test_binary: str, data: Optional[np.ndarray] = None) -> Optional[Tuple[float, float]]:
        """Get y-axis limits for a specific test binary.

        Args:
            test_binary: Test binary name
            data: Optional data array for calculating auto-detected limits

        Returns:
            Tuple of (min, max) limits or None for auto-scaling
        """
        # Get configured limits from plot config if available
        configured_min = None
        configured_max = None

        if self.plot_config and self.plot_config.settings and self.plot_config.settings.y_axis:
            y_limits_config = self.plot_config.settings.y_axis.limits
            if y_limits_config:
                # Check test binary specific limits first
                if y_limits_config.test_binaries and test_binary in y_limits_config.test_binaries:
                    test_binary_config = y_limits_config.test_binaries[test_binary]
                    configured_min = test_binary_config.min
                    configured_max = test_binary_config.max
                else:
                    # Check if it's a P2P operation
                    if 'sendrecv' in test_binary or 'alltoall' in test_binary:
                        configured_min = y_limits_config.p2p_min
                        configured_max = y_limits_config.p2p_max
                    else:
                        # Use default limits
                        configured_min = y_limits_config.default_min
                        configured_max = y_limits_config.default_max

        # Calculate auto-detected limits from data if available
        auto_detected_max = None
        if data is not None and len(data) > 0:
            # Remove NaN values
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                auto_detected_max = np.max(valid_data)

                # Apply auto-detection settings if available
                if self.plot_config and self.plot_config.settings and self.plot_config.settings.y_axis:
                    y_limits_config = self.plot_config.settings.y_axis.limits
                    if y_limits_config:
                        # Add margin
                        if y_limits_config.auto_detect_margin:
                            auto_detected_max *= (1 + y_limits_config.auto_detect_margin)

                        # Round to nearest multiple
                        if y_limits_config.auto_detect_round_to:
                            auto_detected_max = np.ceil(auto_detected_max / y_limits_config.auto_detect_round_to) * y_limits_config.auto_detect_round_to

        # Determine final limits
        final_min = configured_min if configured_min is not None else 0
        final_max = None

        if configured_max is not None and auto_detected_max is not None:
            # Use the maximum of configured and auto-detected limits
            final_max = max(configured_max, auto_detected_max)
        elif configured_max is not None:
            # Use only configured limit
            final_max = configured_max
        elif auto_detected_max is not None:
            # Use only auto-detected limit
            final_max = auto_detected_max

        if final_max is not None:
            return (final_min, final_max)

        return None

    def style_legend(self, ax: Axes, legend_items: List[str]) -> None:
        """Apply legend styling.

        Args:
            ax: Matplotlib axes object
            legend_items: List of legend labels
        """
        if not self.style_config.legend.show:
            return

        # Determine number of columns based on legend size
        num_items = len(legend_items)
        if (self.style_config.legend_positioning.adaptive_columns and
            num_items > self.style_config.legend.columns * 3):
            # Use more columns for large legends
            ncol = min(num_items, self.style_config.legend.columns * 2)
        else:
            ncol = self.style_config.legend.columns

        # Determine font size
        if (self.style_config.legend_positioning.adaptive_font_size and
            num_items > self.style_config.legend.columns * 4):
            # Use smaller font for large legends
            fontsize = max(self.style_config.legend.fontsize - 1, 7)
        else:
            fontsize = self.style_config.legend.fontsize

        # Calculate legend position
        legend_y = self.style_config.legend_positioning.anchor_y_offset

        # Check if there are any labeled artists before creating legend
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            # No labeled artists found, skip legend creation
            logger.debug("No labeled artists found, skipping legend creation")
            return

        # Create legend
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(self.style_config.legend_positioning.anchor_x, legend_y),
            ncol=ncol,
            fontsize=fontsize,
            framealpha=self.style_config.legend.framealpha,
            shadow=self.style_config.legend.shadow,
            fancybox=self.style_config.legend.fancybox,
            edgecolor=self.style_config.legend.edgecolor
        )

        # Adjust subplot to make room for legend
        fig = ax.get_figure()
        if fig:
            legend_height = num_items * self.style_config.legend_positioning.height_per_row / ncol
            legend_height = min(legend_height, self.style_config.legend_positioning.max_legend_height)

            # Adjust bottom margin to accommodate legend
            bottom_margin = self.style_config.legend_positioning.bottom_margin_base + legend_height
            fig.subplots_adjust(bottom=bottom_margin)

    def add_annotations(self, ax: Axes, x_data: np.ndarray, y_data: np.ndarray,
                       label: str) -> None:
        """Add bandwidth annotations to the plot.

        Args:
            ax: Matplotlib axes object
            x_data: X-axis data (message sizes)
            y_data: Y-axis data (bandwidth values)
            label: Label for the data series
        """
        if not self.style_config.annotations.show_final_bw:
            return

        # Find the maximum bandwidth value and its position
        max_idx = np.argmax(y_data)
        max_x = x_data[max_idx]
        max_y = y_data[max_idx]

        # Create annotation text
        annotation_text = f'{max_y:.1f} GB/s'

        # Add annotation
        ax.annotate(
            annotation_text,
            xy=(max_x, max_y),
            xytext=(self.style_config.annotation_positioning.text_offset_x,
                   self.style_config.annotation_positioning.text_offset_y),
            textcoords='offset points',
            fontsize=self.style_config.annotations.fontsize,
            fontweight=self.style_config.annotations.fontweight,
            bbox=dict(
                boxstyle=self.style_config.annotations.bbox_boxstyle,
                facecolor=self.style_config.annotations.bbox_facecolor,
                alpha=self.style_config.annotations.bbox_alpha,
                edgecolor=self.style_config.annotations.bbox_edgecolor,
                linewidth=self.style_config.annotations.bbox_linewidth
            ),
            zorder=self.style_config.annotations.zorder
        )

    def create_lined_page(self) -> matplotlib.figure.Figure:
        """Create a lined page for notes.

        Returns:
            Figure with lined page
        """
        self.apply_global_style()

        figsize = self.get_figure_size()
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # Remove axes
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xticks([])
        ax.set_yticks([])

        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add horizontal lines
        page_layout = self.style_config.page_layout
        for y in range(page_layout.note_grid_start, page_layout.note_grid_end, page_layout.note_grid_step):
            ax.axhline(y=y, color=self.style_config.grid.color,
                      linewidth=self.style_config.grid_lines.note_page_linewidth,
                      alpha=self.style_config.grid.alpha)

        # Add title
        ax.text(self.style_config.text_positioning.title_x,
               self.style_config.text_positioning.title_y,
               'Notes',
               transform=ax.transAxes,
               fontsize=self.style_config.chart_style.font_size + self.style_config.font_adjustments.title_large,
               fontweight='bold',
               ha='center', va='top')

        return fig

    def wrap_text(self, text: str, max_length: int) -> str:
        """Wrap text to fit within specified length.

        Args:
            text: Text to wrap
            max_length: Maximum length per line

        Returns:
            Wrapped text with newlines
        """
        if len(text) <= max_length:
            return text

        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 > max_length:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
                else:
                    # Single word longer than max_length
                    lines.append(word)
                    current_length = 0
            else:
                current_line.append(word)
                current_length += len(word) + 1

        if current_line:
            lines.append(' '.join(current_line))

        return '\n'.join(lines)

    def calculate_table_header_wrap_length(self, num_columns: int) -> int:
        """Calculate optimal wrap length for table headers based on number of columns.

        Args:
            num_columns: Number of columns in the table

        Returns:
            Optimal wrap length for table headers
        """
        text_wrapping_config = self.style_config.text_wrapping
        min_wrap_length = text_wrapping_config.table_header_min_wrap_length
        max_wrap_length = text_wrapping_config.table_header_max_wrap_length

        if num_columns <= 2:
            # Few columns - allow longer headers
            wrap_length = max_wrap_length
        elif num_columns <= 4:
            # Medium number of columns - moderate wrapping
            wrap_length = max(min_wrap_length, max_wrap_length * 0.8)
        else:
            # Many columns - more aggressive wrapping but still reasonable
            wrap_length = max(min_wrap_length, max_wrap_length * 0.6)

        return int(wrap_length)

    def wrap_table_header(self, header_text: str, num_columns: int) -> Tuple[str, int]:
        """Wrap a table header if it exceeds the threshold.

        Args:
            header_text: Header text to potentially wrap
            num_columns: Number of columns in the table

        Returns:
            Tuple of (wrapped_text, number_of_lines)
        """
        text_wrapping_config = self.style_config.text_wrapping
        wrap_threshold = text_wrapping_config.table_header_wrap_threshold

        # Only wrap if header exceeds threshold
        if len(str(header_text)) > wrap_threshold:
            wrap_length = self.calculate_table_header_wrap_length(num_columns)
            wrapped_text = self.wrap_text(header_text, wrap_length)
            num_lines = wrapped_text.count('\n') + 1
        else:
            # Header is short enough - don't wrap
            wrapped_text = str(header_text)
            num_lines = 1

        return wrapped_text, num_lines

    def apply_background_effects(self, ax: Axes) -> None:
        """Apply background effects to the axes if enabled.

        Args:
            ax: Matplotlib axes object
        """
        chart_style = self.style_config.chart_style

        # Apply background gradient if enabled
        if chart_style.background_gradient:
            # Create a subtle gradient background
            import matplotlib.patches as patches
            from matplotlib.colors import LinearSegmentedColormap

            # Create a subtle gradient from white to very light gray
            colors = ['#ffffff', '#f8f8f8']
            cmap = LinearSegmentedColormap.from_list('background_gradient', colors, N=100)

            # Get the axes bounds
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # Create gradient rectangle
            rect = patches.Rectangle((xlim[0], ylim[0]),
                                   xlim[1] - xlim[0],
                                   ylim[1] - ylim[0],
                                   facecolor='none',
                                   edgecolor='none',
                                   alpha=chart_style.background_gradient_alpha,
                                   zorder=-1)

            # Apply gradient fill
            ax.add_patch(rect)
            ax.set_facecolor(cmap(0.5))  # Use middle color of gradient

            logger.debug("Applied background gradient effect")

        # Apply background pattern if enabled
        if chart_style.background_pattern != "none":
            if chart_style.background_pattern == "dots":
                # Add subtle dot pattern
                self._add_dot_pattern(ax)
            elif chart_style.background_pattern == "lines":
                # Add subtle line pattern
                self._add_line_pattern(ax)

            logger.debug(f"Applied background pattern: {chart_style.background_pattern}")

    def _add_dot_pattern(self, ax: Axes) -> None:
        """Add subtle dot pattern to background.

        Args:
            ax: Matplotlib axes object
        """
        import matplotlib.patches as patches

        # Get axes bounds
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Create dot pattern
        dot_size = 0.5
        dot_spacing = 2.0
        alpha = self.style_config.chart_style.background_gradient_alpha

        for x in np.arange(xlim[0], xlim[1], dot_spacing):
            for y in np.arange(ylim[0], ylim[1], dot_spacing):
                dot = patches.Circle((x, y), dot_size,
                                   facecolor='#cccccc',
                                   alpha=alpha,
                                   zorder=-1)
                ax.add_patch(dot)

    def _add_line_pattern(self, ax: Axes) -> None:
        """Add subtle line pattern to background.

        Args:
            ax: Matplotlib axes object
        """
        import matplotlib.patches as patches

        # Get axes bounds
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Create line pattern
        line_spacing = 5.0
        alpha = self.style_config.chart_style.background_gradient_alpha

        # Vertical lines
        for x in np.arange(xlim[0], xlim[1], line_spacing):
            line = patches.Rectangle((x, ylim[0]), 0.1, ylim[1] - ylim[0],
                                   facecolor='#cccccc',
                                   alpha=alpha,
                                   zorder=-1)
            ax.add_patch(line)

        # Horizontal lines
        for y in np.arange(ylim[0], ylim[1], line_spacing):
            line = patches.Rectangle((xlim[0], y), xlim[1] - xlim[0], 0.1,
                                   facecolor='#cccccc',
                                   alpha=alpha,
                                   zorder=-1)
            ax.add_patch(line)

    def estimate_max_command_line_length(self, ax=None) -> int:
        """Estimate the maximum number of monospace characters that fit in the available width.
        Args:
            ax: Optional matplotlib Axes. If not provided, uses default figure size and margins.
        Returns:
            Estimated number of characters per line.
        """
        # Get figure width in inches
        if ax is not None:
            fig = ax.get_figure()
            fig_width_in = fig.get_figwidth()
        else:
            fig_width_in, _ = self.get_figure_size()
        # Get subplot margins (fraction of figure)
        pdf_settings = self.style_config.pdf_settings
        left = pdf_settings.pdf_subplot_left
        right = pdf_settings.pdf_subplot_right
        # Available width in figure fraction
        available_fraction = right - left
        # Available width in inches
        available_width_in = fig_width_in * available_fraction
        # Estimate average monospace character width in points (1 pt = 1/72 in)
        # For monospace, 1 char ~ 0.6 * font size in points
        font_size_pt = self.style_config.chart_style.font_size
        char_width_in = (font_size_pt * 0.6) / 72.0
        # Number of characters that fit
        if char_width_in > 0:
            max_chars = int(available_width_in / char_width_in)
        else:
            max_chars = 80  # fallback
        # Clamp to a reasonable range
        return max(30, min(max_chars, 200))
