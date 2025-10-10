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
PDF report generator using matplotlib.

This module generates PDF reports from performance data using matplotlib
for charts and reportlab for PDF compilation.
"""

import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.figure


from ...models import StyleConfig
from ...adapters.nccl_adapter import NCCLPerformanceAdapter
from ...plotting import ChartGenerator, StyleManager

logger = logging.getLogger(__name__)





def _get_pdf_savefig_params(style_config: StyleConfig) -> Dict[str, Any]:
    """Get consistent savefig parameters from style configuration.

    Args:
        style_config: Style configuration object

    Returns:
        Dictionary of parameters for matplotlib savefig
    """
    pdf_settings = style_config.pdf_settings

    return {
        'dpi': pdf_settings.pdf_dpi,
        'facecolor': pdf_settings.pdf_facecolor,
        'edgecolor': pdf_settings.pdf_edgecolor,
        'bbox_inches': None,  # Use full figure dimensions
        'pad_inches': 0,  # No extra padding
        'metadata': {
            'Creator': 'nvcomms_perf',
            'Producer': 'matplotlib',
        }
    }


def _save_figure_to_pdf(pdf: PdfPages, fig: matplotlib.figure.Figure,
                       style_config: StyleConfig) -> None:
    """Save a figure to the PDF with consistent page settings.

    Args:
        pdf: PdfPages object
        fig: Matplotlib figure to save
        style_config: Style configuration object
    """
    # Skip None figures to avoid empty PDF deprecation warning
    if fig is None:
        return

    savefig_params = _get_pdf_savefig_params(style_config)
    pdf.savefig(fig, **savefig_params)
    plt.close(fig)


def generate_pdf_report(
    dataframe: pd.DataFrame,
    output_path: Path,
    pdf_type: str,
    adapter: NCCLPerformanceAdapter,
    include_empty_pages: bool = True,
    include_data_tables: bool = True,
    include_test_commands: bool = True
) -> None:
    """Generate a PDF report from a pandas DataFrame.

    Args:
        dataframe: The DataFrame containing test results
        output_path: Path where the PDF should be saved
        pdf_type: Type of PDF report to generate (e.g., 'basic_report', 'comprehensive_report')
        adapter: Pre-configured NCCLPerformanceAdapter with plot and style configurations
        include_empty_pages: Whether to include empty lined pages for notes
        include_data_tables: Whether to include data tables after charts
        include_test_commands: Whether to include test command pages
    """
    logger.info(f"Generating PDF report of type: {pdf_type}")
    logger.info(f"DataFrame shape: {dataframe.shape}")
    logger.info(f"Output path: {output_path}")

    # Save current matplotlib warning threshold
    import matplotlib
    original_max_open_warning = matplotlib.rcParams.get('figure.max_open_warning', 20)

    # Temporarily increase the threshold to avoid warnings during PDF generation
    # We properly close all figures, but some methods create multiple figures at once
    matplotlib.rcParams['figure.max_open_warning'] = 100

    # Initialize variables that will be used in finally block
    chart_generator = None
    style_manager = None

    try:
        # Get configurations from the adapter
        plot_config = adapter.plot_config
        style_config = adapter.style_config

        # Ensure configurations are loaded
        if not plot_config or not style_config:
            raise ValueError("Adapter must have both plot and style configurations loaded")

        # Initialize managers
        style_manager = StyleManager(style_config)
        chart_generator = ChartGenerator(adapter, style_manager)

        # Process the data
        adapter._raw_data = dataframe
        processed_data = adapter.process_data()

        logger.info(f"DEBUG: Processed data shape: {processed_data.shape if processed_data is not None else 'None'}")
        if processed_data is not None and not processed_data.empty:
            logger.info(f"DEBUG: Processed data columns: {list(processed_data.columns)[:10]}{'...' if len(processed_data.columns) > 10 else ''}")

        # Get PDF configuration
        pdf_config = plot_config.get_pdf_config_by_id(pdf_type) if plot_config.pdf_configs else None
        if not pdf_config:
            logger.warning(f"PDF configuration for type '{pdf_type}' not found")
            return

        logger.info(f"DEBUG: PDF config found, analysis_profiles: {pdf_config.analysis_profiles}")

        # Log PDF page settings
        pdf_settings = style_config.pdf_settings
        logger.info(f"PDF page settings: {pdf_settings.page_width}x{pdf_settings.page_height} inches, "
                    f"landscape={pdf_settings.use_landscape}, DPI={pdf_settings.pdf_dpi}")

        # Check if we have any content to generate
        has_content = False
        content_summary = []

        # Initialize chart number counter
        chart_number = 0

        # Check if we have analysis profiles to generate
        if pdf_config.analysis_profiles:
            logger.debug(f"Processing {len(pdf_config.analysis_profiles)} analysis profiles")
            for profile_name in pdf_config.analysis_profiles:
                if plot_config.analysis_profiles:
                    profile = plot_config.get_analysis_profile_by_id(profile_name)
                    if profile and profile.test_binaries:
                        # Check if we have matching binaries in the data
                        adapter_binaries = adapter.get_test_binaries()
                        expected_set = set(profile.test_binaries)
                        found_set = set(adapter_binaries)
                        matches = expected_set.intersection(found_set)

                        if matches:
                            logger.debug(f"Profile '{profile_name}': found {len(matches)} matching test binaries")
                            has_content = True
                            content_summary.append(f"Profile: {profile_name}")
                        else:
                            logger.warning(f"No matching test binaries for profile '{profile_name}' (expected: {expected_set}, found: {found_set})")
                else:
                    logger.warning("No analysis_profiles in plot_config")
        else:
            logger.warning("No analysis profiles in PDF config")

        # If no content, create a simple report with just the introduction
        if not has_content:
            logger.info("No chart content found - creating summary report with introduction only")

        # Open PDF writer early to write figures as they are generated
        with PdfPages(str(output_path)) as pdf:
            # 1. Add introduction page
            intro_fig = _create_introduction_page(pdf_config, processed_data, style_manager, adapter)
            _save_figure_to_pdf(pdf, intro_fig, style_config)

            # Add empty page after introduction if requested
            if include_empty_pages:
                empty_fig = style_manager.create_lined_page()
                _save_figure_to_pdf(pdf, empty_fig, style_config)

            # Only generate charts if we have content
            if has_content:

                # Lists to track figures that need to be generated after charts
                pending_data_tables = []

                # 2. Process each analysis profile
                for profile_name in pdf_config.analysis_profiles:
                    profile = plot_config.get_analysis_profile_by_id(profile_name)
                    if not profile:
                        logger.warning(f"Analysis profile '{profile_name}' not found in plot config")
                        continue
                    # For each test binary in the profile
                    for test_binary in profile.test_binaries:
                        # Check if this test binary exists in our data
                        adapter_binaries = adapter.get_test_binaries()
                        if test_binary not in adapter_binaries:
                            logger.warning(f"Test binary '{test_binary}' not found in data, skipping")
                            continue

                        # Generate charts for each chart type in the profile
                        for chart_type in profile.chart_types:
                            logger.debug(f"Generating {chart_type} chart for {test_binary}")
                            try:
                                # Get chart groups (data subsets) for this chart type
                                chart_groups = adapter.get_chart_groups_for_plotting(
                                    chart_type, test_binary
                                )

                                if not chart_groups:
                                    logger.warning(f"No data groups found for {chart_type} - {test_binary}")
                                    continue

                                # Generate a chart for each data group
                                for group_info in chart_groups:

                                    # Generate the chart
                                    chart_number += 1
                                    fig = chart_generator.generate_chart(
                                        chart_type,
                                        test_binary,
                                        group_info,
                                        chart_number
                                    )

                                    if fig:
                                        # Save to PDF
                                        _save_figure_to_pdf(pdf, fig, style_config)

                                        # Track for data tables if enabled
                                        if include_data_tables:
                                            pending_data_tables.append({
                                                'chart_type': chart_type,
                                                'data': group_info['data'],
                                                'test_binary': test_binary,
                                                'filters': group_info.get('filters', {}),
                                                'chart_number': chart_number
                                            })

                                        # Add empty page if requested
                                        if include_empty_pages:
                                            empty_fig = style_manager.create_lined_page()
                                            _save_figure_to_pdf(pdf, empty_fig, style_config)

                            except Exception as e:
                                logger.error(f"Error generating chart {chart_type} for {test_binary}: {e}")
                                logger.exception("Full traceback:")

                # Generate and save all data tables
                if include_data_tables and pending_data_tables:
                    logger.info(f"Generating and saving {len(pending_data_tables)} data tables")
                    for table_info in pending_data_tables:
                        try:
                            # Create chart_group structure expected by the functions
                            chart_group = {
                                'data': table_info['data'],
                                'filters': table_info['filters'],
                                'group_key': f"{table_info['chart_type']}_{table_info['test_binary']}"
                            }

                            table_figures = chart_generator.create_data_table(
                                table_info['chart_type'],
                                table_info['test_binary'],
                                chart_group,
                                table_info.get('chart_number')  # Use the tracked chart number
                            )
                            if table_figures:
                                for fig in table_figures:
                                    _save_figure_to_pdf(pdf, fig, style_config)
                            else:
                                logger.debug(f"Skipping empty data table for {chart_group['group_key']}")
                        except Exception as e:
                            logger.error(f"Failed to generate data table for {table_info['chart_type']}_{table_info['test_binary']}: {e}")

                # Generate and save all command pages
                if include_test_commands and pending_data_tables:
                    logger.info("Generating and saving command pages")
                    for cmd_info in pending_data_tables:
                        try:
                            # Create chart_group structure expected by the functions
                            chart_group = {
                                'data': cmd_info['data'],
                                'filters': cmd_info['filters'],
                                'group_key': f"{cmd_info['chart_type']}_{cmd_info['test_binary']}"
                            }

                            command_figures = chart_generator.create_command_pages_for_chart(
                                cmd_info['chart_type'],
                                cmd_info['test_binary'],
                                chart_group,
                                cmd_info.get('chart_number', 0)  # Use the tracked chart number
                            )
                            if command_figures:
                                for fig in command_figures:
                                    _save_figure_to_pdf(pdf, fig, style_config)
                            else:
                                logger.debug(f"No command pages created for {chart_group['group_key']}")
                        except Exception as e:
                            logger.error(f"Failed to create command pages for {cmd_info['chart_type']}_{cmd_info['test_binary']}: {e}")

        logger.info(f"PDF report saved to: {output_path} ({chart_number} charts generated)")

    finally:
        # Ensure cleanup happens even if an error occurs
        try:
            if chart_generator:
                chart_generator.cleanup()
            if style_manager:
                style_manager.close_all_figures()
            logger.info("PDF generation cleanup completed")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
        # Restore original matplotlib warning threshold
        matplotlib.rcParams['figure.max_open_warning'] = original_max_open_warning





def _create_introduction_page(pdf_config: Any, dataframe: pd.DataFrame,
                             style_manager: StyleManager, adapter: NCCLPerformanceAdapter) -> matplotlib.figure.Figure:
    """Create an introduction page with comprehensive metadata.

    Args:
        pdf_config: PDF configuration
        dataframe: Processed DataFrame
        style_manager: Style manager
        adapter: NCCL adapter for collecting metadata

    Returns:
        Matplotlib figure with introduction page
    """
    # Create figure using the style manager for consistent sizing
    fig = plt.figure(figsize=style_manager.get_figure_size())
    ax = fig.add_subplot(111)

    # Remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Apply subplot adjustments from PDF settings
    pdf_settings = style_manager.style_config.pdf_settings
    fig.subplots_adjust(
        left=pdf_settings.pdf_subplot_left,
        right=pdf_settings.pdf_subplot_right,
        top=pdf_settings.pdf_subplot_top,
        bottom=pdf_settings.pdf_subplot_bottom
    )

    # Add title
    title = pdf_config.name
    ax.text(0.5, 0.95, title, transform=ax.transAxes,
           fontsize=style_manager.style_config.chart_style.font_size +
                    style_manager.style_config.font_adjustments.title_large,
           fontweight='bold', ha='center', va='top')

    # Add description
    description = pdf_config.description
    ax.text(0.5, 0.85, description, transform=ax.transAxes,
           fontsize=style_manager.style_config.chart_style.font_size +
                    style_manager.style_config.font_adjustments.title_medium,
           ha='center', va='top')

    # Collect comprehensive metadata
    metadata = adapter.collect_dataset_metadata()

    # Set up 2-column layout
    base_font_size = style_manager.style_config.chart_style.font_size
    left_col_x = 0.05   # Left column start
    right_col_x = 0.55  # Right column start

    # Starting positions for both columns
    left_y_pos = 0.78
    right_y_pos = 0.78

    # LEFT COLUMN - Dataset Overview and Test Configuration

    # Dataset Overview section
    if 'basic_stats' in metadata:
        basic_stats = metadata['basic_stats']

        # Section header
        ax.text(left_col_x, left_y_pos, "Dataset Overview", transform=ax.transAxes,
               fontsize=base_font_size + 2, fontweight='bold', ha='left', va='top')
        left_y_pos -= 0.04

        # Basic statistics
        overview_items = [
            f"Total Records: {basic_stats.get('total_records', 'Unknown'):,}",
            f"Data Columns: {basic_stats.get('total_columns', 'Unknown')}",
            f"Memory Usage: {basic_stats.get('memory_usage_mb', 0):.1f} MB",
            f"Analysis Profiles: {', '.join(pdf_config.analysis_profiles)}",
        ]

        for item in overview_items:
            ax.text(left_col_x + 0.03, left_y_pos, item, transform=ax.transAxes,
                   fontsize=base_font_size, ha='left', va='top')
            left_y_pos -= 0.03

        left_y_pos -= 0.02  # Section spacing

    # Test Configuration section
    if 'test_info' in metadata:
        test_info = metadata['test_info']

        # Section header
        ax.text(left_col_x, left_y_pos, "Test Configuration", transform=ax.transAxes,
               fontsize=base_font_size + 2, fontweight='bold', ha='left', va='top')
        left_y_pos -= 0.04

        # Test binaries
        if 'test_binaries' in test_info:
            binaries = test_info['test_binaries']
            if len(binaries) <= 3:
                ax.text(left_col_x + 0.03, left_y_pos, f"Test Binaries: {', '.join(binaries)}", transform=ax.transAxes,
                       fontsize=base_font_size, ha='left', va='top')
            else:
                ax.text(left_col_x + 0.03, left_y_pos, f"Test Binaries: {', '.join(binaries[:3])}... ({len(binaries)} total)",
                       transform=ax.transAxes, fontsize=base_font_size, ha='left', va='top')
            left_y_pos -= 0.03

        # Scale information
        if 'scale_info' in test_info:
            scale_info = test_info['scale_info']

            if 'node_range' in scale_info:
                node_range = scale_info['node_range']
                ax.text(left_col_x + 0.03, left_y_pos, f"Node Range: {node_range.get('min', 'Unknown')}-{node_range.get('max', 'Unknown')} nodes",
                       transform=ax.transAxes, fontsize=base_font_size, ha='left', va='top')
                left_y_pos -= 0.03

            if 'process_range' in scale_info:
                process_range = scale_info['process_range']
                ax.text(left_col_x + 0.03, left_y_pos, f"Process Range: {process_range.get('min', 'Unknown')}-{process_range.get('max', 'Unknown')} processes",
                       transform=ax.transAxes, fontsize=base_font_size, ha='left', va='top')
                left_y_pos -= 0.03

            if 'message_size_range' in scale_info:
                size_range = scale_info['message_size_range']
                ax.text(left_col_x + 0.03, left_y_pos, f"Message Size Range: {size_range.get('min_human', 'Unknown')}-{size_range.get('max_human', 'Unknown')}",
                       transform=ax.transAxes, fontsize=base_font_size, ha='left', va='top')
                left_y_pos -= 0.03

    # RIGHT COLUMN - System Information and Performance Metrics

    # System Information section
    if 'system_info' in metadata:
        system_info = metadata['system_info']

        # Section header
        ax.text(right_col_x, right_y_pos, "System Information", transform=ax.transAxes,
               fontsize=base_font_size + 2, fontweight='bold', ha='left', va='top',
               color='black')  # Ensure text is black and visible
        right_y_pos -= 0.04

        # Systems
        if 'systems' in system_info:
            systems = system_info['systems']
            if len(systems) <= 3:
                text = f"Systems: {', '.join(systems)}"
                ax.text(right_col_x + 0.03, right_y_pos, text, transform=ax.transAxes,
                       fontsize=base_font_size, ha='left', va='top')
            else:
                text = f"Systems: {', '.join(systems[:3])}... ({len(systems)} total)"
                ax.text(right_col_x + 0.03, right_y_pos, text,
                       transform=ax.transAxes, fontsize=base_font_size, ha='left', va='top')
            right_y_pos -= 0.03

        # GPU architectures
        if 'gpu_architectures' in system_info:
            gpu_archs = system_info['gpu_architectures']
            text = f"GPU Architectures: {', '.join(gpu_archs)}"
            ax.text(right_col_x + 0.03, right_y_pos, text, transform=ax.transAxes,
                   fontsize=base_font_size, ha='left', va='top')
            right_y_pos -= 0.03

        # Software versions
        if 'nccl_versions' in system_info:
            nccl_versions = system_info['nccl_versions']
            text = f"NCCL Versions: {', '.join(nccl_versions)}"
            ax.text(right_col_x + 0.03, right_y_pos, text, transform=ax.transAxes,
                   fontsize=base_font_size, ha='left', va='top')
            right_y_pos -= 0.03

        if 'cuda_versions' in system_info:
            cuda_versions = system_info['cuda_versions']
            text = f"CUDA Versions: {', '.join(str(v) for v in cuda_versions)}"
            ax.text(right_col_x + 0.03, right_y_pos, text, transform=ax.transAxes,
                   fontsize=base_font_size, ha='left', va='top')
            right_y_pos -= 0.03

        right_y_pos -= 0.02  # Section spacing

    # Data coverage section (if space allows)
    if 'data_coverage' in metadata and right_y_pos > 0.25:
        coverage = metadata['data_coverage']

        # Section header
        ax.text(right_col_x, right_y_pos, "Data Coverage", transform=ax.transAxes,
               fontsize=base_font_size + 2, fontweight='bold', ha='left', va='top')
        right_y_pos -= 0.04

        # Show key NCCL parameters
        key_params = ['NCCL_ALGO', 'NCCL_PROTO', 'local_register']
        for param in key_params:
            if param in coverage and right_y_pos > 0.15:
                param_info = coverage[param]
                unique_count = param_info.get('unique_values', 0)
                ax.text(right_col_x + 0.03, right_y_pos, f"{param}: {unique_count} unique values",
                       transform=ax.transAxes, fontsize=base_font_size, ha='left', va='top')
                right_y_pos -= 0.03

    # Add generation timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.text(0.5, 0.05, f"Generated: {timestamp}", transform=ax.transAxes,
           fontsize=style_manager.style_config.chart_style.font_size +
                    style_manager.style_config.font_adjustments.label_small,
           ha='center', va='bottom', alpha=0.7)

    return fig
