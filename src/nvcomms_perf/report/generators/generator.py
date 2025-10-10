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

"""Base report generator module that dispatches to specific format generators."""

import pandas as pd
from pathlib import Path
import logging
from typing import Literal

logger = logging.getLogger(__name__)


def generate_report(
    dataframe: pd.DataFrame,
    output_path: Path,
    pdf_type: str,
    format: Literal["pdf", "html"] = "pdf",
    **kwargs
) -> None:
    """Generate a report in the specified format.

    This is a dispatcher function that routes to the appropriate format-specific
    generator based on the format parameter.

    Args:
        dataframe: The DataFrame containing test results
        output_path: Path where the report should be saved
        pdf_type: Type of PDF report to generate (e.g., 'basic_report', 'comprehensive_report')
        format: The output format ('pdf' or 'html'). Defaults to 'pdf'
        **kwargs: Additional format-specific arguments
            For PDF:
                - plot_config_path: Optional path to custom plot configuration TOML file
                - style_config_path: Optional path to custom style configuration TOML file
                - include_empty_pages: Whether to include empty lined pages for notes
                - include_data_tables: Whether to include data tables after charts
                - include_test_commands: Whether to include test command pages

    Raises:
        ValueError: If an unsupported format is specified
    """
    logger.info(f"Generating {format.upper()} report of type: {pdf_type}")

    if format == "pdf":
        from .pdf import generate_pdf_report

        # Extract PDF-specific arguments
        include_empty_pages = kwargs.get('include_empty_pages', True)
        include_data_tables = kwargs.get('include_data_tables', True)
        include_test_commands = kwargs.get('include_test_commands', True)

        # Create adapter with configurations
        from ..adapters.nccl_adapter import NCCLPerformanceAdapter
        from ..models import ChartConfigLoader, StyleConfigLoader

        # Load configurations if provided
        plot_config = None
        style_config = None

        plot_config_path = kwargs.get('plot_config_path')
        if plot_config_path:
            plot_config = ChartConfigLoader.load_from_toml(plot_config_path)

        style_config_path = kwargs.get('style_config_path')
        if style_config_path:
            style_config = StyleConfigLoader.load_from_toml(style_config_path)

        # Create adapter
        adapter = NCCLPerformanceAdapter(plot_config=plot_config, style_config=style_config)

        generate_pdf_report(
            dataframe=dataframe,
            output_path=output_path,
            pdf_type=pdf_type,
            adapter=adapter,
            include_empty_pages=include_empty_pages,
            include_data_tables=include_data_tables,
            include_test_commands=include_test_commands
        )
    elif format == "html":
        # HTML support not yet implemented
        raise NotImplementedError("HTML report generation is not yet implemented")
    else:
        raise ValueError(f"Unsupported report format: {format}. Supported formats: 'pdf', 'html'")

    logger.info(f"{format.upper()} report generation completed")
