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
API module for direct report generation.

This module provides clean interfaces for users to make direct API calls
to generate reports without using the CLI interface.
"""

import pandas as pd
import io
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any, IO
import tempfile

from .models import ChartConfigLoader, StyleConfigLoader
from .adapters.nccl_adapter import NCCLPerformanceAdapter
from .generators.pdf import generate_pdf_report

logger = logging.getLogger(__name__)

class ReportAPI:
    """API class for generating performance reports directly from data."""

    def __init__(self):
        self._default_plot_config = None
        self._default_style_config = None

    @staticmethod
    def _parse_dataframe_from_input(dataframe_input: Union[str, bytes, bytearray, memoryview, IO]) -> pd.DataFrame:
        """Parse DataFrame from various input types.

        Args:
            dataframe_input: Input data as string, bytes, or file-like object

        Returns:
            Parsed pandas DataFrame

        Raises:
            TypeError: If input type is not supported
        """
        if isinstance(dataframe_input, str):
            return pd.read_csv(io.StringIO(dataframe_input))
        elif isinstance(dataframe_input, (bytes, bytearray, memoryview)):
            return pd.read_csv(io.StringIO(bytes(dataframe_input).decode()))
        elif hasattr(dataframe_input, 'read'):
            content = dataframe_input.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            return pd.read_csv(io.StringIO(content))
        else:
            raise TypeError("Unsupported input type for DataFrame parsing; must be str, bytes, bytearray, memoryview, or file-like object.")

    @staticmethod
    def _parse_config_from_input(config_input: Union[str, bytes, bytearray, memoryview, IO], config_type: str) -> Any:
        """Parse configuration from various input types.

        Args:
            config_input: Configuration input as string, bytes, or file-like object
            config_type: Type of configuration ('plot' or 'style')

        Returns:
            Parsed configuration object

        Raises:
            TypeError: If input type is not supported
            ValueError: If config_type is unknown
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as temp_file:
            if isinstance(config_input, str):
                temp_file.write(config_input)
            elif isinstance(config_input, (bytes, bytearray, memoryview)):
                temp_file.write(bytes(config_input).decode())
            elif hasattr(config_input, 'read'):
                content = config_input.read()
                if isinstance(content, bytes):
                    content = content.decode("utf-8", errors="replace")
                temp_file.write(content)
            else:
                raise TypeError("Unsupported input type for config parsing; must be str, bytes, bytearray, memoryview, or file-like object.")
            temp_file_path = temp_file.name
        try:
            if config_type == 'plot':
                return ChartConfigLoader.load_from_toml(temp_file_path)
            elif config_type == 'style':
                return StyleConfigLoader.load_from_toml(temp_file_path)
            else:
                raise ValueError(f"Unknown config type: {config_type}")
        finally:
            Path(temp_file_path).unlink(missing_ok=True)

    def _load_default_configs(self):
        """Load default configurations if not already loaded."""
        if self._default_plot_config is None:
            from .cli.report import _get_default_config_path
            default_plot_path = _get_default_config_path('plot')
            self._default_plot_config = ChartConfigLoader.load_from_toml(str(default_plot_path))
        if self._default_style_config is None:
            from .cli.report import _get_default_config_path
            default_style_path = _get_default_config_path('style')
            self._default_style_config = StyleConfigLoader.load_from_toml(str(default_style_path))

    def generate_pdf_report_from_input(
        self,
        dataframe_input: Union[str, bytes, bytearray, memoryview, IO],
        output_path: Union[str, Path],
        pdf_type: str = "basic_report",
        plot_config_input: Optional[Union[str, bytes, bytearray, memoryview, IO]] = None,
        style_config_input: Optional[Union[str, bytes, bytearray, memoryview, IO]] = None,
        include_empty_pages: bool = True,
        include_data_tables: bool = True,
        include_test_commands: bool = True
    ) -> None:
        """Generate PDF report from various input types.

        Args:
            dataframe_input: DataFrame input as string, bytes, or file-like object
            output_path: Path where the PDF should be saved
            pdf_type: Type of PDF report to generate
            plot_config_input: Optional plot configuration input
            style_config_input: Optional style configuration input
            include_empty_pages: Whether to include empty lined pages
            include_data_tables: Whether to include data tables
            include_test_commands: Whether to include test command pages

        Raises:
            Exception: If PDF generation fails
        """
        logger.info("Generating PDF report from DataFrame input")
        logger.info(f"Output path: {output_path}")
        logger.info(f"PDF type: {pdf_type}")

        try:
            dataframe = self._parse_dataframe_from_input(dataframe_input)

            if plot_config_input:
                plot_config = self._parse_config_from_input(plot_config_input, 'plot')
            else:
                self._load_default_configs()
                plot_config = self._default_plot_config

            if style_config_input:
                style_config = self._parse_config_from_input(style_config_input, 'style')
            else:
                self._load_default_configs()
                style_config = self._default_style_config

            adapter = NCCLPerformanceAdapter(plot_config=plot_config, style_config=style_config)
            output_path = Path(output_path)

            generate_pdf_report(
                dataframe=dataframe,
                output_path=output_path,
                pdf_type=pdf_type,
                adapter=adapter,
                include_empty_pages=include_empty_pages,
                include_data_tables=include_data_tables,
                include_test_commands=include_test_commands
            )
            logger.info(f"Successfully generated PDF report: {output_path}")
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
            raise

    def validate_dataframe_input(self, dataframe_input: Union[str, bytes, bytearray, memoryview, IO]) -> Dict[str, Any]:
        """Validate DataFrame input and return metadata.

        Args:
            dataframe_input: DataFrame input to validate

        Returns:
            Dictionary with validation results and metadata
        """
        try:
            df = self._parse_dataframe_from_input(dataframe_input)
            return {
                'valid': True,
                'shape': df.shape,
                'columns': list(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'dtypes': df.dtypes.to_dict()
            }
        except Exception as e:
            return {'valid': False, 'error': str(e)}

    def validate_config_input(self, config_input: Union[str, bytes, bytearray, memoryview, IO], config_type: str) -> Dict[str, Any]:
        """Validate configuration input and return metadata.

        Args:
            config_input: Configuration input to validate
            config_type: Type of configuration ('plot' or 'style')

        Returns:
            Dictionary with validation results and metadata
        """
        try:
            config_obj = self._parse_config_from_input(config_input, config_type)
            if config_type == 'plot':
                metadata = {
                    'valid': True,
                    'config_type': 'plot',
                    'analysis_profiles': [profile.id for profile in config_obj.analysis_profiles] if config_obj.analysis_profiles else [],
                    'pdf_configs': [config.id for config in config_obj.pdf_configs] if config_obj.pdf_configs else []
                }
            elif config_type == 'style':
                metadata = {
                    'valid': True,
                    'config_type': 'style',
                    'pdf_settings': {
                        'page_width': config_obj.pdf_settings.page_width,
                        'page_height': config_obj.pdf_settings.page_height,
                        'use_landscape': config_obj.pdf_settings.use_landscape,
                        'pdf_dpi': config_obj.pdf_settings.pdf_dpi
                    } if config_obj.pdf_settings else {}
                }
            else:
                metadata = {
                    'valid': True,
                    'config_type': config_type
                }
            return metadata
        except Exception as e:
            return {'valid': False, 'config_type': config_type, 'error': str(e)}

    @classmethod
    def read_csv(cls, obj: Union[str, bytes, bytearray, memoryview, IO]) -> pd.DataFrame:
        """Parse a DataFrame from CSV content.

        Args:
            obj: CSV content as string, bytes, or file-like object

        Returns:
            Parsed pandas DataFrame

        Raises:
            TypeError: If input type is not supported
        """
        instance = cls()
        return instance._parse_dataframe_from_input(obj)

    @classmethod
    def read_plot_config(cls, obj: Union[str, bytes, bytearray, memoryview, IO]) -> Any:
        """Parse a plot configuration from TOML content.

        Args:
            obj: TOML content as string, bytes, or file-like object

        Returns:
            Parsed plot configuration object

        Raises:
            TypeError: If input type is not supported
        """
        instance = cls()
        return instance._parse_config_from_input(obj, 'plot')

    @classmethod
    def read_style_config(cls, obj: Union[str, bytes, bytearray, memoryview, IO]) -> Any:
        """Parse a style configuration from TOML content.

        Args:
            obj: TOML content as string, bytes, or file-like object

        Returns:
            Parsed style configuration object

        Raises:
            TypeError: If input type is not supported
        """
        instance = cls()
        return instance._parse_config_from_input(obj, 'style')

# Convenience module-level functions

def generate_pdf_report_from_input(
    dataframe_input: Union[str, bytes, bytearray, memoryview, IO],
    output_path: Union[str, Path],
    pdf_type: str = "basic_report",
    plot_config_input: Optional[Union[str, bytes, bytearray, memoryview, IO]] = None,
    style_config_input: Optional[Union[str, bytes, bytearray, memoryview, IO]] = None,
    **kwargs
) -> None:
    """Generate PDF report from various input types (convenience function)."""
    api = ReportAPI()
    return api.generate_pdf_report_from_input(
        dataframe_input=dataframe_input,
        output_path=output_path,
        pdf_type=pdf_type,
        plot_config_input=plot_config_input,
        style_config_input=style_config_input,
        **kwargs
    )

def validate_dataframe_input(dataframe_input: Union[str, bytes, bytearray, memoryview, IO]) -> Dict[str, Any]:
    """Validate DataFrame input (convenience function)."""
    api = ReportAPI()
    return api.validate_dataframe_input(dataframe_input)

def validate_config_input(config_input: Union[str, bytes, bytearray, memoryview, IO], config_type: str) -> Dict[str, Any]:
    """Validate configuration input (convenience function)."""
    api = ReportAPI()
    return api.validate_config_input(config_input, config_type)

# Existing string-based APIs for backward compatibility

def generate_pdf_report_from_dataframe_string(
    dataframe_str: str,
    output_path: Union[str, Path],
    pdf_type: str = "basic_report",
    plot_config_str: Optional[str] = None,
    style_config_str: Optional[str] = None,
    **kwargs
) -> None:
    """Generate PDF report from DataFrame string (backward compatibility)."""
    return generate_pdf_report_from_input(
        dataframe_input=dataframe_str,
        output_path=output_path,
        pdf_type=pdf_type,
        plot_config_input=plot_config_str,
        style_config_input=style_config_str,
        **kwargs
    )

def validate_dataframe_string(dataframe_str: str) -> Dict[str, Any]:
    """Validate DataFrame string (backward compatibility)."""
    return validate_dataframe_input(dataframe_str)

def validate_config_string(config_str: str, config_type: str) -> Dict[str, Any]:
    """Validate configuration string (backward compatibility)."""
    return validate_config_input(config_str, config_type)

# Existing DataFrame object API for backward compatibility
def generate_pdf_report_from_dataframe_object(
    dataframe: pd.DataFrame,
    output_path: Union[str, Path],
    pdf_type: str = "basic_report",
    plot_config_str: Optional[str] = None,
    style_config_str: Optional[str] = None,
    include_empty_pages: bool = True,
    include_data_tables: bool = True,
    include_test_commands: bool = True
) -> None:
    """Generate PDF report from DataFrame object (backward compatibility)."""
    logger.info("Generating PDF report from DataFrame object")
    logger.info(f"DataFrame shape: {dataframe.shape}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"PDF type: {pdf_type}")

    api = ReportAPI()
    try:
        if plot_config_str:
            plot_config = api._parse_config_from_input(plot_config_str, 'plot')
        else:
            api._load_default_configs()
            plot_config = api._default_plot_config

        if style_config_str:
            style_config = api._parse_config_from_input(style_config_str, 'style')
        else:
            api._load_default_configs()
            style_config = api._default_style_config

        adapter = NCCLPerformanceAdapter(plot_config=plot_config, style_config=style_config)
        output_path = Path(output_path)

        generate_pdf_report(
            dataframe=dataframe,
            output_path=output_path,
            pdf_type=pdf_type,
            adapter=adapter,
            include_empty_pages=include_empty_pages,
            include_data_tables=include_data_tables,
            include_test_commands=include_test_commands
        )
        logger.info(f"Successfully generated PDF report: {output_path}")
    except Exception as e:
        logger.error(f"Failed to generate PDF report: {e}")
        raise
