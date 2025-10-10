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
NCCL adapter for processing NCCL performance test data.

This module provides the NCCL-specific implementation of the base adapter
for processing NCCL test results into formats suitable for plotting.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Set, Union
import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .base_adapter import BasePerformanceAdapter


logger = logging.getLogger(__name__)


class NCCLPerformanceAdapter(BasePerformanceAdapter):
    """Adapter for processing NCCL performance test data.

    This class implements the NCCL-specific logic for processing raw NCCL test
    results (from nccl-tests) into structured DataFrames suitable for creating
    performance plots.
    """

    # Legend parameter shortening lookup table
    _LEGEND_PARAM_SHORTENING = {
        'num_nodes': 'N',
        'num_processes': 'NP',
        'NCCL_TESTS_SPLIT_MASK': 'MASK',
        'local_register': 'UB',
        'NCCL_ALGO': 'ALGO',
        'NCCL_PROTO': 'PROTO',
        'NCCL_MIN_CTAS': 'CTAS',
    }

    # Reverse mapping for shortened parameter names
    _SHORTENED_TO_FULL_PARAM = {
        'N': 'num_nodes',
        'NP': 'num_processes',
        'MASK': 'NCCL_TESTS_SPLIT_MASK',
        'UB': 'local_register',
        'ALGO': 'NCCL_ALGO',
        'PROTO': 'NCCL_PROTO',
        'CTAS': 'NCCL_MIN_CTAS',
    }

    # Base column mappings for NCCL data - these are the possible columns
    # The actual columns used will be determined dynamically based on configuration
    _BASE_NCCL_COLUMNS = {
        'test_binary': 'tool_binary',  # Maps to 'tool_binary' column in CSV
        'size': 'test_result.size',
        'out_of_place_bus_bw': 'test_result.out_of_place.bus_bw',
        'out_of_place_alg_bw': 'test_result.out_of_place.alg_bw',
        'in_place_bus_bw': 'test_result.in_place.bus_bw',
        'in_place_alg_bw': 'test_result.in_place.alg_bw',
        'out_of_place_bus_bw_max': 'test_result.out_of_place.bus_bw_max',
        'out_of_place_alg_bw_max': 'test_result.out_of_place.alg_bw_max',
        'in_place_bus_bw_max': 'test_result.in_place.bus_bw_max',
        'in_place_alg_bw_max': 'test_result.in_place.alg_bw_max',
        'num_nodes': 'num_nodes',
        'num_processes': 'num_processes',
        'local_register': 'test_config.local_register',
        'nccl_tests_split_mask': 'test_env_vars.NCCL_TESTS_SPLIT_MASK',
        'library_version': 'library_version',
        'cuda_version': 'library_info.cuda_runtime_version',
        'system_name': 'system_info.system_name',
        'gpu_arch': 'system_info.gpu_arch_type',
        'test_cmd': 'test_cmd',
    }


    def __init__(self, plot_config: Optional[Any] = None, style_config: Optional[Any] = None):
        """Initialize the NCCL adapter.

        Args:
            plot_config: Optional plot configuration object
            style_config: Optional style configuration object
        """
        super().__init__(plot_config)
        self.style_config = style_config
        self._test_binaries: Optional[Set[str]] = None
        self._required_columns: Optional[Set[str]] = None
        self._active_column_mapping: Optional[Dict[str, str]] = None
        self._color_palette_cache: Optional[List[str]] = None

    def shorten_legend_key(self, legend_key: str) -> str:
        """Shorten a legend key by replacing long parameter names with shorter ones.

        Args:
            legend_key: Original legend key (e.g., "num_nodes=4, num_processes=8, NCCL_ALGO=RING")

        Returns:
            Shortened legend key (e.g., "N=4, NP=8, ALGO=RING")
        """
        if not legend_key:
            return legend_key

        shortened_key = legend_key

        # Replace specific parameter names from lookup table
        for long_name, short_name in self._LEGEND_PARAM_SHORTENING.items():
            # Use regex to match parameter names at word boundaries
            pattern = rf'\b{re.escape(long_name)}\b'
            shortened_key = re.sub(pattern, short_name, shortened_key)

        # Remove "NCCL_" prefix from any remaining NCCL_* parameters
        shortened_key = re.sub(r'\bNCCL_', '', shortened_key)

        logger.debug(f"Shortened legend key: '{legend_key}' -> '{shortened_key}'")
        return shortened_key

    def get_required_columns_for_pdf_type(self, pdf_type: str) -> Set[str]:
        """Get all columns required for a specific PDF report type.

        Args:
            pdf_type: Type of PDF report (e.g., 'basic_report', 'comprehensive_report')

        Returns:
            Set of column names required for the PDF type
        """
        if not self.plot_config:
            logger.warning("No plot configuration provided, using all available columns")
            return set(self._BASE_NCCL_COLUMNS.values())

        required_columns = set()

        # Get PDF configuration
        pdf_config = self.plot_config.get_pdf_config_by_id(pdf_type) if self.plot_config.pdf_configs else None
        if not pdf_config:
            logger.warning(f"PDF type '{pdf_type}' not found in configuration")
            return set(self._BASE_NCCL_COLUMNS.values())

        # Get analysis profiles for this PDF type
        analysis_profiles = pdf_config.analysis_profiles
        logger.info(f"PDF type '{pdf_type}' requires analysis profiles: {analysis_profiles}")

        # For each analysis profile, get the required columns
        for profile_name in analysis_profiles:
            profile_columns = self._get_columns_for_analysis_profile(profile_name)
            required_columns.update(profile_columns)

        # Always include essential columns
        essential_columns = {
            self._BASE_NCCL_COLUMNS['test_binary'],
            self._BASE_NCCL_COLUMNS['size'],
            self._BASE_NCCL_COLUMNS['num_nodes'],
            self._BASE_NCCL_COLUMNS['num_processes'],
        }
        required_columns.update(essential_columns)

        # Always include test_cmd column since it might be needed for command pages
        # This ensures the column is loaded even if not explicitly referenced in chart configs
        test_cmd_column = self._BASE_NCCL_COLUMNS['test_cmd']
        required_columns.add(test_cmd_column)

        # Always include system information columns for metadata collection
        system_info_columns = {
            self._BASE_NCCL_COLUMNS['system_name'],
            self._BASE_NCCL_COLUMNS['gpu_arch'],
            self._BASE_NCCL_COLUMNS['library_version'],
            self._BASE_NCCL_COLUMNS['cuda_version'],
        }
        required_columns.update(system_info_columns)

        logger.info(f"Total required columns for PDF type '{pdf_type}': {len(required_columns)}")
        return required_columns

    def _get_columns_for_analysis_profile(self, profile_name: str) -> Set[str]:
        """Get columns required for an analysis profile.

        Args:
            profile_name: Name of the analysis profile

        Returns:
            Set of column names required for the profile
        """
        if not self.plot_config or not self.plot_config.analysis_profiles:
            return set()

        profile = self.plot_config.get_analysis_profile_by_id(profile_name)
        if not profile:
            logger.warning(f"Analysis profile '{profile_name}' not found in configuration")
            return set()

        required_columns = set()

        # Get columns from each chart type in the profile
        for chart_type in profile.chart_types:
            chart_columns = self._get_columns_for_chart_type(chart_type)
            required_columns.update(chart_columns)

        logger.debug(f"Analysis profile '{profile_name}' requires {len(required_columns)} columns")
        return required_columns

    def _get_columns_for_chart_type(self, chart_type: str) -> Set[str]:
        """Get columns required for a chart type.

        Args:
            chart_type: Name of the chart type

        Returns:
            Set of column names required for the chart type
        """
        if not self.plot_config or not self.plot_config.chart:
            return set()

        chart_config = self.plot_config.get_chart_by_id(chart_type)
        if not chart_config:
            logger.warning(f"Chart type '{chart_type}' not found in configuration")
            return set()

        required_columns = set()

        # Get performance columns specified in chart type
        if chart_config.performance_columns:
            for perf_col in chart_config.performance_columns:
                # Check if this is a derived column from config
                if self._is_derived_performance_column(perf_col):
                    # Get source columns for derived column
                    source_cols = self._get_source_columns_for_derived_column(perf_col)
                    required_columns.update(source_cols)
                else:
                    # Map to actual column name
                    actual_col = self.get_dataframe_column_name(perf_col)
                    required_columns.add(actual_col)

        # Get columns from fixed parameters
        if chart_config.fixed_params:
            for param in chart_config.fixed_params.keys():
                try:
                    actual_col = self.get_dataframe_column_name(param)
                    required_columns.add(actual_col)
                except ValueError:
                    logger.warning(f"Fixed parameter '{param}' not found in available columns.")

        # Get columns from split parameters
        if chart_config.chart_split_params:
            for param in chart_config.chart_split_params.keys():
                try:
                    actual_col = self.get_dataframe_column_name(param)
                    required_columns.add(actual_col)
                except ValueError:
                    logger.warning(f"Split parameter '{param}' not found in available columns.")

        # Get columns from combine parameters
        if chart_config.chart_combine_params:
            for param in chart_config.chart_combine_params.keys():
                try:
                    actual_col = self.get_dataframe_column_name(param)
                    required_columns.add(actual_col)
                except ValueError:
                    logger.warning(f"Combine parameter '{param}' not found in available columns.")

        logger.debug(f"Chart type '{chart_type}' requires {len(required_columns)} columns")
        return required_columns

    def _is_derived_performance_column(self, col_name: str) -> bool:
        """Check if a column is a derived performance column from configuration.

        Args:
            col_name: Column name to check

        Returns:
            True if column is derived from configuration
        """
        if not self.plot_config or not self.plot_config.settings:
            return False

        settings = self.plot_config.settings
        if not settings.data_processing or not settings.data_processing.performance_columns:
            return False

        return col_name in settings.data_processing.performance_columns

    def _get_source_columns_for_derived_column(self, col_name: str) -> Set[str]:
        """Get source columns for a derived performance column.

        Args:
            col_name: Name of the derived column

        Returns:
            Set of source column names
        """
        if not self.plot_config or not self.plot_config.settings:
            return set()

        settings = self.plot_config.settings
        if not settings.data_processing or not settings.data_processing.performance_columns:
            return set()

        perf_config = settings.data_processing.performance_columns.get(col_name)
        if not perf_config or not perf_config.sources:
            return set()

        # Map source column names to actual dataframe column names
        source_columns = set()
        for source in perf_config.sources:
            actual_col = self.get_dataframe_column_name(source)
            source_columns.add(actual_col)

        return source_columns

    def get_required_test_binaries_for_pdf_type(self, pdf_type: str) -> Set[str]:
        """Get test binaries required for a specific PDF report type.

        Args:
            pdf_type: Type of PDF report

        Returns:
            Set of test binary names required for the PDF type
        """
        if not self.plot_config:
            logger.warning("No plot configuration provided")
            return set()

        required_binaries = set()

        # Get PDF configuration
        pdf_config = self.plot_config.get_pdf_config_by_id(pdf_type) if self.plot_config.pdf_configs else None
        if not pdf_config:
            logger.warning(f"PDF type '{pdf_type}' not found in configuration.")
            return set()

        # Get analysis profiles for this PDF type
        analysis_profiles = pdf_config.analysis_profiles

        # For each analysis profile, get the required test binaries
        for profile_name in analysis_profiles:
            profile = self.plot_config.get_analysis_profile_by_id(profile_name)
            if profile:
                required_binaries.update(profile.test_binaries)

        logger.info(f"PDF type '{pdf_type}' requires test binaries: {sorted(required_binaries)}")
        return required_binaries

    def get_dataframe_column_name(self, config_param: str, data_source: Optional[Any] = None) -> str:
        """
        Get the actual dataframe column name for a configuration parameter.

        Args:
            config_param: Parameter name from configuration
            data_source: Optional dataframe or series to check which column exists

        Returns:
            Actual column name in dataframe

        Raises:
            ValueError: If the parameter column is not found in the data source
        """
        # First try direct mapping from base columns
        if config_param in self._BASE_NCCL_COLUMNS:
            return self._BASE_NCCL_COLUMNS[config_param]

        # If data source is provided, check for exact matches only
        if data_source is not None:
            if hasattr(data_source, 'columns'):  # DataFrame
                available_columns = data_source.columns
            elif hasattr(data_source, 'index'):  # Series
                available_columns = data_source.index
            else:
                available_columns = []

            # Try exact match first
            if config_param in available_columns:
                return config_param

            # Handle NCCL environment variables
            if config_param.startswith('NCCL_'):
                env_var_col = f'test_env_vars.{config_param}'
                if env_var_col in available_columns:
                    return env_var_col
                else:
                    # If we reach here, the NCCL environment variable column was not found
                    raise ValueError(f"Configuration parameter '{config_param}' not found in available columns: {list(available_columns)}")

            # If we reach here, config_param was not found in available_columns
            raise ValueError(f"Configuration parameter '{config_param}' not found in available columns: {list(available_columns)}")

        # Handle NCCL environment variables when no data source is provided
        if config_param.startswith('NCCL_'):
            env_var_col = f'test_env_vars.{config_param}'
            return env_var_col  # Return constructed name for NCCL environment variables

        # Return as-is if no mapping found and no data source provided
        return config_param

    def load_csv_with_required_columns(self, file_path: Union[str, Path], pdf_type: str) -> pd.DataFrame:
        """Load CSV data with only the columns required for the specified PDF type.

        Args:
            file_path: Path to the CSV file
            pdf_type: Type of PDF report to determine required columns

        Returns:
            DataFrame with only the required columns loaded
        """
        # Get required columns for this PDF type
        required_columns = self.get_required_columns_for_pdf_type(pdf_type)

        # Load the CSV file
        logger.info(f"Loading CSV with {len(required_columns)} required columns for PDF type '{pdf_type}'")

        # First, peek at the file to see what columns are available
        try:
            # Read just the header to see available columns
            sample_df = pd.read_csv(file_path, nrows=0)
            available_columns = set(sample_df.columns)

            # Find intersection of required and available columns
            columns_to_load = required_columns.intersection(available_columns)
            missing_columns = required_columns - available_columns

            if missing_columns:
                logger.warning(f"Missing columns in CSV: {sorted(missing_columns)}")

            if not columns_to_load:
                raise ValueError("No required columns found in CSV; cannot proceed with loading.")
            
            logger.info(f"Loading {len(columns_to_load)} columns from CSV")

            # Load the data with specified columns
            if columns_to_load:
                dtype_spec = self._get_dtype_spec()
                # Filter dtype_spec to only include columns we're loading
                if dtype_spec:
                    dtype_spec = {k: v for k, v in dtype_spec.items() if k in columns_to_load}

                self._raw_data = pd.read_csv(file_path, usecols=list(columns_to_load), dtype=dtype_spec)  # type: ignore
            else:
                self._raw_data = pd.read_csv(file_path)

            # Store the required columns for later use
            self._required_columns = required_columns

            logger.info(f"Loaded {len(self._raw_data)} rows, {len(self._raw_data.columns)} columns")  # type: ignore
            return self._raw_data  # type: ignore

        except Exception as e:
            logger.error(f"Failed to load CSV with required columns: {e}")
            # Fallback to loading all columns
            return self.load_csv(file_path)

    def process_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Process raw NCCL data into format suitable for plotting.

        Args:
            df: Optional DataFrame to process. If None, uses loaded data.

        Returns:
            Processed DataFrame ready for plotting

        Raises:
            ValueError: If no data available to process
        """
        if df is None:
            if self._raw_data is None:
                raise ValueError("No data available to process")
            df = self._raw_data

        logger.info(f"Processing NCCL data with {len(df)} rows")

        # Create a copy to avoid modifying original
        processed_df = df.copy()

        # Clean and standardize test binary names
        processed_df = self._clean_test_binaries(processed_df)

        # Convert numeric columns
        processed_df = self._convert_numeric_columns(processed_df)

        # Handle NCCL-specific columns
        processed_df = self._process_nccl_columns(processed_df)

        # Create derived performance columns from config
        processed_df = self.create_performance_columns(processed_df)

        # Sort data for consistent output
        processed_df = self._sort_data(processed_df)

        # Store processed data
        self._processed_data = processed_df

        # Cache test binaries
        test_binary_col = self.get_dataframe_column_name('test_binary', processed_df)
        if test_binary_col in processed_df.columns:
            self._test_binaries = set(processed_df[test_binary_col].unique())
        else:
            self._test_binaries = set()

        logger.info(f"Processed data contains {len(self._test_binaries)} test binaries")

        return processed_df

    def _clean_test_binaries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize test binary names.

        Args:
            df: DataFrame to clean

        Returns:
            DataFrame with cleaned test binary names
        """
        test_binary_col = self.get_dataframe_column_name('test_binary', df)

        if test_binary_col not in df.columns:
            logger.warning(f"Test binary column '{test_binary_col}' not found")
            return df

        df[test_binary_col] = df[test_binary_col].str.replace('_perf$', '', regex=True)  # type: ignore
        df[test_binary_col] = df[test_binary_col] + '_perf'

        return df

    def _convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert string columns to numeric where appropriate based on column patterns.

        Args:
            df: DataFrame to convert

        Returns:
            DataFrame with numeric columns converted
        """
        # Dynamically identify columns that should be numeric based on patterns
        for col in df.columns:
            # Convert columns that contain numeric performance data
            if any(pattern in col.lower() for pattern in ['bus_bw', 'alg_bw', 'latency', 'throughput', 'bandwidth']):
                df[col] = pd.to_numeric(df[col], errors='coerce')
            # Convert size/count columns
            elif any(pattern in col.lower() for pattern in ['size', 'num_nodes', 'num_processes', 'count']):
                df[col] = pd.to_numeric(df[col], errors='coerce')
            # Convert columns from dtype specification in config
            elif self.plot_config and hasattr(self.plot_config, 'settings'):
                settings = self.plot_config.settings
                if (settings and settings.data_processing and settings.data_processing.dtypes
                    and col in settings.data_processing.dtypes):
                    dtype = settings.data_processing.dtypes[col]
                    if 'int' in dtype or 'float' in dtype:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def _process_nccl_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process NCCL-specific columns.

        Args:
            df: DataFrame to process

        Returns:
            DataFrame with processed NCCL columns
        """
        # Process NCCL_TESTS_SPLIT_MASK - convert hex to string representation
        split_mask_col = self.get_dataframe_column_name('nccl_tests_split_mask', df)
        if split_mask_col in df.columns:
            df[split_mask_col] = df[split_mask_col].apply(self._format_split_mask)

        # Process local_register column
        local_reg_col = self.get_dataframe_column_name('local_register', df)
        if local_reg_col in df.columns:
            # Ensure it's a string for consistent handling
            df[local_reg_col] = df[local_reg_col].astype(str)

        return df

    def _format_split_mask(self, value: Any) -> str:
        """Format NCCL_TESTS_SPLIT_MASK value.

        Args:
            value: Raw split mask value

        Returns:
            Formatted string representation
        """
        if pd.isna(value):
            return 'None'

        # If it's already a hex string, return as is
        if isinstance(value, str) and value.startswith('0x'):
            return value

        # Try to convert to hex
        try:
            if isinstance(value, (int, float)):
                return f'0x{int(value):X}'
            else:
                # Try to parse as string number
                return f'0x{int(str(value), 0):X}'
        except (ValueError, TypeError):
            return str(value)

    def _sort_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort data based on configuration or default ordering.

        Args:
            df: DataFrame to sort

        Returns:
            Sorted DataFrame
        """
        sort_columns = []

        # Check configuration for sort order
        if self.plot_config and hasattr(self.plot_config, 'settings'):
            settings = self.plot_config.settings
            if settings and settings.chart_ordering and settings.chart_ordering.enable_sorting:
                sort_order = settings.chart_ordering.sort_order
                if sort_order:
                    # Get sort fields from Pydantic model
                    try:
                        # Check if it's a Pydantic model with model_fields
                        if hasattr(sort_order.__class__, 'model_fields'):
                            # Use model_fields to get field names
                            fields = sort_order.__class__.model_fields.keys()
                        else:
                            # Fallback to dir() for non-Pydantic objects
                            fields = [f for f in dir(sort_order)
                                    if not f.startswith('_') and not callable(getattr(sort_order, f))]

                        for field_name in fields:
                            # Skip internal Pydantic attributes
                            if field_name in ['model_config', 'model_fields', 'model_fields_set']:
                                continue

                            field_value = getattr(sort_order, field_name, None)
                            # Only process if the field is enabled (truthy)
                            if field_value:
                                try:
                                    # Map config field name to actual column name
                                    if field_name == 'NCCL_TESTS_SPLIT_MASK':
                                        # Special case for NCCL_TESTS_SPLIT_MASK
                                        col = self.get_dataframe_column_name('nccl_tests_split_mask', df)
                                    else:
                                        col = self.get_dataframe_column_name(field_name, df)

                                    if col in df.columns:
                                        sort_columns.append(col)
                                        logger.debug(f"Added sort column: {field_name} -> {col}")
                                except Exception as e:
                                    logger.warning(f"Failed to map sort field '{field_name}': {e}")
                    except Exception as e:
                        logger.warning(f"Failed to process sort order configuration: {e}")

        # Add size as final sort column
        size_col = self.get_dataframe_column_name('size', df)
        if size_col in df.columns:
            sort_columns.append(size_col)

        # Perform sort if we have columns to sort by
        if sort_columns:
            df = df.sort_values(by=sort_columns)
            logger.debug(f"Sorted data by columns: {sort_columns}")

        return df

    def get_performance_columns(self) -> List[str]:
        """Get list of available performance metric columns from configuration.

        Returns:
            List of column names containing performance metrics
        """
        if self._processed_data is None:
            return []

        perf_columns = []

        # Get performance columns from configuration only
        if self.plot_config and hasattr(self.plot_config, 'settings'):
            settings = self.plot_config.settings
            if settings and settings.data_processing and settings.data_processing.performance_columns:
                for col_name in settings.data_processing.performance_columns.keys():
                    if col_name in self._processed_data.columns:
                        perf_columns.append(col_name)

        return perf_columns

    def get_size_column(self) -> str:
        """Get the column name containing size/message size data.

        Returns:
            Column name for size data
        """
        # Check configuration first
        if self.plot_config and hasattr(self.plot_config, 'settings'):
            settings = self.plot_config.settings
            if settings and settings.data_processing and settings.data_processing.size_column:
                return settings.data_processing.size_column

        # Return dynamically mapped size column
        if self._processed_data is not None:
            return self.get_dataframe_column_name('size', self._processed_data)
        else:
            return self.get_dataframe_column_name('size')

    def get_test_binaries(self) -> List[str]:
        """Get list of unique test binaries in the data.

        Returns:
            List of test binary names
        """
        if self._test_binaries is None:
            if self._processed_data is None:
                return []
            test_binary_col = self.get_dataframe_column_name('test_binary', self._processed_data)
            if test_binary_col in self._processed_data.columns:
                self._test_binaries = set(self._processed_data[test_binary_col].unique())
            else:
                self._test_binaries = set()

        return sorted(list(self._test_binaries))

    def get_metadata_columns(self) -> Dict[str, str]:
        """Get mapping of metadata columns dynamically from available columns.

        Returns:
            Dictionary mapping logical names to actual column names found in data
        """
        if self._processed_data is None:
            return {}

        metadata_mapping = {}
        available_columns = set(self._processed_data.columns)

        # Dynamically find metadata columns based on patterns
        for col in available_columns:
            if 'library_version' in col:
                metadata_mapping['library_version'] = col
            elif 'cuda' in col.lower() and 'version' in col.lower():
                metadata_mapping['cuda_version'] = col
            elif 'system' in col.lower() and 'name' in col.lower():
                metadata_mapping['system_name'] = col
            elif 'gpu' in col.lower() and 'arch' in col.lower():
                metadata_mapping['gpu_arch'] = col
            elif col == 'num_nodes':
                metadata_mapping['num_nodes'] = col
            elif col == 'num_processes':
                metadata_mapping['num_processes'] = col

        return metadata_mapping

    def collect_dataset_metadata(self) -> Dict[str, Any]:
        """Collect comprehensive statistics about the dataset for introduction page.

        Returns:
            Dictionary containing dataset statistics including:
            - basic_stats: Row count, column count, data size info
            - test_info: Test binaries, configurations, scale information
            - system_info: Systems, GPU architectures, software versions
            - data_coverage: Coverage statistics for different parameters
        """
        if self._processed_data is None:
            logger.warning("No processed data available for metadata collection")
            return {}

        df = self._processed_data
        metadata = {}

        # Basic dataset statistics
        metadata['basic_stats'] = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        }

        # Test binary and configuration information
        test_binary_col = self.get_dataframe_column_name('test_binary', df)
        if test_binary_col in df.columns:
            unique_binaries = df[test_binary_col].nunique()
            metadata['test_info'] = {
                'unique_test_binaries': unique_binaries,
                'test_binaries': sorted(df[test_binary_col].unique().tolist()),
                'records_per_binary': df[test_binary_col].value_counts().to_dict()
            }
        else:
            metadata['test_info'] = {'unique_test_binaries': 0, 'test_binaries': []}

        # Scale and configuration information
        scale_info = {}

        # Node and process information
        num_nodes_col = self.get_dataframe_column_name('num_nodes', df)
        if num_nodes_col in df.columns:
            scale_info['node_range'] = {
                'min': int(df[num_nodes_col].min()),
                'max': int(df[num_nodes_col].max()),
                'unique_counts': sorted(df[num_nodes_col].unique().tolist())
            }

        num_processes_col = self.get_dataframe_column_name('num_processes', df)
        if num_processes_col in df.columns:
            scale_info['process_range'] = {
                'min': int(df[num_processes_col].min()),
                'max': int(df[num_processes_col].max()),
                'unique_counts': sorted(df[num_processes_col].unique().tolist())
            }

        # Message size information
        size_col = self.get_dataframe_column_name('size', df)
        if size_col in df.columns:
            scale_info['message_size_range'] = {
                'min_bytes': int(df[size_col].min()),
                'max_bytes': int(df[size_col].max()),
                'min_human': self._format_bytes(df[size_col].min()),  # type: ignore
                'max_human': self._format_bytes(df[size_col].max()),  # type: ignore
                'unique_sizes': len(df[size_col].unique())
            }

        metadata['test_info']['scale_info'] = scale_info

        # System and environment information
        system_info = {}

        # System names
        system_name_col = self.get_dataframe_column_name('system_name', df)
        if system_name_col in df.columns:
            system_info['unique_systems'] = df[system_name_col].nunique()
            system_info['systems'] = sorted(df[system_name_col].unique().tolist())

        # GPU architecture
        gpu_arch_col = self.get_dataframe_column_name('gpu_arch', df)
        if gpu_arch_col in df.columns:
            system_info['gpu_architectures'] = sorted(df[gpu_arch_col].unique().tolist())

        # Software versions
        library_version_col = self.get_dataframe_column_name('library_version', df)
        if library_version_col in df.columns:
            system_info['nccl_versions'] = sorted(df[library_version_col].unique().tolist())

        cuda_version_col = self.get_dataframe_column_name('cuda_version', df)
        if cuda_version_col in df.columns:
            system_info['cuda_versions'] = sorted(df[cuda_version_col].unique().tolist())

        metadata['system_info'] = system_info

        # Data coverage analysis
        coverage_info = {}

        for col in df.columns:
            # Look for NCCL environment variable columns
            if col.startswith('test_env_vars.NCCL_'):
                # Extract the NCCL parameter name from the column
                param_name = col.replace('test_env_vars.', '')
                unique_values = df[col].dropna().unique()
                coverage_info[param_name] = {
                    'unique_values': len(unique_values),
                    'values': sorted(unique_values.tolist()) if len(unique_values) <= 10 else f"{len(unique_values)} unique values"
                }
            # Also check for direct NCCL_ columns (in case they exist without test_env_vars prefix)
            elif col.startswith('NCCL_'):
                unique_values = df[col].dropna().unique()
                coverage_info[col] = {
                    'unique_values': len(unique_values),
                    'values': sorted(unique_values.tolist()) if len(unique_values) <= 10 else f"{len(unique_values)} unique values"
                }

        # Local register usage
        local_reg_col = self.get_dataframe_column_name('local_register', df)
        if local_reg_col in df.columns:
            coverage_info['local_register'] = {
                'unique_values': df[local_reg_col].nunique(),
                'values': sorted(df[local_reg_col].unique().tolist())
            }

        # Split mask usage
        split_mask_col = self.get_dataframe_column_name('nccl_tests_split_mask', df)
        if split_mask_col in df.columns:
            coverage_info['split_mask'] = {
                'unique_values': df[split_mask_col].nunique(),
                'values': sorted(df[split_mask_col].unique().tolist())
            }

        metadata['data_coverage'] = coverage_info

        # Add timestamp if available
        metadata['collection_timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

        logger.info(f"Collected metadata for dataset with {metadata['basic_stats']['total_records']} records")
        return metadata

    def _format_bytes(self, bytes_value: float) -> str:
        """Format bytes value into human-readable string.

        Args:
            bytes_value: Number of bytes

        Returns:
            Human-readable string (e.g., "1.5 MB", "2.0 GB")
        """
        if bytes_value < 1024:
            return f"{bytes_value} B"
        elif bytes_value < 1024**2:
            return f"{bytes_value/1024:.1f} KB"
        elif bytes_value < 1024**3:
            return f"{bytes_value/(1024**2):.1f} MB"
        else:
            return f"{bytes_value/(1024**3):.1f} GB"

    def get_chart_groups_for_plotting(self, chart_type: str, test_binary: str) -> List[Dict[str, Any]]:
        """Get chart groups for plotting based on chart type and test binary.

        This method groups the processed data into separate charts based on the
        chart_split_params configuration. Each group represents data for one chart.

        Args:
            chart_type: Name of the chart type from configuration
            test_binary: Test binary name to filter by

        Returns:
            List of chart group dictionaries, each containing:
            - 'group_key': String identifying the group
            - 'data': DataFrame subset for this group
            - 'filters': Dictionary of filters applied to create this group

        Raises:
            ValueError: If processed data not available or chart type not found
        """
        if self._processed_data is None:
            raise ValueError("No processed data available. Call process_data() first.")

        # Get chart configuration
        if not self.plot_config or not self.plot_config.chart:
            raise ValueError("Plot configuration not available")

        chart_config = self.plot_config.get_chart_by_id(chart_type)
        if not chart_config:
            raise ValueError(f"Chart type '{chart_type}' not found in configuration")

        # Filter data by test binary first
        test_binary_col = self.get_dataframe_column_name('test_binary')
        if test_binary_col not in self._processed_data.columns:
            logger.warning(f"Test binary column '{test_binary_col}' not found")
            return []

        filtered_data = self._processed_data[
            self._processed_data[test_binary_col] == test_binary
        ].copy()

        if filtered_data.empty:
            logger.warning(f"No data found for test binary: {test_binary}")
            return []

        # Apply fixed parameter filters if any
        if chart_config.fixed_params:
            # Filter out parameters that don't exist in the data
            valid_fixed_params = {}
            for param, value in chart_config.fixed_params.items():
                try:
                    param_col = self.get_dataframe_column_name(param, filtered_data)
                    if param_col in filtered_data.columns:
                        # Check if the value exists in the data (case agnostic)
                        data_values = filtered_data[param_col].astype(str).str.lower()  # type: ignore
                        config_value_lower = str(value).lower()
                        if config_value_lower in data_values.values:
                            valid_fixed_params[param] = value
                        else:
                            logger.warning(f"Fixed parameter value '{value}' not found in column '{param_col}' for parameter '{param}'")
                    else:
                        logger.warning(f"Fixed parameter column '{param_col}' not found for parameter '{param}'")
                except ValueError as e:
                    logger.warning(f"Fixed parameter '{param}' not found in available columns.")
                    logger.debug(f"Fixed parameter '{param}' not found in available columns: {e}")

            # Apply filters for valid parameters only
            if valid_fixed_params:
                for param, value in valid_fixed_params.items():
                    param_col = self.get_dataframe_column_name(param, filtered_data)
                    # Make value matching case agnostic
                    filtered_data = filtered_data[filtered_data[param_col].astype(str).str.lower() == str(value).lower()]  # type: ignore
                    if len(filtered_data) == 0:
                        break
            else:
                logger.warning(f"No valid fixed parameters found for {chart_type}, using all data")

        if filtered_data.empty:  # type: ignore
            logger.warning(f"No data remaining after applying fixed parameters for {chart_type}")
            return []

        # Get split parameters - these create separate charts
        split_params = chart_config.chart_split_params or {}

        if not split_params:
            # No split parameters, return single group
            return [{
                'group_key': f"{chart_type}_{test_binary}",
                'data': filtered_data,
                'filters': {}
            }]

        # Build list of split columns that exist in the data
        split_columns = []
        for param, values in split_params.items():
            if values == "all":  # Include all values for this parameter
                try:
                    param_col = self.get_dataframe_column_name(param, filtered_data)
                    if param_col in filtered_data.columns:  # type: ignore
                        split_columns.append(param_col)
                    else:
                        logger.warning(f"Split parameter column '{param_col}' not found for parameter '{param}', skipping")
                except ValueError:
                    logger.warning(f"Split parameter '{param}' not found in data.")
            # Add support for specific values if needed in the future

        if not split_columns:
            # No valid split columns, return single group
            return [{
                'group_key': f"{chart_type}_{test_binary}",
                'data': filtered_data,
                'filters': {}
            }]

        # Group data by split parameters
        chart_groups = []
        grouped = filtered_data.groupby(split_columns, dropna=False)  # type: ignore

        for group_values, group_data in grouped:
            if group_data.empty:  # type: ignore
                continue

            # Create group key from split parameter values
            if len(split_columns) == 1:
                group_values = [group_values]  # Ensure it's always a list

            group_key_parts = [f"{chart_type}", f"{test_binary}"]
            filters = {}

            for i, col in enumerate(split_columns):
                value = group_values[i]  # type: ignore
                # Map back to original parameter name for the key
                param_name = self._get_config_param_name(col)
                # Shorten the parameter name for the group key
                shortened_param = self.shorten_legend_key(param_name)
                group_key_parts.append(f"{shortened_param}={value}")
                filters[param_name] = value

            group_key = "_".join(group_key_parts)

            chart_groups.append({
                'group_key': group_key,
                'data': group_data,
                'filters': filters
            })

        logger.info(f"Created {len(chart_groups)} chart groups for {chart_type} - {test_binary}")

        return chart_groups

    def _get_config_param_name(self, column_name: str) -> str:
        """Get the configuration parameter name for a given column name.

        Args:
            column_name: Column name in the DataFrame

        Returns:
            Configuration parameter name
        """
        # Reverse lookup in the column mapping
        for config_name, mapped_col in self._BASE_NCCL_COLUMNS.items():
            if mapped_col == column_name:
                # Convert back to config parameter format
                if config_name == 'nccl_tests_split_mask':
                    return 'NCCL_TESTS_SPLIT_MASK'
                elif config_name == 'local_register':
                    return 'local_register'
                else:
                    return config_name

        # If not found in mapping, return the column name as-is
        return column_name

    def get_legend_style_info(self, legend_key: str, chart_type: str) -> Dict[str, Any]:
        """Get style information for a legend entry.

        This method determines the color, line style, and marker style for a given
        legend key based on the style configuration.

        Args:
            legend_key: String identifying the legend entry (e.g., "local_register=1")
            chart_type: Type of chart (for chart-specific styling if needed)

        Returns:
            Dictionary containing:
            - 'color': matplotlib color string
            - 'line_style': matplotlib line style string
            - 'marker_style': matplotlib marker style string
        """
        # Default style values
        style_info = {
            'color': 'blue',
            'line_style': '-',
            'marker_style': 'o'
        }

        # Check if we have style configuration
        if not self.plot_config or not self.plot_config.settings:
            return style_info

        settings = self.plot_config.settings
        if not settings.style_mappings:
            return style_info

        style_mappings = settings.style_mappings

        # Parse the legend key to extract parameter values
        # Legend key format: "param1=value1, param2=value2" or "param=value"
        param_values = {}
        for part in legend_key.split(','):
            part = part.strip()
            if '=' in part:
                param, value = part.split('=', 1)
                param = param.strip()
                # Convert shortened parameter names back to full names for style mapping
                full_param = self._SHORTENED_TO_FULL_PARAM.get(param, param)
                param_values[full_param] = value.strip()

        # Get the style column parameter
        style_column = style_mappings.style_column
        if style_column in param_values:
            style_value = param_values[style_column]

            # Get line style
            if style_mappings.line_styles:
                # Try exact match first
                if style_value in style_mappings.line_styles:
                    style_info['line_style'] = style_mappings.line_styles[style_value]
                # Try case-insensitive match for text values
                else:
                    for key, line_style in style_mappings.line_styles.items():
                        if str(key).lower() == str(style_value).lower():
                            style_info['line_style'] = line_style
                            break

            # Get marker style
            if style_mappings.marker_styles:
                # Try exact match first
                if style_value in style_mappings.marker_styles:
                    style_info['marker_style'] = style_mappings.marker_styles[style_value]
                # Try case-insensitive match for text values
                else:
                    for key, marker_style in style_mappings.marker_styles.items():
                        if str(key).lower() == str(style_value).lower():
                            style_info['marker_style'] = marker_style
                            break

        # Check for special color mappings
        if style_mappings.default_colors:
            # Build a key from NCCL algorithm parameters
            algo_key_parts = []
            algo_params = ['NCCL_ALGO', 'NCCL_PROTO', 'NCCL_MIN_CTAS']

            for param in algo_params:
                if param in param_values:
                    algo_key_parts.append(f"{param}={param_values[param]}")

            if algo_key_parts:
                algo_key = '_'.join(algo_key_parts)
                if algo_key in style_mappings.default_colors:
                    style_info['color'] = style_mappings.default_colors[algo_key]

        # Assign colors based on hash if no special color found
        if style_info['color'] == 'blue':
            # Use color palette from style configuration
            colors = self._get_color_palette_from_config()

            # Create a color key that excludes local_register since it's handled by line/marker styles
            color_key_parts = []
            for param, value in param_values.items():
                if param != 'local_register':  # Exclude local_register from color assignment
                    color_key_parts.append(f"{param}={value}")

            # Sort to ensure consistent ordering
            color_key_parts.sort()
            color_key = ", ".join(color_key_parts)

            # Use hash to consistently assign same color to same color key
            color_index = hash(color_key) % len(colors)
            style_info['color'] = colors[color_index]

            logger.debug(f"Color assignment for legend '{legend_key}': color_key='{color_key}', color_index={color_index}, color='{style_info['color']}'")

        logger.debug(f"Style for '{legend_key}': {style_info}")
        return style_info

    def _get_color_palette_from_config(self) -> List[str]:
        """Get color palette from style configuration.

        Returns:
            List of color hex strings from matplotlib color palette
        """
        if self._color_palette_cache is not None:
            return self._color_palette_cache

        # Default fallback colors if no style config
        fallback_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
            '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5'
        ]

        if not self.style_config:
            logger.debug("No style configuration provided, using fallback colors")
            self._color_palette_cache = fallback_colors
            return self._color_palette_cache

        # Get palette name and number of colors from config
        palette_name = self.style_config.color_scheme.palette
        num_colors = self.style_config.color_palette.num_colors

        try:
            # Try to get the color palette from matplotlib
            colors_list: List[str] = []

            try:
                cmap = plt.colormaps[palette_name]

                # Generate colors by sampling the colormap
                try:
                    # Try to get the number of colors in the colormap
                    n_available = getattr(cmap, 'N', None)
                    if n_available is not None and n_available < 256:
                        # For ListedColormap with limited colors, use all available colors
                        indices = np.linspace(0, 1, min(n_available, num_colors))
                    else:
                        # For continuous colormaps, sample evenly
                        indices = np.linspace(0, 1, num_colors)
                except AttributeError:
                    # Fallback to even sampling
                    indices = np.linspace(0, 1, num_colors)

                # Get colors and convert to hex
                for idx in indices:
                    rgba = cmap(idx)
                    hex_color = mcolors.to_hex(rgba)
                    colors_list.append(hex_color)

            except Exception as e:
                logger.warning(f"Could not load palette '{palette_name}': {e}, using fallback colors")
                colors_list = fallback_colors

            # Ensure we have enough colors
            if len(colors_list) < num_colors:
                # Repeat colors if needed
                original_colors = colors_list[:]
                while len(colors_list) < num_colors:
                    colors_list.extend(original_colors)

            self._color_palette_cache = colors_list[:num_colors]
            logger.debug(f"Loaded {len(self._color_palette_cache)} colors from palette '{palette_name}'")

        except Exception as e:
            logger.error(f"Failed to load color palette: {e}")
            self._color_palette_cache = fallback_colors

        return self._color_palette_cache
