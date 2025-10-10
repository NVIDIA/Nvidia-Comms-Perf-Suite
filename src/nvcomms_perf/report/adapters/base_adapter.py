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
Base adapter for processing performance test data.

This module provides a library-agnostic base class for adapting raw performance
test data into formats suitable for plotting and analysis.
"""

import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class DataFrameMetadata:
    """Metadata about a processed DataFrame."""

    source_file: str
    row_count: int
    column_count: int
    test_binaries: List[str]
    performance_columns: List[str]
    size_column: str
    grouping_columns: List[str]


class BasePerformanceAdapter(ABC):
    """Base class for adapting performance test data for plotting.

    This abstract base class provides the framework for processing raw performance
    test data into structured DataFrames suitable for creating plots according to
    the plot configuration specifications.

    The adapter uses dynamic column discovery based on plot configuration to load
    only the necessary data columns for efficient memory usage and processing.
    """

    def __init__(self, plot_config: Optional[Any] = None):
        """Initialize the adapter.

        Args:
            plot_config: Optional plot configuration object
        """
        self.plot_config = plot_config
        self._raw_data: Optional[pd.DataFrame] = None
        self._processed_data: Optional[pd.DataFrame] = None
        self._metadata: Optional[DataFrameMetadata] = None

    def load_csv(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load CSV data from file.

        Args:
            file_path: Path to the CSV file

        Returns:
            Raw DataFrame loaded from CSV

        Raises:
            FileNotFoundError: If CSV file not found
            pd.errors.ParserError: If CSV parsing fails
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        logger.info(f"Loading CSV data from: {file_path}")

        try:
            # Load with optimized dtypes if specified in config
            dtype_spec = self._get_dtype_spec()
            if dtype_spec is not None:
                # Convert string dtypes to numpy dtypes
                dtype_spec = {k: np.dtype(v) if isinstance(v, str) else v for k, v in dtype_spec.items()}

            self._raw_data = pd.read_csv(file_path, dtype=dtype_spec)  # type: ignore

            if self._raw_data is not None:
                # Set the source_file attribute for traceability
                if not hasattr(self._raw_data, 'attrs'):
                    self._raw_data.attrs = {}
                self._raw_data.attrs['source_file'] = str(file_path)

                logger.info(f"Loaded {len(self._raw_data)} rows, {len(self._raw_data.columns)} columns")
            else:
                logger.warning("Loaded DataFrame is None")
                # Create empty DataFrame if None is returned
                self._raw_data = pd.DataFrame()
                if not hasattr(self._raw_data, 'attrs'):
                    self._raw_data.attrs = {}
                self._raw_data.attrs['source_file'] = str(file_path)

            return self._raw_data

        except pd.errors.ParserError as e:
            logger.error(f"Failed to parse CSV file: {e}")
            raise

    def _get_dtype_spec(self) -> Optional[Dict[str, str]]:
        """Get data type specifications from config.

        Returns:
            Dictionary of column names to data types, or None
        """
        if not self.plot_config or not hasattr(self.plot_config, 'settings'):
            return None

        settings = self.plot_config.settings
        if settings and settings.data_processing and settings.data_processing.dtypes:
            return settings.data_processing.dtypes

        return None

    @abstractmethod
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw data into format suitable for plotting.

        This method must be implemented by subclasses to handle library-specific
        data transformations.

        Args:
            df: Raw DataFrame to process

        Returns:
            Processed DataFrame ready for plotting
        """
        pass

    @abstractmethod
    def get_performance_columns(self) -> List[str]:
        """Get list of available performance metric columns from configuration.

        Returns:
            List of column names containing performance metrics defined in config
        """
        pass

    @abstractmethod
    def get_size_column(self) -> str:
        """Get the column name containing size/message size data.

        Returns:
            Column name for size data
        """
        pass

    @abstractmethod
    def get_test_binaries(self) -> List[str]:
        """Get list of unique test binaries in the data.

        Returns:
            List of test binary names
        """
        pass

    @abstractmethod
    def get_required_columns_for_pdf_type(self, pdf_type: str) -> Set[str]:
        """Get all columns required for a specific PDF report type.

        Args:
            pdf_type: Type of PDF report (e.g., 'basic_report', 'comprehensive_report')

        Returns:
            Set of column names required for the PDF type
        """
        pass

    @abstractmethod
    def get_required_test_binaries_for_pdf_type(self, pdf_type: str) -> Set[str]:
        """Get test binaries required for a specific PDF report type.

        Args:
            pdf_type: Type of PDF report

        Returns:
            Set of test binary names required for the PDF type
        """
        pass

    @abstractmethod
    def get_dataframe_column_name(self, config_param: str, data_source: Optional[Any] = None) -> str:
        """Get the actual dataframe column name for a configuration parameter.

        Args:
            config_param: Parameter name from configuration
            data_source: Optional dataframe or series to check which column exists

        Returns:
            Actual column name in dataframe
        """
        pass

    @abstractmethod
    def load_csv_with_required_columns(self, file_path: Union[str, Path], pdf_type: str) -> pd.DataFrame:
        """Load CSV data with only the columns required for the specified PDF type.

        Args:
            file_path: Path to the CSV file
            pdf_type: Type of PDF report to determine required columns

        Returns:
            DataFrame with only the required columns loaded
        """
        pass

    def create_performance_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived performance columns based on configuration.

        Args:
            df: DataFrame to add performance columns to

        Returns:
            DataFrame with additional performance columns
        """
        if not self.plot_config or not hasattr(self.plot_config, 'settings'):
            return df

        settings = self.plot_config.settings
        if not settings or not settings.data_processing:
            return df

        perf_columns = settings.data_processing.performance_columns
        if not perf_columns:
            return df

        df = df.copy()

        # Type guard: ensure df is a DataFrame
        if not isinstance(df, pd.DataFrame):
            logger.warning("Input to create_performance_columns is not a DataFrame, returning empty DataFrame")
            return pd.DataFrame()

        for col_name, col_config in perf_columns.items():
            logger.debug(f"Creating performance column: {col_name}")

            # Get source columns
            source_cols = col_config.sources
            missing_cols = [col for col in source_cols if col not in df.columns]  # type: ignore
            if missing_cols:
                logger.warning(f"Missing source columns for {col_name}: {missing_cols}")
                continue

            # Apply operation
            operation = col_config.operation
            if operation == "none":
                # For "none" operation, just copy the single source column
                if len(source_cols) == 1:
                    df[col_name] = df[source_cols[0]]
                else:
                    logger.warning(f"'none' operation requires exactly one source column, got {len(source_cols)} for {col_name}")
                    continue
            elif operation == "max":
                df[col_name] = df[source_cols].max(axis=1)
            elif operation == "min":
                df[col_name] = df[source_cols].min(axis=1)
            elif operation == "sum":
                df[col_name] = df[source_cols].sum(axis=1)
            elif operation == "mean":
                df[col_name] = df[source_cols].mean(axis=1)
            elif operation == "median":
                df[col_name] = df[source_cols].median(axis=1)
            elif operation == "std":
                df[col_name] = df[source_cols].std(axis=1)
            elif operation == "var":
                df[col_name] = df[source_cols].var(axis=1)
            else:
                logger.warning(f"Unknown operation '{operation}' for column {col_name}")

        return df

    def filter_by_chart_type(self, df: pd.DataFrame, chart_type: str) -> pd.DataFrame:
        """Filter DataFrame based on chart type configuration.

        Args:
            df: DataFrame to filter
            chart_type: Name of the chart type from configuration

        Returns:
            Filtered DataFrame
        """
        # Type guard: ensure df is a DataFrame
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame()

        if not self.plot_config or not hasattr(self.plot_config, 'chart'):
            return df

        chart_config = self.plot_config.get_chart_by_id(chart_type)
        if not chart_config:
            logger.warning(f"Chart type '{chart_type}' not found in configuration")
            return df

        df_filtered = df.copy()

        # Apply fixed parameter filters
        if chart_config.fixed_params:
            for param, value in chart_config.fixed_params.items():
                # Use dynamic column name mapping
                actual_column = self.get_dataframe_column_name(param, df_filtered)
                if actual_column in df_filtered.columns:  # type: ignore
                    df_filtered = df_filtered[df_filtered[actual_column] == value]
                else:
                    logger.warning(f"Fixed parameter column '{param}' (mapped to '{actual_column}') not found")

        # Type guard: ensure return is a DataFrame
        if not isinstance(df_filtered, pd.DataFrame):
            df_filtered = pd.DataFrame()
        return df_filtered

    def get_grouping_columns(self, chart_type: str) -> List[str]:
        """Get columns to group data by for a specific chart type.

        Args:
            chart_type: Name of the chart type

        Returns:
            List of actual column names to group by
        """
        if not self.plot_config or not hasattr(self.plot_config, 'chart'):
            return []

        chart_config = self.plot_config.get_chart_by_id(chart_type)
        if not chart_config:
            return []

        grouping_cols = []

        # Add split parameters with dynamic column mapping
        if chart_config.chart_split_params:
            for param, values in chart_config.chart_split_params.items():
                actual_column = self.get_dataframe_column_name(param)
                if isinstance(values, str):
                    grouping_cols.append(actual_column)
                elif isinstance(values, list):
                    grouping_cols.extend([self.get_dataframe_column_name(str(v)) for v in values if isinstance(v, (str, int))])

        # Add combine parameters with dynamic column mapping
        if chart_config.chart_combine_params:
            for param, values in chart_config.chart_combine_params.items():
                actual_column = self.get_dataframe_column_name(param)
                if isinstance(values, str) and actual_column not in grouping_cols:
                    grouping_cols.append(actual_column)
                elif isinstance(values, list):
                    for v in values:
                        if isinstance(v, (str, int)):
                            mapped_col = self.get_dataframe_column_name(str(v))
                            if mapped_col not in grouping_cols:
                                grouping_cols.append(mapped_col)

        return grouping_cols

    def prepare_for_plotting(
        self,
        chart_type: str,
        performance_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, DataFrameMetadata]:
        """Prepare data for plotting based on chart type.

        Args:
            chart_type: Name of the chart type from configuration
            performance_column: Optional specific performance column to use

        Returns:
            Tuple of (prepared DataFrame, metadata)

        Raises:
            ValueError: If data not loaded or processed
        """
        if self._processed_data is None:
            raise ValueError("No processed data available. Call process_data() first.")

        # Ensure _processed_data is a DataFrame
        if not isinstance(self._processed_data, pd.DataFrame):
            logger.warning("_processed_data is not a DataFrame, creating empty DataFrame")
            self._processed_data = pd.DataFrame()

        # Filter by chart type
        df = self.filter_by_chart_type(self._processed_data, chart_type)

        # Get grouping columns
        grouping_cols = self.get_grouping_columns(chart_type)

        # Get performance columns
        if performance_column:
            perf_cols = [performance_column]
        else:
            perf_cols = self.get_performance_columns()

        # Create metadata
        source_file = 'unknown'
        if self._raw_data is not None and hasattr(self._raw_data, 'attrs'):
            source_file = str(self._raw_data.attrs.get('source_file', 'unknown'))

        # Ensure df is a DataFrame for metadata
        if not isinstance(df, pd.DataFrame):
            column_count = 0
        else:
            column_count = len(df.columns)  # type: ignore

        metadata = DataFrameMetadata(
            source_file=source_file,
            row_count=len(df) if df is not None else 0,
            column_count=column_count,
            test_binaries=self.get_test_binaries(),
            performance_columns=perf_cols,
            size_column=self.get_size_column(),
            grouping_columns=grouping_cols
        )

        # Ensure we always return a DataFrame
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame()

        return df, metadata

    def get_default_performance_column(self) -> Optional[str]:
        """Get the default performance column from configuration.

        Returns:
            Default performance column name, or None
        """
        if not self.plot_config or not hasattr(self.plot_config, 'settings'):
            return None

        settings = self.plot_config.settings
        if settings and settings.data_processing:
            return settings.data_processing.default_performance_column

        return None
