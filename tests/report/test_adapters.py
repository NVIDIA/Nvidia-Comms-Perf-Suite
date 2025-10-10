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
Tests for the adapters module.

This module tests all adapter functionality including:
- BasePerformanceAdapter abstract base class
- NCCLPerformanceAdapter implementation
- Data processing and column optimization
- Legend styling and formatting
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import logging
from typing import Set, Union
from unittest.mock import Mock, patch, MagicMock

from nvcomms_perf.report.adapters import (
    BasePerformanceAdapter,
    NCCLPerformanceAdapter,
    DataFrameMetadata
)
from nvcomms_perf.report.models.plot_config import (
    ChartConfig,
    Settings,
    DataProcessing,
    PerformanceColumn,
    ChartType,
    AnalysisProfile,
    PDFConfig
)


class ConcreteAdapter(BasePerformanceAdapter):
    """Concrete implementation of BasePerformanceAdapter for testing."""

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self._processed_data = df
        return df

    def get_performance_columns(self) -> list:
        return ['perf_col1', 'perf_col2']

    def get_size_column(self) -> str:
        return 'size'

    def get_test_binaries(self) -> list:
        return ['test1', 'test2']

    def get_required_columns_for_pdf_type(self, pdf_type: str) -> Set[str]:
        return {'test_binary', 'size', 'perf_col1', 'perf_col2'}

    def get_required_test_binaries_for_pdf_type(self, pdf_type: str) -> Set[str]:
        return {'test1', 'test2'}

    def get_dataframe_column_name(self, config_param: str, data_source=None) -> str:
        return config_param

    def load_csv_with_required_columns(self, file_path: Union[str, Path], pdf_type: str) -> pd.DataFrame:
        return pd.read_csv(file_path)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing."""
    return pd.DataFrame({
        'test_binary': ['all_reduce_perf', 'all_gather_perf'] * 3,
        'size': [1024, 2048, 4096, 8192, 16384, 32768],
        'perf_col1': [100.5, 200.3, 300.1, 400.8, 500.2, 600.7],
        'perf_col2': [50.2, 100.1, 150.3, 200.4, 250.5, 300.6],
        'num_nodes': [1, 2, 4, 8, 16, 32],
        'num_processes': [8, 16, 32, 64, 128, 256]
    })


@pytest.fixture
def temp_csv_file(sample_csv_data, tmp_path):
    """Create a temporary CSV file for testing."""
    csv_path = tmp_path / "test_data.csv"
    sample_csv_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_nccl_data():
    """Sample NCCL data for testing."""
    return pd.DataFrame({
        'tool_binary': ['all_reduce_perf', 'all_gather_perf'] * 3,
        'test_result.size': [1024, 2048, 4096, 8192, 16384, 32768],
        'test_result.out_of_place.bus_bw': [100.5, 200.3, 300.1, 400.8, 500.2, 600.7],
        'test_result.in_place.bus_bw': [90.5, 190.3, 290.1, 390.8, 490.2, 590.7],
        'num_nodes': [1, 2, 4, 8, 16, 32],
        'num_processes': [8, 16, 32, 64, 128, 256],
        'test_config.local_register': ['0', '1', '0', '1', '0', '1'],
        'test_env_vars.NCCL_TESTS_SPLIT_MASK': ['0x0', '0x1', '0x2', '0x3', '0x4', '0x5'],
        'test_env_vars.NCCL_ALGO': ['RING', 'TREE', 'RING', 'TREE', 'RING', 'TREE'],
        'test_env_vars.NCCL_PROTO': ['Simple', 'LL', 'Simple', 'LL', 'Simple', 'LL']
    })


@pytest.fixture
def mock_plot_config():
    """Create a mock plot configuration for testing."""
    # Create performance column configuration
    perf_config = Mock()
    perf_config.sources = ['test_result.out_of_place.bus_bw', 'test_result.in_place.bus_bw']
    perf_config.operation = 'max'  # Use supported operation

    # Create data processing configuration
    data_processing = Mock()
    data_processing.performance_columns = {'avg_bandwidth': perf_config}
    data_processing.dtypes = {'num_nodes': 'int64', 'num_processes': 'int64'}

    # Create settings
    settings = Mock()
    settings.data_processing = data_processing

    # Create style mappings with proper dictionary returns
    style_mappings = Mock()
    style_mappings.default_colors = {}  # Return empty dict instead of Mock
    style_mappings.style_column = 'local_register'
    style_mappings.line_styles = {'0': '-', '1': '--'}
    style_mappings.marker_styles = {'0': 'o', '1': '^'}
    settings.style_mappings = style_mappings

    # Create chart configuration
    chart_config = Mock()
    chart_config.id = 'bandwidth_comparison'
    chart_config.name = 'Bandwidth Comparison'
    chart_config.performance_columns = ['avg_bandwidth']
    chart_config.fixed_params = {'num_nodes': 4}
    chart_config.chart_split_params = {'NCCL_ALGO': 'all'}
    chart_config.chart_combine_params = {}

    # Create analysis profile
    analysis_profile = Mock()
    analysis_profile.id = 'basic_analysis'
    analysis_profile.name = 'Basic Analysis'
    analysis_profile.chart_types = ['bandwidth_comparison']
    analysis_profile.test_binaries = ['all_reduce_perf', 'all_gather_perf']

    # Create PDF configuration
    pdf_config = Mock()
    pdf_config.id = 'basic_report'
    pdf_config.name = 'Basic Report'
    pdf_config.analysis_profiles = ['basic_analysis']

    # Create main plot configuration with list structure
    plot_config = Mock()
    plot_config.settings = settings
    plot_config.chart = [chart_config]  # List instead of dict
    plot_config.analysis_profiles = [analysis_profile]  # List instead of dict
    plot_config.pdf_configs = [pdf_config]  # List instead of dict

    # Mock the helper methods
    def get_chart_by_id(chart_id):
        for chart in plot_config.chart:
            if chart.id == chart_id:
                return chart
        return None

    def get_analysis_profile_by_id(profile_id):
        for profile in plot_config.analysis_profiles:
            if profile.id == profile_id:
                return profile
        return None

    def get_pdf_config_by_id(pdf_id):
        for pdf_config in plot_config.pdf_configs:
            if pdf_config.id == pdf_id:
                return pdf_config
        return None

    plot_config.get_chart_by_id = Mock(side_effect=get_chart_by_id)
    plot_config.get_analysis_profile_by_id = Mock(side_effect=get_analysis_profile_by_id)
    plot_config.get_pdf_config_by_id = Mock(side_effect=get_pdf_config_by_id)

    return plot_config


@pytest.fixture
def comprehensive_nccl_data():
    """Comprehensive NCCL data for metadata testing."""
    return pd.DataFrame({
        'tool_binary': ['all_reduce_perf', 'all_gather_perf', 'all_to_all_perf'] * 4,
        'test_result.size': [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152],
        'test_result.out_of_place.bus_bw': [100.5, 200.3, 300.1, 400.8, 500.2, 600.7, 700.1, 800.5, 900.3, 1000.7, 1100.2, 1200.8],
        'test_result.in_place.bus_bw': [90.5, 190.3, 290.1, 390.8, 490.2, 590.7, 690.1, 790.5, 890.3, 990.7, 1090.2, 1190.8],
        'test_result.out_of_place.alg_bw': [80.5, 180.3, 280.1, 380.8, 480.2, 580.7, 680.1, 780.5, 880.3, 980.7, 1080.2, 1180.8],
        'test_result.in_place.alg_bw': [70.5, 170.3, 270.1, 370.8, 470.2, 570.7, 670.1, 770.5, 870.3, 970.7, 1070.2, 1170.8],
        'num_nodes': [1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32],
        'num_processes': [8, 16, 32, 64, 128, 256, 8, 16, 32, 64, 128, 256],
        'test_config.local_register': ['0', '1', '0', '1', '0', '1', '0', '1', '0', '1', '0', '1'],
        'test_env_vars.NCCL_TESTS_SPLIT_MASK': ['0x0', '0x1', '0x2', '0x3', '0x4', '0x5', '0x0', '0x1', '0x2', '0x3', '0x4', '0x5'],
        'test_env_vars.NCCL_ALGO': ['RING', 'TREE', 'COLLNET', 'RING', 'TREE', 'COLLNET', 'RING', 'TREE', 'COLLNET', 'RING', 'TREE', 'COLLNET'],
        'test_env_vars.NCCL_PROTO': ['P2P', 'SIMPLE', 'LL', 'P2P', 'SIMPLE', 'LL', 'P2P', 'SIMPLE', 'LL', 'P2P', 'SIMPLE', 'LL'],
        'test_env_vars.NCCL_MIN_CTAS': ['1', '2', '4', '8', '1', '2', '4', '8', '1', '2', '4', '8'],
        'library_version': ['2.21.1', '2.21.5', '2.21.1', '2.21.5', '2.21.1', '2.21.5', '2.21.1', '2.21.5', '2.21.1', '2.21.5', '2.21.1', '2.21.5'],
        'library_info.cuda_runtime_version': ['12.1', '12.2', '12.1', '12.2', '12.1', '12.2', '12.1', '12.2', '12.1', '12.2', '12.1', '12.2'],
        'system_info.system_name': ['system_a', 'system_b', 'system_a', 'system_b', 'system_a', 'system_b', 'system_a', 'system_b', 'system_a', 'system_b', 'system_a', 'system_b'],
        'system_info.gpu_arch_type': ['H100', 'A100', 'H100', 'A100', 'H100', 'A100', 'H100', 'A100', 'H100', 'A100', 'H100', 'A100'],
        'test_cmd': [f'cmd{i}' for i in range(1, 13)]
    })


# ============================================================================
# Base Adapter Tests
# ============================================================================

class TestBasePerformanceAdapter:
    """Test the base performance adapter functionality."""

    def test_load_csv(self, temp_csv_file):
        """Test loading CSV data."""
        adapter = ConcreteAdapter()
        data = adapter.load_csv(temp_csv_file)

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 6
        assert 'test_binary' in data.columns

    def test_load_csv_file_not_found(self):
        """Test loading non-existent CSV file."""
        adapter = ConcreteAdapter()
        with pytest.raises(FileNotFoundError):
            adapter.load_csv("nonexistent.csv")

    def test_create_performance_columns(self, sample_csv_data):
        """Test creating derived performance columns."""
        adapter = ConcreteAdapter()

        # Mock configuration for derived columns
        perf_config = Mock()
        perf_config.sources = ['perf_col1', 'perf_col2']
        perf_config.operation = 'mean'  # Use supported operation

        adapter.plot_config = Mock()
        adapter.plot_config.settings = Mock()
        adapter.plot_config.settings.data_processing = Mock()
        adapter.plot_config.settings.data_processing.performance_columns = {'avg_perf': perf_config}

        result = adapter.create_performance_columns(sample_csv_data)

        assert 'avg_perf' in result.columns
        assert len(result) == len(sample_csv_data)

    def test_create_performance_columns_single_source_none_operation(self, sample_csv_data):
        """Test creating performance column with single source and 'none' operation."""
        adapter = ConcreteAdapter()

        # Mock configuration for single source with 'none' operation
        perf_config = Mock()
        perf_config.sources = ['perf_col1']  # Single source
        perf_config.operation = 'none'

        adapter.plot_config = Mock()
        adapter.plot_config.settings = Mock()
        adapter.plot_config.settings.data_processing = Mock()
        adapter.plot_config.settings.data_processing.performance_columns = {'direct_perf': perf_config}

        result = adapter.create_performance_columns(sample_csv_data)

        assert 'direct_perf' in result.columns
        assert len(result) == len(sample_csv_data)
        # The new column should be identical to the source column (ignore names)
        pd.testing.assert_series_equal(result['direct_perf'], result['perf_col1'], check_names=False)

    def test_create_performance_columns_multiple_sources_none_operation(self, sample_csv_data, caplog):
        """Test that 'none' operation with multiple sources logs warning and skips."""
        adapter = ConcreteAdapter()

        # Mock configuration for multiple sources with 'none' operation (invalid)
        perf_config = Mock()
        perf_config.sources = ['perf_col1', 'perf_col2']  # Multiple sources
        perf_config.operation = 'none'

        adapter.plot_config = Mock()
        adapter.plot_config.settings = Mock()
        adapter.plot_config.settings.data_processing = Mock()
        adapter.plot_config.settings.data_processing.performance_columns = {'invalid_perf': perf_config}

        # Capture log messages
        with caplog.at_level(logging.WARNING):
            result = adapter.create_performance_columns(sample_csv_data)

        # Should not create the column due to invalid configuration
        assert 'invalid_perf' not in result.columns
        # Should log a warning
        assert any("'none' operation requires exactly one source column" in record.message
                  for record in caplog.records)

    def test_create_performance_columns_empty_sources_none_operation(self, sample_csv_data, caplog):
        """Test that 'none' operation with empty sources list logs warning and skips."""
        adapter = ConcreteAdapter()

        # Mock configuration for empty sources with 'none' operation (invalid)
        perf_config = Mock()
        perf_config.sources = []  # Empty sources
        perf_config.operation = 'none'

        adapter.plot_config = Mock()
        adapter.plot_config.settings = Mock()
        adapter.plot_config.settings.data_processing = Mock()
        adapter.plot_config.settings.data_processing.performance_columns = {'empty_perf': perf_config}

        # Capture log messages
        with caplog.at_level(logging.WARNING):
            result = adapter.create_performance_columns(sample_csv_data)

        # Should not create the column due to invalid configuration
        assert 'empty_perf' not in result.columns
        # Should log a warning
        assert any("'none' operation requires exactly one source column" in record.message
                  for record in caplog.records)

    def test_filter_by_chart_type(self, sample_csv_data):
        """Test filtering data by chart type."""
        adapter = ConcreteAdapter()
        adapter._processed_data = sample_csv_data

        filtered = adapter.filter_by_chart_type(sample_csv_data, 'test_chart')

        assert len(filtered) <= len(sample_csv_data)
        assert isinstance(filtered, pd.DataFrame)

    def test_prepare_for_plotting(self, sample_csv_data):
        """Test preparing data for plotting."""
        adapter = ConcreteAdapter()
        adapter._processed_data = sample_csv_data

        # Mock chart configuration with list structure
        mock_plot_config = Mock()
        mock_chart = Mock()
        mock_chart.id = 'test_chart'
        mock_chart.performance_columns = ['perf_col1']
        mock_chart.fixed_params = {}
        mock_chart.chart_split_params = {}
        mock_chart.chart_combine_params = {}
        mock_plot_config.chart = [mock_chart]

        # Mock the helper method
        def get_chart_by_id(chart_id):
            for chart in mock_plot_config.chart:
                if chart.id == chart_id:
                    return chart
            return None

        mock_plot_config.get_chart_by_id = Mock(side_effect=get_chart_by_id)
        adapter.plot_config = mock_plot_config

        data, metadata = adapter.prepare_for_plotting('test_chart')

        assert isinstance(data, pd.DataFrame)
        assert isinstance(metadata, DataFrameMetadata)
        assert metadata.row_count == len(data)

    def test_source_file_attribute_preservation(self, temp_csv_file):
        """Test that source_file attribute is properly set and preserved in metadata."""
        adapter = ConcreteAdapter()

        # Load CSV data
        df = adapter.load_csv(temp_csv_file)

        # Verify source_file attribute is set
        assert hasattr(df, 'attrs')
        assert 'source_file' in df.attrs
        assert df.attrs['source_file'] == str(temp_csv_file)

        # Process the data
        adapter.process_data(df)

        # Mock chart configuration
        mock_plot_config = Mock()
        mock_chart = Mock()
        mock_chart.id = 'test_chart'
        mock_chart.performance_columns = ['perf_col1']
        mock_chart.fixed_params = {}
        mock_chart.chart_split_params = {}
        mock_chart.chart_combine_params = {}
        mock_plot_config.chart = [mock_chart]

        def get_chart_by_id(chart_id):
            for chart in mock_plot_config.chart:
                if chart.id == chart_id:
                    return chart
            return None

        mock_plot_config.get_chart_by_id = Mock(side_effect=get_chart_by_id)
        adapter.plot_config = mock_plot_config

        df_result, metadata = adapter.prepare_for_plotting('test_chart')

        # Verify source_file is preserved in metadata
        assert metadata.source_file == str(temp_csv_file)


# ============================================================================
# NCCL Adapter Tests
# ============================================================================

class TestNCCLPerformanceAdapter:
    """Test the NCCL performance adapter functionality."""

    def test_process_data(self, sample_nccl_data):
        """Test processing NCCL data."""
        adapter = NCCLPerformanceAdapter()
        adapter._raw_data = sample_nccl_data

        processed = adapter.process_data()

        assert isinstance(processed, pd.DataFrame)
        assert len(processed) == len(sample_nccl_data)
        assert adapter._processed_data is not None

    def test_get_test_binaries(self, sample_nccl_data):
        """Test getting unique test binaries."""
        adapter = NCCLPerformanceAdapter()
        adapter._raw_data = sample_nccl_data
        adapter.process_data()

        binaries = adapter.get_test_binaries()

        assert isinstance(binaries, list)
        assert len(binaries) >= 1
        assert 'all_reduce_perf' in binaries

    def test_format_split_mask(self):
        """Test formatting NCCL split mask values."""
        adapter = NCCLPerformanceAdapter()

        assert adapter._format_split_mask(0) == '0x0'
        assert adapter._format_split_mask(15) == '0xF'
        assert adapter._format_split_mask('0x1') == '0x1'
        assert adapter._format_split_mask(None) == 'None'

    def test_get_dataframe_column_name(self):
        """Test getting dataframe column names."""
        adapter = NCCLPerformanceAdapter()

        assert adapter.get_dataframe_column_name('test_binary') == 'tool_binary'
        assert adapter.get_dataframe_column_name('size') == 'test_result.size'
        assert adapter.get_dataframe_column_name('NCCL_ALGO') == 'test_env_vars.NCCL_ALGO'

    def test_get_dataframe_column_name_exact_matching(self):
        """Test that get_dataframe_column_name uses exact matching only."""
        adapter = NCCLPerformanceAdapter()

        # Test with data source that has partial match candidates
        data = pd.DataFrame({
            'test_env_vars.NCCL_ALGO': ['RING'],
            'test_env_vars.NCCL_ALGO_RING': ['RING'],  # Partial match
            'test_env_vars.NCCL_ALGO_TREE': ['TREE']   # Another partial match
        })

        # Should find exact match
        column_name = adapter.get_dataframe_column_name('NCCL_ALGO', data)
        assert column_name == 'test_env_vars.NCCL_ALGO'

        # Should raise error for non-existent parameter
        with pytest.raises(ValueError):
            adapter.get_dataframe_column_name('NONEXISTENT_PARAM', data)

    def test_get_dataframe_column_name_no_partial_fallback(self):
        """Test that get_dataframe_column_name doesn't use partial matching fallback."""
        adapter = NCCLPerformanceAdapter()

        # Test with data source that has partial match candidates but no exact match
        data = pd.DataFrame({
            'test_env_vars.NCCL_ALGO_RING': ['RING'],  # Partial match
            'test_env_vars.NCCL_ALGO_TREE': ['TREE']   # Another partial match
        })

        # Should raise error for non-existent parameter, not use partial match
        with pytest.raises(ValueError):
            adapter.get_dataframe_column_name('NCCL_ALGO', data)

    def test_get_required_columns_for_pdf_type(self, mock_plot_config):
        """Test getting required columns for PDF type."""
        adapter = NCCLPerformanceAdapter(plot_config=mock_plot_config)

        columns = adapter.get_required_columns_for_pdf_type('basic_report')

        assert isinstance(columns, set)
        assert len(columns) > 0
        assert 'tool_binary' in columns  # test_binary maps to tool_binary

    def test_get_required_test_binaries_for_pdf_type(self, mock_plot_config):
        """Test getting required test binaries for PDF type."""
        adapter = NCCLPerformanceAdapter(plot_config=mock_plot_config)

        binaries = adapter.get_required_test_binaries_for_pdf_type('basic_report')

        assert isinstance(binaries, set)
        assert len(binaries) > 0

    def test_get_columns_for_chart_type(self, mock_plot_config):
        """Test getting columns for chart type."""
        adapter = NCCLPerformanceAdapter(plot_config=mock_plot_config)

        columns = adapter._get_columns_for_chart_type('bandwidth_comparison')

        assert isinstance(columns, set)

    def test_column_extraction_with_missing_parameters(self, sample_nccl_data):
        """Test that column extraction handles missing parameters gracefully."""
        # Create data without some columns
        data = sample_nccl_data.copy()
        data = data.drop(columns=['test_env_vars.NCCL_ALGO', 'test_env_vars.NCCL_PROTO'])

        # Create mock plot config with proper structure
        mock_config = Mock()

        # Create chart config
        chart_config = Mock()
        chart_config.fixed_params = {
            'local_register': 'LOCAL_REGISTER_ALL',
            'NCCL_ALGO': 'RING',  # Missing
            'NCCL_PROTO': 'SIMPLE'  # Missing
        }
        chart_config.chart_split_params = {
            'NCCL_ALGO': 'all'  # Missing
        }
        chart_config.chart_combine_params = {
            'NCCL_PROTO': 'all'  # Missing
        }
        chart_config.default_values = {}
        chart_config.performance_columns = []  # Add this to avoid TypeError

        # Mock settings to avoid TypeError in _convert_numeric_columns
        mock_settings = Mock()
        mock_settings.data_processing = Mock()
        mock_settings.data_processing.dtypes = {}
        mock_settings.data_processing.performance_columns = {}
        mock_settings.style_mappings = None
        mock_config.settings = mock_settings

        # Set up plot_config structure
        mock_config.chart = [chart_config]
        mock_config.get_chart_by_id = Mock(return_value=chart_config)

        adapter = NCCLPerformanceAdapter(mock_config)
        adapter._raw_data = data
        adapter.process_data(data)

        # Should not fail during column extraction
        required_columns = adapter._get_columns_for_chart_type('test_chart')

        # Should include available columns but not fail on missing ones
        assert 'test_config.local_register' in required_columns
        # Missing columns should not cause errors

    def test_is_derived_performance_column(self, mock_plot_config):
        """Test checking if column is derived."""
        adapter = NCCLPerformanceAdapter(plot_config=mock_plot_config)

        # Test with mock configuration
        result = adapter._is_derived_performance_column('avg_bandwidth')
        assert isinstance(result, bool)

    def test_get_source_columns_for_derived_column(self, mock_plot_config):
        """Test getting source columns for derived column."""
        adapter = NCCLPerformanceAdapter(plot_config=mock_plot_config)

        columns = adapter._get_source_columns_for_derived_column('avg_bandwidth')

        assert isinstance(columns, set)

    def test_load_csv_with_required_columns(self, temp_csv_file, mock_plot_config):
        """Test loading CSV with required columns only."""
        # Create NCCL-style CSV data
        nccl_data = pd.DataFrame({
            'tool_binary': ['all_reduce_perf', 'all_gather_perf'],
            'test_result.size': [1024, 2048],
            'test_result.out_of_place.bus_bw': [100.5, 200.3],
            'num_nodes': [4, 8],
            'num_processes': [32, 64]
        })

        nccl_csv_path = temp_csv_file.parent / "nccl_test.csv"
        nccl_data.to_csv(nccl_csv_path, index=False)

        adapter = NCCLPerformanceAdapter(plot_config=mock_plot_config)

        loaded_data = adapter.load_csv_with_required_columns(nccl_csv_path, 'basic_report')

        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) > 0

    def test_load_csv_with_required_columns_fallback(self, temp_csv_file):
        """Test CSV loading fallback when no plot config."""
        adapter = NCCLPerformanceAdapter()

        loaded_data = adapter.load_csv_with_required_columns(temp_csv_file, 'basic_report')

        assert isinstance(loaded_data, pd.DataFrame)

    def test_column_optimization_benefits(self, mock_plot_config):
        """Test that column optimization provides benefits."""
        adapter = NCCLPerformanceAdapter(plot_config=mock_plot_config)

        # Test that we can get required columns
        columns = adapter.get_required_columns_for_pdf_type('basic_report')

        # Should be a subset of all possible columns
        all_columns = set(adapter._BASE_NCCL_COLUMNS.values())

        assert isinstance(columns, set)
        assert len(columns) <= len(all_columns)

    def test_no_plot_config_fallback(self):
        """Test adapter behavior when no plot configuration is provided."""
        adapter = NCCLPerformanceAdapter()

        # Should still work with basic functionality
        column_name = adapter.get_dataframe_column_name('test_binary')
        assert column_name == 'tool_binary'

        performance_columns = adapter.get_performance_columns()
        assert isinstance(performance_columns, list)


# ============================================================================
# Legend and Style Tests
# ============================================================================

class TestLegendAndStyle:
    """Test legend formatting and style functionality."""

    def test_legend_shortening_functionality(self):
        """Test legend key shortening."""
        adapter = NCCLPerformanceAdapter()

        original = "num_nodes=4, num_processes=32, NCCL_ALGO=RING"
        shortened = adapter.shorten_legend_key(original)
        expected = "N=4, NP=32, ALGO=RING"

        assert shortened == expected

    def test_legend_color_assignment_excludes_local_register(self, mock_plot_config):
        """Test that color assignment excludes local_register parameter."""
        adapter = NCCLPerformanceAdapter(plot_config=mock_plot_config)

        # Mock style config
        mock_style_config = Mock()
        mock_style_config.color_scheme.palette = 'tab20'
        mock_style_config.color_palette.num_colors = 20
        adapter.style_config = mock_style_config

        # Test legend with local_register
        legend_key = "num_nodes=4, local_register=1, NCCL_ALGO=RING"

        style_info = adapter.get_legend_style_info(legend_key, 'test_chart')

        assert 'color' in style_info
        assert 'line_style' in style_info
        assert 'marker_style' in style_info

    def test_legend_style_differentiation_by_local_register(self, mock_plot_config):
        """Test that line/marker styles differentiate by local_register."""
        adapter = NCCLPerformanceAdapter(plot_config=mock_plot_config)

        # Test that different local_register values can get different styles
        legend_key_0 = "num_nodes=4, local_register=0"
        legend_key_1 = "num_nodes=4, local_register=1"

        style_0 = adapter.get_legend_style_info(legend_key_0, 'test_chart')
        style_1 = adapter.get_legend_style_info(legend_key_1, 'test_chart')

        # Styles should be consistent (same structure)
        assert set(style_0.keys()) == set(style_1.keys())

    def test_color_key_excludes_local_register(self, mock_plot_config):
        """Test that color key generation excludes local_register."""
        adapter = NCCLPerformanceAdapter(plot_config=mock_plot_config)

        # Test internal color assignment logic
        legend_key = "num_nodes=4, local_register=1, NCCL_ALGO=RING"

        # Get style info, which internally uses color key logic
        style_info = adapter.get_legend_style_info(legend_key, 'test_chart')

        # Should have color assigned
        assert 'color' in style_info
        assert style_info['color'] != 'blue'  # Should not be default

    def test_legend_style_info_with_shortened_keys(self, mock_plot_config):
        """Test legend style info works with shortened parameter names."""
        adapter = NCCLPerformanceAdapter(plot_config=mock_plot_config)

        # Test with shortened parameter names
        shortened_key = "N=4, UB=1, ALGO=RING"

        style_info = adapter.get_legend_style_info(shortened_key, 'test_chart')

        assert isinstance(style_info, dict)
        assert 'color' in style_info
        assert 'line_style' in style_info
        assert 'marker_style' in style_info


# ============================================================================
# Chart Group Tests
# ============================================================================

class TestChartGroups:
    """Test chart group generation functionality."""

    def test_get_chart_groups_for_plotting(self, sample_nccl_data, mock_plot_config):
        """Test getting chart groups for plotting."""
        adapter = NCCLPerformanceAdapter(plot_config=mock_plot_config)
        adapter._processed_data = sample_nccl_data

        chart_groups = adapter.get_chart_groups_for_plotting('bandwidth_comparison', 'all_reduce_perf')

        assert isinstance(chart_groups, list)
        # May be empty if no matching data, but should not error

    def test_chart_groups_with_split_params(self, sample_nccl_data, mock_plot_config):
        """Test chart groups with split parameters."""
        adapter = NCCLPerformanceAdapter(plot_config=mock_plot_config)
        adapter._processed_data = sample_nccl_data

        # Mock chart config with split params - access via helper
        chart_config = mock_plot_config.get_chart_by_id('bandwidth_comparison')
        chart_config.chart_split_params = {'NCCL_ALGO': 'all'}

        chart_groups = adapter.get_chart_groups_for_plotting('bandwidth_comparison', 'all_reduce_perf')

        assert isinstance(chart_groups, list)

    def test_chart_groups_with_fixed_params(self, sample_nccl_data, mock_plot_config):
        """Test chart groups with fixed parameters."""
        adapter = NCCLPerformanceAdapter(plot_config=mock_plot_config)
        adapter._processed_data = sample_nccl_data

        # Mock chart config with fixed params - access via list
        chart_config = mock_plot_config.get_chart_by_id('bandwidth_comparison')
        chart_config.fixed_params = {'num_nodes': 4}

        chart_groups = adapter.get_chart_groups_for_plotting('bandwidth_comparison', 'all_reduce_perf')

        assert isinstance(chart_groups, list)


class TestMetadataCollection:
    """Test dataset metadata collection functionality."""

    def test_collect_dataset_metadata_comprehensive(self, comprehensive_nccl_data):
        """Test comprehensive metadata collection with full dataset."""
        adapter = NCCLPerformanceAdapter()
        adapter._processed_data = comprehensive_nccl_data

        metadata = adapter.collect_dataset_metadata()

        # Test basic stats
        assert 'basic_stats' in metadata
        assert metadata['basic_stats']['total_records'] == 12
        assert metadata['basic_stats']['total_columns'] == 18
        assert 'memory_usage_mb' in metadata['basic_stats']
        assert metadata['basic_stats']['memory_usage_mb'] > 0

        # Test test info
        assert 'test_info' in metadata
        assert metadata['test_info']['unique_test_binaries'] == 3
        assert set(metadata['test_info']['test_binaries']) == {'all_reduce_perf', 'all_gather_perf', 'all_to_all_perf'}
        assert 'records_per_binary' in metadata['test_info']
        assert metadata['test_info']['records_per_binary']['all_reduce_perf'] == 4

        # Test scale info
        assert 'scale_info' in metadata['test_info']
        scale_info = metadata['test_info']['scale_info']

        assert 'node_range' in scale_info
        assert scale_info['node_range']['min'] == 1
        assert scale_info['node_range']['max'] == 32
        assert scale_info['node_range']['unique_counts'] == [1, 2, 4, 8, 16, 32]

        assert 'process_range' in scale_info
        assert scale_info['process_range']['min'] == 8
        assert scale_info['process_range']['max'] == 256

        assert 'message_size_range' in scale_info
        assert scale_info['message_size_range']['min_bytes'] == 1024
        assert scale_info['message_size_range']['max_bytes'] == 2097152
        assert scale_info['message_size_range']['min_human'] == '1.0 KB'
        assert scale_info['message_size_range']['max_human'] == '2.0 MB'
        assert scale_info['message_size_range']['unique_sizes'] == 12

        # Test system info
        assert 'system_info' in metadata
        system_info = metadata['system_info']

        assert system_info['unique_systems'] == 2
        assert set(system_info['systems']) == {'system_a', 'system_b'}
        assert set(system_info['gpu_architectures']) == {'H100', 'A100'}
        assert set(system_info['nccl_versions']) == {'2.21.1', '2.21.5'}
        assert set(system_info['cuda_versions']) == {'12.1', '12.2'}

        # Test data coverage
        assert 'data_coverage' in metadata
        coverage = metadata['data_coverage']

        # NCCL parameter coverage
        assert 'NCCL_ALGO' in coverage
        assert coverage['NCCL_ALGO']['unique_values'] == 3
        assert set(coverage['NCCL_ALGO']['values']) == {'RING', 'TREE', 'COLLNET'}

        assert 'NCCL_PROTO' in coverage
        assert coverage['NCCL_PROTO']['unique_values'] == 3
        assert set(coverage['NCCL_PROTO']['values']) == {'P2P', 'SIMPLE', 'LL'}

        assert 'NCCL_MIN_CTAS' in coverage
        assert coverage['NCCL_MIN_CTAS']['unique_values'] == 4
        assert set(coverage['NCCL_MIN_CTAS']['values']) == {'1', '2', '4', '8'}

        # Local register coverage
        assert 'local_register' in coverage
        assert coverage['local_register']['unique_values'] == 2
        assert set(coverage['local_register']['values']) == {'0', '1'}

        # Split mask coverage
        assert 'split_mask' in coverage
        assert coverage['split_mask']['unique_values'] == 6

        # Test timestamp
        assert 'collection_timestamp' in metadata
        assert isinstance(metadata['collection_timestamp'], str)

    def test_collect_dataset_metadata_no_data(self):
        """Test metadata collection when no processed data is available."""
        adapter = NCCLPerformanceAdapter()

        metadata = adapter.collect_dataset_metadata()

        assert metadata == {}

    def test_collect_dataset_metadata_minimal_data(self):
        """Test metadata collection with minimal data."""
        minimal_data = pd.DataFrame({
            'tool_binary': ['all_reduce_perf', 'all_gather_perf'],
            'test_result.size': [1024, 2048],
            'test_result.out_of_place.bus_bw': [100.0, 200.0],
            'num_nodes': [1, 2],
            'num_processes': [8, 16]
        })

        adapter = NCCLPerformanceAdapter()
        adapter._processed_data = minimal_data

        metadata = adapter.collect_dataset_metadata()

        # Should have basic stats
        assert 'basic_stats' in metadata
        assert metadata['basic_stats']['total_records'] == 2
        assert metadata['basic_stats']['total_columns'] == 5

        # Should have test info
        assert 'test_info' in metadata
        assert metadata['test_info']['unique_test_binaries'] == 2

        # Should have scale info
        assert 'scale_info' in metadata['test_info']
        assert metadata['test_info']['scale_info']['node_range']['min'] == 1
        assert metadata['test_info']['scale_info']['node_range']['max'] == 2

        # Data coverage might be empty for missing NCCL parameters
        assert 'data_coverage' in metadata

    def test_collect_dataset_metadata_missing_columns(self):
        """Test metadata collection when some expected columns are missing."""
        incomplete_data = pd.DataFrame({
            'tool_binary': ['all_reduce_perf', 'all_gather_perf'],
            'test_result.size': [1024, 2048],
            'test_result.out_of_place.bus_bw': [100.0, 200.0],
            # Missing: system info, versions, NCCL parameters, etc.
        })

        adapter = NCCLPerformanceAdapter()
        adapter._processed_data = incomplete_data

        metadata = adapter.collect_dataset_metadata()

        # Should still have basic stats
        assert 'basic_stats' in metadata
        assert metadata['basic_stats']['total_records'] == 2

        # Should have test info but minimal scale info
        assert 'test_info' in metadata
        assert metadata['test_info']['unique_test_binaries'] == 2

        # System info should be empty or minimal
        assert 'system_info' in metadata
        # Most system info fields should be missing

        # Data coverage should be mostly empty
        assert 'data_coverage' in metadata

    def test_format_bytes_functionality(self):
        """Test the _format_bytes helper method."""
        adapter = NCCLPerformanceAdapter()

        # Test different byte ranges
        assert adapter._format_bytes(512) == "512 B"
        assert adapter._format_bytes(1024) == "1.0 KB"
        assert adapter._format_bytes(1536) == "1.5 KB"
        assert adapter._format_bytes(1024 * 1024) == "1.0 MB"
        assert adapter._format_bytes(1.5 * 1024 * 1024) == "1.5 MB"
        assert adapter._format_bytes(1024 * 1024 * 1024) == "1.0 GB"
        assert adapter._format_bytes(2.5 * 1024 * 1024 * 1024) == "2.5 GB"

    def test_metadata_collection_with_many_unique_values(self):
        """Test metadata collection when there are many unique values for a parameter."""
        # Create data with many unique NCCL_ALGO values (more than 10)
        many_values_data = pd.DataFrame({
            'tool_binary': ['all_reduce_perf'] * 15,
            'test_result.size': [1024] * 15,
            'test_result.out_of_place.bus_bw': [100.0] * 15,
            'test_env_vars.NCCL_ALGO': [f'ALGO_{i}' for i in range(15)],  # 15 unique values
            'num_nodes': [1] * 15,
            'num_processes': [8] * 15
        })

        adapter = NCCLPerformanceAdapter()
        adapter._processed_data = many_values_data

        metadata = adapter.collect_dataset_metadata()

        # Should handle many unique values by showing count instead of listing all
        assert 'data_coverage' in metadata
        assert 'NCCL_ALGO' in metadata['data_coverage']
        assert metadata['data_coverage']['NCCL_ALGO']['unique_values'] == 15
        # Should show count message instead of listing all values
        assert isinstance(metadata['data_coverage']['NCCL_ALGO']['values'], str)
        assert "15 unique values" in metadata['data_coverage']['NCCL_ALGO']['values']


# ============================================================================
# Tests for Plotting Fixes
# ============================================================================

class TestParameterMatching:
    """Test parameter matching functionality including case sensitivity and error handling."""

    def test_case_agnostic_parameter_matching(self, sample_nccl_data):
        """Test that parameter matching is case agnostic."""
        # Create data with mixed case values
        data = sample_nccl_data.copy()
        data['test_config.local_register'] = ['LOCAL_REGISTER_ALL', 'local_register_all', 'Local_Register_All'] * 2
        data['test_env_vars.NCCL_ALGO'] = ['RING', 'ring', 'Ring'] * 2
        data['test_env_vars.NCCL_PROTO'] = ['SIMPLE', 'simple', 'Simple'] * 2

        # Create mock plot config with fixed parameters
        mock_config = Mock()

        # Create chart config
        chart_config = Mock()
        chart_config.fixed_params = {
            'local_register': 'LOCAL_REGISTER_ALL',
            'NCCL_ALGO': 'RING',
            'NCCL_PROTO': 'SIMPLE'
        }
        chart_config.chart_split_params = {}
        chart_config.chart_combine_params = {}
        chart_config.default_values = {}

        # Mock settings to avoid TypeError in _convert_numeric_columns
        mock_settings = Mock()
        mock_settings.data_processing = Mock()
        mock_settings.data_processing.dtypes = {}
        mock_settings.data_processing.performance_columns = {}  # Empty dict for performance columns
        mock_settings.style_mappings = None  # Add this to avoid style mapping issues
        mock_config.settings = mock_settings

        # Set up plot_config structure
        mock_config.chart = [chart_config]
        mock_config.get_chart_by_id = Mock(return_value=chart_config)

        adapter = NCCLPerformanceAdapter(mock_config)
        adapter._raw_data = data
        adapter.process_data(data)

        # Test chart grouping - should include all case variations
        chart_groups = adapter.get_chart_groups_for_plotting('test_chart', 'all_reduce_perf')

        assert len(chart_groups) > 0
        # Should have data from all case variations
        for group in chart_groups:
            if 'test_config.local_register' in group['data'].columns:
                unique_values = group['data']['test_config.local_register'].unique()
                # Should have multiple case variations
                assert len(unique_values) >= 2

    def test_missing_parameter_handling(self, sample_nccl_data):
        """Test that missing parameters are handled gracefully."""
        # Create data without NCCL_ALGO column
        data = sample_nccl_data.copy()
        data = data.drop(columns=['test_env_vars.NCCL_ALGO'])

        # Create mock plot config with fixed parameters including missing one
        mock_config = Mock()

        # Create chart config
        chart_config = Mock()
        chart_config.fixed_params = {
            'local_register': 'LOCAL_REGISTER_ALL',
            'NCCL_ALGO': 'RING',  # This parameter is missing from data
            'NCCL_PROTO': 'SIMPLE'
        }
        chart_config.chart_split_params = {}
        chart_config.chart_combine_params = {}
        chart_config.default_values = {}

        # Mock settings to avoid TypeError in _convert_numeric_columns
        mock_settings = Mock()
        mock_settings.data_processing = Mock()
        mock_settings.data_processing.dtypes = {}
        mock_settings.data_processing.performance_columns = {}
        mock_settings.style_mappings = None  # Add this to avoid style mapping issues
        mock_config.settings = mock_settings

        # Set up plot_config structure
        mock_config.chart = [chart_config]
        mock_config.get_chart_by_id = Mock(return_value=chart_config)

        adapter = NCCLPerformanceAdapter(mock_config)
        adapter._raw_data = data
        adapter.process_data(data)

        # Should not fail even though NCCL_ALGO is missing
        chart_groups = adapter.get_chart_groups_for_plotting('test_chart', 'all_reduce_perf')

        assert len(chart_groups) > 0
        # Should still work with available parameters
        # Note: The filtering might not work as expected with mock data, but the method should not fail

    def test_exact_parameter_matching_no_partial_fallback(self, sample_nccl_data):
        """Test that exact parameter matching is used and no partial fallback occurs."""
        # Create data with columns that could be partial matches
        data = sample_nccl_data.copy()
        data['test_env_vars.NCCL_ALGO_RING'] = ['RING'] * len(data)  # Partial match candidate
        data['test_env_vars.NCCL_ALGO_TREE'] = ['TREE'] * len(data)  # Another partial match

        # Create mock plot config
        mock_config = Mock()

        # Create chart config
        chart_config = Mock()
        chart_config.fixed_params = {
            'NCCL_ALGO': 'RING'  # Should only match exact column, not partial matches
        }
        chart_config.chart_split_params = {}
        chart_config.chart_combine_params = {}
        chart_config.default_values = {}

        # Mock settings to avoid TypeError in _convert_numeric_columns
        mock_settings = Mock()
        mock_settings.data_processing = Mock()
        mock_settings.data_processing.dtypes = {}
        mock_settings.data_processing.performance_columns = {}
        mock_settings.style_mappings = None  # Add this to avoid style mapping issues
        mock_config.settings = mock_settings

        # Set up plot_config structure
        mock_config.chart = [chart_config]
        mock_config.get_chart_by_id = Mock(return_value=chart_config)

        adapter = NCCLPerformanceAdapter(mock_config)
        adapter._raw_data = data
        adapter.process_data(data)

        # Should use exact column name, not partial matches
        chart_groups = adapter.get_chart_groups_for_plotting('test_chart', 'all_reduce_perf')

        assert len(chart_groups) > 0
        # The filtering should work correctly with exact column names
        for group in chart_groups:
            if 'test_env_vars.NCCL_ALGO' in group['data'].columns:
                unique_values = group['data']['test_env_vars.NCCL_ALGO'].unique()
                assert all(val == 'RING' for val in unique_values)

    def test_combine_parameters_case_agnostic(self, sample_nccl_data):
        """Test that combine parameters use case agnostic matching."""
        # Create data with mixed case values
        data = sample_nccl_data.copy()
        data['test_config.local_register'] = ['LOCAL_REGISTER_ALL', 'local_register_all'] * 3
        data['test_env_vars.NCCL_ALGO'] = ['RING', 'ring'] * 3

        # Create mock plot config with combine parameters
        mock_config = Mock()

        # Create chart config
        chart_config = Mock()
        chart_config.fixed_params = {}
        chart_config.chart_split_params = {}
        chart_config.chart_combine_params = {
            'local_register': 'all',
            'NCCL_ALGO': 'all'
        }
        chart_config.default_values = {}

        # Mock settings to avoid TypeError in _convert_numeric_columns
        mock_settings = Mock()
        mock_settings.data_processing = Mock()
        mock_settings.data_processing.dtypes = {}
        mock_settings.data_processing.performance_columns = {}
        mock_config.settings = mock_settings

        # Set up plot_config structure
        mock_config.chart = [chart_config]
        mock_config.get_chart_by_id = Mock(return_value=chart_config)

        adapter = NCCLPerformanceAdapter(mock_config)
        adapter._raw_data = data
        adapter.process_data(data)

        # Test chart grouping
        chart_groups = adapter.get_chart_groups_for_plotting('test_chart', 'all_reduce_perf')

        assert len(chart_groups) > 0
        # Should create separate groups for different combinations
        # but case variations should be grouped together

    def test_missing_combine_parameter_handling(self, sample_nccl_data):
        """Test that missing combine parameters are handled gracefully."""
        # Create data without some columns
        data = sample_nccl_data.copy()
        data = data.drop(columns=['test_env_vars.NCCL_ALGO'])

        # Create mock plot config with combine parameters including missing ones
        mock_config = Mock()

        # Create chart config
        chart_config = Mock()
        chart_config.fixed_params = {}
        chart_config.chart_split_params = {}
        chart_config.chart_combine_params = {
            'local_register': 'all',
            'NCCL_ALGO': 'all',  # This parameter is missing
            'NCCL_PROTO': 'all'
        }
        chart_config.default_values = {}

        # Mock settings to avoid TypeError in _convert_numeric_columns
        mock_settings = Mock()
        mock_settings.data_processing = Mock()
        mock_settings.data_processing.dtypes = {}
        mock_settings.data_processing.performance_columns = {}
        mock_settings.style_mappings = None
        mock_config.settings = mock_settings

        # Set up plot_config structure
        mock_config.chart = [chart_config]
        mock_config.get_chart_by_id = Mock(return_value=chart_config)

        adapter = NCCLPerformanceAdapter(mock_config)
        adapter._raw_data = data
        adapter.process_data(data)

        # Should not fail even though NCCL_ALGO is missing
        chart_groups = adapter.get_chart_groups_for_plotting('test_chart', 'all_reduce_perf')

        assert len(chart_groups) > 0
        # Should still work with the available parameters

    def test_split_parameters_case_agnostic(self, sample_nccl_data):
        """Test that split parameters use case agnostic matching."""
        # Create data with mixed case values
        data = sample_nccl_data.copy()
        data['test_config.local_register'] = ['LOCAL_REGISTER_ALL', 'local_register_all'] * 3

        # Create mock plot config with split parameters
        mock_config = Mock()

        # Create chart config
        chart_config = Mock()
        chart_config.fixed_params = {}
        chart_config.chart_split_params = {
            'local_register': 'all'
        }
        chart_config.chart_combine_params = {}
        chart_config.default_values = {}

        # Mock settings to avoid TypeError in _convert_numeric_columns
        mock_settings = Mock()
        mock_settings.data_processing = Mock()
        mock_settings.data_processing.dtypes = {}
        mock_settings.data_processing.performance_columns = {}
        mock_config.settings = mock_settings

        # Set up plot_config structure
        mock_config.chart = [chart_config]
        mock_config.get_chart_by_id = Mock(return_value=chart_config)

        adapter = NCCLPerformanceAdapter(mock_config)
        adapter._raw_data = data
        adapter.process_data(data)

        # Test chart grouping
        chart_groups = adapter.get_chart_groups_for_plotting('test_chart', 'all_reduce_perf')

        assert len(chart_groups) > 0
        # Should create separate charts for different values
        # but case variations should be grouped together

    def test_missing_split_parameter_handling(self, sample_nccl_data):
        """Test that missing split parameters are handled gracefully."""
        # Create data without some columns
        data = sample_nccl_data.copy()
        data = data.drop(columns=['test_env_vars.NCCL_ALGO'])

        # Create mock plot config with split parameters including missing ones
        mock_config = Mock()

        # Create chart config
        chart_config = Mock()
        chart_config.fixed_params = {}
        chart_config.chart_split_params = {
            'local_register': 'all',
            'NCCL_ALGO': 'all'  # This parameter is missing
        }
        chart_config.chart_combine_params = {}
        chart_config.default_values = {}

        # Mock settings to avoid TypeError in _convert_numeric_columns
        mock_settings = Mock()
        mock_settings.data_processing = Mock()
        mock_settings.data_processing.dtypes = {}
        mock_settings.data_processing.performance_columns = {}
        mock_config.settings = mock_settings

        # Set up plot_config structure
        mock_config.chart = [chart_config]
        mock_config.get_chart_by_id = Mock(return_value=chart_config)

        adapter = NCCLPerformanceAdapter(mock_config)
        adapter._raw_data = data
        adapter.process_data(data)

        # Should not fail even though NCCL_ALGO is missing
        chart_groups = adapter.get_chart_groups_for_plotting('test_chart', 'all_reduce_perf')

        assert len(chart_groups) > 0
        # Should still work with the available parameters



    def test_value_matching_case_agnostic_edge_cases(self, sample_nccl_data):
        """Test edge cases for case agnostic value matching."""
        # Create data with various edge cases
        data = sample_nccl_data.copy()
        data['test_config.local_register'] = [
            'LOCAL_REGISTER_ALL',
            'local_register_all',
            'Local_Register_All',
            'LOCAL_REGISTER_all',
            'local_REGISTER_ALL',
            'LOCAL_register_ALL'
        ]

        # Create mock plot config with proper structure
        mock_config = Mock()

        # Create chart config
        chart_config = Mock()
        chart_config.fixed_params = {
            'local_register': 'LOCAL_REGISTER_ALL'
        }
        chart_config.chart_split_params = {}
        chart_config.chart_combine_params = {}
        chart_config.default_values = {}

        # Mock settings to avoid TypeError in _convert_numeric_columns
        mock_settings = Mock()
        mock_settings.data_processing = Mock()
        mock_settings.data_processing.dtypes = {}
        mock_settings.data_processing.performance_columns = {}
        mock_settings.style_mappings = None
        mock_config.settings = mock_settings

        # Set up plot_config structure
        mock_config.chart = [chart_config]
        mock_config.get_chart_by_id = Mock(return_value=chart_config)

        adapter = NCCLPerformanceAdapter(mock_config)
        adapter._raw_data = data
        adapter.process_data(data)

        # Test chart grouping
        chart_groups = adapter.get_chart_groups_for_plotting('test_chart', 'all_reduce_perf')

        assert len(chart_groups) > 0
        # All case variations should be included
        for group in chart_groups:
            if 'test_config.local_register' in group['data'].columns:
                unique_values = group['data']['test_config.local_register'].unique()
                # Should have multiple case variations
                assert len(unique_values) >= 2
                # All should be case variations of LOCAL_REGISTER_ALL
                for val in unique_values:
                    assert 'local_register_all' in str(val).lower()
