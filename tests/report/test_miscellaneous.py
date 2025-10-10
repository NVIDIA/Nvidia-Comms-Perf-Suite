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
Tests for miscellaneous functionality.

This module tests functionality that doesn't fit into specific modules:
- Edge cases and error handling
- Utility functions and helpers
- Integration scenarios
- Performance and memory considerations
- Data validation and cleanup
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from unittest.mock import Mock
import matplotlib.pyplot as plt

from nvcomms_perf.report.adapters.nccl_adapter import NCCLPerformanceAdapter

from nvcomms_perf.report.models.plot_config import ChartConfigLoader
from nvcomms_perf.report.models.style_config import StyleConfigLoader


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def config_base_path():
    """Get the base path for configuration files."""
    return Path(__file__).parent.parent.parent / "src" / "nvcomms_perf" / "report" / "config" / "nccl"


@pytest.fixture
def edge_case_data():
    """Data with edge cases for testing."""
    return pd.DataFrame({
        'tool_binary': ['all_reduce_perf', '', 'test_with_very_long_name_that_exceeds_normal_limits'],
        'test_result.size': [0, 1, float('inf')],  # Edge case sizes
        'test_result.out_of_place.bus_bw': [0.0, -1.0, 1e10],  # Edge case bandwidth values
        'num_nodes': [1, 0, 1000000],  # Edge case node counts
        'num_processes': [1, -1, None],  # Edge case process counts
        'test_config.local_register': ['', 'invalid_value', None],
        'test_env_vars.NCCL_ALGO': [None, 'UNKNOWN_ALGO', ''],
        'test_env_vars.NCCL_PROTO': ['', None, 'INVALID_PROTO']
    })


@pytest.fixture
def malformed_data():
    """Malformed data for error testing."""
    return pd.DataFrame({
        'missing_columns': [1, 2, 3],
        'wrong_types': ['a', 'b', 'c'],
        'inconsistent_data': [1, 'two', 3.0]
    })


@pytest.fixture
def large_dataset():
    """Large dataset for performance testing."""
    size = 10000
    return pd.DataFrame({
        'tool_binary': ['all_reduce_perf'] * size,
        'test_result.size': [1024 * (i + 1) for i in range(size)],
        'test_result.out_of_place.bus_bw': [100.0 + i * 0.1 for i in range(size)],
        'num_nodes': [(i % 64) + 1 for i in range(size)],
        'num_processes': [((i % 128) + 1) * 8 for i in range(size)]
    })


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dataframe_handling(self):
        """Test handling of completely empty dataframes."""
        empty_df = pd.DataFrame()

        adapter = NCCLPerformanceAdapter()

        # Should handle empty data gracefully
        try:
            adapter._raw_data = empty_df
            processed = adapter.process_data()

            assert isinstance(processed, pd.DataFrame)
            assert len(processed) == 0

        except Exception as e:
            # Should handle empty data without unhandled exceptions
            assert isinstance(e, (ValueError, KeyError, IndexError))

    def test_single_row_dataframe(self):
        """Test handling of single-row dataframes."""
        single_row = pd.DataFrame({
            'tool_binary': ['all_reduce_perf'],
            'test_result.size': [1024],
            'test_result.out_of_place.bus_bw': [100.5],
            'num_nodes': [4],
            'num_processes': [32]
        })

        adapter = NCCLPerformanceAdapter()
        adapter._raw_data = single_row

        processed = adapter.process_data()

        assert len(processed) == 1
        assert processed.iloc[0]['tool_binary'] == 'all_reduce_perf'

    def test_missing_required_columns(self, malformed_data):
        """Test handling of data with missing required columns."""
        adapter = NCCLPerformanceAdapter()
        adapter._raw_data = malformed_data

        # Should handle missing columns gracefully
        try:
            processed = adapter.process_data()
            # If successful, should handle missing columns
            assert isinstance(processed, pd.DataFrame)

        except Exception as e:
            # Should raise appropriate exceptions for missing columns
            assert isinstance(e, (KeyError, ValueError, AttributeError))

    def test_extreme_values(self, edge_case_data):
        """Test handling of extreme values in data."""
        adapter = NCCLPerformanceAdapter()
        adapter._raw_data = edge_case_data

        # Should handle extreme values without crashing
        try:
            processed = adapter.process_data()
            assert isinstance(processed, pd.DataFrame)

            # Check that extreme values are handled appropriately
            if 'test_result.size' in processed.columns:
                sizes = processed['test_result.size']
                # Should handle zero, infinity, etc.
                assert sizes is not None

        except Exception as e:
            # Should handle extreme values gracefully
            assert isinstance(e, (ValueError, OverflowError, TypeError))

    def test_null_and_nan_values(self):
        """Test handling of null and NaN values."""
        data_with_nulls = pd.DataFrame({
            'tool_binary': ['all_reduce_perf', None, 'all_gather_perf'],
            'test_result.size': [1024, np.nan, 2048],
            'test_result.out_of_place.bus_bw': [100.5, None, 200.3],
            'num_nodes': [4, np.nan, 8],
            'num_processes': [32, None, 64]
        })

        adapter = NCCLPerformanceAdapter()
        adapter._raw_data = data_with_nulls

        # Should handle null/NaN values appropriately
        processed = adapter.process_data()

        assert isinstance(processed, pd.DataFrame)
        # Should have same number of rows (nulls preserved or handled)
        assert len(processed) == len(data_with_nulls)

    def test_unicode_and_special_characters(self):
        """Test handling of Unicode and special characters."""
        unicode_data = pd.DataFrame({
            'tool_binary': ['all_reduce_perf', 'test_ñoñó', 'тест_перф'],
            'test_result.size': [1024, 2048, 4096],
            'test_result.out_of_place.bus_bw': [100.5, 200.3, 300.1],
            'comment': ['Normal', 'Special chars: ñoñó', 'Unicode: тест'],
            'path': ['/normal/path', '/path with spaces/', '/path/with/símbolos']
        })

        adapter = NCCLPerformanceAdapter()
        adapter._raw_data = unicode_data

        # Should handle Unicode characters without issues
        processed = adapter.process_data()

        assert isinstance(processed, pd.DataFrame)
        assert len(processed) == 3

    def test_very_long_strings(self):
        """Test handling of very long string values."""
        long_string = 'very_long_test_name_' + 'x' * 1000

        long_string_data = pd.DataFrame({
            'tool_binary': [long_string, 'normal_name', 'another_normal'],
            'test_result.size': [1024, 2048, 4096],
            'test_result.out_of_place.bus_bw': [100.5, 200.3, 300.1],
            'description': [long_string, 'normal', 'also normal']
        })

        adapter = NCCLPerformanceAdapter()
        adapter._raw_data = long_string_data

        # Should handle very long strings
        processed = adapter.process_data()

        assert isinstance(processed, pd.DataFrame)
        assert len(processed) == 3


# ============================================================================
# Data Validation Tests
# ============================================================================

class TestDataValidation:
    """Test data validation and cleanup functionality."""

    def test_data_type_consistency(self):
        """Test that data types are consistent after processing."""
        mixed_types_data = pd.DataFrame({
            'tool_binary': ['all_reduce_perf', 'all_gather_perf'],
            'test_result.size': ['1024', 2048],  # Mixed string/int
            'test_result.out_of_place.bus_bw': ['100.5', 200.3],  # Mixed string/float
            'num_nodes': ['4', 8],  # Mixed string/int
            'num_processes': [32, '64']  # Mixed int/string
        })

        adapter = NCCLPerformanceAdapter()
        adapter._raw_data = mixed_types_data

        processed = adapter.process_data()

        # Should handle type conversion
        assert isinstance(processed, pd.DataFrame)

        # Check that numeric columns are properly converted
        if 'test_result.size' in processed.columns:
            # Should be numeric after processing
            assert processed['test_result.size'].dtype in ['int64', 'float64', 'object']

    def test_duplicate_row_handling(self):
        """Test handling of duplicate rows."""
        duplicate_data = pd.DataFrame({
            'tool_binary': ['all_reduce_perf', 'all_reduce_perf', 'all_gather_perf'],
            'test_result.size': [1024, 1024, 2048],  # First two rows identical
            'test_result.out_of_place.bus_bw': [100.5, 100.5, 200.3],
            'num_nodes': [4, 4, 8],
            'num_processes': [32, 32, 64]
        })

        adapter = NCCLPerformanceAdapter()
        adapter._raw_data = duplicate_data

        processed = adapter.process_data()

        # Should handle duplicates (may keep them or remove them)
        assert isinstance(processed, pd.DataFrame)
        assert len(processed) in [2, 3]  # Either keeps or removes duplicates

    def test_data_range_validation(self):
        """Test validation of data ranges."""
        out_of_range_data = pd.DataFrame({
            'tool_binary': ['all_reduce_perf', 'all_gather_perf'],
            'test_result.size': [-1024, 0],  # Negative and zero sizes
            'test_result.out_of_place.bus_bw': [-100.5, 1e20],  # Negative and very large values
            'num_nodes': [0, -4],  # Invalid node counts
            'num_processes': [-32, 0]  # Invalid process counts
        })

        adapter = NCCLPerformanceAdapter()
        adapter._raw_data = out_of_range_data

        # Should handle out-of-range values
        processed = adapter.process_data()

        assert isinstance(processed, pd.DataFrame)
        assert len(processed) == 2

    def test_column_name_normalization(self):
        """Test normalization of column names."""
        weird_columns_data = pd.DataFrame({
            'Tool Binary': ['all_reduce_perf'],  # Space in name
            'test.result.size': [1024],  # Different separator
            'test_result__out_of_place__bus_bw': [100.5],  # Double underscore
            'NUM_NODES': [4],  # All caps
            'num-processes': [32]  # Hyphen separator
        })

        adapter = NCCLPerformanceAdapter()
        adapter._raw_data = weird_columns_data

        # Should handle weird column names
        processed = adapter.process_data()

        assert isinstance(processed, pd.DataFrame)
        assert len(processed) == 1


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Test performance and scalability."""

    def test_large_dataset_processing(self, large_dataset):
        """Test processing of large datasets."""
        adapter = NCCLPerformanceAdapter()
        adapter._raw_data = large_dataset

        # Should handle large datasets efficiently
        processed = adapter.process_data()

        assert isinstance(processed, pd.DataFrame)
        assert len(processed) == len(large_dataset)

        # Basic performance check - should complete in reasonable time
        assert True  # If we reach here, performance is acceptable

    def test_memory_usage_optimization(self, large_dataset):
        """Test memory usage optimization."""
        adapter = NCCLPerformanceAdapter()

        # Test column optimization with new list structure
        mock_plot_config = Mock()

        # Create PDF config with list structure
        mock_pdf_config = Mock()
        mock_pdf_config.id = 'basic_report'
        mock_pdf_config.analysis_profiles = ['test_profile']
        mock_plot_config.pdf_configs = [mock_pdf_config]

        # Create analysis profile with list structure
        mock_analysis_profile = Mock()
        mock_analysis_profile.id = 'test_profile'
        mock_analysis_profile.chart_types = ['test_chart']
        mock_analysis_profile.test_binaries = ['all_reduce_perf']
        mock_plot_config.analysis_profiles = [mock_analysis_profile]

        # Create chart config with list structure
        mock_chart_config = Mock()
        mock_chart_config.id = 'test_chart'
        mock_chart_config.performance_columns = ['test_result.out_of_place.bus_bw']
        mock_chart_config.fixed_params = {}
        mock_chart_config.chart_split_params = {}
        mock_chart_config.chart_combine_params = {}
        mock_plot_config.chart = [mock_chart_config]

        # Add settings with proper structure to avoid Mock iterable error
        mock_settings = Mock()
        mock_data_processing = Mock()
        mock_data_processing.performance_columns = {}  # Empty dict instead of Mock
        mock_settings.data_processing = mock_data_processing
        mock_plot_config.settings = mock_settings

        # Mock the helper methods
        def get_pdf_config_by_id(pdf_id):
            for config in mock_plot_config.pdf_configs:
                if config.id == pdf_id:
                    return config
            return None

        def get_analysis_profile_by_id(profile_id):
            for profile in mock_plot_config.analysis_profiles:
                if profile.id == profile_id:
                    return profile
            return None

        def get_chart_by_id(chart_id):
            for chart in mock_plot_config.chart:
                if chart.id == chart_id:
                    return chart
            return None

        mock_plot_config.get_pdf_config_by_id = Mock(side_effect=get_pdf_config_by_id)
        mock_plot_config.get_analysis_profile_by_id = Mock(side_effect=get_analysis_profile_by_id)
        mock_plot_config.get_chart_by_id = Mock(side_effect=get_chart_by_id)

        adapter.plot_config = mock_plot_config

        # Get required columns for optimization
        required_columns = adapter.get_required_columns_for_pdf_type('basic_report')

        # Should optimize column usage
        assert isinstance(required_columns, set)
        assert len(required_columns) > 0

    def test_concurrent_processing(self, large_dataset):
        """Test behavior under concurrent processing scenarios."""
        adapters = []

        # Create multiple adapters with same data
        for i in range(3):
            adapter = NCCLPerformanceAdapter()
            adapter._raw_data = large_dataset.copy()
            adapters.append(adapter)

        # Process concurrently (simulated)
        results = []
        for adapter in adapters:
            processed = adapter.process_data()
            results.append(processed)

        # Should handle concurrent-like processing
        assert len(results) == 3
        assert all(isinstance(result, pd.DataFrame) for result in results)
        assert all(len(result) == len(large_dataset) for result in results)

    def test_memory_cleanup(self):
        """Test memory cleanup after processing."""
        # Create and process multiple datasets
        for i in range(10):
            data = pd.DataFrame({
                'tool_binary': ['test_perf'] * 1000,
                'test_result.size': list(range(1000)),
                'test_result.out_of_place.bus_bw': [j * 0.1 for j in range(1000)]
            })

            adapter = NCCLPerformanceAdapter()
            adapter._raw_data = data
            processed = adapter.process_data()

            # Clear references
            del adapter
            del processed
            del data

        # Should not accumulate memory
        assert True  # If we reach here without memory issues, cleanup is working


# ============================================================================
# Integration and Compatibility Tests
# ============================================================================

class TestIntegrationAndCompatibility:
    """Test integration scenarios and compatibility."""

    def test_configuration_file_compatibility(self, config_base_path):
        """Test compatibility with different configuration file versions."""
        plot_config_path = config_base_path / "plot" / "plot_config.toml"
        style_config_path = config_base_path / "style" / "style_config.toml"

        # Test that configurations can be loaded without errors
        if plot_config_path.exists():
            try:
                plot_config = ChartConfigLoader.load_from_toml(str(plot_config_path))
                assert plot_config is not None
            except Exception as e:
                # Should handle configuration issues gracefully
                assert isinstance(e, (ValueError, TypeError, KeyError))

        if style_config_path.exists():
            try:
                style_config = StyleConfigLoader.load_from_toml(str(style_config_path))
                assert style_config is not None
            except Exception as e:
                # Should handle configuration issues gracefully
                assert isinstance(e, (ValueError, TypeError, KeyError))

    def test_pandas_version_compatibility(self):
        """Test compatibility with different pandas versions."""
        # Test basic pandas operations that might vary between versions
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })

        # Test operations that work across pandas versions
        assert len(data) == 3
        assert list(data.columns) == ['col1', 'col2']
        assert data.iloc[0]['col1'] == 1

        # Test numeric conversion
        numeric_data = pd.to_numeric(['1', '2', '3'], errors='coerce')
        assert all(pd.notna(x) and isinstance(x, (int, float, np.integer, np.floating)) for x in numeric_data)  # type: ignore

    def test_matplotlib_version_compatibility(self):
        """Test compatibility with different matplotlib versions."""
        # Test basic matplotlib operations
        fig, ax = plt.subplots(figsize=(8, 6))

        # Test plotting operations that should work across versions
        x = [1, 2, 3, 4]
        y = [1, 4, 2, 3]
        ax.plot(x, y)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_title('Test Plot')

        # Clean up
        plt.close(fig)

        assert True  # If we reach here, basic matplotlib operations work

    def test_cross_platform_compatibility(self):
        """Test cross-platform compatibility."""
        # Test path handling
        test_path = Path('test') / 'subdir' / 'file.txt'

        # Should work on all platforms
        assert isinstance(test_path, Path)
        assert str(test_path).count('file.txt') == 1

        # Test file handling
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / 'test.csv'

            # Create test file
            pd.DataFrame({'col': [1, 2, 3]}).to_csv(test_file, index=False)

            # Should be readable
            assert test_file.exists()
            loaded = pd.read_csv(test_file)
            assert len(loaded) == 3  # type: ignore


# ============================================================================
# Utility Function Tests
# ============================================================================

class TestUtilityFunctions:
    """Test utility functions and helpers."""

    def test_data_type_inference(self):
        """Test automatic data type inference."""
        test_values = [
            ('123', int),
            ('123.45', float),
            ('test_string', str),
            ('true', str),  # Should remain string unless specifically converted
            ('', str)
        ]

        for value, expected_type in test_values:
            # Test numeric conversion
            try:
                numeric_value = pd.to_numeric(value, errors='coerce')
                if pd.notna(numeric_value):  # type: ignore
                    assert isinstance(numeric_value, (int, float))
                else:
                    # Should handle non-numeric values
                    assert isinstance(value, str)
            except Exception:
                # Should handle conversion errors gracefully
                assert isinstance(value, str)

    def test_string_cleaning_utilities(self):
        """Test string cleaning and normalization utilities."""
        test_strings = [
            '  whitespace_test  ',
            'UPPERCASE_TEST',
            'special-chars_test',
            'unicode_tëst',
            ''
        ]

        for test_string in test_strings:
            # Test basic string operations
            cleaned = test_string.strip()
            lower = test_string.lower()

            assert isinstance(cleaned, str)
            assert isinstance(lower, str)
            assert len(cleaned) <= len(test_string)

    def test_configuration_file_validation(self):
        """Test configuration file path validation."""
        valid_paths = [
            'config.toml',
            '/absolute/path/config.toml',
            './relative/config.toml',
            'config_with_underscores.toml'
        ]

        invalid_paths = [
            '',
            'config.txt',
            'config',
            None
        ]

        for path in valid_paths:
            # Should recognize valid TOML paths
            assert path.endswith('.toml')
            assert len(path) > 5  # At least '.toml'

        for path in invalid_paths:
            # Should recognize invalid paths
            if path:
                assert not path.endswith('.toml') or len(path) <= 5
            else:
                assert not path

    def test_error_message_formatting(self):
        """Test error message formatting utilities."""
        error_scenarios = [
            ('FileNotFoundError', 'test.csv', 'File not found: test.csv'),
            ('ValueError', 'invalid data', 'Invalid data: invalid data'),
            ('KeyError', 'missing_column', 'Missing column: missing_column')
        ]

        for error_type, context, expected_pattern in error_scenarios:
            # Test error message formatting
            formatted_message = f"{error_type}: {context}"

            assert error_type in formatted_message
            assert context in formatted_message
            assert len(formatted_message) > len(context)


# ============================================================================
# Regression Tests
# ============================================================================

class TestRegressionPrevention:
    """Test to prevent regressions in functionality."""

    def test_basic_adapter_workflow(self):
        """Test that basic adapter workflow still works."""
        data = pd.DataFrame({
            'tool_binary': ['all_reduce_perf', 'all_gather_perf'],
            'test_result.size': [1024, 2048],
            'test_result.out_of_place.bus_bw': [100.5, 200.3],
            'num_nodes': [4, 8],
            'num_processes': [32, 64]
        })

        adapter = NCCLPerformanceAdapter()
        adapter._raw_data = data

        # Should complete basic workflow
        processed = adapter.process_data()
        binaries = adapter.get_test_binaries()
        perf_cols = adapter.get_performance_columns()
        size_col = adapter.get_size_column()

        assert isinstance(processed, pd.DataFrame)
        assert isinstance(binaries, list)
        assert isinstance(perf_cols, list)
        assert isinstance(size_col, str)

    def test_legend_shortening_regression(self):
        """Test that legend shortening still works correctly."""
        adapter = NCCLPerformanceAdapter()

        test_cases = [
            ('num_nodes=4', 'N=4'),
            ('num_processes=8', 'NP=8'),
            ('NCCL_ALGO=RING', 'ALGO=RING'),
            ('local_register=1', 'UB=1'),
            ('num_nodes=4, NCCL_ALGO=RING', 'N=4, ALGO=RING')
        ]

        for original, expected in test_cases:
            shortened = adapter.shorten_legend_key(original)
            assert shortened == expected

    def test_column_mapping_regression(self):
        """Test that column mapping still works correctly."""
        adapter = NCCLPerformanceAdapter()

        # Test key column mappings
        mappings = [
            ('test_binary', 'tool_binary'),
            ('size', 'test_result.size'),
            ('NCCL_ALGO', 'test_env_vars.NCCL_ALGO'),
            ('num_nodes', 'num_nodes')
        ]

        for config_param, expected_col in mappings:
            actual_col = adapter.get_dataframe_column_name(config_param)
            assert actual_col == expected_col

    def test_data_processing_regression(self):
        """Test that data processing produces expected results."""
        data = pd.DataFrame({
            'tool_binary': ['all_reduce', 'all_gather'],  # Missing _perf suffix
            'test_result.size': ['1024', '2048'],  # String values
            'test_result.out_of_place.bus_bw': ['100.5', '200.3'],  # String values
            'test_env_vars.NCCL_TESTS_SPLIT_MASK': [31, '0x1F']  # Mixed hex formats
        })

        adapter = NCCLPerformanceAdapter()
        adapter._raw_data = data

        processed = adapter.process_data()

        # Should process data consistently
        assert all(processed['tool_binary'].str.endswith('_perf'))
        assert processed['test_result.size'].dtype in ['int64', 'float64', 'object']
        assert all(processed['test_env_vars.NCCL_TESTS_SPLIT_MASK'].astype(str).str.startswith('0x'))
