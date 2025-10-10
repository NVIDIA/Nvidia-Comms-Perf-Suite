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

"""Test multiple CSV input handling for report generation."""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from nvcomms_perf.report.cli.report import generate_report, ReportType, PDFType, OutputFile
from nvcomms_perf.report.cli.common import InputFile, InputFileType


class TestMultipleCSVInputs:
    """Test cases for handling multiple CSV inputs in report generation."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data for testing."""
        # First CSV data - node count 1-2
        data1 = {
            'tool_binary': ['all_reduce_perf'] * 4,
            'test_result.size': [1024, 2048, 1024, 2048],
            'test_result.out_of_place.bus_bw': [10.5, 20.3, 11.2, 21.0],
            'test_result.in_place.bus_bw': [9.8, 19.7, 10.5, 20.1],
            'num_nodes': [1, 1, 2, 2],
            'num_processes': [8, 8, 16, 16],
            'system_info.system_name': ['System-A'] * 4,
            'library_version': ['2.18.5'] * 4
        }

        # Second CSV data - node count 4
        data2 = {
            'tool_binary': ['all_reduce_perf'] * 4,
            'test_result.size': [1024, 2048, 4096, 8192],
            'test_result.out_of_place.bus_bw': [15.2, 25.6, 35.8, 45.2],
            'test_result.in_place.bus_bw': [14.8, 24.9, 34.5, 44.1],
            'num_nodes': [4] * 4,
            'num_processes': [32] * 4,
            'system_info.system_name': ['System-B'] * 4,
            'library_version': ['2.18.5'] * 4
        }

        # Third CSV data - different test binary
        data3 = {
            'tool_binary': ['all_gather_perf'] * 3,
            'test_result.size': [512, 1024, 2048],
            'test_result.out_of_place.bus_bw': [8.5, 12.3, 18.9],
            'test_result.in_place.bus_bw': [8.1, 11.8, 18.2],
            'num_nodes': [1, 1, 1],
            'num_processes': [8, 8, 8],
            'system_info.system_name': ['System-A'] * 3,
            'library_version': ['2.18.5'] * 3
        }

        return data1, data2, data3

    @pytest.fixture
    def csv_files(self, temp_dir, sample_csv_data):
        """Create CSV files with sample data."""
        data1, data2, data3 = sample_csv_data

        # Create CSV files
        csv1 = temp_dir / "data1.csv"
        csv2 = temp_dir / "data2.csv"
        csv3 = temp_dir / "data3.csv"

        pd.DataFrame(data1).to_csv(csv1, index=False)
        pd.DataFrame(data2).to_csv(csv2, index=False)
        pd.DataFrame(data3).to_csv(csv3, index=False)

        return csv1, csv2, csv3

    def test_multiple_csv_loading(self, csv_files, temp_dir):
        """Test that multiple CSV files are correctly loaded and combined."""
        csv1, csv2, csv3 = csv_files
        output_file = temp_dir / "test_report.pdf"

        # Create InputFile objects
        input_files = [
            InputFile(str(csv1)),
            InputFile(str(csv2)),
            InputFile(str(csv3))
        ]

        # Mock the PDF generation to focus on data loading
        with patch('nvcomms_perf.report.generators.pdf.generate_pdf_report') as mock_pdf_gen:
            # Mock the configuration loading
            with patch('nvcomms_perf.report.cli.report._get_default_config_path') as mock_config_path:
                # Create minimal mock config files
                mock_plot_config = temp_dir / "mock_plot.toml"
                mock_style_config = temp_dir / "mock_style.toml"

                # Write minimal valid TOML content
                mock_plot_config.write_text('[settings]\n[chart]\n[analysis_profiles]\n[pdf_configs]')
                mock_style_config.write_text('[style_mappings]\n[color_scheme]\n[color_palette]')

                def config_path_side_effect(config_type):
                    if config_type == 'plot':
                        return mock_plot_config
                    else:
                        return mock_style_config

                mock_config_path.side_effect = config_path_side_effect

                # Mock configuration objects to avoid complex setup
                with patch('nvcomms_perf.report.models.ChartConfigLoader.load_from_toml') as mock_plot_loader:
                    with patch('nvcomms_perf.report.models.StyleConfigLoader.load_from_toml') as mock_style_loader:
                        # Create minimal mock config objects
                        mock_plot_config_obj = MagicMock()
                        mock_plot_config_obj.pdf_configs = {}
                        mock_plot_config_obj.analysis_profiles = {}
                        mock_plot_config_obj.chart = {}
                        mock_plot_config_obj.settings = None

                        mock_style_config_obj = MagicMock()

                        mock_plot_loader.return_value = mock_plot_config_obj
                        mock_style_loader.return_value = mock_style_config_obj

                        # Call the generate_report function
                        generate_report(
                            inputs=input_files,
                            output=OutputFile(f"pdf:{output_file}"),
                            pdf_type=PDFType.BASIC_REPORT,
                            plot_config=None,
                            style_config=None,
                            show_progress=False,
                            include_empty_pages=True,
                            include_data_tables=True,
                            include_test_commands=True
                        )

                        # Verify that generate_pdf_report was called
                        assert mock_pdf_gen.called

                        # Get the combined DataFrame that was passed to generate_pdf_report
                        call_args = mock_pdf_gen.call_args
                        combined_df = call_args[0][0]  # First positional argument

                        # Verify the combined DataFrame has data from all files
                        assert len(combined_df) == 11  # 4 + 4 + 3 rows

                        # Check that all test binaries are present
                        assert set(combined_df['tool_binary']) == {'all_reduce_perf', 'all_gather_perf'}

                        # Check that all node counts are present
                        assert set(combined_df['num_nodes']) == {1, 2, 4}

                        # Check that all systems are present
                        assert set(combined_df['system_info.system_name']) == {'System-A', 'System-B'}

                        # Verify specific data points to ensure correct merging
                        # Data from csv1
                        csv1_data = combined_df[
                            (combined_df['num_nodes'] == 1) &
                            (combined_df['num_processes'] == 8) &
                            (combined_df['tool_binary'] == 'all_reduce_perf') &
                            (combined_df['test_result.size'] == 1024)
                        ]
                        assert len(csv1_data) == 1
                        assert csv1_data.iloc[0]['test_result.out_of_place.bus_bw'] == 10.5

                        # Data from csv2
                        csv2_data = combined_df[
                            (combined_df['num_nodes'] == 4) &
                            (combined_df['test_result.size'] == 8192)
                        ]
                        assert len(csv2_data) == 1
                        assert csv2_data.iloc[0]['test_result.out_of_place.bus_bw'] == 45.2

                        # Data from csv3
                        csv3_data = combined_df[
                            (combined_df['tool_binary'] == 'all_gather_perf') &
                            (combined_df['test_result.size'] == 2048)
                        ]
                        assert len(csv3_data) == 1
                        assert csv3_data.iloc[0]['test_result.out_of_place.bus_bw'] == 18.9

    def test_duplicate_csv_handling(self, csv_files, temp_dir):
        """Test handling of duplicate CSV files (same content)."""
        csv1, csv2, csv3 = csv_files
        output_file = temp_dir / "test_report.pdf"

        # Create a duplicate of csv1
        csv1_dup = temp_dir / "data1_duplicate.csv"
        shutil.copy(csv1, csv1_dup)

        # Create InputFile objects with duplicate
        input_files = [
            InputFile(str(csv1)),
            InputFile(str(csv1_dup)),  # Duplicate
            InputFile(str(csv2))
        ]

        # The duplicate check should detect this, but report generation should still work
        with patch('nvcomms_perf.report.generators.pdf.generate_pdf_report') as mock_pdf_gen:
            with patch('nvcomms_perf.report.cli.report._get_default_config_path') as mock_config_path:
                # Set up mocks similar to previous test
                mock_plot_config = temp_dir / "mock_plot.toml"
                mock_style_config = temp_dir / "mock_style.toml"
                mock_plot_config.write_text('[settings]\n[chart]\n[analysis_profiles]\n[pdf_configs]')
                mock_style_config.write_text('[style_mappings]\n[color_scheme]\n[color_palette]')

                def config_path_side_effect(config_type):
                    if config_type == 'plot':
                        return mock_plot_config
                    else:
                        return mock_style_config

                mock_config_path.side_effect = config_path_side_effect

                with patch('nvcomms_perf.report.models.ChartConfigLoader.load_from_toml'):
                    with patch('nvcomms_perf.report.models.StyleConfigLoader.load_from_toml'):
                        # This should work - duplicates are included in the combined data
                        generate_report(
                            inputs=input_files,
                            output=OutputFile(f"pdf:{output_file}"),
                            pdf_type=PDFType.BASIC_REPORT,
                            plot_config=None,
                            style_config=None,
                            show_progress=False,
                            include_empty_pages=True,
                            include_data_tables=True,
                            include_test_commands=True
                        )

                        # Should still generate report successfully
                        assert mock_pdf_gen.called

    def test_mixed_csv_sizes(self, temp_dir):
        """Test handling of CSV files with different column sets."""
        # CSV with basic columns
        data_basic = {
            'tool_binary': ['all_reduce_perf'],
            'test_result.size': [1024],
            'test_result.out_of_place.bus_bw': [10.5],
            'num_nodes': [1],
            'num_processes': [8]
        }

        # CSV with additional columns
        data_extended = {
            'tool_binary': ['all_reduce_perf'],
            'test_result.size': [2048],
            'test_result.out_of_place.bus_bw': [20.5],
            'test_result.out_of_place.alg_bw': [19.8],  # Extra column
            'num_nodes': [2],
            'num_processes': [16],
            'test_env_vars.NCCL_ALGO': ['Ring'],  # Extra column
            'library_version': ['2.18.5']  # Extra column
        }

        csv_basic = temp_dir / "basic.csv"
        csv_extended = temp_dir / "extended.csv"

        pd.DataFrame(data_basic).to_csv(csv_basic, index=False)
        pd.DataFrame(data_extended).to_csv(csv_extended, index=False)

        input_files = [
            InputFile(str(csv_basic)),
            InputFile(str(csv_extended))
        ]

        output_file = temp_dir / "test_report.pdf"

        with patch('nvcomms_perf.report.generators.pdf.generate_pdf_report') as mock_pdf_gen:
            with patch('nvcomms_perf.report.cli.report._get_default_config_path') as mock_config_path:
                # Set up mocks
                mock_plot_config = temp_dir / "mock_plot.toml"
                mock_style_config = temp_dir / "mock_style.toml"
                mock_plot_config.write_text('[settings]\n[chart]\n[analysis_profiles]\n[pdf_configs]')
                mock_style_config.write_text('[style_mappings]\n[color_scheme]\n[color_palette]')

                def config_path_side_effect(config_type):
                    if config_type == 'plot':
                        return mock_plot_config
                    else:
                        return mock_style_config

                mock_config_path.side_effect = config_path_side_effect

                with patch('nvcomms_perf.report.models.ChartConfigLoader.load_from_toml'):
                    with patch('nvcomms_perf.report.models.StyleConfigLoader.load_from_toml'):
                        # Patch the adapter's method to include bus_bw columns in required columns
                        with patch('nvcomms_perf.report.adapters.nccl_adapter.NCCLPerformanceAdapter.get_required_columns_for_pdf_type') as mock_required_cols:
                            # Return a set that includes bus_bw columns
                            mock_required_cols.return_value = {
                                'tool_binary', 'test_result.size', 'num_nodes', 'num_processes',
                                'test_result.out_of_place.bus_bw', 'test_result.in_place.bus_bw',
                                'library_version', 'system_info.system_name', 'test_cmd'
                            }

                            generate_report(
                                inputs=input_files,
                                output=OutputFile(f"pdf:{output_file}"),
                                pdf_type=PDFType.BASIC_REPORT,
                                plot_config=None,
                                style_config=None,
                                show_progress=False,
                                include_empty_pages=True,
                                include_data_tables=True,
                                include_test_commands=True
                            )

                            # Get the combined DataFrame
                            call_args = mock_pdf_gen.call_args
                            combined_df = call_args[0][0]

                            # Should have 2 rows
                            assert len(combined_df) == 2

                            # Check that basic columns are present
                            assert 'tool_binary' in combined_df.columns
                            assert 'test_result.size' in combined_df.columns
                            assert 'num_nodes' in combined_df.columns
                            assert 'num_processes' in combined_df.columns

                            # Check if bus_bw column is present (it's in both CSVs)
                            bus_bw_columns = [col for col in combined_df.columns if 'bus_bw' in col]
                            assert len(bus_bw_columns) > 0, f"No bus_bw columns found. Available columns: {list(combined_df.columns)}"

                            # Verify the data values
                            row1 = combined_df.iloc[0]
                            row2 = combined_df.iloc[1]

                            assert row1['tool_binary'] == 'all_reduce_perf'
                            assert row1['test_result.size'] == 1024
                            assert row1['num_nodes'] == 1
                            assert row1['num_processes'] == 8

                            # Check bus_bw values
                            assert row1['test_result.out_of_place.bus_bw'] == 10.5

                            assert row2['tool_binary'] == 'all_reduce_perf'
                            assert row2['test_result.size'] == 2048
                            assert row2['num_nodes'] == 2
                            assert row2['num_processes'] == 16
                            assert row2['test_result.out_of_place.bus_bw'] == 20.5

                            # Note: When using optimized column loading, only columns required
                            # by the PDF type are loaded. Extra columns like 'test_result.out_of_place.alg_bw'
                            # and 'test_env_vars.NCCL_ALGO' won't be present unless they're required
                            # by the configuration. This is the expected behavior for performance optimization.
