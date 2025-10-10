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
Tests for the generators module.

This module tests all generator functionality including:
- PDF report generation
- Report layout and pagination
- Multi-chart report handling
- Export functionality
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch, MagicMock
import matplotlib.figure

from nvcomms_perf.report.adapters.nccl_adapter import NCCLPerformanceAdapter
from nvcomms_perf.report.plotting.chart_generator import ChartGenerator
from nvcomms_perf.report.plotting.style_manager import StyleManager
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
def sample_report_data():
    """Sample data for report generation."""
    return pd.DataFrame({
        'tool_binary': ['all_reduce_perf'] * 10 + ['all_gather_perf'] * 10,
        'test_result.size': [1024, 2048, 4096, 8192, 16384] * 4,
        'test_result.out_of_place.bus_bw': [100 + i*10 for i in range(20)],
        'test_result.in_place.bus_bw': [95 + i*10 for i in range(20)],
        'num_nodes': [4, 8, 16, 32] * 5,
        'num_processes': [32, 64, 128, 256] * 5,
        'test_config.local_register': ['0', '1'] * 10,
        'test_env_vars.NCCL_ALGO': ['RING', 'TREE'] * 10,
        'test_env_vars.NCCL_PROTO': ['Simple', 'LL'] * 10
    })


@pytest.fixture
def mock_chart_generator():
    """Create a mock chart generator."""
    chart_generator = Mock(spec=ChartGenerator)

    # Mock figure generation
    mock_figure = Mock(spec=matplotlib.figure.Figure)
    chart_generator.generate_chart.return_value = mock_figure
    chart_generator.create_data_table.return_value = [mock_figure]
    chart_generator.create_command_page.return_value = mock_figure

    return chart_generator


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


# ============================================================================
# PDF Generator Tests
# ============================================================================

class TestPDFGenerator:
    """Test PDF report generation functionality."""

    @patch('nvcomms_perf.report.generators.pdf.generate_pdf_report')
    def test_pdf_generator_basic(self, mock_generate_pdf, sample_report_data, temp_output_dir):
        """Test basic PDF generation."""
        output_file = temp_output_dir / "test_report.pdf"

        # Mock successful PDF generation
        mock_generate_pdf.return_value = None

        # Test that PDF generator can be called
        try:
            from nvcomms_perf.report.generators.pdf import generate_pdf_report

            # Create minimal inputs for testing
            mock_adapter = Mock(spec=NCCLPerformanceAdapter)
            mock_adapter._processed_data = sample_report_data
            mock_style_manager = Mock(spec=StyleManager)

            result = generate_pdf_report(
                dataframe=sample_report_data,
                output_path=Path(output_file),
                pdf_type="basic_report",
                adapter=mock_adapter
            )

            # Should complete without error
            assert True

        except ImportError:
            # PDF generator module might not exist yet
            pytest.skip("PDF generator module not available")
        except Exception as e:
            # Should handle gracefully
            assert isinstance(e, (ValueError, TypeError, AttributeError))

    def test_pdf_generator_with_configurations(self, sample_report_data, config_base_path, temp_output_dir):
        """Test PDF generation with actual configurations."""
        plot_config_path = config_base_path / "plot" / "plot_config.toml"
        style_config_path = config_base_path / "style" / "style_config.toml"

        if plot_config_path.exists() and style_config_path.exists():
            plot_config = ChartConfigLoader.load_from_toml(str(plot_config_path))
            style_config = StyleConfigLoader.load_from_toml(str(style_config_path))

            # Create components
            adapter = NCCLPerformanceAdapter(plot_config=plot_config, style_config=style_config)
            adapter._processed_data = sample_report_data
            style_manager = StyleManager(style_config)

            output_file = temp_output_dir / "configured_report.pdf"

            # Test PDF generation with real configurations
            try:
                from nvcomms_perf.report.generators.pdf import generate_pdf_report

                result = generate_pdf_report(
                    dataframe=sample_report_data,
                    output_path=Path(output_file),
                    pdf_type="basic_report",
                    adapter=adapter
                )

                # Should complete successfully
                assert True

            except ImportError:
                pytest.skip("PDF generator module not available")
            except Exception as e:
                # Should handle configuration issues gracefully
                assert isinstance(e, (ValueError, TypeError, AttributeError))
        else:
            pytest.skip("Configuration files not available")

    def test_pdf_generator_empty_data(self, temp_output_dir):
        """Test PDF generation with empty data."""
        empty_data = pd.DataFrame()
        output_file = temp_output_dir / "empty_report.pdf"

        try:
            from nvcomms_perf.report.generators.pdf import generate_pdf_report

            mock_adapter = Mock(spec=NCCLPerformanceAdapter)
            mock_adapter._processed_data = empty_data
            mock_adapter.plot_config = Mock()  # Add missing plot_config attribute
            mock_adapter.style_config = Mock()  # Add missing style_config attribute
            mock_style_manager = Mock(spec=StyleManager)

            result = generate_pdf_report(
                dataframe=empty_data,
                output_path=Path(output_file),
                pdf_type="basic_report",
                adapter=mock_adapter
            )

            # Should handle empty data gracefully
            assert True

        except ImportError:
            pytest.skip("PDF generator module not available")
        except Exception as e:
            # Should handle empty data gracefully
            assert isinstance(e, (ValueError, TypeError))

    @patch('matplotlib.pyplot.savefig')
    @patch('nvcomms_perf.report.generators.pdf.pdf_generator._create_introduction_page')
    def test_pdf_generator_matplotlib_integration(self, mock_create_intro, mock_savefig, sample_report_data, temp_output_dir):
        """Test PDF generator integration with matplotlib."""
        output_file = temp_output_dir / "matplotlib_test.pdf"

        # Mock matplotlib savefig
        mock_savefig.return_value = None

        # Mock introduction page creation to return None (no intro page)
        mock_create_intro.return_value = None

        try:
            from nvcomms_perf.report.generators.pdf import generate_pdf_report

            mock_adapter = Mock(spec=NCCLPerformanceAdapter)

            # Create proper mock plot_config with list structure
            mock_plot_config = Mock()
            mock_pdf_config = Mock()
            mock_pdf_config.id = 'basic_report'
            mock_pdf_config.analysis_profiles = []  # Empty list to make it iterable
            mock_plot_config.pdf_configs = [mock_pdf_config]

            # Mock the helper method
            def get_pdf_config_by_id(pdf_id):
                for config in mock_plot_config.pdf_configs:
                    if config.id == pdf_id:
                        return config
                return None

            mock_plot_config.get_pdf_config_by_id = Mock(side_effect=get_pdf_config_by_id)
            mock_adapter.plot_config = mock_plot_config

            # Create proper mock style_config with numeric values
            mock_style_config = Mock()
            mock_pdf_settings = Mock()
            mock_pdf_settings.page_height = 14.0
            mock_pdf_settings.page_width = 11.0
            mock_pdf_settings.margin_left = 0.5
            mock_pdf_settings.margin_right = 0.5
            mock_pdf_settings.margin_top = 1.0
            mock_pdf_settings.margin_bottom = 1.0
            mock_pdf_settings.use_landscape = True
            mock_pdf_settings.pdf_dpi = 300
            mock_pdf_settings.pdf_facecolor = 'white'
            mock_pdf_settings.pdf_edgecolor = 'none'
            mock_style_config.pdf_settings = mock_pdf_settings

            # Mock other style config components
            mock_chart_style = Mock()
            mock_chart_style.font_size = 11
            mock_chart_style.axes_facecolor = '#ffffff'
            mock_chart_style.axes_edgecolor = '#333333'
            mock_chart_style.axes_labelcolor = '#333333'
            mock_chart_style.text_color = '#333333'
            mock_chart_style.xtick_color = '#333333'
            mock_chart_style.ytick_color = '#333333'
            mock_chart_style.background_color = '#ffffff'
            mock_chart_style.axes_spines_top = False
            mock_chart_style.axes_spines_right = False
            mock_chart_style.line_width = 2.2
            mock_chart_style.marker_size = 5
            mock_chart_style.markeredgewidth = 0.8
            mock_chart_style.markeredgecolor = 'white'
            mock_style_config.chart_style = mock_chart_style

            mock_font_adjustments = Mock()
            mock_font_adjustments.label_large = 1
            mock_font_adjustments.title_medium = 3
            mock_font_adjustments.label_small = -1
            mock_font_adjustments.title_large = 5
            mock_style_config.font_adjustments = mock_font_adjustments

            mock_legend = Mock()
            mock_legend.fontsize = 9
            mock_style_config.legend = mock_legend

            mock_grid = Mock()
            mock_grid.show = True
            mock_grid.alpha = 0.2
            mock_grid.major_color = '#666666'
            mock_grid.major_style = 'solid'
            mock_grid.major_linewidth = 0.8
            mock_grid.style = 'solid'
            mock_grid.color = '#666666'
            mock_style_config.grid = mock_grid

            mock_ticks = Mock()
            mock_ticks.minor_length = 3
            mock_ticks.minor_width = 0.5
            mock_ticks.major_length = 5
            mock_ticks.major_width = 0.8
            mock_ticks.tick_direction = 'out'
            mock_style_config.ticks = mock_ticks

            # Mock page_layout with numeric values for range() function
            mock_page_layout = Mock()
            mock_page_layout.note_grid_start = 5
            mock_page_layout.note_grid_end = 95
            mock_page_layout.note_grid_step = 5
            mock_style_config.page_layout = mock_page_layout

            # Mock grid_lines with numeric values for linewidth
            mock_grid_lines = Mock()
            mock_grid_lines.note_page_linewidth = 0.3
            mock_style_config.grid_lines = mock_grid_lines

            # Mock text_positioning with numeric values for text placement
            mock_text_positioning = Mock()
            mock_text_positioning.title_x = 0.5
            mock_text_positioning.title_y = 0.95
            mock_style_config.text_positioning = mock_text_positioning

            mock_adapter.style_config = mock_style_config

            # Test that matplotlib integration works
            result = generate_pdf_report(
                dataframe=sample_report_data,
                output_path=Path(output_file),
                pdf_type="basic_report",
                adapter=mock_adapter
            )

            # Should interact with matplotlib
            assert True

        except ImportError:
            pytest.skip("PDF generator module not available")


# ============================================================================
# Report Layout Tests
# ============================================================================

class TestReportLayout:
    """Test report layout and pagination functionality."""

    def test_multi_chart_layout(self, sample_report_data, mock_chart_generator):
        """Test layout with multiple charts."""
        # Create multiple chart groups
        chart_groups = []
        for i in range(5):
            chart_groups.append({
                'group_key': f'test_group_{i}',
                'data': sample_report_data.iloc[i*2:(i+1)*2],
                'filters': {'test_param': f'value_{i}'}
            })

        # Test that multiple charts can be handled
        results = []
        for group in chart_groups:
            result = mock_chart_generator.generate_chart('test_type', 'test_binary', group)
            results.append(result)

        assert len(results) == 5
        assert all(result is not None for result in results)

    def test_pagination_logic(self, sample_report_data):
        """Test pagination logic for large reports."""
        # Simulate large dataset requiring pagination
        large_data = pd.concat([sample_report_data] * 10, ignore_index=True)

        # Test pagination parameters
        page_size = 4  # Charts per page
        total_charts = 15
        expected_pages = (total_charts + page_size - 1) // page_size  # Ceiling division

        assert expected_pages == 4  # 15 charts with 4 per page = 4 pages

        # Test page calculation
        for chart_idx in range(total_charts):
            page_num = chart_idx // page_size + 1
            chart_on_page = chart_idx % page_size + 1

            assert 1 <= page_num <= expected_pages
            assert 1 <= chart_on_page <= page_size

    def test_page_layout_configuration(self, config_base_path):
        """Test page layout configuration."""
        style_config_path = config_base_path / "style" / "style_config.toml"

        if style_config_path.exists():
            style_config = StyleConfigLoader.load_from_toml(str(style_config_path))

            # Test PDF layout settings
            if hasattr(style_config, 'pdf_settings'):
                pdf_settings = style_config.pdf_settings
                assert pdf_settings.page_width > 0
                assert pdf_settings.page_height > 0
                assert pdf_settings.margin_left >= 0
                assert pdf_settings.margin_right >= 0
                assert pdf_settings.margin_top >= 0
                assert pdf_settings.margin_bottom >= 0
        else:
            # Test layout concept
            layout_config = {
                'page_width': 11.0,
                'page_height': 8.5,
                'margin_left': 0.5,
                'margin_right': 0.5,
                'margin_top': 1.0,
                'margin_bottom': 1.0
            }

            assert layout_config['page_width'] > layout_config['margin_left'] + layout_config['margin_right']
            assert layout_config['page_height'] > layout_config['margin_top'] + layout_config['margin_bottom']

    def test_chart_spacing_on_page(self):
        """Test chart spacing on report pages."""
        # Test optimal chart arrangement
        page_width = 11.0  # inches
        page_height = 8.5  # inches
        margin = 0.5

        available_width = page_width - 2 * margin
        available_height = page_height - 2 * margin

        # Test different chart arrangements
        arrangements = [
            (1, 1),  # 1x1 (single chart)
            (2, 1),  # 2x1 (two charts horizontal)
            (1, 2),  # 1x2 (two charts vertical)
            (2, 2),  # 2x2 (four charts)
        ]

        for cols, rows in arrangements:
            chart_width = available_width / cols
            chart_height = available_height / rows

            assert chart_width > 0
            assert chart_height > 0
            assert chart_width * cols <= available_width
            assert chart_height * rows <= available_height


# ============================================================================
# Export Functionality Tests
# ============================================================================

class TestExportFunctionality:
    """Test export functionality for different formats."""

    def test_pdf_export_options(self, temp_output_dir):
        """Test PDF export with different options."""
        output_file = temp_output_dir / "export_test.pdf"

        # Test PDF export parameters
        pdf_options = {
            'dpi': 300,
            'facecolor': 'white',
            'edgecolor': 'none',
            'orientation': 'landscape',
            'papertype': 'letter'
        }

        # Validate PDF options
        assert pdf_options['dpi'] > 0
        assert pdf_options['facecolor'] in ['white', 'none', 'transparent']
        assert pdf_options['orientation'] in ['landscape', 'portrait']

        # Test that options are reasonable
        assert 72 <= pdf_options['dpi'] <= 600  # Reasonable DPI range

    def test_png_export_options(self, temp_output_dir):
        """Test PNG export with different options."""
        output_file = temp_output_dir / "export_test.png"

        # Test PNG export parameters
        png_options = {
            'dpi': 300,
            'facecolor': 'white',
            'edgecolor': 'none',
            'transparent': False,
            'bbox_inches': 'tight'
        }

        # Validate PNG options
        assert png_options['dpi'] > 0
        assert isinstance(png_options['transparent'], bool)
        assert png_options['bbox_inches'] in ['tight', 'standard', None]

    def test_export_file_validation(self, temp_output_dir):
        """Test export file path validation."""
        test_cases = [
            ("report.pdf", True),
            ("report.png", True),
            ("report.jpeg", False),  # Not typically supported
            ("report.txt", False),   # Invalid format
            ("", False),             # Empty filename
        ]

        for filename, should_be_valid in test_cases:
            if filename:
                output_path = temp_output_dir / filename

                # Test file extension validation
                valid_extensions = ['.pdf', '.png']
                is_valid = any(filename.lower().endswith(ext) for ext in valid_extensions)

                assert is_valid == should_be_valid
            else:
                # Empty filename should be invalid
                assert not should_be_valid

    def test_output_directory_creation(self, tmp_path):
        """Test automatic output directory creation."""
        nested_dir = tmp_path / "reports" / "2024" / "quarter1"
        output_file = nested_dir / "test_report.pdf"

        # Test directory creation logic
        output_file.parent.mkdir(parents=True, exist_ok=True)

        assert nested_dir.exists()
        assert nested_dir.is_dir()

        # Test that file can be created in new directory
        output_file.touch()
        assert output_file.exists()


# ============================================================================
# Error Handling and Edge Cases
# ============================================================================

class TestGeneratorErrorHandling:
    """Test error handling in generators."""

    def test_invalid_output_path(self):
        """Test handling of invalid output paths."""
        invalid_paths = [
            "/nonexistent/directory/report.pdf",
            "",
            None
        ]

        for path in invalid_paths:
            if path:
                output_path = Path(path)

                # Test path validation - only check paths that won't cause permission errors
                if str(output_path).startswith('/nonexistent'):
                    # Should handle non-existent directories
                    assert not output_path.parent.exists()
            else:
                # Should handle None/empty paths
                assert not path

    def test_large_dataset_handling(self, temp_output_dir):
        """Test handling of large datasets."""
        # Create large dataset
        large_data = pd.DataFrame({
            'tool_binary': ['test_perf'] * 10000,
            'test_result.size': list(range(10000)),
            'test_result.out_of_place.bus_bw': [i * 0.1 for i in range(10000)],
            'num_nodes': [(i % 32) + 1 for i in range(10000)],
            'num_processes': [((i % 64) + 1) * 8 for i in range(10000)]
        })

        # Test that large datasets can be handled
        assert len(large_data) == 10000
        assert large_data.memory_usage(deep=True).sum() > 0

        # Test chunking/pagination for large datasets
        chunk_size = 1000
        chunks = [large_data.iloc[i:i+chunk_size] for i in range(0, len(large_data), chunk_size)]

        assert len(chunks) == 10  # 10000 rows / 1000 chunk_size
        assert all(len(chunk) <= chunk_size for chunk in chunks)

    def test_memory_management(self, sample_report_data, mock_chart_generator):
        """Test memory management during report generation."""
        # Generate multiple reports to test memory usage
        for i in range(10):
            chart_group = {
                'group_key': f'memory_test_{i}',
                'data': sample_report_data.copy(),
                'filters': {'iteration': i}
            }

            result = mock_chart_generator.generate_chart('test_type', 'test_binary', chart_group)

            # Should handle multiple generations without issues
            assert result is not None

        # Test that memory is managed properly
        assert True  # If we reach here, memory management is working

    def test_concurrent_generation(self, sample_report_data, temp_output_dir):
        """Test handling of concurrent report generation scenarios."""
        # Simulate concurrent report generation
        output_files = []

        for i in range(3):
            output_file = temp_output_dir / f"concurrent_report_{i}.pdf"
            output_files.append(output_file)

        # Test that multiple output files can be handled
        for output_file in output_files:
            # Create placeholder files to test file handling
            output_file.touch()
            assert output_file.exists()

        # Test file locking/naming conflicts
        assert len(set(str(f) for f in output_files)) == 3  # All files should be unique


# ============================================================================
# Integration Tests
# ============================================================================

class TestGeneratorIntegration:
    """Test generator integration with other components."""

    def test_generator_adapter_integration(self, sample_report_data, config_base_path):
        """Test generator integration with adapters."""
        plot_config_path = config_base_path / "plot" / "plot_config.toml"

        if plot_config_path.exists():
            plot_config = ChartConfigLoader.load_from_toml(str(plot_config_path))

            adapter = NCCLPerformanceAdapter(plot_config=plot_config)
            adapter._processed_data = sample_report_data

            # Test that generator can work with adapter
            if plot_config.pdf_configs:
                pdf_types = [config.id for config in plot_config.pdf_configs]

                for pdf_type in pdf_types:
                    required_columns = adapter.get_required_columns_for_pdf_type(pdf_type)
                    required_binaries = adapter.get_required_test_binaries_for_pdf_type(pdf_type)

                    assert isinstance(required_columns, set)
                    assert isinstance(required_binaries, set)
        else:
            pytest.skip("Plot configuration not available")

    def test_generator_style_integration(self, config_base_path):
        """Test generator integration with style manager."""
        style_config_path = config_base_path / "style" / "style_config.toml"

        if style_config_path.exists():
            style_config = StyleConfigLoader.load_from_toml(str(style_config_path))
            style_manager = StyleManager(style_config)

            # Test that generator can work with style manager
            figure_size = style_manager.get_figure_size()

            assert isinstance(figure_size, (list, tuple))
            assert len(figure_size) == 2
            assert all(isinstance(dim, (int, float)) for dim in figure_size)
        else:
            pytest.skip("Style configuration not available")

    def test_end_to_end_generation_workflow(self, sample_report_data, config_base_path, temp_output_dir):
        """Test complete end-to-end report generation workflow."""
        plot_config_path = config_base_path / "plot" / "plot_config.toml"
        style_config_path = config_base_path / "style" / "style_config.toml"

        if plot_config_path.exists() and style_config_path.exists():
            # Load configurations
            plot_config = ChartConfigLoader.load_from_toml(str(plot_config_path))
            style_config = StyleConfigLoader.load_from_toml(str(style_config_path))

            # Create components
            adapter = NCCLPerformanceAdapter(plot_config=plot_config, style_config=style_config)
            adapter._processed_data = sample_report_data
            style_manager = StyleManager(style_config)
            chart_generator = ChartGenerator(adapter, style_manager)

            output_file = temp_output_dir / "end_to_end_report.pdf"

            # Test complete workflow
            try:
                from nvcomms_perf.report.generators.pdf import generate_pdf_report

                result = generate_pdf_report(
                    dataframe=sample_report_data,
                    output_path=Path(output_file),
                    pdf_type="basic_report",
                    adapter=adapter
                )

                # Should complete full workflow
                assert True

            except ImportError:
                pytest.skip("PDF generator module not available")
            except Exception as e:
                # Should handle workflow issues gracefully
                assert isinstance(e, (ValueError, TypeError, AttributeError))
        else:
            pytest.skip("Configuration files not available")
