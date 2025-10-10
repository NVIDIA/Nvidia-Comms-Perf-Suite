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

import pandas as pd
import importlib.resources
from pathlib import Path
from nvcomms_perf.report.adapters.nccl_adapter import NCCLPerformanceAdapter
from nvcomms_perf.report.models.plot_config import ChartConfigLoader
from nvcomms_perf.report.models.style_config import StyleConfigLoader
from nvcomms_perf.report.plotting.chart_generator import ChartGenerator
from nvcomms_perf.report.plotting.style_manager import StyleManager

def _get_config_path(config_file: str) -> Path:
    """Get the configuration file path using package-relative paths.

    Args:
        config_file: The configuration file name (e.g., 'config/nccl/plot/plot_config.toml')

    Returns:
        Path to the configuration file
    """
    # Try to find the config file using importlib.resources
    try:
        if hasattr(importlib.resources, 'files'):
            package_files = importlib.resources.files('nvcomms_perf.report')
            config_path = package_files / config_file
            if config_path.is_file():
                return Path(str(config_path))
    except Exception:
        pass

    # Try relative to the module
    module_path = Path(__file__).parent.parent.parent.parent / 'src' / 'nvcomms_perf' / 'report'
    config_path = module_path / config_file
    if config_path.exists():
        return config_path

    # Try from project root
    project_root = Path(__file__).parent.parent.parent.parent
    config_path = project_root / 'src' / 'nvcomms_perf' / 'report' / config_file
    if config_path.exists():
        return config_path

    raise FileNotFoundError(f"Could not find configuration file: {config_file}")

def test_empty_figure_handling():
    """Test that empty charts, data tables, and command pages return None instead of empty figures."""
    # Load configurations using package-relative paths
    plot_config_path = _get_config_path('config/nccl/plot/plot_config.toml')
    style_config_path = _get_config_path('config/nccl/style/style_config.toml')

    plot_config = ChartConfigLoader.load_from_toml(str(plot_config_path))
    style_config = StyleConfigLoader.load_from_toml(str(style_config_path))

    # Create adapter
    adapter = NCCLPerformanceAdapter(plot_config=plot_config, style_config=style_config)

    # Create empty data with proper data types
    empty_data = pd.DataFrame({
        'tool_binary': pd.Series(dtype='object'),
        'test_result.size': pd.Series(dtype='object'),
        'test_result.out_of_place.bus_bw': pd.Series(dtype='object'),
        'test_result.in_place.bus_bw': pd.Series(dtype='object'),
        'num_nodes': pd.Series(dtype='object'),
        'num_processes': pd.Series(dtype='object'),
        'test_config.local_register': pd.Series(dtype='object'),
        'test_env_vars.NCCL_TESTS_SPLIT_MASK': pd.Series(dtype='object'),
        'test_env_vars.NCCL_ALGO': pd.Series(dtype='object'),
        'test_env_vars.NCCL_PROTO': pd.Series(dtype='object')
    })

    # Process the data directly
    adapter.process_data(empty_data)

    # Create style manager and chart generator
    style_manager = StyleManager(style_config)
    chart_generator = ChartGenerator(adapter, style_manager)

    # Create a mock chart group with empty data
    chart_group = {
        'group_key': 'test_empty_group',
        'data': empty_data,
        'filters': {}
    }

    # Test that create_data_table returns empty list for empty data
    table_figures = chart_generator.create_data_table('default_comparison', 'all_reduce_perf', chart_group)
    assert table_figures == [], "Empty data table should return empty list, not figures"

    # Test that generate_chart returns None for empty data
    chart_fig = chart_generator.generate_chart('default_comparison', 'all_reduce_perf', chart_group)
    assert chart_fig is None, "Empty chart should return None, not a figure"

    # Test that create_command_page returns None for empty data
    cmd_fig = chart_generator.create_command_page('default_comparison', 'all_reduce_perf', chart_group)
    assert cmd_fig is None, "Empty command page should return None, not a figure"
