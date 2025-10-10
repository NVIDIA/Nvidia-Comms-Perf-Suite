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

"""Test tarball to PDF report generation."""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from nvcomms_perf.report.cli.report import _convert_records_to_dataframe


class TestTarballToPDF:
    """Test tarball parsing and PDF generation."""

    def test_convert_records_with_tool_column(self):
        """Test that records with 'tool' column get converted to 'tool_binary'."""
        # Create mock records
        mock_record = Mock()
        mock_record.model_dump.return_value = {
            'tool': 'nccl-tests',  # Tarball uses 'tool' not 'tool_binary'
            'test_result': [
                {
                    'size': 1024,
                    'out_of_place': {'bus_bw': 100.0, 'alg_bw': 110.0},
                    'in_place': {'bus_bw': 95.0, 'alg_bw': 105.0}
                }
            ],
            'num_nodes': 4,
            'num_processes': 32
        }

        # Convert records
        df = _convert_records_to_dataframe([mock_record])

        # Check that tool_binary column exists
        assert 'tool_binary' in df.columns
        assert df['tool_binary'].iloc[0] == 'nccl-tests'

        # Check that test_result columns are properly prefixed
        assert 'test_result.size' in df.columns
        assert 'test_result.out_of_place.bus_bw' in df.columns
        assert 'test_result.in_place.bus_bw' in df.columns

    def test_convert_records_with_path_in_tool_binary(self):
        """Test that full paths in tool_binary get converted to just binary names."""
        mock_record = Mock()
        mock_record.model_dump.return_value = {
            'tool': '/usr/local/bin/nccl-tests',  # Full path
            'test_result': [{'size': 1024}],
            'num_nodes': 4,
            'num_processes': 32
        }

        df = _convert_records_to_dataframe([mock_record])

        # Check that path is stripped
        assert 'tool_binary' in df.columns
        assert df['tool_binary'].iloc[0] == 'nccl-tests'

    def test_column_normalization_preserves_existing_columns(self):
        """Test that column normalization doesn't break existing correct columns."""
        mock_record = Mock()
        mock_record.model_dump.return_value = {
            'tool_binary': 'nccl-tests',  # Already correct name
            'test_result': [
                {
                    'size': 1024,
                    'out_of_place': {'bus_bw': 100.0},
                    'in_place': {'bus_bw': 95.0}
                }
            ],
            'num_nodes': 4,
            'num_processes': 32
        }

        df = _convert_records_to_dataframe([mock_record])

        # Should still have tool_binary
        assert 'tool_binary' in df.columns
        assert df['tool_binary'].iloc[0] == 'nccl-tests'

        # All other columns should be present
        assert 'test_result.size' in df.columns
        assert 'num_nodes' in df.columns

    @patch('nvcomms_perf.report.cli.report.logger')
    def test_column_renaming_logged(self, mock_logger):
        """Test that column renaming is properly logged."""
        mock_record = Mock()
        mock_record.model_dump.return_value = {
            'tool': 'nccl-tests',
            'test_result': [{'size': 1024}],
            'num_nodes': 4
        }

        _convert_records_to_dataframe([mock_record])

        # Check that renaming was logged
        mock_logger.info.assert_any_call("Renaming column 'tool' to 'tool_binary' for compatibility")
