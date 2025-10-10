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

"""Parser for input files."""

from pathlib import Path
import uuid
import tarfile
from typing import Optional, Callable

from ..factories.perf_record_factories import build_perf_record_from_nccl_test_result
from ..models.metadata_sweep import SweepMetadata
from ..models.metadata_test_set import TestSetMetadata, TestSetLibrary
from ..models.metadata_test import TestMetadata
from ..models.nccl_tests import NcclTestResult
from ..models.perf_record import PerfRecord
from .misc import hash_file

ParserCallback = Callable[[int, int, bool], None]  # records_uploaded, records_total, is_completed


def parse_nccl_test(path: Path, sweep_id: uuid.UUID, callback: Optional[ParserCallback] = None) -> PerfRecord:
    """
    Try to parse the file as JSON first. If it fails, try as plain text. If both fail, raise an error. Then convert to PerfRecord.
    """
    if callback:
        callback(0, 1, False)  # 0 record uploaded, 1 record total, to trigger the progress bar to start

    test_set_metadata = TestSetMetadata(library=TestSetLibrary(name="nccl"))
    try:
        with open(path, "r") as f:
            nccl_result = NcclTestResult.read_json(f)
    except Exception as json_exc:
        try:
            with open(path, "r") as f:
                nccl_result = NcclTestResult.read_plain_output(f)
        except Exception as text_exc:
            raise RuntimeError(
                f"Failed to parse {path} as JSON (error: {json_exc}) or as plain text (error: {text_exc})"
            )

    hash = hash_file(path)

    perf_record = build_perf_record_from_nccl_test_result(
        nccl_result, None, test_set_metadata, None, sweep_id, path.name, None, hash
    )
    perf_record.tool = "nccl-tests"

    if callback:
        callback(1, 1, True)  # 1 record uploaded, 1 record total

    return perf_record


def parse_tarball(
    tarball_path: Path,
    sweep_id: uuid.UUID,
    continue_on_error: bool = False,
    callback: Optional[ParserCallback] = None,
) -> tuple[list[PerfRecord], int, int]:
    perf_records = []
    hash = hash_file(tarball_path)
    total_log_files = 0

    with tarfile.open(tarball_path, "r:gz") as tar:
        # Get all members once
        all_members = tar.getmembers()

        # First pass: Find all valid test set directories (those with test_set_metadata.yml)
        test_set_dirs = set()
        for member in all_members:
            if member.name.endswith("test_set_metadata.yml"):
                test_set_dir = Path(member.name).parent
                test_set_dirs.add(test_set_dir)

        # Second pass: Only collect yml files from valid test set directories
        yml_files_dict = {}

        for member in all_members:
            if member.name.endswith(".yml") and member.isfile():
                test_set_dir = Path(member.name).parent
                if test_set_dir in test_set_dirs:
                    if test_set_dir not in yml_files_dict:
                        yml_files_dict[test_set_dir] = []
                    yml_files_dict[test_set_dir].append(member)
                    if not member.name.endswith("test_set_metadata.yml") and not member.name.endswith(
                        "test_sweep_metadata.yml"
                    ):
                        total_log_files += 1

        if callback:
            callback(0, total_log_files, False)

        for test_set_dir, yml_files in yml_files_dict.items():
            if not yml_files:
                continue

            sweep_metadata = None
            test_set_metadata = None

            # Read metadata files
            for yml_member in yml_files:
                yml_name = yml_member.name.split("/")[-1]
                if yml_name not in ["test_sweep_metadata.yml", "test_set_metadata.yml"]:
                    continue
                if fileobj := tar.extractfile(yml_member):
                    with fileobj as f:
                        try:
                            if yml_name == "test_sweep_metadata.yml":
                                sweep_metadata = SweepMetadata.read_yaml(f)
                            else:
                                test_set_metadata = TestSetMetadata.read_yaml(f)
                        except Exception as e:
                            if continue_on_error:
                                break
                            raise Exception(f"Error parsing {yml_member.name} in {tarball_path}") from e
                else:
                    if continue_on_error:
                        break
                    raise FileNotFoundError(f"{yml_member.name} not found in {test_set_dir}")

            if not sweep_metadata or not test_set_metadata:
                continue

            # Process test files
            for yml_member in yml_files:
                yml_name = yml_member.name.split("/")[-1]
                if yml_name in ["test_sweep_metadata.yml", "test_set_metadata.yml"]:
                    continue

                test_metadata = None
                if fileobj := tar.extractfile(yml_member.name):
                    with fileobj as f:
                        try:
                            test_metadata = TestMetadata.read_yaml(f)
                        except Exception as e:
                            if continue_on_error:
                                continue
                            raise Exception(f"Error parsing {yml_member.name} in {tarball_path}") from e
                else:
                    if continue_on_error:
                        continue
                    raise FileNotFoundError(f"{yml_member.name} not found in tarball")

                if not test_metadata:
                    if continue_on_error:
                        continue
                    raise Exception(f"Error parsing {yml_member.name} in {tarball_path}")

                # Find and read log file
                log_path = yml_member.name.replace(yml_name, test_metadata.log_name)
                try:
                    if fileobj := tar.extractfile(log_path):
                        with fileobj as f:
                            try:
                                nccl_test_result = NcclTestResult.read_plain_output(f)
                            except Exception as e:
                                if continue_on_error:
                                    continue
                                raise Exception(f"Error parsing {log_path} in {tarball_path}") from e
                    else:
                        if continue_on_error:
                            continue
                        raise FileNotFoundError(f"{log_path} not found in tarball")
                except KeyError:
                    if continue_on_error:
                        continue
                    raise FileNotFoundError(f"{log_path} not found in tarball")

                perf_records.append(
                    build_perf_record_from_nccl_test_result(
                        nccl_test_result,
                        test_metadata,
                        test_set_metadata,
                        sweep_metadata,
                        sweep_id,
                        tarball_path.name,
                        None,
                        hash,
                    )
                )

                if callback:
                    callback(len(perf_records), total_log_files, False)

    if callback:
        callback(len(perf_records), total_log_files, True)

    return perf_records, total_log_files, total_log_files - len(perf_records)
