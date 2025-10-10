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

import logging
import uuid
from typing import Any, Optional, Type, Union
from datetime import datetime, timezone
from sqlmodel import SQLModel, Field
from sqlalchemy import Column, DateTime, literal, func
from sqlalchemy.orm import column_property
from sqlalchemy.types import JSON, TypeDecorator
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.mysql import JSON as MySQLJSON
from sqlalchemy.dialects.oracle import CLOB
from sqlalchemy.dialects.mssql import VARCHAR
from pydantic import BaseModel, ConfigDict
from pydantic.types import NonNegativeInt, PositiveInt

logger = logging.getLogger(__name__)

json_type = (
    JSONB()
    .with_variant(JSON(), "sqlite")
    .with_variant(MySQLJSON(), "mysql")
    .with_variant(CLOB(), "oracle")
    .with_variant(VARCHAR(), "mssql")
)


class PydanticType(TypeDecorator):
    """SQLAlchemy type to store Pydantic models as JSON"""

    impl = json_type
    cache_ok = True

    def __init__(self, pydantic_type: Type[BaseModel], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pydantic_type = pydantic_type

    def process_bind_param(self, value, dialect) -> Union[dict[str, Any], Any, None]:
        if value is not None:
            if isinstance(value, BaseModel):
                # Use mode='json' to handle datetime serialization properly
                return value.model_dump(mode="json")
            elif isinstance(value, dict):
                # Validate the dict by creating a Pydantic model, then return the dict
                return self._pydantic_type(**value).model_dump(mode="json")
            return value
        return value

    def process_result_value(self, value, dialect) -> Optional[BaseModel]:
        if value is not None:
            return self._pydantic_type(**value)
        return value

    def copy(self, **kw) -> "PydanticType":
        """Produce a copy of this PydanticType instance."""
        return self.__class__(self._pydantic_type, **kw)


class RecordSystemInfo(BaseModel):
    system_name: str
    gpu_arch_type: Optional[str]
    cpu_model_name: Optional[str]
    cpu_arch_type: Optional[str]
    os_type: Optional[str]
    os_version: Optional[str]
    linux_kernel_version: Optional[str]
    adaptive_routing: Optional[bool]
    cuda_driver_version: Optional[str]

    model_config = ConfigDict(extra="allow")


class RecordNetworkInfo(BaseModel):
    nics: Optional[str]
    switch_name: Optional[str]
    switch_type: Optional[str]
    switch_version: Optional[str]
    network_name: Optional[str]
    network_version: Optional[str]
    network_plugin_name: Optional[str]
    network_plugin_version: Optional[str]
    mofed_version: Optional[str]
    libfabric_version: Optional[str]

    model_config = ConfigDict(extra="allow")


class RecordNcclTestDeviceInfo(BaseModel):
    rank: NonNegativeInt
    group: NonNegativeInt
    pid: PositiveInt
    hostname: str
    device: NonNegativeInt
    device_hex: str
    device_info: str


class RecordNcclTestsMetadataInfo(BaseModel):
    start_time: Optional[datetime] = Field(default=None, nullable=True)
    end_time: Optional[datetime] = Field(default=None, nullable=True)
    devices: Optional[list[RecordNcclTestDeviceInfo]] = Field(default=None, nullable=True)

    default_config: Optional[bool] = Field(default=None, nullable=True)
    iteration: Optional[int] = Field(default=None, nullable=True)

    model_config = ConfigDict(extra="allow")


class RecordNcclTestsConfig(BaseModel):
    # Required fields from NCCL test output header
    nthreads: PositiveInt
    ngpus: PositiveInt
    minbytes: PositiveInt
    maxbytes: PositiveInt
    warmup_iters: NonNegativeInt
    iters: PositiveInt
    agg_iters: PositiveInt
    check: bool  # Maps from "validation" in header
    cudagraph: int  # Maps from "graph" in header

    # Optional fields for additional configuration
    blocking: Optional[bool] = False
    parallel_init: Optional[bool] = False
    local_register: Optional[str] = "disabled"

    model_config = ConfigDict(extra="allow")


class PerfRecord(SQLModel, table=True):
    # test properties
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True, nullable=False, description="Unique identifier for the test record")
    name: Optional[str] = Field(default=None, index=True, nullable=True, description="Name of the test")
    description: Optional[str] = Field(default=None, nullable=True, description="Description of the test")
    log_file: Optional[str] = Field(default=None, nullable=True, description="Path to the log file")

    # test set properties
    test_set_name: Optional[str] = Field(default=None, nullable=True, description="Name of the test set")
    test_set_description: Optional[str] = Field(default=None, nullable=True, description="Description of the test set")

    # sweep properties
    sweep_id: uuid.UUID = Field(index=True, nullable=False, description="Unique identifier for the sweep")
    sweep_name: Optional[str] = Field(default=None, nullable=True, description="Name of the sweep")
    sweep_description: Optional[str] = Field(default=None, nullable=True, description="Description of the sweep")
    sweep_dir: Optional[str] = Field(default=None, nullable=True, description="Path to the sweep directory")

    # test target
    tool: str = Field(index=True, nullable=False, description="tool used to run the test")
    tool_binary: Optional[str] = Field(default=None, nullable=True, description="Path to the tool binary")
    tool_version: Optional[str] = Field(default=None, index=True, nullable=True, description="Version of the tool")
    library: str = Field(index=True, nullable=False, description="library tested by the tool")
    library_binary: Optional[str] = Field(default=None, nullable=True, description="Path to the library binary")
    library_version: Optional[str] = Field(default=None, index=True, nullable=True, description="Version of the library")
    library_info: Optional[dict] = Field(default=None, sa_column=Column(json_type), description="Additional information about the library")

    # test environment
    cluster_name: Optional[str] = Field(default=None, index=True, nullable=True, description="Name of the cluster")
    system_info: Optional[RecordSystemInfo] = Field(
        default=None, sa_column=Column(PydanticType(RecordSystemInfo), nullable=True), description="System information"
    )

    network_info: Optional[RecordNetworkInfo] = Field(
        default=None, sa_column=Column(PydanticType(RecordNetworkInfo), nullable=True), description="Network information"
    )

    num_nodes: Optional[int] = Field(default=None, nullable=True, description="Number of nodes in the test")
    node_list: Optional[str] = Field(default=None, nullable=True, description="List of nodes in the test")
    num_processes: Optional[int] = Field(default=None, nullable=True, description="Number of processes in the test")
    num_process_per_node: Optional[int] = Field(default=None, nullable=True, description="Number of processes per node in the test")
    container: Optional[str] = Field(default=None, nullable=True, description="Container used to run the test")
    slurm_job_id: Optional[int] = Field(default=None, nullable=True, index=True, description="SLURM job ID")
    slurm_job_name: Optional[str] = Field(default=None, nullable=True, description="SLURM job name")
    # test details
    test_cmd: Optional[str] = Field(default=None, nullable=True, description="Command used to run the test")

    test_metadata: Optional[RecordNcclTestsMetadataInfo] = Field(
        default=None, sa_column=Column(PydanticType(RecordNcclTestsMetadataInfo), nullable=True), description="Metadata information about the test"
    )

    test_env_vars: Optional[dict] = Field(default=None, sa_column=Column(json_type), description="Environment variables used to run the test")

    test_config: Optional[RecordNcclTestsConfig] = Field(
        default=None, sa_column=Column(PydanticType(RecordNcclTestsConfig), nullable=True), description="Configuration of the test"
    )

    test_result: list[dict] = Field(sa_column=Column(json_type, nullable=False, comment="list[NcclTestBenchmarkRow]"), description="Result of the test")

    # miscellaneous
    source: Optional[str] = Field(default=None, nullable=True, description="File that contains the test result")
    created_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), sa_column=Column(DateTime(timezone=True), nullable=False), description="Timestamp when the test was created"
    )
    collected_by: Optional[str] = Field(default=None, nullable=True, description="User who collected the test")
    uploaded_by: Optional[str] = Field(default=None, nullable=True, description="User who uploaded the test")
    raw_data_path: Optional[str] = Field(default=None, nullable=True, description="Path to the raw data")
    raw_data_hash: str = Field(nullable=False, index=True, description="Hash of the raw data file")

# Add column properties to the model to combine two fields into a single indexable field, this is used for dashboard filtering
# Use string concatenation with || operator which works in SQLite and other databases for better compatibility
# sqlite before 3.44.0 does not support the concat function
PerfRecord.tool_and_version = column_property(
    PerfRecord.tool + literal(":") + func.coalesce(PerfRecord.tool_version, "Unknown")
)
PerfRecord.library_and_version = column_property(
    PerfRecord.library + literal(":") + func.coalesce(PerfRecord.library_version, "Unknown")
)
