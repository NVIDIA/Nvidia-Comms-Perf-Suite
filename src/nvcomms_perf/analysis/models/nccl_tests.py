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
import re
import shlex
from typing import Any, Literal, Optional, Annotated, Union, IO
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    BeforeValidator,
    computed_field,
    model_validator,
)
from datetime import datetime

logger = logging.getLogger(__name__)


def _convert_nccl_version(version: Optional[int]) -> Optional[str]:
    """
    Convert NCCL version from numeric format to semantic version (X.Y.Z).

    The numeric format is calculated as:
    `#define NCCL_VERSION(X,Y,Z) (((X) <= 2 && (Y) <= 8) ? (X) * 1000 + (Y) * 100 + (Z) : (X) * 10000 + (Y) * 100 + (Z))`
    - For X <= 2 and Y <= 8: X * 1000 + Y * 100 + Z
    - For other cases: X * 10000 + Y * 100 + Z

    Examples:
    - 22702 -> 2.27.2
    - 21800 -> 2.18.0
    """
    if version is None:
        return None

    if version <= 2800:  # X <= 2 and Y <= 8 case
        x = version // 1000
        y = (version % 1000) // 100
        z = version % 100
    else:  # X * 10000 + Y * 100 + Z case
        x = version // 10000
        y = (version % 10000) // 100
        z = version % 100

    return f"{x}.{y}.{z}"


class NcclTestCmdOptions(BaseModel):
    """
    Pydantic model for parsing NCCL test command-line options.

    Long option names are used as field names, and short option names are used as validation aliases.

    Examples:
        # From command string (includes executable)
        options = NcclTestCmdOptions.from_args("./nccl_test -b 32 -e 4096 -i 1024 -J ./output.json")
        options = NcclTestCmdOptions.from_args("all_gather_perf -t1 -g1 -f2 -n20 -w5 -b8 -e16G")

        # From dictionary using short options (validation aliases)
        options = NcclTestCmdOptions(b=32, e=4096, i=1024, J="./output.json")

        # From dictionary using long options (field names)
        options = NcclTestCmdOptions(minbytes=32, maxbytes=4096, stepbytes=1024, output_file="./output.json")

        # Serialize excluding None values
        clean_dict = options.model_dump(exclude_none=True)
        clean_json = options.model_dump_json(exclude_none=True)
    """

    nthreads: Optional[PositiveInt] = Field(default=None, validation_alias="t")
    ngpus: Optional[PositiveInt] = Field(default=None, validation_alias="g")
    minbytes: Optional[PositiveInt] = Field(default=None, validation_alias="b")
    maxbytes: Optional[PositiveInt] = Field(default=None, validation_alias="e")
    stepbytes: Optional[NonNegativeInt] = Field(default=None, validation_alias="i")
    stepfactor: Optional[NonNegativeFloat] = Field(default=None, validation_alias="f")
    iters: Optional[PositiveInt] = Field(default=None, validation_alias="n")
    agg_iters: Optional[PositiveInt] = Field(default=None, validation_alias="m")
    warmup_iters: Optional[NonNegativeInt] = Field(default=None, validation_alias="w")
    run_cycles: Optional[PositiveInt] = Field(default=None, validation_alias="N")
    parallel_init: Optional[bool] = Field(default=None, validation_alias="p")
    check: Optional[bool] = Field(default=None, validation_alias="c")
    op: Optional[str] = Field(default=None, validation_alias="o")
    datatype: Optional[str] = Field(default=None, validation_alias="d")
    root: Optional[int] = Field(default=None, validation_alias="r")
    blocking: Optional[bool] = Field(default=None, validation_alias="z")
    stream_null: Optional[bool] = Field(default=None, validation_alias="y")
    side_comp: Optional[bool] = Field(default=None, validation_alias="k")
    replay: Optional[str] = Field(default=None, validation_alias="l")
    timeout: Optional[Union[int, float]] = Field(default=None, validation_alias="T")
    cudagraph: Optional[int] = Field(default=None, validation_alias="G")
    report_cputime: Optional[bool] = Field(default=None, validation_alias="C")
    out_of_place: Optional[bool] = Field(default=None, validation_alias="O")
    unalign: Optional[int] = Field(default=None, validation_alias="u")
    output_file: Optional[str] = Field(default=None, validation_alias="J")
    average: Optional[int] = Field(default=None, validation_alias="a")
    commblocking: Optional[int] = Field(default=None, validation_alias="B")
    ft_test: Optional[bool] = Field(default=None, validation_alias="F")
    ft_list: Optional[str] = Field(default=None, validation_alias="L")
    tbytes: Optional[PositiveInt] = Field(default=None, validation_alias="s")
    split_share: Optional[int] = Field(default=None, validation_alias="S")
    split_comm: Optional[int] = Field(default=None, validation_alias="P")
    local_register: Optional[str] = Field(default=None, validation_alias="R")
    per_coll_perf: Optional[int] = Field(default=None, validation_alias="A")
    simulate: Optional[bool] = Field(default=None, validation_alias="E")
    init_ids: Optional[str] = Field(default=None, validation_alias="I")
    traffic_class: Optional[int] = Field(default=None, validation_alias="q")
    help: Optional[bool] = Field(default=None, validation_alias="h", exclude=True)

    model_config = ConfigDict(
        extra="allow",
        validate_by_alias=True,
        validate_by_name=True,
        serialize_by_alias=False,
    )

    @classmethod
    def from_args(cls, args: str) -> "NcclTestCmdOptions":
        """
        Parse command-line arguments into NcclTestCmdOptions.

        Args:
            args: Command string including the executable.
                  e.g., "nccl_test -b 32 -e 4096 -i 1024"
                  or "./all_gather_perf -t1 -g1 -f2 -n20 -w5"

        Returns:
            NcclTestCmdOptions instance with parsed options
        """
        import argparse
        import shlex

        def parse_size(value: str) -> int:
            """Parse size values with optional suffixes (K, M, G)."""
            # Match the C parsesize function behavior
            value = value.strip()
            if not value:
                return 0

            # Try to parse as "number suffix" format
            import re

            match = re.match(r"^(\d+(?:\.\d+)?)\s*([KMGkmg]?)$", value)
            if match:
                size = float(match.group(1))
                suffix = match.group(2).upper() if match.group(2) else ""

                if suffix == "K":
                    return int(size * 1024)
                elif suffix == "M":
                    return int(size * 1024 * 1024)
                elif suffix == "G":
                    return int(size * 1024 * 1024 * 1024)
                else:
                    return int(size)

            # Fallback to simple int parsing
            try:
                return int(value, 0)
            except ValueError:
                raise argparse.ArgumentTypeError(f"invalid size specified: {value}")

        def parse_int(value: str) -> int:
            """Parse int values, supporting decimal, hex (0x), and octal (0) formats."""
            # Match C's strtol behavior with base 0
            value = value.strip()
            if value.startswith("0x") or value.startswith("0X"):
                return int(value, 16)
            elif value.startswith("0") and len(value) > 1 and value[1].isdigit():
                return int(value, 8)
            else:
                return int(value, 10)

        def parse_bool(value: str) -> bool:
            """Parse boolean values from 0/1."""
            int_val = parse_int(value)
            return bool(int_val)

        def parse_local_register(value: str) -> str:
            """Parse -R option following NCCL's logic."""
            # Match the C implementation from NCCL
            if value and value[0] == "s":
                return "LOCAL_REGISTER_SEND"
            elif value and value[0] == "r":
                return "LOCAL_REGISTER_RECV"
            elif value and value[0] == "a":
                return "LOCAL_REGISTER_ALL"  # LOCAL_REGISTER_SEND|LOCAL_REGISTER_RECV
            else:
                # Try to parse as int
                try:
                    # Use the same logic as parse_int for consistent int parsing
                    value = value.strip()
                    if value.startswith("0x") or value.startswith("0X"):
                        int_val = int(value, 16)
                    elif value.startswith("0") and len(value) > 1 and value[1].isdigit():
                        int_val = int(value, 8)
                    else:
                        int_val = int(value, 10)

                    if int_val == 0:
                        return "disabled"
                    elif int_val == 1:
                        return "LOCAL_REGISTER_ALL"  # LOCAL_REGISTER_SEND|LOCAL_REGISTER_RECV
                    elif int_val == 2:
                        return "SYMMETRIC_REGISTER_ALL"
                    else:
                        return str(int_val)  # For other values, return as string
                except ValueError:
                    return value

        # Split the command string and skip the executable
        args_list = shlex.split(args)[1:]  # Skip executable

        # Create parser with add_help=False to handle -h ourselves
        parser = argparse.ArgumentParser(add_help=False)

        # Add all arguments
        parser.add_argument("-t", "--nthreads", type=parse_int)
        parser.add_argument("-g", "--ngpus", type=parse_int)
        parser.add_argument("-b", "--minbytes", type=parse_size)
        parser.add_argument("-e", "--maxbytes", type=parse_size)
        parser.add_argument("-i", "--stepbytes", type=parse_int)
        parser.add_argument("-f", "--stepfactor", type=float)
        parser.add_argument("-s", "--tbytes", type=parse_size)
        parser.add_argument("-n", "--iters", type=parse_int)
        parser.add_argument("-m", "--agg_iters", type=parse_int)
        parser.add_argument("-w", "--warmup_iters", type=parse_int)
        parser.add_argument("-N", "--run_cycles", type=parse_int)
        parser.add_argument("-p", "--parallel_init", type=parse_bool)
        parser.add_argument("-c", "--check", type=parse_bool)
        parser.add_argument("-o", "--op", type=str)
        parser.add_argument("-d", "--datatype", type=str)
        parser.add_argument("-r", "--root", type=parse_int)
        parser.add_argument("-z", "--blocking", type=parse_bool)
        parser.add_argument("-y", "--stream_null", type=parse_bool)
        parser.add_argument("-k", "--side_comp", type=parse_bool)
        parser.add_argument("-l", "--replay", type=str)
        parser.add_argument("-T", "--timeout", type=float)
        parser.add_argument("-G", "--cudagraph", type=parse_int)
        parser.add_argument("-C", "--report_cputime", type=parse_bool)
        parser.add_argument("-O", "--out_of_place", type=parse_bool)
        parser.add_argument("-u", "--unalign", type=parse_int)
        parser.add_argument("-J", "--output_file", type=str)
        parser.add_argument("-a", "--average", type=parse_int)
        parser.add_argument("-B", "--commblocking", type=parse_int)
        parser.add_argument("-F", "--ft_test", type=parse_bool)
        parser.add_argument("-L", "--ft_list", type=str)
        parser.add_argument("-S", "--split_share", type=parse_int)
        parser.add_argument("-P", "--split_comm", type=parse_int)
        parser.add_argument("-R", "--local_register", type=parse_local_register)
        parser.add_argument("-A", "--per_coll_perf", type=parse_int)
        parser.add_argument("-E", "--simulate", type=parse_bool)
        parser.add_argument("-I", "--init_ids", type=str)
        parser.add_argument("-q", "--traffic_class", type=parse_int)
        parser.add_argument("-h", "--help", action="store_true")

        try:
            # Parse known args (ignore unknown ones)
            parsed_args, _ = parser.parse_known_args(args_list)

            # Convert to dict, using short option names for validation
            parsed_dict = {}
            for short_opt, long_opt in [
                ("t", "nthreads"),
                ("g", "ngpus"),
                ("b", "minbytes"),
                ("e", "maxbytes"),
                ("i", "stepbytes"),
                ("f", "stepfactor"),
                ("s", "tbytes"),
                ("n", "iters"),
                ("m", "agg_iters"),
                ("w", "warmup_iters"),
                ("N", "run_cycles"),
                ("p", "parallel_init"),
                ("c", "check"),
                ("o", "op"),
                ("d", "datatype"),
                ("r", "root"),
                ("z", "blocking"),
                ("y", "stream_null"),
                ("k", "side_comp"),
                ("l", "replay"),
                ("T", "timeout"),
                ("G", "cudagraph"),
                ("C", "report_cputime"),
                ("O", "out_of_place"),
                ("u", "unalign"),
                ("J", "output_file"),
                ("a", "average"),
                ("B", "commblocking"),
                ("F", "ft_test"),
                ("L", "ft_list"),
                ("S", "split_share"),
                ("P", "split_comm"),
                ("R", "local_register"),
                ("A", "per_coll_perf"),
                ("E", "simulate"),
                ("I", "init_ids"),
                ("q", "traffic_class"),
                ("h", "help"),
            ]:
                value = getattr(parsed_args, long_opt, None)
                if value is not None:
                    parsed_dict[short_opt] = value

            return cls.model_validate(parsed_dict)

        except (argparse.ArgumentError, SystemExit):
            # If argparse fails, return empty options
            return cls()


class NcclTestDevice(BaseModel):
    rank: NonNegativeInt
    group: NonNegativeInt
    pid: PositiveInt
    hostname: str
    device: NonNegativeInt
    device_hex: str
    device_info: str

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        validate_by_alias=True,
        validate_by_name=True,
        serialize_by_alias=False,
    )


class NcclTestConfig(BaseModel):
    n_threads: PositiveInt = Field(validation_alias="nthreads")
    n_gpus: PositiveInt = Field(validation_alias="ngpus")
    min_bytes: PositiveInt = Field(validation_alias="minimum_bytes")
    max_bytes: PositiveInt = Field(validation_alias="maximum_bytes")
    step: NonNegativeInt = Field(validation_alias="step")
    step_type: Literal["factor", "bytes"]
    warmup_iters: NonNegativeInt = Field(validation_alias="warmup_iters")
    iters: PositiveInt = Field(validation_alias="iterations")
    agg_iters: PositiveInt = Field(validation_alias="aggregated_iterations")
    validation: int
    graph: NonNegativeInt = Field(validation_alias="graph")
    blocking_collectives: Optional[bool] = None
    parallel_init: Optional[bool] = None

    devices: list[NcclTestDevice]

    @model_validator(mode="before")
    @classmethod
    def unify_step(cls, values) -> dict:
        if "step_bytes" in values:
            values["step_type"] = "bytes"
            values["step"] = values.pop("step_bytes")
        elif "step_factor" in values:
            values["step_type"] = "factor"
            values["step"] = values.pop("step_factor")
        return values

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        validate_by_alias=True,
        validate_by_name=True,
        serialize_by_alias=False,
    )


class NcclTestBenchmark(BaseModel):
    time: NonNegativeFloat
    alg_bw: NonNegativeFloat
    bus_bw: NonNegativeFloat
    time_max: Optional[NonNegativeFloat] = None
    alg_bw_max: Optional[NonNegativeFloat] = None
    bus_bw_max: Optional[NonNegativeFloat] = None
    nwrong: Any
    side_comp_bw: Any = None
    estimated_time: Any = None

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        validate_by_alias=True,
        validate_by_name=True,
        serialize_by_alias=False,
    )


class NcclTestBenchmarkRow(BaseModel):
    size: NonNegativeInt
    count: NonNegativeInt
    type: str
    redop: str
    root: int
    out_of_place: NcclTestBenchmark
    in_place: NcclTestBenchmark
    actual_iterations: Optional[NonNegativeInt] = None
    experiment_name: Optional[str] = None

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        validate_by_alias=True,
        validate_by_name=True,
        serialize_by_alias=False,
    )

    def flatten(self) -> dict:
        flat = {
            "size": self.size,
            "count": self.count,
            "type": self.type,
            "redop": self.redop,
            "root": self.root,
            **{
                f"out_of_place.{k}": v for k, v in (self.out_of_place.model_dump().items() if self.out_of_place else [])
            },
            **{f"in_place.{k}": v for k, v in (self.in_place.model_dump().items() if self.in_place else [])},
            "actual_iterations": self.actual_iterations,
            "experiment_name": self.experiment_name,
        }
        return flat


class NcclTestOutOfBound(BaseModel):
    count: NonNegativeInt
    okay: Annotated[str, BeforeValidator(lambda x: "true" if x.upper() in ("OK", "TRUE") else "false")]

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        validate_by_alias=True,
        validate_by_name=True,
        serialize_by_alias=False,
    )


class NcclTestAvgBusBw(BaseModel):
    bandwidth: NonNegativeFloat = Field(validation_alias="bandwidith")
    okay: Annotated[
        str, BeforeValidator(lambda x: "unchecked" if not x else "true" if x.upper() in ("OK", "TRUE") else "false")
    ]

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        validate_by_alias=True,
        validate_by_name=True,
        serialize_by_alias=False,
    )


class NcclTestResult(BaseModel):
    """
    Represents the result of an nccl-tests run.

    Note:
        Recommend to use the provided properties (`test_config`, `device_list`, `result_dataframe`, etc.) to access relevant data.
    """

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    args: Annotated[Optional[str], BeforeValidator(lambda x: None if x is None else shlex.join(x) if isinstance(x, list) else x), Field(default=None)]
    env: Annotated[
        Optional[dict[str, str]],
        BeforeValidator(lambda x: {k: v for k, v in (item.split("=", 1) for item in x)} if x else None),
    ] = None
    nccl_version: Annotated[Optional[str], BeforeValidator(_convert_nccl_version)] = None

    config: NcclTestConfig
    results: list[NcclTestBenchmarkRow]

    out_of_bounds: NcclTestOutOfBound
    average_bus_bandwidth: NcclTestAvgBusBw = Field(validation_alias="average_bus_bandwidith")

    errors: Optional[list[str]] = None

    @computed_field
    @property
    def cmd_options(self) -> Optional[NcclTestCmdOptions]:
        if not self.args:
            return None

        # self.args is always a string due to BeforeValidator
        return NcclTestCmdOptions.from_args(self.args)

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        validate_by_alias=True,
        validate_by_name=True,
        serialize_by_alias=False,
    )

    @model_validator(mode="after")
    def validate_num_benchmark_rows(self):
        cfg = self.config
        import math

        if cfg.step_type == "factor":
            if cfg.step <= 1:
                raise ValueError(f"Step must be > 1 for factor step_type, got {cfg.step}")
            expected = math.floor(math.log(cfg.max_bytes / cfg.min_bytes, cfg.step)) + 1
            actual = len(self.results)
            if actual != expected:
                raise ValueError(
                    f"Expected {expected} benchmark rows for maxBytes {cfg.max_bytes}, minBytes {cfg.min_bytes}, step: {cfg.step}({cfg.step_type}), but got {actual}."
                )
        elif cfg.step_type == "bytes":
            if cfg.step <= 0:
                raise ValueError(f"Step must be > 0 for bytes step_type, got {cfg.step}")
            expected = math.floor((cfg.max_bytes - cfg.min_bytes) / cfg.step) + 1
            actual = len(self.results)
            if actual != expected:
                raise ValueError(
                    f"Expected {expected} benchmark rows for maxBytes {cfg.max_bytes}, minBytes {cfg.min_bytes}, step: {cfg.step}({cfg.step_type}), but got {actual}."
                )
        return self

    @classmethod
    def read_json(cls, obj: Union[str, bytes, bytearray, memoryview, IO]) -> "NcclTestResult":
        """
        Parse a NcclTestResult from JSON content.

        Accepts:
        - str: JSON string
        - bytes: JSON content as bytes
        - file-like object: Any object with a .read() method
        """
        if isinstance(obj, (str, bytes)):
            return cls.model_validate_json(obj)
        elif isinstance(obj, (bytearray, memoryview)):
            return cls.model_validate_json(bytes(obj))
        elif hasattr(obj, "read"):
            return cls.model_validate_json(obj.read())
        else:
            raise TypeError(
                "Unsupported input type for read_json; must be str, bytes, bytearray, memoryview, or file-like object."
            )

    @classmethod
    def read_plain_output(cls, obj: Union[str, bytes, bytearray, memoryview, IO]) -> "NcclTestResult":
        """
        Parse a NcclTestResult from plain text NCCL test output.

        Accepts:
        - str: plain text output
        - bytes: plain text as bytes
        - bytearray, memoryview: plain text as bytes-like
        - file-like object: Any object with a .read() method
        """
        if isinstance(obj, str):
            return cls._read_plain_output_str(obj)
        elif isinstance(obj, (bytes, bytearray, memoryview)):
            return cls._read_plain_output_str(bytes(obj).decode())
        elif hasattr(obj, "read"):
            content = obj.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            return cls._read_plain_output_str(content)
        else:
            raise TypeError(
                "Unsupported input type for read_plain_output; must be str, bytes, bytearray, memoryview, or file-like object."
            )

    @classmethod
    def _read_plain_output_str(cls, output: str) -> "NcclTestResult":
        """
        Parse a NcclTestResult from a plain text NCCL test output string.

        Args:
            output (str): Plain text output from an NCCL test.

        Returns:
            NcclTestResult: Parsed result object.

        Raises:
            ValueError: If the string cannot be parsed as a valid NCCL test output.
        """
        all_lines = output.splitlines()

        arguments_pattern = re.compile(
            r"\#\s+nThread\s+(?P<n_threads>\d+)\s+nGpus\s+(?P<n_gpus>\d+)\s+minBytes\s+(?P<min_bytes>\d+)\s+maxBytes\s+(?P<max_bytes>\d+)\s+"
            r"step:\s+(?P<step>\d+)\((?P<step_type>factor|bytes)\)\s+warmup\s+iters:\s+(?P<warmup_iters>\d+)\s+iters:\s+(?P<iters>\d+)\s+"
            r"agg iters:\s+(?P<agg_iters>\d+)\s+validation:\s+(?P<validation>\d+)\s+graph:\s+(?P<graph>\d+)"
        )
        device_info_pattern = re.compile(
            r"\#\s+Rank\s+(?P<rank>\d+)\s+Group\s+(?P<group>\d+)\s+Pid\s+(?P<pid>\d+)\s+on\s+(?P<hostname>\S+)\s+"
            r"device\s+(?P<device>\d+)\s+\[(?P<device_hex>\S+)\]\s+(?P<device_info>.+)"
        )
        result_header_pattern = re.compile(
            r"#\s+size\s+count\s+type\s+redop\s+root\s+time\s+algbw\s+busbw\s+#wrong\s+time\s+algbw\s+busbw\s+#wrong"
        )
        result_header_ex_pattern = re.compile(
            r"#\s+size\s+count\s+type\s+redop\s+root\s+timeAvg\s+algbwAvg\s+busbwAvg\s+timeMax\s+algbwMax\s+busbwMax\s+#wrong\s+timeAvg\s+algbwAvg\s+busbwAvg\s+timeMax\s+algbwMax\s+busbwMax\s+#wrong"
        )
        result_values_pattern = re.compile(
            r"\s+(?P<size>\d+)\s+(?P<count>\d+)\s+(?P<type>[a-zA-Z0-9]+)\s+(?P<redop>[a-zA-Z]+)\s+(?P<root>-?\d+)\s+"
            r"(?P<oop_time>[^\s]+)\s+(?P<oop_algbw>[^\s]+)\s+(?P<oop_busbw>[^\s]+)\s+(?P<oop_wrong>[^\s]+)\s+"
            r"(?P<inp_time>[^\s]+)\s+(?P<inp_algbw>[^\s]+)\s+(?P<inp_busbw>[^\s]+)\s+(?P<inp_wrong>[^\s]+)"
            r"(?:\s+(?P<actual_iterations>\d+))?(?:\s+(?P<experiment_name>.+))?"
        )
        result_values_ex_pattern = re.compile(
            r"\s+(?P<size>\d+)\s+(?P<count>\d+)\s+(?P<type>[a-zA-Z0-9]+)\s+(?P<redop>[a-zA-Z]+)\s+(?P<root>-?\d+)\s+"
            r"(?P<oop_time>[^\s]+)\s+(?P<oop_algbw>[^\s]+)\s+(?P<oop_busbw>[^\s]+)\s+"
            r"(?P<oop_timeMax>[^\s]+)\s+(?P<oop_algbwMax>[^\s]+)\s+(?P<oop_busbwMax>[^\s]+)\s+(?P<oop_wrong>[^\s]+)\s+"
            r"(?P<inp_time>[^\s]+)\s+(?P<inp_algbw>[^\s]+)\s+(?P<inp_busbw>[^\s]+)\s*"
            r"(?P<inp_timeMax>[^\s]+)\s+(?P<inp_algbwMax>[^\s]+)\s+(?P<inp_busbwMax>[^\s]+)\s+(?P<inp_wrong>[^\s]+)"
            r"(?:\s+(?P<actual_iterations>\d+))?(?:\s+(?P<experiment_name>.+))?"
        )
        out_of_bounds_pattern = re.compile(r"# Out of bounds values\s+:\s+(?P<count>\d+)\s+(?P<okay>OK|FAILED)")
        avg_bus_bandwidth_pattern = re.compile(
            r"# Avg bus bandwidth\s+:\s+(?P<bandwidth>[^\s]+)(?:\s+(?P<okay>OK|FAILED))?"
        )

        arguments = next((match.groupdict() for line in all_lines if (match := arguments_pattern.match(line))), {})
        if not arguments:
            raise ValueError("Failed to parse nccl tests output: arguments not found")

        device_list = [match.groupdict() for line in all_lines if (match := device_info_pattern.match(line))]
        if not device_list:
            raise ValueError("Failed to parse nccl tests output: device list not found")

        if any((match := result_header_pattern.match(line)) for line in all_lines):
            result_values = [match.groupdict() for line in all_lines if (match := result_values_pattern.match(line))]
            if not result_values:
                raise ValueError("Failed to parse nccl tests output: result values not found")
        elif any((match := result_header_ex_pattern.match(line)) for line in all_lines):
            result_values = [match.groupdict() for line in all_lines if (match := result_values_ex_pattern.match(line))]
            if not result_values:
                raise ValueError("Failed to parse nccl tests output: result values (extended) not found")
        else:
            raise ValueError("Failed to parse nccl tests output: result header not found")

        oob_values = next((match.groupdict() for line in all_lines if (match := out_of_bounds_pattern.match(line))), {})
        if not oob_values:
            raise ValueError("Failed to parse nccl tests output: out of bounds values not found")

        avg_bus_bandwidth = next(
            (match.groupdict() for line in all_lines if (match := avg_bus_bandwidth_pattern.match(line))), {}
        )
        if not avg_bus_bandwidth:
            raise ValueError("Failed to parse nccl tests output: average bus bandwidth not found")

        config = NcclTestConfig.model_validate(
            {**arguments, "devices": [NcclTestDevice.model_validate(d) for d in device_list]}
        )

        results = []
        for r in result_values:
            row_dict = {
                "size": r["size"],
                "count": r["count"],
                "type": r["type"],
                "redop": r["redop"],
                "root": r["root"],
                "out_of_place": NcclTestBenchmark.model_validate(
                    {
                        "time": r["oop_time"],
                        "alg_bw": r["oop_algbw"],
                        "bus_bw": r["oop_busbw"],
                        "time_max": r.get("oop_timeMax", None),
                        "alg_bw_max": r.get("oop_algbwMax", None),
                        "bus_bw_max": r.get("oop_busbwMax", None),
                        "nwrong": r["oop_wrong"].strip(),
                    }
                ),
                "in_place": NcclTestBenchmark.model_validate(
                    {
                        "time": r["inp_time"],
                        "alg_bw": r["inp_algbw"],
                        "bus_bw": r["inp_busbw"],
                        "time_max": r.get("inp_timeMax", None),
                        "alg_bw_max": r.get("inp_algbwMax", None),
                        "bus_bw_max": r.get("inp_busbwMax", None),
                        "nwrong": r["inp_wrong"].strip(),
                    }
                ),
                "actual_iterations": r.get("actual_iterations", None),
                "experiment_name": r.get("experiment_name", None),
            }
            benchmark_row = NcclTestBenchmarkRow.model_validate(row_dict)
            results.append(benchmark_row)

        test_result = cls.model_validate(
            {
                "config": config,
                "results": results,
                "out_of_bounds": NcclTestOutOfBound.model_validate(oob_values),
                "average_bus_bandwidth": NcclTestAvgBusBw.model_validate(avg_bus_bandwidth),
            }
        )

        return test_result
