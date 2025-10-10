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

import polars as pl
import orjson
from typing import Optional
from ..models.perf_record import PerfRecord


def _unnest_all_with_prefix(df: pl.DataFrame, struct_cols: list[str]) -> pl.DataFrame:
    for col in struct_cols:
        original_columns = set(df.columns)
        df = df.unnest(col)
        new_columns = set(df.columns) - original_columns

        rename_map = {new_col: f"{col}.{new_col}" for new_col in new_columns}
        df = df.rename(rename_map)

    return df

def convert_perfrecords_to_flattened_dataframe(all_records: list[PerfRecord], fields_to_include: Optional[list[str]] = None) -> pl.DataFrame:
    if not all_records:
        return pl.DataFrame()

    if fields_to_include is not None:
        records = [record.model_dump(mode="json", include=set(fields_to_include)) for record in all_records]
    else:
        records = [record.model_dump(mode="json") for record in all_records]

    df = pl.json_normalize(records, max_level=1)

    def _truncate_list(x: list) -> list:
        if len(x) > 32:
            return x[:32] + [f"({len(x) - 32} more)"]
        return x

    if "test_metadata.devices" in df.columns:
        df = df.with_columns(
            pl.col("test_metadata.devices").map_elements(
                lambda x: orjson.dumps(_truncate_list(x.to_list())).decode() if x is not None else None,
                return_dtype=pl.String
            )
        )

    if "test_result" in df.columns:
        # Check if test_result column has any non-null values to explode
        if (df["test_result"].list.len() > 0).any():
            df = df.explode("test_result")
            df = _unnest_all_with_prefix(df, ["test_result"])

    # Group columns by field prefix and order by PerfRecord field order
    model_fields = list(PerfRecord.model_fields.keys())
    existing_cols = df.columns

    ordered_cols = []
    for field in model_fields:
        if fields_to_include is not None and field not in fields_to_include:
            continue
        # Add exact field match first
        if field in existing_cols:
            ordered_cols.append(field)
        # Add all columns with this field as prefix
        prefixed_cols = [col for col in existing_cols if col.startswith(f"{field}.")]
        ordered_cols.extend(prefixed_cols)

    if ordered_cols:
        df = df.select(ordered_cols)

    return df


