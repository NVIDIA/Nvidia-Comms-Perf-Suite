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

from enum import Enum
from typing import List
from pathlib import Path
import os
import typer
from rich.console import Console
import hashlib

console = Console()


class InputFileType(str, Enum):
    """Enum for input file types for report generation"""

    CSV = "csv"
    TARBALL = "tarball"


class InputFile:
    """Represents an input file for report generation."""

    def __init__(self, value: str):
        """Parse input file specification.

        Args:
            value: Either a path to a file or format "<type>:<path>"
        """
        if ":" in value:
            type_str, path = [x.strip() for x in value.split(":", 1)]
            try:
                self.type = InputFileType(type_str)
            except ValueError:
                raise typer.BadParameter(f"Unknown input file type: {type_str}")
        else:
            # Infer type from extension
            path = value
            ext = Path(path).suffix.lower()
            if ext == ".csv":
                self.type = InputFileType.CSV
            elif ext in [".tar", ".tar.gz", ".tgz", ".tar.bz2"]:
                self.type = InputFileType.TARBALL
            else:
                raise typer.BadParameter(
                    f"Cannot infer input file type from extension '{ext}'. "
                    f"Supported file types: CSV (.csv) and tarballs (.tar, .tar.gz, .tgz, .tar.bz2). "
                    f"Please use <type>:<path> format for explicit type specification."
                )

        self.path = Path(path)

        # Validate file
        if not self.path.exists():
            raise typer.BadParameter(f"Input file not found: {self.path}")

        if not self.path.is_file():
            raise typer.BadParameter(f"Input path is not a file: {self.path}")

        if not os.access(self.path, os.R_OK):
            raise typer.BadParameter(f"Input file is not readable: {self.path}")

        self.size = self.path.stat().st_size

        # Compute file hash for duplicate detection
        try:
            self.hash = self._compute_file_hash()
        except Exception as e:
            raise typer.BadParameter(f"Failed to hash input file: {e}")

    def _compute_file_hash(self) -> str:
        """Compute SHA256 hash of the file."""
        sha256_hash = hashlib.sha256()
        with open(self.path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def __repr__(self):
        return f"<InputFile: type={self.type}, path={self.path}>"

    def __str__(self):
        return f"{self.type}:{self.path}"

    def __hash__(self):
        """Return hash based on file content hash."""
        return hash(self.hash)

    def __eq__(self, other):
        """Two InputFiles are equal if they have the same content hash."""
        if not isinstance(other, InputFile):
            return False
        return self.hash == other.hash


def validate_input_files(input_files: List[InputFile]) -> List[InputFile]:
    """Validate input files and check for duplicates.

    Args:
        input_files: List of input files to validate

    Returns:
        List of validated input files
    """
    if not input_files:
        raise typer.BadParameter("At least one input file is required")

    # Check for duplicate paths
    seen_paths = set()
    for input_file in input_files:
        if input_file.path in seen_paths:
            console.print(f"[yellow]Warning: Duplicate input file: {input_file.path}[/yellow]")
        seen_paths.add(input_file.path)

    return input_files


__all__ = ['InputFileType', 'InputFile', 'validate_input_files', 'console']
