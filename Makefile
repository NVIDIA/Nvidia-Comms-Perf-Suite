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

# Python interpreter
PYTHON := python3

# UV executable
UV := uv

# Virtual environment directory
VENV := .venv

# Source directory
SRC_DIR := src

# Test directory
TEST_DIR := tests

# Default target
.DEFAULT_GOAL := help

# Help target
help:
	@echo "Available targets:"
	@echo "  setup      	- Create virtual environment and install dependencies"
	@echo "  build      	- Build the UV app"
	@echo "  test       	- Run pytest suite"
	@echo "  clean      	- Remove build artifacts and virtual environment"
	@echo "  run        	- Run the UV app"
	@echo "  generate-docs	- Generate schema docs for data models"

# Clean up build artifacts and virtual environment
clean:
	rm -rf $(VENV) build dist *.egg-info .pytest_cache .ruff_cache tests/dump_root tests/resources/**/parsed tests/resources/**/translated
	find . -type d -name __pycache__ -exec rm -rf {} +

# Create virtual environment and install dependencies
setup:
	$(UV) venv --seed $(VENV)
	$(UV) sync --extra dev

# Run pytest suite
test:
	$(UV) run pytest --basetemp=$(TEST_DIR)/dump_root $(TEST_DIR)/ -v

# Build wheel package
build:
	$(UV) build

# Generate schema markdown docs
generate-docs:
	$(UV) run nvcomms-perf --generate-schema-docs .

# Ensure these targets are always executed, even if files with the same name exist
.PHONY: help setup clean test
