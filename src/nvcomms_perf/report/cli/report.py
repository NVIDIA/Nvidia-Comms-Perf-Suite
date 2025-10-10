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
from typing import Annotated, Optional
import typer
import pandas as pd
from pathlib import Path
import logging
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

# Use report module's InputFile for function signature
from .common import InputFile, InputFileType, console
# Import parsing functions from analysis module
from ...analysis.cli.common import parse_input_files as analysis_parse_input_files
from ...analysis.cli.common import InputFileType as AnalysisInputFileType

logger = logging.getLogger(__name__)


class ReportType(str, Enum):
    """Enum for report types"""

    PDF = "pdf"


class PDFType(str, Enum):
    """Enum for PDF report types"""

    BASIC_REPORT = "basic_report"
    COMPREHENSIVE_REPORT = "comprehensive_report"


class OutputFile:
    """Represents an output file for report generation."""

    def __init__(self, value: str):
        """Parse output file specification.

        Args:
            value: Either a path or format "<type>:<path>"
        """
        if ":" in value:
            type_str, path = [x.strip() for x in value.split(":", 1)]
            try:
                self.type = ReportType(type_str)
            except ValueError:
                raise typer.BadParameter(f"Unknown report type: {type_str}")
        else:
            # Infer type from extension
            path = value
            ext = Path(path).suffix.lower()
            if ext == ".pdf":
                self.type = ReportType.PDF
            else:
                raise typer.BadParameter(
                    f"Cannot infer report type from extension '{ext}'. "
                    f"Please use <type>:<path> format."
                )

        self.path = Path(path)

    def __repr__(self):
        return f"<OutputFile: type={self.type}, path={self.path}>"


def _get_default_config_path(config_type: str) -> Path:
    """Get the default configuration file path.

    Args:
        config_type: Either 'plot' or 'style'

    Returns:
        Path to the default configuration file
    """
    import importlib.resources

    if config_type == 'plot':
        config_file = 'config/nccl/plot/plot_config.toml'
    elif config_type == 'style':
        config_file = 'config/nccl/style/style_config.toml'
    else:
        raise ValueError(f"Unknown config type: {config_type}")

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
    module_path = Path(__file__).parent.parent.parent
    config_path = module_path / config_file.replace('/', Path('/').as_posix())
    if config_path.exists():
        return config_path

    # Try from project root
    project_root = Path(__file__).parent.parent.parent.parent.parent
    config_path = project_root / 'src' / 'nvcomms_perf' / 'report' / config_file.replace('/', Path('/').as_posix())
    if config_path.exists():
        return config_path

    raise FileNotFoundError(f"Could not find default {config_type} configuration file")


def _convert_records_to_dataframe(records):
    """Convert parsed performance records to a pandas DataFrame.

    This function:
    1. Converts records to dictionaries using model_dump()
    2. Explodes the test_result column to handle multiple results per record
    3. Normalizes nested JSON structures (fully flatten)
    4. Truncates long strings/lists/dicts to 1024 characters
    5. Handles column renaming for compatibility (e.g., 'tool' -> 'tool_binary')

    Args:
        records: List of performance record objects with model_dump() method

    Returns:
        DataFrame with flattened and cleaned data
    """
    # Convert records to DataFrame
    df = pd.DataFrame([record.model_dump() for record in records]).explode("test_result", ignore_index=True)

    # Fully flatten nested JSON structures
    df = pd.json_normalize(df.to_dict(orient="records"), sep=".")

    # Handle column renaming for compatibility
    if 'tool' in df.columns and 'tool_binary' not in df.columns:
        logger.info("Renaming column 'tool' to 'tool_binary' for compatibility")
        df = df.rename(columns={'tool': 'tool_binary'})

    # Clean up tool_binary column if it contains full paths
    if 'tool_binary' in df.columns:
        df['tool_binary'] = df['tool_binary'].apply(lambda x: Path(x).name if isinstance(x, str) else x)

    # Truncate long data
    def truncate_data(x):
        """Truncate strings, dicts, and lists to 1024 characters"""
        if isinstance(x, (str, dict, list)):
            str_repr = str(x)
            return str_repr[:1024] if len(str_repr) > 1024 else str_repr
        return x

    df = df.map(truncate_data)

    return df


def generate_report(
    inputs: Annotated[
        list[InputFile],
        typer.Option(
            ...,
            "--input",
            "-i",
            parser=InputFile,
            show_default=False,
            help=(
                """Input file(s) to generate report from.

                Supported input types:
                csv, tarball

                Format: <type>:<path> OR <path>
                Example:
                - csv:/path/to/results.csv OR /path/to/results.csv
                - tarball:/path/to/results.tar.gz

                Multiple input files can be specified by using --input multiple times.
                Each input file will be processed and combined into the final report.

                For tarballs, all supported file types within the archive will be processed.
                """
            ),
        ),
    ],
    output: Annotated[
        OutputFile,
        typer.Option(
            ...,
            "--output",
            "-o",
            parser=OutputFile,
            show_default=False,
            help=(
                """Output file path for the generated report.

                Supported report types:
                pdf

                Format: <type>:<path> OR <path>
                Example: pdf:/tmp/report.pdf OR /tmp/report.pdf

                If type is not specified, it will be inferred from the file extension.
                """
            ),
        ),
    ],
    pdf_type: Annotated[
        PDFType,
        typer.Option(
            "--pdf-type",
            "-t",
            help="Type of PDF report to generate",
            show_default=True,
        ),
    ] = PDFType.BASIC_REPORT,
    plot_config: Annotated[
        Optional[Path],
        typer.Option(
            "--plot-config",
            "-c",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help=(
                """Path to custom plot configuration TOML file.

                If not provided, uses the default configuration from:
                src/nvcomms_perf/report/config/nccl/plot/plot_config.toml
                """
            ),
            show_default=False,
        ),
    ] = None,
    style_config: Annotated[
        Optional[Path],
        typer.Option(
            "--style-config",
            "-s",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help=(
                """Path to custom style configuration TOML file.

                If not provided, uses the default configuration from:
                src/nvcomms_perf/report/config/nccl/style/style_config.toml
                """
            ),
            show_default=False,
        ),
    ] = None,
    show_progress: Annotated[
        bool,
        typer.Option(
            "--show-progress/--hide-progress",
            "-p/-P",
            help="Show progress bar during report generation",
            show_default=True,
        ),
    ] = True,
    continue_on_error: Annotated[
        bool,
        typer.Option(
            "--continue-on-error/--fail-on-error",
            help="Continue processing other files if one fails to parse",
            show_default=True,
        ),
    ] = True,
    include_empty_pages: Annotated[
        bool,
        typer.Option(
            "--include-empty-pages/--no-empty-pages",
            help="Include empty lined pages for notes after charts",
            show_default=True,
        ),
    ] = True,
    include_data_tables: Annotated[
        bool,
        typer.Option(
            "--include-data-tables/--no-data-tables",
            help="Include data tables showing raw values used in charts",
            show_default=True,
        ),
    ] = True,
    include_test_commands: Annotated[
        bool,
        typer.Option(
            "--include-test-commands/--no-test-commands",
            help="Include pages showing test commands used to generate data",
            show_default=True,
        ),
    ] = True,
) -> None:
    """Generate performance reports from test result files.

    This command can process CSV files and tarballs containing test results.
    All inputs are combined into a single report.
    """

    # Separate CSV files from other file types
    csv_files = []
    other_files = []

    for input_file in inputs:
        if input_file.type == InputFileType.CSV:
            csv_files.append(input_file)
        else:
            # Convert to analysis module's InputFile format
            try:
                # Map our file types to analysis module's file types
                if input_file.type.value == "tarball":
                    analysis_type = AnalysisInputFileType.TARBALL
                else:
                    console.print(f"[red]Unsupported file type for parsing: {input_file.type}[/red]")
                    continue

                # Create analysis InputFile manually
                analysis_input = type('obj', (object,), {
                    'type': analysis_type,
                    'path': input_file.path,
                    'size': input_file.size
                })()
                other_files.append(analysis_input)
            except Exception as e:
                console.print(f"[red]Error converting input file {input_file.path}: {e}[/red]")
                raise typer.Exit(1)

    # Parse non-CSV files if any
    parsed_data = {}
    if other_files:
        successful_parses, failed_parses = analysis_parse_input_files(
            other_files,
            max_workers=None,  # Use default
            show_progress=show_progress,
            continue_on_error=continue_on_error
        )
        parsed_data = successful_parses

        if not successful_parses and not csv_files:
            console.print("[bold red]No files successfully parsed[/bold red]")
            raise typer.Exit(1)

    # Validate configuration files (additional validation for TOML extension)
    if plot_config and not plot_config.suffix.lower() == ".toml":
        raise typer.BadParameter(f"Plot configuration file must be a TOML file: {plot_config}")
    if style_config and not style_config.suffix.lower() == ".toml":
        raise typer.BadParameter(f"Style configuration file must be a TOML file: {style_config}")

    # Import modules before creating Progress to avoid interference
    from ..models import ChartConfigLoader, StyleConfigLoader
    from ..adapters.nccl_adapter import NCCLPerformanceAdapter
    from ..generators.pdf import generate_pdf_report

    # Flush console to clear any buffered output before progress display
    console.clear_live()

    # Track progress completion to prevent duplicate messages
    progress_completed = {
        'config': False,
        'parsing': False,
        'convert': False,
        'combine': False,
        'pdf': False
    }

    logger.debug("Starting report generation with progress tracking")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        disable=not show_progress,
    ) as progress:
        # Load configuration first to determine required columns
        config_task_id = progress.add_task("[cyan]Loading configuration...[/cyan]", total=1)
        logger.debug(f"Created configuration task with ID: {config_task_id}")

        try:
            # Load plot configuration
            if plot_config:
                logger.info(f"Loading custom plot configuration from: {plot_config}")
                plot_config_obj = ChartConfigLoader.load_from_toml(str(plot_config))
            else:
                default_plot_path = _get_default_config_path('plot')
                logger.info(f"Loading default plot configuration from: {default_plot_path}")
                plot_config_obj = ChartConfigLoader.load_from_toml(str(default_plot_path))

            # Load style configuration
            if style_config:
                logger.info(f"Loading custom style configuration from: {style_config}")
                style_config_obj = StyleConfigLoader.load_from_toml(str(style_config))
            else:
                default_style_path = _get_default_config_path('style')
                logger.info(f"Loading default style configuration from: {default_style_path}")
                style_config_obj = StyleConfigLoader.load_from_toml(str(default_style_path))

            # Create NCCL adapter with both configurations
            adapter = NCCLPerformanceAdapter(plot_config=plot_config_obj, style_config=style_config_obj)

            # Update progress only once for configuration loading
            if not progress_completed['config']:
                logger.debug("Updating configuration progress completion")
                progress.update(config_task_id, completed=1, description="[green]✓[/green] Loaded configuration")
                progress_completed['config'] = True
            else:
                logger.debug("Configuration progress already completed, skipping update")

        except Exception as e:
            console.print(f"[red]Error loading configuration: {e}[/red]")
            raise typer.Exit(1)

        # Convert parsed files to DataFrames
        convert_task_id = progress.add_task("[cyan]Converting data to DataFrames...[/cyan]", total=1)
        logger.debug(f"Created conversion task with ID: {convert_task_id}")

        dataframes = []

        # First, handle CSV files with optimized loading
        for csv_file in csv_files:
            try:
                df = adapter.load_csv_with_required_columns(csv_file.path, pdf_type.value)
                dataframes.append(df)
                logger.debug(f"Loaded CSV file directly: {csv_file.path}")
            except Exception as e:
                logger.warning(f"Failed to load CSV with optimized columns, falling back: {e}")
                df = pd.read_csv(str(csv_file.path))
                dataframes.append(df)

        # Then, handle parsed non-CSV files
        for parsed_file in parsed_data.values():
            if hasattr(parsed_file, 'records'):
                all_records = parsed_file.records
                if all_records:
                    # Convert records to DataFrame
                    df = _convert_records_to_dataframe(all_records)
                    dataframes.append(df)
                    logger.debug("Converted parsed data to DataFrame")
                else:
                    logger.warning("No records found in parsed file")

        if not dataframes:
            console.print("[bold red]No data could be loaded from input files[/bold red]")
            raise typer.Exit(1)

        # Update progress for conversion
        if not progress_completed['convert']:
            logger.debug("Updating conversion progress completion")
            progress.update(convert_task_id, completed=1, description=f"[green]✓[/green] Converted {len(dataframes)} files to DataFrames")
            progress_completed['convert'] = True
        else:
            logger.debug("Conversion progress already completed, skipping update")

        # Combine dataframes if multiple inputs
        if len(dataframes) > 1:
            combine_task_id = progress.add_task("[cyan]Combining data...[/cyan]", total=1)
            logger.debug(f"Created data combination task with ID: {combine_task_id}")
            combined_df = pd.concat(dataframes, ignore_index=True)
            if not progress_completed['combine']:
                logger.debug("Updating data combination progress completion")
                progress.update(combine_task_id, completed=1, description="[green]✓[/green] Combined data")
                progress_completed['combine'] = True
            else:
                logger.debug("Data combination progress already completed, skipping update")
        else:
            combined_df = dataframes[0]
            logger.debug("Single CSV file, skipping data combination")

        # Log information about loaded columns
        logger.info(f"Loaded DataFrame with {len(combined_df)} rows and {len(combined_df.columns)} columns")
        logger.info(f"Memory usage: {combined_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

        # Generate report based on type
        if output.type == ReportType.PDF:
            pdf_task_id = progress.add_task("[cyan]Generating PDF report...[/cyan]", total=1)
            logger.debug(f"Created PDF generation task with ID: {pdf_task_id}")

            try:
                # Generate the PDF report
                generate_pdf_report(
                    combined_df,
                    output.path,
                    pdf_type=pdf_type.value,
                    adapter=adapter,  # Pass the already-loaded adapter
                    include_empty_pages=include_empty_pages,
                    include_data_tables=include_data_tables,
                    include_test_commands=include_test_commands
                )

                # Update progress only once for PDF generation completion
                if not progress_completed['pdf']:
                    logger.debug("Updating PDF generation progress completion")
                    progress.update(pdf_task_id, completed=1, description="[green]✓[/green] Generated PDF report")
                    progress_completed['pdf'] = True
                else:
                    logger.debug("PDF generation progress already completed, skipping update")
            except ImportError as e:
                console.print(
                    f"[red]Import error during PDF generation: {e}[/red]"
                )
                raise typer.Exit(1)
            except Exception as e:
                console.print(f"[red]Error generating PDF report: {e}[/red]")
                raise typer.Exit(1)
        else:
            raise NotImplementedError(f"Report type {output.type} not implemented")

    logger.debug("Report generation completed successfully")
    console.print(
        f"[bold green]Successfully generated {output.type.value} report: {output.path}[/bold green]"
    )
