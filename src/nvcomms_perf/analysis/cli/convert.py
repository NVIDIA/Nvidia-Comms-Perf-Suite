# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from enum import Enum
from typing import Annotated, Optional
import typer
from pathlib import Path
import logging
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.console import Console

from .common import check_duplicate_files, parse_input_files, InputFile, InputFileType
from ..utils.conversion import convert_perfrecords_to_flattened_dataframe

logger = logging.getLogger(__name__)
console = Console()


class OutputFileType(str, Enum):
    """Enum for file types to convert to"""

    CSV = "csv"


class OutputFile:
    def __init__(self, value: str):
        if ":" in value:
            type, path = [x.strip() for x in value.split(":", 1)]
            try:
                self.type = OutputFileType(type)
            except ValueError:
                raise typer.BadParameter(f"Unknown output file type: {type}")
        else:
            path = value
            ext = Path(path).suffix.lower()
            if ext == ".csv":
                self.type = OutputFileType.CSV
            else:
                raise typer.BadParameter(
                    f"Cannot infer output file type from extension '{ext}'. Please use <type>:<path>."
                )

        self.path = Path(path)

    def __repr__(self):
        return f"<OutputFile: type={self.type}, path={self.path}>"


def convert_test_results(
    inputs: Annotated[
        list[InputFile],
        typer.Option(
            ...,
            "--input",
            "-i",
            parser=InputFile,
            show_default=False,
            help=(
                f"""Input file(s) to convert.

                Supported input types:
                {", ".join([t.value for t in InputFileType])}

                Format: <type>:<path>
                Example: nccl-tests:/path/to/nccl-tests.log

                Multiple input files can be specified by using --input multiple times.
                Each input file will be processed and combined into the final output.
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
                f"""Output file path to write the converted results.

                Supported output types:
                {", ".join([t.value for t in OutputFileType])}

                Format: <type>:<path> OR <path>
                Example: csv:/tmp/results.csv , /tmp/results.csv

                If type is not specified, it will be inferred from the file extension.
                """
            ),
        ),
    ],
    max_workers: Annotated[
        Optional[int],
        typer.Option(
            "--max-workers",
            help="Maximum number of parallel workers for parsing. Default: MIN(number of CPU cores, number of input files)",
            show_default=False,
        ),
    ] = None,
    show_progress: Annotated[
        bool,
        typer.Option(
            "--show-progress/--hide-progress",
            "-p/-P",
            help="Show progress bar during parsing",
            show_default=True,
        ),
    ] = True,
    continue_on_error: Annotated[
        bool,
        typer.Option(
            "--continue-on-error/--stop-on-error",
            "-c/-C",
            help="Continue processing other files if one fails",
            show_default=True,
        ),
    ] = False,
) -> None:
    """Convert test results to a local file in the specified format."""
    inputs = check_duplicate_files(inputs, show_progress)

    successful_parses, failed_parses = parse_input_files(inputs, max_workers, show_progress, continue_on_error)
    if not continue_on_error and failed_parses:
        console.print("[bold red]Stopped due to errors[/bold red]")
        raise typer.Exit(1)

    if not successful_parses:
        console.print("[bold yellow]No files to process[/bold yellow]")
        raise typer.Exit(1)

    parsed_files = list(successful_parses.values())

    if output.type == OutputFileType.CSV:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            disable=not show_progress,
        ) as progress:
            if show_progress:
                console.print("[cyan]Convert to CSV[/cyan]")
            task_id = progress.add_task("[cyan]Processing records...[/cyan]", total=1)
            all_records = []
            for parsed_file in parsed_files:
                all_records.extend(parsed_file.records)

            df = convert_perfrecords_to_flattened_dataframe(all_records)

            progress.update(task_id, completed=1, description=f"[green]✓[/green] Processed {len(all_records)} records")

            task_id = progress.add_task(f"[cyan]Writing to[/cyan] {output.path.name}", total=1)

            df.write_csv(str(output.path))
            progress.update(task_id, completed=1, description=f"[green]✓[/green] Saved to {output.path.name}")

            if not show_progress:
                console.print(
                    f"Saved {len(all_records)} records from {len(parsed_files)} files to [bold green]{output.path.name}[/bold green]"
                )
    else:
        raise NotImplementedError(f"Output type {output.type} not implemented")
