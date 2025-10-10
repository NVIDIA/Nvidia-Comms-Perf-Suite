from enum import Enum
from typing import Annotated, Optional
import uuid
from pathlib import Path
import os
import typer
from concurrent.futures import ThreadPoolExecutor, as_completed, CancelledError
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TaskID,
)
from rich.console import Console
from pydantic import BaseModel, Field
from ..models.perf_record import PerfRecord

from ..utils.parser import parse_nccl_test, parse_tarball
from ..utils.misc import hash_file

console = Console()


class InputFileType(str, Enum):
    """Enum for file types to be converted"""

    NCCL_TESTS = "nccl-tests"
    TARBALL = "tarball"


class InputFile:
    def __init__(self, value: str):
        if ":" not in value:
            raise typer.BadParameter(f"Input file must be in the format <type>:<path>. Got: {value}")

        type, path = [x.strip() for x in value.split(":", 1)]
        try:
            self.type = InputFileType(type)
        except ValueError:
            raise typer.BadParameter(f"Unknown input file type: {type}")
        self.path = Path(path)

        if not self.path.exists():
            raise typer.BadParameter(f"Input file not found: {self.path}")

        if not self.path.is_file():
            raise typer.BadParameter(f"Input file is not a file: {self.path}")

        if not os.access(self.path, os.R_OK):
            raise typer.BadParameter(f"Input file is not readable: {self.path}")

        self.size = self.path.stat().st_size

        try:
            self.hash = hash_file(self.path)
        except Exception as e:
            raise typer.BadParameter(f"Failed to hash input file: {e}")

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


class ParseResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    sweep_id: uuid.UUID
    records: Annotated[list[PerfRecord], Field(min_length=1)]

    total_records: int
    failed_records: int = 0


def check_duplicate_files(input_files: list[InputFile], show_progress: bool = False) -> list[InputFile]:
    """Check for duplicate files and return a list of files with the same hash."""
    hash_to_inputs = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        disable=not show_progress,
    ) as progress:
        task_id = None
        if show_progress:
            console.print("[cyan]Check for duplicate files[/cyan]")
        task_id = progress.add_task("[cyan]Checking for duplicate files...[/cyan]", total=1)

        for input_file in input_files:
            hash = input_file.hash
            if hash not in hash_to_inputs:
                hash_to_inputs[hash] = []
            hash_to_inputs[hash].append(input_file)

        for _, inputs in hash_to_inputs.items():
            if len(inputs) > 1:
                console.print("[yellow]:warning: The following files are identical, will use the first one:[/yellow]")
                for input in inputs:
                    console.print(f"[dim]  {input.path}[/dim]")

        progress.update(task_id, completed=1, description="Duplicate check complete")

    return [inputs[0] for _, inputs in hash_to_inputs.items()]


def _parse_input_file(
    input_file: InputFile, progress: Progress, task_id: TaskID, continue_on_error: bool = False
) -> ParseResult:
    """Parse a single input file and return sweep_id and records."""
    sweep_id = uuid.uuid4()
    perf_records = []
    total_records = 0
    failed_records = 0

    def _progress_callback(records_uploaded: int, records_total: int, is_completed: bool) -> None:
        if records_uploaded == 0:
            progress.start_task(task_id)
            progress.update(
                task_id, total=records_total, description=f"[cyan]Parsing[/cyan] [dim]{input_file.path.name}[/dim]"
            )
        elif is_completed:
            if records_uploaded == records_total:
                progress.update(
                    task_id,
                    completed=records_total,
                    description=f"[green]✓ Parsed[/green] [dim]{input_file.path.name}[/dim]",
                )
            else:
                progress.update(
                    task_id,
                    completed=records_uploaded,
                    description=f"[yellow]:warning: Partially parsed[/yellow] [dim]{input_file.path.name}[/dim]",
                )
            progress.stop_task(task_id)
        else:
            progress.update(task_id, completed=records_uploaded)

    try:
        if input_file.type == InputFileType.NCCL_TESTS:
            perf_records = [parse_nccl_test(input_file.path, sweep_id, _progress_callback if progress else None)]
            total_records = 1
        elif input_file.type == InputFileType.TARBALL:
            perf_records, total_records, failed_records = parse_tarball(
                input_file.path, sweep_id, continue_on_error, _progress_callback if progress else None
            )
        else:
            raise ValueError(f"Input type {input_file.type} not supported to parse")
    except Exception:
        progress.update(task_id, description=f"[red]✗ Failed[/red] [dim]{input_file.path.name}[/dim]")
        raise

    return ParseResult(
        sweep_id=sweep_id, records=perf_records, total_records=total_records, failed_records=failed_records
    )


def parse_input_files(
    input_files: list[InputFile],
    max_workers: Optional[int] = None,
    show_progress: bool = False,
    continue_on_error: bool = False,
) -> tuple[dict[InputFile, ParseResult], dict[InputFile, Exception]]:
    """Parse a list of input files and return (successful, failed) dictionaries.

    Returns:
        Tuple of (successful_dict, failed_dict) where:
        - successful_dict: maps InputFile to ParseResult
        - failed_dict: maps InputFile to Exception
    """

    if not input_files:
        return {}, {}

    successful_parses = {}
    partial_parses = {}
    failed_parses = {}

    if max_workers is None:
        max_workers = os.cpu_count() or 4
    max_workers = min(max_workers, len(input_files))  # Don't use more workers than files

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(compact=True, elapsed_when_finished=True),
        console=console,
        disable=not show_progress,
    ) as progress:
        if show_progress:
            console.print(f"[cyan]Parse {len(input_files)} files[/cyan]")
        parse_tasks = {
            input_file: progress.add_task(f"[cyan]Extracting[/cyan] [dim]{input_file.path.name}[/dim]", start=False)
            for input_file in input_files
        }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_input = {
                executor.submit(_parse_input_file, input_file, progress, task_id, continue_on_error): input_file
                for input_file, task_id in parse_tasks.items()
            }

            for future in as_completed(future_to_input):
                input_file = future_to_input[future]
                try:
                    parse_result = future.result()
                    if parse_result.failed_records > 0:
                        partial_parses[input_file] = parse_result
                    else:
                        successful_parses[input_file] = parse_result
                except CancelledError as e:
                    progress.update(
                        parse_tasks[input_file],
                        description=f"[yellow]✗ Cancelled[/yellow] [dim]{input_file.path.name}[/dim]",
                    )
                    failed_parses[input_file] = e
                except Exception as e:
                    failed_parses[input_file] = e
                    if not continue_on_error:
                        for remaining_future, remaining_input in future_to_input.items():
                            if not remaining_future.done():
                                remaining_future.cancel()

    for input_file, e in failed_parses.items():
        if not isinstance(e, CancelledError):
            console.print(f"[red]✗ Failed to parse[/red] [dim]{input_file.path.name}[/dim]: {e}")

    return {**successful_parses, **partial_parses}, failed_parses
