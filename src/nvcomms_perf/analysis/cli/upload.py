import logging
from typing import Annotated, Optional
import typer
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    TaskID,
    TransferSpeedColumn,
    MofNCompleteColumn,
    DownloadColumn,
)
from rich.console import Console
import os
from concurrent.futures import ThreadPoolExecutor, as_completed, CancelledError

from ..models.perf_record import PerfRecord
from ..storage.sql_db import SQLDBClient, SQLDBOptions
from ..storage.s3 import S3Client, S3Options, S3FileLocation
from .common import check_duplicate_files, parse_input_files, InputFile, InputFileType, ParseResult

logger = logging.getLogger(__name__)
console = Console()


def _check_files_existence_in_db(
    input_files: list[InputFile], sqldb: SQLDBClient, show_progress: bool = False
) -> list[InputFile]:
    """Check if files already exist in the database or S3"""
    deduplicated_input_files = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        disable=not show_progress,
    ) as progress:
        if show_progress:
            console.print("[cyan]Check for file existence in database[/cyan]")
        parse_tasks = {
            input_file: progress.add_task(f"[cyan]Checking[/cyan] [dim]{input_file.path.name}[/dim]", total=1)
            for input_file in input_files
        }

        for input_file, task_id in parse_tasks.items():
            exists = sqldb.record_exists(input_file.hash)
            if exists:
                progress.update(
                    task_id,
                    completed=1,
                    description=f"[yellow]:warning:[/yellow] [dim]{input_file.path.name}[/dim] [yellow](exists)[/yellow]",
                )
                if not show_progress:
                    console.print(
                        f"[yellow]:warning:[/yellow] [dim]{input_file.path.name}[/dim] already exists in the database"
                    )
            else:
                deduplicated_input_files.append(input_file)
                progress.update(task_id, completed=1, description=f"[green]✓[/green] [dim]{input_file.path.name}[/dim]")

    return deduplicated_input_files


def _upload_single_file_to_s3(
    input_file: InputFile,
    s3_key: str,
    s3_client: S3Client,
    progress: Progress,
    task_id: TaskID,
) -> S3FileLocation:
    """Upload a single file to S3 with progress tracking.

    Args:
        input_file: The input file
        s3_key: The S3 key
        s3_client: The S3 client (thread-safe)
        progress: Optional progress bar
        task_id: Optional task ID for progress tracking

    Returns:
        S3FileLocation

    Raises:
        Exception if upload fails
    """

    def _progress_callback(bytes_transferred: int, total_bytes: int) -> None:
        progress.update(task_id, completed=bytes_transferred)
        # Update description when upload completes
        if bytes_transferred == total_bytes:
            progress.update(task_id, description=f"[green]✓[/green] Uploaded [dim]{input_file.path.name}[/dim]")

    try:
        return s3_client.upload_file(
            local_path=input_file.path, s3_key=s3_key, callback=_progress_callback if progress else None
        )
    except Exception:
        progress.update(task_id, description=f"[red]✗[/red] Failed [dim]{input_file.path.name}[/dim]")
        raise


def _upload_files_to_s3(
    parsed_files: dict[InputFile, ParseResult],
    s3_client: S3Client,
    max_workers: Optional[int] = None,
    show_progress: bool = False,
    continue_on_error: bool = False,
) -> tuple[dict[InputFile, S3FileLocation], dict[InputFile, Exception]]:
    """Upload multiple files to S3 using thread pool.

    Args:
        parsed_files: List of ParsedFile objects to upload
        s3_client: The S3 client (thread-safe)
        max_workers: Maximum number of upload threads
        show_progress: Whether to show progress bars
        continue_on_error: Whether to continue if a file fails

    Returns:
        Tuple of (successful_uploads, failed_uploads) where:
        - successful_uploads: maps InputFile to S3FileLocation
        - failed_uploads: maps InputFile to Exception
    """
    if not parsed_files:
        return {}, {}

    successful_uploads = {}
    failed_uploads = {}

    if max_workers is None:
        max_workers = os.cpu_count() or 4
    max_workers = min(max_workers, len(parsed_files), 10)  # Cap at 10 for S3 uploads

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(compact=True, elapsed_when_finished=True),
        console=console,
        disable=not show_progress,
    ) as progress:
        upload_tasks = {}
        if show_progress:
            console.print(f"[cyan]Upload {len(parsed_files)} files to S3[/cyan]")
        upload_tasks = {
            input_file: progress.add_task(
                f"[cyan]Uploading [dim]{input_file.path.name}[/dim]...", total=input_file.size
            )
            for input_file in parsed_files.keys()
        }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_input = {}
            for input_file, parse_result in parsed_files.items():
                s3_key = f"{str(parse_result.sweep_id)}/{input_file.path.name}"
                task_id = upload_tasks.get(input_file)
                if task_id is None:
                    raise RuntimeError(f"Internal error: input file {input_file} not found in upload_tasks")
                future = executor.submit(_upload_single_file_to_s3, input_file, s3_key, s3_client, progress, task_id)
                future_to_input[future] = input_file

            for future in as_completed(future_to_input):
                input_file = future_to_input[future]
                parse_result = parsed_files.get(input_file, None)
                if parse_result is None:
                    raise RuntimeError(f"Internal error: input file {input_file} not found in parsed_files")

                try:
                    location = future.result()
                    successful_uploads[input_file] = location
                except CancelledError as e:
                    progress.update(
                        upload_tasks[input_file],
                        description=f"[yellow]✗ Cancelled[/yellow] [dim]{input_file.path.name}[/dim]",
                    )
                    failed_uploads[input_file] = e
                except Exception as e:
                    failed_uploads[input_file] = e
                    if not continue_on_error:
                        for remaining_future, remaining_input in future_to_input.items():
                            if not remaining_future.done():
                                remaining_future.cancel()

    if not show_progress:
        for input_file, e in failed_uploads.items():
            console.print(f"[red]✗ Failed to upload [dim]{input_file.path.name}[/dim][/red]: {e}")

    return successful_uploads, failed_uploads


def _upload_records_to_db(records: list[PerfRecord], sqldb: SQLDBClient, show_progress: bool = False) -> int:
    """Upload records to database with progress tracking."""

    success_cnt = 0

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
            console.print("[cyan]Upload records to database[/cyan]")
        task_id = progress.add_task("[cyan]Uploading...", total=len(records))

        def _progress_callback(records_uploaded: int, records_total: int) -> None:
            nonlocal success_cnt
            success_cnt = records_uploaded
            progress.update(task_id, completed=records_uploaded)
            if records_uploaded == records_total:
                progress.update(task_id, description="[green]✓[/green] Uploaded")

        try:
            sqldb.upload_records(records, batch_size=100, callback=_progress_callback)
        except Exception as e:
            progress.update(task_id, description="[red]✗ Failed[/red]")
            if not show_progress:
                console.print(f"[red]✗ Database upload failed after {success_cnt} records: {e}[/red]")

    return success_cnt


def upload_test_results(
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
    db_url: Annotated[
        Optional[str],
        typer.Option(
            "--db-url",
            envvar="NVCOMMS_PERF_DB_URL",
            help="Database URL, following the format of the SQLAlchemy database URL: e.g. postgresql+psycopg://user:pass@host:port/db",
            rich_help_panel="SQL Database",
            show_default=False,
        ),
    ] = None,
    db_schema: Annotated[
        Optional[str],
        typer.Option(
            "--db-schema",
            envvar="NVCOMMS_PERF_DB_SCHEMA",
            help="Database schema to use, only used for PostgreSQL.",
            rich_help_panel="SQL Database",
            show_default=True,
        ),
    ] = None,
    s3_endpoint: Annotated[
        Optional[str],
        typer.Option(
            "--s3-endpoint",
            envvar="NVCOMMS_PERF_S3_ENDPOINT",
            help="S3 endpoint URL",
            rich_help_panel="S3",
            show_default=False,
        ),
    ] = None,
    s3_region: Annotated[
        Optional[str],
        typer.Option(
            "--s3-region",
            envvar="NVCOMMS_PERF_S3_REGION",
            help="S3 region",
            rich_help_panel="S3",
            show_default=False,
        ),
    ] = None,
    s3_bucket: Annotated[
        Optional[str],
        typer.Option(
            "--s3-bucket",
            envvar="NVCOMMS_PERF_S3_BUCKET",
            help="S3 bucket name",
            rich_help_panel="S3",
            show_default=False,
        ),
    ] = None,
    s3_access_key: Annotated[
        Optional[str],
        typer.Option(
            "--s3-access-key",
            envvar="NVCOMMS_PERF_S3_ACCESS_KEY",
            help=(
                """S3 access key id.

                when both option and envvar are omitted, AWS_ACCESS_KEY_ID is used"""
            ),
            rich_help_panel="S3",
            show_default=False,
        ),
    ] = None,
    s3_secret_key: Annotated[
        Optional[str],
        typer.Option(
            "--s3-secret-key",
            envvar="NVCOMMS_PERF_S3_SECRET_KEY",
            help=(
                """S3 secret access key.

                when both option and envvar are omitted, AWS_SECRET_ACCESS_KEY is used"""
            ),
            rich_help_panel="S3",
            show_default=False,
        ),
    ] = None,
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
    """
    Upload test results to a database (PostgreSQL, MySQL, SQLite) and/or S3.
    """

    sqldb = None
    s3 = None

    if db_url:
        try:
            sqldb_options = SQLDBOptions(db_url=db_url, db_schema=db_schema)
            sqldb = SQLDBClient(sqldb_options, create_tables=True)
        except Exception as e:
            logger.error(f"Error initializing SQL database: {e}")
            raise typer.Exit(1)

    if s3_bucket:
        try:
            s3_options = S3Options(
                bucket=s3_bucket,
                region=s3_region,
                endpoint=s3_endpoint,
                access_key=s3_access_key,
                secret_key=s3_secret_key,
            )
            s3 = S3Client(s3_options)
        except Exception as e:
            logger.error(f"Error initializing S3 client: {e}")
            raise typer.Exit(1)

    if not sqldb and not s3:
        logger.error("Either --db-url or --s3-bucket must be provided")
        raise typer.Exit(1)

    inputs = check_duplicate_files(inputs, show_progress)

    if sqldb:
        inputs = _check_files_existence_in_db(inputs, sqldb, show_progress)

    if not inputs:
        console.print("[yellow]No new files to process[/yellow]")
        raise typer.Exit(0)

    successful_parses, failed_parses = parse_input_files(inputs, max_workers, show_progress, continue_on_error)
    if not continue_on_error and failed_parses:
        console.print("[bold red]Stopped due to errors[/bold red]")
        raise typer.Exit(1)

    if not successful_parses:
        console.print("[bold red]No files have been parsed successfully[/bold red]")
        raise typer.Exit(1)

    input_files_to_use = list(successful_parses.keys())

    successful_uploads = {}
    if s3:
        successful_uploads, failed_uploads = _upload_files_to_s3(
            successful_parses,
            s3,
            max_workers=max_workers,
            show_progress=show_progress,
            continue_on_error=continue_on_error,
        )
        if not continue_on_error and failed_uploads:
            raise typer.Exit(1)

        if not successful_uploads:
            console.print("[yellow]No files have been uploaded successfully[/yellow]")
            raise typer.Exit(1)

        input_files_to_use = list(successful_uploads.keys())

    all_records = []

    if s3:
        # When S3 is used, iterate over successful uploads
        for input_file, s3_location in successful_uploads.items():
            parse_result = successful_parses.get(input_file, None)
            if parse_result is None:
                raise RuntimeError(f"Internal error: input file {input_file} not found in successful_parses")

            records_to_add = parse_result.records
            s3_path = f"s3://{s3_location.bucket}/{s3_location.key}"
            # Update raw_data_path for all records from this file
            for record in records_to_add:
                record.raw_data_path = s3_path

            all_records.extend(records_to_add)
    else:
        # When S3 is not used, iterate over successful parses
        for input_file in input_files_to_use:
            parse_result = successful_parses.get(input_file, None)
            if parse_result is None:
                raise RuntimeError(f"Internal error: input file {input_file} not found in successful_parses")

            all_records.extend(parse_result.records)

    if not all_records:
        console.print("[yellow]Didn't find any records to upload to database[/yellow]")
        raise typer.Exit(0)

    if sqldb:
        total_records_uploaded = _upload_records_to_db(all_records, sqldb, show_progress)
    else:
        total_records_uploaded = 0

    console.print("\n" + "=" * 50)
    console.print("[bold]UPLOAD SUMMARY[/bold]")
    console.print("=" * 50)
    console.print(f"Database: [cyan]{sqldb}[/cyan]")
    console.print(f"Total files processed: [cyan]{len(successful_parses)}[/cyan]")
    console.print(f"Total files uploaded to S3: [cyan]{len(successful_uploads)}[/cyan]")
    console.print(f"Total records to upload to database: [cyan]{total_records_uploaded}[/cyan]")

    console.print("\n[bold]Processed Files:[/bold]")
    for input_file in input_files_to_use:
        console.print(f"\n  File: [cyan]{input_file.path.name}[/cyan]")
        console.print(f"  Sweep ID: [dim]{successful_parses[input_file].sweep_id}[/dim]")
        console.print(f"  Records: [green]{len(successful_parses[input_file].records)}[/green]")

        if s3:
            s3_location = successful_uploads.get(input_file, None)
            if s3_location:
                console.print(f"  S3 Path: [blue]s3://{s3_location.bucket}/{s3_location.key}[/blue]")

    if sqldb:
        sqldb.close()
