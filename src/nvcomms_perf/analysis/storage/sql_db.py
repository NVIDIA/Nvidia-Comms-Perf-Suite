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

from pydantic import BaseModel, ConfigDict
from typing import Optional, Any, Callable
from sqlmodel import SQLModel, create_engine, Session, select, text
from sqlalchemy.engine.url import make_url
from sqlalchemy import and_, exists
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from contextlib import contextmanager
import logging
from ..models.perf_record import PerfRecord

logger = logging.getLogger(__name__)


class SQLDBOptions(BaseModel):
    db_url: str
    db_schema: Optional[str] = None
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: float = 30.0
    pool_recycle: int = 3600  # Recycle connections after 1 hour
    echo: bool = False

    model_config = ConfigDict(frozen=True)


class SQLDBClient:
    def __init__(self, options: SQLDBOptions, create_tables: bool = False):
        """
        Initialize SQL database client with optimized settings.

        Args:
            options: Database configuration options
            create_tables: Whether to create tables on initialization (default: False)
        """
        self.db_url = make_url(options.db_url)
        self.schema = options.db_schema

        # Set schema on all tables in metadata (schema is global for this project)
        if self.schema and self.db_url.drivername.startswith("postgresql"):
            for table in SQLModel.metadata.tables.values():
                if not table.schema:
                    table.schema = self.schema

        # Configure connection pool based on database type
        pool_class = NullPool if self.db_url.drivername == "sqlite" else QueuePool

        engine_kwargs: dict[str, Any] = {
            "echo": options.echo,
            "pool_pre_ping": True,  # Verify connections before using
            "future": True,  # Use SQLAlchemy 2.0 style
        }

        # Add pooling options for non-SQLite databases
        if pool_class == QueuePool:
            engine_kwargs.update(
                {
                    "poolclass": pool_class,
                    "pool_size": options.pool_size,
                    "max_overflow": options.max_overflow,
                    "pool_timeout": options.pool_timeout,
                    "pool_recycle": options.pool_recycle,
                }
            )

        try:
            self.engine = create_engine(self.db_url, **engine_kwargs)

            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            if create_tables:
                self._create_tables()

        except OperationalError as e:
            logger.error(f"Failed to connect to database: {e}")
            raise ConnectionError(f"Database connection failed: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize database client: {e}")
            raise

    def __str__(self):
        return self.db_url.render_as_string(hide_password=True)

    def _create_tables(self):
        """Create database tables if they don't exist."""
        try:
            # Create schema first if needed (PostgreSQL)
            if self.schema and self.db_url.drivername.startswith("postgresql"):
                with self.engine.connect() as conn:
                    conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self.schema}"))
                    conn.commit()

            SQLModel.metadata.create_all(self.engine)
            logger.debug("Database tables created successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to create tables: {e}")
            raise

    def close(self):
        """Close the database connection pool."""
        self.engine.dispose()
        logger.debug("Database connection pool closed")

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and close connections."""
        self.close()
        return False  # Don't suppress exceptions

    @contextmanager
    def get_session(self):
        """Context manager for database sessions with automatic cleanup."""
        session = Session(self.engine)
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def upload_records(
        self, records: list[PerfRecord], batch_size: int = 1000, callback: Optional[Callable[[int, int], None]] = None
    ) -> int:
        """
        Upload records in batches for better performance.

        Args:
            records: List of PerfRecord objects to upload
            batch_size: Number of records to insert per batch
            callback: Optional callback function that receives (records_uploaded, total_records).
                     Called after each batch is successfully uploaded.

        Returns:
            Number of records uploaded
        """
        if not records:
            raise ValueError("No records to upload.")

        total_records = len(records)
        total_uploaded = 0
        total_batches = (total_records + batch_size - 1) // batch_size  # Ceiling division

        try:
            with self.get_session() as session:
                # Process in batches
                for i in range(0, total_records, batch_size):
                    batch = records[i : i + batch_size]
                    session.bulk_save_objects(batch)
                    session.commit()
                    total_uploaded += len(batch)

                    current_batch = i // batch_size + 1
                    logger.debug(f"Uploaded batch {current_batch}/{total_batches} ({len(batch)} records)")

                    # Call progress callback if provided
                    if callback:
                        callback(total_uploaded, total_records)

        except SQLAlchemyError as e:
            logger.error(f"Failed to upload records: {e}")
            raise

        logger.info(f"Successfully uploaded {total_uploaded} records")
        return total_uploaded

    def record_exists(self, file_hash: str) -> bool:
        """Check if a record with the given file hash exists."""
        with self.get_session() as session:
            stmt = select(exists().where(PerfRecord.raw_data_hash == file_hash))  # type: ignore
            return session.exec(stmt).one()

    def get_records(
        self,
        where_clause: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> list[PerfRecord]:
        """
        Get records with flexible filtering options.

        Args:
            where_clause: Raw SQL where clause (use with caution)
            filters: Safe dictionary-based filters
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of PerfRecord objects
        """
        with self.get_session() as session:
            statement = select(PerfRecord)

            # Apply filters
            if filters:
                conditions = []
                for field, value in filters.items():
                    if hasattr(PerfRecord, field):
                        conditions.append(getattr(PerfRecord, field) == value)
                if conditions:
                    statement = statement.where(and_(*conditions))

            # Apply raw where clause if provided
            if where_clause:
                statement = statement.where(text(where_clause))

            # Apply pagination
            if offset:
                statement = statement.offset(offset)
            if limit:
                statement = statement.limit(limit)

            result = session.exec(statement).all()
            return list(result)

    def get_records_with_params(
        self, where_clause: str, params: dict[str, Any], limit: Optional[int] = None
    ) -> list[PerfRecord]:
        """
        Get records using parameterized SQL for complex queries.

        Args:
            where_clause: SQL where clause with named parameters
            params: Parameter values
            limit: Maximum number of records to return

        Example:
            where_clause = "tool = :tool AND num_nodes > :min_nodes"
            params = {"tool": "nccl-tests", "min_nodes": 2}
        """
        with self.get_session() as session:
            statement = select(PerfRecord).where(text(where_clause).bindparams(**params))

            if limit:
                statement = statement.limit(limit)

            result = session.exec(statement).all()
            return list(result)

    def count_records(self, filters: Optional[dict[str, Any]] = None) -> int:
        """Get count of records matching filters."""
        from sqlalchemy import func

        with self.get_session() as session:
            statement = select(func.count()).select_from(PerfRecord)

            if filters:
                conditions = []
                for field, value in filters.items():
                    if hasattr(PerfRecord, field):
                        conditions.append(getattr(PerfRecord, field) == value)
                if conditions:
                    statement = statement.where(and_(*conditions))

            result = session.exec(statement).one()
            return result
