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

from pydantic import BaseModel
from typing import Optional, Callable, Any
from pathlib import Path
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError, BotoCoreError
import logging
import threading
from datetime import datetime

logger = logging.getLogger(__name__)


class S3Options(BaseModel):
    bucket: str
    region: Optional[str] = None
    endpoint: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    # Performance options
    max_pool_connections: int = 50
    max_retries: int = 3
    multipart_threshold: int = 1024 * 1024 * 25  # 25MB
    multipart_chunksize: int = 1024 * 1024 * 25  # 25MB
    max_concurrency: int = 10
    use_threads: bool = True


class S3FileLocation(BaseModel):
    url: str
    bucket: str
    key: str
    region: str
    endpoint: str


class S3FileInfo(BaseModel):
    """Information about a file stored in S3."""

    key: str
    size: int
    last_modified: datetime
    etag: str
    content_type: Optional[str] = None
    metadata: Optional[dict[str, str]] = None

    @property
    def filename(self) -> str:
        """Get the filename from the key."""
        return self.key.split("/")[-1]

    @property
    def directory(self) -> str:
        """Get the directory path from the key."""
        parts = self.key.split("/")[:-1]
        return "/".join(parts) if parts else ""

    def __str__(self) -> str:
        """String representation of the file info."""
        return f"S3FileInfo(key={self.key}, size={self.size:,} bytes, modified={self.last_modified})"


class S3Client:
    def __init__(self, options: S3Options):
        """
        Initialize S3 client with optimized settings.

        Args:
            options: S3 configuration options
        """
        self.bucket = options.bucket
        self.options = options  # Store options for thread-local client creation
        self._thread_local = threading.local()  # Thread-local storage for boto3 clients

        # Configure transfer settings for optimal performance
        self.transfer_config = TransferConfig(
            multipart_threshold=options.multipart_threshold,
            multipart_chunksize=options.multipart_chunksize,
            max_concurrency=options.max_concurrency,
            use_threads=options.use_threads,
        )

        # Initialize and test connection for the main thread
        self._get_s3_client()
        self._verify_bucket_access()

    def _get_s3_client(self):
        """Get or create a thread-local boto3 S3 client."""
        # Check if this thread already has a client
        if not hasattr(self._thread_local, "s3_client"):
            # Create a new client for this thread
            config = Config(
                region_name=self.options.region,
                retries={
                    "max_attempts": self.options.max_retries,
                    "mode": "adaptive",  # Adaptive retry mode for better handling
                },
                max_pool_connections=self.options.max_pool_connections,
            )

            client_kwargs = {
                "service_name": "s3",
                "config": config,
            }

            if self.options.endpoint:
                client_kwargs["endpoint_url"] = self.options.endpoint
            if self.options.access_key:
                client_kwargs["aws_access_key_id"] = self.options.access_key
            if self.options.secret_key:
                client_kwargs["aws_secret_access_key"] = self.options.secret_key

            try:
                self._thread_local.s3_client = boto3.client(**client_kwargs)
                logger.debug(f"Created new S3 client for thread {threading.current_thread().name}")
            except (ClientError, BotoCoreError) as e:
                logger.error(f"Failed to initialize S3 client for thread: {e}")
                raise ConnectionError(f"S3 connection failed: {e}")

        return self._thread_local.s3_client

    @property
    def s3(self):
        """Thread-safe property to get the current thread's S3 client."""
        return self._get_s3_client()

    def _verify_bucket_access(self):
        """Verify that we can access the configured bucket."""
        try:
            self.s3.head_bucket(Bucket=self.bucket)
            logger.debug(f"Successfully verified access to bucket: {self.bucket}")
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code in ("404", "NoSuchBucket"):
                raise ValueError(f"Bucket '{self.bucket}' does not exist")
            elif error_code in ("403", "AccessDenied"):
                raise PermissionError(f"Access denied to bucket '{self.bucket}'")
            else:
                raise

    def upload_file(
        self,
        local_path: Path,
        s3_key: str,
        callback: Optional[Callable[[int, int], None]] = None,
        extra_args: Optional[dict] = None,
    ) -> S3FileLocation:
        """
        Upload a file to the configured S3 bucket with optimized settings.

        Args:
            local_path: Path to the local file to upload
            s3_key: S3 object key (path in the bucket)
            callback: Optional callback function that receives (bytes_transferred, total_bytes).
                     Thread safety is automatically handled for multi-threaded uploads.
            extra_args: Optional extra arguments for upload (e.g., ServerSideEncryption)

        Returns:
            Dictionary with upload details
        """
        if not local_path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")

        file_size = local_path.stat().st_size
        logger.debug(f"Uploading {local_path} ({file_size:,} bytes) to s3://{self.bucket}/{s3_key}")

        # Create thread-safe progress callback if provided
        upload_callback: Optional[Callable[[int], Any]] = None
        if callback is not None:
            bytes_transferred = 0
            lock = threading.Lock()

            def _progress_callback(bytes_amount: int) -> None:
                nonlocal bytes_transferred, file_size
                with lock:
                    bytes_transferred += bytes_amount
                callback(bytes_transferred, file_size)

            upload_callback = _progress_callback

        try:
            # Use upload_file with transfer config for automatic multipart handling
            self.s3.upload_file(
                Filename=str(local_path),
                Bucket=self.bucket,
                Key=s3_key,
                Config=self.transfer_config,
                Callback=upload_callback,
                ExtraArgs=extra_args or {},
            )

            logger.debug(f"Successfully uploaded {local_path.name} to S3")

            return S3FileLocation(
                url=f"s3://{self.bucket}/{s3_key}",
                bucket=self.bucket,
                key=s3_key,
                region=self.s3.meta.region_name,
                endpoint=self.s3.meta.endpoint_url,
            )

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            logger.error(f"Failed to upload {local_path}: {error_code} - {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error uploading {local_path}: {e}")
            raise

    def download_file(self, s3_key: str, local_path: Path, callback: Optional[Callable[[int], Any]] = None) -> Path:
        """
        Download a file from S3 with optimized settings.

        Args:
            s3_key: S3 object key
            local_path: Local path to save the file
            callback: Optional callback for progress tracking

        Returns:
            Path to the downloaded file
        """
        logger.debug(f"Downloading s3://{self.bucket}/{s3_key} to {local_path}")

        try:
            # Ensure parent directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)

            self.s3.download_file(
                Bucket=self.bucket, Key=s3_key, Filename=str(local_path), Config=self.transfer_config, Callback=callback
            )

            logger.info(f"Successfully downloaded {s3_key}")
            return local_path

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                raise FileNotFoundError(f"S3 object not found: s3://{self.bucket}/{s3_key}")
            logger.error(f"Failed to download {s3_key}: {error_code} - {e}")
            raise

    def file_exists(self, s3_key: str) -> bool:
        """Check if a file exists in S3."""
        try:
            self.s3.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def get_file_info(self, s3_key: str) -> S3FileInfo:
        """Get metadata about a file in S3."""
        try:
            response = self.s3.head_object(Bucket=self.bucket, Key=s3_key)
            return S3FileInfo(
                key=s3_key,
                size=response["ContentLength"],
                last_modified=response["LastModified"],
                etag=response["ETag"].strip('"'),
                content_type=response.get("ContentType", "binary/octet-stream"),
                metadata=response.get("Metadata", {}),
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                raise FileNotFoundError(f"S3 object not found: s3://{self.bucket}/{s3_key}")
            raise

    def list_files(self, prefix: str = "", max_keys: int = 1000) -> list[S3FileInfo]:
        """List files in the bucket with a given prefix."""
        try:
            paginator = self.s3.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix, PaginationConfig={"MaxItems": max_keys})

            files = []
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        files.append(
                            S3FileInfo(
                                key=obj["Key"],
                                size=obj["Size"],
                                last_modified=obj["LastModified"],
                                etag=obj["ETag"].strip('"'),
                            )
                        )

            return files

        except ClientError as e:
            logger.error(f"Failed to list files: {e}")
            raise

    def delete_file(self, s3_key: str) -> None:
        """Delete a file from S3."""
        try:
            self.s3.delete_object(Bucket=self.bucket, Key=s3_key)
            logger.info(f"Deleted s3://{self.bucket}/{s3_key}")
        except ClientError as e:
            logger.error(f"Failed to delete {s3_key}: {e}")
            raise
