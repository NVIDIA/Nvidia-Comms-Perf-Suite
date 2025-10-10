"""CLI interface for analysis."""

from .convert import convert_test_results
from .upload import upload_test_results

__all__ = ["convert_test_results", "upload_test_results"]
