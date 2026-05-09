"""Builder error types."""

from __future__ import annotations


class BuildError(Exception):
    """Raised when the builder cannot produce a valid release file."""
