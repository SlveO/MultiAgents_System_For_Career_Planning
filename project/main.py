#!/usr/bin/env python3
"""Backward-compatible entry point for the canonical completion-MVP CLI."""

from .assistant_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
