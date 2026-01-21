# l1obs/logging_utils.py
from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logging(verbose: bool = False, very_verbose: bool = False) -> None:
    """
    Configure root logging. Call once from CLI early.

    verbose: INFO
    very_verbose: DEBUG
    default: WARNING
    """
    if very_verbose:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    handler = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
