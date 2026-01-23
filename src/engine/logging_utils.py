"""Logging helpers for SOZLab runs."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Tuple


def setup_run_logger(output_dir: str, name: str = "sozlab") -> Tuple[logging.Logger, str]:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "sozlab.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Remove existing file handlers to avoid duplicate logs across runs.
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)

    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("=== SOZLab run started %s ===", datetime.now().isoformat())
    return logger, log_path
