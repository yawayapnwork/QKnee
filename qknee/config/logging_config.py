"""
Central logging setup for Q-Knee.

Replaces the project's scattered `print(...)` calls and ad-hoc
`logging.basicConfig(...)` calls (previously duplicated in nearly every
module's `if __name__ == "__main__":` block) with one configuration point,
driven by `qknee/config/config.yaml`'s `logging` section.

Usage:
    from qknee.config.logging_config import setup_logging, get_logger

    setup_logging()                 # call once, e.g. at process entry point
    logger = get_logger(__name__)
    logger.info("Loaded %d samples", n_samples)
"""

from __future__ import annotations

import logging
from typing import Optional

from qknee.config.loader import QKneeConfig, load_config

_CONFIGURED = False


def setup_logging(config: Optional[QKneeConfig] = None, level: Optional[str] = None) -> None:
    """Configures the root logger once per process.

    Args:
        config: A pre-loaded `QKneeConfig`; if omitted, `load_config()` is used.
        level: Optional override for `config.logging.level`
            (e.g. "DEBUG" for a verbose local run), independent of config.yaml.

    Idempotent: subsequent calls are no-ops so importing this module from
    multiple entry points (API, UI, CLI scripts, tests) never double-attaches
    handlers or produces duplicate log lines.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    config = config or load_config()
    resolved_level = (level or config.logging.level).upper()

    logging.basicConfig(
        level=getattr(logging, resolved_level, logging.INFO),
        format=config.logging.format,
        datefmt=config.logging.datefmt,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Returns a module-scoped logger, configuring root logging on first use
    if `setup_logging()` hasn't been called explicitly yet."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
