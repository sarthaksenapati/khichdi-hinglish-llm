"""Shared utilities: seeding and logging.

Every script in this project imports from here. Reproducibility starts
with one canonical seed function -- never seed ad hoc inside scripts.
"""
from __future__ import annotations

import logging
import os
import random
import sys

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed every RNG we touch: Python, NumPy, PyTorch (CPU + CUDA).

    Note: this gives *statistical* reproducibility, not bitwise -- CUDA
    kernels (e.g. atomicAdd in reductions) are nondeterministic by
    default. For training runs that's the accepted standard. If you ever
    need bitwise determinism, additionally set
    torch.use_deterministic_algorithms(True) and accept the slowdown.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Console logger with a consistent format across all scripts.

    Why not print(): logs carry timestamps + module names, so when you
    paste a training log into a bug report (or your blog), the sequence
    of events is unambiguous.
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid duplicate handlers on re-import (notebooks!)
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger
