#!/usr/bin/env python3
"""Prepare the shared s02 cohort without loading either speech classifier."""

import sys
from train import main


if __name__ == "__main__":
    raise SystemExit(main([*sys.argv[1:], "--prepare-only"]))
