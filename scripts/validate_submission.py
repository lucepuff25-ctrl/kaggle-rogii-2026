#!/usr/bin/env python3
"""Validate a ROGII submission against the official sample order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii.submission import validate_submission_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument(
        "--sample",
        type=Path,
        default=ROOT / "data/raw/sample_submission.csv",
    )
    args = parser.parse_args()
    result = validate_submission_file(args.submission.resolve(), args.sample.resolve())
    result["status"] = "ok"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
