"""Stable numeric fingerprints for ROGII typewell curves."""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
import struct


TYPEWELL_FINGERPRINT_ALGORITHM = "sha256-tvt-gr-float64be-v1"
_DOMAIN_PREFIX = b"rogii-typewell-float64be-v1\0"


def _finite_float64(value: str, *, column: str, row_number: int, path: Path) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"non-numeric {column} at row {row_number} in {path}") from exc
    if not math.isfinite(number):
        raise ValueError(f"non-finite {column} at row {row_number} in {path}")
    # Canonicalize signed zero while preserving every other IEEE-754 float64 value.
    return 0.0 if number == 0.0 else number


def typewell_numeric_fingerprint(path: str | Path) -> str:
    """Hash ordered TVT/GR rows after numeric float64 normalization.

    Only ``TVT`` and ``GR`` participate. Values are encoded as big-endian
    IEEE-754 float64 pairs, so textual variants such as ``1.0`` and ``1.000``
    are identical while row order remains significant.
    """
    source = Path(path)
    digest = hashlib.sha256(_DOMAIN_PREFIX)
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        missing = [column for column in ("TVT", "GR") if column not in fields]
        if missing:
            raise ValueError(f"missing required typewell columns {missing} in {source}")
        for row_number, row in enumerate(reader, start=2):
            tvt = _finite_float64(row["TVT"], column="TVT", row_number=row_number, path=source)
            gr = _finite_float64(row["GR"], column="GR", row_number=row_number, path=source)
            digest.update(struct.pack(">dd", tvt, gr))
    return digest.hexdigest()
