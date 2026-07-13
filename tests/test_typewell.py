from pathlib import Path

import pytest

from rogii.typewell import typewell_numeric_fingerprint


def write_typewell(path: Path, rows: str, header: str = "TVT,GR,Geology") -> Path:
    path.write_text(f"{header}\n{rows}", encoding="utf-8")
    return path


def test_numeric_formatting_and_geology_do_not_change_fingerprint(tmp_path: Path) -> None:
    first = write_typewell(tmp_path / "first.csv", "1.0,2.00,A\n3,4.000,B\n")
    second = write_typewell(tmp_path / "second.csv", "1.000,2,C\n3.0,4,D\n")
    assert typewell_numeric_fingerprint(first) == typewell_numeric_fingerprint(second)


def test_row_order_changes_fingerprint(tmp_path: Path) -> None:
    first = write_typewell(tmp_path / "first.csv", "1,2,A\n3,4,B\n")
    reversed_rows = write_typewell(tmp_path / "reversed.csv", "3,4,B\n1,2,A\n")
    assert typewell_numeric_fingerprint(first) != typewell_numeric_fingerprint(reversed_rows)


@pytest.mark.parametrize("bad_value", ["not-a-number", "NaN", "Inf", "-Inf"])
def test_invalid_or_nonfinite_values_raise(tmp_path: Path, bad_value: str) -> None:
    path = write_typewell(tmp_path / "bad.csv", f"{bad_value},2,A\n")
    with pytest.raises(ValueError):
        typewell_numeric_fingerprint(path)


def test_missing_required_column_raises(tmp_path: Path) -> None:
    path = write_typewell(tmp_path / "missing.csv", "1,A\n", header="TVT,Geology")
    with pytest.raises(ValueError, match="missing required typewell columns"):
        typewell_numeric_fingerprint(path)
