"""Streaming adapter for UCI dataset 859's nested ZIP layout."""

from __future__ import annotations

import csv
import hashlib
import math
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO, TextIOWrapper
from pathlib import Path
from zipfile import BadZipFile, ZipFile

_NESTED_ARCHIVE = "TATDescriptionDataset.zip"
_CSV_MEMBERS = {
    "TATDescriptionDataset/MacBookPro1.csv": "macbook_pro_1",
    "TATDescriptionDataset/MacBookPro2.csv": "macbook_pro_2",
    "TATDescriptionDataset/RasberryPi.csv": "raspberry_pi_4b",
    "TATDescriptionDataset/VM.csv": "ubuntu_virtualbox_vm",
}
_EXPECTED_HEADER = ["Time", "Execution Time"]


class UciMecParseError(ValueError):
    """The artifact or a CSV row violates the frozen UCI-859 contract."""


@dataclass(frozen=True, slots=True)
class UciMecExecutionRecord:
    sample_id: str
    server_id: str
    source_member: str
    source_row_number: int
    observed_at: datetime
    turnaround_time_s: float


def _parse_row(
    row: list[str], *, member: str, row_number: int, server_id: str
) -> UciMecExecutionRecord:
    if len(row) != 2:
        raise UciMecParseError(f"{member}:{row_number} must have two columns")
    try:
        observed_at = datetime.strptime(row[0], "%a %b %d %H:%M:%S %Y")
    except ValueError as error:
        raise UciMecParseError(
            f"{member}:{row_number} has an invalid source timestamp"
        ) from error
    try:
        turnaround = float(row[1])
    except ValueError as error:
        raise UciMecParseError(
            f"{member}:{row_number} execution time must be numeric"
        ) from error
    if not math.isfinite(turnaround) or turnaround <= 0:
        raise UciMecParseError(
            f"{member}:{row_number} execution time must be finite and positive"
        )
    sample_id = hashlib.sha256(f"{member}:{row_number}".encode()).hexdigest()
    return UciMecExecutionRecord(
        sample_id=sample_id,
        server_id=server_id,
        source_member=member,
        source_row_number=row_number,
        observed_at=observed_at,
        turnaround_time_s=turnaround,
    )


def iter_uci_mec_archive(path: Path) -> Iterator[UciMecExecutionRecord]:
    """Yield 4,000 measurements without extracting either ZIP to disk."""

    try:
        with ZipFile(path) as outer:
            if outer.testzip() is not None:
                raise UciMecParseError("outer ZIP CRC validation failed")
            names = outer.namelist()
            if names != [_NESTED_ARCHIVE]:
                raise UciMecParseError("outer ZIP must contain only the nested archive")
            nested_payload = outer.read(_NESTED_ARCHIVE)
    except (BadZipFile, KeyError) as error:
        raise UciMecParseError("invalid UCI outer ZIP") from error

    try:
        with ZipFile(BytesIO(nested_payload)) as archive:
            if archive.testzip() is not None:
                raise UciMecParseError("nested ZIP CRC validation failed")
            present_csv = {
                name for name in archive.namelist() if name.lower().endswith(".csv")
            }
            if present_csv != set(_CSV_MEMBERS):
                raise UciMecParseError("nested ZIP CSV members do not match contract")
            for member, server_id in _CSV_MEMBERS.items():
                with archive.open(member) as raw, TextIOWrapper(
                    raw, encoding="utf-8-sig", newline=""
                ) as text:
                    reader = csv.reader(text)
                    header = next(reader, None)
                    if header != _EXPECTED_HEADER:
                        raise UciMecParseError(f"unexpected header in {member}")
                    for row_number, row in enumerate(reader, start=2):
                        yield _parse_row(
                            row,
                            member=member,
                            row_number=row_number,
                            server_id=server_id,
                        )
    except BadZipFile as error:
        raise UciMecParseError("invalid UCI nested ZIP") from error
