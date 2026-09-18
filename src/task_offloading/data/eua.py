"""Privacy-minimizing streaming adapter for the EUA location snapshot."""

from __future__ import annotations

import csv
import hashlib
import math
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from io import TextIOWrapper
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

_EDGE_SUFFIX = "/edge-servers/site.csv"
_USER_SUFFIX = "/users/users-aus.csv"
_EDGE_HEADER = [
    "SITE_ID",
    "LATITUDE",
    "LONGITUDE",
    "NAME",
    "STATE",
    "LICENSING_AREA_ID",
    "POSTCODE",
    "SITE_PRECISION",
    "ELEVATION",
    "HCIS_L2",
]
_USER_HEADER = ["IP", "Latitude", "Longitude", "PostCode", "City", "State", "Country"]


class EuaParseError(ValueError):
    """The EUA snapshot or a selected coordinate row is invalid."""


class EuaEntityKind(StrEnum):
    EDGE_SERVER = "edge_server"
    USER = "user"


@dataclass(frozen=True, slots=True)
class EuaLocationRecord:
    """Only coordinates and row-derived IDs; raw IP/site/name fields are absent."""

    source_entity_id: str
    entity_kind: EuaEntityKind
    source_member: str
    source_row_number: int
    latitude_deg: float
    longitude_deg: float


def _safe_members(archive: ZipFile) -> tuple[str, ...]:
    names = tuple(item.filename for item in archive.infolist())
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise EuaParseError(f"unsafe archive member: {name}")
    return names


def _selected_member(names: tuple[str, ...], suffix: str) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise EuaParseError(f"expected exactly one archive member ending in {suffix}")
    return matches[0]


def _coordinate(value: str, *, latitude: bool, location: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise EuaParseError(f"{location} coordinate must be numeric") from error
    limit = 90.0 if latitude else 180.0
    if not math.isfinite(parsed) or not -limit <= parsed <= limit:
        raise EuaParseError(f"{location} coordinate is outside valid bounds")
    return parsed


def _iter_member(
    archive: ZipFile,
    *,
    member: str,
    kind: EuaEntityKind,
    expected_header: list[str],
    latitude_key: str,
    longitude_key: str,
) -> Iterator[EuaLocationRecord]:
    with archive.open(member) as raw, TextIOWrapper(
        raw, encoding="utf-8-sig", newline=""
    ) as text:
        reader = csv.DictReader(text)
        if reader.fieldnames != expected_header:
            raise EuaParseError(f"unexpected header in {member}")
        for row_number, row in enumerate(reader, start=2):
            location = f"{member}:{row_number}"
            latitude = _coordinate(
                row[latitude_key], latitude=True, location=location
            )
            longitude = _coordinate(
                row[longitude_key], latitude=False, location=location
            )
            identifier = hashlib.sha256(location.encode()).hexdigest()
            yield EuaLocationRecord(
                source_entity_id=identifier,
                entity_kind=kind,
                source_member=member,
                source_row_number=row_number,
                latitude_deg=latitude,
                longitude_deg=longitude,
            )


def iter_eua_archive(path: Path) -> Iterator[EuaLocationRecord]:
    """Yield observed site/user coordinates while discarding names and raw IPs."""

    try:
        with ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise EuaParseError("EUA ZIP CRC validation failed")
            names = _safe_members(archive)
            edge_member = _selected_member(names, _EDGE_SUFFIX)
            user_member = _selected_member(names, _USER_SUFFIX)
            yield from _iter_member(
                archive,
                member=edge_member,
                kind=EuaEntityKind.EDGE_SERVER,
                expected_header=_EDGE_HEADER,
                latitude_key="LATITUDE",
                longitude_key="LONGITUDE",
            )
            yield from _iter_member(
                archive,
                member=user_member,
                kind=EuaEntityKind.USER,
                expected_header=_USER_HEADER,
                latitude_key="Latitude",
                longitude_key="Longitude",
            )
    except BadZipFile as error:
        raise EuaParseError("invalid EUA ZIP") from error
