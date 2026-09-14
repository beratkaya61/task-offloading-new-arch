"""Privacy-minimizing parser for the headerless BUPT mobile trace."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath

_MIN_PSEUDONYM_KEY_BYTES = 16
_SUPPORTED_CITIES = frozenset({"beijing", "guangzhou", "shanghai"})


class BuptParseError(ValueError):
    """A raw BUPT row does not satisfy the frozen safe-prefix contract."""


@dataclass(frozen=True, slots=True)
class BuptTraceRecord:
    """Training-safe fields; raw identifiers, IPs, URLs and user agents are absent."""

    sample_id: str
    source_path: str
    source_row_number: int
    city: str
    device_id: str
    station_id: str
    arrival_time: datetime
    duration_s: int
    input_bits: int
    output_bits: int
    radio_access_type: int
    destination_port: int
    status_code: int


def _parse_non_negative_int(value: str, field_name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise BuptParseError(f"{field_name} must be an integer") from error
    if parsed < 0:
        raise BuptParseError(f"{field_name} must be non-negative")
    return parsed


def _parse_timestamp(value: str, field_name: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise BuptParseError(f"{field_name} must be an ISO timestamp") from error


def _validate_ip(value: str, field_name: str) -> None:
    try:
        ipaddress.ip_address(value)
    except ValueError as error:
        raise BuptParseError(f"{field_name} must be an IP address") from error


def _pseudonymize(namespace: str, value: str, key: bytes) -> str:
    if len(key) < _MIN_PSEUDONYM_KEY_BYTES:
        raise ValueError(
            f"pseudonymization key must be at least {_MIN_PSEUDONYM_KEY_BYTES} bytes"
        )
    payload = f"{namespace}:{value}".encode()
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def parse_bupt_line(
    line: str,
    *,
    source_path: str,
    source_row_number: int,
    pseudonymization_key: bytes,
) -> BuptTraceRecord:
    """Parse the stable 18-field prefix and discard the unsafe free-text tail.

    The upstream files are headerless and do not quote commas in URL/User-Agent
    fields. Splitting at most 18 times preserves the stable numeric prefix while
    intentionally treating the remaining text as opaque and never returning it.
    """

    if source_row_number <= 0:
        raise ValueError("source_row_number must be positive")
    fields = line.rstrip("\r\n").split(",", 18)
    if len(fields) != 19:
        raise BuptParseError("row must contain the stable 18-field prefix")

    raw_device_id = fields[0]
    if not raw_device_id:
        raise BuptParseError("device identifier must not be empty")

    path = PurePosixPath(source_path)
    city = path.parent.name.lower()
    if city not in _SUPPORTED_CITIES:
        raise BuptParseError("source path must identify a supported city")

    lac = _parse_non_negative_int(fields[1], "location_area_code")
    cell_id = _parse_non_negative_int(fields[2], "cell_identity")
    start_time = _parse_timestamp(fields[7], "start_time")
    end_time = _parse_timestamp(fields[8], "end_time")
    if end_time < start_time:
        raise BuptParseError("end_time must not precede start_time")

    duration_s = _parse_non_negative_int(fields[9], "duration_s")
    uplink_bytes = _parse_non_negative_int(fields[10], "uplink_bytes")
    downlink_bytes = _parse_non_negative_int(fields[11], "downlink_bytes")
    radio_access_type = _parse_non_negative_int(fields[12], "radio_access_type")
    source_port = _parse_non_negative_int(fields[14], "source_port")
    destination_port = _parse_non_negative_int(fields[16], "destination_port")
    status_code = _parse_non_negative_int(fields[17], "status_code")
    if source_port > 65535 or destination_port > 65535:
        raise BuptParseError("network ports must be at most 65535")

    _validate_ip(fields[5], "sgsn_ip")
    _validate_ip(fields[6], "ggsn_ip")
    _validate_ip(fields[13], "source_ip")
    _validate_ip(fields[15], "destination_ip")

    station_source = f"{city}:{lac}:{cell_id}"
    sample_source = f"{source_path}:{source_row_number}"
    return BuptTraceRecord(
        sample_id=hashlib.sha256(sample_source.encode()).hexdigest(),
        source_path=source_path,
        source_row_number=source_row_number,
        city=city,
        device_id=_pseudonymize("device", raw_device_id, pseudonymization_key),
        station_id=_pseudonymize("station", station_source, pseudonymization_key),
        arrival_time=start_time,
        duration_s=duration_s,
        input_bits=uplink_bytes * 8,
        output_bits=downlink_bytes * 8,
        radio_access_type=radio_access_type,
        destination_port=destination_port,
        status_code=status_code,
    )
