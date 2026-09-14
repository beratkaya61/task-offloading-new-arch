"""BUPT safe-prefix parsing and privacy-boundary tests."""

from __future__ import annotations

from datetime import datetime

import pytest

from task_offloading.data.bupt import BuptParseError, parse_bupt_line

KEY = b"phase-3-test-key-do-not-use-in-production"
SOURCE = "root/Data/beijing/BY_USER_20_XJL_20150801.csv"


def valid_line(
    *,
    start: str = "2015-08-01 12:00:00",
    end: str = "2015-08-01 12:00:03",
) -> str:
    prefix = [
        "encrypted-device",
        "41001",
        "12001",
        "3gnet",
        "encrypted-imei",
        "192.0.2.1",
        "198.51.100.1",
        start,
        end,
        "3",
        "125",
        "50",
        "1",
        "10.0.0.1",
        "55000",
        "203.0.113.2",
        "443",
        "200",
    ]
    return ",".join(prefix) + ",https://private.example/a,b,user-agent,text/plain"


def replace_prefix_field(index: int, value: str) -> str:
    fields = valid_line().split(",", 18)
    fields[index] = value
    return ",".join(fields)


def test_safe_parser_derives_training_fields_and_discards_tail() -> None:
    record = parse_bupt_line(
        valid_line(),
        source_path=SOURCE,
        source_row_number=7,
        pseudonymization_key=KEY,
    )

    assert record.arrival_time == datetime(2015, 8, 1, 12, 0)
    assert record.input_bits == 1000
    assert record.output_bits == 400
    assert record.destination_port == 443
    assert record.city == "beijing"
    representation = repr(record)
    assert "private.example" not in representation
    assert "encrypted-device" not in representation
    assert "203.0.113.2" not in representation


def test_pseudonyms_are_stable_but_key_scoped() -> None:
    first = parse_bupt_line(
        valid_line(),
        source_path=SOURCE,
        source_row_number=1,
        pseudonymization_key=KEY,
    )
    repeated = parse_bupt_line(
        valid_line(),
        source_path=SOURCE,
        source_row_number=1,
        pseudonymization_key=KEY,
    )
    other_key = parse_bupt_line(
        valid_line(),
        source_path=SOURCE,
        source_row_number=1,
        pseudonymization_key=b"another-project-key-1234",
    )

    assert first.device_id == repeated.device_id
    assert first.station_id == repeated.station_id
    assert first.device_id != other_key.device_id
    assert first.station_id != other_key.station_id


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("too,few,fields", "stable 18-field prefix"),
        (valid_line(start="invalid"), "start_time"),
        (
            valid_line(start="2015-08-01 12:00:04", end="2015-08-01 12:00:03"),
            "must not precede",
        ),
    ],
)
def test_malformed_rows_are_rejected(line: str, message: str) -> None:
    with pytest.raises(BuptParseError, match=message):
        parse_bupt_line(
            line,
            source_path=SOURCE,
            source_row_number=1,
            pseudonymization_key=KEY,
        )


def test_unknown_city_is_rejected() -> None:
    with pytest.raises(BuptParseError, match="supported city"):
        parse_bupt_line(
            valid_line(),
            source_path="root/Data/unknown/file.csv",
            source_row_number=1,
            pseudonymization_key=KEY,
        )


@pytest.mark.parametrize(
    ("line", "message"),
    [
        (replace_prefix_field(0, ""), "must not be empty"),
        (replace_prefix_field(1, "not-an-int"), "must be an integer"),
        (replace_prefix_field(9, "-1"), "must be non-negative"),
        (replace_prefix_field(13, "not-an-ip"), "must be an IP address"),
        (replace_prefix_field(16, "65536"), "at most 65535"),
    ],
)
def test_invalid_safe_prefix_fields_are_rejected(line: str, message: str) -> None:
    with pytest.raises(BuptParseError, match=message):
        parse_bupt_line(
            line,
            source_path=SOURCE,
            source_row_number=1,
            pseudonymization_key=KEY,
        )


def test_short_key_and_invalid_row_number_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least 16 bytes"):
        parse_bupt_line(
            valid_line(),
            source_path=SOURCE,
            source_row_number=1,
            pseudonymization_key=b"short",
        )
    with pytest.raises(ValueError, match="positive"):
        parse_bupt_line(
            valid_line(),
            source_path=SOURCE,
            source_row_number=0,
            pseudonymization_key=KEY,
        )
