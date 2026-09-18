"""Privacy-safe adapters for the locally validated UCI and EUA artifacts."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from task_offloading.data import (
    EuaEntityKind,
    EuaParseError,
    UciMecParseError,
    iter_eua_archive,
    iter_uci_mec_archive,
)


def make_uci_archive(
    path: Path,
    *,
    header: str = "Time,Execution Time",
    timestamp: str = "Thu Nov 26 14:47:38 2020",
    value: str = "0.25",
) -> None:
    nested_buffer = BytesIO()
    with ZipFile(nested_buffer, "w", ZIP_DEFLATED) as nested:
        filenames = (
            "MacBookPro1.csv",
            "MacBookPro2.csv",
            "RasberryPi.csv",
            "VM.csv",
        )
        for filename in filenames:
            nested.writestr(
                f"TATDescriptionDataset/{filename}",
                f"{header}\n{timestamp},{value}\n",
            )
    with ZipFile(path, "w", ZIP_DEFLATED) as outer:
        outer.writestr("TATDescriptionDataset.zip", nested_buffer.getvalue())


def make_eua_archive(
    path: Path,
    *,
    edge_latitude: str = "-37.8",
    user_longitude: str = "145.1",
    edge_header: str | None = None,
    unsafe_member: bool = False,
) -> None:
    root = "eua-dataset-commit"
    actual_edge_header = edge_header or (
        "SITE_ID,LATITUDE,LONGITUDE,NAME,STATE,LICENSING_AREA_ID,POSTCODE,"
        "SITE_PRECISION,ELEVATION,HCIS_L2"
    )
    user_header = "IP,Latitude,Longitude,PostCode,City,State,Country"
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr(
            f"{root}/edge-servers/site.csv",
            f"{actual_edge_header}\n"
            f"raw-site,{edge_latitude},144.9,Raw Name,VIC,2,3000,Exact,1,X\n",
        )
        archive.writestr(
            f"{root}/users/users-aus.csv",
            f"{user_header}\n1.2.3.4,-37.9,{user_longitude},3000,Raw City,VIC,AU\n",
        )
        if unsafe_member:
            archive.writestr("../escape.txt", "unsafe")


def test_uci_nested_archive_maps_four_servers(tmp_path: Path) -> None:
    path = tmp_path / "uci.zip"
    make_uci_archive(path)
    records = tuple(iter_uci_mec_archive(path))
    assert len(records) == 4
    assert {record.server_id for record in records} == {
        "macbook_pro_1",
        "macbook_pro_2",
        "raspberry_pi_4b",
        "ubuntu_virtualbox_vm",
    }
    assert all(record.turnaround_time_s == 0.25 for record in records)
    assert len({record.sample_id for record in records}) == 4


@pytest.mark.parametrize(
    ("header", "value", "message"),
    [
        ("Wrong,Header", "0.25", "unexpected header"),
        ("Time,Execution Time", "not-number", "must be numeric"),
        ("Time,Execution Time", "0", "finite and positive"),
        ("Time,Execution Time", "nan", "finite and positive"),
    ],
)
def test_uci_rejects_schema_and_value_errors(
    tmp_path: Path, header: str, value: str, message: str
) -> None:
    path = tmp_path / "uci-invalid.zip"
    make_uci_archive(path, header=header, value=value)
    with pytest.raises(UciMecParseError, match=message):
        tuple(iter_uci_mec_archive(path))


def test_uci_rejects_wrong_outer_layout(tmp_path: Path) -> None:
    path = tmp_path / "wrong.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("unexpected.txt", "content")
    with pytest.raises(UciMecParseError, match="nested archive"):
        tuple(iter_uci_mec_archive(path))


@pytest.mark.parametrize(
    ("timestamp", "value", "message"),
    [
        ("not-a-timestamp", "0.25", "invalid source timestamp"),
        ("Thu Nov 26 14:47:38 2020", "0.25,extra", "two columns"),
    ],
)
def test_uci_rejects_malformed_rows(
    tmp_path: Path, timestamp: str, value: str, message: str
) -> None:
    path = tmp_path / "uci-malformed-row.zip"
    make_uci_archive(path, timestamp=timestamp, value=value)
    with pytest.raises(UciMecParseError, match=message):
        tuple(iter_uci_mec_archive(path))


def test_uci_rejects_invalid_outer_and_nested_zip(tmp_path: Path) -> None:
    invalid_outer = tmp_path / "not-a-zip.zip"
    invalid_outer.write_bytes(b"not a zip")
    with pytest.raises(UciMecParseError, match="invalid UCI outer ZIP"):
        tuple(iter_uci_mec_archive(invalid_outer))

    invalid_nested = tmp_path / "invalid-nested.zip"
    with ZipFile(invalid_nested, "w") as archive:
        archive.writestr("TATDescriptionDataset.zip", b"not a nested zip")
    with pytest.raises(UciMecParseError, match="invalid UCI nested ZIP"):
        tuple(iter_uci_mec_archive(invalid_nested))


def test_uci_requires_exact_nested_csv_set(tmp_path: Path) -> None:
    nested_buffer = BytesIO()
    with ZipFile(nested_buffer, "w") as nested:
        nested.writestr(
            "TATDescriptionDataset/MacBookPro1.csv",
            "Time,Execution Time\nThu Nov 26 14:47:38 2020,0.25\n",
        )
    path = tmp_path / "missing-csv.zip"
    with ZipFile(path, "w") as outer:
        outer.writestr("TATDescriptionDataset.zip", nested_buffer.getvalue())
    with pytest.raises(UciMecParseError, match="CSV members do not match"):
        tuple(iter_uci_mec_archive(path))


def test_eua_adapter_discards_raw_identifiers_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "eua.zip"
    make_eua_archive(path)
    edge, user = tuple(iter_eua_archive(path))
    assert edge.entity_kind is EuaEntityKind.EDGE_SERVER
    assert user.entity_kind is EuaEntityKind.USER
    assert edge.latitude_deg == -37.8
    assert user.longitude_deg == 145.1
    serialized = repr((edge, user))
    assert "raw-site" not in serialized
    assert "Raw Name" not in serialized
    assert "1.2.3.4" not in serialized
    assert "Raw City" not in serialized


@pytest.mark.parametrize(
    ("edge_latitude", "user_longitude"),
    [("91", "145.1"), ("-37.8", "181"), ("not-number", "145.1")],
)
def test_eua_rejects_invalid_coordinates(
    tmp_path: Path, edge_latitude: str, user_longitude: str
) -> None:
    path = tmp_path / "eua-invalid.zip"
    make_eua_archive(
        path, edge_latitude=edge_latitude, user_longitude=user_longitude
    )
    with pytest.raises(EuaParseError, match="coordinate"):
        tuple(iter_eua_archive(path))


def test_eua_rejects_path_traversal_member(tmp_path: Path) -> None:
    path = tmp_path / "eua-unsafe.zip"
    make_eua_archive(path, unsafe_member=True)
    with pytest.raises(EuaParseError, match="unsafe archive member"):
        tuple(iter_eua_archive(path))


def test_eua_requires_both_selected_members(tmp_path: Path) -> None:
    path = tmp_path / "eua-missing.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("root/users/users-aus.csv", "IP,Latitude,Longitude\n")
    with pytest.raises(EuaParseError, match="exactly one"):
        tuple(iter_eua_archive(path))


def test_eua_rejects_changed_header_and_invalid_zip(tmp_path: Path) -> None:
    changed_header = tmp_path / "eua-changed-header.zip"
    make_eua_archive(changed_header, edge_header="LATITUDE,LONGITUDE")
    with pytest.raises(EuaParseError, match="unexpected header"):
        tuple(iter_eua_archive(changed_header))

    invalid_zip = tmp_path / "eua-invalid-zip.zip"
    invalid_zip.write_bytes(b"not a zip")
    with pytest.raises(EuaParseError, match="invalid EUA ZIP"):
        tuple(iter_eua_archive(invalid_zip))
