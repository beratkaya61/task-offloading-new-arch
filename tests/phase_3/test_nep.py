"""Out-of-core NEP row, range, and reference-integrity validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from task_offloading.data.nep import (
    NepProfileError,
    profile_nep_accepted_rows,
    profile_nep_directory,
)


def write_fixture(
    root: Path, *, cpu_rate: str = "0.5", unknown_vm: bool = False
) -> Path:
    data = root / "Full_trace"
    data.mkdir(parents=True)
    (data / "PM.csv").write_text(
        "pm_name,site_id,cores,memory,storage\n"
        "pm-1,site-1,8,16384,100000\n",
        encoding="utf-8",
    )
    (data / "VM.csv").write_text(
        "vm_id,uid,pm_name,site_id,status,image_id,cores,memory,storage,"
        "os_type,os_name,start_time,end_time\n"
        "vm-1,user-1,pm-1,site-1,running,image-1,2,2048,20000,linux,"
        "ubuntu,1590969600,1593561600\n",
        encoding="utf-8",
    )
    vm_id = "unknown-vm" if unknown_vm else "vm-1"
    (data / "VM_CPU.csv").write_text(
        "vm_id,site_id,cpu_rate,report_ts\n"
        f"{vm_id},site-1,{cpu_rate},1590969600\n",
        encoding="utf-8",
    )
    (data / "VM_BANDWIDTH.csv").write_text(
        "vm_id,site_id,pub_down_flow,pub_up_flow,pub_down_bw,pub_up_bw,"
        "pri_down_flow,pri_up_flow,pri_down_bw,pri_up_bw,report_ts\n"
        "vm-1,site-1,1,2,100.5,50.25,3,4,20.0,10.0,1590969600\n",
        encoding="utf-8",
    )
    (data / "VM_BW_THREE_MONTHS.csv").write_text(
        "vm_id,report_time,pub_down_bw,pub_up_bw,pri_down_bw,pri_up_bw\n"
        "vm-1,2020-06-01 00:00:00,100.5,50.25,20.0,10.0\n",
        encoding="utf-8",
    )
    (data / "SITE_RTT.csv").write_text(
        ",from_site_id,to_site_id,rtt,loss,type,biz_ts,biz_time,create_time\n"
        "0,site-1,site-1,1.5,0.0,icmp,1590969600,"
        "2020-06-01 00:00:00,2020-06-01 00:00:01\n",
        encoding="utf-8",
    )
    return root


def test_nep_full_fixture_passes_schema_range_and_reference_checks(
    tmp_path: Path,
) -> None:
    spill = tmp_path / "spill"
    profile = profile_nep_directory(
        write_fixture(tmp_path / "data"), temp_directory=spill
    )
    assert profile.is_valid
    assert spill.is_dir()
    assert profile.total_rows == 6
    serialized = profile.as_dict()
    assert serialized["engine"]["out_of_core"] is True
    assert all(member["row_count"] == 1 for member in serialized["members"])


@pytest.mark.parametrize("cpu_rate", ["not-number", "-0.1", "1.1", "nan"])
def test_nep_profile_rejects_invalid_cpu_values(
    tmp_path: Path, cpu_rate: str
) -> None:
    profile = profile_nep_directory(write_fixture(tmp_path, cpu_rate=cpu_rate))
    cpu = next(member for member in profile.members if member.filename == "VM_CPU.csv")
    assert not cpu.is_valid
    assert cpu.invalid_row_count == 1


def test_nep_profile_detects_unknown_vm_reference(tmp_path: Path) -> None:
    profile = profile_nep_directory(write_fixture(tmp_path, unknown_vm=True))
    cpu = next(member for member in profile.members if member.filename == "VM_CPU.csv")
    assert not cpu.is_valid
    check = next(check for check in cpu.checks if check.name == "unknown_vm_rows")
    assert check.count == 1


def test_nep_profile_rejects_header_and_file_set_drift(tmp_path: Path) -> None:
    root = write_fixture(tmp_path)
    cpu = root / "Full_trace" / "VM_CPU.csv"
    cpu.write_text("wrong,header\n1,2\n", encoding="utf-8")
    with pytest.raises(NepProfileError, match="unexpected header"):
        profile_nep_directory(root)

    root = write_fixture(tmp_path / "other")
    (root / "Full_trace" / "extra.csv").write_text("x\n1\n", encoding="utf-8")
    with pytest.raises(NepProfileError, match="file set mismatch"):
        profile_nep_directory(root)


@pytest.mark.parametrize(
    "filename",
    [
        "PM.csv",
        "VM.csv",
        "VM_CPU.csv",
        "VM_BANDWIDTH.csv",
        "VM_BW_THREE_MONTHS.csv",
        "SITE_RTT.csv",
    ],
)
def test_nep_acceptance_policy_keeps_valid_lineage(
    tmp_path: Path, filename: str
) -> None:
    root = write_fixture(tmp_path)
    assert profile_nep_accepted_rows(root, filename) == 1


@pytest.mark.parametrize(
    ("cpu_rate", "unknown_vm"),
    [("1.1", False), ("0.5", True)],
)
def test_nep_acceptance_policy_rejects_invalid_cpu_or_lineage(
    tmp_path: Path, cpu_rate: str, unknown_vm: bool
) -> None:
    root = write_fixture(
        tmp_path, cpu_rate=cpu_rate, unknown_vm=unknown_vm
    )
    assert profile_nep_accepted_rows(root, "VM_CPU.csv") == 0


def test_nep_acceptance_policy_rejects_invalid_parent_pm(tmp_path: Path) -> None:
    root = write_fixture(tmp_path)
    pm = root / "Full_trace" / "PM.csv"
    pm.write_text(
        "pm_name,site_id,cores,memory,storage\npm-1,site-1,0,16384,100000\n",
        encoding="utf-8",
    )
    assert profile_nep_accepted_rows(root, "PM.csv") == 0
    assert profile_nep_accepted_rows(root, "VM.csv") == 0
    assert profile_nep_accepted_rows(root, "VM_CPU.csv") == 0


def test_nep_acceptance_policy_rejects_unknown_member(tmp_path: Path) -> None:
    root = write_fixture(tmp_path)
    with pytest.raises(NepProfileError, match="unknown NEP member"):
        profile_nep_accepted_rows(root, "UNKNOWN.csv")
