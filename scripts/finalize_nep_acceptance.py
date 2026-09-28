"""Add resumable, exact acceptance counts to an existing NEP full profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from task_offloading.data.nep import (
    NEP_ACCEPTANCE_POLICY_VERSION,
    profile_nep_accepted_rows,
)

MEMBER_ORDER = (
    "PM.csv",
    "VM.csv",
    "VM_CPU.csv",
    "VM_BANDWIDTH.csv",
    "VM_BW_THREE_MONTHS.csv",
    "SITE_RTT.csv",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted-root", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument(
        "--temp-directory", type=Path, default=Path(".duckdb_tmp")
    )
    return parser.parse_args()


def _load_profile(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value: Any = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("NEP profile must be a JSON object")
    full = value.get("full_row_profile")
    if not isinstance(full, dict) or not isinstance(full.get("members"), list):
        raise ValueError("NEP full-row profile is missing")
    return value


def _write_profile(path: Path, profile: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    serialized = json.dumps(profile, ensure_ascii=False, indent=2) + "\n"
    temporary.write_text(serialized, encoding="utf-8", newline="\n")
    temporary.replace(path)


def main() -> None:
    args = _arguments()
    profile = _load_profile(args.profile)
    full: dict[str, Any] = profile["full_row_profile"]
    members: list[dict[str, Any]] = full["members"]
    by_name = {member["filename"]: member for member in members}
    if set(by_name) != set(MEMBER_ORDER):
        raise ValueError("NEP profile member set does not match the frozen contract")

    args.temp_directory.mkdir(parents=True, exist_ok=True)
    for filename in MEMBER_ORDER:
        member = by_name[filename]
        acceptance = member.get("acceptance")
        if (
            isinstance(acceptance, dict)
            and acceptance.get("policy_version")
            == NEP_ACCEPTANCE_POLICY_VERSION
        ):
            print(f"skip {filename}: existing acceptance count", flush=True)
            continue
        print(f"start {filename}", flush=True)
        with TemporaryDirectory(
            prefix="nep_acceptance_", dir=args.temp_directory
        ) as temporary:
            accepted = profile_nep_accepted_rows(
                args.extracted_root,
                filename,
                temp_directory=Path(temporary),
            )
        rows = int(member["row_count"])
        if accepted < 0 or accepted > rows:
            raise ValueError(f"invalid acceptance count for {filename}")
        member["acceptance"] = {
            "policy_version": NEP_ACCEPTANCE_POLICY_VERSION,
            "accepted_row_count": accepted,
            "rejected_row_count": rows - accepted,
        }
        _write_profile(args.profile, profile)
        print(
            f"done {filename}: accepted={accepted} rejected={rows - accepted}",
            flush=True,
        )

    full["acceptance_policy"] = {
        "version": NEP_ACCEPTANCE_POLICY_VERSION,
        "rule": (
            "Accept only type/range-valid rows with complete PM-to-VM-to-site "
            "lineage; never repair or invent identifiers."
        ),
    }
    full["accepted_rows_total"] = sum(
        int(member["acceptance"]["accepted_row_count"]) for member in members
    )
    full["rejected_rows_total"] = sum(
        int(member["acceptance"]["rejected_row_count"]) for member in members
    )
    _write_profile(args.profile, profile)


if __name__ == "__main__":
    main()
