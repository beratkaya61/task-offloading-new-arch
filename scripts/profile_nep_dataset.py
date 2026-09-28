"""Generate aggregate-only NEP full-row validation evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from task_offloading.data.nep import profile_nep_directory


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted-root", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument(
        "--temp-directory", type=Path, default=Path(".duckdb_tmp")
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    with args.profile.open(encoding="utf-8") as handle:
        base: Any = json.load(handle)
    if not isinstance(base, dict):
        raise ValueError("base NEP profile must be a JSON object")

    args.temp_directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix="nep_profile_", dir=args.temp_directory
    ) as temporary:
        full_profile = profile_nep_directory(
            args.extracted_root, temp_directory=Path(temporary)
        )
    validation = base.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("base NEP profile validation must be an object")
    validation["full_row_schema_validation"] = (
        "completed_all_rows_valid"
        if full_profile.is_valid
        else "completed_with_violations"
    )
    base["full_row_profile"] = full_profile.as_dict()
    serialized = json.dumps(base, ensure_ascii=False, indent=2) + "\n"
    args.profile.write_text(serialized, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
