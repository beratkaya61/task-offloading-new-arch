"""Out-of-core validation for the NEP-large/full extracted CSV files.

The archive expands to more than 60 GB, so validation is expressed as DuckDB
aggregations instead of loading rows into Python memory. Raw identifiers never
leave the external dataset directory; only counts, ranges, and integrity checks
are returned.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import duckdb


class NepProfileError(ValueError):
    """The extracted NEP dataset violates the frozen structural contract."""


class NepColumnKind(StrEnum):
    """Supported strict conversions for raw NEP columns."""

    TEXT = "text"
    INTEGER = "integer"
    FLOAT = "float"
    TIMESTAMP = "timestamp"


@dataclass(frozen=True, slots=True)
class NepColumnContract:
    """Expected type and admissible numeric range for one source column."""

    source_name: str
    name: str
    kind: NepColumnKind
    nullable: bool = False
    minimum: float | int | None = None
    maximum: float | int | None = None
    profile_distinct: bool = True


@dataclass(frozen=True, slots=True)
class NepMemberContract:
    """Exact header and typed fields for one archive member."""

    filename: str
    columns: tuple[NepColumnContract, ...]

    @property
    def source_header(self) -> tuple[str, ...]:
        return tuple(column.source_name for column in self.columns)


@dataclass(frozen=True, slots=True)
class NepColumnProfile:
    """Aggregate-only validation evidence for one source column."""

    name: str
    kind: str
    nullable: bool
    empty_count: int
    distinct_count: int | None
    parse_error_count: int
    non_finite_count: int
    below_minimum_count: int
    above_maximum_count: int
    minimum: int | float | str | None
    maximum: int | float | str | None

    @property
    def invalid_value_count(self) -> int:
        required_empty = 0 if self.nullable else self.empty_count
        return (
            required_empty
            + self.parse_error_count
            + self.non_finite_count
            + self.below_minimum_count
            + self.above_maximum_count
        )


@dataclass(frozen=True, slots=True)
class NepCheck:
    """Cross-column or cross-file check without exposing source values."""

    name: str
    count: int
    severity: str
    description: str


@dataclass(frozen=True, slots=True)
class NepMemberProfile:
    """Aggregate validation result for one full CSV member."""

    filename: str
    size_bytes: int
    row_count: int
    invalid_row_count: int
    columns: tuple[NepColumnProfile, ...]
    checks: tuple[NepCheck, ...]

    @property
    def is_valid(self) -> bool:
        return self.invalid_row_count == 0 and all(
            check.count == 0 for check in self.checks if check.severity == "error"
        )


@dataclass(frozen=True, slots=True)
class NepDatasetProfile:
    """Privacy-safe full-row evidence for all six NEP members."""

    extracted_root: str
    members: tuple[NepMemberProfile, ...]

    @property
    def total_rows(self) -> int:
        return sum(member.row_count for member in self.members)

    @property
    def is_valid(self) -> bool:
        return all(member.is_valid for member in self.members)

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": {
                "name": "duckdb",
                "version": duckdb.__version__,
                "out_of_core": True,
            },
            "extracted_root": self.extracted_root,
            "total_rows": self.total_rows,
            "is_valid": self.is_valid,
            "members": [
                {
                    "filename": member.filename,
                    "size_bytes": member.size_bytes,
                    "row_count": member.row_count,
                    "invalid_row_count": member.invalid_row_count,
                    "is_valid": member.is_valid,
                    "columns": [
                        {
                            "name": column.name,
                            "kind": column.kind,
                            "nullable": column.nullable,
                            "empty_count": column.empty_count,
                            "distinct_count": column.distinct_count,
                            "parse_error_count": column.parse_error_count,
                            "non_finite_count": column.non_finite_count,
                            "below_minimum_count": column.below_minimum_count,
                            "above_maximum_count": column.above_maximum_count,
                            "minimum": column.minimum,
                            "maximum": column.maximum,
                            "invalid_value_count": column.invalid_value_count,
                        }
                        for column in member.columns
                    ],
                    "checks": [
                        {
                            "name": check.name,
                            "count": check.count,
                            "severity": check.severity,
                            "description": check.description,
                        }
                        for check in member.checks
                    ],
                }
                for member in self.members
            ],
        }


def _column(
    name: str,
    kind: NepColumnKind,
    *,
    source_name: str | None = None,
    nullable: bool = False,
    minimum: float | int | None = None,
    maximum: float | int | None = None,
    profile_distinct: bool = True,
) -> NepColumnContract:
    return NepColumnContract(
        source_name=name if source_name is None else source_name,
        name=name,
        kind=kind,
        nullable=nullable,
        minimum=minimum,
        maximum=maximum,
        profile_distinct=profile_distinct,
    )


NEP_MEMBER_CONTRACTS: tuple[NepMemberContract, ...] = (
    NepMemberContract(
        "PM.csv",
        (
            _column("pm_name", NepColumnKind.TEXT),
            _column("site_id", NepColumnKind.TEXT),
            _column("cores", NepColumnKind.INTEGER, minimum=1),
            _column("memory", NepColumnKind.INTEGER, minimum=1),
            _column("storage", NepColumnKind.INTEGER, minimum=1),
        ),
    ),
    NepMemberContract(
        "VM.csv",
        (
            _column("vm_id", NepColumnKind.TEXT),
            _column("uid", NepColumnKind.TEXT, nullable=True),
            _column("pm_name", NepColumnKind.TEXT),
            _column("site_id", NepColumnKind.TEXT),
            _column("status", NepColumnKind.TEXT, nullable=True),
            _column("image_id", NepColumnKind.TEXT, nullable=True),
            _column("cores", NepColumnKind.INTEGER, minimum=0),
            _column("memory", NepColumnKind.INTEGER, minimum=0),
            _column("storage", NepColumnKind.INTEGER, minimum=0),
            _column("os_type", NepColumnKind.TEXT, nullable=True),
            _column("os_name", NepColumnKind.TEXT, nullable=True),
            _column("start_time", NepColumnKind.INTEGER, minimum=0),
            _column("end_time", NepColumnKind.INTEGER, minimum=0),
        ),
    ),
    NepMemberContract(
        "VM_CPU.csv",
        (
            _column("vm_id", NepColumnKind.TEXT, profile_distinct=False),
            _column("site_id", NepColumnKind.TEXT, profile_distinct=False),
            _column(
                "cpu_rate", NepColumnKind.FLOAT, minimum=0.0, maximum=1.0
            ),
            _column("report_ts", NepColumnKind.INTEGER, minimum=1),
        ),
    ),
    NepMemberContract(
        "VM_BANDWIDTH.csv",
        (
            _column("vm_id", NepColumnKind.TEXT, profile_distinct=False),
            _column("site_id", NepColumnKind.TEXT, profile_distinct=False),
            _column("pub_down_flow", NepColumnKind.INTEGER, minimum=0),
            _column("pub_up_flow", NepColumnKind.INTEGER, minimum=0),
            _column("pub_down_bw", NepColumnKind.FLOAT, minimum=0.0),
            _column("pub_up_bw", NepColumnKind.FLOAT, minimum=0.0),
            _column("pri_down_flow", NepColumnKind.INTEGER, minimum=0),
            _column("pri_up_flow", NepColumnKind.INTEGER, minimum=0),
            _column("pri_down_bw", NepColumnKind.FLOAT, minimum=0.0),
            _column("pri_up_bw", NepColumnKind.FLOAT, minimum=0.0),
            _column("report_ts", NepColumnKind.INTEGER, minimum=1),
        ),
    ),
    NepMemberContract(
        "VM_BW_THREE_MONTHS.csv",
        (
            _column("vm_id", NepColumnKind.TEXT, profile_distinct=False),
            _column("report_time", NepColumnKind.TIMESTAMP),
            _column("pub_down_bw", NepColumnKind.FLOAT, minimum=0.0),
            _column("pub_up_bw", NepColumnKind.FLOAT, minimum=0.0),
            _column("pri_down_bw", NepColumnKind.FLOAT, minimum=0.0),
            _column("pri_up_bw", NepColumnKind.FLOAT, minimum=0.0),
        ),
    ),
    NepMemberContract(
        "SITE_RTT.csv",
        (
            _column(
                "row_index",
                NepColumnKind.INTEGER,
                source_name="",
                minimum=0,
            ),
            _column("from_site_id", NepColumnKind.TEXT, profile_distinct=False),
            _column("to_site_id", NepColumnKind.TEXT, profile_distinct=False),
            _column("rtt", NepColumnKind.FLOAT, minimum=0.0),
            _column(
                "loss",
                NepColumnKind.FLOAT,
                nullable=True,
                minimum=0.0,
                maximum=100.0,
            ),
            _column(
                "type",
                NepColumnKind.TEXT,
                nullable=True,
                profile_distinct=False,
            ),
            _column("biz_ts", NepColumnKind.INTEGER, minimum=1),
            _column("biz_time", NepColumnKind.TIMESTAMP),
            _column("create_time", NepColumnKind.TIMESTAMP),
        ),
    ),
)

_CONTRACT_BY_NAME = {contract.filename: contract for contract in NEP_MEMBER_CONTRACTS}
NEP_ACCEPTANCE_POLICY_VERSION = "1.0.0"


def _quote_identifier(value: str) -> str:
    return f'"{value.replace(chr(34), chr(34) * 2)}"'


def _read_csv_sql(contract: NepMemberContract) -> str:
    columns = ", ".join(
        f"'{column.name}': 'VARCHAR'" for column in contract.columns
    )
    return (
        "read_csv(?, header = true, auto_detect = false, "
        f"columns = {{{columns}}}, strict_mode = true, null_padding = false)"
    )


def _empty_expression(column: NepColumnContract) -> str:
    name = f"src.{_quote_identifier(column.name)}"
    return f"({name} IS NULL OR trim({name}) = '')"


def _cast_expression(column: NepColumnContract) -> str:
    name = f"src.{_quote_identifier(column.name)}"
    target = {
        NepColumnKind.INTEGER: "BIGINT",
        NepColumnKind.FLOAT: "DOUBLE",
        NepColumnKind.TIMESTAMP: "TIMESTAMP",
    }[column.kind]
    return f"try_cast({name} AS {target})"


def _invalid_expression(column: NepColumnContract) -> str:
    empty = _empty_expression(column)
    expressions: list[str] = []
    if not column.nullable:
        expressions.append(empty)
    if column.kind is not NepColumnKind.TEXT:
        cast = _cast_expression(column)
        expressions.append(f"(NOT {empty} AND {cast} IS NULL)")
        if column.kind is NepColumnKind.FLOAT:
            expressions.append(f"({cast} IS NOT NULL AND NOT isfinite({cast}))")
        if column.minimum is not None:
            expressions.append(f"({cast} < {column.minimum})")
        if column.maximum is not None:
            expressions.append(f"({cast} > {column.maximum})")
    return "(" + " OR ".join(expressions or ["false"]) + ")"


def _json_scalar(value: object) -> int | float | str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def _column_selects(column: NepColumnContract) -> list[str]:
    empty = _empty_expression(column)
    name = f"src.{_quote_identifier(column.name)}"
    selects = [f"count_if({empty})"]
    if column.kind is NepColumnKind.TEXT:
        distinct = "NULL"
        if column.profile_distinct:
            distinct = f"count(DISTINCT CASE WHEN NOT {empty} THEN {name} END)"
        selects.extend(
            [
                distinct,
                "0",
                "0",
                "0",
                "0",
                "NULL",
                "NULL",
            ]
        )
        return selects

    cast = _cast_expression(column)
    parse_error = f"(NOT {empty} AND {cast} IS NULL)"
    finite = "true"
    if column.kind is NepColumnKind.FLOAT:
        finite = f"({cast} IS NOT NULL AND isfinite({cast}))"
    below = "false"
    above = "false"
    if column.minimum is not None:
        below = f"({finite} AND {cast} < {column.minimum})"
    if column.maximum is not None:
        above = f"({finite} AND {cast} > {column.maximum})"
    non_finite = "false"
    if column.kind is NepColumnKind.FLOAT:
        non_finite = f"({cast} IS NOT NULL AND NOT isfinite({cast}))"
    selects.extend(
        [
            "NULL",
            f"count_if({parse_error})",
            f"count_if({non_finite})",
            f"count_if({below})",
            f"count_if({above})",
            f"min(CASE WHEN {finite} THEN {cast} END)",
            f"max(CASE WHEN {finite} THEN {cast} END)",
        ]
    )
    return selects


def _read_header(path: Path) -> tuple[str, ...]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return tuple(next(csv.reader(handle)))
    except (OSError, UnicodeError, StopIteration, csv.Error) as error:
        raise NepProfileError(f"cannot read NEP header from {path}") from error


def _profile_member(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
    contract: NepMemberContract,
    *,
    joins: str = "",
    checks: tuple[tuple[str, str, str, str], ...] = (),
) -> NepMemberProfile:
    if _read_header(path) != contract.source_header:
        raise NepProfileError(f"unexpected header in {contract.filename}")

    selections = ["count(*)"]
    for column in contract.columns:
        selections.extend(_column_selects(column))
    row_invalid = " OR ".join(
        _invalid_expression(column) for column in contract.columns
    )
    selections.append(f"count_if({row_invalid})")
    selections.extend(expression for _, expression, _, _ in checks)
    query = (
        f"SELECT {', '.join(selections)} "
        f"FROM {_read_csv_sql(contract)} AS src {joins}"
    )
    try:
        result = connection.execute(query, [str(path)]).fetchone()
    except duckdb.Error as error:
        message = f"failed to profile {contract.filename}: {error}"
        raise NepProfileError(message) from error
    if result is None:
        raise NepProfileError(f"no profile result for {contract.filename}")

    cursor = 0
    row_count = int(result[cursor])
    cursor += 1
    column_profiles: list[NepColumnProfile] = []
    for column in contract.columns:
        values = result[cursor : cursor + 8]
        cursor += 8
        column_profiles.append(
            NepColumnProfile(
                name=column.name,
                kind=column.kind.value,
                nullable=column.nullable,
                empty_count=int(values[0]),
                distinct_count=None if values[1] is None else int(values[1]),
                parse_error_count=int(values[2]),
                non_finite_count=int(values[3]),
                below_minimum_count=int(values[4]),
                above_maximum_count=int(values[5]),
                minimum=_json_scalar(values[6]),
                maximum=_json_scalar(values[7]),
            )
        )
    invalid_row_count = int(result[cursor])
    cursor += 1
    check_results = tuple(
        NepCheck(name, int(result[cursor + index]), severity, description)
        for index, (name, _, severity, description) in enumerate(checks)
    )
    return NepMemberProfile(
        filename=contract.filename,
        size_bytes=path.stat().st_size,
        row_count=row_count,
        invalid_row_count=invalid_row_count,
        columns=tuple(column_profiles),
        checks=check_results,
    )


def _reference_check(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
    contract: NepMemberContract,
    *,
    joins: str,
    where: str,
    name: str,
    severity: str,
    description: str,
) -> NepCheck:
    query = (
        f"SELECT count(*) FROM {_read_csv_sql(contract)} AS src "
        f"{joins} WHERE {where}"
    )
    try:
        result = connection.execute(query, [str(path)]).fetchone()
    except duckdb.Error as error:
        message = f"failed reference checks for {contract.filename}: {error}"
        raise NepProfileError(message) from error
    if result is None:
        raise NepProfileError(f"no reference result for {contract.filename}")
    return NepCheck(name, int(result[0]), severity, description)


def _connect(
    temp_directory: Path | None, *, memory_limit: str = "2GB"
) -> duckdb.DuckDBPyConnection:
    """Create one bounded connection for a single profiling stage."""

    connection = duckdb.connect(":memory:")
    connection.execute("SET enable_progress_bar = false")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute("SET threads = 1")
    connection.execute("SET memory_limit = ?", [memory_limit])
    if temp_directory is not None:
        temp_directory.mkdir(parents=True, exist_ok=True)
        connection.execute("SET temp_directory = ?", [str(temp_directory.resolve())])
        connection.execute("SET max_temp_directory_size = '30GB'")
    return connection


def _isolated_reference_check(
    root: Path,
    temp_directory: Path | None,
    path: Path,
    contract: NepMemberContract,
    *,
    joins: str,
    where: str,
    name: str,
    severity: str,
    description: str,
) -> NepCheck:
    """Run one large reference scan without retaining a prior scan's buffers."""

    # Reference joins need a larger hash-table working set than the pure
    # range scan. Six GiB fits the measured 16 GiB profiling host while still
    # leaving headroom for Windows and the rest of the pipeline.
    connection = _connect(temp_directory, memory_limit="6GB")
    pm_contract = _CONTRACT_BY_NAME["PM.csv"]
    vm_contract = _CONTRACT_BY_NAME["VM.csv"]
    try:
        connection.execute(
            "CREATE TEMP TABLE vm_dim AS "
            "SELECT vm_id, min(site_id) AS site_id, count(*) AS source_rows "
            f"FROM {_read_csv_sql(vm_contract)} "
            "GROUP BY vm_id",
            [str(root / "VM.csv")],
        )
        connection.execute(
            "CREATE TEMP TABLE site_dim AS "
            "SELECT DISTINCT site_id "
            f"FROM {_read_csv_sql(pm_contract)}",
            [str(root / "PM.csv")],
        )
        return _reference_check(
            connection,
            path,
            contract,
            joins=joins,
            where=where,
            name=name,
            severity=severity,
            description=description,
        )
    finally:
        connection.close()


def _resolve_data_root(path: Path) -> Path:
    nested = path / "Full_trace"
    return nested if nested.is_dir() else path


def _validate_file_set(root: Path) -> None:
    expected = {contract.filename for contract in NEP_MEMBER_CONTRACTS}
    present = {path.name for path in root.iterdir() if path.is_file()}
    if present != expected:
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        raise NepProfileError(
            f"NEP extracted file set mismatch; missing={missing}, extra={extra}"
        )


def profile_nep_directory(
    path: Path, *, temp_directory: Path | None = None
) -> NepDatasetProfile:
    """Scan every extracted NEP row and return only aggregate validation evidence."""

    root = _resolve_data_root(path)
    if not root.is_dir():
        raise NepProfileError(f"NEP extracted directory does not exist: {root}")
    _validate_file_set(root)
    for contract in NEP_MEMBER_CONTRACTS:
        if _read_header(root / contract.filename) != contract.source_header:
            raise NepProfileError(f"unexpected header in {contract.filename}")

    connection = _connect(temp_directory)
    pm_contract = _CONTRACT_BY_NAME["PM.csv"]
    vm_contract = _CONTRACT_BY_NAME["VM.csv"]
    try:
        connection.execute(
            "CREATE TEMP TABLE pm_dim AS "
            "SELECT pm_name, min(site_id) AS site_id, count(*) AS source_rows "
            f"FROM {_read_csv_sql(pm_contract)} "
            "GROUP BY pm_name",
            [str(root / "PM.csv")],
        )
        connection.execute(
            "CREATE TEMP TABLE vm_dim AS "
            "SELECT vm_id, min(site_id) AS site_id, count(*) AS source_rows "
            f"FROM {_read_csv_sql(vm_contract)} "
            "GROUP BY vm_id",
            [str(root / "VM.csv")],
        )
        connection.execute(
            "CREATE TEMP TABLE site_dim AS "
            "SELECT DISTINCT site_id FROM pm_dim"
        )

        profiles: tuple[NepMemberProfile, ...] = (
            _profile_member(
                connection,
                root / "PM.csv",
                pm_contract,
                checks=(
                    (
                        "duplicate_pm_id_rows",
                        "count(*) - count(DISTINCT src.pm_name)",
                        "error",
                        "Physical-machine identifiers must be unique.",
                    ),
                ),
            ),
            _profile_member(
                connection,
                root / "VM.csv",
                vm_contract,
                joins="LEFT JOIN pm_dim pm ON src.pm_name = pm.pm_name",
                checks=(
                    (
                        "duplicate_vm_id_rows",
                        "count(*) - count(DISTINCT src.vm_id)",
                        "error",
                        "VM identifiers must be unique.",
                    ),
                    (
                        "unknown_pm_rows",
                        "count_if(pm.pm_name IS NULL)",
                        "error",
                        "Every VM must reference a known physical machine.",
                    ),
                    (
                        "pm_site_mismatch_rows",
                        "count_if(pm.pm_name IS NOT NULL "
                        "AND src.site_id <> pm.site_id)",
                        "error",
                        "A VM site must match its physical machine site.",
                    ),
                    (
                        "zero_start_time_rows",
                        "count_if(try_cast(src.start_time AS BIGINT) = 0)",
                        "info",
                        "Zero start times are retained as source sentinels, not dates.",
                    ),
                    (
                        "end_before_start_rows",
                        "count_if(try_cast(src.end_time AS BIGINT) < "
                        "try_cast(src.start_time AS BIGINT))",
                        "info",
                        "Source lifecycle values requiring replay-policy handling.",
                    ),
                ),
            ),
            _profile_member(
                connection,
                root / "VM_CPU.csv",
                _CONTRACT_BY_NAME["VM_CPU.csv"],
            ),
            _profile_member(
                connection,
                root / "VM_BANDWIDTH.csv",
                _CONTRACT_BY_NAME["VM_BANDWIDTH.csv"],
            ),
            _profile_member(
                connection,
                root / "VM_BW_THREE_MONTHS.csv",
                _CONTRACT_BY_NAME["VM_BW_THREE_MONTHS.csv"],
            ),
            _profile_member(
                connection,
                root / "SITE_RTT.csv",
                _CONTRACT_BY_NAME["SITE_RTT.csv"],
                checks=(
                    (
                        "self_pair_rows",
                        "count_if(src.from_site_id = src.to_site_id)",
                        "info",
                        "Self-site RTT rows are disclosed separately.",
                    ),
                ),
            ),
        )

        # DuckDB can retain CSV buffers after the 63 GB range scan. A fresh
        # connection keeps the subsequent hash/semi-join checks bounded; this
        # is materially different from merely clearing Python references.
        connection.close()
        reference_checks = {
            "VM_CPU.csv": (
                _isolated_reference_check(
                    root,
                    temp_directory,
                    root / "VM_CPU.csv",
                    _CONTRACT_BY_NAME["VM_CPU.csv"],
                    joins="ANTI JOIN vm_dim vm USING (vm_id)",
                    where="true",
                    name="unknown_vm_rows",
                    severity="error",
                    description="Every CPU observation must reference a known VM.",
                ),
                _isolated_reference_check(
                    root,
                    temp_directory,
                    root / "VM_CPU.csv",
                    _CONTRACT_BY_NAME["VM_CPU.csv"],
                    joins="JOIN vm_dim vm USING (vm_id)",
                    where="src.site_id <> vm.site_id",
                    name="vm_site_mismatch_rows",
                    severity="error",
                    description=(
                        "CPU observation site must match the VM affiliation table."
                    ),
                ),
            ),
            "VM_BANDWIDTH.csv": (
                _isolated_reference_check(
                    root,
                    temp_directory,
                    root / "VM_BANDWIDTH.csv",
                    _CONTRACT_BY_NAME["VM_BANDWIDTH.csv"],
                    joins="ANTI JOIN vm_dim vm USING (vm_id)",
                    where="true",
                    name="unknown_vm_rows",
                    severity="error",
                    description=(
                        "Every bandwidth observation must reference a known VM."
                    ),
                ),
                _isolated_reference_check(
                    root,
                    temp_directory,
                    root / "VM_BANDWIDTH.csv",
                    _CONTRACT_BY_NAME["VM_BANDWIDTH.csv"],
                    joins="JOIN vm_dim vm USING (vm_id)",
                    where="src.site_id <> vm.site_id",
                    name="vm_site_mismatch_rows",
                    severity="error",
                    description=(
                        "Bandwidth observation site must match VM affiliation."
                    ),
                ),
            ),
            "VM_BW_THREE_MONTHS.csv": (
                _isolated_reference_check(
                    root,
                    temp_directory,
                    root / "VM_BW_THREE_MONTHS.csv",
                    _CONTRACT_BY_NAME["VM_BW_THREE_MONTHS.csv"],
                    joins="ANTI JOIN vm_dim vm USING (vm_id)",
                    where="true",
                    name="unknown_vm_rows",
                    severity="error",
                    description=(
                        "Every three-month bandwidth row must reference a known VM."
                    ),
                ),
            ),
            "SITE_RTT.csv": (
                _isolated_reference_check(
                    root,
                    temp_directory,
                    root / "SITE_RTT.csv",
                    _CONTRACT_BY_NAME["SITE_RTT.csv"],
                    joins=(
                        "ANTI JOIN site_dim src_site "
                        "ON src.from_site_id = src_site.site_id"
                    ),
                    where="true",
                    name="unknown_from_site_rows",
                    severity="error",
                    description="Every RTT source must reference a known site.",
                ),
                _isolated_reference_check(
                    root,
                    temp_directory,
                    root / "SITE_RTT.csv",
                    _CONTRACT_BY_NAME["SITE_RTT.csv"],
                    joins=(
                        "ANTI JOIN site_dim dst_site "
                        "ON src.to_site_id = dst_site.site_id"
                    ),
                    where="true",
                    name="unknown_to_site_rows",
                    severity="error",
                    description="Every RTT destination must reference a known site.",
                ),
            ),
        }
        profiles = tuple(
            replace(
                profile,
                checks=profile.checks + reference_checks.get(profile.filename, ()),
            )
            for profile in profiles
        )
    finally:
        connection.close()
    return NepDatasetProfile(root.name, profiles)


def _accepted_dimensions(
    connection: duckdb.DuckDBPyConnection, root: Path
) -> None:
    pm_contract = _CONTRACT_BY_NAME["PM.csv"]
    vm_contract = _CONTRACT_BY_NAME["VM.csv"]
    pm_invalid = " OR ".join(
        _invalid_expression(column) for column in pm_contract.columns
    )
    vm_invalid = " OR ".join(
        _invalid_expression(column) for column in vm_contract.columns
    )
    connection.execute(
        "CREATE TEMP TABLE accepted_pm AS "
        "SELECT src.pm_name, src.site_id "
        f"FROM {_read_csv_sql(pm_contract)} AS src "
        f"WHERE NOT ({pm_invalid})",
        [str(root / "PM.csv")],
    )
    connection.execute(
        "CREATE TEMP TABLE accepted_vm AS "
        "SELECT src.vm_id, src.site_id "
        f"FROM {_read_csv_sql(vm_contract)} AS src "
        "JOIN accepted_pm pm ON src.pm_name = pm.pm_name "
        f"WHERE NOT ({vm_invalid}) AND src.site_id = pm.site_id",
        [str(root / "VM.csv")],
    )
    connection.execute(
        "CREATE TEMP TABLE accepted_site AS "
        "SELECT DISTINCT site_id FROM accepted_pm"
    )


def profile_nep_accepted_rows(
    path: Path,
    filename: str,
    *,
    temp_directory: Path | None = None,
) -> int:
    """Count rows satisfying the frozen value and reference acceptance policy."""

    root = _resolve_data_root(path)
    if not root.is_dir():
        raise NepProfileError(f"NEP extracted directory does not exist: {root}")
    _validate_file_set(root)
    try:
        contract = _CONTRACT_BY_NAME[filename]
    except KeyError as error:
        raise NepProfileError(f"unknown NEP member: {filename}") from error
    if _read_header(root / filename) != contract.source_header:
        raise NepProfileError(f"unexpected header in {filename}")

    connection = _connect(temp_directory, memory_limit="6GB")
    try:
        _accepted_dimensions(connection, root)
        invalid = " OR ".join(
            _invalid_expression(column) for column in contract.columns
        )
        joins = ""
        reference_condition = "true"
        if filename in {"VM_CPU.csv", "VM_BANDWIDTH.csv"}:
            joins = "JOIN accepted_vm vm USING (vm_id)"
            reference_condition = "src.site_id = vm.site_id"
        elif filename == "VM_BW_THREE_MONTHS.csv":
            joins = "JOIN accepted_vm vm USING (vm_id)"
        elif filename == "SITE_RTT.csv":
            joins = (
                "JOIN accepted_site src_site "
                "ON src.from_site_id = src_site.site_id "
                "JOIN accepted_site dst_site "
                "ON src.to_site_id = dst_site.site_id"
            )
        elif filename == "VM.csv":
            joins = (
                "JOIN accepted_pm pm ON src.pm_name = pm.pm_name "
                "AND src.site_id = pm.site_id"
            )

        query = (
            f"SELECT count(*) FROM {_read_csv_sql(contract)} AS src "
            f"{joins} WHERE NOT ({invalid}) AND {reference_condition}"
        )
        result = connection.execute(query, [str(root / filename)]).fetchone()
        if result is None:
            raise NepProfileError(f"no acceptance result for {filename}")
        return int(result[0])
    except duckdb.Error as error:
        raise NepProfileError(
            f"failed acceptance count for {filename}: {error}"
        ) from error
    finally:
        connection.close()
