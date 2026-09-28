"""Data provenance, normalization, checksum, and split contracts."""

from task_offloading.data.bupt import (
    BuptParseError,
    BuptTraceRecord,
    parse_bupt_line,
)
from task_offloading.data.eua import (
    EuaEntityKind,
    EuaLocationRecord,
    EuaParseError,
    iter_eua_archive,
)
from task_offloading.data.manifest import (
    assert_status_transition,
    checksum_matches,
    load_and_validate_manifest,
    sha256_file,
    source_is_analysis_ready,
    validate_manifest_semantics,
)
from task_offloading.data.nep import (
    NEP_ACCEPTANCE_POLICY_VERSION,
    NepProfileError,
    profile_nep_accepted_rows,
    profile_nep_directory,
)
from task_offloading.data.normalization import (
    StandardizationStats,
    TrainOnlyStandardizer,
)
from task_offloading.data.splitting import (
    assert_no_leakage,
    audit_split,
    chronological_split,
    content_sha256,
    group_holdout_split,
)
from task_offloading.data.types import (
    BenchmarkKind,
    DataSample,
    DataSplit,
    LeakageReport,
    ProvenanceKind,
    SplitAssignment,
    SplitPlan,
    SplitStrategy,
)
from task_offloading.data.uci_mec import (
    UciMecExecutionRecord,
    UciMecParseError,
    iter_uci_mec_archive,
)

__all__ = [
    "NEP_ACCEPTANCE_POLICY_VERSION",
    "BenchmarkKind",
    "BuptParseError",
    "BuptTraceRecord",
    "DataSample",
    "DataSplit",
    "EuaEntityKind",
    "EuaLocationRecord",
    "EuaParseError",
    "LeakageReport",
    "NepProfileError",
    "ProvenanceKind",
    "SplitAssignment",
    "SplitPlan",
    "SplitStrategy",
    "StandardizationStats",
    "TrainOnlyStandardizer",
    "UciMecExecutionRecord",
    "UciMecParseError",
    "assert_no_leakage",
    "assert_status_transition",
    "audit_split",
    "checksum_matches",
    "chronological_split",
    "content_sha256",
    "group_holdout_split",
    "iter_eua_archive",
    "iter_uci_mec_archive",
    "load_and_validate_manifest",
    "parse_bupt_line",
    "profile_nep_accepted_rows",
    "profile_nep_directory",
    "sha256_file",
    "source_is_analysis_ready",
    "validate_manifest_semantics",
]
