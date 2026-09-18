"""Deterministic and leakage-aware semantic benchmark primitives.

The corpus contains authoring metadata, but annotators receive only a strict
allow-list of fields. Human annotations are validated against both the frozen
JSON schema and the exact task text before they can become gold candidates.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

ANNOTATION_FIELDS = (
    "domain",
    "latency",
    "privacy",
    "execution_policy",
    "reliability",
    "accuracy",
    "energy_priority",
    "divisibility",
)
ANNOTATOR_IDS = frozenset({"annotator_a", "annotator_b"})
PACKET_KEYS = frozenset(
    {
        "packet_version",
        "protocol_version",
        "annotator_id",
        "presentation_order",
        "task_id",
        "language",
        "task_text",
        "user_policy_text",
        "task_content_sha256",
    }
)
_SENSITIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("url", re.compile(r"(?:https?://|www\.)", re.IGNORECASE)),
    ("email", re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")),
    ("ipv4", re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")),
    ("imei_or_imsi", re.compile(r"\b(?:imei|imsi)\b", re.IGNORECASE)),
    (
        "mac_address",
        re.compile(r"\b(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}\b", re.IGNORECASE),
    ),
)


@dataclass(frozen=True, slots=True)
class CorpusAudit:
    """Evidence about pilot size, balance, uniqueness, and privacy hygiene."""

    item_count: int
    origin_counts: tuple[tuple[str, int], ...]
    domain_stratum_counts: tuple[tuple[str, int], ...]
    phenomenon_counts: tuple[tuple[str, int], ...]
    issues: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.issues


@dataclass(frozen=True, slots=True)
class SubmissionAudit:
    """Validation result for one actual human's completed annotation file."""

    annotator_id: str
    record_count: int
    issues: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.issues


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def corpus_task_sha256(task: Mapping[str, Any]) -> str:
    """Bind an annotation to exactly the text and policy shown to the human."""

    visible = {
        "task_id": task["task_id"],
        "language": task["language"],
        "task_text": task["task_text"],
        "user_policy_text": task["user_policy_text"],
    }
    return hashlib.sha256(_canonical_json(visible)).hexdigest()


def load_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    """Load non-empty JSONL records and identify the exact failing line."""

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL line at {path}:{line_number}")
            try:
                value: Any = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {error.msg}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"JSONL record must be an object at {path}:{line_number}"
                )
            records.append(value)
    if not records:
        raise ValueError(f"JSONL file is empty: {path}")
    return tuple(records)


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Write canonical, UTF-8 JSONL for stable checksums."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            serialized = json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write(f"{serialized}\n")


def _normalized_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def _schema_issues(
    records: Sequence[Mapping[str, Any]], schema: Mapping[str, Any], label: str
) -> list[str]:
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    issues: list[str] = []
    for index, record in enumerate(records, start=1):
        for error in validator.iter_errors(record):
            location = ".".join(str(item) for item in error.absolute_path) or "$"
            issues.append(f"{label} {index} at {location}: {error.message}")
    return issues


def audit_corpus(
    records: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
    *,
    expected_count: int = 20,
) -> CorpusAudit:
    """Audit a pilot without treating design strata as human gold labels."""

    issues = _schema_issues(records, schema, "corpus record")
    task_ids: list[str] = []
    normalized_texts: list[str] = []
    origin_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    phenomenon_counts: Counter[str] = Counter()

    for index, record in enumerate(records, start=1):
        task_id = record.get("task_id")
        task_text = record.get("task_text")
        if isinstance(task_id, str):
            task_ids.append(task_id)
        if isinstance(task_text, str):
            policy = record.get("user_policy_text")
            combined = f"{task_text}\n{policy if isinstance(policy, str) else ''}"
            normalized_texts.append(_normalized_text(combined))
            for pattern_name, pattern in _SENSITIVE_PATTERNS:
                if pattern.search(combined):
                    issues.append(
                        f"corpus record {index} contains forbidden {pattern_name}"
                    )

        provenance = record.get("provenance")
        if isinstance(provenance, dict):
            origin = provenance.get("origin")
            if isinstance(origin, str):
                origin_counts[origin] += 1

        strata = record.get("design_strata")
        if isinstance(strata, dict):
            domain = strata.get("scenario_domain")
            if isinstance(domain, str):
                domain_counts[domain] += 1
            phenomena = strata.get("semantic_phenomena")
            if isinstance(phenomena, list):
                phenomenon_counts.update(
                    item for item in phenomena if isinstance(item, str)
                )

    if len(records) != expected_count:
        issues.append(f"expected {expected_count} records, found {len(records)}")
    if len(task_ids) != len(set(task_ids)):
        issues.append("task_id values must be unique")
    if len(normalized_texts) != len(set(normalized_texts)):
        issues.append("normalized task+policy texts must be unique")

    required_domains = {
        "healthcare",
        "industrial",
        "transportation",
        "public_safety",
        "smart_city",
        "agriculture",
        "consumer",
        "other",
    }
    underrepresented = sorted(
        domain for domain in required_domains if domain_counts[domain] < 2
    )
    if underrepresented:
        issues.append(
            "every scenario-domain stratum needs at least two pilot items: "
            + ", ".join(underrepresented)
        )

    required_phenomena = {
        "ambiguity",
        "explicit_accuracy",
        "explicit_divisibility",
        "explicit_energy",
        "explicit_execution_policy",
        "explicit_latency",
        "explicit_privacy",
        "explicit_reliability",
        "missing_information",
        "negation",
        "numeric_threshold",
    }
    missing_phenomena = sorted(required_phenomena - phenomenon_counts.keys())
    if missing_phenomena:
        issues.append(
            "pilot lacks required semantic phenomena: "
            + ", ".join(missing_phenomena)
        )

    return CorpusAudit(
        item_count=len(records),
        origin_counts=tuple(sorted(origin_counts.items())),
        domain_stratum_counts=tuple(sorted(domain_counts.items())),
        phenomenon_counts=tuple(sorted(phenomenon_counts.items())),
        issues=tuple(issues),
    )


def build_blind_packet(
    corpus: Sequence[Mapping[str, Any]], *, annotator_id: str, seed: int
) -> tuple[dict[str, Any], ...]:
    """Project corpus rows onto the only fields an independent human may see."""

    if annotator_id not in ANNOTATOR_IDS:
        raise ValueError(f"unsupported annotator id: {annotator_id}")
    if seed < 0:
        raise ValueError("seed must be >= 0")
    ordered = sorted(
        corpus,
        key=lambda item: hashlib.sha256(
            f"{seed}:{annotator_id}:{item['task_id']}".encode()
        ).hexdigest(),
    )
    return tuple(
        {
            "packet_version": "1.0.0",
            "protocol_version": "1.0.0",
            "annotator_id": annotator_id,
            "presentation_order": index,
            "task_id": item["task_id"],
            "language": item["language"],
            "task_text": item["task_text"],
            "user_policy_text": item["user_policy_text"],
            "task_content_sha256": corpus_task_sha256(item),
        }
        for index, item in enumerate(ordered, start=1)
    )


def packet_issues(
    packet: Sequence[Mapping[str, Any]],
    corpus: Sequence[Mapping[str, Any]],
    *,
    annotator_id: str,
) -> tuple[str, ...]:
    """Detect missing tasks, ordering drift, and authoring/gold leakage."""

    issues: list[str] = []
    corpus_by_id = {str(item["task_id"]): item for item in corpus}
    seen_ids: list[str] = []
    for index, row in enumerate(packet, start=1):
        extra = set(row) - PACKET_KEYS
        missing = PACKET_KEYS - set(row)
        if extra:
            issues.append(f"packet row {index} leaks fields: {sorted(extra)}")
        if missing:
            issues.append(f"packet row {index} misses fields: {sorted(missing)}")
            continue
        if row["annotator_id"] != annotator_id:
            issues.append(f"packet row {index} has wrong annotator id")
        if row["presentation_order"] != index:
            issues.append(f"packet row {index} has non-contiguous order")
        task_id = str(row["task_id"])
        seen_ids.append(task_id)
        source = corpus_by_id.get(task_id)
        if source is None:
            issues.append(f"packet row {index} references unknown task {task_id}")
            continue
        for key in ("language", "task_text", "user_policy_text"):
            if row[key] != source[key]:
                issues.append(f"packet row {index} changed {key}")
        if row["task_content_sha256"] != corpus_task_sha256(source):
            issues.append(f"packet row {index} has a stale task hash")
    if len(seen_ids) != len(set(seen_ids)):
        issues.append("packet contains duplicate task ids")
    if set(seen_ids) != set(corpus_by_id):
        issues.append("packet task ids do not match the corpus")
    return tuple(issues)


def _evidence_issues(
    annotation: Mapping[str, Any], source: Mapping[str, Any], task_id: str
) -> list[str]:
    issues: list[str] = []
    task_text = str(source["task_text"])
    policy = source.get("user_policy_text")
    visible_text = f"{task_text}\n{policy if isinstance(policy, str) else ''}"
    for field in ANNOTATION_FIELDS:
        label = annotation.get(field)
        if not isinstance(label, dict):
            continue
        evidence = label.get("evidence")
        if not isinstance(evidence, list):
            continue
        for snippet in evidence:
            if isinstance(snippet, str) and snippet not in visible_text:
                issues.append(
                    f"{task_id}.{field} evidence is not an exact visible substring"
                )
    return issues


def _abstention_issues(annotation: Mapping[str, Any], task_id: str) -> list[str]:
    expected: set[str] = set()
    for field in ANNOTATION_FIELDS:
        label = annotation.get(field)
        if isinstance(label, dict) and label.get("status") in {
            "not_stated",
            "ambiguous",
        }:
            expected.add(field)
    actual = annotation.get("abstained_fields")
    if not isinstance(actual, list) or set(actual) != expected:
        return [
            f"{task_id}.abstained_fields must exactly match not_stated/ambiguous fields"
        ]
    return []


def audit_submission(
    records: Sequence[Mapping[str, Any]],
    corpus: Sequence[Mapping[str, Any]],
    wrapper_schema: Mapping[str, Any],
    semantic_schema: Mapping[str, Any],
    *,
    annotator_id: str,
) -> SubmissionAudit:
    """Validate a completed independent-human submission before it is locked."""

    issues = _schema_issues(records, wrapper_schema, "submission record")
    corpus_by_id = {str(item["task_id"]): item for item in corpus}
    seen: list[str] = []
    semantic_records: list[Mapping[str, Any]] = []

    for index, record in enumerate(records, start=1):
        task_id = record.get("task_id")
        if not isinstance(task_id, str):
            continue
        seen.append(task_id)
        source = corpus_by_id.get(task_id)
        if source is None:
            issues.append(
                f"submission record {index} references unknown task {task_id}"
            )
            continue
        if record.get("annotator_id") != annotator_id:
            issues.append(f"submission record {index} has wrong annotator id")
        if record.get("task_content_sha256") != corpus_task_sha256(source):
            issues.append(f"submission record {index} has a stale task hash")
        annotation = record.get("annotation")
        if not isinstance(annotation, dict):
            continue
        semantic_records.append(annotation)
        if annotation.get("task_id") != task_id:
            issues.append(f"submission record {index} task ids disagree")
        if annotation.get("language") != source["language"]:
            issues.append(f"submission record {index} language disagrees")
        issues.extend(_evidence_issues(annotation, source, task_id))
        issues.extend(_abstention_issues(annotation, task_id))

    issues.extend(_schema_issues(semantic_records, semantic_schema, "annotation"))
    if len(seen) != len(set(seen)):
        issues.append("submission contains duplicate task ids")
    if set(seen) != set(corpus_by_id):
        issues.append("submission task ids do not match the corpus")
    return SubmissionAudit(annotator_id, len(records), tuple(issues))
