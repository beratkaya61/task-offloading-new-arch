"""Offline form generation and release gates for human annotation.

The generated form contains only the already-blinded packet.  It has no
network dependency and never receives corpus authoring metadata, another
annotator's answers, or model output.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from typing import Any

from task_offloading.semantic.benchmark import (
    ANNOTATION_FIELDS,
    ANNOTATOR_IDS,
    PACKET_KEYS,
)

_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_SUBMISSION_SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
_PAYLOAD_MARKER = "__ANNOTATION_FORM_PAYLOAD__"


@dataclass(frozen=True, slots=True)
class ReleaseAudit:
    """Human-review status that must pass before packets are distributed."""

    item_count: int
    review_status_counts: tuple[tuple[str, int], ...]
    issues: tuple[str, ...]

    @property
    def is_ready(self) -> bool:
        return not self.issues


def audit_annotation_release(
    corpus: Sequence[Mapping[str, Any]],
) -> ReleaseAudit:
    """Require an explicit human text review before annotation starts."""

    counts: Counter[str] = Counter()
    issues: list[str] = []
    for index, item in enumerate(corpus, start=1):
        provenance = item.get("provenance")
        if not isinstance(provenance, Mapping):
            issues.append(f"corpus record {index} has no provenance object")
            continue
        status = provenance.get("human_text_review")
        if not isinstance(status, str):
            issues.append(f"corpus record {index} has no human_text_review status")
            continue
        counts[status] += 1

    pending = counts["pending"]
    rejected = counts["rejected"]
    approved = counts["approved"]
    if pending:
        issues.append(f"{pending} corpus texts still await human review")
    if rejected:
        issues.append(f"{rejected} corpus texts were rejected by human review")
    if approved != len(corpus):
        issues.append(
            "every corpus text must have human_text_review=approved before release"
        )

    return ReleaseAudit(
        item_count=len(corpus),
        review_status_counts=tuple(sorted(counts.items())),
        issues=tuple(issues),
    )


def _validate_form_inputs(
    packet: Sequence[Mapping[str, Any]],
    *,
    annotator_id: str,
    packet_sha256: str,
) -> None:
    if annotator_id not in ANNOTATOR_IDS:
        raise ValueError(f"unsupported annotator id: {annotator_id}")
    if not _SHA256_PATTERN.fullmatch(packet_sha256):
        raise ValueError("packet_sha256 must be a lowercase SHA-256 value")
    if not packet:
        raise ValueError("annotation packet cannot be empty")

    for index, row in enumerate(packet, start=1):
        if set(row) != PACKET_KEYS:
            raise ValueError(
                f"packet row {index} does not contain the exact blind field set"
            )
        if row.get("annotator_id") != annotator_id:
            raise ValueError(f"packet row {index} has the wrong annotator id")
        if row.get("presentation_order") != index:
            raise ValueError(f"packet row {index} has non-contiguous order")
    packet_versions = {row.get("packet_version") for row in packet}
    protocol_versions = {row.get("protocol_version") for row in packet}
    if len(packet_versions) != 1 or not isinstance(next(iter(packet_versions)), str):
        raise ValueError("packet rows must share one packet version")
    if len(protocol_versions) != 1 or not isinstance(
        next(iter(protocol_versions)), str
    ):
        raise ValueError("packet rows must share one protocol version")


def _validate_initial_responses(
    packet: Sequence[Mapping[str, Any]],
    initial_responses: Mapping[str, Any],
) -> None:
    task_ids = {str(row["task_id"]) for row in packet}
    unknown_tasks = set(initial_responses) - task_ids
    if unknown_tasks:
        raise ValueError(
            "initial responses contain unknown task ids: "
            + ", ".join(sorted(unknown_tasks))
        )

    for task_id, fields in initial_responses.items():
        if not isinstance(fields, Mapping):
            raise ValueError(f"initial response for {task_id} must be an object")
        unknown_fields = set(fields) - set(ANNOTATION_FIELDS)
        if unknown_fields:
            raise ValueError(
                f"initial response for {task_id} contains unknown fields: "
                + ", ".join(sorted(unknown_fields))
            )
        for field_name, field in fields.items():
            if not isinstance(field, Mapping):
                raise ValueError(
                    f"initial response {task_id}.{field_name} must be an object"
                )


def render_annotation_form(
    packet: Sequence[Mapping[str, Any]],
    *,
    annotator_id: str,
    packet_sha256: str,
    initial_responses: Mapping[str, Any] | None = None,
    submission_slug: str | None = None,
) -> str:
    """Render a deterministic offline form, optionally with an annotator draft."""

    _validate_form_inputs(
        packet,
        annotator_id=annotator_id,
        packet_sha256=packet_sha256,
    )
    responses = initial_responses or {}
    _validate_initial_responses(packet, responses)
    packet_version = str(packet[0]["packet_version"])
    protocol_version = str(packet[0]["protocol_version"])
    if submission_slug is not None and not _SUBMISSION_SLUG_PATTERN.fullmatch(
        submission_slug
    ):
        raise ValueError("submission_slug must be a safe lowercase filename stem")
    payload = {
        "form_version": "1.2.0" if responses else "1.1.0",
        "record_version": "1.0.0",
        "packet_version": packet_version,
        "protocol_version": protocol_version,
        "schema_version": "1.0.0",
        "annotator_id": annotator_id,
        "form_mode": "assisted_review" if responses else "blind_annotation",
        "packet_sha256": packet_sha256,
        "submission_slug": submission_slug,
        "tasks": [dict(row) for row in packet],
        "initial_responses": responses,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).replace("<", "\\u003c")

    template = (
        resources.files("task_offloading.semantic")
        .joinpath("pilot_annotation_form.html")
        .read_text(encoding="utf-8")
    )
    if template.count(_PAYLOAD_MARKER) != 1:
        raise RuntimeError("annotation form template has an invalid payload marker")
    return template.replace(_PAYLOAD_MARKER, serialized)
