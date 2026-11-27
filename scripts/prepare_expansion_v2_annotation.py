# ruff: noqa: E501
"""Prepare the blinded two-annotator packets for the expansion v2 candidate.

Lean design (decided 2026-11-27, two annotators):

* validation/test: every family, 2 of 4 semantic profiles (seeded), both
  wording views -> 384 + 384 rows; every held-out subject is kept;
* training audit: 48 domain-stratified train families x 2 profiles x 2 views;
* pilot: 40 train rows from families outside the audit, identical for A and B;
* hidden repeats: 24 already-assigned rows re-shown in late sessions under a
  new opaque id to measure intra-annotator consistency;
* annotator B additionally receives the locked Human-60 under protocol 1.3.

Annotators only receive the offline HTML forms.  Task ids in packets are
opaque (``v2x_NNNN``); the blinding key that maps them to split, family and
wording view stays with the coordinator.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from task_offloading.data import sha256_file
from task_offloading.semantic import (
    build_blind_packet,
    load_jsonl,
    packet_issues,
    render_annotation_form,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "semantic_benchmark" / "expanded_v2"
OUT = DATA / "annotation"
FORMS = ROOT / "artifacts" / "semantic_annotation" / "expanded_v2"
MAIN = ROOT / "data" / "semantic_benchmark" / "main"
PROTOCOL = "2.0.0"
PACKET_VERSION = "2.0.0"
SEED = 20261128
ANNOTATORS = ("annotator_a", "annotator_b")
PROFILES_PER_HELDOUT_FAMILY = 2
AUDIT_FAMILIES_PER_DOMAIN = 6
PILOT_ROWS_PER_DOMAIN = 5
SESSIONS = 7
HIDDEN_REPEATS = 24


def _rng(*parts: object) -> random.Random:
    digest = hashlib.sha256("|".join(map(str, (SEED, *parts))).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _select(tasks: list[dict[str, Any]], assignments: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (main rows with role, pilot rows)."""

    by_family: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for task in tasks:
        family = str(task["grouping"]["scenario_family_id"])
        by_family[family][str(task["grouping"]["semantic_profile_id"])].append(task)

    def annotated_views(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Canonical plus the natural view; the template paraphrase is the fallback."""

        by_view = {str(r["design_strata"]["wording_view"]): r for r in rows}
        second = by_view.get("natural") or by_view["paraphrase"]
        return [by_view["canonical"], second]

    def two_profiles(family: str) -> list[dict[str, Any]]:
        profiles = sorted(by_family[family])
        chosen = sorted(_rng("profiles", family).sample(profiles, PROFILES_PER_HELDOUT_FAMILY))
        return [row for profile in chosen for row in annotated_views(by_family[family][profile])]

    domain_of = {str(t["grouping"]["scenario_family_id"]): str(t["design_strata"]["scenario_domain"]) for t in tasks}
    train_by_domain: dict[str, list[str]] = defaultdict(list)
    for family, split in sorted(assignments.items()):
        if split == "train":
            train_by_domain[domain_of[family]].append(family)

    main: list[dict[str, Any]] = []
    for family, split in sorted(assignments.items()):
        if split in {"validation", "test"}:
            main.extend({**row, "_role": f"{split}_reference"} for row in two_profiles(family))
    pilot: list[dict[str, Any]] = []
    for domain in sorted(train_by_domain):
        families = list(train_by_domain[domain])
        _rng("train_order", domain).shuffle(families)
        audit = families[:AUDIT_FAMILIES_PER_DOMAIN]
        pilot_families = families[AUDIT_FAMILIES_PER_DOMAIN : AUDIT_FAMILIES_PER_DOMAIN + PILOT_ROWS_PER_DOMAIN]
        for family in audit:
            main.extend({**row, "_role": "training_audit"} for row in two_profiles(family))
        for family in pilot_families:
            # Pilot uses only the frozen template views so later wording additions
            # (e.g. LLM natural rewrites) never change an already released pilot.
            rows = [
                row
                for profile in sorted(by_family[family])
                for row in sorted(by_family[family][profile], key=lambda r: r["task_id"])
                if row["design_strata"]["wording_view"] in {"canonical", "paraphrase"}
            ]
            pilot.append({**_rng("pilot", family).choice(rows), "_role": "pilot"})
    return main, pilot


def main() -> None:
    tasks = list(load_jsonl(DATA / "candidates.tasks.jsonl"))
    split_manifest = json.loads((DATA / "split_manifest.candidate.json").read_text(encoding="utf-8"))
    assignments = dict(split_manifest["family_assignments"])
    main_rows, pilot_rows = _select(tasks, assignments)

    # Session assignment is shared by A and B (enables per-session agreement QA);
    # the presentation order inside a session differs per annotator.
    order = list(main_rows)
    _rng("sessions").shuffle(order)
    sessions: list[list[dict[str, Any]]] = [order[i::SESSIONS] for i in range(SESSIONS)]
    early = [row for session in sessions[:2] for row in session]
    repeats = _rng("repeats").sample(early, HIDDEN_REPEATS)
    for index, row in enumerate(repeats):
        sessions[SESSIONS - 2 + index % 2].append({**row, "_role": "hidden_repeat", "_repeat_of": row["task_id"]})

    # Separate alias spaces: the pilot (v2p_) is independent of the main pool (v2x_).
    pilot_entries: list[tuple[str, dict[str, Any]]] = [("pilot", row) for row in pilot_rows]
    main_entries = [(f"session_{i + 1:02d}", row) for i, session in enumerate(sessions) for row in session]
    pilot_numbers = list(range(1, len(pilot_entries) + 1))
    _rng("pilot_aliases").shuffle(pilot_numbers)
    main_numbers = list(range(1, len(main_entries) + 1))
    _rng("aliases").shuffle(main_numbers)
    entries = pilot_entries + main_entries
    aliases = [f"v2p_{n:04d}" for n in pilot_numbers] + [f"v2x_{n:04d}" for n in main_numbers]
    key: list[dict[str, Any]] = []
    blinded: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (session, row), alias in zip(entries, aliases, strict=True):
        blinded[session].append(
            {"task_id": alias, "language": row["language"], "task_text": row["task_text"], "user_policy_text": row["user_policy_text"]}
        )
        key.append(
            {
                "alias": alias,
                "session": session,
                "role": row["_role"],
                "task_id": row["task_id"],
                "repeat_of_task_id": row.get("_repeat_of"),
                "split": row["design_strata"]["split"],
                "scenario_family_id": row["grouping"]["scenario_family_id"],
                "subject_group_id": row["grouping"]["subject_group_id"],
                "wording_view": row["design_strata"]["wording_view"],
            }
        )

    packet_index: dict[str, Any] = {}
    for annotator in ANNOTATORS:
        for session, corpus in sorted(blinded.items()):
            packet = build_blind_packet(
                corpus,
                annotator_id=annotator,
                seed=_rng("packet", annotator, session).randrange(2**31),
                packet_version=PACKET_VERSION,
                protocol_version=PROTOCOL,
            )
            issues = packet_issues(packet, corpus, annotator_id=annotator, packet_version=PACKET_VERSION, protocol_version=PROTOCOL)
            if issues:
                raise ValueError(f"{annotator}/{session} packet audit failed: {issues}")
            packet_path = OUT / "packets" / f"{annotator}.{session}.jsonl"
            write_jsonl(packet_path, packet)
            slug = f"{annotator}.expanded_v2.{session.replace('_', '-')}"
            form_path = FORMS / annotator / f"{session}.html"
            form_path.parent.mkdir(parents=True, exist_ok=True)
            form_path.write_text(
                render_annotation_form(packet, annotator_id=annotator, packet_sha256=sha256_file(packet_path), submission_slug=slug, collect_naturalness=True),
                encoding="utf-8",
            )
            packet_index[f"{annotator}/{session}"] = {
                "packet": packet_path.relative_to(ROOT).as_posix(),
                "packet_sha256": sha256_file(packet_path),
                "form_local_ignored_path": form_path.relative_to(ROOT).as_posix(),
                "task_count": len(packet),
            }

    # Annotator B: second independent annotation of the locked Human-60 (protocol 1.3).
    split = json.loads((MAIN / "split_manifest.v1.json").read_text(encoding="utf-8"))
    evaluation_ids = set(split["single_human_evaluation"]["task_ids"])
    human60 = [t for t in load_jsonl(MAIN / "tasks.v1.jsonl") if t["task_id"] in evaluation_ids]
    h60_packet = build_blind_packet(human60, annotator_id="annotator_b", seed=42002, packet_version="1.3.0", protocol_version="1.3.0")
    h60_issues = packet_issues(h60_packet, human60, annotator_id="annotator_b", packet_version="1.3.0", protocol_version="1.3.0")
    if h60_issues:
        raise ValueError(f"Human-60 B packet audit failed: {h60_issues}")
    h60_path = MAIN / "packets" / "annotator_b.evaluation.v1.3.jsonl"
    write_jsonl(h60_path, h60_packet)
    h60_form = FORMS / "annotator_b" / "human60_second_annotation.html"
    h60_form.write_text(
        render_annotation_form(h60_packet, annotator_id="annotator_b", packet_sha256=sha256_file(h60_path), submission_slug="annotator_b.blind_evaluation.protocol-1.3.0"),
        encoding="utf-8",
    )

    write_jsonl(OUT / "blinding_key.coordinator_only.jsonl", sorted(key, key=lambda r: r["alias"]))
    roles: dict[str, int] = defaultdict(int)
    for row in key:
        roles[row["role"]] += 1
    per_annotator = len(key)
    _json(
        OUT / "annotation_plan.v2.json",
        {
            "protocol_version": PROTOCOL,
            "created_on": "2026-11-27",
            "seed": SEED,
            "annotators": list(ANNOTATORS),
            "design": __doc__,
            "role_counts": dict(sorted(roles.items())),
            "unique_main_tasks": len(main_rows),
            "tasks_per_annotator_including_pilot_and_repeats": per_annotator,
            "annotator_b_extra_human60_tasks": len(h60_packet),
            "sessions": {name: len(rows) for name, rows in sorted(blinded.items())},
            "blinding_key": (OUT / "blinding_key.coordinator_only.jsonl").relative_to(ROOT).as_posix(),
            "blinding_key_sha256": sha256_file(OUT / "blinding_key.coordinator_only.jsonl"),
            "packets": packet_index,
            "human60_annotator_b": {
                "packet": h60_path.relative_to(ROOT).as_posix(),
                "packet_sha256": sha256_file(h60_path),
                "form_local_ignored_path": h60_form.relative_to(ROOT).as_posix(),
                "task_count": len(h60_packet),
            },
            "record_schema": "schemas/human_annotation_record.v2.schema.json",
            "never_share_with_annotators": ["blinding_key.coordinator_only.jsonl", "candidates.construction_labels.jsonl", "split_manifest.candidate.json"],
        },
    )
    print(f"roles={dict(sorted(roles.items()))} per_annotator={per_annotator} sessions={ {k: len(v) for k, v in sorted(blinded.items())} }")
    print(f"forms={FORMS}")


if __name__ == "__main__":
    main()
