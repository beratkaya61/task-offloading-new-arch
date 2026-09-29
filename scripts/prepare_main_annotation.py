"""Prepare A's one-time 60-task blind main-evaluation form."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from task_offloading.data import sha256_file
from task_offloading.semantic import (
    annotation_form_seed,
    audit_submission,
    authoring_preannotation,
    build_blind_packet,
    corpus_task_sha256,
    load_jsonl,
    packet_issues,
    render_annotation_form,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "data" / "semantic_benchmark" / "main"
TASKS = MAIN / "tasks.v1.jsonl"
SPLIT = MAIN / "split_manifest.v1.json"
PACKETS = MAIN / "packets"
PREANNOTATIONS = MAIN / "preannotations" / "machine_development.v1.jsonl"
MANIFEST = MAIN / "annotation_packets.v1.json"
FORMS = ROOT / "artifacts" / "semantic_annotation" / "main"
WRAPPER_SCHEMA = ROOT / "schemas" / "human_annotation_record.schema.json"
SEMANTIC_SCHEMA = ROOT / "schemas" / "semantic_requirements.schema.json"
PROTOCOL_VERSION = "1.3.0"
PACKET_VERSION = "1.3.0"
SEEDS = {
    "annotator_a_evaluation": 42001,
}


def _load_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value: Any = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_bytes(content)
    if temporary.read_bytes() != content:
        raise OSError(f"temporary write verification failed: {temporary}")
    temporary.replace(path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    content = (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_write(path, content)


def _subset(
    corpus: tuple[dict[str, Any], ...], task_ids: list[str]
) -> tuple[dict[str, Any], ...]:
    by_id = {str(task["task_id"]): task for task in corpus}
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("split contains duplicate task ids")
    unknown = set(task_ids) - set(by_id)
    if unknown:
        raise ValueError("split contains unknown task ids: " + ", ".join(unknown))
    return tuple(by_id[task_id] for task_id in task_ids)


def _build_packet(
    corpus: tuple[dict[str, Any], ...], annotator_id: str, seed: int
) -> tuple[dict[str, Any], ...]:
    packet = build_blind_packet(
        corpus,
        annotator_id=annotator_id,
        seed=seed,
        packet_version=PACKET_VERSION,
        protocol_version=PROTOCOL_VERSION,
    )
    issues = packet_issues(
        packet,
        corpus,
        annotator_id=annotator_id,
        packet_version=PACKET_VERSION,
        protocol_version=PROTOCOL_VERSION,
    )
    if issues:
        raise ValueError("main blind packet audit failed:\n- " + "\n- ".join(issues))
    return packet


def _preannotation_records(
    corpus: tuple[dict[str, Any], ...],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    seeds: dict[str, dict[str, Any]] = {}
    human_shaped: list[dict[str, Any]] = []
    for task in corpus:
        annotation = authoring_preannotation(task)
        task_id = str(task["task_id"])
        task_hash = corpus_task_sha256(task)
        records.append(
            {
                "preannotation_version": "1.0.0",
                "method": "deterministic_authoring_template_extractor_v1",
                "is_human_annotation": False,
                "is_ground_truth": False,
                "task_id": task_id,
                "task_content_sha256": task_hash,
                "annotation": annotation,
            }
        )
        seeds[task_id] = annotation_form_seed(annotation)
        human_shaped.append(
            {
                "record_version": "1.0.0",
                "protocol_version": PROTOCOL_VERSION,
                "annotator_id": "annotator_a",
                "task_id": task_id,
                "task_content_sha256": task_hash,
                "started_at_utc": "2026-09-29T00:00:00Z",
                "submitted_at_utc": "2026-09-29T00:00:01Z",
                "annotation": annotation,
            }
        )

    audit = audit_submission(
        human_shaped,
        corpus,
        _load_object(WRAPPER_SCHEMA),
        _load_object(SEMANTIC_SCHEMA),
        annotator_id="annotator_a",
        expected_protocol_version=PROTOCOL_VERSION,
    )
    if not audit.is_valid:
        raise ValueError(
            "machine preannotation audit failed:\n- " + "\n- ".join(audit.issues)
        )
    return records, seeds


def _write_form(
    path: Path,
    packet: tuple[dict[str, Any], ...],
    *,
    annotator_id: str,
    packet_path: Path,
    submission_slug: str,
    initial_responses: dict[str, dict[str, Any]] | None = None,
) -> None:
    rendered = render_annotation_form(
        packet,
        annotator_id=annotator_id,
        packet_sha256=sha256_file(packet_path),
        initial_responses=initial_responses,
        submission_slug=submission_slug,
    )
    _atomic_write(path, rendered.encode("utf-8"))


def main() -> None:
    corpus = load_jsonl(TASKS)
    split = _load_object(SPLIT)
    development_ids = list(split["machine_labeled_development"]["task_ids"])
    evaluation_ids = list(split["single_human_evaluation"]["task_ids"])
    development = _subset(corpus, development_ids)
    evaluation = _subset(corpus, evaluation_ids)

    packets = {
        "annotator_a_evaluation": _build_packet(
            evaluation,
            "annotator_a",
            SEEDS["annotator_a_evaluation"],
        ),
    }

    packet_paths = {
        "annotator_a_evaluation": PACKETS / "annotator_a.evaluation.v1.3.jsonl",
    }
    for key, packet in packets.items():
        write_jsonl(packet_paths[key], packet)

    preannotations, _ = _preannotation_records(development)
    write_jsonl(PREANNOTATIONS, preannotations)

    form_paths = {
        "annotator_a_evaluation": FORMS
        / "annotator_a.blind_evaluation.protocol-1.3.0.html",
    }
    _write_form(
        form_paths["annotator_a_evaluation"],
        packets["annotator_a_evaluation"],
        annotator_id="annotator_a",
        packet_path=packet_paths["annotator_a_evaluation"],
        submission_slug="annotator_a.blind_evaluation.protocol-1.3.0",
    )
    manifest: dict[str, Any] = {
        "manifest_version": "1.0.0",
        "created_on": "2026-09-29",
        "protocol_version": PROTOCOL_VERSION,
        "packet_version": PACKET_VERSION,
        "corpus": {
            "path": TASKS.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(TASKS),
            "task_count": 240,
        },
        "split": {
            "path": SPLIT.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(SPLIT),
        },
        "preannotations": {
            "path": PREANNOTATIONS.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(PREANNOTATIONS),
            "record_count": len(preannotations),
            "method": "deterministic_authoring_template_extractor_v1",
            "is_human_annotation": False,
            "is_ground_truth": False,
            "evaluation_task_count": 0,
        },
        "packets": {
            key: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
                "seed": SEEDS[key],
                "task_count": len(packets[key]),
            }
            for key, path in packet_paths.items()
        },
        "forms": {
            key: {
                "local_ignored_path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
                "tracked_by_git": False,
                "regenerate_with": "python scripts/prepare_main_annotation.py",
            }
            for key, path in form_paths.items()
        },
        "workflow": {
            "step_1": "annotator_a completes the blind 60-task evaluation form",
            "step_2": "validate and lock A's single-human reference file",
            "step_3": "report existing pilot A/B/C evidence separately",
            "step_4": "do not request new tasks from annotator B or C",
        },
        "additional_annotator_release_status": "not_planned",
        "human_label_claims": {
            "a_evaluation": "single_human_reference_60",
            "main_gold": "not_available",
            "existing_pilot_evidence": "round_1_a_b_c_and_round_2_a_b",
            "development": "machine_generated_weak_labels_180",
        },
        "main_annotation_status": "forms_prepared_not_collected",
    }
    _atomic_json(MANIFEST, manifest)
    print(f"manifest={MANIFEST}")
    print(f"manifest_sha256={sha256_file(MANIFEST)}")
    for key, path in form_paths.items():
        print(f"{key}_form={path}")
        print(f"{key}_form_sha256={sha256_file(path)}")


if __name__ == "__main__":
    main()
