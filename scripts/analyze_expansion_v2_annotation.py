# ruff: noqa: E501
"""Audit expansion v2 submissions and report agreement, consistency and gates.

Drop exported files into ``data/semantic_benchmark/expanded_v2/annotation/submissions``
as ``annotator_a.<session>.jsonl`` / ``annotator_b.<session>.jsonl`` (session =
``pilot`` or ``session_01`` ... ``session_07``).  The script can run after every
session; sessions that only one annotator has submitted are audited but not
compared yet.

Outputs (``annotation/reports``):
* ``agreement_report.json`` — per-session and pooled A/B agreement (kappa per
  field), naturalness agreement, hidden-repeat intra-annotator consistency and
  human-vs-construction agreement on training-audit rows;
* ``disagreements.<session>.jsonl`` — cells for the separate adjudication step.

Labels are never shown to annotators; this script is for the coordinator only.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from task_offloading.semantic import (
    ANNOTATION_FIELDS,
    audit_submission,
    disagreement_rows,
    load_jsonl,
    pairwise_agreement,
    write_jsonl,
)
from task_offloading.semantic.baselines import semantic_signature
from task_offloading.semantic.experiment import write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "semantic_benchmark" / "expanded_v2"
ANN = DATA / "annotation"
SUBMISSIONS = ANN / "submissions"
REPORTS = ANN / "reports"
WRAPPER = ROOT / "schemas" / "human_annotation_record.v2.schema.json"
SEMANTIC = ROOT / "schemas" / "semantic_requirements.schema.json"
KAPPA_MINIMUM = 0.70


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _corpus(packet_path: Path) -> list[dict[str, Any]]:
    return [
        {k: row[k] for k in ("task_id", "language", "task_text", "user_policy_text")}
        for row in load_jsonl(packet_path)
    ]


def _field_kappas(agreement: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Status kappa over all tasks and value kappa over both-explicit tasks."""

    out: dict[str, dict[str, Any]] = {}
    for field in agreement["fields"]:
        value = field["value"]
        out[field["field"]] = {
            "status_kappa": field["status"]["kappa"],
            "status_percent": field["status"]["percent_agreement"],
            "value_kappa": value["kappa"] if value else None,
            "value_n": value["sample_size"] if value else 0,
            "exact_semantic_agreement": field["exact_semantic_agreement"],
        }
    execution = agreement["execution_policy"]
    out["execution_policy"] = {
        "status_kappa": execution["status"]["kappa"],
        "status_percent": execution["status"]["percent_agreement"],
        "mean_allowed_target_jaccard": execution["mean_allowed_target_jaccard"],
        "exact_semantic_agreement": execution["exact_semantic_agreement"],
    }
    return out


def _min_kappa(entry: dict[str, Any]) -> float | None:
    values = [entry.get("status_kappa"), entry.get("value_kappa")]
    present = [v for v in values if v is not None]
    return min(present) if present else None


def main() -> None:
    plan = _object(ANN / "annotation_plan.v2.json")
    key = {r["alias"]: r for r in load_jsonl(ANN / "blinding_key.coordinator_only.jsonl")}
    construction = {r["task_id"]: r["annotation"] for r in load_jsonl(DATA / "candidates.construction_labels.jsonl")}
    wrapper, semantic = _object(WRAPPER), _object(SEMANTIC)
    sessions = sorted(plan["sessions"])
    records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    audits: dict[str, Any] = {}
    for annotator in ("annotator_a", "annotator_b"):
        records[annotator] = {}
        for session in sessions:
            path = SUBMISSIONS / f"{annotator}.{session}.jsonl"
            if not path.exists():
                continue
            rows = list(load_jsonl(path))
            corpus = _corpus(ROOT / plan["packets"][f"{annotator}/{session}"]["packet"])
            audit = audit_submission(rows, corpus, wrapper, semantic, annotator_id=annotator, expected_protocol_version="2.0.0")
            audits[f"{annotator}/{session}"] = {"records": audit.record_count, "issues": list(audit.issues)}
            if audit.is_valid:
                records[annotator][session] = rows

    report: dict[str, Any] = {"audits": audits, "sessions": {}, "kappa_minimum": KAPPA_MINIMUM}
    pooled_a: list[dict[str, Any]] = []
    pooled_b: list[dict[str, Any]] = []
    for session in sessions:
        a, b = records["annotator_a"].get(session), records["annotator_b"].get(session)
        if not (a and b):
            continue
        agreement = pairwise_agreement(a, b).to_dict()
        natural_a = {r["task_id"]: r["naturalness"] for r in a}
        natural_b = {r["task_id"]: r["naturalness"] for r in b}
        report["sessions"][session] = {
            "task_count": agreement["task_count"],
            "exact_record_agreement": agreement["exact_record_agreement"],
            "field_kappa": _field_kappas(agreement),
            "naturalness_percent_agreement": sum(natural_a[t] == natural_b[t] for t in natural_a) / len(natural_a),
            "naturalness_counts": {"annotator_a": Counter(natural_a.values()), "annotator_b": Counter(natural_b.values())},
            "both_unnatural": sorted(t for t in natural_a if natural_a[t] == natural_b[t] == "unnatural"),
        }
        write_jsonl(REPORTS / f"disagreements.{session}.jsonl", list(disagreement_rows(a, b)))
        if session != "pilot":
            pooled_a.extend(a)
            pooled_b.extend(b)

    if pooled_a and pooled_b:
        main_a = [r for r in pooled_a if key[r["task_id"]]["role"] != "hidden_repeat"]
        main_b = [r for r in pooled_b if key[r["task_id"]]["role"] != "hidden_repeat"]
        pooled = pairwise_agreement(main_a, main_b).to_dict()
        kappas = _field_kappas(pooled)
        report["pooled_main"] = {
            "task_count": pooled["task_count"],
            "exact_record_agreement": pooled["exact_record_agreement"],
            "field_kappa": kappas,
            "fields_below_minimum": [
                f for f, entry in kappas.items() if (k := _min_kappa(entry)) is not None and k < KAPPA_MINIMUM
            ],
        }

    consistency: dict[str, Any] = {}
    for annotator, by_session in records.items():
        by_task: dict[str, dict[str, Any]] = {}
        for rows in by_session.values():
            for row in rows:
                by_task[key[row["task_id"]]["task_id"] + ("#repeat" if key[row["task_id"]]["role"] == "hidden_repeat" else "")] = row
        pairs = [(by_task[t], by_task[t + "#repeat"]) for t in list(by_task) if t + "#repeat" in by_task]
        if pairs:
            same = [
                sum(semantic_signature(x["annotation"], f) == semantic_signature(y["annotation"], f) for f in ANNOTATION_FIELDS) / len(ANNOTATION_FIELDS)
                for x, y in pairs
            ]
            consistency[annotator] = {"repeat_pairs": len(pairs), "mean_field_self_agreement": sum(same) / len(same)}
    report["intra_annotator_consistency"] = consistency

    audit_rows: dict[str, list[float]] = {}
    for annotator, by_session in records.items():
        for rows in by_session.values():
            for row in rows:
                entry = key[row["task_id"]]
                if entry["role"] != "training_audit":
                    continue
                label = construction[entry["task_id"]]
                audit_rows.setdefault(annotator, []).append(
                    sum(semantic_signature(row["annotation"], f) == semantic_signature(label, f) for f in ANNOTATION_FIELDS) / len(ANNOTATION_FIELDS)
                )
    report["human_vs_construction_training_audit"] = {
        annotator: {"rows": len(values), "mean_field_agreement": sum(values) / len(values)}
        for annotator, values in audit_rows.items()
    }
    write_json_atomic(REPORTS / "agreement_report.json", json.loads(json.dumps(report, default=dict)))
    print(json.dumps({k: v for k, v in report.items() if k != "audits"}, ensure_ascii=False, indent=1, default=dict)[:4000])
    bad = {k: v["issues"][:3] for k, v in audits.items() if v["issues"]}
    if bad:
        print(f"SUBMISSION ISSUES: {bad}")


if __name__ == "__main__":
    main()
