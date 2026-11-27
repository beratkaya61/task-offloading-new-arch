# ruff: noqa: E501
"""Materialize the context-disjoint v2 semantic expansion candidate.

v2 fixes two leakage paths found in the v1 candidate:

1. 244/600 v1 families reused subjects/operations of the locked evaluation
   scenarios (Human-60 and the Comp-600/Independent-188 sets derived from it).
2. v1 split *families*, but a subject or operation appeared in several
   families, so 98/120 subjects and 103/120 operations crossed splits.

v2 bans every evaluation context everywhere, keeps clean historical contexts in
train only, and splits new subjects (with all their operations) as a unit.
The build fails closed when any leakage audit is non-zero.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from task_offloading.semantic import load_jsonl, write_jsonl
from task_offloading.semantic.expansion import (
    build_context_disjoint_families,
    build_expanded_rows,
    evaluation_context_terms,
    select_training_audit_families,
)
from task_offloading.semantic.experiment import write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "semantic_expansion_v2.json"
BLUEPRINT = ROOT / "configs" / "main_semantic_scenarios.v1.json"
MAIN = ROOT / "data" / "semantic_benchmark" / "main"
V3 = MAIN / "benchmark_v3"
OUTPUT = ROOT / "data" / "semantic_benchmark" / "expanded_v2"
DATA_DIR = OUTPUT


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _visible(row: dict[str, Any]) -> str:
    return f"{row['task_text']}\n{row.get('user_policy_text') or ''}".strip()


def historical_texts() -> tuple[set[str], set[str]]:
    """Return (locked evaluation texts, historical training/calibration texts)."""

    split = _object(MAIN / "split_manifest.v1.json")
    evaluation_ids = set(split["single_human_evaluation"]["task_ids"])
    main_rows = list(load_jsonl(MAIN / "tasks.v1.jsonl"))
    evaluation = {_visible(r) for r in main_rows if r["task_id"] in evaluation_ids}
    training = {_visible(r) for r in main_rows if r["task_id"] not in evaluation_ids}
    for path in sorted(V3.glob("eval_*.tasks.jsonl")):
        evaluation |= {_visible(r) for r in load_jsonl(path)}
    robustness = MAIN / "paraphrase_robustness"
    for path in sorted(robustness.glob("evaluation_tasks*.jsonl")):
        evaluation |= {_visible(r) for r in load_jsonl(path)}
    for name in ("train_augmented_tasks.jsonl", "calibration_tasks.jsonl"):
        training |= {_visible(r) for r in load_jsonl(V3 / name)}
    return evaluation, training


def leakage_audit(
    tasks: list[dict[str, Any]],
    families: list[dict[str, Any]],
    *,
    evaluation_texts: set[str],
    training_texts: set[str],
    banned_subjects: set[str],
    banned_operations: set[str],
) -> dict[str, int]:
    family = {str(f["task_id"]): f for f in families}
    by_split: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    hashes: Counter[str] = Counter()
    group_splits: dict[str, set[str]] = defaultdict(set)
    profile_splits: dict[str, set[str]] = defaultdict(set)
    exact_eval = exact_history_heldout = banned_hits = 0
    banned_terms = [t.casefold() for t in (*banned_subjects, *banned_operations)]
    for task in tasks:
        split = str(task["design_strata"]["split"])
        source = family[str(task["grouping"]["scenario_family_id"])]["provenance"]
        by_split[split]["subject"].add(str(source["subject"]))
        by_split[split]["operation"].add(str(source["operation"]))
        group_splits[str(task["grouping"]["subject_group_id"])].add(split)
        profile_splits[str(task["grouping"]["semantic_profile_id"])].add(split)
        text = _visible(task)
        hashes[hashlib.sha256(text.encode("utf-8")).hexdigest()] += 1
        exact_eval += text in evaluation_texts
        if split != "train":
            exact_history_heldout += text in training_texts
        folded = text.casefold()
        banned_hits += any(term in folded for term in banned_terms)
    cross_subject = cross_operation = 0
    splits = sorted(by_split)
    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            cross_subject += len(by_split[left]["subject"] & by_split[right]["subject"])
            cross_operation += len(
                by_split[left]["operation"] & by_split[right]["operation"]
            )
    heldout_terms = {
        term.casefold()
        for split in ("validation", "test")
        for kind in ("subject", "operation")
        for term in by_split[split][kind]
    }
    history = [t.casefold() for t in (*evaluation_texts, *training_texts)]
    heldout_terms_in_history = sum(
        any(term in text for text in history) for term in heldout_terms
    )
    return {
        "duplicate_visible_texts": sum(count - 1 for count in hashes.values()),
        "subject_group_cross_split": sum(len(v) > 1 for v in group_splits.values()),
        "semantic_profile_cross_split": sum(len(v) > 1 for v in profile_splits.values()),
        "subject_cross_split": cross_subject,
        "operation_cross_split": cross_operation,
        "exact_text_overlap_with_locked_evaluation": exact_eval,
        "heldout_exact_text_overlap_with_historical_training": exact_history_heldout,
        "rows_containing_banned_evaluation_context_terms": banned_hits,
        "heldout_subject_or_operation_found_in_historical_texts": heldout_terms_in_history,
    }


def add_natural_views(
    tasks: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    *,
    evaluation_texts: set[str],
    banned_terms: list[str],
) -> dict[str, Any]:
    """Append accepted LLM natural rewrites as a third wording view per profile.

    Labels are transferred from the canonical row and flagged as pending
    human verification; rewrites that touch locked evaluation texts or banned
    evaluation-context terms are dropped (never allowed to fail open).
    """

    rewrites: dict[str, dict[str, Any]] = {}
    for path in sorted((DATA_DIR / "natural_rewrites").glob("*.jsonl")):
        if path.name.endswith(".smoke.jsonl"):
            continue
        for row in load_jsonl(path):
            if row["accepted"]:
                rewrites[str(row["source_task_id"])] = row
    label_by_id = {str(r["task_id"]): r for r in labels}
    seen_texts = {_visible(t) for t in tasks}
    added: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    generators: dict[str, set[str]] = defaultdict(set)
    for canonical in [t for t in tasks if t["design_strata"]["wording_view"] == "canonical"]:
        rewrite = rewrites.get(str(canonical["task_id"]))
        if rewrite is None:
            continue
        split = str(canonical["design_strata"]["split"])
        if rewrite["split"] != split:
            raise ValueError(f"rewrite split mismatch for {canonical['task_id']}")
        text = str(rewrite["rewrite"]).strip()
        folded = text.casefold()
        if text in evaluation_texts or text in seen_texts or any(term in folded for term in banned_terms):
            dropped[split] += 1
            continue
        seen_texts.add(text)
        profile_id = str(canonical["grouping"]["semantic_profile_id"])
        natural_id = f"{profile_id}_natural"
        row = copy.deepcopy(canonical)
        row["task_id"] = natural_id
        row["task_text"] = text
        row["user_policy_text"] = None
        row["design_strata"]["wording_view"] = "natural"
        row["provenance"] = {
            **row["provenance"],
            "origin": "llm_natural_rewrite",
            "source_task_id": canonical["task_id"],
            "generator_model": rewrite["generator_model"],
            "generator_revision": rewrite["generator_revision"],
            "label_claim": "construction_label_transferred_pending_human_verification",
        }
        source_label = label_by_id[str(canonical["task_id"])]
        annotation = copy.deepcopy(source_label["annotation"])
        annotation["task_id"] = natural_id
        tasks.append(row)
        labels.append({"task_id": natural_id, "split": split, "annotation": annotation})
        added[split] += 1
        generators[split].add(str(rewrite["generator_model"]))
    return {
        "added_by_split": dict(sorted(added.items())),
        "dropped_by_split": dict(sorted(dropped.items())),
        "generator_by_split": {k: sorted(v) for k, v in sorted(generators.items())},
    }


def main() -> None:
    config = _object(CONFIG)
    blueprint = _object(BLUEPRINT)
    contexts = _object(ROOT / str(config["contexts_file"]))
    seed = int(config["seed"])
    evaluation_texts, training_texts = historical_texts()
    banned_subjects, banned_operations = evaluation_context_terms(
        blueprint, sorted(evaluation_texts)
    )
    families, assignments = build_context_disjoint_families(
        blueprint,
        contexts,
        banned_subjects=banned_subjects,
        banned_operations=banned_operations,
        subject_split_per_domain=config["subject_split_per_domain"],
        seed=seed,
    )
    tasks, labels = build_expanded_rows(
        families,
        assignments,
        profiles_per_family=int(config["semantic_profiles_per_family"]),
        seed=seed,
        profile_prefix="exp2_profile",
        protocol="semantic_expansion_v2",
    )
    natural_report = add_natural_views(
        tasks,
        labels,
        evaluation_texts=evaluation_texts,
        banned_terms=[t.casefold() for t in (*banned_subjects, *banned_operations)],
    )
    audit = leakage_audit(
        tasks,
        families,
        evaluation_texts=evaluation_texts,
        training_texts=training_texts,
        banned_subjects=banned_subjects,
        banned_operations=banned_operations,
    )
    if any(audit.values()):
        raise ValueError(f"leakage audit failed: {audit}")

    task_split = {str(r["task_id"]): str(r["design_strata"]["split"]) for r in tasks}
    audit_families = select_training_audit_families(
        families,
        assignments,
        count=int(config["human_annotation"]["training_audit_families"]),
        seed=seed,
    )
    audit_set = set(audit_families)
    annotators = int(config["human_annotation"]["independent_annotators"])
    annotation_items = [
        {
            "task_id": str(task["task_id"]),
            "scenario_family_id": str(task["grouping"]["scenario_family_id"]),
            "subject_group_id": str(task["grouping"]["subject_group_id"]),
            "split": task_split[str(task["task_id"])],
            "annotation_role": (
                f"{task_split[str(task['task_id'])]}_reference"
                if task_split[str(task["task_id"])] != "train"
                else "training_audit"
            ),
            "independent_annotators_required": annotators,
            "adjudication_on_disagreement": True,
        }
        for task in tasks
        if task_split[str(task["task_id"])] != "train"
        or str(task["grouping"]["scenario_family_id"]) in audit_set
    ]
    group_of = {
        str(f["task_id"]): str(f["grouping"]["subject_group_id"]) for f in families
    }
    group_counts = Counter(
        split for split in {group_of[k]: v for k, v in assignments.items()}.values()
    )
    origin_counts = Counter(
        (str(f["provenance"]["context_origin"]), assignments[str(f["task_id"])])
        for f in families
    )
    manifest = {
        "protocol_version": config["protocol_version"],
        "status": config["status"],
        "supersedes": config["supersedes"],
        "seed": seed,
        "independent_unit": config["independent_unit"],
        "related_row_unit": "semantic_profile_x_wording_view",
        "subject_group_counts": dict(sorted(group_counts.items())),
        "family_counts": dict(sorted(Counter(assignments.values()).items())),
        "row_counts": dict(sorted(Counter(task_split.values()).items())),
        "family_origin_counts": {
            f"{origin}:{split}": count
            for (origin, split), count in sorted(origin_counts.items())
        },
        "total_subject_groups": len(set(group_of.values())),
        "total_families": len(families),
        "total_rows": len(tasks),
        "banned_evaluation_contexts": {
            "subjects": sorted(banned_subjects),
            "operations": sorted(banned_operations),
        },
        "leakage_audit": audit,
        "natural_views": natural_report,
        "labels": "by_construction_pending_blind_human_annotation",
        "family_assignments": dict(sorted(assignments.items())),
        "comparison_contract": config["comparison_contract"],
    }
    annotation_manifest = {
        "protocol_version": config["protocol_version"],
        "blind_to_model_predictions": True,
        "independent_annotators": annotators,
        "adjudicator_on_disagreement": config["human_annotation"][
            "adjudicator_on_disagreement"
        ],
        "training_audit_families": audit_families,
        "unique_tasks_to_annotate": len(annotation_items),
        "minimum_independent_assignments": len(annotation_items) * annotators,
        "items": annotation_items,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT / "candidates.tasks.jsonl", tasks)
    write_jsonl(OUTPUT / "candidates.construction_labels.jsonl", labels)
    write_jsonl(OUTPUT / "scenario_families.jsonl", families)
    write_json_atomic(OUTPUT / "split_manifest.candidate.json", manifest)
    write_json_atomic(OUTPUT / "annotation_manifest.candidate.json", annotation_manifest)
    print(
        f"subject_groups={dict(sorted(group_counts.items()))} "
        f"families={manifest['family_counts']} rows={manifest['row_counts']}"
    )
    print(f"banned_eval_contexts={len(banned_subjects)} leakage_audit={audit}")
    print(f"natural_views={natural_report}")
    print(
        f"human_annotation_tasks={len(annotation_items)} "
        f"minimum_assignments={annotation_manifest['minimum_independent_assignments']}"
    )


if __name__ == "__main__":
    main()
