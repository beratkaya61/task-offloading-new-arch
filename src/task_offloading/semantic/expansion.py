# ruff: noqa: RUF001
"""Leakage-safe construction of the semantic benchmark expansion candidate pool.

The pool is deliberately described in two units: scenario families and related
text rows.  A family contains several semantic contracts and wording views, so
the row count must never be presented as an independent-sample count.

Labels produced here are labels by construction.  They become human reference
labels only after the blind annotation and adjudication protocol recorded in
``configs/semantic_expansion_v1.json`` has been completed.
"""

from __future__ import annotations

import copy
import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from task_offloading.semantic.compositional import compose_tasks
from task_offloading.semantic.paraphrase import paraphrase_task

SPLITS = ("train", "validation", "test")
CONTEXT_ROTATIONS = (0, 2, 5, 8, 11)


def build_scenario_families(blueprint: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Create 75 review candidates per domain from curated subject/operation lists."""

    domains = blueprint.get("domains")
    if not isinstance(domains, Mapping) or len(domains) != 8:
        raise ValueError("the expansion requires exactly eight domains")
    families: list[dict[str, Any]] = []
    for domain, raw_config in domains.items():
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"invalid domain configuration: {domain}")
        contexts = raw_config.get("contexts")
        if not isinstance(contexts, list) or len(contexts) != 15:
            raise ValueError(f"{domain} must provide 15 curated contexts")
        data_name = str(raw_config.get("data", "")).strip()
        if not data_name:
            raise ValueError(f"{domain} has no data description")
        pairs = [(str(row[0]), str(row[1])) for row in contexts]
        for rotation_index, rotation in enumerate(CONTEXT_ROTATIONS):
            for subject_index, (subject, _) in enumerate(pairs):
                operation = pairs[(subject_index + rotation) % len(pairs)][1]
                family_index = rotation_index * len(pairs) + subject_index + 1
                family_id = f"exp1_{domain}_{family_index:03d}"
                families.append(
                    {
                        "task_id": family_id,
                        "language": "tr",
                        "task_text": (
                            f"{subject}, {data_name} üzerinde {operation} görevini "
                            "yürütür."
                        ),
                        "user_policy_text": None,
                        "design_strata": {"scenario_domain": str(domain)},
                        "grouping": {"scenario_family_id": family_id},
                        "provenance": {
                            "origin": "structured_expansion_candidate",
                            "generation_method": (
                                "curated_subject_operation_recombination_requires_"
                                "human_naturalness_review"
                            ),
                            "human_text_review": "pending",
                            "subject": subject,
                            "operation": operation,
                            "data_name": data_name,
                        },
                    }
                )
    if len(families) != 600:
        raise AssertionError("expansion must produce 600 scenario families")
    return families


def assign_family_splits(
    families: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    per_domain: Mapping[str, int],
) -> dict[str, str]:
    """Assign complete scenario families to deterministic domain-balanced splits."""

    if set(per_domain) != set(SPLITS) or sum(per_domain.values()) != 75:
        raise ValueError("per-domain split counts must cover 75 families")
    by_domain: dict[str, list[str]] = defaultdict(list)
    for family in families:
        domain = str(family["design_strata"]["scenario_domain"])
        by_domain[domain].append(str(family["task_id"]))
    assignments: dict[str, str] = {}
    for domain_index, domain in enumerate(sorted(by_domain)):
        ids = sorted(by_domain[domain])
        random.Random(seed + domain_index).shuffle(ids)
        start = 0
        for split in SPLITS:
            stop = start + int(per_domain[split])
            for family_id in ids[start:stop]:
                assignments[family_id] = split
            start = stop
        if start != len(ids):
            raise AssertionError(f"unassigned families in {domain}")
    return assignments


def _family(
    *,
    family_id: str,
    domain: str,
    subject: str,
    operation: str,
    data_name: str,
    subject_group_id: str,
    context_origin: str,
) -> dict[str, Any]:
    return {
        "task_id": family_id,
        "language": "tr",
        "task_text": f"{subject}, {data_name} üzerinde {operation} görevini yürütür.",
        "user_policy_text": None,
        "design_strata": {"scenario_domain": domain},
        "grouping": {
            "scenario_family_id": family_id,
            "subject_group_id": subject_group_id,
        },
        "provenance": {
            "origin": "structured_expansion_candidate",
            "generation_method": (
                "curated_subject_owned_operations_requires_human_naturalness_review"
            ),
            "context_origin": context_origin,
            "human_text_review": "pending",
            "subject": subject,
            "operation": operation,
            "data_name": data_name,
        },
    }


def evaluation_context_terms(
    blueprint: Mapping[str, Any], evaluation_texts: Sequence[str]
) -> tuple[set[str], set[str]]:
    """Return blueprint subjects/operations that occur in any locked evaluation text.

    A blueprint context is considered an evaluation context when either its
    subject or its operation appears in an evaluation text.  Both strings are
    then banned everywhere in the expansion, so no split can contain a scenario
    that the historical evaluation sets were built from.
    """

    subjects: set[str] = set()
    operations: set[str] = set()
    folded = [text.casefold() for text in evaluation_texts]
    for raw_config in dict(blueprint["domains"]).values():
        for subject, operation in raw_config["contexts"]:
            subject_key = str(subject).casefold()
            operation_key = str(operation).casefold()
            if any(subject_key in text or operation_key in text for text in folded):
                subjects.add(str(subject))
                operations.add(str(operation))
    return subjects, operations


def build_context_disjoint_families(
    blueprint: Mapping[str, Any],
    new_contexts: Mapping[str, Any],
    *,
    banned_subjects: set[str],
    banned_operations: set[str],
    subject_split_per_domain: Mapping[str, int],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Create families whose subjects *and* operations never cross splits.

    * Historical blueprint contexts that touch a locked evaluation set are
      dropped entirely; the remaining historical contexts go to ``train`` only.
    * Every new subject owns its operations.  Subjects (with all their
      operations) are assigned to exactly one split, so validation and test
      consist exclusively of devices and operations never seen anywhere else.
    """

    if set(subject_split_per_domain) != {"validation", "test"}:
        raise ValueError("subject split must specify validation and test counts")
    blueprint_domains = dict(blueprint["domains"])
    new_domains = dict(new_contexts["domains"])
    if set(new_domains) != set(blueprint_domains):
        raise ValueError("new contexts must cover exactly the blueprint domains")
    historical_subjects = {
        str(subject)
        for config in blueprint_domains.values()
        for subject, _ in config["contexts"]
    }
    historical_operations = {
        str(operation)
        for config in blueprint_domains.values()
        for _, operation in config["contexts"]
    }
    seen_subjects: set[str] = set()
    seen_operations: set[str] = set()
    families: list[dict[str, Any]] = []
    assignments: dict[str, str] = {}
    for domain_index, domain in enumerate(sorted(blueprint_domains)):
        data_name = str(blueprint_domains[domain]["data"])
        kept = 0
        for subject, operation in blueprint_domains[domain]["contexts"]:
            if subject in banned_subjects or operation in banned_operations:
                continue
            kept += 1
            family_id = f"exp2_{domain}_hist_{kept:02d}"
            families.append(
                _family(
                    family_id=family_id,
                    domain=domain,
                    subject=str(subject),
                    operation=str(operation),
                    data_name=data_name,
                    subject_group_id=family_id,
                    context_origin="historical_development_blueprint",
                )
            )
            assignments[family_id] = "train"
        entries = list(new_domains[domain])
        for subject, operations in entries:
            subject = str(subject)
            if subject in historical_subjects or subject in seen_subjects:
                raise ValueError(f"subject is not new/unique: {subject}")
            if subject[:1] in {"I", "İ"}:
                raise ValueError(f"subject breaks Turkish casefolding: {subject}")
            seen_subjects.add(subject)
            for operation in operations:
                operation = str(operation)
                if operation in historical_operations or operation in seen_operations:
                    raise ValueError(f"operation is not new/unique: {operation}")
                seen_operations.add(operation)
        order = list(range(len(entries)))
        random.Random(seed + 1000 + domain_index).shuffle(order)
        test_n = int(subject_split_per_domain["test"])
        validation_n = int(subject_split_per_domain["validation"])
        if test_n + validation_n >= len(entries):
            raise ValueError(f"{domain} has too few new subjects for the split")
        split_of = {}
        for rank, index in enumerate(order):
            split_of[index] = (
                "test"
                if rank < test_n
                else "validation"
                if rank < test_n + validation_n
                else "train"
            )
        for index, (subject, operations) in enumerate(entries):
            group_id = f"exp2_{domain}_new_{index + 1:02d}"
            for operation_index, operation in enumerate(operations):
                family_id = f"{group_id}_op{operation_index + 1}"
                families.append(
                    _family(
                        family_id=family_id,
                        domain=domain,
                        subject=str(subject),
                        operation=str(operation),
                        data_name=data_name,
                        subject_group_id=group_id,
                        context_origin="new_curated_context_v2",
                    )
                )
                assignments[family_id] = split_of[index]
    return families, assignments


def _annotation_with_id(annotation: Mapping[str, Any], task_id: str) -> dict[str, Any]:
    value = copy.deepcopy(dict(annotation))
    value["task_id"] = task_id
    return value


def build_expanded_rows(
    families: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, str],
    *,
    profiles_per_family: int,
    seed: int,
    profile_prefix: str = "exp1_profile",
    protocol: str = "semantic_expansion_v1",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build paired canonical/paraphrased rows with construction labels."""

    candidates, labels = compose_tasks(
        families,
        variants_per_task=profiles_per_family * 4,
        wording="canonical",
        seed=seed,
        prefix=profile_prefix,
    )
    tasks_out: list[dict[str, Any]] = []
    labels_out: list[dict[str, Any]] = []
    family_by_id = {str(row["task_id"]): row for row in families}
    candidates_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        candidates_by_family[str(row["provenance"]["base_task_id"])].append(row)
    selected: list[tuple[int, dict[str, Any]]] = []
    for base_id, rows in sorted(candidates_by_family.items()):
        signatures: set[str] = set()
        selection_index = 0
        for row in rows:
            annotation = labels[str(row["task_id"])]
            signature = json.dumps(
                {
                    key: value
                    for key, value in annotation.items()
                    if key
                    not in {
                        "task_id",
                        "overall_confidence",
                        "abstained_fields",
                    }
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            if signature in signatures:
                continue
            signatures.add(signature)
            selected.append((selection_index, row))
            selection_index += 1
            if selection_index == profiles_per_family:
                break
        if selection_index != profiles_per_family:
            raise ValueError(f"not enough unique semantic profiles for {base_id}")

    for profile_index, row in selected:
        base_id = str(row["provenance"]["base_task_id"])
        split = assignments[base_id]
        profile_id = f"{base_id}_profile_{profile_index:02d}"
        canonical_id = f"{profile_id}_canonical"
        canonical_row = copy.deepcopy(row)
        canonical_row["task_id"] = canonical_id
        canonical_row["grouping"] = {
            "scenario_family_id": base_id,
            "subject_group_id": str(
                family_by_id[base_id]["grouping"].get("subject_group_id", base_id)
            ),
            "semantic_profile_id": profile_id,
            "paraphrase_family_id": profile_id,
        }
        canonical_row["design_strata"]["split"] = split
        canonical_row["design_strata"]["wording_view"] = "canonical"
        canonical_row["provenance"].update(
            {
                "candidate_protocol": protocol,
                "label_claim": "label_by_construction_pending_human_review",
            }
        )
        canonical_label = _annotation_with_id(labels[str(row["task_id"])], canonical_id)

        pool: Literal["train", "test"] = (
            "train" if split == "train" else "test"
        )
        paraphrased, applied = paraphrase_task(
            canonical_row, pool=pool, variant=profile_index
        )
        source = family_by_id[base_id]["provenance"]
        rewritten_intro = (
            f"{str(source['operation']).capitalize()} amacıyla "
            f"{source['data_name']}, {str(source['subject']).casefold()} "
            "tarafından işlenir."
        )
        _, separator, remainder = str(paraphrased["task_text"]).partition(". ")
        paraphrased["task_text"] = rewritten_intro + (
            f" {remainder}" if separator else ""
        )
        paraphrase_id = f"{profile_id}_paraphrase"
        paraphrased["task_id"] = paraphrase_id
        paraphrased["design_strata"] = dict(paraphrased["design_strata"])
        paraphrased["design_strata"]["wording_view"] = "paraphrase"
        paraphrased["provenance"] = dict(paraphrased["provenance"])
        paraphrased["provenance"].update(
            {
                "paraphrase_pool": pool,
                "paraphrase_rules_applied": list(applied),
                "label_claim": (
                    "construction_label_transferred_under_invariance_pending_"
                    "human_review"
                ),
            }
        )
        paraphrase_label = _annotation_with_id(canonical_label, paraphrase_id)

        tasks_out.extend((canonical_row, paraphrased))
        labels_out.extend(
            (
                {
                    "task_id": canonical_id,
                    "split": split,
                    "annotation": canonical_label,
                },
                {
                    "task_id": paraphrase_id,
                    "split": split,
                    "annotation": paraphrase_label,
                },
            )
        )
    return tasks_out, labels_out


def select_training_audit_families(
    families: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, str],
    *,
    count: int,
    seed: int,
) -> list[str]:
    """Select whole train families with domain stratification for human audit."""

    if count % 8:
        raise ValueError("training audit family count must be divisible by 8")
    by_domain: dict[str, list[str]] = defaultdict(list)
    domain_by_id = {
        str(row["task_id"]): str(row["design_strata"]["scenario_domain"])
        for row in families
    }
    for family_id, split in assignments.items():
        if split == "train":
            by_domain[domain_by_id[family_id]].append(family_id)
    chosen: list[str] = []
    quota = count // 8
    for domain_index, domain in enumerate(sorted(by_domain)):
        values = sorted(by_domain[domain])
        random.Random(seed + 100 + domain_index).shuffle(values)
        chosen.extend(values[:quota])
    return sorted(chosen)


__all__ = [
    "SPLITS",
    "assign_family_splits",
    "build_context_disjoint_families",
    "build_expanded_rows",
    "build_scenario_families",
    "evaluation_context_terms",
    "select_training_audit_families",
]
