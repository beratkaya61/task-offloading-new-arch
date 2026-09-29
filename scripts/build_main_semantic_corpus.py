# ruff: noqa: RUF001
"""Build the balanced 240-task corpus and freeze its label-blind 180/60 split."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from task_offloading.data import sha256_file
from task_offloading.semantic import audit_corpus, load_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "data" / "semantic_benchmark" / "main"
TASKS = MAIN / "tasks.v1.jsonl"
SPLIT = MAIN / "split_manifest.v1.json"
MANIFEST = MAIN / "corpus_manifest.v1.json"
BLUEPRINT = ROOT / "configs" / "main_semantic_scenarios.v1.json"
DESIGN = ROOT / "configs" / "semantic_annotation_design.v1.json"
SCHEMA = ROOT / "schemas" / "main_semantic_task.schema.json"
PROTOCOL = ROOT / "docs" / "ANNOTATION_PROTOCOL_AMENDMENT_1_3.md"
PILOT = ROOT / "data" / "semantic_benchmark" / "pilot" / "tasks.v1.jsonl"

DOMAINS = (
    "healthcare",
    "industrial",
    "transportation",
    "public_safety",
    "smart_city",
    "agriculture",
    "consumer",
    "other",
)
ORIGINS = (
    "standard_derived",
    "trace_conditioned",
    "llm_assisted_synthetic",
    "adversarial",
)
PAYLOADS = (
    ("input_bits.p10", "0,5"),
    ("input_bits.p25", "1,2"),
    ("input_bits.p50", "4,1"),
    ("input_bits.p75", "9,3"),
    ("input_bits.p90", "20,6"),
    ("input_bits.p95", "36,5"),
)
PATTERN_PHENOMENA: tuple[tuple[str, ...], ...] = (
    ("explicit_latency", "numeric_threshold"),
    ("explicit_privacy", "explicit_execution_policy", "negation"),
    ("explicit_reliability", "explicit_accuracy", "numeric_threshold"),
    ("explicit_energy", "explicit_divisibility", "qualitative_requirement"),
    ("missing_information",),
    ("ambiguity", "explicit_latency", "numeric_threshold"),
    ("ambiguity", "explicit_execution_policy", "negation"),
    ("ambiguity", "explicit_accuracy", "qualitative_requirement"),
    ("explicit_privacy", "explicit_execution_policy"),
    ("explicit_reliability", "qualitative_requirement"),
    ("explicit_privacy", "qualitative_requirement"),
    ("explicit_execution_policy", "negation"),
    ("explicit_latency", "explicit_energy", "numeric_threshold"),
    ("explicit_accuracy", "numeric_threshold"),
    ("explicit_divisibility", "qualitative_requirement"),
)


def _load_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value: Any = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    content = (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_bytes(content)
    if temporary.read_bytes() != content:
        raise OSError(f"temporary write verification failed: {temporary}")
    temporary.replace(path)


def _origin(domain_index: int, pattern_index: int) -> str:
    if pattern_index in {5, 6, 7} or (
        pattern_index == 4 and domain_index < 6
    ):
        return "adversarial"
    if pattern_index in {0, 8, 12} or (
        pattern_index == 13 and domain_index < 6
    ):
        return "trace_conditioned"
    if pattern_index in {2, 3, 9} or (
        pattern_index == 1 and domain_index < 6
    ):
        return "standard_derived"
    return "llm_assisted_synthetic"


def _constraints(pattern_index: int, family_index: int) -> list[str]:
    latency_ms = (25, 80, 150, 500, 800, 3000)[family_index % 6]
    clauses = {
        0: [f"Sonuç en geç {latency_ms} ms içinde hazır olmalıdır."],
        1: [
            "İşlenen kayıt hassas veridir.",
            "Cihaz veya yakın edge kullanılabilir; genel bulut kullanılamaz.",
        ],
        2: [
            "İsteklerin en az %99,95'i başarıyla tamamlanmalıdır.",
            "Sonucun F1 skoru en az 0,97 olmalıdır.",
        ],
        3: [
            "Pil tüketimini azaltmak birinci önceliktir.",
            "Veri parçaları bağımsız ve paralel işlenebilir.",
        ],
        4: [],
        5: [
            "Yanıt anlık verilmelidir.",
            "Dört saniyelik gecikme tamamen kabul edilebilir.",
        ],
        6: [
            "Analiz yalnız bulutta yapılmalıdır.",
            "Aynı kaydın buluta gönderilmesi kesinlikle yasaktır.",
        ],
        7: [
            "Sonuç yüksek doğrulukta olmalıdır.",
            "Yaklaşık ve düşük doğruluklu bir sonuç da yeterlidir.",
        ],
        8: [
            "Kullanılan veri kamuya açıktır.",
            "Görev cihaz, yakın edge veya bulutta çalıştırılabilir.",
        ],
        9: ["Bağlantı kesintilerine karşı yüksek güvenilirlik sağlanmalıdır."],
        10: ["İşlenen ham kayıt kısıtlı veridir."],
        11: [
            "Yakın edge sunucusunda işlem yapılabilir ve genel bulut kullanılamaz."
        ],
        12: [
            "Sonuç 500 ms içinde hazırlanmalıdır.",
            "Enerji tüketimi ile yanıt süresi dengelenmelidir.",
        ],
        13: ["Sonucun F1 skoru en az 0,95 olmalıdır."],
        14: ["Görev parçaları birbirinden bağımsız işlenebilir."],
    }
    return list(clauses[pattern_index])


def _render_text(
    *,
    subject: str,
    operation: str,
    data: str,
    origin: str,
    payload: str,
    pattern_index: int,
    family_index: int,
    variant: int,
) -> tuple[str, str | None]:
    payload_phrase = (
        f"yaklaşık {payload} kbitlik " if origin == "trace_conditioned" else ""
    )
    if variant == 0:
        intro = (
            f"{subject}, {payload_phrase}{data} üzerinde {operation} görevini yürütür."
        )
    else:
        intro = (
            f"{operation.capitalize()} için {payload_phrase}{data}, "
            f"{subject.casefold()} tarafından işlenir."
        )
    constraints = _constraints(pattern_index, family_index)
    if variant == 1 and constraints:
        task_text = " ".join([intro, *constraints[:-1]])
        user_policy = constraints[-1]
    else:
        task_text = " ".join([intro, *constraints])
        user_policy = None
    return task_text, user_policy


def _build_tasks(blueprint: Mapping[str, Any]) -> list[dict[str, Any]]:
    domain_values = blueprint.get("domains")
    if not isinstance(domain_values, Mapping) or set(domain_values) != set(DOMAINS):
        raise ValueError("blueprint must contain exactly the eight frozen domains")

    tasks: list[dict[str, Any]] = []
    family_index = 0
    task_index = 0
    for domain_index, domain in enumerate(DOMAINS):
        domain_config = domain_values[domain]
        if not isinstance(domain_config, Mapping):
            raise ValueError(f"invalid domain blueprint: {domain}")
        data = str(domain_config.get("data", ""))
        contexts = domain_config.get("contexts")
        if not isinstance(contexts, list) or len(contexts) != 15:
            raise ValueError(f"{domain} must contain exactly 15 contexts")

        for pattern_index, context in enumerate(contexts):
            if not isinstance(context, list) or len(context) != 2:
                raise ValueError(f"invalid context in {domain}: {context}")
            subject, operation = (str(value) for value in context)
            family_index += 1
            origin = _origin(domain_index, pattern_index)
            aggregate_field, payload = PAYLOADS[(family_index - 1) % len(PAYLOADS)]
            for variant in (0, 1):
                task_index += 1
                task_text, user_policy = _render_text(
                    subject=subject,
                    operation=operation,
                    data=data,
                    origin=origin,
                    payload=payload,
                    pattern_index=pattern_index,
                    family_index=family_index,
                    variant=variant,
                )
                trace_conditioning = None
                source_reference_ids: list[str] = []
                if origin == "trace_conditioned":
                    source_reference_ids = ["bupt_edge_computing_dataset"]
                    trace_conditioning = {
                        "profile_id": "bupt_04e664f_workload_quantiles_v1",
                        "aggregate_fields": [aggregate_field],
                    }
                elif origin == "standard_derived":
                    source_reference_ids = ["etsi_mec_overview"]

                tasks.append(
                    {
                        "corpus_version": "main-1.0.0",
                        "task_id": f"main_tr_{task_index:03d}",
                        "language": "tr",
                        "task_text": task_text,
                        "user_policy_text": user_policy,
                        "provenance": {
                            "origin": origin,
                            "generation_method": "llm_assisted_structured_design",
                            "source_reference_ids": source_reference_ids,
                            "trace_conditioning": trace_conditioning,
                            "human_text_review": "pending",
                        },
                        "grouping": {
                            "scenario_family_id": f"main_sf_{family_index:03d}",
                            "template_family_id": f"main_tf_{pattern_index + 1:03d}",
                            "paraphrase_family_id": f"main_pf_{family_index:03d}",
                            "source_entity_id": None,
                        },
                        "design_strata": {
                            "scenario_domain": domain,
                            "semantic_phenomena": list(
                                PATTERN_PHENOMENA[pattern_index]
                            ),
                        },
                    }
                )
    return tasks


def _family_rows(
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    families: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for task in tasks:
        grouping = task["grouping"]
        if not isinstance(grouping, Mapping):
            raise ValueError("task grouping must be an object")
        families[str(grouping["scenario_family_id"])].append(task)
    if len(families) != 120 or any(len(rows) != 2 for rows in families.values()):
        raise ValueError("main corpus must contain 120 two-task scenario families")
    return dict(families)


def _select_evaluation_families(
    tasks: Sequence[Mapping[str, Any]], seed: int
) -> tuple[str, ...]:
    families = _family_rows(tasks)
    by_domain: dict[str, list[str]] = defaultdict(list)
    for family_id, rows in families.items():
        strata = rows[0]["design_strata"]
        if not isinstance(strata, Mapping):
            raise ValueError("design_strata must be an object")
        by_domain[str(strata["scenario_domain"])].append(family_id)

    domain_quotas = {
        domain: 4 if index < 6 else 3 for index, domain in enumerate(DOMAINS)
    }
    origin_targets = {
        "standard_derived": 8,
        "trace_conditioned": 8,
        "llm_assisted_synthetic": 7,
        "adversarial": 7,
    }
    all_phenomena = {value for values in PATTERN_PHENOMENA for value in values}
    best: tuple[int, tuple[str, ...]] | None = None
    for attempt in range(2000):
        rng = random.Random(seed + attempt)
        selected: list[str] = []
        for domain in DOMAINS:
            candidates = sorted(by_domain[domain])
            rng.shuffle(candidates)
            selected.extend(candidates[: domain_quotas[domain]])
        origin_counts: Counter[str] = Counter()
        phenomena: set[str] = set()
        for family_id in selected:
            row = families[family_id][0]
            provenance = row["provenance"]
            strata = row["design_strata"]
            if not isinstance(provenance, Mapping) or not isinstance(strata, Mapping):
                raise ValueError("invalid family metadata")
            origin_counts[str(provenance["origin"])] += 1
            raw_phenomena = strata["semantic_phenomena"]
            if isinstance(raw_phenomena, list):
                phenomena.update(str(value) for value in raw_phenomena)
        score = sum(
            abs(origin_counts[origin] - target)
            for origin, target in origin_targets.items()
        )
        score += 100 * len(all_phenomena - phenomena)
        candidate = (score, tuple(sorted(selected)))
        if best is None or candidate < best:
            best = candidate
            if score == 0:
                break
    if best is None or best[0] != 0:
        raise ValueError("could not build a fully balanced evaluation partition")
    return best[1]


def _build_split(tasks: Sequence[Mapping[str, Any]], seed: int) -> dict[str, Any]:
    evaluation_families = set(_select_evaluation_families(tasks, seed))
    development_ids: list[str] = []
    evaluation_ids: list[str] = []
    development_families: set[str] = set()
    for task in tasks:
        grouping = task["grouping"]
        if not isinstance(grouping, Mapping):
            raise ValueError("task grouping must be an object")
        family_id = str(grouping["scenario_family_id"])
        task_id = str(task["task_id"])
        if family_id in evaluation_families:
            evaluation_ids.append(task_id)
        else:
            development_ids.append(task_id)
            development_families.add(family_id)
    if len(development_ids) != 180 or len(evaluation_ids) != 60:
        raise ValueError("main partition must be exactly 180/60")
    if development_families & evaluation_families:
        raise ValueError("scenario-family leakage across the main partition")
    return {
        "split_version": "1.1.0",
        "created_on": "2026-09-29",
        "selection_algorithm": "seeded_domain_origin_stratified_family_search_v1",
        "selection_seed": seed,
        "selection_uses_human_or_model_labels": False,
        "group_keys": [
            "grouping.scenario_family_id",
            "grouping.paraphrase_family_id",
            "grouping.source_entity_id",
        ],
        "machine_labeled_development": {
            "task_count": len(development_ids),
            "family_count": len(development_families),
            "task_ids": development_ids,
            "scenario_family_ids": sorted(development_families),
        },
        "single_human_evaluation": {
            "task_count": len(evaluation_ids),
            "family_count": len(evaluation_families),
            "task_ids": evaluation_ids,
            "scenario_family_ids": sorted(evaluation_families),
        },
    }


def _count_partition(
    tasks: Sequence[Mapping[str, Any]], task_ids: set[str]
) -> dict[str, dict[str, int]]:
    domains: Counter[str] = Counter()
    origins: Counter[str] = Counter()
    phenomena: Counter[str] = Counter()
    for task in tasks:
        if str(task["task_id"]) not in task_ids:
            continue
        strata = task["design_strata"]
        provenance = task["provenance"]
        if not isinstance(strata, Mapping) or not isinstance(provenance, Mapping):
            raise ValueError("invalid task metadata")
        domains[str(strata["scenario_domain"])] += 1
        origins[str(provenance["origin"])] += 1
        raw = strata["semantic_phenomena"]
        if isinstance(raw, list):
            phenomena.update(str(value) for value in raw)
    return {
        "domains": dict(sorted(domains.items())),
        "origins": dict(sorted(origins.items())),
        "semantic_phenomena": dict(sorted(phenomena.items())),
    }


def main() -> None:
    blueprint = _load_object(BLUEPRINT)
    tasks = _build_tasks(blueprint)
    schema = _load_object(SCHEMA)
    audit = audit_corpus(tasks, schema, expected_count=240)
    if not audit.is_valid:
        raise ValueError("main corpus audit failed:\n- " + "\n- ".join(audit.issues))

    pilot_texts = {
        (str(row["task_text"]) + "\n" + str(row.get("user_policy_text") or ""))
        .casefold()
        .strip()
        for row in load_jsonl(PILOT)
    }
    main_texts = {
        (str(row["task_text"]) + "\n" + str(row.get("user_policy_text") or ""))
        .casefold()
        .strip()
        for row in tasks
    }
    if pilot_texts & main_texts:
        raise ValueError("main corpus reuses an exact pilot task")

    MAIN.mkdir(parents=True, exist_ok=True)
    write_jsonl(TASKS, tasks)
    split = _build_split(tasks, seed=20260929)
    _atomic_json(SPLIT, split)
    development_ids = set(split["machine_labeled_development"]["task_ids"])
    evaluation_ids = set(split["single_human_evaluation"]["task_ids"])
    manifest: dict[str, Any] = {
        "manifest_version": "1.0.0",
        "created_on": "2026-09-29",
        "corpus_version": "main-1.0.0",
        "status": "generated_awaiting_human_text_review",
        "annotation_release_allowed": False,
        "task_count": len(tasks),
        "scenario_family_count": 120,
        "human_text_review": {"pending": 240, "approved": 0, "rejected": 0},
        "tasks": {
            "path": TASKS.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(TASKS),
        },
        "schema": {
            "path": SCHEMA.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(SCHEMA),
        },
        "blueprint": {
            "path": BLUEPRINT.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(BLUEPRINT),
        },
        "annotation_design": {
            "path": DESIGN.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(DESIGN),
        },
        "protocol": {
            "version": "1.3.0",
            "path": PROTOCOL.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(PROTOCOL),
        },
        "split": {
            "path": SPLIT.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(SPLIT),
        },
        "design_metadata_is_gold": False,
        "generation_model_may_define_gold": False,
        "pilot_exact_text_overlap_count": 0,
        "overall_counts": {
            "domains": dict(audit.domain_stratum_counts),
            "origins": dict(audit.origin_counts),
            "semantic_phenomena": dict(audit.phenomenon_counts),
        },
        "partition_counts": {
            "machine_labeled_development": _count_partition(
                tasks, development_ids
            ),
            "single_human_evaluation": _count_partition(tasks, evaluation_ids),
        },
        "sources": [
            {
                "id": "etsi_mec_overview",
                "url": "https://www.etsi.org/technologies/multi-access-edge-computing",
                "use": "scenario inspiration only; no verbatim source text",
            },
            {
                "id": "bupt_edge_computing_dataset",
                "profile": "configs/data_profiles/bupt_04e664f_workload_quantiles.json",
                "use": (
                    "privacy-safe input-bit quantiles only; no row-level data "
                    "or semantic labels"
                ),
            },
        ],
    }
    _atomic_json(MANIFEST, manifest)
    print(f"tasks={TASKS}")
    print(f"tasks_sha256={sha256_file(TASKS)}")
    print(f"split={SPLIT}")
    print(f"split_sha256={sha256_file(SPLIT)}")
    print(f"manifest={MANIFEST}")
    print(f"manifest_sha256={sha256_file(MANIFEST)}")


if __name__ == "__main__":
    main()
