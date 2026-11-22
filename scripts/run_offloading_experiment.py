"""Semantic-on/off offloading experiment on trace-driven hybrid episodes.

Design (fixed before looking at any result):

* Training episodes: BUPT days < 24, semantic texts from the 180 development
  tasks (weak labels as both truth and belief).  PPO is trained once per seed
  and condition: ``ppo_semantic`` (requirement features + semantic mask) and
  ``ppo_no_semantic`` (no requirement features, no semantic mask).
* Test episodes: BUPT days >= 24, semantic texts from the 60 locked evaluation
  tasks.  *Truth* = protocol-normalized single-human reference.  *Belief* =
  output of a semantic source (reference, rule, TF-IDF+augmentation, constant
  profile) on the original wording or on the held-out paraphrases, with or
  without the conservative fallback.
* Every policy and source is evaluated on the *same* test episodes (paired
  design); violations are always judged against the truth.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pickle
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.spaces import Discrete
from sb3_contrib import MaskablePPO

from task_offloading.domain import ExecutionTier, OffloadOutcome, TaskRequirements
from task_offloading.env import ScenarioPoolEnv, TaskOffloadingEnv
from task_offloading.experiment_config import (
    ENERGY_REFERENCE_J,
    EXPERIMENT_SEED,
    LATENCY_REFERENCE_S,
    PPO_HYPERPARAMETERS,
    PPO_SEEDS,
    PPO_TIMESTEPS,
    TEST_EPISODES,
    TRAIN_EPISODES,
)
from task_offloading.policies import fixed_target, greedy, random_valid
from task_offloading.semantic import load_jsonl
from task_offloading.semantic.compositional import compose_tasks
from task_offloading.semantic.experiment import load_json_object, ordered_subset
from task_offloading.semantic.preannotation import authoring_preannotation
from task_offloading.sim import PhysicsConfig, Scenario
from task_offloading.sim.trace_hybrid import (
    HybridConfig,
    PhysicalPools,
    SemanticItem,
    ablate_physical_pools,
    annotation_to_requirements,
    build_episode,
    load_nep_layer,
    load_physical_pools,
    split_eua_entity_holdout,
)
from task_offloading.system_metrics import system_metrics

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "data" / "semantic_benchmark" / "main"
DATASETS = Path("D:/task_offloading_datasets")
POOL_CACHE = ROOT / "artifacts" / "trace_hybrid_pools.v1.pkl"
NEP_PROFILE = ROOT / "configs" / "data_profiles" / "nep_quantiles_v1.json"
OUTPUT = ROOT / "artifacts" / "offloading_experiment"
PSEUDONYM_KEY = b"thesis-local-only-bupt-pseudonym-key-v1"
SEED = EXPERIMENT_SEED
PHYSICS = PhysicsConfig(
    latency_reference_s=LATENCY_REFERENCE_S, energy_reference_j=ENERGY_REFERENCE_J
)
NO_SEMANTIC_PHYSICS = dataclasses.replace(PHYSICS, mask_semantic_permissions=False)


def _dataset_archive(root: Path, name: str, candidates: Sequence[Path]) -> Path:
    """Resolve one pinned dataset layout without an unbounded directory scan."""

    for relative in candidates:
        path = root / relative
        if path.is_file():
            return path
    tried = "\n".join(f"- {root / relative}" for relative in candidates)
    raise FileNotFoundError(f"{name} archive was not found; tried:\n{tried}")


def _dataset_archives(root: Path) -> tuple[Path, Path, Path]:
    uci = _dataset_archive(
        root,
        "UCI MEC",
        (
            Path("UCI_MEC_859/TATDescriptionDataset.zip"),
            Path("uci/TATDescriptionDataset.zip"),
        ),
    )
    bupt = _dataset_archive(
        root,
        "BUPT Edge Computing",
        (
            Path(
                "BUPT_Edge_Computing_Dataset/"
                "04e664fab9cdb2a58d04ebc615cd74405e6062e2/"
                "Edge-Computing-Dataset-"
                "04e664fab9cdb2a58d04ebc615cd74405e6062e2.zip"
            ),
            Path(
                "bupt/Edge-Computing-Dataset-"
                "04e664fab9cdb2a58d04ebc615cd74405e6062e2.zip"
            ),
        ),
    )
    eua = _dataset_archive(
        root,
        "EUA",
        (
            Path(
                "EUA_Dataset/61238e00a6f01019c8a6ba1619675337a879a3d4/"
                "eua-dataset-61238e00a6f01019c8a6ba1619675337a879a3d4.zip"
            ),
            Path(
                "eua/eua-dataset-61238e00a6f01019c8a6ba1619675337a879a3d4.zip"
            ),
        ),
    )
    return uci, bupt, eua


def _pools(config: HybridConfig, datasets_root: Path = DATASETS) -> PhysicalPools:
    if POOL_CACHE.exists():
        with POOL_CACHE.open("rb") as handle:
            cached: PhysicalPools = pickle.load(handle)
        return cached
    uci, bupt, eua = _dataset_archives(datasets_root)
    pools = load_physical_pools(
        uci_zip=uci,
        bupt_zip=bupt,
        eua_zip=eua,
        window=config.episode_tasks,
        key=PSEUDONYM_KEY,
    )
    POOL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with POOL_CACHE.open("wb") as handle:
        pickle.dump(pools, handle)
    return pools


V3 = MAIN / "benchmark_v3"
BELIEF_MODELS = {
    # short name: (prediction file stem, confidence provenance)
    "constant_profile": ("constant_profile", "calibrated_and_raw"),
    "rule": ("rule_keyword", "calibrated_and_raw"),
    "tfidf_aug": ("tfidf_lr_augmented", "calibrated_and_raw"),
    "e5_frozen_aug": ("e5_frozen_lr_augmented", "calibrated_and_raw"),
    # These cached predictions predate v3.2 and have no matching calibration
    # predictions/checkpoint. Their names explicitly prevent a calibrated claim.
    "e5_finetuned_v31": ("e5_finetuned_augmented", "legacy_raw"),
    "llm_fewshot": ("llm_fewshot_Qwen2.5-1.5B-Instruct", "legacy_raw"),
}
POOL_SETS = {
    "compositional": ("compositional600_canonical", "compositional600_paraphrase"),
    "compositional_subset200": (
        "compositional600_canonical",
        "compositional600_paraphrase",
    ),
    "independent": ("compositional600_canonical", "independent_paraphrase"),
    "human60": ("human60_original", "human60_paraphrase"),
}
SUBSET_SEED = 20261120
SUBSET_SIZE = 200


def subset200_ids() -> list[str]:
    """The pre-registered 200-task subset (same draw as the paraphrase builder)."""

    tasks = load_jsonl(V3 / "eval_compositional600_canonical.tasks.jsonl")
    chosen = np.random.default_rng(SUBSET_SEED).choice(
        len(tasks), size=SUBSET_SIZE, replace=False
    )
    return [tasks[int(i)]["task_id"] for i in sorted(chosen.tolist())]


def _by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {row["task_id"]: row["annotation"] for row in load_jsonl(path)}


def _semantic_sources(
    truth_kind: str = "design",
    pool: str = "compositional",
    prediction_variant: str = "calibrated",
    heldout_domains: frozenset[str] = frozenset(),
) -> tuple[list[SemanticItem], dict[str, TaskRequirements], dict[str, dict[str, Any]]]:
    """Training items, test truth, and test beliefs per (source, wording, fallback).

    Beliefs are the frozen v3 predictions (``benchmark_v3/predictions``); the
    extractors are never re-fitted here, so the semantic layer is identical to
    the one reported in the semantic benchmark.
    """

    split = load_json_object(MAIN / "split_manifest.v1.json")
    tasks = load_jsonl(MAIN / "tasks.v1.jsonl")
    dev = list(
        ordered_subset(tasks, list(split["machine_labeled_development"]["task_ids"]))
    )
    if heldout_domains:
        dev = [
            task
            for task in dev
            if str(task["design_strata"]["scenario_domain"])
            not in heldout_domains
        ]
        if not dev:
            raise ValueError("heldout domains remove every development task")
    eval_ids = list(split["single_human_evaluation"]["task_ids"])
    evaluation = ordered_subset(tasks, eval_ids)
    weak = _by_id(MAIN / "preannotations" / "machine_development.v1.jsonl")

    # PPO training semantics: development texts (weak labels) plus canonical
    # compositional development tasks (labels by construction); belief = truth.
    train_annotations = [weak[t["task_id"]] for t in dev]
    comp_train, comp_labels = compose_tasks(
        dev,
        variants_per_task=2,
        wording="canonical",
        seed=SEED + 101,
        prefix="ppo_train",
    )
    train_annotations += [comp_labels[t["task_id"]] for t in comp_train]
    train_items = [
        SemanticItem(
            str(a["task_id"]),
            annotation_to_requirements(a),
            annotation_to_requirements(a),
        )
        for a in train_annotations
    ]

    canonical_set, paraphrase_set = POOL_SETS[pool]
    if pool == "independent":
        truth_annotations = _by_id(V3 / "eval_independent_paraphrase.labels.jsonl")
    elif pool.startswith("compositional"):
        truth_annotations = _by_id(V3 / f"eval_{canonical_set}.labels.jsonl")
        if pool == "compositional_subset200":
            keep = set(subset200_ids())
            truth_annotations = {
                k: v for k, v in truth_annotations.items() if k in keep
            }
    elif truth_kind == "design":
        truth_annotations = {
            t["task_id"]: authoring_preannotation(t) for t in evaluation
        }
    else:
        truth_annotations = _by_id(
            MAIN / "reference_variants" / "annotator_a.protocol_normalized.v1.3.1.jsonl"
        )
    if heldout_domains:
        if pool == "independent":
            domain_tasks = load_jsonl(V3 / "eval_independent_paraphrase.tasks.jsonl")
        elif pool.startswith("compositional"):
            domain_tasks = load_jsonl(V3 / f"eval_{canonical_set}.tasks.jsonl")
        else:
            domain_tasks = evaluation
        domain_by_id = {
            str(task["task_id"]): str(task["design_strata"]["scenario_domain"])
            for task in domain_tasks
        }
        truth_annotations = {
            task_id: annotation
            for task_id, annotation in truth_annotations.items()
            if domain_by_id.get(task_id) in heldout_domains
        }
        if not truth_annotations:
            raise ValueError("heldout domains select no evaluation tasks")
    truth = {i: annotation_to_requirements(a) for i, a in truth_annotations.items()}

    if prediction_variant not in {"calibrated", "raw", "both"}:
        raise ValueError("prediction_variant must be calibrated, raw, or both")
    prediction_roots: list[tuple[str, str]] = []
    if prediction_variant in {"calibrated", "both"}:
        prediction_roots.append(("", "predictions"))
    if prediction_variant in {"raw", "both"}:
        prediction_roots.append(("_raw", "predictions_raw"))

    annotations: dict[str, dict[str, Any]] = {"reference": truth_annotations}
    for short, (model, confidence_provenance) in BELIEF_MODELS.items():
        model_roots = prediction_roots
        if confidence_provenance == "legacy_raw":
            model_roots = (
                [("_raw", "predictions")]
                if prediction_variant in {"raw", "both"}
                else []
            )
        for suffix, directory in model_roots:
            for wording, set_name in (
                ("canonical", canonical_set),
                ("paraphrase", paraphrase_set),
            ):
                path = V3 / directory / f"{model}.{set_name}.jsonl"
                subset_path = V3 / directory / f"{model}.{set_name}.subset200.jsonl"
                if pool == "independent" and wording == "canonical":
                    continue
                if not path.exists() and subset_path.exists():
                    path = subset_path
                if path.exists() and not (
                    short == "constant_profile" and wording == "paraphrase"
                ):
                    by_task = _by_id(path)
                    if all(k in by_task for k in truth_annotations):
                        annotations[f"{short}_{wording}{suffix}"] = {
                            k: by_task[k] for k in truth_annotations
                        }
    beliefs: dict[str, dict[str, Any]] = {}
    for source, by_task in annotations.items():
        for conservative in (False, True):
            name = f"{source}{'+fallback' if conservative else ''}"
            beliefs[name] = {
                task_id: annotation_to_requirements(
                    annotation, conservative=conservative
                )
                for task_id, annotation in by_task.items()
            }
    return train_items, truth, beliefs


def _with_beliefs(
    scenario: Scenario, used: Sequence[str], beliefs: Mapping[str, TaskRequirements]
) -> Scenario:
    tasks = tuple(
        dataclasses.replace(task, believed_requirements=beliefs[task_id])
        for task, task_id in zip(scenario.tasks, used, strict=True)
    )
    return dataclasses.replace(scenario, tasks=tasks)


def _run(
    scenarios: Sequence[Scenario],
    choose: Callable[[TaskOffloadingEnv, dict[str, Any]], int],
    *,
    physics: PhysicsConfig,
    semantic_observation: bool,
    noise: float = 0.0,
) -> dict[str, Any]:
    """Pooled metrics plus per-episode values for paired bootstrap CIs."""

    outcomes: list[OffloadOutcome] = []
    overhead_ms: list[float] = []
    reward_components: list[Any] = []
    per_episode: dict[str, list[float]] = {}
    for scenario in scenarios:
        env = TaskOffloadingEnv(
            scenario,
            physics,
            semantic_observation=semantic_observation,
            preview_noise_sigma=noise,
            noise_seed=SEED,
        )
        if not isinstance(env.action_space, Discrete):
            raise TypeError("task offloading requires a discrete action space")
        available_actions = int(env.action_space.n)
        observation, _ = env.reset(seed=0)
        done = False
        episode: list[OffloadOutcome] = []
        episode_overhead_ms: list[float] = []
        episode_reward_components: list[Any] = []
        while not done:
            decision_started = time.perf_counter_ns()
            action = choose(env, observation)
            episode_overhead_ms.append(
                (time.perf_counter_ns() - decision_started) / 1_000_000
            )
            observation, _, done, _, info = env.step(action)
            episode.append(info["outcome"])
            episode_reward_components.append(info["reward_components"])
        outcomes.extend(episode)
        overhead_ms.extend(episode_overhead_ms)
        reward_components.extend(episode_reward_components)
        episode_metrics = system_metrics(
            episode,
            arrival_times_s=[task.arrival_time_s for task in scenario.tasks],
            source_device_ids=[task.source_device_id for task in scenario.tasks],
            available_actions=available_actions,
            decision_overhead_ms=episode_overhead_ms,
        )
        for field in dataclasses.fields(episode_reward_components[0]):
            episode_metrics[f"reward_component_{field.name}_mean"] = float(
                np.mean(
                    [
                        getattr(component, field.name)
                        for component in episode_reward_components
                    ]
                )
            )
        episode_metrics["reward_cost_mean"] = float(
            np.mean(
                [
                    component.total_cost(physics.reward_weights)
                    for component in episode_reward_components
                ]
            )
        )
        for key, value in episode_metrics.items():
            per_episode.setdefault(key, []).append(value)
    result: dict[str, Any] = dict(
        system_metrics(
            outcomes,
            available_actions=available_actions,
            decision_overhead_ms=overhead_ms,
        )
    )
    episode_mean_metrics = {
        "completed_throughput_tps",
        "successful_throughput_tps",
        "simulated_horizon_s",
        "device_energy_episode_total_j",
        "source_success_jain_fairness",
        "active_source_count",
    }
    for key in episode_mean_metrics:
        result[key] = float(np.mean(per_episode[key]))
    result["device_energy_run_total_j"] = float(
        sum(item.device_energy_j for item in outcomes)
    )
    for field in dataclasses.fields(reward_components[0]):
        result[f"reward_component_{field.name}_mean"] = float(
            np.mean([getattr(component, field.name) for component in reward_components])
        )
    result["reward_cost_mean"] = float(
        np.mean(
            [
                component.total_cost(physics.reward_weights)
                for component in reward_components
            ]
        )
    )
    result["per_episode"] = per_episode
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=PPO_SEEDS)
    parser.add_argument("--timesteps", type=int, default=PPO_TIMESTEPS)
    parser.add_argument("--train-episodes", type=int, default=TRAIN_EPISODES)
    parser.add_argument("--test-episodes", type=int, default=TEST_EPISODES)
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--truth", choices=("design", "human"), default="design")
    parser.add_argument("--pool", choices=tuple(POOL_SETS), default="compositional")
    parser.add_argument(
        "--eval-only-from",
        default=None,
        help="tag of saved checkpoints to re-evaluate instead of training",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "reuse existing condition/seed checkpoints for --tag and train missing ones"
        ),
    )
    parser.add_argument("--nep", action="store_true", help="enable NEP layer")
    parser.add_argument("--preview-noise", type=float, default=0.0)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("TASK_OFFLOADING_DATASETS", DATASETS)),
        help="root containing the pinned UCI/BUPT/EUA archives",
    )
    parser.add_argument(
        "--prediction-variant",
        choices=("calibrated", "raw", "both"),
        default="calibrated",
        help="use isotonic-calibrated confidence, raw confidence, or both",
    )
    parser.add_argument(
        "--conditions",
        default="ppo_semantic,ppo_no_semantic,ppo_semantic_penalty",
    )
    parser.add_argument(
        "--penalty-all-beliefs",
        action="store_true",
        help="evaluate penalty PPO on every belief (default: H4 reference only)",
    )
    parser.add_argument(
        "--beliefs",
        default="",
        help="comma list of belief sources to evaluate (default: all)",
    )
    parser.add_argument(
        "--stress-share",
        type=float,
        default=0.0,
        help="share of test tasks drawn from execution-restricted texts",
    )
    parser.add_argument(
        "--physical-ablation",
        choices=(
            "none",
            "without_uci_heterogeneity",
            "without_bupt_variability",
            "without_eua_variability",
        ),
        default="none",
        help="remove one measured/observed source layer while preserving RNG shape",
    )
    parser.add_argument(
        "--eua-entity-holdout",
        action="store_true",
        help="train/test on disjoint seeded 80/20 EUA location rows",
    )
    parser.add_argument(
        "--heldout-domains",
        default="",
        help=(
            "comma-separated semantic domains excluded from PPO training and "
            "used exclusively in evaluation"
        ),
    )
    args = parser.parse_args()

    started = time.perf_counter()
    if args.nep:
        nep_layer = load_nep_layer(NEP_PROFILE)
        config = HybridConfig(
            nep_percentiles=nep_layer["nep_percentiles"],
            nep_edge_load_quantiles=nep_layer["nep_edge_load_quantiles"],
            nep_rtt_ms_quantiles=nep_layer["nep_rtt_ms_quantiles"],
        )
    else:
        config = HybridConfig()
    pools = ablate_physical_pools(
        _pools(HybridConfig(), args.data_root),
        args.physical_ablation,
        fit_seed=SEED,
    )
    if args.eua_entity_holdout and args.physical_ablation == "without_eua_variability":
        parser.error(
            "--eua-entity-holdout cannot be combined with --physical-ablation "
            "without_eua_variability"
        )
    if args.eua_entity_holdout:
        train_pools, test_pools = split_eua_entity_holdout(pools, seed=SEED)
    else:
        train_pools = test_pools = pools
    heldout_domains = frozenset(
        value.strip() for value in args.heldout_domains.split(",") if value.strip()
    )
    train_items, truth, beliefs = _semantic_sources(
        args.truth,
        args.pool,
        args.prediction_variant,
        heldout_domains,
    )
    if args.beliefs:
        wanted = {b.strip() for b in args.beliefs.split(",") if b.strip()}
        beliefs = {k: v for k, v in beliefs.items() if k.split("+")[0] in wanted}
    noise = args.preview_noise
    print(f"loaded pools/semantics in {time.perf_counter() - started:.0f}s", flush=True)

    rng = np.random.default_rng(SEED)
    train_scenarios = tuple(
        build_episode(
            rng=rng,
            pools=train_pools,
            windows=train_pools.train_windows,
            semantic_items=train_items,
            config=config,
        )[0]
        for _ in range(args.train_episodes)
    )
    test_rng = np.random.default_rng(SEED + 1)
    all_items = [SemanticItem(i, truth[i], truth[i]) for i in sorted(truth)]
    restricted = [
        item
        for item in all_items
        if item.true_requirements.allowed_tiers != frozenset(ExecutionTier)
    ]
    others = [item for item in all_items if item not in restricted]
    test_items = list(all_items)
    if args.stress_share > 0 and restricted:
        # Replicate restricted texts so they make up ``stress_share`` of draws.
        copies = max(
            1,
            round(
                args.stress_share
                * len(others)
                / ((1 - args.stress_share) * len(restricted))
            ),
        )
        test_items = others + restricted * copies
    base_test = [
        build_episode(
            rng=test_rng,
            pools=test_pools,
            windows=test_pools.test_windows,
            semantic_items=test_items,
            config=config,
        )
        for _ in range(args.test_episodes)
    ]
    test_by_belief = {
        name: [_with_beliefs(s, used, belief) for s, used in base_test]
        for name, belief in beliefs.items()
    }
    reference_test = test_by_belief["reference"]

    partial_path = OUTPUT / f"results_{args.tag}.partial.json"
    results: dict[str, Any] = {"baselines": {}, "ppo": {}}
    if args.resume and partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        previous = partial.get("results")
        if isinstance(previous, dict) and isinstance(previous.get("ppo"), dict):
            results["ppo"] = previous["ppo"]

    def heuristic(policy: Any) -> Callable[[TaskOffloadingEnv, dict[str, Any]], int]:
        return lambda env, _obs: policy(env)

    for label in ("local", "edge_0", "cloud"):
        results["baselines"][f"always_{label}"] = _run(
            reference_test,
            heuristic(fixed_target(label)),
            physics=NO_SEMANTIC_PHYSICS,
            semantic_observation=False,
            noise=noise,
        )
    results["baselines"]["random_valid"] = _run(
        reference_test,
        heuristic(random_valid(SEED)),
        physics=NO_SEMANTIC_PHYSICS,
        semantic_observation=False,
        noise=noise,
    )
    results["baselines"]["greedy_no_semantic"] = _run(
        reference_test,
        heuristic(greedy(semantic=False)),
        physics=NO_SEMANTIC_PHYSICS,
        semantic_observation=False,
        noise=noise,
    )
    for name, scenarios in test_by_belief.items():
        results["baselines"][f"greedy_semantic[{name}]"] = _run(
            scenarios,
            heuristic(greedy(semantic=True)),
            physics=PHYSICS,
            semantic_observation=True,
            noise=noise,
        )
    print(f"baselines done {time.perf_counter() - started:.0f}s", flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    condition_specs = {
        # name: (semantic observation, semantic mask)
        "ppo_semantic": (True, True),
        "ppo_no_semantic": (False, False),
        "ppo_semantic_penalty": (True, False),
    }
    for condition in [c.strip() for c in args.conditions.split(",") if c.strip()]:
        semantic, masked = condition_specs[condition]
        physics = PHYSICS if masked else NO_SEMANTIC_PHYSICS
        for seed in range(args.seeds):
            sets = test_by_belief if semantic else {"reference": reference_test}
            if condition == "ppo_semantic_penalty" and not args.penalty_all_beliefs:
                sets = {"reference": reference_test}
            result_keys = [
                f"{condition}[{name}]" if semantic else condition for name in sets
            ]
            if result_keys and all(
                len(results["ppo"].get(key, [])) > seed for key in result_keys
            ):
                print(f"{condition} seed {seed} restored from partial", flush=True)
                continue
            env = ScenarioPoolEnv(
                train_scenarios,
                physics,
                semantic_observation=semantic,
                preview_noise_sigma=noise,
                noise_seed=SEED,
            )
            checkpoint_tag = args.eval_only_from or args.tag
            checkpoint = OUTPUT / f"{condition}_seed{seed}_{checkpoint_tag}.zip"
            if args.eval_only_from or (args.resume and checkpoint.exists()):
                model = MaskablePPO.load(
                    checkpoint,
                    env=env,
                    device="cpu",
                )
            else:
                model = MaskablePPO(
                    "MultiInputPolicy",
                    env,
                    seed=SEED + seed,
                    verbose=0,
                    **PPO_HYPERPARAMETERS,
                    device="cpu",
                )
                model.learn(total_timesteps=args.timesteps)
                model.save(checkpoint)

            def ppo_choice(
                env: TaskOffloadingEnv, obs: dict[str, Any], model: Any = model
            ) -> int:
                action, _ = model.predict(
                    obs, action_masks=env.action_masks(), deterministic=True
                )
                return int(action)

            for name, scenarios in sets.items():
                key = f"{condition}[{name}]" if semantic else condition
                results["ppo"].setdefault(key, []).append(
                    _run(
                        scenarios,
                        ppo_choice,
                        physics=physics,
                        semantic_observation=semantic,
                        noise=noise,
                    )
                )
            print(
                f"{condition} seed {seed} done {time.perf_counter() - started:.0f}s",
                flush=True,
            )
            partial_path.write_text(
                json.dumps(
                    {
                        "experiment_version": args.tag,
                        "incomplete": True,
                        "completed_condition": condition,
                        "completed_seed": seed,
                        "results": results,
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )

    summary = {
        "experiment_version": args.tag,
        "truth": args.truth if args.pool == "human60" else "label_by_construction",
        "pool": args.pool,
        "stress_share": args.stress_share,
        "nep_layer": args.nep,
        "preview_noise_sigma": noise,
        "prediction_variant": args.prediction_variant,
        "physical_ablation": args.physical_ablation,
        "eua_entity_holdout": args.eua_entity_holdout,
        "eua_train_location_rows": len(train_pools.user_edge_distances_m),
        "eua_test_location_rows": len(test_pools.user_edge_distances_m),
        "heldout_domains": sorted(heldout_domains),
        "fallback_confidence_floor": 0.6,
        "conditions": args.conditions,
        "design": __doc__,
        "seeds": args.seeds,
        "timesteps": args.timesteps,
        "checkpoints_from": args.eval_only_from or args.tag,
        "resumed_existing_checkpoints": args.resume,
        "train_episodes": args.train_episodes,
        "test_episodes": args.test_episodes,
        "tasks_per_episode": config.episode_tasks,
        "hybrid_config": dataclasses.asdict(config),
        "physics_config": dataclasses.asdict(PHYSICS),
        "results": results,
    }
    path = OUTPUT / f"results_{args.tag}.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"wrote {path} in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
