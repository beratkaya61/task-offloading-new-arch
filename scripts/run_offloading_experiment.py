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
import pickle
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
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
    annotation_to_requirements,
    build_episode,
    load_nep_layer,
    load_physical_pools,
)

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "data" / "semantic_benchmark" / "main"
DATASETS = Path("D:/task_offloading_datasets")
UCI = DATASETS / "uci" / "TATDescriptionDataset.zip"
BUPT = (
    DATASETS
    / "bupt"
    / "Edge-Computing-Dataset-04e664fab9cdb2a58d04ebc615cd74405e6062e2.zip"
)
EUA = DATASETS / "eua" / "eua-dataset-61238e00a6f01019c8a6ba1619675337a879a3d4.zip"
POOL_CACHE = ROOT / "artifacts" / "trace_hybrid_pools.v1.pkl"
NEP_PROFILE = ROOT / "configs" / "data_profiles" / "nep_quantiles_v1.json"
OUTPUT = ROOT / "artifacts" / "offloading_experiment"
PSEUDONYM_KEY = b"thesis-local-only-bupt-pseudonym-key-v1"
SEED = EXPERIMENT_SEED
PHYSICS = PhysicsConfig(
    latency_reference_s=LATENCY_REFERENCE_S, energy_reference_j=ENERGY_REFERENCE_J
)
NO_SEMANTIC_PHYSICS = dataclasses.replace(PHYSICS, mask_semantic_permissions=False)


def _pools(config: HybridConfig) -> PhysicalPools:
    if POOL_CACHE.exists():
        with POOL_CACHE.open("rb") as handle:
            cached: PhysicalPools = pickle.load(handle)
        return cached
    pools = load_physical_pools(
        uci_zip=UCI,
        bupt_zip=BUPT,
        eua_zip=EUA,
        window=config.episode_tasks,
        key=PSEUDONYM_KEY,
    )
    POOL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with POOL_CACHE.open("wb") as handle:
        pickle.dump(pools, handle)
    return pools


V3 = MAIN / "benchmark_v3"
BELIEF_MODELS = {
    "constant_profile": "constant_profile",
    "rule": "rule_keyword",
    "tfidf_aug": "tfidf_lr_augmented",
    "e5_frozen_aug": "e5_frozen_lr_augmented",
    "e5_finetuned_aug": "e5_finetuned_augmented",
    "llm_fewshot": "llm_fewshot_Qwen2.5-1.5B-Instruct",
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
    truth_kind: str = "design", pool: str = "compositional"
) -> tuple[list[SemanticItem], dict[str, TaskRequirements], dict[str, dict[str, Any]]]:
    """Training items, test truth, and test beliefs per (source, wording, fallback).

    Beliefs are the frozen v3 predictions (``benchmark_v3/predictions``); the
    extractors are never re-fitted here, so the semantic layer is identical to
    the one reported in the semantic benchmark.
    """

    split = load_json_object(MAIN / "split_manifest.v1.json")
    tasks = load_jsonl(MAIN / "tasks.v1.jsonl")
    dev = ordered_subset(tasks, list(split["machine_labeled_development"]["task_ids"]))
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
    truth = {i: annotation_to_requirements(a) for i, a in truth_annotations.items()}

    annotations: dict[str, dict[str, Any]] = {"reference": truth_annotations}
    for short, model in BELIEF_MODELS.items():
        for wording, set_name in (
            ("canonical", canonical_set),
            ("paraphrase", paraphrase_set),
        ):
            path = V3 / "predictions" / f"{model}.{set_name}.jsonl"
            subset_path = V3 / "predictions" / f"{model}.{set_name}.subset200.jsonl"
            if pool == "independent" and wording == "canonical":
                continue
            if not path.exists() and subset_path.exists():
                path = subset_path
            if path.exists() and not (
                short == "constant_profile" and wording == "paraphrase"
            ):
                by_task = _by_id(path)
                if all(k in by_task for k in truth_annotations):
                    annotations[f"{short}_{wording}"] = {
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


def _metrics(outcomes: Sequence[OffloadOutcome]) -> dict[str, float]:
    latency = np.asarray([item.total_latency_s for item in outcomes])
    tiers = [item.tier for item in outcomes]
    return {
        "decisions": float(len(outcomes)),
        "success_rate": float(np.mean([item.success for item in outcomes])),
        "hard_violation_rate": float(
            np.mean([item.hard_constraint_violated for item in outcomes])
        ),
        "deadline_miss_rate": float(
            np.mean([item.deadline_missed for item in outcomes])
        ),
        "reliability_violation_rate": float(
            np.mean([item.reliability_violated for item in outcomes])
        ),
        "accuracy_violation_rate": float(
            np.mean([item.accuracy_violated for item in outcomes])
        ),
        "failure_rate": float(np.mean([not item.completed for item in outcomes])),
        "latency_mean_ms": float(1000 * latency.mean()),
        "latency_p95_ms": float(1000 * np.percentile(latency, 95)),
        "device_energy_mean_mj": float(
            1000 * np.mean([item.device_energy_j for item in outcomes])
        ),
        "share_local": float(np.mean([t is ExecutionTier.DEVICE for t in tiers])),
        "share_edge": float(np.mean([t is ExecutionTier.EDGE for t in tiers])),
        "share_cloud": float(np.mean([t is ExecutionTier.CLOUD for t in tiers])),
    }


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
    per_episode: dict[str, list[float]] = {}
    for scenario in scenarios:
        env = TaskOffloadingEnv(
            scenario,
            physics,
            semantic_observation=semantic_observation,
            preview_noise_sigma=noise,
            noise_seed=SEED,
        )
        observation, _ = env.reset(seed=0)
        done = False
        episode: list[OffloadOutcome] = []
        while not done:
            action = choose(env, observation)
            observation, _, done, _, info = env.step(action)
            episode.append(info["outcome"])
        outcomes.extend(episode)
        for key, value in _metrics(episode).items():
            per_episode.setdefault(key, []).append(value)
    result: dict[str, Any] = dict(_metrics(outcomes))
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
    parser.add_argument("--nep", action="store_true", help="enable NEP layer")
    parser.add_argument("--preview-noise", type=float, default=0.0)
    parser.add_argument(
        "--conditions",
        default="ppo_semantic,ppo_no_semantic,ppo_semantic_penalty",
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
    args = parser.parse_args()

    started = time.perf_counter()
    config = HybridConfig(**load_nep_layer(NEP_PROFILE)) if args.nep else HybridConfig()
    pools = _pools(HybridConfig())
    train_items, truth, beliefs = _semantic_sources(args.truth, args.pool)
    if args.beliefs:
        wanted = {b.strip() for b in args.beliefs.split(",") if b.strip()}
        beliefs = {k: v for k, v in beliefs.items() if k.split("+")[0] in wanted}
    noise = args.preview_noise
    print(f"loaded pools/semantics in {time.perf_counter() - started:.0f}s", flush=True)

    rng = np.random.default_rng(SEED)
    train_scenarios = tuple(
        build_episode(
            rng=rng,
            pools=pools,
            windows=pools.train_windows,
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
            pools=pools,
            windows=pools.test_windows,
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

    results: dict[str, Any] = {"baselines": {}, "ppo": {}}

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
            env = ScenarioPoolEnv(
                train_scenarios,
                physics,
                semantic_observation=semantic,
                preview_noise_sigma=noise,
                noise_seed=SEED,
            )
            if args.eval_only_from:
                model = MaskablePPO.load(
                    OUTPUT / f"{condition}_seed{seed}_{args.eval_only_from}.zip",
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
                model.save(OUTPUT / f"{condition}_seed{seed}_{args.tag}.zip")

            def ppo_choice(
                env: TaskOffloadingEnv, obs: dict[str, Any], model: Any = model
            ) -> int:
                action, _ = model.predict(
                    obs, action_masks=env.action_masks(), deterministic=True
                )
                return int(action)

            sets = test_by_belief if semantic else {"reference": reference_test}
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

    summary = {
        "experiment_version": args.tag,
        "truth": args.truth if args.pool == "human60" else "label_by_construction",
        "pool": args.pool,
        "stress_share": args.stress_share,
        "nep_layer": args.nep,
        "preview_noise_sigma": noise,
        "conditions": args.conditions,
        "design": __doc__,
        "seeds": args.seeds,
        "timesteps": args.timesteps,
        "checkpoints_from": args.eval_only_from or args.tag,
        "train_episodes": args.train_episodes,
        "test_episodes": args.test_episodes,
        "tasks_per_episode": config.episode_tasks,
        "hybrid_config": dataclasses.asdict(config),
        "results": results,
    }
    path = OUTPUT / f"results_{args.tag}.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"wrote {path} in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
