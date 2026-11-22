"""One-at-a-time reward-weight sensitivity on fixed test trajectories.

This is a scoring sensitivity analysis, not PPO retraining. It asks whether
the observed policy ranking is an artifact of one scalarization after every
policy has already acted on the same paired episodes.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from task_offloading.domain import RewardWeights

FIELDS = tuple(RewardWeights.__dataclass_fields__)
RESAMPLES = 2000


def _episode_components(entry: Any) -> dict[str, np.ndarray]:
    runs = entry if isinstance(entry, list) else [entry]
    return {
        field: np.mean(
            [
                np.asarray(
                    run["per_episode"][f"reward_component_{field}_mean"],
                    dtype=float,
                )
                for run in runs
            ],
            axis=0,
        )
        for field in FIELDS
    }


def _cost(
    components: Mapping[str, np.ndarray], weights: Mapping[str, float]
) -> np.ndarray:
    result = np.zeros_like(components[FIELDS[0]], dtype=float)
    for field in FIELDS:
        result = result + components[field] * weights[field]
    return result


def _profiles(base: Mapping[str, float]) -> dict[str, dict[str, float]]:
    output = {"base": dict(base)}
    for field in FIELDS:
        for factor in (0.5, 2.0):
            profile = dict(base)
            profile[field] *= factor
            output[f"{field} x{factor:g}"] = profile
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.results.read_text(encoding="utf-8"))
    rows: dict[str, Any] = {**data["results"]["baselines"], **data["results"]["ppo"]}
    available = {
        name: entry
        for name, entry in rows.items()
        if "reward_component_latency_norm_mean"
        in (entry[0] if isinstance(entry, list) else entry)["per_episode"]
    }
    if not available:
        raise ValueError("result file does not contain reward component episodes")
    components = {
        name: _episode_components(entry) for name, entry in available.items()
    }
    base = dataclasses.asdict(RewardWeights())
    stored = data.get("physics_config", {}).get("reward_weights")
    if isinstance(stored, dict) and set(stored) == set(FIELDS):
        base = {field: float(stored[field]) for field in FIELDS}

    reference_ppo = "ppo_semantic[reference]"
    reference_greedy = "greedy_semantic[reference]"
    rng = np.random.default_rng(20261120)
    episode_count = len(next(iter(components.values()))[FIELDS[0]])
    draws = rng.integers(0, episode_count, size=(RESAMPLES, episode_count))
    lines = [
        f"# Reward Sensitivity - {data['experiment_version']}",
        "",
        "Bu tablo ayni test trajectories uzerindeki reward components'i yeniden "
        "agirliklandirir; PPO yeniden egitilmemistir. Bu nedenle scalar cost "
        "ranking robustness kanitidir, training sensitivity iddiasi degildir.",
        "",
        "| Weight profile | En dusuk cost kosulu | Cost | PPO-reference rank | "
        "Greedy-reference rank | PPO - greedy cost [%95 GA] |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for profile_name, weights in _profiles(base).items():
        costs = {name: _cost(value, weights) for name, value in components.items()}
        ranking = sorted(costs, key=lambda name: float(costs[name].mean()))
        best = ranking[0]
        ppo_rank = (
            ranking.index(reference_ppo) + 1 if reference_ppo in ranking else None
        )
        greedy_rank = (
            ranking.index(reference_greedy) + 1 if reference_greedy in ranking else None
        )
        if reference_ppo in costs and reference_greedy in costs:
            delta = costs[reference_ppo] - costs[reference_greedy]
            samples = delta[draws].mean(axis=1)
            low, high = np.percentile(samples, [2.5, 97.5])
            contrast = f"{delta.mean():+.3f} [{low:+.3f}, {high:+.3f}]"
        else:
            contrast = "n/a"
        lines.append(
            f"| {profile_name} | {best} | {costs[best].mean():.3f} | "
            f"{ppo_rank or 'n/a'} | {greedy_rank or 'n/a'} | {contrast} |"
        )
    lines += [
        "",
        "Base weights: `" + json.dumps(base, sort_keys=True) + "`.",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
