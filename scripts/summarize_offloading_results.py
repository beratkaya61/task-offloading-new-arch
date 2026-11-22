# ruff: noqa: RUF001
"""Publication tables for paired offloading experiments with Holm correction.

The unit of resampling is the complete test episode. PPO runs are averaged
over training seeds within an episode before paired bootstrap resampling; the
pooled metric's across-seed standard deviation is shown separately.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

Metric = tuple[str, str, float, int]
PRIMARY: tuple[Metric, ...] = (
    ("success_rate", "Başarı ↑", 100, 1),
    ("hard_violation_rate", "Kesin kısıt ihlali ↓", 100, 2),
    ("deadline_miss_rate", "Deadline kaçırma ↓", 100, 1),
    ("reliability_violation_rate", "Güvenilirlik ihlali ↓", 100, 1),
    ("accuracy_violation_rate", "Doğruluk ihlali ↓", 100, 1),
    ("drop_rate", "Drop ↓", 100, 2),
)
TAIL_AND_RESOURCE: tuple[Metric, ...] = (
    ("latency_mean_ms", "Ort. latency (ms) ↓", 1, 1),
    ("latency_p50_ms", "p50 (ms) ↓", 1, 1),
    ("latency_p95_ms", "p95 (ms) ↓", 1, 1),
    ("latency_p99_ms", "p99 (ms) ↓", 1, 1),
    ("queue_wait_mean_ms", "Ort. queue wait (ms) ↓", 1, 1),
    ("completed_throughput_tps", "Completed task/s ↑", 1, 2),
    ("successful_throughput_tps", "Successful task/s ↑", 1, 2),
    ("device_energy_mean_mj", "Ort. device energy (mJ) ↓", 1, 2),
    ("device_energy_episode_total_j", "Episode energy (J) ↓", 1, 3),
    ("device_energy_per_success_mj", "Energy/success (mJ) ↓", 1, 2),
)
POLICY_DIAGNOSTICS: tuple[Metric, ...] = (
    ("action_entropy_normalized", "Normalized action entropy ↑", 1, 3),
    ("action_max_share", "Collapse/max action share ↓", 100, 1),
    ("action_load_jain_fairness", "Jain load fairness ↑", 1, 3),
    ("source_success_jain_fairness", "Jain source-success fairness ↑", 1, 3),
    ("active_source_count", "Active sources", 1, 0),
    ("decision_overhead_mean_ms", "Decision overhead mean (ms) ↓", 1, 3),
    ("decision_overhead_p95_ms", "Decision overhead p95 (ms) ↓", 1, 3),
)
REWARD_COMPONENTS: tuple[Metric, ...] = (
    ("reward_component_latency_norm_mean", "Latency", 1, 3),
    ("reward_component_energy_norm_mean", "Energy", 1, 3),
    ("reward_component_deadline_miss_mean", "Deadline", 1, 3),
    ("reward_component_soft_constraint_violation_mean", "Soft constraint", 1, 3),
    ("reward_component_failure_or_drop_mean", "Failure/drop", 1, 3),
    ("reward_component_hard_constraint_violation_mean", "Hard constraint", 1, 3),
    ("reward_cost_mean", "Weighted total cost ↓", 1, 3),
)
RESAMPLES = 2000


def _available_metrics(
    rows: Mapping[str, Any], metrics: Sequence[Metric]
) -> list[Metric]:
    first = next(iter(rows.values()))
    first_run = first[0] if isinstance(first, list) else first
    episode_metrics = first_run["per_episode"]
    return [item for item in metrics if item[0] in episode_metrics]


def _episodes(entry: Any, metrics: Sequence[Metric]) -> dict[str, np.ndarray]:
    runs = entry if isinstance(entry, list) else [entry]
    return {
        metric: np.mean(
            [np.asarray(run["per_episode"][metric], dtype=float) for run in runs],
            axis=0,
        )
        for metric, *_ in metrics
    }


def _seed_std(entry: Any, metric: str) -> float | None:
    if not isinstance(entry, list) or len(entry) < 2:
        return None
    return float(np.std([run[metric] for run in entry], ddof=1))


def _ci(values: np.ndarray, draws: np.ndarray) -> tuple[float, float, float]:
    samples = values[draws].mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def _contrast(
    left: np.ndarray, right: np.ndarray, draws: np.ndarray
) -> tuple[float, float, float, float]:
    delta = left - right
    samples = delta[draws].mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    p_value = float(
        min(1.0, 2 * min(float(np.mean(samples <= 0)), float(np.mean(samples >= 0))))
    )
    return float(delta.mean()), float(low), float(high), p_value


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values in the input order."""

    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _p_text(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _metric_table(
    title: str,
    rows: Mapping[str, Any],
    episodes: Mapping[str, Mapping[str, np.ndarray]],
    metrics: Sequence[Metric],
    draws: np.ndarray,
) -> list[str]:
    if not metrics:
        return []
    lines = [
        f"## {title}",
        "",
        "| Koşul | " + " | ".join(label for _, label, _, _ in metrics) + " |",
        "|---|" + "---:|" * len(metrics),
    ]
    for name, entry in rows.items():
        values = []
        for metric, _, scale, digits in metrics:
            mean, low, high = _ci(episodes[name][metric], draws)
            text = (
                f"{scale * mean:.{digits}f} "
                f"[{scale * low:.{digits}f}, {scale * high:.{digits}f}]"
            )
            std = _seed_std(entry, metric)
            if std is not None:
                text += f" (±{scale * std:.{digits}f})"
            values.append(text)
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    lines.append("")
    return lines


def _family_contrasts(
    *,
    base: str,
    prefix: str,
    rows: Mapping[str, Any],
    episodes: Mapping[str, Mapping[str, np.ndarray]],
    draws: np.ndarray,
) -> list[str]:
    names = [name for name in rows if name.startswith(prefix)]
    if base not in rows or not names:
        return []
    metrics = (
        ("success_rate", "Başarı farkı (puan)"),
        ("hard_violation_rate", "Kesin ihlal farkı (puan)"),
        ("deadline_miss_rate", "Deadline farkı (puan)"),
    )
    contrasts: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    adjusted: dict[tuple[str, str], float] = {}
    for metric, _ in metrics:
        family = [
            _contrast(episodes[name][metric], episodes[base][metric], draws)
            for name in names
        ]
        for name, values, p_adj in zip(
            names, family, holm_adjust([item[3] for item in family]), strict=True
        ):
            contrasts[(name, metric)] = values
            adjusted[(name, metric)] = p_adj

    lines = [
        f"## `{base}` koşuluna göre eşleştirilmiş fark",
        "",
        "Her metrik sütunundaki p-değerleri o koşul ailesi içinde Holm ile "
        "düzeltilmiştir.",
        "",
        "| Koşul | " + " | ".join(label for _, label in metrics) + " |",
        "|---|" + "---:|" * len(metrics),
    ]
    for name in names:
        cells = []
        for metric, _ in metrics:
            mean, low, high, raw = contrasts[(name, metric)]
            cells.append(
                f"{100 * mean:+.1f} [{100 * low:+.1f}, {100 * high:+.1f}]; "
                f"p={_p_text(raw)}, p_Holm={_p_text(adjusted[(name, metric)])}"
            )
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _ppo_vs_greedy(
    rows: Mapping[str, Any],
    episodes: Mapping[str, Mapping[str, np.ndarray]],
    draws: np.ndarray,
) -> list[str]:
    pairs = []
    for ppo in rows:
        if not ppo.startswith("ppo_semantic["):
            continue
        greedy = ppo.replace("ppo_semantic[", "greedy_semantic[", 1)
        if greedy in rows:
            pairs.append((ppo, greedy))
    if not pairs:
        return []
    metrics = ("success_rate", "hard_violation_rate")
    results: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    adjusted: dict[tuple[str, str], float] = {}
    for metric in metrics:
        values = [
            _contrast(episodes[ppo][metric], episodes[greedy][metric], draws)
            for ppo, greedy in pairs
        ]
        for (ppo, _), value, p_adj in zip(
            pairs, values, holm_adjust([item[3] for item in values]), strict=True
        ):
            results[(ppo, metric)] = value
            adjusted[(ppo, metric)] = p_adj
    lines = [
        "## PPO − greedy doğrudan karşılaştırması",
        "",
        "Pozitif başarı farkı PPO lehinedir; negatif ihlal farkı PPO lehinedir. "
        "p-değerleri metrik ailesi içinde Holm-düzeltilmiştir.",
        "",
        "| Aynı semantik koşul | Başarı farkı (puan) | Kesin ihlal farkı (puan) |",
        "|---|---:|---:|",
    ]
    for ppo, greedy in pairs:
        cells = []
        for metric in metrics:
            mean, low, high, raw = results[(ppo, metric)]
            cells.append(
                f"{100 * mean:+.1f} [{100 * low:+.1f}, {100 * high:+.1f}]; "
                f"p={_p_text(raw)}, p_Holm={_p_text(adjusted[(ppo, metric)])}"
            )
        lines.append(f"| {ppo} vs {greedy} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.results.read_text(encoding="utf-8"))
    rows: dict[str, Any] = {**data["results"]["baselines"], **data["results"]["ppo"]}
    all_metrics = tuple(
        item
        for group in (PRIMARY, TAIL_AND_RESOURCE, POLICY_DIAGNOSTICS, REWARD_COMPONENTS)
        for item in group
    )
    present = _available_metrics(rows, all_metrics)
    episodes = {name: _episodes(entry, present) for name, entry in rows.items()}
    count = len(next(iter(episodes.values()))["success_rate"])
    rng = np.random.default_rng(20261120)
    draws = rng.integers(0, count, size=(RESAMPLES, count))

    lines = [
        f"# Offloading Deneyi — {data['experiment_version']}",
        "",
        f"Test: {data['test_episodes']} bölüm × {data['tasks_per_episode']} görev "
        f"(toplam {count * data['tasks_per_episode']} karar), havuz: "
        f"`{data.get('pool', 'human60')}`, gerçek: `{data.get('truth')}`, "
        f"NEP: `{data.get('nep_layer', False)}`, preview noise σ: "
        f"`{data.get('preview_noise_sigma', 0)}`, prediction confidence: "
        f"`{data.get('prediction_variant', 'legacy/raw')}`. PPO: {data['seeds']} "
        f"tohum × {data['timesteps']} adım. Değerler bölüm düzeyinde eşleştirilmiş "
        f"bootstrap ortalaması [%95 GA, {RESAMPLES} tekrar]; PPO için ayrıca "
        "tohumlar arası standart sapma (±) verilir.",
        "",
        "`Drop`, fiziksel olarak tamamlanmayan görev oranıdır; `throughput`, "
        "simüle edilen bölüm ufku başına tamamlanan/semantik olarak başarılı görev "
        "sayısıdır. Action entropy ve Jain load fairness tüm kullanılabilir hedefleri "
        "kapsar. Tek kaynak cihazlı mevcut topolojide source-success fairness "
        "mekanik olarak 1 olabilir; bu nedenle `Active sources` ile birlikte "
        "yorumlanmalıdır. Decision overhead yalnız policy seçiminin duvar saati "
        "maliyetidir, fiziksel latency'ye eklenmez.",
        "",
    ]
    for title, group in (
        ("Birincil sonuçlar", PRIMARY),
        ("Tail latency, queue, throughput ve energy", TAIL_AND_RESOURCE),
        ("Policy diagnostics, fairness, collapse ve overhead", POLICY_DIAGNOSTICS),
        ("Reward component decomposition", REWARD_COMPONENTS),
    ):
        metrics = [item for item in group if item in present]
        lines += _metric_table(title, rows, episodes, metrics, draws)
    lines += _family_contrasts(
        base="ppo_no_semantic",
        prefix="ppo_semantic[",
        rows=rows,
        episodes=episodes,
        draws=draws,
    )
    lines += _family_contrasts(
        base="greedy_no_semantic",
        prefix="greedy_semantic[",
        rows=rows,
        episodes=episodes,
        draws=draws,
    )
    lines += _ppo_vs_greedy(rows, episodes, draws)
    text = "\n".join(lines).rstrip() + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    print(text)


if __name__ == "__main__":
    main()


__all__ = ["holm_adjust"]
