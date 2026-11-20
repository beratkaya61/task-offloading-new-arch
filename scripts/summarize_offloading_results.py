# ruff: noqa: E501, RUF001
"""Publication-style tables for the offloading experiment (paired, with CIs).

Unit of resampling: the test episode (all policies saw the same episodes).
PPO values are averaged over training seeds per episode before resampling;
the across-seed standard deviation of the pooled metric is reported too.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    ("success_rate", "Başarı ↑", 100, 1),
    ("hard_violation_rate", "Kesin kısıt ihlali ↓", 100, 2),
    ("deadline_miss_rate", "Deadline kaçırma ↓", 100, 1),
    ("reliability_violation_rate", "Güvenilirlik ihlali ↓", 100, 1),
    ("accuracy_violation_rate", "Doğruluk ihlali ↓", 100, 1),
    ("latency_mean_ms", "Ort. gecikme (ms) ↓", 1, 1),
    ("device_energy_mean_mj", "Cihaz enerjisi (mJ) ↓", 1, 2),
)
RESAMPLES = 2000


def _episodes(entry: Any) -> dict[str, np.ndarray]:
    runs = entry if isinstance(entry, list) else [entry]
    return {
        metric: np.mean(
            [np.asarray(run["per_episode"][metric], dtype=float) for run in runs],
            axis=0,
        )
        for metric, *_ in METRICS
    }


def _seed_std(entry: Any, metric: str) -> float | None:
    if not isinstance(entry, list) or len(entry) < 2:
        return None
    return float(np.std([run[metric] for run in entry], ddof=1))


def _ci(values: np.ndarray, draws: np.ndarray) -> tuple[float, float, float]:
    samples = values[draws].mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.results.read_text(encoding="utf-8"))
    rows: dict[str, Any] = {**data["results"]["baselines"], **data["results"]["ppo"]}
    episodes = {name: _episodes(entry) for name, entry in rows.items()}
    count = len(next(iter(episodes.values()))["success_rate"])
    rng = np.random.default_rng(20261120)
    draws = rng.integers(0, count, size=(RESAMPLES, count))

    def fmt(name: str, metric: str, scale: float, digits: int) -> str:
        mean, low, high = _ci(episodes[name][metric], draws)
        text = f"{scale * mean:.{digits}f} [{scale * low:.{digits}f}, {scale * high:.{digits}f}]"
        std = _seed_std(rows[name], metric)
        return text if std is None else f"{text} (±{scale * std:.{digits}f})"

    def diff(name: str, base: str, metric: str, scale: float, digits: int) -> str:
        delta = episodes[name][metric] - episodes[base][metric]
        mean, low, high = _ci(delta, draws)
        samples = delta[draws].mean(axis=1)
        p = float(min(1.0, 2 * min(np.mean(samples <= 0), np.mean(samples >= 0))))
        p_text = "<0.001" if p < 0.001 else f"{p:.3f}"
        return (
            f"{scale * mean:+.{digits}f} [{scale * low:+.{digits}f}, "
            f"{scale * high:+.{digits}f}]; p={p_text}"
        )

    lines = [
        f"# Offloading Deneyi — {data['experiment_version']}",
        "",
        f"Test: {data['test_episodes']} bölüm × {data['tasks_per_episode']} görev "
        f"(toplam {count * data['tasks_per_episode']} karar), havuz: "
        f"`{data.get('pool', 'human60')}`, gerçek: `{data.get('truth')}`, "
        f"kısıt-yoğunluk payı: {data.get('stress_share', 0)}. PPO: {data['seeds']} tohum × "
        f"{data['timesteps']} adım. Değerler: ortalama [bölüm düzeyinde eşleştirilmiş "
        f"bootstrap %95 GA, {RESAMPLES} tekrar]; PPO için (± tohumlar arası std). "
        "Oranlar yüzde, gecikme ms, enerji mJ.",
        "",
        "## Tüm koşullar",
        "",
        "| Koşul | " + " | ".join(label for _, label, _, _ in METRICS) + " |",
        "|---|" + "---:|" * len(METRICS),
    ]
    for name in rows:
        lines.append(
            f"| {name} | "
            + " | ".join(fmt(name, m, s, d) for m, _, s, d in METRICS)
            + " |"
        )
    for base in ("ppo_no_semantic", "greedy_no_semantic"):
        if base not in rows:
            continue
        family = "ppo_semantic[" if base.startswith("ppo") else "greedy_semantic["
        lines += [
            "",
            f"## Semantiksiz `{base}` koşuluna göre eşleştirilmiş fark",
            "",
            "| Koşul | Başarı farkı (puan) | Kesin ihlal farkı (puan) | Deadline farkı (puan) |",
            "|---|---|---|---|",
        ]
        for name in rows:
            if name.startswith(family):
                lines.append(
                    f"| {name} | {diff(name, base, 'success_rate', 100, 1)} | "
                    f"{diff(name, base, 'hard_violation_rate', 100, 2)} | "
                    f"{diff(name, base, 'deadline_miss_rate', 100, 1)} |"
                )
    text = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    print(text)


if __name__ == "__main__":
    main()
