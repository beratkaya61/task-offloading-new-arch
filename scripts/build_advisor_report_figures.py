"""Build figures embedded in the thesis-advisor progress report.

The values are the frozen publication-facing v3.2 semantic, NEP-v2 offloading,
and Phase 7 generalization results.  Keeping the figure source in the repository
prevents hand-edited charts from drifting away from the evidence tables.
"""

# ruff: noqa: RUF001 -- Turkish figure labels intentionally contain dotless i.

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "advisor_assets"
MPL_CONFIG = ROOT / "artifacts" / "matplotlib_cache"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def semantic_figure() -> None:
    methods = [
        "Kural",
        "TF-IDF\n(+veri)",
        "E5 donuk\n(+veri)",
        "E5 ince ayar\n(+veri)",
        "Qwen2.5\n1.5B 4-shot",
    ]
    original = np.array([78.0, 78.2, 76.3, 70.2, 48.3])
    independent = np.array([54.6, 80.9, 73.8, np.nan, 42.0])

    x = np.arange(len(methods))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    groups = [
        ax.bar(
            x - width / 2,
            original,
            width,
            label="Single-human-60: original",
            color="#315b7d",
        ),
        ax.bar(
            x + width / 2,
            independent,
            width,
            label="Independent-188 paraphrase*",
            color="#4d9278",
        ),
    ]
    for group in groups:
        for bar in group:
            value = bar.get_height()
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 1.2,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    ax.text(
        x[3] + width / 2,
        3.5,
        "ölçülmedi",
        ha="center",
        fontsize=7,
        color="#555555",
    )
    ax.set_ylabel("Sekiz alan ortalama macro-F1 (%)")
    ax.set_xticks(x, methods)
    ax.set_ylim(0, 90)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.set_title("Semantic extraction: original ve independent wording")
    fig.text(
        0.01,
        0.01,
        "* Independent metinler insan tarafından yeniden etiketlenmedi; "
        "parent intent labels anlam/sayı koruma filtrelerinden sonra aktarıldı.",
        fontsize=7,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUTPUT / "semantic_macro_f1.png", bbox_inches="tight")
    plt.close(fig)


def offloading_figure() -> None:
    methods = [
        "Greedy\nno semantic",
        "PPO\nno semantic",
        "Greedy\nreference",
        "PPO\nreference",
        "PPO rule\nparaphrase",
        "PPO TF-IDF\n(+data)",
        "PPO E5 frozen\n(+data)",
    ]
    success = np.array([78.8, 76.2, 91.7, 91.2, 79.5, 85.7, 86.8])
    hard_violation = np.array([7.45, 0.01, 0.00, 0.00, 14.18, 4.08, 5.03])
    colors = [
        "#8a8f95",
        "#687079",
        "#4d9278",
        "#315b7d",
        "#c55b55",
        "#d98b39",
        "#6d7fb3",
    ]

    x = np.arange(len(methods))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.4, 7.0), sharex=True)
    success_bars = ax1.bar(x, success, color=colors)
    violation_bars = ax2.bar(x, hard_violation, color=colors)
    for bar, value in zip(success_bars, success, strict=True):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.7,
            f"{value:.1f}",
            ha="center",
            fontsize=7,
        )
    for bar, value in zip(violation_bars, hard_violation, strict=True):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.35,
            f"{value:.2f}",
            ha="center",
            fontsize=7,
        )
    ax1.set_ylabel("Görev başarısı (%)")
    ax1.set_ylim(70, 95)
    ax1.grid(axis="y", alpha=0.22)
    ax1.set_title("NEP-v2 exact-preview: aynı 3.840 kararda semantic source etkisi")
    ax2.set_ylabel("Hard constraint ihlali (%)")
    ax2.set_ylim(0, 18)
    ax2.grid(axis="y", alpha=0.22)
    ax2.set_xticks(x, methods, fontsize=8)
    ax2.set_xlabel("Koşul")
    fig.tight_layout()
    fig.savefig(OUTPUT / "offloading_success_and_violations.png", bbox_inches="tight")
    plt.close(fig)


def generalization_figure() -> None:
    """Plot strict-holdout semantic uplift and frozen-source counterfactuals."""

    holdout_labels = ["EUA entity\nholdout", "İki-domain\nholdout"]
    holdout_delta = np.array([15.03, 13.41])
    holdout_low = np.array([13.82, 12.37])
    holdout_high = np.array([16.25, 14.51])
    source_labels = [
        "NEP background\nremoved",
        "UCI compute\nheterogeneity\nremoved",
        "BUPT workload\nvariability\nfixed",
        "EUA topology\nvariability\nfixed",
    ]
    source_delta = np.array([-0.48, -8.92, 0.92, 0.31])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.9))
    x1 = np.arange(len(holdout_labels))
    errors = np.vstack((holdout_delta - holdout_low, holdout_high - holdout_delta))
    bars1 = ax1.bar(x1, holdout_delta, color=["#315b7d", "#4d9278"])
    ax1.errorbar(
        x1,
        holdout_delta,
        yerr=errors,
        fmt="none",
        ecolor="#263746",
        capsize=5,
        linewidth=1.3,
    )
    for bar, value in zip(bars1, holdout_delta, strict=True):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.45,
            f"+{value:.2f}",
            ha="center",
            fontsize=8,
        )
    ax1.set_xticks(x1, holdout_labels)
    ax1.set_ylabel("PPO reference − no-semantic success (point)")
    ax1.set_ylim(0, 18)
    ax1.grid(axis="y", alpha=0.22)
    ax1.set_title("Strict retraining holdout")

    x2 = np.arange(len(source_labels))
    colors = ["#c55b55" if value < 0 else "#4d9278" for value in source_delta]
    bars2 = ax2.bar(x2, source_delta, color=colors)
    for bar, value in zip(bars2, source_delta, strict=True):
        offset = 0.25 if value >= 0 else -0.65
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:+.2f}",
            ha="center",
            fontsize=8,
        )
    ax2.axhline(0, color="#334455", linewidth=0.9)
    ax2.set_xticks(x2, source_labels, fontsize=8)
    ax2.set_ylabel("Base'e göre Δ PPO-reference success (point)")
    ax2.set_ylim(-10.5, 2.1)
    ax2.grid(axis="y", alpha=0.22)
    ax2.set_title("Frozen-policy source counterfactual")

    fig.tight_layout()
    fig.savefig(OUTPUT / "generalization_and_sources.png", bbox_inches="tight")
    plt.close(fig)


def architecture_figure() -> None:
    fig, ax = plt.subplots(figsize=(10.4, 5.3))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 5.3)
    ax.axis("off")

    boxes = [
        (0.25, 3.55, 1.7, 1.0, "Türkçe görev\nmetni", "#d8e7f3"),
        (
            2.35,
            3.35,
            2.05,
            1.4,
            "İzole semantik\nçıkarıcı\n(rule / TF-IDF / E5 / LLM)",
            "#d7eadf",
        ),
        (
            4.85,
            3.35,
            2.05,
            1.4,
            "Dondurulmuş\n8 alanlı sözleşme\n+ confidence",
            "#f5dfbd",
        ),
        (
            0.85,
            0.65,
            2.35,
            1.35,
            "Trace-driven fizik\nBUPT + NEP + UCI + EUA",
            "#e4dff1",
        ),
        (
            4.15,
            0.55,
            2.25,
            1.55,
            "Action mask +\ntek fizik çekirdeği\n+ PPO / greedy",
            "#f3d4d1",
        ),
        (
            7.15,
            0.55,
            2.65,
            1.55,
            "Local / edge / cloud\nGecikme · enerji · başarı\nHard ihlal · fairness",
            "#d9e5e3",
        ),
    ]
    for x, y, width, height, label, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.05,rounding_size=0.08",
            linewidth=1.2,
            edgecolor="#334455",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=9
        )

    arrows = [
        ((1.95, 4.05), (2.35, 4.05)),
        ((4.40, 4.05), (4.85, 4.05)),
        ((5.88, 3.35), (5.42, 2.10)),
        ((3.20, 1.32), (4.15, 1.32)),
        ((6.40, 1.32), (7.15, 1.32)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.5,
                color="#334455",
            )
        )
    ax.text(
        5.2,
        5.0,
        "Temel izolasyon: LLM ağ durumunu, doğru eylemi, ödülü veya "
        "oracle sonucunu görmez",
        ha="center",
        fontsize=10,
        weight="bold",
        color="#7a2e2e",
    )
    ax.text(2.65, 2.55, "Semantik katman", ha="center", fontsize=8, color="#315b7d")
    ax.text(
        5.5, 2.55, "Karar ve fizik katmanı", ha="center", fontsize=8, color="#7a2e2e"
    )
    fig.tight_layout()
    fig.savefig(OUTPUT / "research_architecture.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _style()
    semantic_figure()
    offloading_figure()
    architecture_figure()
    generalization_figure()
    print(f"Wrote advisor figures to {OUTPUT}")


if __name__ == "__main__":
    main()
