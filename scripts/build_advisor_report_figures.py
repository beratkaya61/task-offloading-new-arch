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
        "Rule",
        "TF-IDF\n(+data)",
        "E5 frozen\n(+data)",
        "E5 fine-tuned\n(+data)",
        "Qwen2.5\n1.5B 4-shot",
    ]
    original = np.array([78.0, 78.2, 76.3, 70.2, 48.3])
    independent = np.array([54.6, 80.9, 73.8, np.nan, 42.0])

    x = np.arange(len(methods))
    width = 0.34
    # Keep the physical canvas close to the DOCX text width so Pandoc does not
    # shrink labels to an unreadable size when embedding the figure.
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
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
                    fontsize=8,
                )
    ax.text(
        x[3] + width / 2,
        3.5,
        "not evaluated",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    ax.set_ylabel("Eight-field macro-F1 (%)")
    ax.set_xticks(x, methods)
    ax.set_ylim(0, 90)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.set_title("Semantic extraction: original vs. independent wording")
    fig.text(
        0.01,
        0.01,
        "* Independent texts were not re-annotated by a human; parent-intent "
        "labels were transferred after meaning/number preservation filters.",
        fontsize=8,
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.5), sharex=True)
    success_bars = ax1.bar(x, success, color=colors)
    violation_bars = ax2.bar(x, hard_violation, color=colors)
    for bar, value in zip(success_bars, success, strict=True):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.7,
            f"{value:.1f}",
            ha="center",
            fontsize=8,
        )
    for bar, value in zip(violation_bars, hard_violation, strict=True):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.35,
            f"{value:.2f}",
            ha="center",
            fontsize=8,
        )
    ax1.set_ylabel("Task success rate (%)")
    ax1.set_ylim(70, 95)
    ax1.grid(axis="y", alpha=0.22)
    ax1.set_title(
        "NEP-v2 exact-preview: semantic-source effect on 3,840 paired decisions"
    )
    ax2.set_ylabel("Hard constraint violation rate (%)")
    ax2.set_ylim(0, 18)
    ax2.grid(axis="y", alpha=0.22)
    ax2.set_xticks(x, methods, fontsize=8)
    ax2.set_xlabel("Condition")
    fig.tight_layout()
    fig.savefig(OUTPUT / "offloading_success_and_violations.png", bbox_inches="tight")
    plt.close(fig)


def generalization_figure() -> None:
    """Plot strict-holdout semantic uplift and frozen-source counterfactuals."""

    holdout_labels = ["EUA entity\nholdout", "Two-domain\nholdout"]
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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 7.2))
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
    ax2.set_ylabel("Δ PPO-reference success vs. base (point)")
    ax2.set_ylim(-10.5, 2.1)
    ax2.grid(axis="y", alpha=0.22)
    ax2.set_title("Frozen-policy source counterfactual")

    fig.tight_layout()
    fig.savefig(OUTPUT / "generalization_and_sources.png", bbox_inches="tight")
    plt.close(fig)


def architecture_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 5.3)
    ax.axis("off")

    boxes = [
        (0.25, 3.55, 1.7, 1.0, "Turkish task\ntext", "#d8e7f3", 8.5),
        (
            2.35,
            3.35,
            2.05,
            1.4,
            "Isolated semantic\nextractor\nrule / TF-IDF /\nE5 / LLM",
            "#d7eadf",
            7.5,
        ),
        (
            4.85,
            3.35,
            2.05,
            1.4,
            "Frozen 8-field\nrequirement\ncontract\n+ confidence",
            "#f5dfbd",
            7.5,
        ),
        (
            0.85,
            0.65,
            2.35,
            1.35,
            "Trace-driven physics\nBUPT + NEP\n+ UCI + EUA",
            "#e4dff1",
            7.5,
        ),
        (
            4.15,
            0.55,
            2.25,
            1.55,
            "Action masking\n+ shared physics core\n+ PPO / greedy",
            "#f3d4d1",
            7.5,
        ),
        (
            7.15,
            0.55,
            2.65,
            1.55,
            "Local / edge / cloud\nLatency · energy\n"
            "Success · hard violations\nFairness",
            "#d9e5e3",
            7.2,
        ),
    ]
    for x, y, width, height, label, color, fontsize in boxes:
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
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
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
        "Core isolation: the LLM cannot observe network state, correct action, "
        "reward, or oracle outcome",
        ha="center",
        fontsize=10,
        weight="bold",
        color="#7a2e2e",
    )
    ax.text(2.25, 2.55, "Semantic layer", ha="center", fontsize=8, color="#315b7d")
    ax.text(
        6.6,
        2.55,
        "Decision and physics layer",
        ha="center",
        fontsize=8,
        color="#7a2e2e",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT / "research_architecture.png", bbox_inches="tight")
    plt.close(fig)


def end_to_end_flow_figure() -> None:
    """Render the frozen end-to-end research pipeline as a flow diagram."""

    # Portrait-first layout: the DOCX embeds figures at about 5.83 inches.
    # A wide three-lane canvas would reduce 8 pt labels to roughly 3 pt.
    fig, ax = plt.subplots(figsize=(7.2, 9.6))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 15)
    ax.axis("off")

    lane_specs = [
        (10.55, 4.05, "OFFLINE SEMANTIC PREPARATION", "#eaf2f8", "#315b7d"),
        (5.65, 4.35, "TRACE-DRIVEN DECISION AND PHYSICS", "#f8f0ea", "#8a4f39"),
        (0.45, 4.65, "EVALUATION AND PUBLICATION", "#edf5ef", "#3f745f"),
    ]
    for y, height, label, face, edge in lane_specs:
        lane = FancyBboxPatch(
            (0.12, y),
            8.76,
            height,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.0,
            edgecolor=edge,
            facecolor=face,
            alpha=0.72,
        )
        ax.add_patch(lane)
        ax.text(
            0.3,
            y + height - 0.3,
            label,
            fontsize=10,
            weight="bold",
            color=edge,
        )

    def box(
        x: float,
        y: float,
        width: float,
        height: float,
        label: str,
        color: str,
        *,
        fontsize: float = 9.2,
    ) -> None:
        node = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.05,rounding_size=0.07",
            linewidth=1.2,
            edgecolor="#334455",
            facecolor=color,
        )
        ax.add_patch(node)
        ax.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
        )

    def arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        label: str | None = None,
        dashed: bool = False,
        color: str = "#334455",
    ) -> None:
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.35,
            linestyle="--" if dashed else "-",
            color=color,
            connectionstyle="arc3,rad=0.0",
        )
        ax.add_patch(patch)
        if label:
            ax.text(
                (start[0] + end[0]) / 2,
                (start[1] + end[1]) / 2 + 0.1,
                label,
                ha="center",
                va="bottom",
                fontsize=8.0,
                color=color,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
            )

    semantic_color = "#d8e7f3"
    contract_color = "#d8eadf"
    physical_color = "#f1dfcf"
    policy_color = "#f3d4d1"
    evaluation_color = "#dcebdc"

    # Lane 1: language is converted into a frozen policy input.  Reference
    # requirements are a separate evaluation condition, never an online oracle.
    box(0.4, 12.65, 2.25, 1.05, "Task text +\nuser policy text", semantic_color)
    box(
        3.35,
        12.65,
        2.3,
        1.05,
        "Semantic extractors\nrule · TF-IDF · E5 · LLM\n(+ reference condition)",
        semantic_color,
        fontsize=8.7,
    )
    box(
        6.35,
        12.65,
        2.25,
        1.05,
        "8-field requirement contract\nvalue · status · evidence\nraw confidence",
        contract_color,
        fontsize=8.5,
    )
    box(
        3.35,
        10.95,
        2.3,
        1.05,
        "Development-only\nconfidence calibration\n+ fallback rule",
        contract_color,
        fontsize=8.7,
    )
    box(
        6.35,
        10.95,
        2.25,
        1.05,
        "Versioned frozen cache\nmodel · prompt · parser\nraw / calibrated",
        contract_color,
        fontsize=8.7,
    )

    arrow((2.65, 13.18), (3.35, 13.18))
    arrow((5.65, 13.18), (6.35, 13.18))
    arrow((7.48, 12.65), (4.5, 12.0))
    arrow((5.65, 11.48), (6.35, 11.48))
    ax.text(
        1.5,
        11.48,
        "Isolation gate\nNo network state\nNo reward or oracle\nNo correct action",
        ha="center",
        va="center",
        fontsize=9.0,
        weight="bold",
        color="#7a2e2e",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#fff4f2",
            "edgecolor": "#a85b50",
        },
    )

    # Lane 2: every condition receives the same exogenous event; only the
    # believed requirement source and policy condition vary.
    box(
        0.4,
        8.25,
        2.25,
        1.05,
        "Physical sources\nBUPT · NEP · UCI · EUA\nprovenance + splits",
        physical_color,
        fontsize=8.7,
    )
    box(
        3.35,
        8.25,
        2.3,
        1.05,
        "Seeded exogenous events\narrival · workload · RTT\ncompute · topology",
        physical_color,
        fontsize=8.7,
    )
    box(
        6.35,
        8.25,
        2.25,
        1.05,
        "Observation builder\nnetwork + queue state\n+ believed requirements",
        physical_color,
        fontsize=8.7,
    )
    box(
        0.4,
        6.3,
        2.25,
        1.05,
        "Hard-constraint verifier\n+ action masking\n(or penalty ablation)",
        policy_color,
        fontsize=8.7,
    )
    box(
        3.35,
        6.3,
        2.3,
        1.05,
        "Policy family\nfixed · random · greedy\nPPO (5 seeds)",
        policy_color,
        fontsize=8.7,
    )
    box(
        6.35,
        6.3,
        2.25,
        1.05,
        "Shared deterministic\ntransition core\nlocal · edge · cloud",
        policy_color,
        fontsize=8.7,
    )

    arrow((2.65, 8.78), (3.35, 8.78))
    arrow((5.65, 8.78), (6.35, 8.78))
    arrow((7.48, 8.25), (1.52, 7.35))
    arrow((2.65, 6.83), (3.35, 6.83))
    arrow((5.65, 6.83), (6.35, 6.83))
    arrow(
        (7.48, 10.95),
        (7.48, 9.3),
        label="selected semantic belief",
        dashed=True,
        color="#315b7d",
    )
    arrow(
        (4.5, 8.25),
        (7.48, 7.35),
        label="same seeded physical event",
        dashed=True,
        color="#8a4f39",
    )

    # Lane 3: belief drives action selection, while truth is used only for
    # scoring.  This prevents semantic error from being hidden by evaluation.
    box(
        0.4,
        3.55,
        2.25,
        1.05,
        "Reference requirements\nEvaluation truth only\n(single-human upper bound)",
        evaluation_color,
        fontsize=8.5,
    )
    box(
        3.35,
        3.55,
        2.3,
        1.05,
        "Outcome scorer\ncompletion + QoS\nconstraint violations",
        evaluation_color,
        fontsize=8.7,
    )
    box(
        6.35,
        3.55,
        2.25,
        1.05,
        "System metrics\nsuccess · latency · queue\nenergy · fairness · collapse",
        evaluation_color,
        fontsize=8.5,
    )
    box(
        1.65,
        1.5,
        2.55,
        1.05,
        "Statistical protocol\npaired bootstrap · 95% CI\nHolm correction",
        evaluation_color,
        fontsize=8.7,
    )
    box(
        4.8,
        1.5,
        3.15,
        1.05,
        "Publication evidence\nphase reports · figures\n"
        "hypotheses · advisor MD/DOCX\nSHA-256 manifest",
        evaluation_color,
        fontsize=8.0,
    )

    arrow((2.65, 4.08), (3.35, 4.08), label="truth")
    arrow((5.65, 4.08), (6.35, 4.08))
    arrow((7.48, 3.55), (2.93, 2.55))
    arrow((4.2, 2.03), (4.8, 2.03))
    arrow(
        (7.48, 6.3),
        (4.5, 4.6),
        label="physical outcome",
        dashed=True,
        color="#8a4f39",
    )

    ax.set_title(
        "End-to-end system and experimental flow",
        fontsize=14,
        weight="bold",
        pad=10,
    )
    ax.text(
        4.5,
        0.72,
        "Believed requirements drive the policy; reference requirements "
        "score the outcome.\n"
        "This separation exposes semantic-extraction errors instead of hiding them.",
        ha="center",
        fontsize=9.0,
        color="#334455",
        wrap=True,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT / "end_to_end_system_flow.png", bbox_inches="tight")
    plt.close(fig)


def _validate_outputs() -> None:
    """Fail fast on missing, unreadable, blank, or undersized report figures."""

    names = (
        "research_architecture.png",
        "end_to_end_system_flow.png",
        "semantic_macro_f1.png",
        "offloading_success_and_violations.png",
        "generalization_and_sources.png",
    )
    for name in names:
        path = OUTPUT / name
        image = plt.imread(path)
        if image.ndim not in {2, 3} or min(image.shape[:2]) < 900:
            raise RuntimeError(f"advisor figure has invalid dimensions: {path}")
        if not np.isfinite(image).all() or float(np.std(image)) < 0.01:
            raise RuntimeError(f"advisor figure is blank or corrupt: {path}")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _style()
    semantic_figure()
    offloading_figure()
    architecture_figure()
    end_to_end_flow_figure()
    generalization_figure()
    _validate_outputs()
    print(f"Wrote advisor figures to {OUTPUT}")


if __name__ == "__main__":
    main()
