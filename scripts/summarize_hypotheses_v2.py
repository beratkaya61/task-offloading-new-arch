"""Generate the pre-registered H1-H6 decision matrix from frozen results."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "offloading_experiment"
EXACT = ARTIFACTS / "results_v2_nep.json"
NOISY = ARTIFACTS / "results_v2_nep_noise.json"
OUTPUT = ROOT / "phase_reports" / "HYPOTHESIS_DECISIONS_V2.md"
RESAMPLES = 2000


class Contrast(TypedDict):
    value: float
    ci95: tuple[float, float]
    p_value_two_sided: float


def _rows(data: Mapping[str, Any]) -> dict[str, Any]:
    return {**data["results"]["baselines"], **data["results"]["ppo"]}


def _episodes(entry: Any, metric: str) -> np.ndarray:
    runs = entry if isinstance(entry, list) else [entry]
    return np.asarray(
        np.mean(
            [np.asarray(run["per_episode"][metric], dtype=float) for run in runs],
            axis=0,
        ),
        dtype=float,
    )


def _contrast(
    rows: Mapping[str, Any], left: str, right: str, metric: str
) -> Contrast:
    delta = _episodes(rows[left], metric) - _episodes(rows[right], metric)
    rng = np.random.default_rng(20261120)
    draws = rng.integers(0, len(delta), size=(RESAMPLES, len(delta)))
    samples = delta[draws].mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    p_value = min(
        1.0,
        2 * min(float(np.mean(samples <= 0)), float(np.mean(samples >= 0))),
    )
    return {
        "value": float(delta.mean()),
        "ci95": (float(low), float(high)),
        "p_value_two_sided": float(p_value),
    }


def _fmt(result: Contrast) -> str:
    low, high = result["ci95"]
    return f"{100 * result['value']:+.2f} [{100 * low:+.2f}, {100 * high:+.2f}]"


def main() -> None:
    exact_data = json.loads(EXACT.read_text(encoding="utf-8"))
    noisy_data = json.loads(NOISY.read_text(encoding="utf-8"))
    exact = _rows(exact_data)
    noisy = _rows(noisy_data)

    h1_ppo = _contrast(
        exact, "ppo_semantic[reference]", "ppo_no_semantic", "success_rate"
    )
    h1_greedy = _contrast(
        exact,
        "greedy_semantic[reference]",
        "greedy_no_semantic",
        "success_rate",
    )
    h2_ppo = _contrast(
        exact,
        "ppo_semantic[rule_paraphrase]",
        "ppo_semantic[tfidf_aug_paraphrase]",
        "hard_violation_rate",
    )
    h2_greedy = _contrast(
        exact,
        "greedy_semantic[rule_paraphrase]",
        "greedy_semantic[tfidf_aug_paraphrase]",
        "hard_violation_rate",
    )
    h3: dict[str, dict[str, Contrast]] = {}
    for policy in ("ppo_semantic", "greedy_semantic"):
        for source in ("tfidf_aug_paraphrase", "e5_frozen_aug_paraphrase"):
            base = f"{policy}[{source}]"
            fallback = f"{policy}[{source}+fallback]"
            h3[f"{policy}/{source}"] = {
                "success": _contrast(exact, fallback, base, "success_rate"),
                "hard": _contrast(exact, fallback, base, "hard_violation_rate"),
            }
    h4_hard = _contrast(
        exact,
        "ppo_semantic[reference]",
        "ppo_semantic_penalty[reference]",
        "hard_violation_rate",
    )
    h4_success = _contrast(
        exact,
        "ppo_semantic[reference]",
        "ppo_semantic_penalty[reference]",
        "success_rate",
    )
    h5 = _contrast(
        noisy,
        "ppo_semantic[reference]",
        "greedy_semantic[reference]",
        "success_rate",
    )

    h1_supported = h1_ppo["ci95"][0] > 0 and h1_greedy["ci95"][0] > 0
    h2_supported = h2_ppo["ci95"][0] > 0 and h2_greedy["ci95"][0] > 0
    h3_supported = all(
        item["success"]["ci95"][0] > -0.02 and item["hard"]["ci95"][1] < 0
        for item in h3.values()
    )
    h4_supported = h4_hard["ci95"][1] < 0 and h4_success["ci95"][0] > -0.01
    h5_supported = h5["ci95"][0] > -0.01
    h6_supported = h1_supported and h2_supported

    lines = [
        "# Pre-registered System Hypotheses - v2 Decision Matrix",
        "",
        "Kaynaklar: `results_v2_nep.json` (exact-preview) ve "
        "`results_v2_nep_noise.json` (sigma=0.5 noisy-preview). Farklar paired "
        "episode bootstrap, 2,000 resample ve %95 percentile CI ile hesaplanmistir. "
        "Yuzde-oran farklari percentage point cinsindedir.",
        "",
        "| Hipotez | On-kayitli test | Sonuc | Karar |",
        "|---|---|---|---|",
        f"| H1 | Reference semantic success > no-semantic, PPO ve greedy | "
        f"PPO {_fmt(h1_ppo)}; greedy {_fmt(h1_greedy)} | "
        f"{'Desteklendi' if h1_supported else 'Desteklenmedi'} |",
        f"| H2 | Rule hard violation > TF-IDF(+data), PPO ve greedy | "
        f"PPO {_fmt(h2_ppo)}; greedy {_fmt(h2_greedy)} | "
        f"{'Desteklendi' if h2_supported else 'Desteklenmedi'} |",
        "| H3 | Calibrated fallback hard violation'i azaltir ve success CI alt "
        f"siniri > -2 pp | Asagidaki model/policy tablosu | "
        f"{'Desteklendi' if h3_supported else 'Desteklenmedi'} |",
        f"| H4 | Mask-PPO hard violation < penalty-PPO ve success alt sinir > -1 pp | "
        f"hard {_fmt(h4_hard)}; success {_fmt(h4_success)} | "
        f"{'Desteklendi' if h4_supported else 'Desteklenmedi'} |",
        f"| H5 | Noisy-preview reference PPO - greedy success CI alt siniri > -1 pp | "
        f"{_fmt(h5)} | {'Desteklendi' if h5_supported else 'Desteklenmedi'} |",
        f"| H6 | NEP katmaninda H1 ve H2 yonu korunur | H1/H2 NEP'de ayni yonde | "
        f"{'Desteklendi' if h6_supported else 'Desteklenmedi'} |",
        "",
        "## H3 fallback ayrintisi",
        "",
        "Fallback-baseline farkidir; hard violation icin negatif, success icin "
        "CI alt sinirinin -2 pp'den buyuk olmasi beklenir.",
        "",
        "| Policy/source | Success farki [CI] | Hard violation farki [CI] | "
        "Iki kosul birlikte gecti mi? |",
        "|---|---:|---:|---|",
    ]
    for name, item in h3.items():
        passed = (
            item["success"]["ci95"][0] > -0.02
            and item["hard"]["ci95"][1] < 0
        )
        lines.append(
            f"| {name} | {_fmt(item['success'])} | {_fmt(item['hard'])} | "
            f"{'Evet' if passed else 'Hayir'} |"
        )
    lines += [
        "",
        "## Yorum sinirlari",
        "",
        "- H4'te iki reference PPO kosulu da sifir hard violation verdi; esitlik, "
        "on-kayitli 'mask daha dusuk' yonunu desteklemez.",
        "- H5 point estimate margin civarinda olsa da CI alt siniri -1 pp'nin "
        "altindadir; non-inferiority ilan edilmez.",
        "- H3 fine-tuned E5 parcasi calibrated independent prediction/checkpoint "
        "olmadigi icin tamamlanamadi. Mevcut TF-IDF/E5-frozen kosullarinin da "
        "tamami kurali gecmediginden H3 genel karari degismez.",
        "- Holm-adjusted p-values full publication tablolarinda raw p-values ile "
        "birlikte raporlanmistir.",
        "",
    ]
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
