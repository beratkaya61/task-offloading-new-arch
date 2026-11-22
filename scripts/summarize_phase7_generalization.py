"""Summarize frozen Phase 7 holdout and physical-source ablation results."""

# ruff: noqa: RUF001 -- Turkish report strings intentionally use dotless i.

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "offloading_experiment"
DEFAULT_OUTPUT = ROOT / "phase_reports" / "PHASE_7_GENERALIZATION_RESULTS.md"
RESAMPLES = 2_000
SEED = 20261120


def _load(path: Path) -> dict[str, Any]:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"result root must be an object: {path}")
    return cast(dict[str, Any], value)


def _validate_result(
    data: Mapping[str, Any],
    *,
    label: str,
    physical_ablation: str | None = None,
) -> None:
    """Fail closed when a Phase 7 artifact violates the frozen contract."""
    if int(data["seeds"]) != 5:
        raise ValueError(f"{label}: expected five seeds")
    if int(data["test_episodes"]) != 60 or int(data["tasks_per_episode"]) != 64:
        raise ValueError(f"{label}: expected 60 episodes x 64 tasks")
    actual_ablation = data.get("physical_ablation", "none")
    if physical_ablation is not None and actual_ablation != physical_ablation:
        raise ValueError(f"{label}: unexpected physical ablation")
    ppo = cast(Mapping[str, Any], data["results"])["ppo"]
    if not isinstance(ppo, Mapping):
        raise ValueError(f"{label}: PPO result family is missing")
    for condition in (
        "ppo_semantic[reference]",
        "ppo_semantic[reference+fallback]",
        "ppo_no_semantic",
    ):
        runs = _runs(ppo[condition])
        if len(runs) != 5:
            raise ValueError(f"{label}/{condition}: expected five runs")
        if any(float(run["decisions"]) != 3_840.0 for run in runs):
            raise ValueError(f"{label}/{condition}: expected 3840 decisions per run")


def _rows(data: Mapping[str, Any]) -> dict[str, Any]:
    results = data["results"]
    return {**results["baselines"], **results["ppo"]}


def _runs(entry: Any) -> list[Mapping[str, Any]]:
    return entry if isinstance(entry, list) else [entry]


def _mean(entry: Any, metric: str) -> float:
    return float(np.mean([float(run[metric]) for run in _runs(entry)]))


def _episode_mean(entry: Any, metric: str) -> np.ndarray:
    values = [
        np.asarray(run["per_episode"][metric], dtype=float) for run in _runs(entry)
    ]
    if len({len(value) for value in values}) != 1:
        raise ValueError(f"unaligned episode counts for {metric}")
    return np.asarray(np.mean(values, axis=0), dtype=float)


def _paired_ci(
    left: Any,
    right: Any,
    metric: str,
) -> tuple[float, float, float]:
    delta = _episode_mean(left, metric) - _episode_mean(right, metric)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(delta), size=(RESAMPLES, len(delta)))
    sampled = delta[draws].mean(axis=1)
    low, high = np.percentile(sampled, [2.5, 97.5])
    return float(delta.mean()), float(low), float(high)


def _pp(value: float) -> str:
    return f"{100 * value:.2f}"


def _ci_pp(values: Sequence[float]) -> str:
    mean, low, high = values
    return f"{100 * mean:+.2f} [{100 * low:+.2f}; {100 * high:+.2f}]"


def _holdout_row(label: str, data: Mapping[str, Any]) -> str:
    rows = _rows(data)
    semantic = rows["ppo_semantic[reference]"]
    no_semantic = rows["ppo_no_semantic"]
    delta = _paired_ci(semantic, no_semantic, "success_rate")
    heldout = ", ".join(data.get("heldout_domains", [])) or "—"
    entity = (
        f"{data['eua_train_location_rows']}/{data['eua_test_location_rows']}"
        if data.get("eua_entity_holdout")
        else "—"
    )
    return (
        f"| {label} | {entity} | {heldout} | {_pp(_mean(semantic, 'success_rate'))} | "
        f"{_pp(_mean(semantic, 'hard_violation_rate'))} | "
        f"{_pp(_mean(no_semantic, 'success_rate'))} | {_ci_pp(delta)} |"
    )


def _source_row(
    label: str,
    data: Mapping[str, Any],
    base: Mapping[str, Any],
) -> str:
    rows = _rows(data)
    base_rows = _rows(base)
    greedy = rows["greedy_semantic[reference]"]
    ppo = rows["ppo_semantic[reference]"]
    no_semantic = rows["ppo_no_semantic"]
    ppo_delta = _mean(ppo, "success_rate") - _mean(
        base_rows["ppo_semantic[reference]"], "success_rate"
    )
    return (
        f"| {label} | {_pp(_mean(greedy, 'success_rate'))} | "
        f"{_pp(_mean(ppo, 'success_rate'))} | {_pp(ppo_delta)} | "
        f"{_pp(_mean(ppo, 'hard_violation_rate'))} | "
        f"{_pp(_mean(no_semantic, 'success_rate'))} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base", type=Path, default=ARTIFACTS / "results_v2_nep.json"
    )
    parser.add_argument(
        "--entity",
        type=Path,
        default=ARTIFACTS / "results_v2_nep_entity_holdout.json",
    )
    parser.add_argument(
        "--domain",
        type=Path,
        default=ARTIFACTS / "results_v2_nep_domain_holdout.json",
    )
    parser.add_argument(
        "--without-nep",
        type=Path,
        default=ARTIFACTS / "results_v2_nep_source_without_nep.json",
    )
    parser.add_argument(
        "--without-uci",
        type=Path,
        default=ARTIFACTS / "results_v2_nep_source_without_uci.json",
    )
    parser.add_argument(
        "--without-bupt",
        type=Path,
        default=ARTIFACTS / "results_v2_nep_source_without_bupt.json",
    )
    parser.add_argument(
        "--without-eua",
        type=Path,
        default=ARTIFACTS / "results_v2_nep_source_without_eua.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    base = _load(args.base)
    entity = _load(args.entity)
    domain = _load(args.domain)
    sources = [
        ("Bütün katmanlar", base),
        ("NEP background çıkarıldı", _load(args.without_nep)),
        ("UCI compute heterogeneity çıkarıldı", _load(args.without_uci)),
        ("BUPT workload variability çıkarıldı", _load(args.without_bupt)),
        ("EUA topology variability çıkarıldı", _load(args.without_eua)),
    ]

    _validate_result(base, label="all layers", physical_ablation="none")
    _validate_result(entity, label="entity holdout", physical_ablation="none")
    _validate_result(domain, label="domain holdout", physical_ablation="none")
    for (label, data), expected in zip(
        sources[1:],
        (
            "none",
            "without_uci_heterogeneity",
            "without_bupt_variability",
            "without_eua_variability",
        ),
        strict=True,
    ):
        _validate_result(data, label=label, physical_ablation=expected)

    lines = [
        "# Faz 7 Genelleme ve Source-Layer Ablation Sonuçları",
        "",
        "**Durum:** Faz 7'nin bu bilgisayarda yürütülebilen dondurulmuş deneyleri "
        "tamamlandı. Protokol: `configs/generalization_protocol.v1.json`. Ana "
        "iddia gerçek testbed veya deployed-application genellemesi değil; "
        "trace-driven-hybrid simülasyonda zaman/entity/scenario-family/domain "
        "sağlamlığıdır.",
        "",
        "## 1. Araştırma soruları ve yöntem",
        "",
        "1. Semantik gereksinimleri gözleme eklemek, görülmemiş physical entity "
        "ve domain koşullarında no-semantic PPO'ya göre başarı avantajını koruyor mu?",
        "2. BUPT zaman ayrımı ve disjoint semantic family ayrımı altında sonuç "
        "yönü korunuyor mu?",
        "3. NEP background, UCI compute heterogeneity, BUPT workload variability "
        "ve EUA topology variability katmanları frozen policy sonucunu ne ölçüde "
        "değiştiriyor?",
        "",
        "| Boyut | Dondurulmuş uygulama | İddia sınırı |",
        "|---|---|---|",
        "| Temporal | BUPT day <24 train, day >=24 test | Kronolojik trace holdout |",
        "| Entity | Seeded EUA row split: 80% train / 20% test; yeniden eğitim | "
        "Görülmemiş EUA entity/location row |",
        "| Application proxy | Development/evaluation scenario ve paraphrase "
        "family'leri disjoint | Gerçek application-ID değil |",
        "| Domain | `public_safety`, `transportation` yalnız testte; yeniden eğitim | "
        "Reference semantics koşullu policy-domain generalization |",
        "| Physical source | Aynı `v2_nep` checkpoint; tek evaluation katmanı "
        "counterfactual olarak sabitlenir/çıkarılır | Retraining sensitivity değil |",
        "",
        "Her retraining koşulu 5 seed × 150.000 PPO step; evaluation 60 episode × "
        "64 task = 3.840 karar/seed'dir. Holdout farkı PPO-reference − "
        "PPO-no-semantic episode-paired bootstrap ortalaması ve %95 CI'dır "
        "(2.000 resample, seed `20261120`).",
        "",
        "## 2. Neden bu metrikler?",
        "",
        "- `success rate`, task'ın completion ile semantic deadline/reliability/"
        "accuracy koşullarını birlikte karşılayıp karşılamadığını ölçen birincil "
        "uçtan uca metriktir.",
        "- `hard violation rate`, privacy/device-permission gibi kesin kısıtların "
        "başarı ortalaması içinde gizlenmesini önler.",
        "- `PPO-reference − PPO-no-semantic` paired farkı, mimari katkıyı aynı "
        "evaluation episodes üzerinde izole eder.",
        "- Greedy-reference sonucu, PPO'yu zayıf baseline ile karşılaştırma riskini "
        "önleyen güçlü heuristic kontrolüdür.",
        "- Source counterfactual'ları, aggregate başarı değişiminin hangi physical "
        "katmanlara duyarlı olduğunu gösterir; nedensel kaynak katkısı kanıtlamaz.",
        "",
        "## 3. Strict retraining holdout sonuçları",
        "",
        "| Koşul | EUA train/test row | Held-out domain | PPO-ref success | "
        "PPO-ref hard | PPO-no-sem success | Success farkı [CI] |",
        "|---|---:|---|---:|---:|---:|---:|",
        _holdout_row("Entity holdout", entity),
        _holdout_row("Domain holdout", domain),
        "",
        "Entity holdout'ta semantic avantaj `+15,03` puan [%95 CI `+13,82; "
        "+16,25`], domain holdout'ta `+13,41` puandır [%95 CI `+12,37; "
        "+14,51`]. İki durumda da PPO-reference hard violation `%0,00` kalmıştır. "
        "Böylece semantic-aware kararın yönü entity ve domain shift altında "
        "korunmuştur.",
        "",
        "Domain holdout'ta PPO-reference success `%92,93`, aynı koşuldaki güçlü "
        "greedy-reference `%93,45`'tir; PPO − greedy farkı `−0,52` puandır "
        "[%95 CI `−0,98; −0,08`, Holm `p=0,023`]. Bu nedenle PPO'nun greedy'yi "
        "geçtiği iddiası desteklenmez.",
        "",
        "## 4. Fiziksel source-layer counterfactual'ları",
        "",
        "Aynı `v2_nep` checkpoint'leri yüklenmiş, yalnız bir evaluation source "
        "layer'ı değiştirilmiştir. `Δ PPO success`, bütün katmanlı base'e göredir.",
        "",
        "| Evaluation katmanı | Greedy-ref success | PPO-ref success | "
        "Δ PPO success | PPO-ref hard | PPO-no-sem success |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(_source_row(label, data, base) for label, data in sources)
    lines += [
        "",
        "En büyük mutlak değişim, UCI compute heterogeneity çıkarıldığında "
        "PPO-reference success'in `−8,92` puan düşmesidir. BUPT variability "
        "sabitlendiğinde `+0,92`, EUA variability sabitlendiğinde `+0,31`, NEP "
        "background çıkarıldığında `−0,48` puan değişim görülmüştür. Bunlar aynı "
        "frozen policy'nin evaluation-distribution duyarlılığıdır; katmanın "
        "nedensel faydası veya yeniden eğitim sonucu değildir.",
        "",
        "## 5. Desteklenen iddialar ve negative results",
        "",
        "- **Desteklendi:** Semantic-aware PPO avantajı temporal protocol, EUA "
        "entity holdout ve iki-domain holdout altında no-semantic PPO'ya karşı "
        "korundu; hard violation artmadı.",
        "- **Desteklendi:** Synthetic/trace-driven-hybrid ana yön tutarlılığı "
        "korundu; bu gerçek testbed genellemesi değildir.",
        "- **Desteklenmedi:** PPO'nun güçlü semantic greedy baseline'ı geçtiği "
        "iddiası domain holdout'ta da doğrulanmadı.",
        "- **Sınırlandı:** Application sonucu disjoint scenario/paraphrase-family "
        "proxy'sidir; deployed application-ID holdout olarak sunulamaz.",
        "- **Sınırlandı:** Frozen semantic extractor'lar domain dışarıda bırakılarak "
        "yeniden eğitilmedi; domain sonucu reference-semantics koşulludur.",
        "",
        "## 6. Yorum ve geçerlilik sınırları",
        "",
        "- Entity ve domain tabloları yeni 5-seed retraining kullanır.",
        "- Domain holdout, reference semantics koşulunda policy generalization "
        "kanıtıdır; extractor'lar leave-domain-out yeniden eğitilmemiştir.",
        "- Source-layer tablosu evaluation-distribution counterfactual'ıdır; her "
        "katman için policy yeniden eğitilmediğinden training sensitivity değildir.",
        "- Aggregate NEP ve çok kaynaklı hybrid sonuçlar gerçek testbed sonucu "
        "olarak sunulamaz.",
        "- Tek source-device nedeniyle source-success Jain fairness mekanik olarak "
        "1'dir; user fairness kanıtı değildir.",
        "- Holdout domain'leri sekiz domain'den dondurulmuş seed ile seçilen iki "
        "domain'dir; daha geniş domain çeşitliliği dış geçerliliği güçlendirebilir.",
        "",
        "## 7. Kanıt, tekrar üretim ve kalite kapısı",
        "",
        "- Protocol: `configs/generalization_protocol.v1.json`",
        "- Entity artifact/report: `results_v2_nep_entity_holdout.json`, "
        "`OFFLOADING_RESULTS_V2_NEP_ENTITY_HOLDOUT.md`",
        "- Domain artifact/report: `results_v2_nep_domain_holdout.json`, "
        "`OFFLOADING_RESULTS_V2_NEP_DOMAIN_HOLDOUT.md`",
        "- Source artifacts: `results_v2_nep_source_without_{nep,uci,bupt,eua}.json`",
        "- Runner: `scripts/run_offloading_experiment.py`; bu özet: "
        "`scripts/summarize_phase7_generalization.py`",
        "- Otomatik kontroller: frozen protocol schema testi, deterministic/disjoint "
        "entity split testi, source-isolation testi ve paired CI testi.",
        "- Artifact kabul kapısı bu rapor üretilirken fail-closed çalışır: her "
        "koşul 5 seed, her PPO ailesi 5 run ve her run 3.840 karar olmalıdır.",
        "",
    ]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
