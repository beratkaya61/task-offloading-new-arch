# ruff: noqa: RUF001, RUF002
"""Label-preserving requirement paraphrases (CheckList-style invariance tests).

The controlled corpus states requirements with a small set of template
sentences.  A real user would phrase the same requirement differently, and an
extractor that only matches template wording would fail.  This module rewrites
each recognised requirement sentence into a meaning-preserving alternative.

Two *disjoint* paraphrase pools exist per pattern:

* ``train`` — used only to augment the 180 machine-labeled development tasks;
* ``test``  — used only to build the perturbed evaluation set.

Because each rewrite keeps the requirement and every number unchanged, the
original reference labels remain valid for the perturbed text ("invariance"
test, Ribeiro et al., ACL 2020).  The perturbed set is machine-generated and
must be reported as such; it is not a new human annotation.

Provenance (2026-11-20): both pools were written together, once, before any
model was evaluated on paraphrases.  A later extension of the *train* pool,
written after the test pool and first results had been seen, was withdrawn
because it constituted experimenter leakage; it must not be re-added.
Known limitation: some cue words (e.g. "hassas", "kısıtlı", "açık") occur in
both pools, so paraphrase results are an optimistic estimate of robustness.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

Pool = Literal["train", "test"]


@dataclass(frozen=True, slots=True)
class ParaphraseRule:
    rule_id: str
    pattern: re.Pattern[str]
    train: tuple[str, ...]
    test: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.train or not self.test:
            raise ValueError(f"{self.rule_id}: both pools must be non-empty")
        if set(self.train) & set(self.test):
            raise ValueError(f"{self.rule_id}: train/test pools overlap")


def _rule(
    rule_id: str, pattern: str, train: list[str], test: list[str]
) -> ParaphraseRule:
    return ParaphraseRule(rule_id, re.compile(pattern), tuple(train), tuple(test))


NUM = r"(?P<n>\d+(?:[.,]\d+)?)"

RULES: tuple[ParaphraseRule, ...] = (
    _rule(
        "accuracy_f1_min",
        rf"Sonucun F1 skoru en az {NUM} olmalıdır\.",
        [
            "Sonucun F1 skoru {n} değerinin altında olmamalıdır.",
            "Elde edilen F1 skoru en az {n} olmalıdır.",
        ],
        [
            "F1 ölçütüyle başarım {n} ya da üzeri olmalıdır.",
            "Çıktının F1 değeri asgari {n} olarak tutturulmalıdır.",
        ],
    ),
    _rule(
        "latency_ms_deadline",
        rf"Sonuç (?:en geç )?{NUM} ms içinde (?:hazır olmalıdır|hazırlanmalıdır)\.",
        [
            "Yanıt en fazla {n} ms içinde dönmelidir.",
            "Sonuç {n} ms geçmeden üretilmelidir.",
        ],
        [
            "İşlem {n} ms'lik süre sınırını aşmamalıdır.",
            "Çıktının hazırlanma süresi {n} ms'yi geçmemelidir.",
        ],
    ),
    _rule(
        "privacy_sensitive",
        r"İşlenen kayıt hassas veridir\.",
        [
            "Kayıt hassas veri niteliğindedir.",
            "Bu görevde işlenen bilgiler hassas veri sınıfındadır.",
        ],
        [
            "Söz konusu veriler hassas kategorisinde değerlendirilir.",
            "İşlenen içerik hassas bilgi barındırır.",
        ],
    ),
    _rule(
        "policy_device_edge_no_cloud",
        r"Cihaz veya yakın edge kullanılabilir; genel bulut kullanılamaz\.",
        [
            "İşlem cihazda ya da yakındaki edge sunucusunda yapılabilir; "
            "genel buluta çıkılamaz.",
            "Cihaz ve yakın edge serbesttir, genel bulut kullanılamaz.",
        ],
        [
            "Yürütme yalnızca cihaz üzerinde veya yakın edge düğümünde olabilir; "
            "kamu bulutu kullanılmamalıdır.",
            "Görev uç cihazda ya da yakın edge'de koşabilir ancak genel bulut "
            "yasaktır.",
        ],
    ),
    _rule(
        "privacy_restricted",
        r"İşlenen ham kayıt kısıtlı veridir\.",
        [
            "Ham kayıt kısıtlı veri sınıfındadır.",
            "İşlenen ham bilgiler kısıtlı veri olarak sınıflanır.",
        ],
        [
            "Ham veriler erişimi kısıtlı kategoridedir.",
            "İşlenen ham içerik kısıtlı statüsündedir.",
        ],
    ),
    _rule(
        "policy_edge_no_cloud",
        r"Yakın edge sunucusunda işlem yapılabilir ve genel bulut kullanılamaz\.",
        [
            "İşlem yakın edge sunucusunda yapılabilir; genel buluta gönderilemez.",
            "Yakın edge sunucusu kullanılabilir, genel bulut kullanılamaz.",
        ],
        [
            "Yakındaki edge düğümünde çalıştırma serbesttir, genel bulut ise "
            "devre dışıdır.",
            "Edge sunucusu kullanılabilir fakat genel bulut kullanımı yasaktır.",
        ],
    ),
    _rule(
        "reliability_percent",
        rf"İsteklerin en az %{NUM}'(?:i|ı|u|ü) başarıyla tamamlanmalıdır\.",
        [
            "Başarı oranı en az %{n} olmalıdır.",
            "Taleplerin en az %{n} oranında başarıyla tamamlanması gerekir.",
        ],
        [
            "Başarı oranı %{n} seviyesinin altına düşmemelidir.",
            "İşlemlerin %{n} veya daha fazlası hatasız bitmelidir.",
        ],
    ),
    _rule(
        "energy_high",
        r"Pil tüketimini azaltmak birinci önceliktir\.",
        [
            "Pil tüketiminin azaltılması en önemli önceliktir.",
            "Pil ömrünü korumak birinci önceliktir.",
        ],
        [
            "Batarya kullanımını en aza indirmek her şeyden önemlidir.",
            "Enerji tasarrufu en üst önceliktir.",
        ],
    ),
    _rule(
        "divisible_parallel",
        r"(?:Veri parçaları bağımsız ve paralel işlenebilir|"
        r"Görev parçaları birbirinden bağımsız işlenebilir)\.",
        [
            "Veri bölümleri bağımsız ve paralel işlenebilir.",
            "Görev parçalara ayrılıp bağımsız işlenebilir.",
        ],
        [
            "İş, birbirinden bağımsız alt görevlere bölünerek eşzamanlı yürütülebilir.",
            "Girdi parçalara ayrılabilir ve her parça ayrı ayrı işlenebilir.",
        ],
    ),
    _rule(
        "latency_instant",
        r"Yanıt anlık verilmelidir\.",
        [
            "Yanıt hemen verilmelidir.",
            "Sonuç anında üretilmelidir.",
        ],
        [
            "Cevap gecikmesiz, o an döndürülmelidir.",
            "Sonuç beklemeye mahal vermeden derhal iletilmelidir.",
        ],
    ),
    _rule(
        "latency_four_seconds_ok",
        r"Dört saniyelik gecikme tamamen kabul edilebilir\.",
        [
            "Dört saniyelik gecikme kabul edilebilir düzeydedir.",
            "Dört saniye kadar beklemek sorun değildir.",
        ],
        [
            "Dört saniyelik bir bekleme süresi tolere edilebilir.",
            "Sonucun dört saniye gecikmesi bir problem yaratmaz.",
        ],
    ),
    _rule(
        "policy_cloud_only",
        r"Analiz yalnız bulutta yapılmalıdır\.",
        [
            "Analiz sadece bulutta yapılmalıdır.",
            "İşlem yalnızca bulut ortamında yürütülmelidir.",
        ],
        [
            "Hesaplama tamamen bulut üzerinde gerçekleştirilmelidir; başka bir "
            "yerde çalıştırılamaz.",
            "Analizin bulut dışında yapılmasına izin verilmez.",
        ],
    ),
    _rule(
        "policy_cloud_forbidden",
        r"Aynı kaydın buluta gönderilmesi kesinlikle yasaktır\.",
        [
            "Aynı kaydın buluta aktarılması kesinlikle yasaktır.",
            "Bu kayıt hiçbir koşulda buluta gönderilemez.",
        ],
        [
            "Kaydın bulut ortamına çıkarılmasına asla izin verilmez.",
            "Söz konusu veri kesinlikle buluta yüklenmemelidir.",
        ],
    ),
    _rule(
        "accuracy_high_qualitative",
        r"Sonuç yüksek doğrulukta olmalıdır\.",
        [
            "Sonucun doğruluğu yüksek olmalıdır.",
            "Yüksek doğrulukta bir sonuç gereklidir.",
        ],
        [
            "Çıktının isabet oranı yüksek tutulmalıdır.",
            "Çıktının doğruluk düzeyi yüksek olmalıdır.",
        ],
    ),
    _rule(
        "accuracy_low_ok",
        r"Yaklaşık ve düşük doğruluklu bir sonuç da yeterlidir\.",
        [
            "Yaklaşık, düşük doğruluklu bir sonuç da yeterlidir.",
            "Düşük doğrulukta yaklaşık bir sonuç kabul edilir.",
        ],
        [
            "Kaba bir tahmin de iş görür; yüksek isabet gerekmez.",
            "Sonucun kesin olması gerekmez, yaklaşık bir değer yeterlidir.",
        ],
    ),
    _rule(
        "privacy_public",
        r"Kullanılan veri kamuya açıktır\.",
        [
            "Kullanılan veri kamuya açık niteliktedir.",
            "Veri kamuya açık kaynaklardan alınmıştır.",
        ],
        [
            "Bu veriler açık veri olarak yayımlanmıştır.",
            "İşlenen bilgiler herkesin erişimine açıktır.",
        ],
    ),
    _rule(
        "policy_all_allowed",
        r"Görev cihaz, yakın edge veya bulutta çalıştırılabilir\.",
        [
            "Görev cihazda, yakın edge'de ya da bulutta çalıştırılabilir.",
            "Cihaz, yakın edge ve bulut seçeneklerinin hepsi kullanılabilir.",
        ],
        [
            "Yürütme yeri konusunda kısıt yoktur; cihaz, edge ve bulut serbesttir.",
            "İşlem uç cihazda, yakındaki edge düğümünde veya bulut ortamında "
            "yapılabilir.",
        ],
    ),
    _rule(
        "reliability_high_qualitative",
        r"Bağlantı kesintilerine karşı yüksek güvenilirlik sağlanmalıdır\.",
        [
            "Bağlantı kopmalarına karşı yüksek güvenilirlik gerekir.",
            "Kesintilere karşı yüksek güvenilirlikte çalışılmalıdır.",
        ],
        [
            "Ağ kopmalarında bile hizmetin sürekliliği güçlü biçimde korunmalıdır.",
            "Sistem bağlantı sorunlarında dahi yüksek ölçüde güvenilir kalmalıdır.",
        ],
    ),
    _rule(
        "energy_balanced",
        r"Enerji tüketimi ile yanıt süresi dengelenmelidir\.",
        [
            "Enerji tüketimi ve yanıt süresi dengelenmelidir.",
            "Enerji kullanımı ile gecikme arasında denge kurulmalıdır.",
        ],
        [
            "Pil harcaması ile hız arasında makul bir orta yol bulunmalıdır.",
            "Güç tüketimi ve tepki süresi birlikte gözetilerek dengelenmelidir.",
        ],
    ),
)


def _choice(key: str, options: tuple[str, ...]) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return options[int.from_bytes(digest[:4], "big") % len(options)]


def paraphrase_text(text: str, *, pool: Pool, key: str) -> tuple[str, list[str]]:
    """Rewrite every recognised requirement sentence; return text and rule ids."""

    applied: list[str] = []
    for rule in RULES:
        options = rule.train if pool == "train" else rule.test

        def replace(
            match: re.Match[str],
            rule: ParaphraseRule = rule,
            options: tuple[str, ...] = options,
        ) -> str:
            template = _choice(f"{key}|{rule.rule_id}|{match.start()}", options)
            applied.append(rule.rule_id)
            groups = match.groupdict()
            return template.format(**{k: v for k, v in groups.items() if v})

        text = rule.pattern.sub(replace, text)
    return text, applied


def paraphrase_task(
    task: Mapping[str, Any], *, pool: Pool, variant: int = 0
) -> tuple[dict[str, Any], list[str]]:
    """Return a copy of ``task`` with both visible text fields paraphrased."""

    result = dict(task)
    applied: list[str] = []
    for field in ("task_text", "user_policy_text"):
        value = task.get(field)
        if isinstance(value, str):
            key = f"{task['task_id']}|{pool}|{variant}|{field}"
            result[field], used = paraphrase_text(value, pool=pool, key=key)
            applied.extend(used)
    return result, applied


__all__ = ["RULES", "ParaphraseRule", "paraphrase_task", "paraphrase_text"]
