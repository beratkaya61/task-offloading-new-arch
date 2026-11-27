# ruff: noqa: E501, RUF001, RUF003
"""Constraint-preserving checks for LLM natural rewrites of expansion tasks.

The generator lives in ``scripts/generate_natural_rewrites.py`` (GPU).  This
module holds the deterministic, CPU-only parts so they are unit-testable:
prompt construction and the acceptance filter.  The filter is a cheap
pre-screen; blind human annotation remains the authority on meaning.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

REWRITE_SYSTEM_PROMPT = (
    "Sen Türkçe yazan bir teknik editörsün. Sana bir IoT/uç bilişim görev açıklaması "
    "verilecek. Bu açıklamayı, gerçek bir kullanıcının ya da kurumun yazacağı doğal ve "
    "akıcı Türkçeyle yeniden yaz."
)
REWRITE_RULES = (
    "Kurallar:\n"
    "1. Bütün sayıları, yüzdeleri, ms/saniye değerlerini ve F1 gibi metrik adlarını aynen koru.\n"
    "2. İzin ve yasak ifadelerini (cihaz, edge/uç sunucu, bulut) anlamını değiştirmeden koru.\n"
    "3. Veri gizliliği, enerji/pil, güvenilirlik, doğruluk ve paralel işlenebilirlik hakkındaki her bilgiyi koru.\n"
    "4. Yeni bir kısıt ekleme, hiçbir kısıtı çıkarma; metinde çelişki varsa çelişkiyi de koru.\n"
    "5. Cihazı ve yapılan işi koru ama cümle yapısını ve kelime seçimini belirgin biçimde değiştir.\n"
    "6. En fazla dört cümlelik tek bir paragraf yaz. Başlık, madde işareti, açıklama veya not ekleme.\n"
    "Yalnız yeniden yazılmış metni döndür."
)

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_FORBID = r"(kullanılamaz|kullanılmamalı|yasak|gönderilemez|gönderilmemeli|gönderilmesi|olmamalı|olamaz|izin verilmez|izin yok|çıkmamalı|dışına)"


def rewrite_messages(source_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": f"{REWRITE_RULES}\n\nMetin:\n{source_text}"},
    ]


def _numbers(text: str) -> set[str]:
    return {n.replace(".", ",") for n in _NUMBER.findall(text)}


def _has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def clean_output(raw: str) -> str:
    """Strip quotes/labels a chat model may wrap around the rewrite."""

    text = raw.strip()
    # A chat model that misses its end-of-turn token restarts with a role tag.
    text = re.split(r"(?<=[.!?\"”])\s*(?:assistant|user|system)\b", text, maxsplit=1)[0]
    text = re.sub(r"^(yeniden yazılmış metin|metin)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip('"“”«»').strip()
    return re.sub(r"\s+", " ", text)


def rewrite_issues(source: str, rewrite: str, annotation: Mapping[str, Any]) -> list[str]:
    """Return reasons to reject ``rewrite``; an empty list means accepted."""

    issues: list[str] = []
    src = re.sub(r"\s+", " ", source).strip()
    out = rewrite.strip()
    folded = out.casefold()
    if not out:
        return ["empty"]
    if out.casefold() == src.casefold():
        issues.append("identical_to_source")
    if not 0.6 <= len(out) / max(len(src), 1) <= 2.5:
        issues.append("length_ratio")
    if "\n" in rewrite.strip() or re.search(r"(^|\s)[-*•]\s", out) or _has(r"\bnot\s*:", out):
        issues.append("formatting_or_commentary")
    missing = _numbers(src) - _numbers(out)
    if missing:
        issues.append(f"numbers_missing:{','.join(sorted(missing))}")
    if len(re.findall(r"\b(the|and|with|must|should|data|task)\b", folded)) >= 2:
        issues.append("english_text")

    def status(field: str) -> str:
        return str(annotation[field].get("status"))

    privacy = annotation["privacy"]
    if status("privacy") == "explicit":
        marker = {"public": r"(kamu|açık)", "sensitive": r"hassas", "restricted": r"kısıtl", "internal": r"(kurum|iç)"}[str(privacy.get("data_class"))]
        if not _has(marker, folded):
            issues.append("privacy_marker")
    execution = annotation["execution_policy"]
    if status("execution_policy") != "not_stated":
        if not _has(r"bulut", folded):
            issues.append("execution_cloud_mention")
        if execution.get("cloud") == "forbidden" and not _has(_FORBID, folded):
            issues.append("execution_forbid_marker")
        if execution.get("edge") == "allowed" and not _has(r"(edge|uç)", folded):
            issues.append("execution_edge_mention")
    if status("energy_priority") != "not_stated" and not _has(r"(pil|enerji|batarya)", folded):
        issues.append("energy_marker")
    if status("divisibility") != "not_stated" and not _has(r"(paralel|bağımsız|parça|bölün|eşzamanlı)", folded):
        issues.append("divisibility_marker")
    if status("accuracy") != "not_stated" and not _has(r"(f1|doğru|isabet|hata)", folded):
        issues.append("accuracy_marker")
    if status("reliability") != "not_stated" and not _has(r"(başarı|güvenilir|tamamlan|hatasız|kesinti)", folded):
        issues.append("reliability_marker")
    if status("latency") != "not_stated" and not _has(r"(ms|milisaniye|saniye|anlık|hemen|gecikme|süre)", folded):
        issues.append("latency_marker")

    # Added constraints change the label: a field that is not stated must not
    # gain a marker that the source text did not already contain (device names
    # such as "enerji sayacı" legitimately carry some of these words).
    source_folded = src.casefold()
    for field, pattern in _ADDED_MARKERS.items():
        if status(field) == "not_stated" and _has(pattern, folded) and not _has(pattern, source_folded):
            issues.append(f"added_{field}")
    return issues


_ADDED_MARKERS = {
    "latency": r"(\bms\b|milisaniye|saniye|anlık|gecikme|gerçek zamanlı|hızlı yanıt)",
    "privacy": r"(hassas|gizli|kişisel veri|kamuya açık|kısıtlı)",
    "execution_policy": r"(bulut|edge|uç sunucu|yerel olarak|cihaz üzerinde)",
    "reliability": r"(güvenilir|başarı oranı|başarıyla tamamlan|kesintisiz|hatasız)",
    "accuracy": r"(doğrulu[kğ]|\bf1\b|isabet)",
    "energy_priority": r"(enerji|\bpil\b|pil |batarya|güç tüketimi)",
    "divisibility": r"(paralel|bağımsız|parçalara|eşzamanlı)",
}


__all__ = ["REWRITE_RULES", "REWRITE_SYSTEM_PROMPT", "clean_output", "rewrite_issues", "rewrite_messages"]
